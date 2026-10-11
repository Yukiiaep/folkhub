#!/usr/bin/env python3
"""SNOWSPIN! logo pack generator.

Renders a Roblox-style "SNOWSPIN!" logo plus every layer of it as separate,
pixel-aligned 1920x1080 transparent PNGs, then zips the pack.

    pip install numpy pillow scipy
    python3 generate_logo.py            # -> SNOWSPIN_Logo_Pack/ + SNOWSPIN_Logo_Pack.zip

Everything is drawn procedurally at 2x and box-downsampled for anti-aliasing.
"""
import math
import os
import shutil
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
FONT = os.path.join(HERE, "fonts", "LuckiestGuy-Regular.ttf")
OUT = os.path.join(HERE, "SNOWSPIN_Logo_Pack")

SS = 2                      # supersampling factor
W, H = 1920, 1080
WW, HH = W * SS, H * SS


def hexc(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], np.float32) / 255


NAVY = hexc("#0b1640")
WHITE = hexc("#ffffff")


def grad(stops, t):
    """Multi-stop colour gradient evaluated at t (any shape) -> t.shape + (3,)."""
    t = np.clip(np.asarray(t, np.float32), 0, 1)
    pos = [s[0] for s in stops]
    cols = [hexc(s[1]) for s in stops]
    return np.stack([np.interp(t, pos, [c[i] for c in cols]) for i in range(3)], -1).astype(np.float32)


# ----------------------------------------------------------------------------
# mask helpers (float masks at supersampled resolution, radii in 1x pixels)
# ----------------------------------------------------------------------------
def _bbox(m, pad):
    rows = np.where(m.max(1) > 0.002)[0]
    cols = np.where(m.max(0) > 0.002)[0]
    if len(rows) == 0:
        return None
    return (max(rows[0] - pad, 0), min(rows[-1] + pad + 1, m.shape[0]),
            max(cols[0] - pad, 0), min(cols[-1] + pad + 1, m.shape[1]))


def dilate(m, r):
    r = r * SS
    out = np.zeros_like(m)
    bb = _bbox(m, int(r) + 3)
    if bb is None:
        return out
    y0, y1, x0, x1 = bb
    sub = m[y0:y1, x0:x1]
    d = ndi.distance_transform_edt(sub < 0.5)
    out[y0:y1, x0:x1] = np.maximum(sub, np.clip(r + 0.5 - d, 0, 1))
    return out


def erode(m, r):
    r = r * SS
    out = np.zeros_like(m)
    bb = _bbox(m, 3)
    if bb is None:
        return out
    y0, y1, x0, x1 = bb
    sub = m[y0:y1, x0:x1]
    d = ndi.distance_transform_edt(sub >= 0.5)
    out[y0:y1, x0:x1] = np.minimum(sub, np.clip(d - r + 0.5, 0, 1))
    return out


def shift(m, dx, dy):
    """Integer shift in supersampled pixels, zero-filled."""
    out = np.zeros_like(m)
    h, w = m.shape[:2]
    xs0, xs1 = max(0, -dx), min(w, w - dx)
    ys0, ys1 = max(0, -dy), min(h, h - dy)
    out[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx] = m[ys0:ys1, xs0:xs1]
    return out


def blur(m, sigma):
    sigma = sigma * SS
    out = np.zeros_like(m)
    bb = _bbox(m, int(sigma * 3.5) + 2)
    if bb is None:
        return out
    y0, y1, x0, x1 = bb
    out[y0:y1, x0:x1] = ndi.gaussian_filter(m[y0:y1, x0:x1], sigma)
    return out


def sm(d):
    """Anti-aliased coverage from a signed distance (1x units)."""
    return np.clip(0.5 - d * SS, 0, 1).astype(np.float32)


def d_seg(X, Y, ax, ay, bx, by):
    px, py = X - ax, Y - ay
    vx, vy = bx - ax, by - ay
    t = np.clip((px * vx + py * vy) / (vx * vx + vy * vy + 1e-9), 0, 1)
    return np.hypot(px - vx * t, py - vy * t)


def d_rrect(U, V, hu, hv, r):
    qx = np.abs(U) - (hu - r)
    qy = np.abs(V) - (hv - r)
    return np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - r


def down(a):
    h, w = a.shape[:2]
    return a.reshape(h // SS, SS, w // SS, SS, *a.shape[2:]).mean((1, 3))


# ----------------------------------------------------------------------------
# layers
# ----------------------------------------------------------------------------
class Layer:
    """Full-canvas 1x premultiplied RGBA + 'line' coverage (navy outlines).

    The line mask drives the one-colour (white fill) versions: lines become
    transparent knock-out gaps, everything else becomes solid white.
    """

    def __init__(self, name, c, l=None, fx=False):
        self.name, self.c, self.fx = name, c.astype(np.float32), fx
        self.l = np.zeros(c.shape[:2], np.float32) if l is None else l.astype(np.float32)


def mk_layer(name, mask, color, line=False, op=1.0, fx=False):
    a = (mask * op).astype(np.float32)
    col = np.broadcast_to(np.asarray(color, np.float32), mask.shape + (3,))
    c = np.concatenate([col * a[..., None], a[..., None]], -1)
    return Layer(name, down(c), down(a) if line else None, fx)


def comp(layers):
    res = np.zeros((H, W, 4), np.float32)
    for L in layers:
        res = L.c + res * (1 - L.c[..., 3:])
    return res


def comp_white(layers):
    res = np.zeros((H, W, 4), np.float32)
    for L in layers:
        if L.fx:
            continue
        a = L.c[..., 3]
        wa = np.clip(a - L.l, 0, 1)
        res = res * (1 - a[..., None]) + wa[..., None]
    return res


def save(c, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    a = c[..., 3:]
    rgb = np.where(a > 1e-5, c[..., :3] / np.maximum(a, 1e-5), 0)
    img = np.concatenate([rgb, a], -1)
    Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8), "RGBA").save(path, optimize=True)


class Painter:
    """Paints into a sub-rectangle of the supersampled canvas."""

    def __init__(self, x0, y0, x1, y1):
        x0, y0 = max(int(x0), 0), max(int(y0), 0)
        x1, y1 = min(int(x1), W), min(int(y1), H)
        self.ox, self.oy = x0 * SS, y0 * SS
        self.w, self.h = (x1 - x0) * SS, (y1 - y0) * SS
        self.c = np.zeros((self.h, self.w, 4), np.float32)
        self.l = np.zeros((self.h, self.w), np.float32)
        self.Y = ((np.arange(self.h, dtype=np.float32) + self.oy + 0.5) / SS)[:, None]
        self.X = ((np.arange(self.w, dtype=np.float32) + self.ox + 0.5) / SS)[None, :]

    def zeros(self):
        return np.zeros((self.h, self.w), np.float32)

    def poly(self, pts):
        k = 2  # extra supersampling for polygon edges
        im = Image.new("L", (self.w * k, self.h * k), 0)
        ImageDraw.Draw(im).polygon([((x * SS - self.ox) * k, (y * SS - self.oy) * k) for x, y in pts], fill=255)
        return np.asarray(im.reduce(k), np.float32) / 255

    def frame(self, cx, cy, ang):
        """Rotated coordinates: U along angle `ang` (radians, screen space), V across."""
        dx, dy = self.X - cx, self.Y - cy
        ca, sa = math.cos(ang), math.sin(ang)
        return dx * ca + dy * sa, -dx * sa + dy * ca

    def fill(self, m, col, op=1.0, line=False):
        a = (m * op).astype(np.float32)
        a3 = a[..., None]
        col = np.asarray(col, np.float32)
        self.c[..., :3] = col * a3 + self.c[..., :3] * (1 - a3)
        self.c[..., 3] = a + self.c[..., 3] * (1 - a)
        self.l = self.l * (1 - a) + (a if line else 0)

    def line(self, m, col=NAVY):
        self.fill(m, col, line=True)

    def outlined(self, m, col, w=6):
        self.line(dilate(m, w))
        self.fill(m, col)

    def layer(self, name, fx=False):
        c = np.zeros((H, W, 4), np.float32)
        l = np.zeros((H, W), np.float32)
        y0, x0 = self.oy // SS, self.ox // SS
        c[y0:y0 + self.h // SS, x0:x0 + self.w // SS] = down(self.c)
        l[y0:y0 + self.h // SS, x0:x0 + self.w // SS] = down(self.l)
        return Layer(name, c, l, fx)


# ----------------------------------------------------------------------------
# reusable motifs
# ----------------------------------------------------------------------------
def d_snowflake(X, Y, cx, cy, r, rot=0.0):
    d = np.full(np.broadcast(X, Y).shape, 1e9, np.float32)
    for k in range(6):
        a = rot + k * math.pi / 3
        ex, ey = cx + r * math.cos(a), cy + r * math.sin(a)
        d = np.minimum(d, d_seg(X, Y, cx, cy, ex, ey))
        for f, bl in ((0.52, 0.30), (0.78, 0.20)):
            bx, by = cx + f * r * math.cos(a), cy + f * r * math.sin(a)
            for s in (-1, 1):
                b = a + s * math.pi / 4
                d = np.minimum(d, d_seg(X, Y, bx, by, bx + bl * r * math.cos(b), by + bl * r * math.sin(b)))
    return np.minimum(d, np.hypot(X - cx, Y - cy) - 0.12 * r)


def snowflake(p, cx, cy, r, rot=0.0, th=None, ow=4):
    th = th if th is not None else max(r * 0.085, 2.5)
    d = d_snowflake(p.X, p.Y, cx, cy, r, rot)
    p.line(sm(d - th - ow))
    p.fill(sm(d - th), grad([(0, "#ffffff"), (1, "#c4ecff")], (p.Y - (cy - r)) / (2 * r)))


def star4(p, cx, cy, r, col=WHITE, glow=True):
    pts = []
    for k in range(8):
        a = -math.pi / 2 + k * math.pi / 4
        rr = r if k % 2 == 0 else r * 0.2
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    m = p.poly(pts)
    if glow:
        p.fill(blur(dilate(m, 2), r * 0.25), hexc("#bff3ff"), op=0.85)
    p.fill(m, col)


# ----------------------------------------------------------------------------
# TEXT
# ----------------------------------------------------------------------------
TEXT = "SNOWSPIN!"
SNOW_FACE = [(0, "#e8fdff"), (0.38, "#86e3ff"), (0.72, "#33abff"), (1, "#1a66f0")]
SPIN_FACE = [(0, "#fde6ff"), (0.38, "#eaa2ff"), (0.72, "#b85cff"), (1, "#7333ee")]
SNOW_EXT = [(0, "#1c64e0"), (1, "#0a2a80")]
SPIN_EXT = [(0, "#5b25d0"), (1, "#2a0d74")]
SNOW_SHADE, SPIN_SHADE = hexc("#1057d8"), hexc("#5a24c9")

FACE_W = 1440          # width of the letter faces (1x)
TEXT_CY = 548          # vertical centre of the word
ARC = 24               # arch height
XSQ = 0.92             # horizontal squash of glyphs
TRACK = 0.035          # letter spacing (fraction of font size)
JIT_Y = [2, -7, 4, -5, 5, -6, 3, -5, 0]
JIT_R = [-2.5, 2.0, -1.5, 2.5, -2.5, 2.0, -2.0, 2.5, 6.0]

STROKE, EXT_OUT, OUT_STROKE, RIM = 11, 7, 13, 5
EXT_DEPTH, EXT_DX = 26, 0.2


def build_text():
    probe = ImageFont.truetype(FONT, 200)
    bbs = [probe.getbbox(ch) for ch in TEXT]
    wsum = sum((b[2] - b[0]) * XSQ for b in bbs) + TRACK * 200 * (len(TEXT) - 1)
    fs = 200 * FACE_W / wsum
    font = ImageFont.truetype(FONT, int(round(fs * SS)))

    F = np.zeros((HH, WW), np.float32)
    face = np.zeros((HH, WW, 3), np.float32)
    gloss = np.zeros((HH, WW), np.float32)
    shade = np.zeros((HH, WW), np.float32)
    shade_col = np.zeros((HH, WW, 3), np.float32)
    lab = np.zeros((HH, WW), np.int32)
    letters = []

    widths = [(b[2] - b[0]) * XSQ * fs / 200 for b in bbs]
    x = W / 2 - FACE_W / 2
    hw = FACE_W / 2
    for i, ch in enumerate(TEXT):
        cx = x + widths[i] / 2
        x += widths[i] + TRACK * fs
        t = (cx - W / 2) / hw
        cy = TEXT_CY - ARC * (1 - t * t) + JIT_Y[i]
        slope = 2 * ARC * t / hw
        ang = -math.degrees(math.atan(slope)) + JIT_R[i]
        snow_word = i < 4

        l, tp, r, b = font.getbbox(ch)
        pad = 30 * SS
        gw, gh = r - l, b - tp
        im = Image.new("L", (gw + 2 * pad, gh + 2 * pad), 0)
        ImageDraw.Draw(im).text((pad - l, pad - tp), ch, font=font, fill=255)
        nw = int(round((gw + 2 * pad) * XSQ))
        im = im.resize((nw, gh + 2 * pad), Image.LANCZOS)
        m = np.asarray(im, np.float32) / 255
        hgt, wid = m.shape
        Yl = np.arange(hgt, dtype=np.float32)[:, None]
        Xl = np.arange(wid, dtype=np.float32)[None, :]
        gx0, gx1 = pad * XSQ, wid - pad * XSQ
        yn = (Yl - pad) / gh
        xn = (Xl - gx0) / (gx1 - gx0)

        rgb = np.broadcast_to(grad(SNOW_FACE if snow_word else SPIN_FACE, yn), (hgt, wid, 3))
        split = 0.47 - 0.10 * xn
        gl = erode(m, 8) * np.clip((split - yn) * gh, 0, 1) * (0.62 - 0.40 * np.clip(yn / split, 0, 1))
        sh = m * (1 - shift(m, 0, -15 * SS))

        def rot(arr, mode):
            img = Image.fromarray((np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8), mode)
            return np.asarray(img.rotate(ang, Image.BICUBIC, expand=True), np.float32) / 255

        mr, rgbr, glr, shr = rot(m, "L"), rot(np.ascontiguousarray(rgb), "RGB"), rot(gl, "L"), rot(sh, "L")
        rh, rw = mr.shape
        tx, ty = int(round(cx * SS - rw / 2)), int(round(cy * SS - rh / 2))
        sl = (slice(ty, ty + rh), slice(tx, tx + rw))
        mm = mr[..., None]
        face[sl] = rgbr * mm + face[sl] * (1 - mm)
        F[sl] = np.maximum(F[sl], mr)
        gloss[sl] = gloss[sl] * (1 - mr) + glr
        shade[sl] = shade[sl] * (1 - mr) + shr
        shade_col[sl][mr > 0.3] = SNOW_SHADE if snow_word else SPIN_SHADE
        lab[sl][mr > 0.5] = i + 1
        letters.append(dict(ch=ch, m=mr, tx=tx, ty=ty, snow=snow_word))

    return F, face, gloss, shade, shade_col, lab, letters


def build_snow(letters):
    rng = np.random.default_rng(7)
    S = np.zeros((HH, WW), np.float32)
    for L in letters:
        m = L["m"]
        bm = m > 0.5
        has = bm.any(0)
        top = bm.argmax(0).astype(np.float32)
        top[~has] = np.nan
        k = 4 * SS
        slope = np.full_like(top, np.inf)
        slope[k:-k] = np.abs(top[2 * k:] - top[:-2 * k]) / (2 * k)
        valid = has & (slope < 0.85)
        valid = ndi.binary_opening(valid, np.ones(5 * SS, bool))
        runs, n = ndi.label(valid)
        Yl = np.arange(m.shape[0], dtype=np.float32)[:, None]
        snow = np.zeros_like(m)
        mdil = dilate(m, 3)
        for ri in range(1, n + 1):
            xs = np.where(runs == ri)[0]
            if len(xs) < 34 * SS:
                continue
            x0, x1 = xs[0], xs[-1] + 1
            u = (np.arange(x0, x1) - x0) / (x1 - x0)
            taper = np.clip(np.minimum(u, 1 - u) / 0.13, 0, 1)
            ph = rng.uniform(0, 6.28)
            below = (13 + 5 * np.sin(u * 2 * math.pi * rng.uniform(1.2, 2.2) + ph)) * taper
            for _ in range(rng.integers(1, 3)):
                c = rng.uniform(0.28, 0.72)
                below += rng.uniform(13, 22) * np.exp(-((u - c) / 0.055) ** 2) * taper
            above = (STROKE - 1 + 2.5 * np.sin(u * 9 + ph)) * np.sqrt(taper)
            t = np.nan_to_num(top[x0:x1])
            col = (Yl >= t - above * SS) & (Yl <= t + below * SS)
            col &= (Yl < t) | (mdil[:, x0:x1] > 0.5)
            snow[:, x0:x1] = np.maximum(snow[:, x0:x1], col.astype(np.float32))
        snow = blur(snow, 2.2)
        snow = np.clip((snow - 0.5) * 3 + 0.5, 0, 1)
        sl = (slice(L["ty"], L["ty"] + m.shape[0]), slice(L["tx"], L["tx"] + m.shape[1]))
        S[sl] = np.maximum(S[sl], snow)
    return S


ICICLES = [  # (letter index, position across letter, length, width)
    (0, 0.50, 58, 28), (2, 0.40, 46, 22), (2, 0.66, 66, 28), (4, 0.42, 52, 26),
    (6, 0.50, 70, 28), (7, 0.22, 44, 22), (7, 0.80, 56, 24),
]


def build():
    F, face, gloss, shade, shade_col, lab, letters = build_text()
    S = build_snow(letters)
    T = np.maximum(F, S)
    D = dilate(T, STROKE)

    # word ownership (nearest letter) for colouring merged regions
    bb = _bbox(D, int((EXT_DEPTH + 60) * SS))
    y0, y1, x0, x1 = bb
    _, inds = ndi.distance_transform_edt(lab[y0:y1, x0:x1] == 0, return_indices=True)
    owner = np.zeros_like(lab)
    owner[y0:y1, x0:x1] = lab[y0:y1, x0:x1][inds[0], inds[1]]
    snow_word = (owner >= 1) & (owner <= 4)

    # 3D extrusion
    N = EXT_DEPTH * SS
    ext = np.zeros_like(D)
    depth = np.full(D.shape, float(N), np.float32)
    for k in range(N, 0, -1):
        sh = shift(D, int(round(k * EXT_DX)), k)
        depth[sh > 0.5] = k
        ext = np.maximum(ext, sh)
    dn = (depth / N)[..., None]
    ext_col = np.where(snow_word[..., None],
                       grad(SNOW_EXT, 0) * (1 - dn) + grad(SNOW_EXT, 1) * dn,
                       grad(SPIN_EXT, 0) * (1 - dn) + grad(SPIN_EXT, 1) * dn)
    E2 = dilate(np.maximum(ext, D), EXT_OUT)

    # icicles hanging off the extrusion outline
    ip = Painter(0, 0, W, H)
    ici_all = np.zeros((HH, WW), np.float32)
    for li, f, ln, wd in ICICLES:
        L = letters[li]
        cols = np.where((L["m"] > 0.5).any(0))[0]
        xc = L["tx"] + cols[0] + f * (cols[-1] - cols[0])
        xa, xb = int(xc - wd * SS / 2), int(xc + wd * SS / 2)
        bottoms = [np.where(E2[:, xx] > 0.5)[0].max() for xx in range(xa, xb + 1, SS)]
        top = min(bottoms) / SS - 6
        x = xc / SS
        pts = [(x - wd / 2, top - 12), (x + wd / 2, top - 12), (x + wd * 0.36, top + ln * 0.42),
               (x + 2, top + ln), (x - wd * 0.40, top + ln * 0.38)]
        m = ip.poly(pts)
        ici_all = np.maximum(ici_all, dilate(m, 5))
        ip.line(dilate(m, 5))
        ip.fill(m, grad([(0, "#ffffff"), (1, "#7fd3ff")], (ip.Y - top) / ln))
        hl = ip.poly([(x - wd * 0.30, top), (x - wd * 0.12, top), (x - wd * 0.02, top + ln * 0.62),
                      (x - wd * 0.14, top + ln * 0.42)])
        ip.fill(hl * m, WHITE, op=0.9)
    O3 = dilate(np.maximum(E2, ici_all), OUT_STROKE)
    O4 = dilate(O3, RIM)

    ys = np.where(O4.max(1) > 0)[0]
    ty0, ty1 = ys[0] / SS, ys[-1] / SS
    Yg = ((np.arange(HH, dtype=np.float32) + 0.5) / SS)[:, None]

    layers = []
    sh_m = blur(shift(O4, 9 * SS, 20 * SS), 7)
    layers.append(mk_layer("01_Drop_Shadow", sh_m, hexc("#061033"), op=0.6, fx=True))
    layers.append(mk_layer("02_Outer_Glow", blur(dilate(O4, 6), 20), hexc("#7fe8ff"), op=0.75, fx=True))
    layers.append(mk_layer("03_Outer_Rim", O4, NAVY, line=True))
    layers.append(mk_layer("04_Outer_Stroke", O3, grad([(0, "#ffffff"), (0.55, "#eefaff"), (1, "#a9defc")],
                                                         (Yg - ty0) / (ty1 - ty0))))
    layers.append(ip.layer("05_Icicles"))
    layers.append(mk_layer("06_Extrusion_Outline", E2, NAVY, line=True))
    layers.append(mk_layer("07_3D_Extrusion", ext, ext_col))
    layers.append(mk_layer("08_Inner_Outline", np.clip(D - T, 0, 1), NAVY, line=True))
    layers.append(mk_layer("09_Gradient_Fill", F, face))
    layers.append(mk_layer("10_Inner_Shadow", shade, shade_col, op=0.55))
    layers.append(mk_layer("11_Gloss_Highlight", gloss * F, WHITE))

    sp = Painter(0, 0, W, H)
    sp.line(np.clip(dilate(S, 3.5) - S, 0, 1) * np.maximum(F, S))
    sp.fill(S, WHITE)
    band = S * (1 - shift(S, 0, -7 * SS))
    sp.fill(band, hexc("#bfe4ff"))
    layers.append(sp.layer("12_Snow_Caps"))

    kp = Painter(0, 0, W, H)
    for li, fx_, fy_, r in [(0, 0.10, 0.20, 30), (2, 0.86, 0.22, 22), (5, 0.12, 0.18, 26),
                            (8, 0.92, 0.10, 34), (3, 0.70, 0.86, 18), (6, 0.15, 0.84, 16)]:
        L = letters[li]
        rows = np.where((L["m"] > 0.5).any(1))[0]
        cols = np.where((L["m"] > 0.5).any(0))[0]
        sx = (L["tx"] + cols[0] + fx_ * (cols[-1] - cols[0])) / SS
        sy = (L["ty"] + rows[0] + fy_ * (rows[-1] - rows[0])) / SS
        star4(kp, sx, sy, r)
    layers.append(kp.layer("13_Sparkles"))

    owner1 = owner[::SS, ::SS]
    return layers, owner1


# ----------------------------------------------------------------------------
# BACKGROUND ELEMENTS
# ----------------------------------------------------------------------------
def aurora():
    out = []
    for name, y0, amp, ph, th, x0, x1, stops in [
        ("Aurora_Ribbon_Violet", 215, 34, 2.2, 70, 330, 1600, [(0, "#7a4dff"), (0.55, "#c77dff"), (1, "#ffb3ec")]),
        ("Aurora_Ribbon_Green", 270, 42, 0.0, 84, 280, 1640, [(0, "#2fb8ff"), (0.55, "#4ff0c8"), (1, "#c9ffe6")]),
    ]:
        p = Painter(x0 - 20, y0 - amp - th - 40, x1 + 20, y0 + amp + th)
        xs = np.linspace(x0, x1, 240)
        u = (xs - x0) / (x1 - x0)
        yc = y0 + amp * np.sin(u * 2 * math.pi * 1.25 + ph)
        t = th * np.clip(np.minimum(u, 1 - u) / 0.14, 0, 1) ** 0.7
        rays = 12 * np.abs(np.sin(xs * 0.045 + ph)) * np.clip(np.minimum(u, 1 - u) / 0.14, 0, 1)
        top = yc - t / 2 - rays
        bot = yc + t / 2
        m = p.poly(list(zip(xs, top)) + list(zip(xs[::-1], bot[::-1])))
        ycg = np.interp(p.X, xs, yc)
        tg = np.maximum(np.interp(p.X, xs, t), 1)
        v = np.clip((p.Y - (ycg - tg / 2 - 12)) / (tg + 12), 0, 1)
        p.line(dilate(m, 4.5))
        p.fill(m, grad(stops, v))
        streak = sm(np.abs(np.sin(p.X * 0.045 + ph)) * 9 - 3.0) * np.clip(1 - v * 1.3, 0, 1)
        p.fill(m * streak, WHITE, op=0.35)
        p.fill(m * sm(np.abs(p.Y - (ycg + tg * 0.32)) - 2.5), WHITE, op=0.55)
        out.append(p.layer(name))
    return "01_Aurora", out


def swirl():
    out = []
    cx, cy, rx, ry = 960, 560, 905, 300
    for name, a0, a1, flip in [("Spin_Swirl_Left", 148, 224, False), ("Spin_Swirl_Right", -44, 32, True)]:
        p = Painter(0, 150, W, 980)
        for lane, (dr, tmax, op) in enumerate([(0, 30, 1.0), (-48, 14, 1.0)]):
            outer, inner = [], []
            n = 80
            for k in range(n + 1):
                s = k / n
                a = math.radians(a0 + (a1 - a0) * s)
                head = s if not flip else 1 - s
                t = tmax * math.sin(math.pi * min(1.0, head * 1.15)) ** 0.8 if head < 0.87 else tmax * (1 - head) / 0.13 * 0.9
                t = max(t, 0.5)
                rr_x, rr_y = rx + dr, ry + dr * 0.35
                ca, sa = math.cos(a), math.sin(a)
                nx, ny = ca * ry, sa * rx
                nl = math.hypot(nx, ny)
                nx, ny = nx / nl, ny / nl
                px, py = cx + rr_x * ca, cy + rr_y * sa
                outer.append((px + nx * t / 2, py + ny * t / 2))
                inner.append((px - nx * t / 2, py - ny * t / 2))
            m = p.poly(outer + inner[::-1])
            ang = np.arctan2((p.Y - cy) / ry, (p.X - cx) / rx)
            deg = np.degrees(ang)
            if not flip:
                deg = np.where(deg < 0, deg + 360, deg)
            s = np.clip((deg - a0) / (a1 - a0), 0, 1)
            if flip:
                s = 1 - s
            p.line(dilate(m, 4.5))
            p.fill(m, grad([(0, "#2f7dff"), (0.45, "#7fe6ff"), (1, "#ffffff")], s), op=op)
        out.append(p.layer(name))
    return "02_Spin_Swirl", out


def mountain_range(p, pts, stops, snow_frac, bottom):
    rng = np.random.default_rng(len(pts))
    m = p.poly(pts)
    p.line(dilate(m, 6))
    p.fill(m, grad(stops, (p.Y - min(y for _, y in pts)) / (bottom - min(y for _, y in pts))))
    snows, shades = [], []
    for i in range(1, len(pts) - 1):
        (xa, ya), (xp, yp), (xb, yb) = pts[i - 1], pts[i], pts[i + 1]
        if yp >= ya or yp >= yb:
            continue
        shades.append(p.poly([(xp, yp), (xb, yb), (xb, bottom + 20), (xp, bottom + 20)]))
        f = snow_frac
        lx, ly = xp + f * (xa - xp), yp + f * (ya - yp)
        rx_, ry_ = xp + f * (xb - xp), yp + f * (yb - yp)
        zig = []
        nz = 5
        for k in range(1, nz):
            u = k / nz
            zx = rx_ + (lx - rx_) * u
            zy = ry_ + (ly - ry_) * u + (16 if k % 2 else -8) + rng.uniform(-4, 4)
            zig.append((zx, zy))
        snows.append(p.poly([(xp, yp), (rx_, ry_)] + zig + [(lx, ly)]))
    for sh in shades:
        p.fill(sh * m, hexc("#16205a"), op=0.22)
    for sn, sh in zip(snows, shades):
        s = sn * m
        p.line(np.clip(dilate(s, 3), 0, 1) * erode(m, 1))
        p.fill(s, WHITE)
        p.fill(s * sh, hexc("#cfe0ff"))


def mountains():
    out = []
    p = Painter(280, 120, 1640, 700)
    mountain_range(p, [(340, 660), (455, 440), (540, 470), (650, 300), (735, 350), (850, 235), (935, 290),
                       (1060, 215), (1150, 300), (1260, 245), (1370, 360), (1460, 330), (1590, 660)],
                   [(0, "#e3ecff"), (1, "#8aa2e8")], 0.38, 660)
    out.append(p.layer("Mountains_Back_Range"))
    p = Painter(280, 120, 1640, 700)
    mountain_range(p, [(390, 660), (500, 430), (590, 470), (720, 320), (800, 370), (960, 175), (1110, 350),
                       (1205, 300), (1320, 430), (1420, 400), (1540, 660)],
                   [(0, "#b7d8ff"), (0.6, "#5d8ff0"), (1, "#3a5fcf")], 0.36, 660)
    out.append(p.layer("Mountains_Front_Peak"))
    return "03_Mountains", out


def snowboard(name, top, bottom, width, stops, accent, strap):
    (tx, ty), (bx, by) = top, bottom
    cx, cy = (tx + bx) / 2, (ty + by) / 2
    L = math.hypot(tx - bx, ty - by)
    ang = math.atan2(ty - by, tx - bx)          # U points toward the top tip
    pad = max(L, width) / 2 + 20
    p = Painter(cx - pad, cy - pad, cx + pad, cy + pad)
    U, V = p.frame(cx, cy, ang)
    waist = 1 - 0.07 * np.cos(np.clip(U / (L / 2), -1, 1) * math.pi / 2) ** 2
    d = d_rrect(U, V / waist, L / 2, width / 2, width * 0.46)
    sheet = sm(d + 7)
    p.line(sm(d - 6))
    p.fill(sm(d), hexc("#e6edf8"))
    p.fill(sheet, grad(stops, (U + L / 2) / L))
    # graphic stripes
    s = U * 0.85 + V * 0.55
    for c, wd, col in [(-0.02 * L, 11, WHITE), (0.06 * L, 6, WHITE), (-0.36 * L, 9, accent), (-0.42 * L, 5, accent)]:
        p.fill(sheet * sm(np.abs(s - c) - wd), col)
    # emblem near the nose
    ex, ey = cx + 0.36 * L * math.cos(ang), cy + 0.36 * L * math.sin(ang)
    d_sf = d_snowflake(p.X, p.Y, ex, ey, width * 0.30, ang)
    p.fill(sm(d_sf - 4.5), NAVY, op=0.55)
    p.fill(sm(d_sf - 3.2), WHITE)
    # gloss
    p.fill(sheet * sm(np.abs(V + width * 0.27) - width * 0.05) * sm(np.abs(U) - L * 0.44), WHITE, op=0.35)
    # bindings
    for ub in (-0.19 * L, 0.17 * L):
        db = d_rrect(U - ub, V, 27, width * 0.53, 14)
        p.line(sm(db - 4))
        p.fill(sm(db), hexc("#26304f"))
        p.fill(sm(d_rrect(U - ub - 9, V, 9, width * 0.42, 6)), hexc("#3b4877"))
        for o in (-11, 12):
            ds = d_rrect(U - ub - o, V, 6.5, width * 0.47, 4)
            p.line(sm(ds - 3))
            p.fill(sm(ds), strap)
    return p.layer(name)


def snowboards():
    return "04_Snowboards", [
        snowboard("Snowboard_Left", (195, 120), (500, 640), 150,
                  [(0, "#ff3d6e"), (0.55, "#ff7a3d"), (1, "#ffc93d")], hexc("#2b2f7a"), hexc("#ffe14d")),
        snowboard("Snowboard_Right", (1725, 120), (1420, 640), 150,
                  [(0, "#2a4dff"), (0.55, "#1fb4ff"), (1, "#69f2ff")], hexc("#ff5fae"), hexc("#ff5fae")),
    ]


def crystal(p, bx, by, ln, wd, deg, light, dark):
    a = math.radians(deg)
    dx, dy = math.sin(a), -math.cos(a)
    px, py = math.cos(a), math.sin(a)

    def P(u, v):
        return (bx + u * dx + v * px, by + u * dy + v * py)

    m = p.poly([P(0, -wd / 2), P(0.72 * ln, -wd / 2), P(ln, 0), P(0.72 * ln, wd / 2), P(0, wd / 2)])
    left = p.poly([P(0, -wd / 2), P(0.72 * ln, -wd / 2), P(ln, 0), P(0, 0)])
    p.line(dilate(m, 5))
    p.fill(m, hexc(dark))
    p.fill(left * m, hexc(light))
    p.line(sm(d_seg(p.X, p.Y, *P(0, 0), *P(ln, 0)) - 1.6) * erode(m, 2))
    p.fill(p.poly([P(0.12 * ln, -0.34 * wd), P(0.62 * ln, -0.34 * wd), P(0.66 * ln, -0.24 * wd),
                   P(0.12 * ln, -0.22 * wd)]), WHITE, op=0.8)


CYAN = ("#c8f8ff", "#38bff5")
VIOLET = ("#f2d6ff", "#9b5cf0")


def crystal_cluster(name, box, mound, shards):
    p = Painter(*box)
    for s in shards:
        crystal(p, *s)
    mx, base, mw, mh = mound

    def d_ell(cx, cy, rx, ry):
        return (np.hypot((p.X - cx) / rx, (p.Y - cy) / ry) - 1) * min(rx, ry)

    dm = np.minimum(d_ell(mx, base, mw / 2, mh),
                    np.minimum(d_ell(mx - mw * 0.3, base, mw * 0.25, mh * 0.72),
                               d_ell(mx + mw * 0.31, base, mw * 0.24, mh * 0.66)))
    dm = np.maximum(dm, p.Y - base)
    p.line(sm(dm - 5))
    p.fill(sm(dm), WHITE)
    p.fill(sm(dm) * np.clip((p.Y - (base - mh * 0.45)) / (mh * 0.45), 0, 1), hexc("#b9dbff"))
    p.fill(sm(dm) * sm(np.abs(p.Y - (base - mh * 0.62)) - 2) * sm(np.abs(p.X - mx) - mw * 0.18), hexc("#d6ebff"))
    return p.layer(name)


def crystals():
    return "05_Ice_Crystals", [
        crystal_cluster("Ice_Crystals_Center", (740, 600, 1180, 990), (960, 942, 340, 46), [
            (838, 922, 110, 36, -52, *CYAN), (1084, 922, 120, 38, 50, *CYAN),
            (900, 916, 175, 50, -26, *VIOLET), (1026, 916, 185, 50, 24, *VIOLET),
            (962, 914, 240, 64, 0, *CYAN)]),
        crystal_cluster("Ice_Crystals_Left", (520, 740, 720, 990), (615, 962, 160, 28), [
            (590, 948, 90, 30, -30, *VIOLET), (640, 946, 120, 36, 14, *CYAN)]),
        crystal_cluster("Ice_Crystals_Right", (1200, 740, 1410, 990), (1305, 962, 160, 28), [
            (1330, 948, 92, 30, 30, *CYAN), (1282, 946, 124, 36, -12, *VIOLET)]),
    ]


def pine(name, x, base, h, w):
    p = Painter(x - w / 2 - 20, base - h - 20, x + w / 2 + 20, base + 20)
    th, tw = 0.14 * h, 0.17 * w
    trunk = sm(d_rrect(p.X - x, p.Y - (base - th / 2), tw / 2, th / 2, 4))
    p.line(sm(d_rrect(p.X - x, p.Y - (base - th / 2), tw / 2, th / 2, 4) - 5))
    p.fill(trunk, grad([(0, "#9a6038"), (1, "#5e3620")], (p.Y - base + th) / th))
    top = base - h
    canopy = h - th * 0.55
    tiers = [(0.42, 1.00, 0.50), (0.20, 0.72, 0.41), (0.00, 0.44, 0.31)]
    for k, (fa, fb, fw) in enumerate(tiers):
        ya, yb, hw = top + fa * canopy, top + fb * canopy, fw * w
        bottom = [(x + hw - j * 2 * hw / 6, yb + (7 if j % 2 else -2)) for j in range(7)]
        m = p.poly([(x, ya)] + bottom)
        p.line(dilate(m, 5))
        p.fill(m, grad([(0, "#46e08c"), (1, "#138a56")], (p.Y - ya) / (yb - ya)))
        p.fill(p.poly([(x, ya), (x + hw, yb + 8), (x + hw * 0.05, yb + 8)]) * m, hexc("#0c4a3a"), op=0.25)
        sf = 0.42 if k == 2 else 0.34
        sy = ya + sf * (yb - ya)
        shw = hw * sf * 1.05
        wav = [(x + shw - j * 2 * shw / 5, sy + (10 if j % 2 else -3)) for j in range(6)]
        s = p.poly([(x, ya - 6)] + wav) * dilate(m, 1)
        p.line(np.clip(dilate(s, 2.5) - s, 0, 1) * m)
        p.fill(s, WHITE)
        p.fill(s * p.poly([(x, ya), (x + shw + 4, sy + 12), (x, sy + 12)]), hexc("#cfe4ff"))
    return p.layer(name)


def trees():
    return "06_Pine_Trees", [
        pine("Pine_Tree_Left_Back", 120, 930, 250, 170),
        pine("Pine_Tree_Left_Big", 250, 960, 360, 230),
        pine("Pine_Tree_Left_Small", 395, 975, 210, 145),
        pine("Pine_Tree_Right_Back", 1805, 930, 260, 175),
        pine("Pine_Tree_Right_Big", 1670, 960, 350, 225),
        pine("Pine_Tree_Right_Small", 1525, 975, 200, 140),
    ]


def snowball(name, cx, cy, r):
    p = Painter(cx - r - 12, cy - r - 12, cx + r + 12, cy + r + 12)
    d = np.hypot(p.X - cx, p.Y - cy) - r
    p.line(sm(d - 5))
    p.fill(sm(d), hexc("#f3faff"))
    d2 = np.hypot(p.X - cx + r * 0.22, p.Y - cy + r * 0.25) - r * 0.98
    p.fill(sm(d) * (1 - sm(d2)), hexc("#b4d8ff"))
    p.fill(sm(np.hypot((p.X - cx + r * 0.38) / 1.4, p.Y - cy + r * 0.42) - r * 0.2), WHITE)
    return p.layer(name)


def snowballs():
    return "07_Snowballs", [
        snowball("Snowball_1", 478, 958, 32),
        snowball("Snowball_2", 528, 975, 21),
        snowball("Snowball_3", 1440, 965, 27),
    ]


SNOWFLAKES = [(70, 420, 34, 0.2), (1850, 450, 30, 0.5), (140, 640, 20, 0.1), (1780, 640, 22, 0.3),
              (700, 1010, 22, 0.4), (1220, 1015, 24, 0.0), (420, 90, 22, 0.25), (1500, 95, 24, 0.6),
              (960, 70, 26, 0.15)]


def snowflakes():
    out = []
    for i, (x, y, r, rot) in enumerate(SNOWFLAKES):
        p = Painter(x - r - 12, y - r - 12, x + r + 12, y + r + 12)
        snowflake(p, x, y, r, rot)
        out.append(p.layer(f"Snowflake_{i + 1}"))
    return "08_Snowflakes", out


# ----------------------------------------------------------------------------
# preview backgrounds
# ----------------------------------------------------------------------------
def bg_preview(kind):
    Y = np.linspace(0, 1, H, dtype=np.float32)[:, None]
    X = np.linspace(0, 1, W, dtype=np.float32)[None, :]
    if kind == "dark":
        rgb = grad([(0, "#0b1a4a"), (0.6, "#1d3f9e"), (1, "#3f7fe0")], Y + 0 * X)
    elif kind == "light":
        rgb = grad([(0, "#bfe8ff"), (1, "#f2fbff")], Y + 0 * X)
    else:
        rgb = grad([(0, "#2b2f3a"), (1, "#14161c")], Y + 0 * X)
    return rgb


def over_bg(c, kind):
    return c[..., :3] + bg_preview(kind) * (1 - c[..., 3:])


def save_jpg(rgb, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray((np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8), "RGB").save(path, quality=92)


# ----------------------------------------------------------------------------
def main():
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    print("text ...")
    text_layers, owner = build()
    groups = []
    for fn in (aurora, swirl, mountains, snowboards, crystals, trees, snowballs, snowflakes):
        print(fn.__name__, "...")
        groups.append(fn())
    bg_layers = [L for _, ls in groups for L in ls]

    full = comp(bg_layers + text_layers)
    text = comp(text_layers)
    save(full, f"{OUT}/01_Full_Logo.png")
    save(comp_white(bg_layers + text_layers), f"{OUT}/02_Full_Logo_White.png")
    save(text, f"{OUT}/03_Text_Logo.png")
    save(comp_white(text_layers), f"{OUT}/04_Text_Logo_White.png")

    for L in text_layers:
        save(L.c, f"{OUT}/05_Text_Effects/{L.name}.png")

    text_nofx = comp([L for L in text_layers if not L.fx])
    for i, ch in enumerate(TEXT):
        nm = "Exclamation" if ch == "!" else ch
        sel = (owner == i + 1)[..., None].astype(np.float32)
        save(text_nofx * sel, f"{OUT}/06_Text_Letters/{i + 1:02d}_{nm}.png")

    save(comp(bg_layers), f"{OUT}/07_Background_Elements/00_All_Background_Elements.png")
    for gname, ls in groups:
        base = f"{OUT}/07_Background_Elements/{gname}"
        save(comp(ls), f"{base}/{gname[3:]}_Group.png")
        for L in ls:
            save(L.c, f"{base}/{L.name}.png")

    for kind in ("dark", "light", "charcoal"):
        save_jpg(over_bg(full, kind), f"{OUT}/08_Previews/Full_Logo_on_{kind}.jpg")
    save_jpg(over_bg(text, "dark"), f"{OUT}/08_Previews/Text_Logo_on_dark.jpg")
    save_jpg(over_bg(comp_white(bg_layers + text_layers), "dark"), f"{OUT}/08_Previews/Full_Logo_White_on_dark.jpg")
    save_jpg(over_bg(comp_white(text_layers), "dark"), f"{OUT}/08_Previews/Text_Logo_White_on_dark.jpg")

    with open(f"{OUT}/README.txt", "w") as f:
        f.write(readme(groups, text_layers))
    shutil.copy(os.path.join(HERE, "fonts", "LICENSE-LuckiestGuy.txt"), f"{OUT}/FONT_LICENSE_LuckiestGuy.txt")

    zpath = OUT + ".zip"
    if os.path.exists(zpath):
        os.remove(zpath)
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(OUT):
            for fn in sorted(files):
                fp = os.path.join(root, fn)
                z.write(fp, os.path.relpath(fp, os.path.dirname(OUT)))
    print("wrote", zpath)


def readme(groups, text_layers):
    lines = [
        "SNOWSPIN! - Logo Pack",
        "=====================",
        "",
        "Every PNG is 1920x1080 with a transparent background and is pixel-aligned:",
        "stack any of them in the same position and they line up exactly.",
        "",
        "01_Full_Logo.png            Full logo, all layers flattened",
        "02_Full_Logo_White.png      Full logo, single-colour white fill (outlines knocked out)",
        "03_Text_Logo.png            Text logo with all effects",
        "04_Text_Logo_White.png      Text logo, single-colour white fill",
        "05_Text_Effects/            Every text effect on its own PNG",
        "06_Text_Letters/            Each finished letter on its own PNG (no shadow/glow; use 05 for those)",
        "07_Background_Elements/     Each background group (+ group flattened) and each child element",
        "08_Previews/                JPG previews on sample backgrounds (not transparent)",
        "",
        "Layer order (bottom -> top) to rebuild 01_Full_Logo.png:",
    ]
    n = 1
    for gname, ls in groups:
        for L in ls:
            lines.append(f"  {n:2d}. 07_Background_Elements/{gname}/{L.name}.png")
            n += 1
    for L in text_layers:
        lines.append(f"  {n:2d}. 05_Text_Effects/{L.name}.png")
        n += 1
    lines += [
        "",
        "Font: Luckiest Guy by Astigmatic (Apache License 2.0, see FONT_LICENSE_LuckiestGuy.txt).",
        "Generated by branding/snowspin-logo/generate_logo.py",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
