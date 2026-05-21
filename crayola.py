#!/usr/bin/env python3
""
Abstract Chaos Image Generator
Produces high-definition colorful images with many independent entropy sources.
Author: http://github.com/wifiknight45
"""

import os
import sys
import time
import math
import random
import hashlib
import multiprocessing as mp
from functools import partial
from dataclasses import dataclass, asdict

import numpy as np
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
from noise import pnoise2, snoise2
from scipy import ndimage
import cv2  # optional: for camera capture and advanced transforms
from numba import njit, prange

# -------------------------
# Utilities and entropy
# -------------------------
def mix_entropy(*sources: bytes) -> int:
    """Mix multiple entropy sources into a 64-bit integer seed."""
    h = hashlib.blake2b(digest_size=16)
    for s in sources:
        h.update(s)
    digest = h.digest()
    return int.from_bytes(digest[:8], 'big')

def get_system_entropy(user_seed: int = None, use_camera: bool = False) -> int:
    """Collect entropy from OS, time, pid, optional camera frame, and user seed."""
    parts = []
    parts.append(os.urandom(32))
    parts.append(str(time.time_ns()).encode())
    parts.append(str(os.getpid()).encode())
    if user_seed is not None:
        parts.append(str(user_seed).encode())
    if use_camera:
        try:
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            cap.release()
            if ret:
                parts.append(frame.tobytes()[:4096])
        except Exception:
            pass
    return mix_entropy(*parts)

# -------------------------
# Palette utilities
# -------------------------
def random_palette(n=6, seed=None):
    rnd = random.Random(seed)
    base_h = rnd.random()
    palette = []
    for i in range(n):
        h = (base_h + rnd.random()*0.5) % 1.0
        s = 0.6 + rnd.random()*0.4
        v = 0.6 + rnd.random()*0.4
        palette.append(hsv_to_rgb(h, s, v))
    return palette

def hsv_to_rgb(h, s, v):
    """Return RGB tuple 0-255 from HSV floats 0-1."""
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(r*255), int(g*255), int(b*255))

def palette_map(img_gray, palette):
    """Map grayscale image (0..255) to palette by interpolation."""
    arr = np.asarray(img_gray).astype(np.float32) / 255.0
    n = len(palette)
    # create linear palette ramp
    ramp = np.array(palette, dtype=np.float32) / 255.0
    idx = arr * (n - 1)
    i0 = np.floor(idx).astype(int)
    frac = idx - i0
    i1 = np.clip(i0 + 1, 0, n - 1)
    out = (1 - frac[..., None]) * ramp[i0] + frac[..., None] * ramp[i1]
    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(out)

# -------------------------
# Noise and fBm
# -------------------------
def perlin_layer(width, height, scale=1.0, octaves=4, lacunarity=2.0, gain=0.5, seed=0):
    arr = np.zeros((height, width), dtype=np.float32)
    freq = scale / min(width, height)
    for y in range(height):
        for x in range(width):
            nx = x * freq
            ny = y * freq
            val = 0.0
            amp = 1.0
            f = 1.0
            for o in range(octaves):
                val += amp * pnoise2(nx * f + seed, ny * f + seed, repeatx=1024, repeaty=1024)
                amp *= gain
                f *= lacunarity
            arr[y, x] = val
    # normalize to 0..255
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-12)
    return (arr * 255).astype(np.uint8)

# -------------------------
# Reaction-diffusion (Gray-Scott)
# -------------------------
def gray_scott(width, height, steps=2000, feed=0.0367, kill=0.0649, seed=0):
    # initialize
    U = np.ones((height, width), dtype=np.float32)
    V = np.zeros((height, width), dtype=np.float32)
    # random perturbation
    rng = np.random.RandomState(seed)
    cx, cy = width//2, height//2
    r = min(width, height)//10
    for _ in range(10):
        x = rng.randint(cx-r, cx+r)
        y = rng.randint(cy-r, cy+r)
        rr = rng.randint(r//4, r)
        yy, xx = np.ogrid[-y:height-y, -x:width-x]
        mask = xx*xx + yy*yy <= rr*rr
        V[mask] = 0.25 + rng.rand()*0.5
        U[mask] = 0.5 - rng.rand()*0.4
    # laplacian kernel
    lap = lambda Z: (
        -Z + 0.2*(np.roll(Z,1,0)+np.roll(Z,-1,0)+np.roll(Z,1,1)+np.roll(Z,-1,1))
        + 0.05*(np.roll(Z,1,0)*0 + np.roll(Z,-1,0)*0)
    )
    for i in range(steps):
        Lu = ndimage.laplace(U)
        Lv = ndimage.laplace(V)
        uvv = U * V * V
        U += (0.16 * Lu - uvv + feed * (1 - U))
        V += (0.08 * Lv + uvv - (feed + kill) * V)
        if i % 200 == 0 and i > 0:
            # small random perturbation to increase chaos
            U += (rng.rand(*U.shape)-0.5)*0.02
            V += (rng.rand(*V.shape)-0.5)*0.02
    img = np.clip((V - V.min()) / (V.max() - V.min() + 1e-12) * 255, 0, 255).astype(np.uint8)
    return img

# -------------------------
# Cellular automata (2D totalistic)
# -------------------------
@njit(parallel=True)
def ca_step(grid, birth, survive):
    h, w = grid.shape
    out = np.zeros_like(grid)
    for y in prange(h):
        for x in range(w):
            s = 0
            for dy in (-1,0,1):
                for dx in (-1,0,1):
                    if dy==0 and dx==0:
                        continue
                    ny = (y+dy) % h
                    nx = (x+dx) % w
                    s += grid[ny, nx]
            if grid[y,x] == 1:
                out[y,x] = 1 if s in survive else 0
            else:
                out[y,x] = 1 if s in birth else 0
    return out

def cellular_automata(width, height, steps=200, seed=0):
    rng = np.random.RandomState(seed)
    grid = (rng.rand(height, width) > 0.7).astype(np.uint8)
    birth = set([3])
    survive = set([2,3])
    for i in range(steps):
        grid = ca_step(grid, birth, survive)
    img = (grid * 255).astype(np.uint8)
    return img

# -------------------------
# Voronoi diagram
# -------------------------
def voronoi(width, height, points=200, seed=0):
    rng = np.random.RandomState(seed)
    pts = rng.randint(0, max(width, height), size=(points, 2))
    xs = np.arange(width)
    ys = np.arange(height)
    xv, yv = np.meshgrid(xs, ys)
    arr = np.full((height, width), 255, dtype=np.uint8)
    for i, (px, py) in enumerate(pts):
        d = (xv - px)**2 + (yv - py)**2
        if i == 0:
            best = d
            arr = np.full_like(d, i, dtype=np.uint16)
        else:
            mask = d < best
            arr[mask] = i
            best[mask] = d[mask]
    # normalize indices to 0..255
    arr = (arr.astype(np.float32) / arr.max() * 255).astype(np.uint8)
    return arr

# -------------------------
# Chaotic attractor projection
# -------------------------
def lorenz_projection(width, height, steps=200000, seed=0):
    rng = np.random.RandomState(seed)
    sigma = 10.0 + rng.rand()*10.0
    rho = 28.0 + rng.rand()*10.0
    beta = 8.0/3.0 + rng.rand()*1.0
    x = rng.rand()*0.1
    y = rng.rand()*0.1
    z = rng.rand()*0.1
    xs = []
    ys = []
    dt = 0.01
    for i in range(steps):
        dx = sigma*(y-x)
        dy = x*(rho-z)-y
        dz = x*y - beta*z
        x += dx*dt
        y += dy*dt
        z += dz*dt
        if i % 10 == 0:
            xs.append(x)
            ys.append(z)  # project x,z
    xs = np.array(xs)
    ys = np.array(ys)
    xs = (xs - xs.min()) / (xs.max() - xs.min() + 1e-12)
    ys = (ys - ys.min()) / (ys.max() - ys.min() + 1e-12)
    img = np.zeros((height, width), dtype=np.uint8)
    ix = (xs * (width - 1)).astype(int)
    iy = (ys * (height - 1)).astype(int)
    for a, b in zip(ix, iy):
        img[b, a] = min(255, img[b, a] + 8)
    img = ndimage.gaussian_filter(img, sigma=1.0)
    return img

# -------------------------
# Random spline strokes
# -------------------------
def random_splines(width, height, n_curves=30, seed=0):
    rng = np.random.RandomState(seed)
    canvas = np.zeros((height, width), dtype=np.float32)
    for i in range(n_curves):
        pts = []
        k = rng.randint(3, 8)
        for j in range(k):
            pts.append((rng.randint(0, width-1), rng.randint(0, height-1)))
        # draw polyline with thickness
        thickness = rng.randint(1, max(1, min(width, height)//100))
        img = np.zeros((height, width), dtype=np.uint8)
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(img, a, b, color=255, thickness=thickness)
        canvas += img.astype(np.float32)
    canvas = np.clip(canvas / canvas.max() * 255, 0, 255).astype(np.uint8)
    return canvas

# -------------------------
# Layer composition and postprocessing
# -------------------------
def blend_images(base, top, mode='overlay', alpha=0.5):
    a = np.asarray(base).astype(np.float32) / 255.0
    b = np.asarray(top).astype(np.float32) / 255.0
    if mode == 'overlay':
        out = np.where(a <= 0.5, 2*a*b, 1 - 2*(1-a)*(1-b))
    elif mode == 'add':
        out = np.clip(a + b*alpha, 0, 1)
    elif mode == 'multiply':
        out = a * (b*alpha + (1-alpha))
    elif mode == 'screen':
        out = 1 - (1-a)*(1-b)
    else:
        out = a*(1-alpha) + b*alpha
    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(out)

def postprocess(img: Image.Image, seed=0):
    rnd = random.Random(seed)
    # slight warp
    if rnd.random() < 0.5:
        arr = np.asarray(img)
        h, w = arr.shape[:2]
        dx = (np.sin(np.linspace(0, math.pi*2, w)) * (rnd.random()*10)).astype(np.float32)
        map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (map_x + dx).astype(np.float32)
        arr = cv2.remap(arr, map_x, map_y.astype(np.float32), interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        img = Image.fromarray(arr)
    # color jitter
    enh = ImageEnhance.Color(img)
    img = enh.enhance(0.8 + rnd.random()*1.6)
    # contrast
    enh = ImageEnhance.Contrast(img)
    img = enh.enhance(0.8 + rnd.random()*1.6)
    # unsharp
    if rnd.random() < 0.6:
        img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    # film grain
    arr = np.asarray(img).astype(np.float32)
    grain = (rnd.random()*0.08) * 255.0 * (np.random.RandomState(seed).randn(*arr.shape[:2]) * 0.5 + 0.5)
    if arr.ndim == 3:
        arr[...,0] = np.clip(arr[...,0] + grain, 0, 255)
        arr[...,1] = np.clip(arr[...,1] + grain, 0, 255)
        arr[...,2] = np.clip(arr[...,2] + grain, 0, 255)
    else:
        arr = np.clip(arr + grain, 0, 255)
    img = Image.fromarray(arr.astype(np.uint8))
    return img

# -------------------------
# Main generator
# -------------------------
@dataclass
class GenConfig:
    width: int = 3840
    height: int = 2160
    seed: int = None
    use_camera_entropy: bool = False
    layers: int = 6
    out_dir: str = "output"
    filename_prefix: str = "chaos"

def generate_image(cfg: GenConfig):
    # collect entropy
    sys_seed = get_system_entropy(cfg.seed, use_camera=cfg.use_camera_entropy)
    rnd = random.Random(sys_seed)
    # create base layers
    layers = []
    for i in range(cfg.layers):
        choice = rnd.choice(['perlin', 'fbm', 'gray_scott', 'ca', 'voronoi', 'lorenz', 'splines'])
        layer_seed = mix_entropy(sys_seed.to_bytes(8,'big'), str(i).encode())
        s = int(layer_seed & 0xffffffff)
        if choice == 'perlin':
            arr = perlin_layer(cfg.width, cfg.height, scale=1.0 + rnd.random()*8.0, octaves=3 + rnd.randint(0,3), seed=s)
            img = Image.fromarray(arr).convert('L')
        elif choice == 'fbm':
            arr = perlin_layer(cfg.width, cfg.height, scale=0.5 + rnd.random()*4.0, octaves=5, seed=s)
            img = Image.fromarray(arr).convert('L')
        elif choice == 'gray_scott':
            arr = gray_scott(cfg.width//2, cfg.height//2, steps=800 + rnd.randint(0,1200), feed=0.02 + rnd.random()*0.06, kill=0.045 + rnd.random()*0.06, seed=s)
            img = Image.fromarray(cv2.resize(arr, (cfg.width, cfg.height), interpolation=cv2.INTER_CUBIC)).convert('L')
        elif choice == 'ca':
            arr = cellular_automata(cfg.width//2, cfg.height//2, steps=200 + rnd.randint(0,400), seed=s)
            img = Image.fromarray(cv2.resize(arr, (cfg.width, cfg.height), interpolation=cv2.INTER_NEAREST)).convert('L')
        elif choice == 'voronoi':
            arr = voronoi(cfg.width, cfg.height, points=200 + rnd.randint(0,800), seed=s)
            img = Image.fromarray(arr).convert('L')
        elif choice == 'lorenz':
            arr = lorenz_projection(cfg.width, cfg.height, steps=200000 + rnd.randint(0,200000), seed=s)
            img = Image.fromarray(arr).convert('L')
        else:
            arr = random_splines(cfg.width, cfg.height, n_curves=20 + rnd.randint(0,80), seed=s)
            img = Image.fromarray(arr).convert('L')
        layers.append((img, choice))
    # colorize and blend
    base = Image.new('RGB', (cfg.width, cfg.height), (0,0,0))
    palette = random_palette(n=6, seed=sys_seed)
    for idx, (img, kind) in enumerate(layers):
        # map grayscale to palette
        pal_img = palette_map(img, random_palette(n=6, seed=sys_seed ^ idx))
        # random blend mode
        mode = rnd.choice(['overlay','add','multiply','screen','mix'])
        alpha = 0.3 + rnd.random()*0.8
        base = blend_images(base, pal_img, mode=mode, alpha=alpha)
        # occasional warp
        if rnd.random() < 0.25:
            base = postprocess(base, seed=int(sys_seed ^ idx))
    # final postprocess
    final = postprocess(base, seed=sys_seed)
    # optional quantize/dither
    if rnd.random() < 0.4:
        final = final.convert('P', palette=Image.ADAPTIVE, colors=256).convert('RGB')
    # save
    os.makedirs(cfg.out_dir, exist_ok=True)
    fname = f"{cfg.filename_prefix}_{int(sys_seed & 0xffffffff):08x}.png"
    outpath = os.path.join(cfg.out_dir, fname)
    final.save(outpath, format='PNG', compress_level=1)
    # strip metadata by re-saving via Pillow (already minimal)
    return outpath

# -------------------------
# CLI and batch generation
# -------------------------
def worker_task(i, cfg_dict):
    cfg = GenConfig(**cfg_dict)
    cfg.seed = cfg.seed ^ i if cfg.seed is not None else None
    return generate_image(cfg)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate chaotic abstract images")
    parser.add_argument('--count', type=int, default=4, help='Number of images to generate')
    parser.add_argument('--width', type=int, default=3840)
    parser.add_argument('--height', type=int, default=2160)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--out', type=str, default='output')
    parser.add_argument('--camera', action='store_true', help='Use camera frame as entropy (optional)')
    args = parser.parse_args()

    cfg = GenConfig(width=args.width, height=args.height, seed=args.seed, use_camera_entropy=args.camera, out_dir=args.out)
    cfg_dict = asdict(cfg)
    tasks = list(range(args.count))
    with mp.Pool(processes=min(mp.cpu_count(), args.count)) as pool:
        results = pool.map(partial(worker_task, cfg_dict=cfg_dict), tasks)
    print("Generated:", results)

if __name__ == '__main__':
    main()
