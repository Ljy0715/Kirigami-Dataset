import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# ====================== USER CONFIG ======================
FILES = [
    r"Simulation Data/Circle_Cut_patch_results_LongE_PR_cleaned.xlsx",
    r"Simulation Data/Inclined_Cut_patch_results_LongE_PR_cleaned.xlsx",
    r"Simulation Data/Rectangle_patch_results_LongE_PR.xlsx",
    r"Simulation Data/Rotate_Triangle_patch_results_LongE_PR_cleaned.xlsx",
    r"Simulation Data/Rotate_Rectangle_patch_results_LongE_PR_cleaned.xlsx",
    r"Simulation Data/Rotate_Circle_patch_results_LongE_PR_cleaned.xlsx",
    r"Simulation Data/Cross_Slash_patch_results_LongE_PR_cleaned.xlsx",
    r"Simulation Data/jyotshna_patch_results_LongE_PR_cleaned.xlsx",
]

SCALE_AB = False         # keep False; a,b are already percentages in your files
SAVE_PNG = False         # Save per-file plot PNGs
SHOW_LEGEND = False      # Legend can be huge; keep False unless needed
MAX_LEGEND_COLS = 6
DPI = 220
N_FIT_POINTS = 200       # smoothness of fitted curve
IMG_SIZE = 50            # All images are 50x50 as requested
IMG_FOLDER = "generated_images_50"  # where pattern PNGs are saved
# =========================================================


# -------------- utilities: data cleaning & fitting --------------
def _nice_num(x):
    try:
        xf = float(x)
        return str(int(xf)) if xf.is_integer() else f"{xf:g}"
    except Exception:
        return str(x)

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map_lower = {c.lower(): c for c in df.columns}

    def find_col(cands, req=True):
        for c in cands:
            if c.lower() in col_map_lower:
                return col_map_lower[c.lower()]
        if req:
            raise KeyError(f"Missing any of: {cands}")
        return None

    col_a  = find_col(["a"])
    col_b  = find_col(["b"])
    col_E  = find_col(["Longitudinal_Strain", "Longitudinal Strain", "LongE", "E"])
    col_pr = find_col(["Poisson_Ratio", "Poisson Ratio", "PR"])
    col_theta = find_col(["theta","θ","angle_deg","angle (deg)","angle","theta_deg","rotation","rot_angle"], req=False)

    out = df.rename(columns={
        col_a: "a", col_b: "b", col_E: "Longitudinal_Strain", col_pr: "Poisson_Ratio"
    })
    if col_theta is not None:
        out = out.rename(columns={col_theta: "theta"})
    return out

def clean_df(df: pd.DataFrame, scale_ab: bool) -> pd.DataFrame:
    df = standardize_columns(df).copy()

    # Do NOT rescale a,b here; they are percentages already
    if scale_ab:
        # (kept for compatibility, but generally leave SCALE_AB=False)
        df["a"] = df["a"] * 1
        df["b"] = df["b"] * 1

    # remove rows where E == 0 OR PR is NaN
    df = df[~((df["Longitudinal_Strain"] == 0) | (df["Poisson_Ratio"].isna()))]

    for c in ["a","b","Longitudinal_Strain","Poisson_Ratio","theta"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    keep = ["a","b","Longitudinal_Strain","Poisson_Ratio"]
    if "theta" in df.columns:
        keep.append("theta")
    df = df.dropna(subset=keep)
    return df

def fit_quadratic(E, PR):
    coeffs = np.polyfit(E, PR, 2)
    c2, c1, c0 = coeffs
    return float(c0), float(c1), float(c2)

# -------------------- utilities: drawing --------------------
def draw_line(image, x1, y1, x2, y2):
    """Bresenham-like integer line on numpy array (1=white)."""
    x1=int(round(x1)); y1=int(round(y1)); x2=int(round(x2)); y2=int(round(y2))
    n = max(abs(x2-x1), abs(y2-y1))
    if n <= 0:
        if 0<=x1<image.shape[1] and 0<=y1<image.shape[0]: image[y1,x1] = 1
        return image
    xs = np.round(np.linspace(x1, x2, n+1)).astype(int)
    ys = np.round(np.linspace(y1, y2, n+1)).astype(int)
    h, w = image.shape
    mask = (xs>=0)&(xs<w)&(ys>=0)&(ys<h)
    image[ys[mask], xs[mask]] = 1
    return image

def draw_polyline(image, xs, ys):
    xs = np.round(xs).astype(int); ys = np.round(ys).astype(int)
    for i in range(len(xs)-1):
        image = draw_line(image, xs[i], ys[i], xs[i+1], ys[i+1])
    return image

# ---------- percent -> pixels (key change you requested) ----------
def pct_to_px(val_pct: float, img_size: int = IMG_SIZE) -> float:
    """Convert percentage of whole image to pixels on a IMG_SIZE grid."""
    return (float(val_pct) / 100.0) * img_size

# ---------------- pattern generators (use pixel units) ----------------
# def gen_inclined(a_percent, b_percent, theta_deg, img_size=IMG_SIZE):
#     """
#     a_percent, b_percent: 切口长度的百分比 (0~100)
#     theta_deg: 旋转角度 (度)
#     img_size: 图像大小 (像素)
#     """
#     # 把百分比转换为像素长度
#     a_px = (a_percent / 100.0) * img_size
#     b_px = (b_percent / 100.0) * img_size
#
#     img = np.zeros((img_size, img_size), dtype=np.uint8)
#     C = img_size // 2
#     th = np.deg2rad(theta_deg)
#     th_b = -th + np.pi / 2
#
#     # 1) 中间长切口 (b_px 长度)
#     dx_b = int(round((b_px / 2) * np.cos(th_b)))
#     dy_b = int(round((b_px / 2) * np.sin(th_b)))
#     x1, y1 = C - dx_b, C - dy_b
#     x2, y2 = C + dx_b, C + dy_b
#     img = draw_line(img, x1, y1, x2, y2)
#
#     # 2) 左下短切口 (a_px 长度)
#     dx_a = int(round((a_px / 2) * np.cos(th)))
#     dy_a = int(round((a_px / 2) * np.sin(th)))
#     x1, y1 = 0, img_size - 1
#     x2, y2 = x1 + dx_a, y1 - dy_a
#     img = draw_line(img, x1, y1, x2, y2)
#
#     # 3) 右上短切口 (对称)
#     x3, y3 = img_size - 1, 0
#     x4, y4 = x3 - dx_a, y3 + dy_a
#     img = draw_line(img, x3, y3, x4, y4)
#
#     # --- 反转像素：切口黑色，背景白色 ---
#     return img

def gen_inclined(a_px, b_px, theta_deg, img_size=IMG_SIZE):
    """
    a_px, b_px: 切口长度的像素值
    theta_deg: 旋转角度 (度)
    img_size: 图像大小 (像素)
    """
    img = np.zeros((img_size, img_size), dtype=np.uint8)
    C = img_size // 2
    th = np.deg2rad(theta_deg)
    th_b = -th + np.pi / 2

    # 1) 中间长切口 (b_px 长度)
    dx_b = int(round((b_px / 2) * np.cos(th_b)))
    dy_b = int(round((b_px / 2) * np.sin(th_b)))
    x1, y1 = C - dx_b, C - dy_b
    x2, y2 = C + dx_b, C + dy_b
    img = draw_line(img, x1, y1, x2, y2)

    # 2) 左下短切口 (a_px 长度)
    dx_a = int(round((a_px / 2) * np.cos(th)))
    dy_a = int(round((a_px / 2) * np.sin(th)))
    x1, y1 = 0, img_size - 1
    x2, y2 = x1 + dx_a, y1 - dy_a
    img = draw_line(img, x1, y1, x2, y2)

    # 3) 右上短切口 (对称)
    x3, y3 = img_size - 1, 0
    x4, y4 = x3 - dx_a, y3 + dy_a
    img = draw_line(img, x3, y3, x4, y4)
    # plt.imshow(img, cmap="gray", origin="upper", vmin=0, vmax=1)
    # plt.show()

    return img



def gen_circle_cut_radius(radius_px, img_size=IMG_SIZE):
    """Filled circle with radius in pixels (centered)."""
    img = np.zeros((img_size, img_size), dtype=np.uint8)
    cx = (img_size-1)/2.0
    cy = (img_size-1)/2.0
    yy, xx = np.meshgrid(np.arange(img_size), np.arange(img_size), indexing='ij')
    dist = np.sqrt((xx-cx)**2 + (yy-cy)**2)
    img[dist <= float(radius_px/2)] = 1
    return img

def gen_cross_slash(a_px, b_px, theta_deg, img_size=IMG_SIZE):
    img = np.zeros((img_size, img_size), dtype=np.uint8)
    C = img_size//2
    th = np.deg2rad(theta_deg)
    th_perp = th + np.pi/2

    # horizontal-ish at angle th, length a_px
    Lh = a_px/2
    dx1 = int(round(Lh*np.cos(th)))
    dy1 = int(round(Lh*np.sin(th)))
    img = draw_line(img, C-dx1, C+dy1, C+dx1, C-dy1)

    # vertical-ish at angle th+90, length b_px
    Lv = b_px/2
    dx2 = int(round(Lv*np.cos(th_perp)))
    dy2 = int(round(Lv*np.sin(th_perp)))
    img = draw_line(img, C-dx2, C+dy2, C+dx2, C-dy2)
    return img

def gen_jyotshna(a_px, b_px, img_size=IMG_SIZE):
    img = np.zeros((img_size, img_size), dtype=np.uint8)
    C = img_size//2

    # center vertical slit length b_px
    dy = int(round(b_px/2))
    img = draw_line(img, C, C-dy, C, C+dy)

    # top & bottom horizontal length a_px
    half_a = int(round(a_px/2))
    # img = draw_line(img, C-half_a, 0, C+half_a, 0)
    img = draw_line(img, C-half_a, img_size-1, C+half_a, img_size-1)

    # middle left/right a_px/2
    img = draw_line(img, 0, C, max(0, half_a), C)
    img = draw_line(img, (img_size-1)-half_a, C, img_size-1, C)

    # four corners b_px/2 vertical
    half_b = int(round(b_px/2))
    # left-top
    img = draw_line(img, 0, 0, 0, min(img_size-1, 0 + half_b))
    # # right-top
    # img = draw_line(img, img_size-1, 0, img_size-1, min(img_size-1, 0 + half_b))
    # left-bottom
    img = draw_line(img, 0, max(0, img_size-1-half_b), 0, img_size-1)
    # # right-bottom
    # img = draw_line(img, img_size-1, max(0, img_size-1-half_b), img_size-1, img_size-1)
    return img

def gen_rectangle_filled(a_px, b_px, theta_deg, img_size=IMG_SIZE):
    # rotated filled rectangle width=a_px height=b_px
    img = np.zeros((img_size, img_size), dtype=np.uint8)
    cx = (img_size-1)/2.0; cy = (img_size-1)/2.0
    th = np.deg2rad(theta_deg)
    R = np.array([[np.cos(-th), -np.sin(-th)],[np.sin(-th), np.cos(-th)]])
    yy, xx = np.meshgrid(np.arange(img_size), np.arange(img_size), indexing='ij')
    x = xx - cx; y = yy - cy
    xr = R[0,0]*x + R[0,1]*y
    yr = R[1,0]*x + R[1,1]*y
    img[(np.abs(xr)<=a_px/2) & (np.abs(yr)<=b_px/2)] = 1
    return img

def gen_rotate_circle(a_px, b_px, theta_deg_unused, img_size=IMG_SIZE):
    """Two arcs: top centered at (-b_px, 0) and lower at (b_px, -b_px), radius=a_px."""
    img = np.zeros((img_size, img_size), dtype=np.uint8)
    Cx = (img_size-1)/2.0; Cy = (img_size-1)/2.0
    R = float(a_px)
    c_top = np.array([-float(b_px), 0.0])
    c_low = np.array([ float(b_px),-float(b_px)])

    n_pts = max(80, int(round(np.pi*max(1,R))))
    ang_top = np.linspace(0, np.pi, n_pts)
    ang_low = np.linspace(np.pi, 2*np.pi, n_pts)
    top_xy = np.stack([c_top[0] + R*np.cos(ang_top),   c_top[1] + R*np.sin(ang_top)], axis=0)
    low_xy = np.stack([c_low[0] + R*np.cos(ang_low),   c_low[1] + R*np.sin(ang_low)], axis=0)

    img = draw_polyline(img, Cx + top_xy[0],  Cy - top_xy[1])
    img = draw_polyline(img, Cx + low_xy[0],  Cy - low_xy[1])
    return img

def gen_rotate_rectangle(a_px, b_px, theta_deg, img_size=IMG_SIZE):
    img = np.zeros((img_size, img_size), dtype=np.uint8)
    Cx = (img_size-1)/2.0; Cy = (img_size-1)/2.0
    L = a_px/2.0
    th = np.deg2rad(theta_deg)
    Rm = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])

    # edges with b-directional tangential shifts per MATLAB logic
    top_p1   = np.array([-L - b_px, +L]); top_p2   = np.array([+L - b_px, +L])
    right_p1 = np.array([+L, -L + b_px]); right_p2 = np.array([+L, +L + b_px])
    bot_p1   = np.array([-L + b_px, -L]); bot_p2   = np.array([+L + b_px, -L])
    left_p1  = np.array([-L, -L - b_px]); left_p2  = np.array([-L, +L - b_px])

    segs = [(top_p1, top_p2),(right_p1,right_p2),(bot_p1,bot_p2),(left_p1,left_p2)]
    for p1, p2 in segs:
        P = np.stack([p1, p2], axis=1)            # 2x2
        Prot = Rm @ P
        x1 = int(round(Cx + Prot[0,0])); y1 = int(round(Cy - Prot[1,0]))
        x2 = int(round(Cx + Prot[0,1])); y2 = int(round(Cy - Prot[1,1]))
        img = draw_line(img, x1, y1, x2, y2)
    return img

def gen_rotate_triangle(a_px, b_px, theta_deg, img_size=IMG_SIZE):
    img = np.zeros((img_size, img_size), dtype=np.uint8)
    Cx = (img_size-1)/2.0; Cy = (img_size-1)/2.0
    h = a_px*np.sqrt(3)/2

    vL = np.array([-a_px/2, 0.0])
    vR = np.array([ a_px/2, 0.0])
    vB = np.array([ 0.0,   -h])

    centroid = (vL + vR + vB)/3.0
    vL -= centroid; vR -= centroid; vB -= centroid

    top   = np.stack([vL, vR], axis=0)
    right = np.stack([vB, vR], axis=0)
    left  = np.stack([vL, vB], axis=0)

    # translate along tangents by b_px
    t_top = np.array([-1.0, 0.0]); t_top /= np.linalg.norm(t_top)
    top = top + b_px*t_top

    t_right = (vR - vB); t_right /= np.linalg.norm(t_right)
    right = right + b_px*t_right

    t_left = (vB - vL); t_left /= np.linalg.norm(t_left)
    left = left + b_px*t_left

    th = np.deg2rad(theta_deg)
    Rm = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
    for seg in [top, right, left]:
        P = (Rm @ seg.T).T
        x1 = int(round(Cx + P[0,0])); y1 = int(round(Cy - P[0,1]))
        x2 = int(round(Cx + P[1,0])); y2 = int(round(Cy - P[1,1]))
        img = draw_line(img, x1, y1, x2, y2)
    return img

# -------------- choose generator by filename (convert % -> px here) --------------
def make_pattern_image(file_stem: str, a_percent: float, b_percent: float, theta: float|None, img_size=IMG_SIZE):
    name = file_stem.lower()

    # Convert percentages to pixels on IMG_SIZE grid
    a_px = pct_to_px(a_percent, img_size)
    b_px = pct_to_px(b_percent, img_size)

    # Clamp to sensible bounds inside the image
    def clamp_len(v): return max(1.0, min(v, img_size - 2))
    a_px = clamp_len(a_px)
    b_px = clamp_len(b_px)

    if "circle_cut" in name and "rotate" not in name:
        # Interpret 'a' as the circle radius percentage of the whole image size
        return gen_circle_cut_radius(a_px, img_size)

    if "incline" in name or "inclined" in name:
        th = 0.0 if theta is None else theta
        return gen_inclined(a_px, b_px, th, img_size)

    if "cross_slash" in name or "crossslash" in name:
        th = 0.0 if theta is None else theta
        return gen_cross_slash(a_px, b_px, th, img_size)

    if "jyotshna" in name:
        return gen_jyotshna(a_px, b_px, img_size)

    if "rectangle_patch" in name and "rotate" not in name:
        th = 0.0 if theta is None else theta
        return gen_rectangle_filled(a_px, b_px, th, img_size)

    if "rotate_circle" in name:
        # a = radius (% of whole image), b = offset (% of whole image)
        return gen_rotate_circle(a_px, b_px, 0.0, img_size)

    if "rotate_rectangle" in name:
        th = 0.0 if theta is None else theta
        return gen_rotate_rectangle(a_px, b_px, th, img_size)

    if "rotate_triangle" in name:
        th = 0.0 if theta is None else theta
        return gen_rotate_triangle(a_px, b_px, th, img_size)

    # default fallback
    th = 0.0 if theta is None else theta
    return gen_cross_slash(a_px, b_px, th, img_size)


# --------- main: fit, write coefs, generate images (paths stored) ----------
def add_coeffs_and_images(df: pd.DataFrame, file_stem: str, img_dir: Path) -> tuple[pd.DataFrame, dict]:
    """
    For each (a,b[,theta]) group (a,b are percentages of whole image):
      - fit c0,c1,c2
      - mark only the FIRST row with c0,c1,c2
      - generate 50x50 image for that parameter set, save PNG
      - put image file path into new column 'image_path' ONLY on the first row
    """
    out = df.copy()
    for c in ["c0","c1","c2","image_path"]:
        if c not in out.columns:
            out[c] = np.nan
    # ensure 'image_path' can hold strings
    out["image_path"] = out["image_path"].astype("object")

    group_keys = ["a","b"] + (["theta"] if "theta" in out.columns else [])
    out = out.sort_values(group_keys + ["Longitudinal_Strain"]).copy()

    row_images = {}
    stem = file_stem

    for keys, g in out.groupby(group_keys, sort=False):
        idx_first = g.index[0]
        E = g["Longitudinal_Strain"].to_numpy(dtype=float)
        PR = g["Poisson_Ratio"].to_numpy(dtype=float)
        c0, c1, c2 = fit_quadratic(E, PR)
        out.loc[idx_first, ["c0","c1","c2"]] = [c0, c1, c2]

        # parameters (still percentages here)
        if isinstance(keys, tuple):
            a_pct, b_pct = float(keys[0]), float(keys[1])
            theta = float(keys[2]) if len(keys) == 3 else None
        else:
            a_pct = float(keys); b_pct = float(keys)  # unlikely
            theta = None

        # generate image based on percentage → pixels conversion
        img = make_pattern_image(stem, a_pct, b_pct, theta, img_size=IMG_SIZE)

        # save PNG
        label_parts = [f"a{_nice_num(a_pct)}pct","b"+_nice_num(b_pct)+"pct"]
        if theta is not None: label_parts.append("th"+_nice_num(theta))
        fname = f"{stem}_" + "_".join(label_parts) + ".png"
        png_path = img_dir / fname
        Image.fromarray((img*255).astype(np.uint8)).save(png_path)

        out.loc[idx_first, "image_path"] = str(png_path)
        row_images[idx_first] = str(png_path)

    return out, row_images

def plot_scatter_and_fit(df: pd.DataFrame, xlsx: Path):
    group_keys = ["a","b"] + (["theta"] if "theta" in df.columns else [])
    groups = df.groupby(group_keys, sort=True)
    n_groups = len(groups)

    plt.figure(figsize=(12, 8))
    for keys, g in groups:
        g = g.sort_values("Longitudinal_Strain")
        E = g["Longitudinal_Strain"].to_numpy(dtype=float)
        PR = g["Poisson_Ratio"].to_numpy(dtype=float)

        if isinstance(keys, tuple):
            parts = [f"a={_nice_num(keys[0])}%", f"b={_nice_num(keys[1])}%"]
            if len(keys) == 3: parts.append(f"θ={_nice_num(keys[2])}")
            label = ", ".join(parts)
        else:
            label = f"{group_keys[0]}={_nice_num(keys)}%"

        plt.scatter(E, PR, s=16, alpha=0.9, label=None)

        c0, c1, c2 = fit_quadratic(E, PR)
        E_fit = np.linspace(E.min(), E.max(), N_FIT_POINTS)
        PR_fit = c0 + c1*E_fit + c2*(E_fit**2)
        plt.plot(E_fit, PR_fit, linewidth=1.3, label=label)

    plt.axhline(0.0, linestyle="--", linewidth=1.0)
    theta_note = " with θ" if "theta" in df.columns else ""
    plt.xlabel("Longitudinal Strain")
    plt.ylabel("Poisson's Ratio")
    plt.title(f"{xlsx.stem} — Scatter + Quadratic Fit (groups: {n_groups}{theta_note})")
    plt.tight_layout()

    if SHOW_LEGEND:
        plt.legend(fontsize=8, ncol=MAX_LEGEND_COLS, loc="upper left", bbox_to_anchor=(0, -0.1))

    if SAVE_PNG:
        out_png = xlsx.with_suffix(".png")
        plt.savefig(out_png, dpi=DPI, bbox_inches="tight")
        print(f"[OK] Saved plot → {out_png}")
        plt.close()
    else:
        plt.show()

def write_excel_with_images(df: pd.DataFrame, out_xlsx: Path, row_images: dict):
    """
    Write DataFrame with c0,c1,c2 and then embed the corresponding image
    in the NEXT column after c2 (i.e., a new column 'image') at the first-row of each group.
    Requires xlsxwriter; pip install xlsxwriter
    """
    cols = list(df.columns)
    if "image" not in cols:
        insert_pos = cols.index("c2") + 1
        cols.insert(insert_pos, "image")
        df = df.reindex(columns=cols)
        df["image"] = pd.Series([None]*len(df), dtype="object")

    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Sheet1")
        ws  = writer.sheets["Sheet1"]
        image_col_idx = df.columns.get_loc("image")
        ws.set_column(image_col_idx, image_col_idx, 18)

        for row_idx, img_path in row_images.items():
            if not img_path or not os.path.exists(img_path):
                continue
            # 在 write_excel_with_images() 里，to_excel 之前加：
            row_pos = {idx: i for i, idx in enumerate(df.index)}
            for row_idx, img_path in row_images.items():
                if not img_path or not os.path.exists(img_path):
                    continue
                excel_row = row_pos[row_idx] + 1  # ← 用表内行号，而不是原始 index
                excel_col = image_col_idx
                ws.insert_image(excel_row, excel_col, img_path, {
                    "x_scale": 0.6, "y_scale": 0.6, "positioning": 2
                })

            excel_row = row_idx + 1  # +1 for header row
            excel_col = image_col_idx
            ws.insert_image(excel_row, excel_col, img_path, {
                "x_scale": 0.6, "y_scale": 0.6, "positioning": 2
            })
        ws.write(0, image_col_idx, "image")

# ============================== MAIN ==============================
if __name__ == "__main__":
    os.makedirs(IMG_FOLDER, exist_ok=True)
    img_dir = Path(IMG_FOLDER)

    for f in FILES:
        xlsx = Path(f)
        if not xlsx.exists():
            print(f"[WARN] File not found: {xlsx}")
            continue

        # 1) read & clean
        raw = pd.read_excel(xlsx)
        df = clean_df(raw, SCALE_AB)

        # 2) add coefs on FIRST row of each group + generate & record image path
        df_with, row_images = add_coeffs_and_images(df, xlsx.stem, img_dir)

        # 3) write new Excel with embedded images in the next column to coefficients
        out_xlsx = xlsx.with_name(xlsx.stem + "_with_coefs_and_images.xlsx")
        write_excel_with_images(df_with, out_xlsx, row_images)
        print(f"[OK] Wrote coefficients + embedded images → {out_xlsx}")

        # 4) plot scatter + fit (one figure per file)
        plot_scatter_and_fit(df_with, xlsx)
