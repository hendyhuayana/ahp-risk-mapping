"""
Multicriteria (AHP) risk mapping for climate hazards.

Computes criterion weights with the Analytic Hierarchy Process,
checks consistency, normalizes raster criteria and builds a
classified risk index (1 = very low ... 5 = very high).

Usage:
    python mcda_risk.py --demo
    python mcda_risk.py --config config.json
"""
import argparse
import json

import numpy as np

# Saaty's random consistency index
RANDOM_INDEX = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24,
                7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
CLASS_NAMES = ["Very low", "Low", "Medium", "High", "Very high"]


def ahp_weights(matrix):
    """Return weights, lambda_max, CI and CR for a pairwise comparison matrix."""
    m = np.asarray(matrix, dtype=float)
    n = m.shape[0]
    if m.shape != (n, n):
        raise ValueError("Pairwise matrix must be square")
    if not np.allclose(m * m.T, 1.0, atol=1e-3):
        raise ValueError("Matrix must be reciprocal: a_ij = 1 / a_ji")
    values, vectors = np.linalg.eig(m)
    i = int(np.argmax(values.real))
    w = np.abs(vectors[:, i].real)
    w = w / w.sum()
    lam = float(values[i].real)
    ci = (lam - n) / (n - 1) if n > 2 else 0.0
    cr = ci / RANDOM_INDEX[n] if RANDOM_INDEX.get(n) else 0.0
    return w, lam, ci, cr


def normalize(arr, invert=False):
    """Min-max scale to 0-1, ignoring NaN. invert=True for criteria where low = more risk."""
    lo, hi = np.nanmin(arr), np.nanmax(arr)
    out = (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr, dtype=float)
    return 1.0 - out if invert else out


def weighted_overlay(layers, weights):
    stack = np.stack(layers)
    return np.tensordot(weights, stack, axes=1)


def classify(index, n_classes=5, method="quantile"):
    """Classes 1..n. method: 'quantile' (equal area) or 'equal' (equal interval). NaN stays NaN."""
    if method == "quantile":
        breaks = np.nanquantile(index, np.linspace(0, 1, n_classes + 1)[1:-1])
    else:
        lo, hi = np.nanmin(index), np.nanmax(index)
        breaks = np.linspace(lo, hi, n_classes + 1)[1:-1]
    classes = np.digitize(index, breaks).astype(float) + 1
    classes[np.isnan(index)] = np.nan
    return classes


def read_raster(path):
    import rasterio  # only needed for real data
    with rasterio.open(path) as src:
        arr = src.read(1).astype(float)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        return arr, src.profile


def write_raster(path, arr, profile):
    import rasterio
    profile = profile.copy()
    profile.update(dtype="float32", count=1, nodata=-9999, compress="lzw")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.where(np.isnan(arr), -9999, arr).astype("float32"), 1)


def report(names, weights, lam, ci, cr):
    print("Criterion weights (AHP)")
    for name, w in zip(names, weights):
        print(f"  {name:<22} {w:.3f}")
    print(f"lambda_max = {lam:.3f}  CI = {ci:.3f}  CR = {cr:.3f}")
    print("Consistency OK (CR < 0.10)" if cr < 0.10 else
          "WARNING: CR >= 0.10, review the pairwise judgements")


def summary(classes):
    valid = classes[~np.isnan(classes)]
    print("Area share by class")
    for k, name in enumerate(CLASS_NAMES, start=1):
        print(f"  {name:<10} {np.mean(valid == k) * 100:5.1f} %")


def run_demo():
    rng = np.random.default_rng(42)
    y, x = np.mgrid[0:200, 0:200]
    criteria = {
        "rainfall_anomaly": np.sin(x / 30) + rng.normal(0, 0.2, x.shape),
        "slope": (y / 200) ** 1.5 + rng.normal(0, 0.05, x.shape),
        "distance_to_river": np.hypot(x - 100, y - 60),
        "soil_saturation": np.cos(y / 40) + rng.normal(0, 0.2, x.shape),
    }
    invert = {"distance_to_river": True}
    matrix = [[1, 3, 5, 3],
              [1/3, 1, 3, 1],
              [1/5, 1/3, 1, 1/3],
              [1/3, 1, 3, 1]]
    names = list(criteria)
    w, lam, ci, cr = ahp_weights(matrix)
    report(names, w, lam, ci, cr)
    layers = [normalize(criteria[n], invert.get(n, False)) for n in names]
    classes = classify(weighted_overlay(layers, w), method="equal")
    summary(classes)
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(["#1a9850", "#a6d96a", "#fee08b", "#f46d43", "#a50026"])
        plt.figure(figsize=(6, 5))
        im = plt.imshow(classes, cmap=cmap, vmin=0.5, vmax=5.5)
        cb = plt.colorbar(im, ticks=range(1, 6))
        cb.ax.set_yticklabels(CLASS_NAMES)
        plt.title("Demo risk index (synthetic data)")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig("demo_risk.png", dpi=120)
        print("Saved demo_risk.png")
    except ImportError:
        pass


def run_config(path):
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    names = [c["name"] for c in cfg["criteria"]]
    w, lam, ci, cr = ahp_weights(cfg["pairwise_matrix"])
    report(names, w, lam, ci, cr)
    layers, profile = [], None
    for c in cfg["criteria"]:
        arr, prof = read_raster(c["path"])
        if profile is None:
            profile = prof
        elif arr.shape != (profile["height"], profile["width"]):
            raise ValueError(f"{c['path']} is not aligned with the first raster")
        layers.append(normalize(arr, c.get("invert", False)))
    index = weighted_overlay(layers, w)
    classes = classify(index, cfg.get("n_classes", 5), cfg.get("classification", "quantile"))
    write_raster(cfg.get("output_index", "risk_index.tif"), index, profile)
    write_raster(cfg.get("output_classes", "risk_classes.tif"), classes, profile)
    summary(classes)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--demo", action="store_true", help="run with synthetic data")
    p.add_argument("--config", help="JSON config with criteria rasters and AHP matrix")
    a = p.parse_args()
    if a.config:
        run_config(a.config)
    else:
        run_demo()
