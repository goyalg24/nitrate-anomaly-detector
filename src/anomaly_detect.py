#!/usr/bin/env python3
"""
Sliding-window percentile anomaly detection for Nitrate time series.
- Fixed window size W, step=1, overlapping
- Percentile thresholds via numpy.percentile(..., method="linear")
- Labels only the newly added point for windows i>=2 (and optionally for the first window as well)
- Evaluates vs ground truth (expects exactly 77 anomaly events unless your gt column differs)
- Saves plot, metrics, and (optionally) a README.md filled from a template.

Usage:
  python src/anomaly_detect.py --data data/data.csv --value-col NO3N --gt-col Student_Flag \
      --W 1200 --q 99.0 --one-sided upper --initial-label newonly --outdir outputs

Optional grid search:
  python src/anomaly_detect.py --data data/data.csv --value-col NO3N --gt-col Student_Flag \
      --sweep --W-min 500 --W-max 2000 --W-step 250 --q-grid 97,98,98.5,99,99.5 \
      --target-normal 0.80 --target-anomaly 0.75

Author: Gaurav Goyal
"""
import argparse
import os
import json
from pathlib import Path
from typing import Tuple, Dict, Optional, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Matplotlib default style constraints per environment: avoid seaborn, keep single plots
plt.rcParams.update({'figure.dpi': 140})

TARGET_TOTAL_ANOMALIES = 77  # Provided spec

# Default locations, resolved relative to the repository root (parent of src/)
# so the script finds data/ and writes to outputs/ wherever it is run from.
REPO_ROOT = Path(__file__).resolve().parent.parent


def _default_path(p: Path) -> str:
    """Return p relative to the current directory when possible (keeps paths
    recorded in run JSON / README short, e.g. 'outputs/...'), else absolute."""
    try:
        return os.path.relpath(p)
    except ValueError:  # e.g. different drive on Windows
        return str(p)


DEFAULT_DATA = _default_path(REPO_ROOT / "data" / "data.csv")
DEFAULT_OUTDIR = _default_path(REPO_ROOT / "outputs")

def parse_args():
    p = argparse.ArgumentParser(description="Sliding-window percentile anomaly detection")
    p.add_argument("--data", type=str, default=DEFAULT_DATA, help="Path to CSV (default: data/data.csv)")
    p.add_argument("--value-col", type=str, default="Nitrate", help="Column name for the time series values")
    p.add_argument("--gt-col", type=str, default=None, help="Ground-truth column (0/1). If not given, attempts auto-detect.")
    p.add_argument("--index-col", type=str, default=None, help="Optional index/time column for plotting")
    p.add_argument("--W", type=int, default=1200, help="Window size (e.g., 500–2000). Used unless --sweep given.")
    p.add_argument("--q", type=float, default=99.0, help="Percentile for threshold (upper-tail)")
    p.add_argument("--two-sided", action="store_true", help="Use two-sided thresholds instead of one-sided")
    p.add_argument("--q-low", type=float, default=1.0, help="Lower-tail percentile if two-sided")
    p.add_argument("--one-sided", type=str, choices=["upper", "lower"], default="upper",
                   help="If not --two-sided, choose which tail to flag as anomalies")
    p.add_argument("--initial-label", type=str, choices=["all", "newonly"], default="newonly",
                   help='For the first window: label "all" points or only the newly added one ("newonly")')
    p.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR, help="Directory for outputs (default: outputs/)")
    p.add_argument("--save-readme", action="store_true", help="Fill README.md from template after run")
    # Sweep / search options
    p.add_argument("--sweep", action="store_true", help="Search over W and q grids")
    p.add_argument("--W-min", type=int, default=500)
    p.add_argument("--W-max", type=int, default=2000)
    p.add_argument("--W-step", type=int, default=250)
    p.add_argument("--q-grid", type=str, default="97,98,98.5,99,99.5",
                   help="Comma-separated percentiles to try for upper-tail; e.g., '95,97,98,99,99.5'")
    p.add_argument("--target-normal", type=float, default=0.80, help="Target TN/N")
    p.add_argument("--target-anomaly", type=float, default=0.75, help="Target TP/P")
    return p.parse_args()


def load_data(path: str, value_col: str, gt_col: Optional[str]) -> Tuple[pd.Series, Optional[pd.Series], Optional[pd.Series]]:
    df = pd.read_csv(path)
    # Try to coerce numeric
    if value_col not in df.columns:
        raise ValueError(f"Value column '{value_col}' not found. Available: {list(df.columns)}")
    x = pd.to_numeric(df[value_col], errors="coerce")
    # Ground truth autodetect if needed
    gt = None
    if gt_col is not None and gt_col in df.columns:
        gt = pd.to_numeric(df[gt_col], errors="coerce").fillna(0).astype(int)
    else:
        candidates = ["ground_truth", "GroundTruth", "Label", "label", "is_anomaly", "Anomaly", "anomaly"]
        for c in candidates:
            if c in df.columns:
                gt = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
                gt_col = c
                break
    # Optional index/time column for plotting
    time_col_series = None
    # If a distinct index/time column exists (not mandatory)
    # We won't raise if missing; index will be integer positions
    return x.reset_index(drop=True), (gt.reset_index(drop=True) if gt is not None else None), time_col_series


def sliding_percentile_detect(
    x: np.ndarray,
    W: int,
    q: float,
    two_sided: bool = False,
    q_low: float = 1.0,
    one_sided: str = "upper",
    initial_label: str = "newonly",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      preds: np.ndarray of shape (n,), 0/1 predicted labels (1 = anomaly)
      thresholds: np.ndarray of shape (n,), per-point threshold used (np.nan where undefined)
    """
    n = len(x)
    preds = np.zeros(n, dtype=int)
    thresholds = np.full(n, np.nan, dtype=float)

    if n < W:
        raise ValueError(f"Series length n={n} is smaller than window size W={W}.")

    # First window
    w0 = x[0:W]
    if two_sided:
        t_low = np.percentile(w0, q_low, method="linear")
        t_high = np.percentile(w0, q, method="linear")
        T1 = (t_low, t_high)
    else:
        T1 = np.percentile(w0, q, method="linear")
    # Labeling for the first window
    if initial_label == "all":
        if two_sided:
            low, high = T1
            preds[0:W] = ((x[0:W] <= low) | (x[0:W] >= high)).astype(int)
            thresholds[0:W] = high  # store upper threshold for reference
        else:
            if one_sided == "upper":
                preds[0:W] = (x[0:W] >= T1).astype(int)
            else:
                preds[0:W] = (x[0:W] <= T1).astype(int)
            thresholds[0:W] = T1
    else:
        # Label only the newly added point for the first window => index W-1
        idx = W - 1
        if two_sided:
            low, high = T1
            preds[idx] = int((x[idx] <= low) or (x[idx] >= high))
            thresholds[idx] = high
        else:
            if one_sided == "upper":
                preds[idx] = int(x[idx] >= T1)
            else:
                preds[idx] = int(x[idx] <= T1)
            thresholds[idx] = T1
        # prior points 0..W-2 remain 0 (normal) by convention

    # Subsequent windows: slide by 1, label only the new point at i+W-1
    for i in range(1, n - W + 1):
        window = x[i : i + W]
        if two_sided:
            t_low = np.percentile(window, q_low, method="linear")
            t_high = np.percentile(window, q, method="linear")
            low, high = t_low, t_high
        else:
            T = np.percentile(window, q, method="linear")

        new_idx = i + W - 1
        if two_sided:
            preds[new_idx] = int((x[new_idx] <= low) or (x[new_idx] >= high))
            thresholds[new_idx] = high
        else:
            if one_sided == "upper":
                preds[new_idx] = int(x[new_idx] >= T)
            else:
                preds[new_idx] = int(x[new_idx] <= T)
            thresholds[new_idx] = T

    return preds, thresholds


def compute_metrics(preds: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    # Ensure binary ints
    preds = preds.astype(int)
    gt = gt.astype(int)
    if len(preds) != len(gt):
        m = min(len(preds), len(gt))
        preds = preds[:m]
        gt = gt[:m]
    TP = int(((preds == 1) & (gt == 1)).sum())
    FP = int(((preds == 1) & (gt == 0)).sum())
    FN = int(((preds == 0) & (gt == 1)).sum())
    TN = int(((preds == 0) & (gt == 0)).sum())
    P = TP + FN
    N = TN + FP
    normal_acc = TN / N if N > 0 else float("nan")
    anomaly_acc = TP / P if P > 0 else float("nan")
    return {"TP": TP, "FP": FP, "FN": FN, "TN": TN, "P": P, "N": N,
            "normal_acc": normal_acc, "anomaly_acc": anomaly_acc}


def plot_results(idx: np.ndarray, x: np.ndarray, preds: np.ndarray, gt: Optional[np.ndarray], outpath: str):
    plt.figure()
    plt.plot(idx, x, linewidth=1.0, label="Nitrate")
    # Predicted anomalies
    pa = np.where(preds == 1)[0]
    if pa.size > 0:
        plt.scatter(idx[pa], x[pa], s=12, marker='o', label="Predicted anomaly")
    # Ground truth anomalies
    if gt is not None:
        ga = np.where(gt == 1)[0]
        if ga.size > 0:
            plt.scatter(idx[ga], x[ga], s=25, marker='x', label="Ground truth anomaly")
    plt.xlabel("Index")
    plt.ylabel("Nitrate")
    plt.title("Sliding-window Percentile Anomaly Detection")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def try_one_run(x: pd.Series, gt: Optional[pd.Series], W: int, q: float,
                two_sided: bool, q_low: float, one_sided: str, initial_label: str,
                outdir: str, tag: str) -> Dict:
    os.makedirs(outdir, exist_ok=True)
    xn = x.to_numpy()
    preds, thresholds = sliding_percentile_detect(
        xn, W=W, q=q, two_sided=two_sided, q_low=q_low,
        one_sided=one_sided, initial_label=initial_label
    )
    metrics = None
    if gt is not None:
        metrics = compute_metrics(preds, gt.to_numpy())
    # Save plot
    figpath = os.path.join(outdir, f"anomaly_plot_{tag}.png")
    idx = np.arange(len(xn))
    plot_results(idx, xn, preds, (gt.to_numpy() if gt is not None else None), figpath)
    # Save predictions
    out_csv = os.path.join(outdir, f"predictions_{tag}.csv")
    out_df = pd.DataFrame({
        "index": idx,
        "value": xn,
        "pred": preds,
        "threshold": thresholds
    })
    if gt is not None:
        out_df["gt"] = gt.values
    out_df.to_csv(out_csv, index=False)
    result = {
        "W": W, "q": q, "two_sided": two_sided, "q_low": q_low, "one_sided": one_sided,
        "initial_label": initial_label, "plot": figpath, "pred_csv": out_csv,
        "metrics": metrics
    }
    with open(os.path.join(outdir, f"run_{tag}.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result


def sweep_search(x: pd.Series, gt: pd.Series, args) -> Dict:
    q_vals = [float(s.strip()) for s in args.q_grid.split(",")]
    W_vals = list(range(args.W_min, args.W_max + 1, args.W_step))
    results: List[Dict] = []
    for W in W_vals:
        for q in q_vals:
            tag = f"W{W}_q{q}"
            res = try_one_run(x, gt, W, q, args.two_sided, args.q_low, args.one_sided,
                              args.initial_label, args.outdir, tag)
            res["tag"] = tag
            results.append(res)

    # Rank: prefer configs meeting targets; otherwise maximize harmonic mean of accuracies
    def score(m):
        if m is None:
            return -1.0
        na = m["normal_acc"]
        aa = m["anomaly_acc"]
        if np.isnan(na) or np.isnan(aa):
            return -1.0
        # If meets targets, add +1 to prioritize, then HM
        meets = (na >= args.target_normal) and (aa >= args.target_anomaly)
        hm = 2*na*aa/(na+aa) if (na+aa) > 0 else 0.0
        return hm + (1.0 if meets else 0.0)

    best = max(results, key=lambda r: score(r["metrics"]) if r["metrics"] else -1.0)
    # Save sweep summary
    rows = []
    for r in results:
        m = r["metrics"] or {}
        rows.append({
            "tag": r.get("tag"),
            "W": r["W"],
            "q": r["q"],
            "normal_acc": m.get("normal_acc"),
            "anomaly_acc": m.get("anomaly_acc"),
            "TP": m.get("TP"), "FP": m.get("FP"), "FN": m.get("FN"), "TN": m.get("TN"),
            "plot": r["plot"], "pred_csv": r["pred_csv"]
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.outdir, "sweep_results.csv"), index=False)
    return {"best": best, "all_results_csv": os.path.join(args.outdir, "sweep_results.csv")}


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    x, gt, _ = load_data(args.data, args.value_col, args.gt_col)

    if gt is not None:
        P = int(gt.sum())
        if P != TARGET_TOTAL_ANOMALIES:
            print(f"[WARN] Ground-truth positives sum to {P}, expected {TARGET_TOTAL_ANOMALIES}. Proceeding anyway.")

    if args.sweep:
        if gt is None:
            raise ValueError("Ground-truth column required for --sweep evaluation.")
        search_res = sweep_search(x, gt, args)
        best = search_res["best"]
        best_tag = best.get("tag", "best")
        print("=== Best configuration from sweep ===")
        print(json.dumps(best, indent=2))
        # Optionally write README if requested
        if args.save_readme:
            write_readme(args, x, gt, best, sweep_csv=os.path.join(args.outdir, "sweep_results.csv"))
        return

    # Single run
    tag = f"W{args.W}_q{args.q}_{'twoSided' if args.two_sided else args.one_sided}_{args.initial_label}"
    res = try_one_run(x, gt, args.W, args.q, args.two_sided, args.q_low,
                      args.one_sided, args.initial_label, args.outdir, tag)
    print("=== Run summary ===")
    print(json.dumps(res, indent=2))

    if args.save_readme:
        write_readme(args, x, gt, res, sweep_csv=None)


def write_readme(args, x: pd.Series, gt: Optional[pd.Series], run_result: Dict, sweep_csv: Optional[str] = None):
    # Fill README_TEMPLATE.md into README.md
    template_path = os.path.join(os.path.dirname(__file__), "README_TEMPLATE.md")
    if not os.path.exists(template_path):
        print("[WARN] README_TEMPLATE.md not found; skipping README creation.")
        return
    with open(template_path, "r") as f:
        tmpl = f.read()

    m = run_result.get("metrics") or {}
    filled = tmpl
    replacements = {
        "{{W}}": str(run_result.get("W")),
        "{{q}}": str(run_result.get("q")),
        "{{INITIAL_LABEL}}": run_result.get("initial_label", "newonly"),
        "{{RULE}}": ("two-sided (q_low={}, q_high={})".format(args.q_low, run_result.get("q"))
                     if args.two_sided else f"one-sided ({run_result.get('one_sided','upper')} tail)"),
        "{{P}}": str(m.get("P", "NA")),
        "{{N}}": str(m.get("N", "NA")),
        "{{TP}}": str(m.get("TP", "NA")),
        "{{FP}}": str(m.get("FP", "NA")),
        "{{FN}}": str(m.get("FN", "NA")),
        "{{TN}}": str(m.get("TN", "NA")),
        "{{NORMAL_ACC}}": f"{m.get('normal_acc', float('nan')):.4f}" if "normal_acc" in m else "NA",
        "{{ANOMALY_ACC}}": f"{m.get('anomaly_acc', float('nan')):.4f}" if "anomaly_acc" in m else "NA",
        "{{PLOT_PATH}}": run_result.get("plot", ""),
        "{{PRED_PATH}}": run_result.get("pred_csv", ""),
        "{{DATA_FILE}}": os.path.basename(args.data)
    }
    for k, v in replacements.items():
        filled = filled.replace(k, v)

    if sweep_csv:
        filled = filled.replace("{{SWEEP_CSV}}", sweep_csv)
    else:
        filled = filled.replace("{{SWEEP_CSV}}", "N/A")

    out_path = os.path.join(args.outdir, "README.md")
    with open(out_path, "w") as f:
        f.write(filled)
    print(f"[INFO] Wrote README to: {out_path}")


if __name__ == "__main__":
    main()
