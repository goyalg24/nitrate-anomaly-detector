# Nitrate Anomaly Detection (Sliding-Window Percentile)

A threshold-based anomaly detector for the Nitrate (`NO3N`) time series. It uses
fixed-size, overlapping sliding windows (step size = 1) and an adaptive percentile
threshold computed with `numpy.percentile(..., method="linear")`. Each threshold is
computed only from the values in the current window, so no future data leaks in.
If a ground-truth column is available, the script compares its predictions against it
(TP/FP/FN/TN, normal accuracy `TN/N` and anomaly accuracy `TP/P`). The script can also
run a grid search over the window size `W` and the percentile `q`.

## How it works

- Choose a window size `W` (e.g. 500–2000) and a percentile `q`.
- First window `[0, W-1]`: compute `T₁ = percentile(x[0:W], q)`. With
  `--initial-label newonly` (the default) only the point at index `W-1` is labeled;
  with `--initial-label all`, every point in the first window is labeled using `T₁`.
- Each later window `i` covers `[i, i+W-1]`. Its threshold `Tᵢ` is computed from that
  window only, and only the newly added point at index `i+W-1` is labeled.
- Rule: one-sided upper tail (`x ≥ T`, the default), one-sided lower tail
  (`--one-sided lower`, `x ≤ T`), or two-sided (`--two-sided`, using `--q-low` for
  the lower tail and `--q` for the upper tail).

## Project structure

```
.
├── data/
│   └── data.csv                 # Date, NO3N, Student_Flag (30,790 rows)
├── outputs/                     # results of the committed run (W=1200, q=99, upper, newonly)
│   ├── README.md                # README auto-filled from the template by --save-readme
│   ├── anomaly_plot_W1200_q99.0_upper_newonly.png
│   ├── predictions_W1200_q99.0_upper_newonly.csv
│   └── run_W1200_q99.0_upper_newonly.json
├── src/
│   ├── anomaly_detect.py        # detector, evaluation, plotting, grid search
│   └── README_TEMPLATE.md       # template used by --save-readme
├── requirements.txt
└── README.md
```

## Setup

Requires Python 3 (NumPy ≥ 1.22 for `np.percentile(..., method=...)`).

```bash
pip install -r requirements.txt
```

## Usage

The script uses `data/data.csv` and `outputs/` by default. It finds both relative to
the repository root, so you can run it from any directory. The value and ground-truth
columns in `data/data.csv` are `NO3N` and `Student_Flag`. The script's built-in
defaults (`Nitrate`, plus auto-detection of the ground-truth column) do not match
these names, so pass them explicitly:

```bash
# Single run: one-sided upper tail, W=1200, q=99, label only new points
python src/anomaly_detect.py --value-col NO3N --gt-col Student_Flag \
    --W 1200 --q 99.0 --one-sided upper --initial-label newonly

# Also write outputs/README.md from src/README_TEMPLATE.md
python src/anomaly_detect.py --value-col NO3N --gt-col Student_Flag \
    --W 1200 --q 99.0 --save-readme

# Grid search over W and q (writes one result set per configuration + outputs/sweep_results.csv)
python src/anomaly_detect.py --value-col NO3N --gt-col Student_Flag \
    --sweep --W-min 500 --W-max 2000 --W-step 250 \
    --q-grid 97,98,98.5,99,99.5 --target-normal 0.80 --target-anomaly 0.75

# Two-sided thresholds
python src/anomaly_detect.py --value-col NO3N --gt-col Student_Flag \
    --W 1200 --two-sided --q-low 1.0 --q 99.0
```

Other options: `--data <csv>`, `--outdir <dir>`. Run `python src/anomaly_detect.py --help`
to see them all.

Each run writes these files to `outputs/` (`<tag>` encodes the settings, e.g.
`W1200_q99.0_upper_newonly`):

- `anomaly_plot_<tag>.png`: the series, with predicted anomalies (circles) and
  ground-truth anomalies (crosses)
- `predictions_<tag>.csv`: `index, value, pred, threshold[, gt]`
- `run_<tag>.json`: the settings, output paths and metrics

## Results in this repository

The committed outputs come from one run with `W = 1200`, `q = 99.0`, a one-sided
upper-tail rule and `newonly` first-window labeling. That run was made without a
ground-truth column, so `run_W1200_q99.0_upper_newonly.json` has `"metrics": null`
and `outputs/README.md` shows `NA` for every metric. Re-run the single-run command
above with `--gt-col Student_Flag` to evaluate against the labels.

Note: the script warns when the ground-truth positives do not total 77, a number
taken from the original assignment spec. `Student_Flag` in `data/data.csv` has 141
positives, so the warning will appear, but the metrics are still computed.

## Author

Gaurav Goyal
