# Nitrate Anomaly Detection (Sliding-Window Percentile)

This repository implements a **threshold-based** anomaly detector for the **Nitrate** time series with **fixed-size, overlapping windows** (step size = 1). Percentiles are computed using `numpy.percentile(..., method="linear")`.

---

## How it Works

- Choose a **window size** `W` (e.g., 500–2000).  
- For the first window `[0, W-1]`, compute `T₁ = percentile(x[0:W], q)` (linear method).  
  - Labeling choice:
    - **newonly**: label only the point at index `W-1` using `T₁` (chosen here: `newonly`).
    - **all**: label all points `[0:W-1]` using `T₁`.
- For each subsequent window `i` covering `[i, i+W-1]`, compute `Tᵢ` from that window only (**no leakage**).
  - **Label only the newly added point** at index `i+W-1` using `Tᵢ`.
- Rule used here: **one-sided (upper tail)** with `q = 99.0`.
  - (If two-sided: `q_low` for lower tail and `q` for upper tail; otherwise one-sided upper or lower tail.)

---

## Chosen Hyperparameters & Rationale

- **Window size `W`**: **1200**  
  Rationale: balances **responsiveness** to local distribution shifts and **stability** against noise. In practice, windows around 500–2000 work well on long environmental series; `1200` preserved enough local structure without overreacting to spikes.
- **Percentile `q`**: **99.0** (one-sided (upper tail))  
  Rationale: upper-tail anomalies in Nitrate typically manifest as rare peaks. High percentiles (97–99.5) flag such peaks while keeping false positives manageable.

If a grid search was run, see **`N/A`** for all tried configurations.

---

## Metrics (vs Ground Truth)

Let **P = TP + FN** (total anomalies) and **N = TN + FP** (total normals).  
Target performance: Normal accuracy ≥ 80%, Anomaly accuracy ≥ 75%.

- **P**: NA (expected 77)  
- **N**: NA
- **TP**: NA, **FP**: NA, **FN**: NA, **TN**: NA
- **Normal accuracy**: `TN / N` = **NA**
- **Anomaly accuracy**: `TP / P` = **NA**

> Note: If your GT column sums to a value other than 77, the script will warn but still compute metrics.

---

## Outputs

- **Plot** with anomalies highlighted: `outputs/anomaly_plot_W1200_q99.0_upper_newonly.png`
- **Predictions CSV** (`index, value, pred, threshold, [gt]`): `outputs/predictions_W1200_q99.0_upper_newonly.csv`

Example plot (anomalies as circles; GT anomalies as crosses):
![detected anomalies](outputs/anomaly_plot_W1200_q99.0_upper_newonly.png)

---

## Reproducible Run

Assuming your data file is **`data.csv`** and the Nitrate column is named **`Nitrate`** (default):

```bash
# Single run (one-sided upper tail), labeling only new points (spec-compliant):
python src/anomaly_detect.py --data data/data.csv --value-col Nitrate --gt-col GroundTruth \
    --W 1200 --q 99.0 --one-sided upper --initial-label newonly \
    --outdir outputs --save-readme
```

**Optional grid search** over `W` and `q` to hit targets:

```bash
python src/anomaly_detect.py --data data/data.csv --value-col Nitrate --gt-col GroundTruth \
    --sweep --W-min 500 --W-max 2000 --W-step 250 \
    --q-grid 97,98,98.5,99,99.5 \
    --target-normal 0.80 --target-anomaly 0.75 \
    --outdir outputs --save-readme
```

---

## Design Choices

- **Windowing**: fixed size, step=1, fully overlapping; threshold computed from **current** window only.  
- **Labeling**: only the **newly added point** per window (**spec requirement**). First window labeling mode: `newonly`.  
- **Percentile method**: `method="linear"` (NumPy).  
- **NaNs**: expected data file is already cleaned; any residual NaNs in the value column are coerced and ignored by the procedure when present in windows.

Two-sided option (if needed):  
```bash
python src/anomaly_detect.py --data data/data.csv --value-col Nitrate --gt-col GroundTruth \
    --W 1200 --two-sided --q-low 1.0 --q 99.0 --initial-label newonly --outdir outputs
```

---

## File Structure

```
.
├── data/
│   └── data.csv
├── src/
│   ├── anomaly_detect.py
│   └── README_TEMPLATE.md
├── requirements.txt
└── outputs/
    ├── anomaly_plot_*.png
    ├── predictions_*.csv
    ├── run_*.json
    └── sweep_results.csv   # if --sweep is used
```
