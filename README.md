# DSCI Drought Analysis — Massachusetts

Weekly Drought Severity and Coverage Index (DSCI) visualization with functional median detection using Modified Band Depth (MBD2), moving-average smoothing, and spline-based derivative analysis.

---

## What is DSCI?

The **Drought Severity and Coverage Index** is a single weekly number summarizing how bad and how widespread drought is across a state. It is computed from the US Drought Monitor's five drought categories:

| Column | Meaning |
|--------|---------|
| D0 | Abnormally dry |
| D1 | Moderate drought |
| D2 | Severe drought |
| D3 | Extreme drought |
| D4 | Exceptional drought |

Each column is the **percentage of the state's area** at that level *or worse* (cumulative). So the formula:

```
DSCI = D0 + D1 + D2 + D3 + D4
```

implicitly double-counts worse droughts — a region in D4 contributes to all five columns, giving it 5× the weight. That's intentional: it ranges from **0** (no drought anywhere) to **500** (entire state in exceptional drought).

---

## What Are We Trying to Do Visually?

We have ~25 years of weekly DSCI data. Each year is a **curve** — 52 points (weeks) mapping to DSCI values. So we're treating each year as a **function of time**:

```
y(t) = DSCI(t),   t = 1, 2, …, 52
```

We plot all these curves together and find the **median curve** — the one that is most "central" or "typical" among all years.

---

## Why Not Just Use the Pointwise Mean or Median?

You could take the average DSCI at each week independently. But that ignores the **shape** of the curve. A year might be typical not because it hits the average at every single week, but because its *overall pattern* — when droughts peak, how they evolve — is closest to most other years.

This is where **functional data analysis** comes in. Each year isn't just 52 numbers — it's a *function*, and we want a depth measure that respects that.

---

## What is Band Depth?

Band depth is a way to measure how "deep" (central) a curve is relative to a set of curves. The intuition is simple:

> A curve is deep if many pairs of other curves form a "band" (envelope) that contains it.

Imagine drawing two curves from different years — they form a tube/band between them. If your curve lies **entirely inside** that band, it scores a point. The more bands contain your curve, the deeper (more central) it is.

The **deepest curve** = the one most surrounded by others = the functional median.

---

## Why MBD2 Specifically?

Three variants of band depth exist:

- **BD2** — checks if a curve fits *completely* inside a band formed by 2 others. Too strict: if any two curves cross, the band collapses to a point and nothing fits inside it. Fails on noisy data like ours.
- **BD3** — uses bands formed by 3 curves. More robust, but O(n⁴) to compute for all curves — very slow.
- **MBD2** — **Modified** Band Depth. Instead of requiring *complete* containment, it measures the *proportion of time* a curve spends inside a band. Much more robust to crossings, and runs in O(mn log n) — fast enough for 25 years × 52 weeks.

> Reference: López-Pintado & Romo (2009), *On the concept of depth for functional data*, JASA.

---

## Code Walkthrough

### Core Functions

#### `load_and_compute_dsci(path)`

Parses dates, computes DSCI, and extracts year and ISO week number:

```python
df["DSCI"] = df["D0"] + df["D1"] + df["D2"] + df["D3"] + df["D4"]
df["Year"] = df["MapDate"].dt.year
df["Week"] = df["MapDate"].dt.isocalendar().week.astype(int)
```

Uses `dt.year` (not ISO year) to avoid late-December rows being misattributed to the next year.

---

#### `build_matrix(df, start_year, end_year, max_weeks=52)`

```python
curve = np.interp(
    np.arange(1, max_weeks + 1),   # target: exactly weeks 1-52
    ydf["Week"].values,             # actual weeks we have
    ydf["DSCI"].values,             # actual DSCI values
)
```

Some years have 53 ISO weeks, some have gaps. `np.interp` standardizes every year to exactly 52 evenly-spaced points. Returns a **matrix of shape (n_years × 52)** — one row per year, one column per week.

Years where `max(DSCI) < 1` are skipped (corrupt/missing data).

---

#### `modified_band_depth(matrix)`

```python
for j in range(m):               # for each of the 52 weeks
    col = matrix[:, j]           # DSCI values of all years at week j

    for i, g in enumerate(col):  # for each year's value at this week
        a = # count of years strictly below g
        c = # count of years strictly above g
        b = # count of years equal to g

        depths[i] += a*c + a*b + b*c + b*(b-1)/2
```

At each time point `j`, for curve `i` with value `g`:
- `a` = how many curves are below it
- `c` = how many are above it
- `b` = how many are equal to it

The expression `a*c + a*b + b*c + b*(b-1)/2` counts how many **pairs** of curves form a band containing `g`. Summing over all 52 weeks and normalizing by `m * (n*(n-1)/2)` gives a score in [0, 1]. Highest score = functional median.

---

#### `apply_moving_average(matrix, window)`

Smooths each year's curve with a centered rolling mean of the given window size. Handles edge effects with `np.convolve`.

---

#### `visualize_curves(matrix, years, depths, title, spline_curve, save_png, fname)`

Plots all curves color-coded by depth rank:

| Curve | Color | Style |
|-------|-------|-------|
| Median (deepest) | crimson | solid, lw=2.5 |
| 2nd deepest | royalblue | solid, lw=2.0 |
| 3rd deepest | forestgreen | solid, lw=2.0 |
| 3 outliers (shallowest) | mediumpurple | dashed, lw=1.5 |
| All others | steelblue | lw=0.8, alpha=0.45 |
| Spline overlay (optional) | black | dash-dot, lw=3.0 |

---

#### `print_depth_summary(depths, years)`

Prints a ranked table of years from most typical (highest depth) to most anomalous (lowest depth).

---

### Analysis Steps

#### Step 1a — Raw MBD2

Computes MBD2 directly on the raw (unsmoothed) 52-week curves. Saves:

```
step1a_dsci_MA_2001_2025.png
```

---

#### Step 1b — Moving Average Smoothing

Applies a centered moving average with window sizes 3, 4, and 5 weeks, then recomputes MBD2 on each smoothed version. Saves:

```
step1b_dsci_MA_ma3.png
step1b_dsci_MA_ma4.png
step1b_dsci_MA_ma5.png
```

---

#### Step 1c — Spline Smoothing and Derivative

Takes the raw median curve (from Step 1a), fits a `scipy.interpolate.UnivariateSpline`, overlays it on the raw data, and also plots its derivative (rate of change of DSCI over weeks). Saves two plots:

```
step1c_dsci_MA_spline.png      # raw curves + spline overlay
step1c_dsci_MA_derivative.png  # dDSCI/dWeek for the median
```

---

## Usage

Set configuration at the top of the notebook:

```python
DATA_FILE  = "data.csv"   # path to your data
STATE      = "MA"         # used in plot titles and filenames
START_YEAR = 2001
END_YEAR   = 2025
```

Then run all cells in order. Steps 1a → 1b → 1c build on each other.

---

## Project Structure

```
.
├── data.csv                    # US Drought Monitor weekly state data
├── dsci_central_depth.ipynb    # Main Jupyter notebook
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── .gitignore
```

---

## Requirements

```
numpy
pandas
matplotlib
scipy
jupyter
```

Install with:

```bash
pip install -r requirements.txt
```

---

## Data Source

[US Drought Monitor](https://droughtmonitor.unl.edu/) — weekly drought statistics by state.
