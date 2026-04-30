# DSCI Drought Analysis — Massachusetts

Weekly Drought Severity and Coverage Index (DSCI) visualization with functional median detection using Modified Band Depth.

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

### Step 1 — `load_and_compute_dsci()`

```python
df["DSCI"] = df["D0"] + df["D1"] + df["D2"] + df["D3"] + df["D4"]
df["Year"] = df["MapDate"].dt.year
df["Week"] = df["MapDate"].dt.isocalendar().week.astype(int)
```

Parses dates, computes DSCI, and extracts year and week number. We use `dt.year` (not ISO year) to avoid late-December rows being misattributed to the next year.

---

### Step 2 — `build_matrix()`

```python
curve = np.interp(
    np.arange(1, max_weeks + 1),   # target: exactly weeks 1-52
    ydf["Week"].values,             # actual weeks we have
    ydf["DSCI"].values,             # actual DSCI values
)
```

Some years have 53 ISO weeks, some have gaps. `np.interp` standardizes every year to exactly 52 evenly-spaced points. The result is a **matrix of shape (n_years × 52)** — one row per year, one column per week.

Years where `max(DSCI) < 1` are skipped — those are corrupt/missing data (2003 and 2011 in the MA dataset are all zeros).

---

### Step 3 — `modified_band_depth()`

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

The expression `a*c + a*b + b*c + b*(b-1)/2` counts how many **pairs** of curves form a band that contains `g` at this time point. Summing over all 52 weeks and dividing by total possible pairs × weeks gives a score in [0, 1]. The curve with the **highest score** is the median.

---

### Step 4 — `plot_dsci(start_year, end_year)`

The single entry point that chains everything together:

1. Loads and filters data to the requested year range
2. Builds the n_years × 52 matrix
3. Computes MBD2 depths for all curves
4. Plots all curves in a blue gradient (older = lighter, newer = darker) with the deepest curve in red
5. Prints a ranking of most-typical and most-anomalous years

---

## Usage

Place `data.csv` in the same folder as the notebook, then call:

```python
plot_dsci(2001, 2025)              # plot full range
plot_dsci(2015, 2025)              # last 10 years only
plot_dsci(2001, 2025, save_png=True)  # also save as .png
```

### Configuration (top of notebook)

```python
DATA_FILE = "data.csv"   # path to your data
STATE     = "MA"         # used in plot title
```

---

## Project Structure

```
.
├── data.csv                  # US Drought Monitor weekly state data
├── dsci_analysis.ipynb       # Main Jupyter notebook
├── README.md                 # This file
└── .gitignore
```

---

## Requirements

```
numpy
pandas
matplotlib
jupyter
```

Install with:

```bash
pip install numpy pandas matplotlib jupyter
```

---

## Data Source

[US Drought Monitor](https://droughtmonitor.unl.edu/) — weekly drought statistics by state.