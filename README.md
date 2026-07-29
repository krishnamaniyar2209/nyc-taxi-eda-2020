# 🚕 NYC Yellow Taxi Trips: Exploratory Data Analysis (2020)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![TFDV](https://img.shields.io/badge/TFDV-Data%20Validation-orange?logo=tensorflow)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![University](https://img.shields.io/badge/Pace%20University-CS672-blue)

> Exploratory Data Analysis of 9.59M NYC Yellow Taxi trip records from 2020, quantifying the COVID-19 impact with TFDV-powered statistical validation. Built for CS672: Introduction to Deep Learning at Pace University (Fall 2025).

---

## 📋 Table of Contents
- [Overview](#-overview)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
- [Key Findings](#-key-findings)
- [Data Validation with TFDV](#-data-validation-with-tfdv)
- [Limitations & Next Steps](#-limitations--next-steps)
- [Installation](#-installation)
- [Usage](#-usage)
- [Technologies Used](#-technologies-used)
- [Author](#-author)

---

## 🔬 Overview

A comprehensive EDA of NYC Yellow Taxi data from 2020, a year reshaped by COVID-19, using both the **classic Pandas/NumPy** approach and **TensorFlow Data Validation (TFDV)** for statistical profiling and anomaly detection.

- ✅ **Task 1** Data preparation: filtering, outlier removal, feature engineering
- ✅ **Task 2** Data-type listing: numeric vs. categorical
- ✅ **Task 3** Classic EDA plus TFDV statistics, correlations, feature importance
- ✅ **Task 4** COVID-19 time-window analysis (March vs. May)
- ✅ **Extra Credit** January (pre-COVID) vs. March 2020 baseline

---

## 📊 Dataset

**Source:** [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)

| Month | Role | Raw trips | After filtering | Retained |
|---|---|---|---|---|
| January 2020 | Pre-COVID baseline (Extra Credit) | 6,405,008 | **6,294,833** | 98.3% |
| March 2020 | Training dataset | 3,007,687 | **2,955,781** | 98.3% |
| May 2020 | Evaluation dataset | 348,415 | **335,709** | 96.4% |
| **Total** | | 9,761,110 | **9,586,323** | 98.2% |

Filters are deliberately conservative, retaining 96 to 98% of raw rows. This matters: the ridership collapse documented below is a real signal, not an artifact of aggressive cleaning.

### Engineered Features
| Feature | Definition |
|---|---|
| `trip_duration_min` | (dropoff − pickup) in seconds ÷ 60 |
| `avg_mph` | 60 × distance ÷ duration |
| `tip_rate` | tip ÷ fare |

### Filters Applied
- `trip_distance` in (0, 150] miles
- `trip_duration_min` in (0, 360] minutes
- `fare_amount` > 0
- `avg_mph` in (1, 80] mph

---

## 📁 Project Structure
```
nyc-taxi-eda-2020/
│
├── Exploratory_Data_Analysis_of_NYC_Yellow_Taxi_Trips_(2020).ipynb
├── README.md
├── requirements.txt
└── .gitignore

# Generated locally by the notebook, not committed:
#   taxi2020/yellow_tripdata_2020-{01,03,05}.parquet   # raw TLC downloads (~1 GB+)
#   prepared/train_mar2020.parquet
#   prepared/eval_may2020.parquet
```

---

## 🔬 Methodology

### Task 1: Data Preparation
Parquet files are downloaded directly from the NYC TLC CDN, restricted to 17 permitted columns, then passed through the `prep()` function for feature engineering and filtering.

### Task 2: Data-Type Listing

A dtype-based listing returns **all 19 columns as numeric, with zero categorical**:

```
Numeric: ['month', 'VendorID', 'RatecodeID', 'payment_type', 'PULocationID',
          'DOLocationID', 'passenger_count', 'trip_distance', 'trip_duration_min',
          'avg_mph', 'fare_amount', 'tip_amount', 'tolls_amount', 'total_amount',
          'extra', 'mta_tax', 'improvement_surcharge', 'congestion_surcharge', 'tip_rate']
Categorical: []
Datetime: []
```

That result is mechanically correct and analytically misleading. Five of those columns are **integer-encoded categoricals** that a dtype check cannot see:

| Column | Actual nature |
|---|---|
| `VendorID` | Categorical, provider identifier |
| `RatecodeID` | Categorical, rate class |
| `payment_type` | Categorical, payment method |
| `PULocationID` | Categorical, 265 taxi zones |
| `DOLocationID` | Categorical, 265 taxi zones |

Treating these as numeric would imply that zone 264 is "more" than zone 12, which is meaningless. Any modeling built on this data must encode them explicitly. The mutual-information step below accounts for this by flagging low-cardinality integer columns as discrete.

### Task 3: EDA
- **Classic:** distribution histograms, correlation heatmap, mutual-information feature importance for `trip_duration_min` on a 200,000-row sample
- **TFDV:** side-by-side March vs. May statistics on 300,000-row samples, schema inferred from March, May validated against it

### Task 4: COVID Time-Window Analysis
March 2020 compared against May 2020, with January 2020 added as a pre-outbreak baseline.

---

## 💡 Key Findings

### The pandemic arrived in two distinct phases

| Metric | Jan → Mar | Mar → May |
|---|---|---|
| **Trips** | **−53.0%** | **−88.6%** |
| Avg. Distance | −0.6% | **+30.8%** |
| Avg. Duration | −0.4% | −9.3% |
| **Avg. Speed** | +2.6% | **+37.3%** |
| Avg. Total ($) | −0.5% | −0.2% |
| **Median Tip Rate** | −0.9% | **−72.1%** |

**Volume collapsed first, behaviour changed second.** By March, ridership had already fallen 53% from January while every per-trip characteristic held within 3% of baseline. The trips that still happened looked exactly like January trips; there were simply half as many. Only by May did the character of a taxi journey change, with distances up 31%, speeds up 37%, and tipping down 72%.

People stopped taking taxis well before the nature of taxi travel shifted. Cumulatively, January to May, ridership fell **94.7%**.

### Absolute values

| Metric | January | March | May |
|---|---|---|---|
| Trips | 6,294,833 | 2,955,781 | 335,709 |
| Avg. Distance (mi) | 2.931 | 2.914 | 3.812 |
| Avg. Duration (min) | 13.187 | 13.139 | 11.919 |
| Avg. Speed (mph) | 11.879 | 12.193 | 16.746 |
| Avg. Total ($) | 18.570 | 18.476 | 18.437 |
| Median Tip Rate | 0.235 | 0.233 | 0.065 |

Average fare stayed remarkably flat across all three months, within 0.7%, even as trips got longer and faster. Longer distances and shorter durations roughly cancelled out in the fare calculation.

### Feature Importance: Mutual Information wrt `trip_duration_min`

| Rank | Feature | MI |
|---|---|---|
| 1 | `avg_mph` | 0.956 |
| 2 | **`trip_distance`** | **0.803** |
| 3 | `tip_amount` | 0.689 |
| 4 | `tip_rate` | 0.648 |
| 5 | `tolls_amount` | 0.077 |
| 6 | `RatecodeID` | 0.055 |
| 7 | `congestion_surcharge` | 0.022 |
| 8+ | `month`, `VendorID`, `payment_type`, `passenger_count` | 0.000 |

> `avg_mph` is computed as `60 × distance ÷ duration`, so duration is on both sides of the equation. Its top ranking is mechanical and should be discarded. **`trip_distance` is the strongest genuinely independent predictor.** The four zero-scoring features carry no information about trip duration at all.

---

## 🔎 Data Validation with TFDV

Statistics were generated for both months, a schema was inferred from March 2020 (the training month), and May 2020 was validated against it:

```python
mar_stats = tfdv.generate_statistics_from_dataframe(mar.sample(300_000, random_state=1))
may_stats = tfdv.generate_statistics_from_dataframe(may.sample(300_000, random_state=1))

tfdv.visualize_statistics(lhs_statistics=mar_stats, rhs_statistics=may_stats,
                          lhs_name="March 2020", rhs_name="May 2020")

schema = tfdv.infer_schema(mar_stats)
tfdv.display_schema(schema)

anomalies = tfdv.validate_statistics(statistics=may_stats, schema=schema)
tfdv.display_anomalies(anomalies)
```

**Result: no anomalies found.**

This is the most instructive outcome in the project. May 2020 had 88.6% fewer trips, 31% longer average distances, 37% higher speeds, and a 72% collapse in median tipping. It passed schema validation without a single flag.

**Schema validation catches structural drift, not distributional drift.** Column types, presence, and domains were all stable between the two months, which is all TFDV's schema check examines. The distributions underneath had transformed completely. A production pipeline relying on schema validation alone would have passed May 2020 through as healthy data.

Catching this class of change requires distribution-level monitoring: TFDV's own drift and skew comparators (L-infinity distance thresholds on categorical features, Jensen-Shannon divergence on numeric ones), or explicit alerting on summary statistics. Schema checks are a floor, not a ceiling.

### Two details in the inferred schema

**`RatecodeID` is the only optional feature.** Every other column inferred as `required`; `RatecodeID` came back `FLOAT` / `optional` / `single`, meaning TFDV observed missing values in March. It is therefore the one column most likely to trip a future anomaly check, and the one worth watching if this pipeline were productionised.

**`__index_level_0__` is not a real feature.** It is a pandas index artifact introduced by the parquet round-trip and silently promoted into the schema. Calling `reset_index(drop=True)` before generating statistics removes it. Left in place, it would become part of the contract that future data is validated against.

---

## ⚠️ Limitations & Next Steps

1. **No distribution-drift check was run.** Given that schema validation passed cleanly, the natural follow-up is TFDV's drift comparators to detect the change that the schema check missed. This is the single most valuable addition.
2. **Only three months were analyzed.** January, March, and May give a coarse picture. A month-by-month series across all of 2020 would show the recovery curve, not just the collapse.
3. **Correlation and MI were computed on March only.** Whether feature relationships held stable through the pandemic is untested, and the May behaviour shift suggests they may not have.
4. **Tip rate is understated for cash trips.** Cash tips are not recorded in TLC data, so `tip_rate` measures card-tipping behaviour. Part of the 72% drop may reflect a shift in payment mix rather than tipping generosity alone. `payment_type` is available and this is testable.
5. **No zone-level geographic analysis was performed.** `PULocationID` and `DOLocationID` are present in the prepared data, the correlation heatmap, and the TFDV statistics, but were excluded from the mutual-information ranking on cardinality grounds and never analysed geographically. Manhattan versus outer-borough shifts are likely a large part of the distance and speed story.
6. **Statistics were computed on samples.** TFDV used 300,000-row samples and mutual information used 200,000 rows. Adequate for stable estimates, but not the full population.

---

## ⚙️ Installation
```bash
git clone https://github.com/krishnamaniyar2209/nyc-taxi-eda-2020.git
cd nyc-taxi-eda-2020
pip install -r requirements.txt
jupyter notebook "Exploratory_Data_Analysis_of_NYC_Yellow_Taxi_Trips_(2020).ipynb"
```

---

## 🚀 Usage
1. Open the notebook in Jupyter or Google Colab
2. The notebook downloads the TLC parquet files directly from the CDN into `taxi2020/`, no manual download needed. Expect roughly 1 GB across the three months
3. Run all cells top to bottom. Cleaning, EDA, TFDV validation, and the COVID comparison generate automatically
4. Prepared datasets are written to `prepared/`. Both directories are excluded from version control by `.gitignore`

---

## 🛠️ Technologies Used

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.10.18 | Core language |
| Pandas | 1.5.3 | Data manipulation and classic EDA |
| NumPy | 1.26.4 | Numerical operations |
| Matplotlib | 3.7.5 | Visualization |
| scikit-learn | 1.3.2 | Mutual-information feature importance |
| TensorFlow | 2.18.0 | TFDV backend |
| TFDV | 1.14.0 | Statistical profiling, schema inference, validation |
| PyArrow | latest | Parquet reading |

---

## 👤 Author

**Krishna Maniyar**, Data Analyst
- 🎓 Pace University, Seidenberg School of CSIS, MS in Data Science
- 📘 CS672: Introduction to Deep Learning (Fall 2025)
- 📧 maniyarkrishnakm22@gmail.com
- 🔗 [GitHub](https://github.com/krishnamaniyar2209) · [LinkedIn](https://www.linkedin.com/in/krishnamaniyar/) · [Portfolio](https://krishnamaniyar2209.github.io/)

---

<p align="center">Made with ❤️ for CS672 @ Pace University</p>
