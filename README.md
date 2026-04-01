# 🚕 NYC Yellow Taxi Trips — Exploratory Data Analysis (2020)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TFDV-Data%20Validation-orange?logo=tensorflow)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![University](https://img.shields.io/badge/Pace%20University-CS672-blue)

> Exploratory Data Analysis of NYC Yellow Taxi trip records from 2020, with COVID-19 impact analysis and TFDV-powered statistical validation — built for CS672: Introduction to Deep Learning at Pace University (Spring 2026).

---

## 📋 Table of Contents
- [Overview](#-overview)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
- [Key Findings](#-key-findings)
- [Installation](#-installation)
- [Usage](#-usage)
- [Technologies Used](#-technologies-used)
- [Author](#-author)

---

## 🔬 Overview

This project performs a comprehensive EDA on NYC Yellow Taxi data from 2020 — a year dramatically impacted by the COVID-19 pandemic. The analysis uses both the **classic Pandas/NumPy approach** and **TensorFlow Data Validation (TFDV)** for statistical profiling.

- ✅ **Task 1** — Data preparation: missing values, outliers, numeric transformation
- ✅ **Task 2** — Data type listing: numeric vs categorical
- ✅ **Task 3** — Classic EDA + TFDV statistics, correlations, feature importance
- ✅ **Task 4** — COVID-19 time window analysis (March = "new normal")
- ✅ **Extra Credit** — January vs March 2020 baseline comparison

---

## 📊 Dataset

**Source:** [NYC TLC Trip Record Data](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page)

| Month | Role | Trips |
|---|---|---|
| January 2020 | Pre-COVID baseline (Extra Credit) | ~6.7M |
| March 2020 | Training dataset | ~2,955,781 |
| May 2020 | Evaluation dataset | ~335,709 |

### Features Used

| Feature | Type | Description |
|---|---|---|
| `tpep_pickup_datetime` | Datetime | Pickup timestamp |
| `tpep_dropoff_datetime` | Datetime | Dropoff timestamp |
| `trip_distance` | Numeric | Distance in miles |
| `trip_duration_min` | Numeric (engineered) | Duration in minutes |
| `avg_mph` | Numeric (engineered) | Average speed |
| `fare_amount` | Numeric | Base fare |
| `total_amount` | Numeric | Total charge |
| `tip_amount` | Numeric | Tip amount |
| `tip_rate` | Numeric (engineered) | Tip as fraction of fare |
| `payment_type` | Categorical | Payment method |
| `VendorID` | Categorical | Vendor |
| `RatecodeID` | Categorical | Rate code |
| `PULocationID` | Categorical | Pickup zone |
| `DOLocationID` | Categorical | Dropoff zone |

---

## 📁 Project Structure
```
nyc-taxi-eda-2020/
│
├── Exploratory_Data_Analysis_of_NYC_Yellow_Taxi_Trips_(2020).ipynb
├── prepared/
│   ├── train_mar2020.parquet
│   └── eval_may2020.parquet
├── README.md
└── requirements.txt
```

---

## 🔬 Methodology

### Task 1 — Data Preparation
- Parquet files downloaded directly from NYC TLC CDN
- Trip duration engineered: `(dropoff - pickup).total_seconds() / 60`
- Filters applied: distance (0–150mi), duration (0–360min), fare > 0
- Speed outliers removed (>80mph)
- Missing values handled per column

### Task 2 — Data Type Listing
```
Numeric    : trip_distance, trip_duration_min, avg_mph, fare_amount,
             tip_amount, tolls_amount, total_amount, tip_rate,
             pickup_hour, pickup_weekday, is_weekend, is_rush_hour, ...
Categorical: VendorID, RatecodeID, payment_type,
             PULocationID, DOLocationID
```

### Task 3 — EDA
**Classic approach:**
- Distribution histograms for all key numeric features
- Correlation heatmap (numeric features)
- Mutual information feature importance for `trip_duration_min`

**TFDV approach:**
- Side-by-side statistics: March vs May
- Schema inference from March (training)
- Anomaly detection on May (evaluation)

### Task 4 — COVID Time Window Analysis
March 2020 = pandemic outbreak in the US → "new normal" for NYC Taxi

---

## 💡 Key Findings

### March vs May 2020

| Metric | March | May | Δ |
|---|---|---|---|
| **Trips** | 2,955,781 | 335,709 | **−88.6%** |
| **Avg Distance (mi)** | 2.914 | 3.812 | +30.8% |
| **Avg Duration (min)** | 13.139 | 11.919 | −9.3% |
| **Avg Speed (mph)** | 12.193 | 16.746 | **+37.3%** |
| **Avg Total ($)** | 18.476 | 18.437 | −0.2% |
| **Median Tip Rate** | 0.233 | 0.065 | **−72.1%** |

### Jan vs March 2020 (Extra Credit)
- January represents the pre-COVID baseline
- March shows early pandemic impact — dramatic trip volume drop
- Speed increased significantly as streets emptied

### Feature Importance (Mutual Information)
Top predictors of `trip_duration_min`:
1. `trip_distance`
2. `avg_mph`
3. `fare_amount`
4. `pickup_hour`
5. `is_rush_hour`

---

## ⚙️ Installation
```bash
git clone https://github.com/krishnamaniyar2209/nyc-taxi-eda-2020.git
cd nyc-taxi-eda-2020
pip install -r requirements.txt
jupyter notebook "Exploratory_Data_Analysis_of_NYC_Yellow_Taxi_Trips_(2020).ipynb"
```

### requirements.txt
```
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
scikit-learn>=1.1.0
tensorflow>=2.10.0
tensorflow-data-validation>=1.3.0
pyarrow>=9.0.0
jupyter>=1.0.0
```

---

## 🛠️ Technologies Used

| Tool | Purpose |
|---|---|
| Pandas / NumPy | Classic EDA |
| Matplotlib | Visualization |
| TFDV | Statistical validation & anomaly detection |
| scikit-learn | Mutual information feature importance |
| PyArrow | Parquet file reading |

---

## 👤 Author

**Krishna Maniyar**
- 🎓 Pace University — Seidenberg School of CSIS
- 📘 CS672: Introduction to Deep Learning | Spring 2026
- 🔗 [GitHub](https://github.com/krishnamaniyar2209)

---

<p align="center">Made with ❤️ for CS672 @ Pace University</p>
