# 🚕 NYC Yellow Taxi Trips — Exploratory Data Analysis (2020)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![TFDV](https://img.shields.io/badge/TFDV-Data%20Validation-orange?logo=tensorflow)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![University](https://img.shields.io/badge/Pace%20University-CS672-blue)

> Exploratory Data Analysis of ~9.6M NYC Yellow Taxi trip records from 2020, quantifying the COVID-19 impact with TFDV-powered statistical validation — built for CS672: Introduction to Deep Learning at Pace University (Fall 2025).

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

A comprehensive EDA of NYC Yellow Taxi data from 2020 — a year reshaped by COVID-19 — using both the **classic Pandas/NumPy** approach and **TensorFlow Data Validation (TFDV)** for statistical profiling and anomaly detection.

- ✅ **Task 1** — Data preparation: filtering, outlier removal, feature engineering
- ✅ **Task 2** — Data-type listing: numeric vs. categorical
- ✅ **Task 3** — Classic EDA + TFDV statistics, correlations, feature importance
- ✅ **Task 4** — COVID-19 time-window analysis (March → May)
- ✅ **Extra Credit** — January (pre-COVID) vs. March 2020 baseline

---

## 📊 Dataset

**Source:** [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)

*(Trip counts below are after cleaning filters; ~9.6M trips total.)*

| Month | Role | Trips (filtered) |
|---|---|---|
| January 2020 | Pre-COVID baseline (Extra Credit) | 6,294,833 |
| March 2020 | Training dataset | 2,955,781 |
| May 2020 | Evaluation dataset | 335,709 |

### Engineered Features
`trip_duration_min` (dropoff − pickup), `avg_mph` (distance ÷ duration), `tip_rate` (tip ÷ fare).

---

## 📁 Project Structure
```
nyc-taxi-eda-2020/
│
├── Exploratory_Data_Analysis_of_NYC_Yellow_Taxi_Trips_(2020).ipynb
├── README.md
└── requirements.txt

# Generated locally by the notebook (not committed — large parquet files):
#   prepared/train_mar2020.parquet
#   prepared/eval_may2020.parquet
```

---

## 🔬 Methodology

### Task 1 — Data Preparation
- Parquet files downloaded from the NYC TLC CDN
- Engineered `trip_duration_min = (dropoff − pickup) / 60`
- Filters: distance 0–150 mi, duration 0–360 min, fare > 0, speed ≤ 80 mph

### Task 2 — Data-Type Listing
Numeric: `trip_distance, trip_duration_min, avg_mph, fare_amount, tip_amount, tolls_amount, total_amount, tip_rate, ...`
Categorical: `VendorID, RatecodeID, payment_type, PULocationID, DOLocationID`

### Task 3 — EDA
- **Classic:** distribution histograms, correlation heatmap, mutual-information feature importance for `trip_duration_min`
- **TFDV:** side-by-side March vs. May statistics, schema inference from March, anomaly detection on May

### Task 4 — COVID Time-Window Analysis
March 2020 (US outbreak) compared against May 2020 to quantify the pandemic's effect on ridership and trip characteristics.

---

## 💡 Key Findings

### March vs. May 2020
| Metric | March | May | Δ |
|---|---|---|---|
| **Trips** | 2,955,781 | 335,709 | **−88.6%** |
| Avg Distance (mi) | 2.914 | 3.812 | +30.8% |
| Avg Duration (min) | 13.139 | 11.919 | −9.3% |
| **Avg Speed (mph)** | 12.193 | 16.746 | **+37.3%** |
| Avg Total ($) | 18.476 | 18.437 | −0.2% |
| **Median Tip Rate** | 0.233 | 0.065 | **−72.1%** |

> As lockdowns emptied the streets, **ridership collapsed ~89%** while **average speed jumped ~37%** (less congestion). The **72% drop in median tip rate** suggests a shift in trip and payment behavior during the pandemic.

### Jan vs. March 2020 (Extra Credit)
January (pre-COVID) and March were nearly identical (trips 6.29M vs. 2.96M, but similar distance/speed/tip) — the dramatic change came **between March and May**, pinpointing the pandemic's onset.

### Feature Importance — Mutual Information wrt `trip_duration_min`
| Rank | Feature | MI |
|---|---|---|
| 1 | `avg_mph` | 0.956 |
| 2 | `trip_distance` | 0.803 |
| 3 | `tip_amount` | 0.689 |
| 4 | `tip_rate` | 0.648 |
| 5 | `tolls_amount` | 0.077 |

> Note: `avg_mph` is derived from duration, so its high score is partly mechanical; `trip_distance` is the strongest *independent* predictor.

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
2. The notebook downloads the TLC parquet files directly — no manual download needed
3. Run all cells top to bottom — cleaning, EDA, TFDV validation, and the COVID comparison generate automatically

---

## 🛠️ Technologies Used
| Tool | Purpose |
|---|---|
| Pandas / NumPy | Classic EDA |
| Matplotlib | Visualization |
| TFDV | Statistical validation & anomaly detection |
| scikit-learn | Mutual-information feature importance |
| PyArrow | Parquet reading |

---

## 👤 Author

**Krishna Maniyar** — Data Analyst
- 🎓 Pace University — Seidenberg School of CSIS, MS in Data Science
- 📘 CS672: Introduction to Deep Learning (Fall 2025)
- 📧 krishnamaniyarkm22@gmail.com
- 🔗 [GitHub](https://github.com/krishnamaniyar2209) · [LinkedIn](https://www.linkedin.com/in/krishnamaniyar/) · [Portfolio](https://krishnamaniyar2209.github.io/)

---

<p align="center">Made with ❤️ for CS672 @ Pace University</p>
