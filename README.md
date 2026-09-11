[English](README.md) | [Bahasa Indonesia](README.id.md)

# Cirebon Regency Food Price Monitoring & Forecasting

This project delivers a data science–based system for monitoring, auditing data quality, and forecasting staple food prices across Cirebon Regency, Indonesia. Designed as a production-ready dashboard, it helps the general public understand market dynamics while enabling government agencies to detect field price reporting anomalies.

---

## Tech Stack

The system is built using modern Python libraries following standard industry practices:
* **Data Analysis & Manipulation**: `pandas` (ETL, pivoting, data wrangling) and `numpy` (numerical operations and vectorized calculations).
* **Machine Learning Modeling**: `xgboost` (XGBoost Regressor for fast, accurate multivariate time series regression) and `scikit-learn` (linear models and preprocessing).
* **Visualization & Dashboard**: `streamlit` (web application framework) and `plotly` (interactive time series charts and audit trail plots).
* **Scraping & External APIs**: `requests` (Open-Meteo weather data and news feeds) and `importlib` (daily module hot-reloading).

---

## System Architecture & Components

The system decouples backend data processing (ETL/Modeling) from the user interface to ensure high performance and UI stability:

1. **Automated Scraping & ETL Pipeline (`update_harian.py`)**
   * Executes automatically on a schedule (Cron Job / GitHub Actions) daily at **13:00 WIB (06:00 UTC)**.
   * Programmatically fetches daily staple food price data from the official Ministry of Trade Kepokmas portal for Cirebon Regency.
   * Automates data cleansing, ingests the latest news sentiment and weather metrics, retrains the forecasting models, and updates the repository database.

2. **Interactive Dashboard (`app.py`)**
   * A visual web application powered by Streamlit serving pre-calculated data without client-side model retraining overhead.
   * Key modules: Actual Price Trends, Anomaly Detection (Audit Trail), Market Price Disparity, Data Quality Diagnostics, and AI Q&A (Natural Language Query / Text-to-Pandas).

---

## Data Cleansing & Anomaly Detection Methodology

Raw records reported by market surveyors frequently contain errors due to manual input typos (e.g., missing zeroes). The automated pipeline sanitizes incoming records through:
* **Automated Magnitude Correction**: Automatically scales typo values missing trailing zeroes (e.g., input Rp 3,500 recorded as Rp 350 is adjusted to an expected valid band).
* **Time Series Outlier Detection**: Applies the **Interquartile Range (IQR)** statistical method on daily price deltas to flag unrealistic spikes caused by data entry mistakes.
* **Gap Imputation**: Outlier records are dropped and replaced via linear interpolation, while missing calendar dates are populated using Forward Fill (`ffill`) to preserve continuous time-series continuity.

---

## Forecasting Model: 14-Day Multi-Factor XGBoost

Traditional market food prices fluctuate heavily over short horizons. The system constrains predictions to a realistic **14-day forward horizon** using an **XGBoost Regressor** that incorporates multi-domain exogenous drivers:

* **Weather & Climate Cycles (Open-Meteo API)**: Daily `Curah_Hujan` (precipitation) and `Suhu_Rata` (mean temperature) to account for harvest disruptions.
* **Public Sentiment (Google News RSS)**: Sentiment scoring from local agricultural and market news headlines (supply shortages or surplus availability).
* **Policy & Institutional Demand**: School calendar and institutional meal program active/inactive statuses impacting regional staple food demand.
* **Annual Seasonality**: `Hari_Dalam_Tahun` (day of year) to capture harvest and lean seasons.
* **Religious Holidays (HBKN)**: `Jarak_Ke_Hari_Raya` (distance to major holidays such as Eid al-Fitr, Christmas, and New Year) to anticipate cyclical demand surges.
* **Autoregressive Price Structure**: Historical price lags (`Lag_1`, `Lag_2`, `Lag_7`) to model price inertia and market memory.

---

## Validation & Accuracy Metrics

Model performance is continuously tracked against actual market data using standard evaluation metrics:
* **Mean Absolute Percentage Error (MAPE)**: Average percentage deviation between model predictions and realized market prices.
* **Mean Absolute Error (MAE)**: Average absolute nominal error in Indonesian Rupiah (IDR).

---

## Local Installation & Execution

### 1. Prerequisites
Ensure Python 3.9+ is installed. Install all required dependencies:
```bash
pip install pandas numpy streamlit plotly xgboost scikit-learn requests beautifulsoup4
```

### 2. Initial Data Pipeline & Model Training
Before launching the dashboard for the first time, run the pipeline script to clean the raw historical dataset (`master_historis_pangan_cirebon.csv`) and compute the 14-day forecasts:
```bash
python update_harian.py
```
This produces the cleansed dataset (`master_historis_pangan_cirebon_clean.csv`) alongside model evaluation files (`validation_detail.csv`, `validation_metrics.csv`, and `forecast_14_hari.csv`).

### 3. Launching the Dashboard
Once the initial pipeline completes, start the Streamlit application:
```bash
streamlit run app.py
```
Access the dashboard via your web browser at `http://localhost:8501`.

### 4. Repository Structure
* `app.py`: Streamlit frontend interface and Natural Language Query engine.
* `update_harian.py`: Scheduled ETL pipeline, anomaly cleansing, and direct forecasting runner.
* `master_historis_pangan_cirebon.csv`: Raw historical price master database (~25 MB).
* `README.md`: English documentation and guide.
* `README.id.md`: Indonesian documentation and guide.
* `master_historis_pangan_cirebon_clean.csv` *(Generated)*: Cleansed and feature-enriched dataset.
* `forecast_14_hari.csv` *(Generated)*: Pre-computed 14-day forecast table for all market-commodity pairs.
* `validation_metrics.csv` & `validation_detail.csv` *(Generated)*: Model evaluation metrics and validation audit log.
