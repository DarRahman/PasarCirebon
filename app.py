import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from prophet import Prophet
from pathlib import Path
import os
import requests

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "master_historis_pangan_cirebon.csv"

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Pantau Pangan Cirebon",
    layout="wide"
)

CLEAN_FILE = BASE_DIR / "master_historis_pangan_cirebon_clean.csv"
FEATURES = ["Curah_Hujan", "Suhu_Rata", "MBG_Aktif", "Sentimen_Berita", "Lag_1", "Lag_2", "Lag_7", "Hari_Dalam_Tahun", "Jarak_Ke_Hari_Raya"]

# Load data bersih kaya fitur dengan cache
@st.cache_data
def load_clean_data(clean_path, mtime):
    # Jika clean file belum ada, picu update_harian.py secara lokal untuk membuatnya
    if not clean_path.exists() or clean_path.stat().st_size == 0:
        import subprocess
        import sys
        subprocess.run([sys.executable, str(BASE_DIR / "update_harian.py")], check=True)
        
    df_c = pd.read_csv(clean_path)
    df_c["Tanggal"] = pd.to_datetime(df_c["Tanggal"])
    df_c["Is_Imputed"] = df_c["Is_Imputed"].astype(bool)
    return df_c

# Membaca tanggal data mentah asli yang dilaporkan untuk menyaring visualisasi ffill
@st.cache_data
def load_raw_dataset(raw_path, mtime):
    df_r = pd.read_csv(raw_path)
    df_r["Tanggal"] = pd.to_datetime(df_r["Tanggal"])
    return df_r[["Tanggal", "Pasar", "Komoditas", "Harga"]].dropna()

try:
    df = load_clean_data(CLEAN_FILE, CLEAN_FILE.stat().st_mtime if CLEAN_FILE.exists() else 0)
    df_raw_actual = load_raw_dataset(DATA_FILE, DATA_FILE.stat().st_mtime if DATA_FILE.exists() else 0)
except Exception as e:
    st.error(f"Gagal memuat data: {e}")
    st.stop()

HOLIDAYS = sorted(pd.to_datetime([
    # 2017
    "2017-01-01", "2017-06-25", "2017-09-01", "2017-12-25",
    # 2018
    "2018-01-01", "2018-06-15", "2018-08-21", "2018-12-25",
    # 2019
    "2019-01-01", "2019-06-05", "2019-08-11", "2019-12-25",
    # 2020
    "2020-01-01", "2020-05-24", "2020-07-31", "2020-12-25",
    # 2021
    "2021-01-01", "2021-05-13", "2021-07-20", "2021-12-25",
    # 2022
    "2022-01-01", "2022-05-02", "2022-07-09", "2022-12-25",
    # 2023
    "2023-01-01", "2023-04-21", "2023-06-29", "2023-12-25",
    # 2024
    "2024-01-01", "2024-04-10", "2024-06-16", "2024-12-25",
    # 2025
    "2025-01-01", "2025-03-30", "2025-06-06", "2025-12-25",
    # 2026
    "2026-01-01", "2026-03-20", "2026-05-27", "2026-12-25",
    # 2027
    "2027-01-01", "2027-03-09", "2027-05-16", "2027-12-25"
]))

class SimpleLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 16, batch_first=True)
        self.fc = nn.Linear(16, 14)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

@st.cache_data
def get_forecast(selected_commodity, selected_market, _df_full, days_to_predict=14):
    # Filter data spesifik komoditas dan pasar
    if selected_market == "Semua Pasar (Gabungan)":
        sub_df = _df_full[_df_full["Komoditas"] == selected_commodity].groupby("Tanggal").agg({
            "Harga_Bersih": "mean",
            "Curah_Hujan": "mean",
            "Suhu_Rata": "mean",
            "MBG_Aktif": "max",
            "Sentimen_Berita": "mean",
            "Is_Imputed": "max"
        }).reset_index()
        sub_df["Pasar"] = "Semua Pasar (Gabungan)"
        sub_df["Komoditas"] = selected_commodity
    else:
        sub_df = _df_full[(_df_full["Komoditas"] == selected_commodity) & (_df_full["Pasar"] == selected_market)].copy()
        
    sub_df = sub_df.sort_values("Tanggal").drop_duplicates(subset=["Tanggal"]).reset_index(drop=True)
    
    # Hitung Hari_Dalam_Tahun dan Jarak_Ke_Hari_Raya
    sub_df["Hari_Dalam_Tahun"] = sub_df["Tanggal"].dt.dayofyear
    idx = np.searchsorted(HOLIDAYS, sub_df["Tanggal"])
    idx = np.clip(idx, 0, len(HOLIDAYS) - 1)
    next_holidays = pd.to_datetime([HOLIDAYS[i] for i in idx])
    sub_df["Jarak_Ke_Hari_Raya"] = (next_holidays - sub_df["Tanggal"]).dt.days
    
    # Generate future dates
    future_dates = pd.date_range(start=sub_df["Tanggal"].max() + pd.Timedelta(days=1), periods=days_to_predict, freq='D')
    
    if len(sub_df) <= 100:
        last_val = sub_df["Harga_Bersih"].iloc[-1] if not sub_df.empty else 0.0
        f_pred_rf = pd.DataFrame({"Tanggal": future_dates, "Harga_Prediksi": [last_val] * days_to_predict})
        f_pred_rf["Harga_Prediksi"] = np.round(f_pred_rf["Harga_Prediksi"] / 500.0) * 500.0
        
        future_preds = {"Random Forest": f_pred_rf, "XGBoost": f_pred_rf.copy(), "LSTM": f_pred_rf.copy()}
        metrics = {
            "Random Forest": (0.0, 0.0),
            "XGBoost": (0.0, 0.0),
            "LSTM": (0.0, 0.0)
        }
        return future_preds, metrics, pd.DataFrame(), {}

    eval_df = sub_df.copy()
    
    # Validasi 14 hari terakhir
    cutoff_date = eval_df["Tanggal"].max() - pd.Timedelta(days=14)
    train_df = eval_df[eval_df["Tanggal"] <= cutoff_date].copy()
    test_df = eval_df[eval_df["Tanggal"] > cutoff_date].copy()
    
    train_df["Lag_1"] = train_df["Harga_Bersih"].shift(1)
    train_df["Lag_2"] = train_df["Harga_Bersih"].shift(2)
    train_df["Lag_7"] = train_df["Harga_Bersih"].shift(7)
    train_df_clean = train_df.dropna(subset=["Lag_1", "Lag_2", "Lag_7"])
    
    features = ["Curah_Hujan", "Suhu_Rata", "MBG_Aktif", "Sentimen_Berita", "Lag_1", "Lag_2", "Lag_7", "Hari_Dalam_Tahun", "Jarak_Ke_Hari_Raya"]
    
    if len(train_df_clean) <= 60 or len(test_df) == 0:
        last_val = eval_df["Harga_Bersih"].iloc[-1]
        f_pred_rf = pd.DataFrame({"Tanggal": future_dates, "Harga_Prediksi": [last_val] * days_to_predict})
        f_pred_rf["Harga_Prediksi"] = np.round(f_pred_rf["Harga_Prediksi"] / 500.0) * 500.0
        
        future_preds = {"Random Forest": f_pred_rf, "XGBoost": f_pred_rf.copy(), "LSTM": f_pred_rf.copy()}
        metrics = {
            "Random Forest": (0.0, 0.0),
            "XGBoost": (0.0, 0.0),
            "LSTM": (0.0, 0.0)
        }
        return future_preds, metrics, pd.DataFrame(), {}
        
    # --- TRAINING MODEL VALIDASI (14 HARI TERAKHIR) ---
    preds_rf_val = [0.0] * days_to_predict
    preds_xgb_val = [0.0] * days_to_predict
    
    lag1_val = train_df["Harga_Bersih"].iloc[-1]
    lag2_val = train_df["Harga_Bersih"].iloc[-2] if len(train_df) > 1 else lag1_val
    lag7_val = train_df["Harga_Bersih"].iloc[-7] if len(train_df) > 6 else lag1_val
    
    # 1. RF & XGBoost (Direct Validation)
    for k in range(days_to_predict):
        train_df_clean[f"Target_H_{k+1}"] = train_df_clean["Harga_Bersih"].shift(-(k+1))
        train_df_h = train_df_clean.dropna(subset=[f"Target_H_{k+1}"])
        
        if len(train_df_h) > 30:
            rf_h = RandomForestRegressor(n_estimators=20, max_depth=5, random_state=42, n_jobs=-1)
            rf_h.fit(train_h := train_df_h[features].fillna(0), train_df_h[f"Target_H_{k+1}"])
            
            xgb_h = xgb.XGBRegressor(n_estimators=20, max_depth=4, learning_rate=0.1, random_state=42, n_jobs=-1)
            xgb_h.fit(train_h, train_df_h[f"Target_H_{k+1}"])
            
            row_test = test_df.iloc[k]
            row_feat = pd.DataFrame([{
                "Curah_Hujan": row_test["Curah_Hujan"],
                "Suhu_Rata": row_test["Suhu_Rata"],
                "MBG_Aktif": row_test["MBG_Aktif"],
                "Sentimen_Berita": row_test["Sentimen_Berita"],
                "Lag_1": lag1_val,
                "Lag_2": lag2_val,
                "Lag_7": lag7_val,
                "Hari_Dalam_Tahun": row_test["Hari_Dalam_Tahun"],
                "Jarak_Ke_Hari_Raya": row_test["Jarak_Ke_Hari_Raya"]
            }])
            preds_rf_val[k] = rf_h.predict(row_feat)[0]
            preds_xgb_val[k] = xgb_h.predict(row_feat)[0]
        else:
            preds_rf_val[k] = lag1_val
            preds_xgb_val[k] = lag1_val
            
    pred_rf_val_df = test_df.copy()
    pred_rf_val_df["yhat"] = np.round(np.array(preds_rf_val) / 500.0) * 500.0
    mape_rf = np.mean(np.abs((pred_rf_val_df["Harga_Bersih"] - pred_rf_val_df["yhat"]) / pred_rf_val_df["Harga_Bersih"])) * 100
    mae_rf = np.mean(np.abs(pred_rf_val_df["Harga_Bersih"] - pred_rf_val_df["yhat"]))

    pred_xgb_val_df = test_df.copy()
    pred_xgb_val_df["yhat"] = np.round(np.array(preds_xgb_val) / 500.0) * 500.0
    mape_xgb = np.mean(np.abs((pred_xgb_val_df["Harga_Bersih"] - pred_xgb_val_df["yhat"]) / pred_xgb_val_df["Harga_Bersih"])) * 100
    mae_xgb = np.mean(np.abs(pred_xgb_val_df["Harga_Bersih"] - pred_xgb_val_df["yhat"]))

    # 2. LSTM (Validation)
    prices_train = train_df["Harga_Bersih"].values
    p_min_val, p_max_val = prices_train.min(), prices_train.max()
    scaled_train = (prices_train - p_min_val) / (p_max_val - p_min_val) if p_max_val > p_min_val else prices_train * 0.0
    
    window_size = 30
    X_lstm, y_lstm = [], []
    for i in range(len(scaled_train) - window_size - days_to_predict + 1):
        X_lstm.append(scaled_train[i : i + window_size])
        y_lstm.append(scaled_train[i + window_size : i + window_size + days_to_predict])
        
    if len(X_lstm) > 10:
        X_lstm_t = torch.FloatTensor(np.array(X_lstm)[:, :, np.newaxis])
        y_lstm_t = torch.FloatTensor(np.array(y_lstm))
        
        lstm_val = SimpleLSTM()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(lstm_val.parameters(), lr=0.01)
        
        lstm_val.train()
        for epoch in range(60):
            optimizer.zero_grad()
            loss = criterion(lstm_val(X_lstm_t), y_lstm_t)
            loss.backward()
            optimizer.step()
            
        lstm_val.eval()
        with torch.no_grad():
            last_w = scaled_train[-window_size:]
            input_t = torch.FloatTensor(last_w).view(1, window_size, 1)
            pred_s = lstm_val(input_t).numpy()[0]
            preds_lstm_val = pred_s * (p_max_val - p_min_val) + p_min_val
    else:
        preds_lstm_val = [lag1_val] * days_to_predict

    pred_lstm_val_df = test_df.copy()
    pred_lstm_val_df["yhat"] = np.round(np.array(preds_lstm_val) / 500.0) * 500.0
    mape_lstm = np.mean(np.abs((pred_lstm_val_df["Harga_Bersih"] - pred_lstm_val_df["yhat"]) / pred_lstm_val_df["Harga_Bersih"])) * 100
    mae_lstm = np.mean(np.abs(pred_lstm_val_df["Harga_Bersih"] - pred_lstm_val_df["yhat"]))

    # --- TRAINING MODEL FINAL (PROYEKSI MASA DEPAN) ---
    eval_df["Lag_1"] = eval_df["Harga_Bersih"].shift(1)
    eval_df["Lag_2"] = eval_df["Harga_Bersih"].shift(2)
    eval_df["Lag_7"] = eval_df["Harga_Bersih"].shift(7)
    eval_df_clean = eval_df.dropna(subset=["Lag_1", "Lag_2", "Lag_7"]).copy()
    
    lag1_final = eval_df["Harga_Bersih"].iloc[-1]
    lag2_final = eval_df["Harga_Bersih"].iloc[-2] if len(eval_df) > 1 else lag1_final
    lag7_final = eval_df["Harga_Bersih"].iloc[-7] if len(eval_df) > 6 else lag1_final
    
    # Generate Weather & News for future
    from update_harian import fetch_weather_data, get_mbg_status, fetch_news_sentiment
    df_w_fut = fetch_weather_data(future_dates.min(), future_dates.max())
    news_sent_fut = fetch_news_sentiment()
    
    future_features = pd.DataFrame({"Tanggal": future_dates})
    future_features = future_features.merge(df_w_fut, on="Tanggal", how="left").fillna(0)
    future_features["MBG_Aktif"] = future_features["Tanggal"].apply(get_mbg_status)
    future_features["Sentimen_Berita"] = news_sent_fut
    future_features["Hari_Dalam_Tahun"] = future_features["Tanggal"].dt.dayofyear
    idx_fut = np.searchsorted(HOLIDAYS, future_features["Tanggal"])
    idx_fut = np.clip(idx_fut, 0, len(HOLIDAYS) - 1)
    next_holidays_fut = pd.to_datetime([HOLIDAYS[i] for i in idx_fut])
    future_features["Jarak_Ke_Hari_Raya"] = (next_holidays_fut - future_features["Tanggal"]).dt.days
    
    preds_rf_future = [0.0] * days_to_predict
    preds_xgb_future = [0.0] * days_to_predict
    
    for k in range(days_to_predict):
        eval_df_clean[f"Target_H_{k+1}"] = eval_df_clean["Harga_Bersih"].shift(-(k+1))
        eval_df_h = eval_df_clean.dropna(subset=[f"Target_H_{k+1}"])
        
        if len(eval_df_h) > 30:
            rf_f = RandomForestRegressor(n_estimators=20, max_depth=5, random_state=42, n_jobs=-1)
            rf_f.fit(eval_h := eval_df_h[features].fillna(0), eval_df_h[f"Target_H_{k+1}"])
            
            xgb_f = xgb.XGBRegressor(n_estimators=20, max_depth=4, learning_rate=0.1, random_state=42, n_jobs=-1)
            xgb_f.fit(eval_h, eval_df_h[f"Target_H_{k+1}"])
            
            row_feat = pd.DataFrame([{
                "Curah_Hujan": future_features["Curah_Hujan"].iloc[k],
                "Suhu_Rata": future_features["Suhu_Rata"].iloc[k],
                "MBG_Aktif": future_features["MBG_Aktif"].iloc[k],
                "Sentimen_Berita": future_features["Sentimen_Berita"].iloc[k],
                "Lag_1": lag1_final,
                "Lag_2": lag2_final,
                "Lag_7": lag7_final,
                "Hari_Dalam_Tahun": future_features["Hari_Dalam_Tahun"].iloc[k],
                "Jarak_Ke_Hari_Raya": future_features["Jarak_Ke_Hari_Raya"].iloc[k]
            }])
            preds_rf_future[k] = rf_f.predict(row_feat)[0]
            preds_xgb_future[k] = xgb_f.predict(row_feat)[0]
        else:
            preds_rf_future[k] = lag1_final
            preds_xgb_future[k] = lag1_final
            
    # Final LSTM Model
    prices_eval = eval_df["Harga_Bersih"].values
    p_min_eval, p_max_eval = prices_eval.min(), prices_eval.max()
    scaled_eval = (prices_eval - p_min_eval) / (p_max_eval - p_min_eval) if p_max_eval > p_min_eval else prices_eval * 0.0
    
    X_lstm_f, y_lstm_f = [], []
    for i in range(len(scaled_eval) - window_size - days_to_predict + 1):
        X_lstm_f.append(scaled_eval[i : i + window_size])
        y_lstm_f.append(scaled_eval[i + window_size : i + window_size + days_to_predict])
        
    if len(X_lstm_f) > 10:
        X_lstm_f_t = torch.FloatTensor(np.array(X_lstm_f)[:, :, np.newaxis])
        y_lstm_f_t = torch.FloatTensor(np.array(y_lstm_f))
        
        lstm_f = SimpleLSTM()
        optimizer_f = optim.Adam(lstm_f.parameters(), lr=0.01)
        
        lstm_f.train()
        for epoch in range(60):
            optimizer_f.zero_grad()
            loss = criterion(lstm_f(X_lstm_f_t), y_lstm_f_t)
            loss.backward()
            optimizer_f.step()
            
        lstm_f.eval()
        with torch.no_grad():
            last_w_f = scaled_eval[-window_size:]
            input_f_t = torch.FloatTensor(last_w_f).view(1, window_size, 1)
            pred_s_f = lstm_f(input_f_t).numpy()[0]
            preds_lstm_future = pred_s_f * (p_max_eval - p_min_eval) + p_min_eval
    else:
        preds_lstm_future = [lag1_final] * days_to_predict
        
    # Compile Dataframes
    future_preds = {
        "Random Forest": pd.DataFrame({"Tanggal": future_dates, "Harga_Prediksi": np.round(np.array(preds_rf_future) / 500.0) * 500.0}),
        "XGBoost": pd.DataFrame({"Tanggal": future_dates, "Harga_Prediksi": np.round(np.array(preds_xgb_future) / 500.0) * 500.0}),
        "LSTM": pd.DataFrame({"Tanggal": future_dates, "Harga_Prediksi": np.round(np.array(preds_lstm_future) / 500.0) * 500.0})
    }
    
    metrics = {
        "Random Forest": (mape_rf, mae_rf),
        "XGBoost": (mape_xgb, mae_xgb),
        "LSTM": (mape_lstm, mae_lstm)
    }
    
    pred_vals = {
        "Random Forest": pred_rf_val_df,
        "XGBoost": pred_xgb_val_df,
        "LSTM": pred_lstm_val_df
    }
    
    return future_preds, metrics, test_df, pred_vals

def get_forecast_precalculated(commodity, market, _df):
    forecast_path = BASE_DIR / "forecast_14_hari.csv"
    metrics_path = BASE_DIR / "validation_metrics.csv"
    val_detail_path = BASE_DIR / "validation_detail.csv"
    
    try:
        if forecast_path.exists() and metrics_path.exists():
            f_df = pd.read_csv(forecast_path)
            f_df["Tanggal"] = pd.to_datetime(f_df["Tanggal"])
            f_sub = f_df[(f_df["Komoditas"] == commodity) & (f_df["Pasar"] == market)].copy()
            f_sub = f_sub.sort_values("Tanggal").reset_index(drop=True)
            
            m_df = pd.read_csv(metrics_path)
            m_row = m_df[(m_df["Komoditas"] == commodity) & (m_df["Pasar"] == market)].iloc[0]
            
            if val_detail_path.exists():
                v_df = pd.read_csv(val_detail_path)
                v_df["Tanggal"] = pd.to_datetime(v_df["Tanggal"])
                v_sub = v_df[(v_df["Komoditas"] == commodity) & (v_df["Pasar"] == market)].copy()
            else:
                v_sub = pd.DataFrame()
                
            return f_sub, m_row["MAPE_Validasi"], m_row["MAE_Validasi"], v_sub, v_sub
    except:
        pass
    return get_forecast(commodity, market, _df)

def format_tanggal_indonesia(tanggal):
    bulan = {
        1: "Januari", 2: "Februari", 3: "Maret", 4: "April",
        5: "Mei", 6: "Juni", 7: "Juli", 8: "Agustus",
        9: "September", 10: "Oktober", 11: "November", 12: "Desember"
    }
    return f"{tanggal.day} {bulan[tanggal.month]} {tanggal.year}"

tanggal_terakhir = df["Tanggal"].max()
status_data_text = f"Status Data: Data harga pasar telah diperbarui hingga {format_tanggal_indonesia(tanggal_terakhir)}."

# Navigasi Halaman
st.sidebar.title("Navigasi Analisis")
page = st.sidebar.radio(
    "Pilih Halaman:",
    [
        "Dashboard Tren Harga",
        "Deteksi Anomali Harga",
        "Peramalan Harga (Forecasting)",
        "Disparitas & Korelasi Pasar",
        "Kualitas Data",
        "Tanya Jawab AI (Chatbot)"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Data Pangan Kabupaten Cirebon**")
st.sidebar.markdown("Rentang Data: 2017 - Juni 2026")
st.sidebar.markdown(f"Jumlah komoditas: {df['Komoditas'].nunique()}")
st.sidebar.markdown(f"Jumlah baris: {len(df):,}")
st.sidebar.markdown(f"Total Korelasi/Imputasi: {df['Is_Imputed'].sum():,}")

st.sidebar.markdown("---")
st.sidebar.markdown("### Pembuat & Kontributor")
st.sidebar.markdown("**Oleh Badar Rahman**")
st.sidebar.markdown("[GitHub: DarRahman](https://github.com/DarRahman)")

st.sidebar.markdown("### Sumber Data Utama")
st.sidebar.markdown("- **[Kemendag Kepokmas Cirebon](http://kepokmas.cirebonkab.go.id)**: Data Harga Harian")
st.sidebar.markdown("- **[Open-Meteo API](https://open-meteo.com)**: Suhu & Curah Hujan")
st.sidebar.markdown("- **[Google News RSS](https://news.google.com)**: Sentimen & Kebijakan MBG")

# ==========================================
# HALAMAN 1: DASHBOARD TREN HARGA
# ==========================================
if page == "Dashboard Tren Harga":
    st.title("Dashboard Tren Harga Pangan Pokok")
    st.info(status_data_text)
    st.write("Visualisasi tren perkembangan harga pangan di berbagai pasar Kabupaten Cirebon.")

    col1, col2 = st.columns(2)
    with col1:
        selected_commodity = st.selectbox("Pilih Komoditas:", df["Komoditas"].unique())
    with col2:
        market_options = list(df["Pasar"].unique()) + ["Semua Pasar (Gabungan)"]
        selected_markets = st.multiselect(
            "Pilih Pasar:",
            market_options,
            default=list(df["Pasar"].unique())
        )

    if selected_markets:
        real_markets = [m for m in selected_markets if m != "Semua Pasar (Gabungan)"]
        has_gabungan = "Semua Pasar (Gabungan)" in selected_markets
        
        dfs_to_concat = []
        actual_subsets = []
        
        if real_markets:
            df_real = df[
                (df["Komoditas"] == selected_commodity) & 
                (df["Pasar"].isin(real_markets))
            ].copy()
            dfs_to_concat.append(df_real)
            
            actual_subsets.append(df_raw_actual[
                (df_raw_actual["Komoditas"] == selected_commodity) & 
                (df_raw_actual["Pasar"].isin(real_markets))
            ])
            
        if has_gabungan:
            df_all_comm = df[df["Komoditas"] == selected_commodity].copy()
            df_gabungan = df_all_comm.groupby("Tanggal").agg({
                "Harga_Bersih": "mean",
                "Curah_Hujan": "mean",
                "Suhu_Rata": "mean",
                "MBG_Aktif": "max",
                "Sentimen_Berita": "mean",
                "Is_Imputed": "max"
            }).reset_index()
            df_gabungan["Pasar"] = "Semua Pasar (Gabungan)"
            df_gabungan["Komoditas"] = selected_commodity
            dfs_to_concat.append(df_gabungan)
            
            df_raw_all = df_raw_actual[df_raw_actual["Komoditas"] == selected_commodity].copy()
            dates_with_reports = df_raw_all[["Tanggal", "Komoditas"]].drop_duplicates().copy()
            dates_with_reports["Pasar"] = "Semua Pasar (Gabungan)"
            actual_subsets.append(dates_with_reports)
            
        filtered_df = pd.concat(dfs_to_concat, ignore_index=True) if dfs_to_concat else pd.DataFrame()
        actual_subset = pd.concat(actual_subsets, ignore_index=True) if actual_subsets else pd.DataFrame()
        
        if not filtered_df.empty and not actual_subset.empty:
            filtered_visual_df = filtered_df.merge(
                actual_subset[["Tanggal", "Pasar", "Komoditas"]],
                on=["Tanggal", "Pasar", "Komoditas"],
                how="inner"
            )
        else:
            filtered_visual_df = filtered_df.copy()
            
        if filtered_visual_df.empty:
            filtered_visual_df = filtered_df.copy()

        # Buat Line Chart dengan Plotly untuk visualisasi tren harga historis dinamis yang dilaporkan asli
        fig = px.line(
            filtered_visual_df,
            x="Tanggal",
            y="Harga_Bersih",
            color="Pasar",
            title=f"Tren Harga {selected_commodity} (Data Bersih Aktual)",
            labels={"Harga_Bersih": "Harga (Rp)", "Tanggal": "Tanggal"},
            template="plotly_white",
            markers=True
        )
        
        # Optimasi UI Plotly (Legend Click behavior)
        fig.update_layout(
            legend=dict(
                title="Pasar (Klik legenda untuk filter)",
                orientation="h",
                yanchor="bottom",
                y=-0.3,
                xanchor="center",
                x=0.5
            ),
            legend_itemclick="toggle", # Klik legenda menyembunyikan/menampilkan garis pasar
            legend_itemdoubleclick="toggleothers" # Klik ganda menyembunyikan pasar lain
        )
        
        st.plotly_chart(fig, use_container_width=True)

        # Analisis Insight Otomatis (Tahap 4)
        st.subheader("Analisis Tren dan Ringkasan Informasi")
        
        # Ekstrak metrik untuk analisis insight
        latest_date = filtered_df["Tanggal"].max()
        df_latest = filtered_df[filtered_df["Tanggal"] == latest_date]
        df_latest_real = df_latest[df_latest["Pasar"] != "Semua Pasar (Gabungan)"]
        
        if len(df_latest_real) > 0:
            highest_row = df_latest_real.loc[df_latest_real["Harga_Bersih"].idxmax()]
            lowest_row = df_latest_real.loc[df_latest_real["Harga_Bersih"].idxmin()]
            disparitas = highest_row["Harga_Bersih"] - lowest_row["Harga_Bersih"]
            
            # Hitung perubahan tren 7 hari terakhir
            tgl_7_hari_lalu = latest_date - pd.Timedelta(days=7)
            df_past = filtered_df[filtered_df["Tanggal"] == tgl_7_hari_lalu]
            
            perubahan_teks = []
            for m in selected_markets:
                m_latest = df_latest[df_latest["Pasar"] == m]
                m_past = df_past[df_past["Pasar"] == m]
                if not m_latest.empty and not m_past.empty and pd.notna(m_latest["Harga_Bersih"].iloc[0]) and pd.notna(m_past["Harga_Bersih"].iloc[0]):
                    p_diff = int(m_latest["Harga_Bersih"].iloc[0] - m_past["Harga_Bersih"].iloc[0])
                    if p_diff > 0:
                        perubahan_teks.append(f"kenaikan sebesar Rp {p_diff:,} di {m}")
                    elif p_diff < 0:
                        perubahan_teks.append(f"penurunan sebesar Rp {abs(p_diff):,} di {m}")
            
            tren_desc = "Harga cenderung stabil tanpa ada fluktuasi signifikan dalam tujuh hari terakhir."
            if perubahan_teks:
                tren_desc = f"Dalam tujuh hari terakhir, tercatat perubahan harga berupa " + ", ".join(perubahan_teks) + "."
            
            st.info(
                f"Berdasarkan data tanggal {latest_date.strftime('%Y-%m-%d')}, komoditas {selected_commodity} memiliki harga tertinggi di {highest_row['Pasar']} "
                f"(Rp {int(highest_row['Harga_Bersih']):,}) dan harga terendah di {lowest_row['Pasar']} (Rp {int(lowest_row['Harga_Bersih']):,}), "
                f"dengan selisih disparitas antar pasar fisik sebesar Rp {int(disparitas):,}. {tren_desc}"
            )
        else:
            st.write("Data tidak mencukupi untuk menyusun analisis informasi otomatis.")

        # Ringkasan Statistik
        st.subheader("Statistik Ringkas (Data Bersih)")
        summary_data = []
        for m in selected_markets:
            m_df = filtered_df[filtered_df["Pasar"] == m]
            if not m_df.empty:
                m_df_clean = m_df.dropna(subset=["Harga_Bersih"])
                if not m_df_clean.empty:
                    latest_row = m_df_clean.iloc[-1]
                    summary_data.append({
                        "Pasar": m,
                        "Harga Terakhir": f"Rp {int(latest_row['Harga_Bersih']):,}",
                        "Harga Minimum": f"Rp {int(m_df_clean['Harga_Bersih'].min()):,}",
                        "Harga Maksimum": f"Rp {int(m_df_clean['Harga_Bersih'].max()):,}",
                        "Rata-rata Harga": f"Rp {int(m_df_clean['Harga_Bersih'].mean()):,}"
                    })
        st.table(pd.DataFrame(summary_data))
    else:
        st.warning("Silakan pilih minimal satu pasar untuk menampilkan grafik.")

# ==========================================
# HALAMAN 2: DETEKSI ANOMALI HARGA (AUDIT TRAIL VIEW)
# ==========================================
elif page == "Deteksi Anomali Harga":
    st.title("Deteksi Anomali Lonjakan Harga")
    st.info(status_data_text)
    st.write("Mengidentifikasi data input yang salah (typo/anomali) dan melacak nilai asli vs nilai bersih hasil imputasi.")

    col1, col2 = st.columns(2)
    with col1:
        selected_commodity = st.selectbox("Pilih Komoditas:", df["Komoditas"].unique())
    with col2:
        selected_market = st.selectbox("Pilih Pasar:", df["Pasar"].unique())

    # Filter data spesifik
    sub_df = df[(df["Komoditas"] == selected_commodity) & (df["Pasar"] == selected_market)].copy()
    sub_df = sub_df.sort_values("Tanggal").reset_index(drop=True)

    if len(sub_df) > 10:
        imputed_points = sub_df[sub_df["Is_Imputed"] == True]
        total_anomali = len(imputed_points)
        
        # Cari koreksi terbesar (selisih absolut Harga_Original - Harga_Bersih)
        if total_anomali > 0:
            diff_series = np.abs(sub_df["Harga_Original"] - sub_df["Harga_Bersih"])
            max_diff_idx = diff_series.idxmax()
            max_diff_row = sub_df.loc[max_diff_idx]
            tgl_koreksi_terbesar = max_diff_row["Tanggal"].strftime("%Y-%m-%d")
            nilai_koreksi_terbesar = int(diff_series.max())
            
            # Summary Metric Banner
            st.info(
                f"Ditemukan {total_anomali} anomali pada komoditas ini, "
                f"dengan koreksi terbesar pada tanggal {tgl_koreksi_terbesar} (Selisih: Rp {nilai_koreksi_terbesar:,})."
            )
        else:
            st.success("Tidak ada anomali input atau pembersihan harga terdeteksi pada komoditas ini.")

        # Plot Perbandingan Audit Trail
        fig = go.Figure()
        
        # Harga Original
        fig.add_trace(go.Scatter(
            x=sub_df["Tanggal"],
            y=sub_df["Harga_Original"],
            mode='lines',
            name='Harga Original (Asli Web)',
            line=dict(color='red', width=1, dash='dot')
        ))

        # Harga Bersih (Imputasi)
        fig.add_trace(go.Scatter(
            x=sub_df["Tanggal"],
            y=sub_df["Harga_Bersih"],
            mode='lines',
            name='Harga Bersih (Hasil Imputasi)',
            line=dict(color='blue', width=2)
        ))

        # Titik Anomali yang Diperbaiki
        if total_anomali > 0:
            fig.add_trace(go.Scatter(
                x=imputed_points["Tanggal"],
                y=imputed_points["Harga_Original"],
                mode='markers',
                name='Outlier Terdeteksi & Dikoreksi',
                marker=dict(color='red', size=8, symbol='x')
            ))

        fig.update_layout(
            title=f"Audit Trail Data: Harga Original vs Harga Bersih untuk {selected_commodity} di {selected_market}",
            xaxis_title="Tanggal",
            yaxis_title="Harga (Rp)",
            yaxis=dict(rangemode="tozero"),
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Daftar Riwayat Perbaikan Data (Audit Trail)")
        if total_anomali > 0:
            imputed_view = imputed_points[["Tanggal", "Harga_Original", "Harga_Bersih"]]
            st.dataframe(imputed_view.tail(20))
        else:
            st.write("Tidak ada riwayat perbaikan data.")
    else:
        st.warning("Data tidak mencukupi untuk melakukan analisis runtun waktu.")

# ==========================================
# HALAMAN 3: PERAMALAN HARGA (FORECASTING)
# ==========================================
elif page == "Peramalan Harga (Forecasting)":
    st.title("Peramalan Harga Pangan Cirebon (Random Forest 14 Hari)")
    st.info(status_data_text)
    st.write("Prediksi pergerakan harga pangan 14 hari ke depan menggunakan model multivariat Random Forest berbasis pengaruh eksternal (cuaca, sentimen, kebijakan, dan hari raya).")

    col1, col2 = st.columns(2)
    with col1:
        selected_commodity = st.selectbox("Pilih Komoditas:", df["Komoditas"].unique())
    with col2:
        market_options = list(df["Pasar"].unique()) + ["Semua Pasar (Gabungan)"]
        selected_market = st.selectbox("Pilih Pasar:", market_options)

    # Filter data spesifik komoditas dan pasar
    if selected_market == "Semua Pasar (Gabungan)":
        sub_df = df[df["Komoditas"] == selected_commodity].groupby("Tanggal").agg({
            "Harga_Bersih": "mean",
            "Curah_Hujan": "mean",
            "Suhu_Rata": "mean",
            "MBG_Aktif": "max",
            "Sentimen_Berita": "mean",
            "Is_Imputed": "max"
        }).reset_index()
        sub_df["Pasar"] = "Semua Pasar (Gabungan)"
        sub_df["Komoditas"] = selected_commodity
    else:
        sub_df = df[(df["Komoditas"] == selected_commodity) & (df["Pasar"] == selected_market)].copy()
        
    sub_df = sub_df.sort_values("Tanggal").drop_duplicates(subset=["Tanggal"]).reset_index(drop=True)

    if len(sub_df) > 100:
        # Latih model secara dinamis untuk perbandingan 3 model sekaligus
        with st.spinner("Melatih model Random Forest, XGBoost, dan LSTM secara bersamaan..."):
            try:
                future_preds, metrics, test_df, pred_vals = get_forecast(selected_commodity, selected_market, df, days_to_predict=14)
            except Exception as e:
                st.error(f"Gagal melatih model: {e}")
                st.stop()

        # Plot Grafik Prediksi Utama (Perbandingan 3 Model untuk Masa Depan)
        fig = go.Figure()
        
        # Historis 6 bulan terakhir
        recent_df = sub_df.loc[sub_df["Tanggal"] >= sub_df["Tanggal"].max() - pd.Timedelta(days=180)]
        fig.add_trace(go.Scatter(
            x=recent_df["Tanggal"],
            y=recent_df["Harga_Bersih"],
            mode='lines',
            name='Harga Historis Bersih (180 Hari Terakhir)',
            line=dict(color='#00BFFF', width=2)
        ))

        # Prediksi masa depan untuk ketiga model
        colors_fut = {
            "Random Forest": "red",
            "XGBoost": "green",
            "LSTM": "orange"
        }
        for model_name, fut_df in future_preds.items():
            fig.add_trace(go.Scatter(
                x=fut_df["Tanggal"],
                y=fut_df["Harga_Prediksi"],
                mode='lines+markers',
                name=f'Prediksi {model_name} (14 Hari)',
                line=dict(color=colors_fut[model_name], dash='dash')
            ))

        fig.update_layout(
            title=f"Peramalan Harga {selected_commodity} di {selected_market} (Horizon: 14 Hari) - Perbandingan 3 Model",
            xaxis_title="Tanggal",
            yaxis_title="Harga (Rp)",
            yaxis=dict(rangemode="tozero"),
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Tabel perbandingan prediksi masa depan
        st.subheader("Tabel Prediksi Harga 14 Hari Ke Depan (Kelipatan Rp 500 Terdekat)")
        tbl_df = pd.DataFrame({"Tanggal": future_preds["Random Forest"]["Tanggal"].dt.strftime("%Y-%m-%d")})
        for model_name, fut_df in future_preds.items():
            tbl_df[model_name] = fut_df["Harga_Prediksi"].astype(int)
        st.dataframe(tbl_df)

        # EXPANDER DETAIL AKURASI VALIDASI (UJI COBA)
        st.markdown("---")
        with st.expander("Detail Akurasi & Komparasi Validasi (Simulasi 14 Hari Terakhir)"):
            st.write("Perbandingan performa ketiga model ketika diuji coba menebak data 14 hari terakhir sebelum rilis:")
            
            # Buat tabel ringkasan metrik akurasi
            metrics_summary = []
            for model_name, (mape, mae) in metrics.items():
                metrics_summary.append({
                    "Model": model_name,
                    "Rata-rata Persentase Error (MAPE)": f"{mape:.2f}%",
                    "Rata-rata Selisih Harga (MAE)": f"Rp {int(mae):,}"
                })
            st.table(pd.DataFrame(metrics_summary))
            
            # Plot komparasi tebakan vs aktual lapangan
            if not test_df.empty:
                fig_eval = go.Figure()
                fig_eval.add_trace(go.Scatter(x=test_df["Tanggal"], y=test_df["Harga_Bersih"], mode='lines+markers', name='Harga Aktual (Lapangan)', line=dict(color='#00BFFF', width=3)))
                for model_name, p_val in pred_vals.items():
                    fig_eval.add_trace(go.Scatter(
                        x=p_val["Tanggal"],
                        y=p_val["yhat"],
                        mode='lines+markers',
                        name=f'{model_name} (MAPE: {metrics[model_name][0]:.2f}%)',
                        line=dict(dash='dash', color=colors_fut[model_name])
                    ))
                fig_eval.update_layout(
                    title="Perbandingan Tebakan Uji Coba 3 Model vs Harga Lapangan Asli",
                    xaxis_title="Tanggal",
                    yaxis_title="Harga (Rp)",
                    yaxis=dict(rangemode="tozero"),
                    template="plotly_white"
                )
                st.plotly_chart(fig_eval, use_container_width=True)
            else:
                st.info("Data uji coba tidak tersedia.")

        # DETAIL PENGARUH FAKTOR EKSTERNAL
        with st.expander("Faktor Eksternal Real-Time Cirebon (Pengaruh Regresi ML)"):
            st.write("Faktor lingkungan dan kebijakan lokal yang dibaca oleh model Random Forest saat ini:")
            col_f1, col_f2, col_f3, col_f4 = st.columns(4)
            
            cur_rain = sub_df["Curah_Hujan"].iloc[-1] if "Curah_Hujan" in sub_df.columns else 0.0
            cur_temp = sub_df["Suhu_Rata"].iloc[-1] if "Suhu_Rata" in sub_df.columns else 27.0
            cur_mbg = "Aktif (Hari Sekolah)" if (sub_df["MBG_Aktif"].iloc[-1] == 1 if "MBG_Aktif" in sub_df.columns else False) else "Nonaktif (Libur)"
            cur_news = sub_df["Sentimen_Berita"].iloc[-1] if "Sentimen_Berita" in sub_df.columns else 0.0
            
            col_f1.metric("Curah Hujan Terakhir", f"{cur_rain:.1f} mm")
            col_f2.metric("Suhu Rata-Rata Cirebon", f"{cur_temp:.1f} °C")
            col_f3.metric("Status Makan Gratis (MBG)", cur_mbg)
            
            news_label = "Netral"
            if cur_news < -0.1:
                news_label = "Negatif (Isu Mahal/Langka)"
            elif cur_news > 0.1:
                news_label = "Positif (Pasokan Aman)"
            col_f4.metric("Sentimen Berita Pangan", news_label)
            
            try:
                import importlib
                import update_harian
                importlib.reload(update_harian)
                headlines, _ = update_harian.fetch_news_headlines_and_sentiment()
            except Exception as e:
                st.error(f"Error loading headlines: {e}")
                headlines = []
                
            if headlines:
                st.write("---")
                st.write("**Berita Pangan Cirebon Terkini (Google News RSS):**")
                for hl in headlines[:5]:
                    score_indicator = "🔴 Negatif" if hl["score"] < 0 else "🟢 Positif" if hl["score"] > 0 else "⚪ Netral"
                    st.markdown(f"- **[{hl['title']}]({hl['link']})**")
                    st.markdown(f"  *Sentimen: {score_indicator} ({hl['score']:.1f}) | Rilis: {hl['pub_date']}*")

        # EVALUASI PASCA-DEPLOYMENT
        st.markdown("---")
        with st.expander("Evaluasi Akurasi Pasca-Deployment (Mulai 30 Juni 2026)"):
            st.write("Skoring akurasi riil dari ketiga model peramalan sejak deployment sistem pada tanggal **30 Juni 2026**.")
            
            post_deploy_actual = sub_df[sub_df["Tanggal"] >= '2026-06-30'].copy()
            post_deploy_train = sub_df[sub_df["Tanggal"] < '2026-06-30'].copy()
            features = FEATURES
            
            if len(post_deploy_actual) >= 1 and len(post_deploy_train) > 60:
                n_days = len(post_deploy_actual)
                
                # Hitung Hari_Dalam_Tahun dan Jarak_Ke_Hari_Raya
                post_deploy_train["Hari_Dalam_Tahun"] = post_deploy_train["Tanggal"].dt.dayofyear
                idx_pd = np.searchsorted(HOLIDAYS, post_deploy_train["Tanggal"])
                idx_pd = np.clip(idx_pd, 0, len(HOLIDAYS) - 1)
                next_hols_pd = pd.to_datetime([HOLIDAYS[i] for i in idx_pd])
                post_deploy_train["Jarak_Ke_Hari_Raya"] = (next_hols_pd - post_deploy_train["Tanggal"]).dt.days
                
                post_deploy_actual["Hari_Dalam_Tahun"] = post_deploy_actual["Tanggal"].dt.dayofyear
                idx_pd_a = np.searchsorted(HOLIDAYS, post_deploy_actual["Tanggal"])
                idx_pd_a = np.clip(idx_pd_a, 0, len(HOLIDAYS) - 1)
                next_hols_pd_a = pd.to_datetime([HOLIDAYS[i] for i in idx_pd_a])
                post_deploy_actual["Jarak_Ke_Hari_Raya"] = (next_hols_pd_a - post_deploy_actual["Tanggal"]).dt.days
                
                post_deploy_train["Lag_1"] = post_deploy_train["Harga_Bersih"].shift(1)
                post_deploy_train["Lag_2"] = post_deploy_train["Harga_Bersih"].shift(2)
                post_deploy_train["Lag_7"] = post_deploy_train["Harga_Bersih"].shift(7)
                post_deploy_train_clean = post_deploy_train.dropna(subset=["Lag_1", "Lag_2", "Lag_7"])
                
                # Predict dengan Direct Forecasting untuk RF & XGB
                lag1_pd = post_deploy_train["Harga_Bersih"].iloc[-1]
                lag2_pd = post_deploy_train["Harga_Bersih"].iloc[-2] if len(post_deploy_train) > 1 else lag1_pd
                lag7_pd = post_deploy_train["Harga_Bersih"].iloc[-7] if len(post_deploy_train) > 6 else lag1_pd
                
                preds_rf_pd = [0.0] * len(post_deploy_actual)
                preds_xgb_pd = [0.0] * len(post_deploy_actual)
                
                for k in range(len(post_deploy_actual)):
                    post_deploy_train_clean[f"Target_H_{k+1}"] = post_deploy_train_clean["Harga_Bersih"].shift(-(k+1))
                    pd_train_h = post_deploy_train_clean.dropna(subset=[f"Target_H_{k+1}"])
                    
                    if len(pd_train_h) > 30:
                        rf_h_pd = RandomForestRegressor(n_estimators=30, max_depth=6, random_state=42, n_jobs=-1)
                        rf_h_pd.fit(pd_train_h[features].fillna(0), pd_train_h[f"Target_H_{k+1}"])
                        
                        xgb_h_pd = xgb.XGBRegressor(n_estimators=30, max_depth=5, learning_rate=0.1, random_state=42, n_jobs=-1)
                        xgb_h_pd.fit(pd_train_h[features].fillna(0), pd_train_h[f"Target_H_{k+1}"])
                        
                        row_act = post_deploy_actual.iloc[k]
                        row_feat = pd.DataFrame([{
                            "Curah_Hujan": row_act["Curah_Hujan"],
                            "Suhu_Rata": row_act["Suhu_Rata"],
                            "MBG_Aktif": row_act["MBG_Aktif"],
                            "Sentimen_Berita": row_act["Sentimen_Berita"],
                            "Lag_1": lag1_pd,
                            "Lag_2": lag2_pd,
                            "Lag_7": lag7_pd,
                            "Hari_Dalam_Tahun": row_act["Hari_Dalam_Tahun"],
                            "Jarak_Ke_Hari_Raya": row_act["Jarak_Ke_Hari_Raya"]
                        }])
                        preds_rf_pd[k] = rf_h_pd.predict(row_feat)[0]
                        preds_xgb_pd[k] = xgb_h_pd.predict(row_feat)[0]
                    else:
                        preds_rf_pd[k] = lag1_pd
                        preds_xgb_pd[k] = lag1_pd
                
                # Predict dengan LSTM untuk post-deploy
                prices_pd = post_deploy_train["Harga_Bersih"].values
                p_min_pd, p_max_pd = prices_pd.min(), prices_pd.max()
                scaled_pd = (prices_pd - p_min_pd) / (p_max_pd - p_min_pd) if p_max_pd > p_min_pd else prices_pd * 0.0
                
                window_size = 30
                X_lstm_pd, y_lstm_pd = [], []
                for i in range(len(scaled_pd) - window_size - len(post_deploy_actual) + 1):
                    X_lstm_pd.append(scaled_pd[i : i + window_size])
                    y_lstm_pd.append(scaled_pd[i + window_size : i + window_size + len(post_deploy_actual)])
                    
                if len(X_lstm_pd) > 10:
                    X_lstm_pd_t = torch.FloatTensor(np.array(X_lstm_pd)[:, :, np.newaxis])
                    y_lstm_pd_t = torch.FloatTensor(np.array(y_lstm_pd))
                    
                    lstm_pd = SimpleLSTM()
                    # Menyesuaikan input/output size LSTM untuk post-deploy
                    lstm_pd.fc = nn.Linear(16, len(post_deploy_actual))
                    
                    criterion = nn.MSELoss()
                    optimizer_pd = optim.Adam(lstm_pd.parameters(), lr=0.01)
                    
                    lstm_pd.train()
                    for epoch in range(60):
                        optimizer_pd.zero_grad()
                        loss = criterion(lstm_pd(X_lstm_pd_t), y_lstm_pd_t)
                        loss.backward()
                        optimizer_pd.step()
                        
                    lstm_pd.eval()
                    with torch.no_grad():
                        last_w_pd = scaled_pd[-window_size:]
                        input_pd_t = torch.FloatTensor(last_w_pd).view(1, window_size, 1)
                        pred_s_pd = lstm_pd(input_pd_t).numpy()[0]
                        preds_lstm_pd = pred_s_pd * (p_max_pd - p_min_pd) + p_min_pd
                else:
                    preds_lstm_pd = [lag1_pd] * len(post_deploy_actual)
                
                # Format Dataframes
                pd_pred_rf = post_deploy_actual.copy()
                pd_pred_rf["yhat"] = np.round(np.array(preds_rf_pd[:len(post_deploy_actual)]) / 500.0) * 500.0
                pd_mape_rf = np.mean(np.abs((pd_pred_rf["Harga_Bersih"] - pd_pred_rf["yhat"]) / pd_pred_rf["Harga_Bersih"])) * 100
                pd_mae_rf = np.mean(np.abs(pd_pred_rf["Harga_Bersih"] - pd_pred_rf["yhat"]))

                pd_pred_xgb = post_deploy_actual.copy()
                pd_pred_xgb["yhat"] = np.round(np.array(preds_xgb_pd[:len(post_deploy_actual)]) / 500.0) * 500.0
                pd_mape_xgb = np.mean(np.abs((pd_pred_xgb["Harga_Bersih"] - pd_pred_xgb["yhat"]) / pd_pred_xgb["Harga_Bersih"])) * 100
                pd_mae_xgb = np.mean(np.abs(pd_pred_xgb["Harga_Bersih"] - pd_pred_xgb["yhat"]))

                pd_pred_lstm = post_deploy_actual.copy()
                pd_pred_lstm["yhat"] = np.round(np.array(preds_lstm_pd[:len(post_deploy_actual)]) / 500.0) * 500.0
                pd_mape_lstm = np.mean(np.abs((pd_pred_lstm["Harga_Bersih"] - pd_pred_lstm["yhat"]) / pd_pred_lstm["Harga_Bersih"])) * 100
                pd_mae_lstm = np.mean(np.abs(pd_pred_lstm["Harga_Bersih"] - pd_pred_lstm["yhat"]))

                st.write(f"Berikut hasil perbandingan performa **3 model** untuk **{n_days} hari** data riil sejak deployment:")
                
                # Tampilkan tabel metrik pasca-deployment
                pd_metrics_summary = [
                    {"Model": "Random Forest", "Rata-rata Persentase Error Riil (MAPE)": f"{pd_mape_rf:.2f}%", "Rata-rata Selisih Harga Riil (MAE)": f"Rp {int(pd_mae_rf):,}"},
                    {"Model": "XGBoost", "Rata-rata Persentase Error Riil (MAPE)": f"{pd_mape_xgb:.2f}%", "Rata-rata Selisih Harga Riil (MAE)": f"Rp {int(pd_mae_xgb):,}"},
                    {"Model": "LSTM", "Rata-rata Persentase Error Riil (MAPE)": f"{pd_mape_lstm:.2f}%", "Rata-rata Selisih Harga Riil (MAE)": f"Rp {int(pd_mae_lstm):,}"}
                ]
                st.table(pd.DataFrame(pd_metrics_summary))
                
                # Plot komparasi pasca-deployment
                fig_pd = go.Figure()
                fig_pd.add_trace(go.Scatter(x=post_deploy_actual["Tanggal"], y=post_deploy_actual["Harga_Bersih"], mode='lines+markers', name='Harga Aktual (Lapangan)', line=dict(color='#00BFFF', width=3)))
                
                colors_fut = {"Random Forest": "red", "XGBoost": "green", "LSTM": "orange"}
                fig_pd.add_trace(go.Scatter(x=pd_pred_rf["Tanggal"], y=pd_pred_rf["yhat"], mode='lines+markers', name=f'Random Forest (MAPE: {pd_mape_rf:.2f}%)', line=dict(dash='dash', color=colors_fut["Random Forest"])))
                fig_pd.add_trace(go.Scatter(x=pd_pred_xgb["Tanggal"], y=pd_pred_xgb["yhat"], mode='lines+markers', name=f'XGBoost (MAPE: {pd_mape_xgb:.2f}%)', line=dict(dash='dash', color=colors_fut["XGBoost"])))
                fig_pd.add_trace(go.Scatter(x=pd_pred_lstm["Tanggal"], y=pd_pred_lstm["yhat"], mode='lines+markers', name=f'LSTM (MAPE: {pd_mape_lstm:.2f}%)', line=dict(dash='dash', color=colors_fut["LSTM"])))
                
                fig_pd.update_layout(
                    title="Komparasi Prediksi 3 Model vs Aktual Pasca-Deployment (Mulai 30 Juni 2026)",
                    xaxis_title="Tanggal",
                    yaxis_title="Harga (Rp)",
                    yaxis=dict(rangemode="tozero"),
                    template="plotly_white"
                )
                st.plotly_chart(fig_pd, use_container_width=True)
            else:
                st.info("Belum ada cukup data aktual pasca-deployment (sejak 30 Juni 2026) untuk diuji akurasinya. Harap tunggu scraping cron harian berjalan.")
    else:
        st.warning("Data historis terlalu sedikit untuk melakukan peramalan (minimal dibutuhkan 100 hari data).")

# ==========================================
# HALAMAN 4: DISPARITAS & KORELASI PASAR
# ==========================================
elif page == "Disparitas & Korelasi Pasar":
    st.title("Analisis Disparitas & Korelasi Harga Antar Pasar")
    st.info(status_data_text)
    st.write("Analisis korelasi pergerakan harga pangan bersih untuk melacak keterkaitan supply chain antarpasar.")

    selected_comm_corr = st.selectbox("Pilih Komoditas:", df["Komoditas"].unique(), key="corr_comm_select")
    
    # Filter 180 hari terakhir untuk melihat korelasi terbaru
    df_comm = df[df["Komoditas"] == selected_comm_corr].copy()
    max_date = df_comm["Tanggal"].max()
    df_recent = df_comm[df_comm["Tanggal"] >= max_date - pd.Timedelta(days=180)].copy()
    
    if not df_recent.empty:
        # Pivot untuk mendapatkan kolom per pasar
        df_pivot = df_recent.pivot(index="Tanggal", columns="Pasar", values="Harga_Bersih")
        
        # Hitung korelasi
        corr_matrix = df_pivot.corr().fillna(0.0)
        
        st.subheader(f"Matriks Korelasi Pergerakan Harga {selected_comm_corr} Antar Pasar")
        
        # Heatmap menggunakan plotly go dengan skala warna terang (dari biru/low ke merah/high, yaitu 'RdBu_r' yang merupakan standar korelasi)
        fig_heat = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=list(corr_matrix.columns),
            y=list(corr_matrix.index),
            colorscale='RdBu_r',
            zmin=0.0, zmax=1.0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            hoverongaps=False,
            colorbar=dict(title="Korelasi")
        ))
        fig_heat.update_layout(
            xaxis_title="Pasar",
            yaxis_title="Pasar"
        )
        st.plotly_chart(fig_heat, use_container_width=True)
        
        # Tampilkan Tingkat Disparitas Harga Bersih Saat Ini
        st.subheader("Tingkat Disparitas Harga Bersih Saat Ini")
        
        # Mengambil harga bersih pada tanggal terakhir tercatat untuk komoditas tersebut per pasar
        latest_prices = df_comm.sort_values("Tanggal").groupby("Pasar").last()["Harga_Bersih"]
        
        disparity_df = pd.DataFrame({
            "Harga Terakhir Bersih (Rp)": latest_prices.apply(lambda x: f"Rp {int(x):,}")
        })
        st.dataframe(disparity_df)
        
        max_market = latest_prices.idxmax()
        min_market = latest_prices.idxmin()
        selisih = latest_prices.max() - latest_prices.min()
        
        st.info(
            f"Pada tanggal terakhir data tercatat, **{selected_comm_corr}** memiliki harga bersih tertinggi di **{max_market}** "
            f"dan terendah di **{min_market}**, dengan selisih disparitas bersih sebesar **Rp {int(selisih):,}**."
        )
    else:
        st.warning("Data tidak cukup untuk melakukan analisis disparitas.")

# ==========================================
# ==========================================
# HALAMAN 5: KUALITAS DATA
# ==========================================
elif page == "Kualitas Data":
    st.title("Kualitas Data dan Audit Pipeline")
    st.info(status_data_text)
    st.write("Ringkasan kualitas data hasil scraping, pembersihan anomali, resampling harian, dan imputasi harga.")

    raw_df = pd.read_csv(DATA_FILE)
    raw_df["Tanggal"] = pd.to_datetime(raw_df["Tanggal"])

    total_raw = len(raw_df)
    total_clean = len(df)
    total_imputed = int(df["Is_Imputed"].sum())
    rasio_imputasi = (total_imputed / total_clean * 100) if total_clean else 0.0
    total_resampled = max(total_clean - total_raw, 0)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Baris Mentah", f"{total_raw:,}")
    col2.metric("Baris Setelah Resampling", f"{total_clean:,}")
    col3.metric("Baris Tambahan Ffill", f"{total_resampled:,}")
    col4.metric("Rasio Imputasi", f"{rasio_imputasi:.2f}%")

    st.subheader("Kualitas Data per Pasar")
    
    # Hitung kelengkapan data mentah secara teoritis per pasar
    stats_list = []
    for market in df["Pasar"].unique():
        df_m = df[df["Pasar"] == market]
        # Cari data mentah aktual yang dilaporkan untuk pasar ini
        raw_m = df_raw_actual[df_raw_actual["Pasar"] == market]
        
        # Kelengkapan mentah = (jumlah baris mentah unik di komoditas & tanggal / total baris bersih) * 100
        # Di mana total baris bersih adalah total data setelah resampling ffill
        # Untuk menyederhanakan dan mencocokkan nilai persis di UI:
        # Kelengkapan_Mentah_% = (Baris_Mentah / Jumlah_Baris_Clean) * 100
        baris_clean = len(df_m)
        
        # Hitung jumlah baris mentah unik
        # Dapatkan irisan dari df_raw_actual yang ada di df_m untuk pasar ini
        # Karena kita ingin membandingkan data mentah yang ada di master_historis_pangan_cirebon.csv
        # Let's count actual raw rows for this market
        baris_mentah = len(raw_df[raw_df["Pasar"] == market])
        
        imputed_m = int(df_m["Is_Imputed"].sum())
        rasio_imp = (imputed_m / baris_clean * 100) if baris_clean else 0.0
        kelengkapan = (baris_mentah / baris_clean * 100) if baris_clean else 0.0
        
        # Mendapatkan Tanggal Awal dan Akhir
        tgl_awal = df_m["Tanggal"].min()
        tgl_akhir = df_m["Tanggal"].max()
        
        stats_list.append({
            "Pasar": market,
            "Baris_Mentah": baris_mentah,
            "Jumlah_Baris_Clean": baris_clean,
            "Kelengkapan_Mentah_%": f"{kelengkapan:.2f}%",
            "Tanggal_Awal": tgl_awal.strftime("%Y-%m-%d") if not pd.isnull(tgl_awal) else "N/A",
            "Tanggal_Akhir": tgl_akhir.strftime("%Y-%m-%d") if not pd.isnull(tgl_akhir) else "N/A",
            "Jumlah_Komoditas": df_m["Komoditas"].nunique(),
            "Jumlah_Imputasi": imputed_m,
            "Rasio_Imputasi_%": f"{rasio_imp:.2f}%",
            "val_imputation": imputed_m
        })
        
    df_stats = pd.DataFrame(stats_list)
    st.dataframe(df_stats.drop(columns=["val_imputation"]))
    
    st.subheader("Jumlah Imputasi Harga per Pasar")
    fig_imp = go.Figure(data=go.Bar(
        x=df_stats["Pasar"],
        y=df_stats["val_imputation"],
        marker=dict(color='#1f77b4') # Warna biru sesuai screenshot lama
    ))
    fig_imp.update_layout(
        xaxis_title="Pasar",
        yaxis_title="Jumlah Imputasi"
    )
    st.plotly_chart(fig_imp, use_container_width=True)
    
    st.subheader("Komoditas dengan Anomali Terbanyak")
    # Kelompokkan komoditas dengan jumlah anomali terbanyak
    anomaly_comm = []
    for comm in df["Komoditas"].unique():
        df_c = df[df["Komoditas"] == comm]
        total_imp = int(df_c["Is_Imputed"].sum())
        if total_imp > 0:
            # Cari pasar terdampak (jumlah pasar unik yang memiliki imputasi)
            markets_affected = df_c[df_c["Is_Imputed"] == True]["Pasar"].nunique()
            # Tanggal terakhir terdeteksi
            last_date = df_c[df_c["Is_Imputed"] == True]["Tanggal"].max()
            
            anomaly_comm.append({
                "Komoditas": comm,
                "Jumlah_Imputasi": total_imp,
                "Pasar_Terdampak": markets_affected,
                "Tanggal_Terakhir": last_date.strftime("%Y-%m-%d") if not pd.isnull(last_date) else "N/A"
            })
    if anomaly_comm:
        df_anomaly = pd.DataFrame(anomaly_comm).sort_values("Jumlah_Imputasi", ascending=False).head(10)
        st.dataframe(df_anomaly)
    else:
        st.info("Tidak ada data anomali terdeteksi.")
        
    st.subheader("Catatan Audit Trail Imputasi Terbaru")
    st.write("Daftar data harga pasar terbaru yang diidentifikasi sebagai anomali (outlier) dan diperbaiki secara otomatis oleh sistem:")
    
    # Hanya ambil baris data terimputasi yang memiliki Harga_Original (bukan hasil resampling harian kosong)
    # ATAU jika Harga_Bersih bernilai valid (bukan NaN/None)
    df_imputed = df[(df["Is_Imputed"] == True) & (df["Harga_Bersih"].notnull())].copy()
    if not df_imputed.empty:
        df_imputed_sorted = df_imputed.sort_values("Tanggal", ascending=False).head(50)
        
        # Merge dengan data mentah aktual untuk mengambil Harga_Original fisik jika dilaporkan
        df_audit = pd.merge(
            df_imputed_sorted,
            df_raw_actual,
            on=["Tanggal", "Pasar", "Komoditas"],
            how="left"
        )
        
        # Kolom 'Harga' berasal dari df_raw_actual (data mentah asli).
        # Kolom 'Harga_Original' berasal dari df (data bersih, yang mungkin sudah di-resample).
        # Jika data mentah asli ada, tampilkan itu. Jika kosong (karena resampling), gunakan nilai baseline Harga_Original.
        # Karena df_raw_actual dan df sama-sama memiliki kolom 'Harga' sebelum digabungkan (yaitu df_raw_actual['Harga'] dan df['Harga']),
        # hasil merge mengganti namanya menjadi 'Harga_y' (dari df_raw_actual) dan 'Harga_x' (dari df).
        harga_mentah_key = "Harga_y" if "Harga_y" in df_audit.columns else "Harga"
        df_audit["Harga_Original_Display"] = df_audit[harga_mentah_key].fillna(df_audit["Harga_Original"])
        
        # Format harga
        df_audit_display = pd.DataFrame({
            "Tanggal": df_audit["Tanggal"].dt.strftime("%Y-%m-%d"),
            "Pasar": df_audit["Pasar"],
            "Komoditas": df_audit["Komoditas"],
            "Harga_Original": df_audit["Harga_Original_Display"],
            "Harga_Bersih": df_audit["Harga_Bersih"]
        })
        
        # Drop rows where Harga_Original is NaN/None agar tidak merusak log visualisasi
        df_audit_display = df_audit_display.dropna(subset=["Harga_Original", "Harga_Bersih"])
        
        st.dataframe(df_audit_display)
    else:
        st.success("Tidak ada tindakan imputasi terdeteksi.")

    # Bagian Baru: Kelengkapan Update Tanggal Terakhir per Pasar
    st.subheader("Kelengkapan Update Tanggal Terakhir per Pasar")
    max_date_all = df["Tanggal"].max()
    stats_dates = []
    for market in df["Pasar"].unique():
        df_m = df[df["Pasar"] == market]
        last_date = df_m["Tanggal"].max()
        rows_last_date = len(df_m[df_m["Tanggal"] == last_date])
        comm_total = df_m["Komoditas"].nunique()
        status = "Terkini" if last_date == max_date_all else "Terlambat"
        stats_dates.append({
            "Pasar": market,
            "Tanggal_Terakhir_Clean": last_date.strftime("%Y-%m-%d") if not pd.isnull(last_date) else "N/A",
            "Baris_Tanggal_Terakhir": rows_last_date,
            "Komoditas_Total": comm_total,
            "Status": status
        })
    df_dates = pd.DataFrame(stats_dates)
    st.dataframe(df_dates)
    
    # Tampilkan banner kesamaan update tanggal jika seluruh pasar sudah sinkron
    if all(d["Status"] == "Terkini" for d in stats_dates):
        st.success("Seluruh pasar memiliki tanggal pembaruan terbaru yang sama (setelah ffill).")
    else:
        st.warning("Terdapat perbedaan tanggal pembaruan terakhir antarpasar.")


elif page == "Tanya Jawab AI (Chatbot)":
    st.title("💬 Tanya Jawab AI (Chatbot Data Pangan)")
    st.markdown(f"**Oleh Badar Rahman** (GitHub: [@DarRahman](https://github.com/DarRahman))")
    st.write("Analisis data pangan Kabupaten Cirebon menggunakan model kecerdasan buatan berbasis natural language query.")

    # Deteksi API Keys
    env_openai_key = os.environ.get("OPENAI_API_KEY", "")
    if not env_openai_key:
        try:
            with open(r"C:\Users\Pongo\.hermes\.env") as f:
                for line in f:
                    if "OPENAI_API_KEY" in line:
                        env_openai_key = line.split("=")[1].strip().strip('"').strip("'")
        except:
            pass
            
    if "sk_9router" in env_openai_key:
        env_openai_key = ""

    env_groq_key = os.environ.get("GROQ_API_KEY", "")
    env_gemini_key = os.environ.get("GEMINI_API_KEY", "")

    # Sidebar LLM Config
    st.sidebar.subheader("Konfigurasi Chatbot")
    llm_provider = st.sidebar.selectbox(
        "Pilih Provider LLM:", 
        ["Gemini (Google)", "Groq (Llama 3.3)", "OpenAI (GPT-4o)"]
    )

    # Pilih key default
    default_key = ""
    if llm_provider == "OpenAI (GPT-4o)":
        default_key = env_openai_key
    elif llm_provider == "Groq (Llama 3.3)":
        default_key = env_groq_key
    elif llm_provider == "Gemini (Google)":
        default_key = env_gemini_key

    api_key_input = st.sidebar.text_input(
        "Masukkan API Key:", 
        value=default_key, 
        type="password", 
        help="Silakan masukkan API Key Anda jika tidak terdeteksi otomatis dari environment."
    )

    sys_prompt = """
Anda adalah asisten AI yang menganalisis dataset pangan Kabupaten Cirebon.
Dataset historis disimpan dalam DataFrame pandas bernama `df`.
Kolom yang tersedia:
- 'Tanggal': tanggal data (format datetime64)
- 'Pasar': nama pasar
- 'Komoditas': nama komoditas
- 'Harga_Bersih': harga bersih dalam rupiah (integer kelipatan 500)
- 'Is_Imputed': boolean penanda data koreksi anomali

Batas Akhir Data Historis:
- Data historis di `df` hanya tersedia sampai tanggal 2 Juli 2026.
- PENTING: Untuk pertanyaan mengenai perkiraan harga, ramalan, atau nilai pada tanggal setelah 2 Juli 2026 (seperti 3 Juli, 4 Juli, dst.), Anda harus menggunakan fungsi peramalan `get_forecast` di bawah. Jangan memfilter df langsung karena hasilnya akan kosong (NaN).

Anda memiliki akses ke fungsi peramalan (forecasting) 14 hari ke depan:
- `get_forecast(komoditas, pasar)`: melakukan peramalan Random Forest 14 hari ke depan. Mengembalikan tuple (future_pred_df, mape, mae, test_df, pred_val_df).
  - future_pred_df adalah DataFrame dengan kolom ['Tanggal', 'Harga_Prediksi'].
  - champion_name adalah nama model terbaik (string).
  - best_mape adalah nilai MAPE terkecil (float).

Tugas Anda:
1. Jika pertanyaan membutuhkan analisis data kuantitatif dari `df` ATAU membutuhkan peramalan/prediksi harga masa depan setelah 2 Juli 2026, kembalikan HANYA satu baris kode Python yang valid dan dapat langsung dijalankan oleh `eval()` untuk mendapatkan jawaban tersebut.
Contoh:
- Tanya: "Berapa rata-rata harga Bawang Bombay di Pasar Sumber?"
  Kembalikan: [PANDAS] df[(df['Pasar']=='Pasar Sumber') & (df['Komoditas']=='Bawang Bombay')]['Harga_Bersih'].mean()
- Tanya: "Pasar mana yang paling murah untuk komoditas Daging Ayam Broiler pada 1 Juli 2026?"
  Kembalikan: [PANDAS] df[(df['Komoditas']=='Daging Ayam Broiler') & (df['Tanggal']=='2026-07-01')].sort_values('Harga_Bersih').iloc[0]['Pasar']
- Tanya: "Bagaimana peramalan harga Beras Medium di Pasar Sumber 14 hari ke depan?"
  Kembalikan: [PANDAS] get_forecast('Beras Medium', 'Pasar Sumber')[0]
- Tanya: "harga daging ayam broiler tanggal 4 july kira2 berapa? untuk semua pasar"
  Kembalikan: [PANDAS] get_forecast('Daging Ayam Broiler', 'Semua Pasar (Gabungan)')[0].loc[get_forecast('Daging Ayam Broiler', 'Semua Pasar (Gabungan)')[0]['Tanggal'] == '2026-07-04', 'Harga_Prediksi'].values[0]
- Tanya: "Apa model forecasting terbaik untuk Bawang Merah di Pasar Pasalaran?"
  Kembalikan: [PANDAS] get_forecast('Bawang Merah', 'Pasar Pasalaran')[1]

2. Jika pertanyaan tidak dapat dijawab dengan satu baris kode pandas/get_forecast atau merupakan obrolan biasa (seperti menyapa atau menjelaskan teori data science), kembalikan penjelasan teks biasa.
Contoh:
- Tanya: "Apa itu RAG?"
  Kembalikan: [TEXT] RAG adalah metode ujian buka buku untuk AI...

Format Output wajib dimulai dengan [PANDAS] diikuti baris kode tunggal, atau [TEXT] diikuti penjelasan teks biasa. Jangan menyertakan penjelasan tambahan jika menggunakan format [PANDAS].
"""

    def call_llm(provider, api_key, prompt):
        if provider == "Gemini (Google)":
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
            headers = {"Content-Type": "application/json"}
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.0}
            }
            r = requests.post(url, json=payload, headers=headers, timeout=15)
            if r.status_code == 200:
                res = r.json()
                return res['candidates'][0]['content']['parts'][0]['text']
            else:
                raise Exception(f"Gemini API Error {r.status_code}: {r.text}")
                
        elif provider == "Groq (Llama 3.3)":
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0
            }
            r = requests.post(url, json=payload, headers=headers, timeout=15)
            if r.status_code == 200:
                res = r.json()
                return res['choices'][0]['message']['content']
            else:
                raise Exception(f"Groq API Error {r.status_code}: {r.text}")
                
        elif provider == "OpenAI (GPT-4o)":
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0
            }
            r = requests.post(url, json=payload, headers=headers, timeout=15)
            if r.status_code == 200:
                res = r.json()
                return res['choices'][0]['message']['content']
            else:
                raise Exception(f"OpenAI API Error {r.status_code}: {r.text}")

    if not api_key_input:
        st.warning("Silakan masukkan API Key di sidebar terlebih dahulu.")
    else:
        st.info("API Key terdeteksi. AI siap menjawab pertanyaan Anda.")
        user_query = st.text_input("Ketik pertanyaan Anda tentang data pangan Cirebon:")
        
        if st.button("Kirim"):
            if user_query:
                with st.spinner("AI sedang memproses pertanyaan..."):
                    try:
                        markets_str = ", ".join(list(df["Pasar"].unique()))
                        commodities_str = ", ".join(list(df["Komoditas"].unique()[:10]))
                        
                        prompt_full = f"{sys_prompt}\n\nKonteks:\n- Pasar yang tersedia: {markets_str}\n- Contoh komoditas: {commodities_str}\n\nPertanyaan User: {user_query}"
                        
                        raw_response = call_llm(llm_provider, api_key_input, prompt_full)
                        raw_response = raw_response.strip()
                        
                        # Parsing response
                        if raw_response.startswith("[PANDAS]"):
                            code_to_eval = raw_response.replace("[PANDAS]", "").strip()
                            # Eksekusi kode
                            result = eval(code_to_eval, {"df": df, "pd": pd, "np": np, "get_forecast": lambda c, m: get_forecast_precalculated(c, m, df)})
                            
                            st.success("Query Data Sukses!")
                            st.write("**Kode Pandas yang Dijalankan AI:**")
                            st.code(code_to_eval, language="python")
                            
                            st.write("**Hasil:**")
                            if isinstance(result, (pd.DataFrame, pd.Series)):
                                st.dataframe(result)
                            elif isinstance(result, (int, float, np.integer, np.floating)):
                                st.metric("Nilai Terhitung", f"Rp {result:,.0f}" if result > 1000 else f"{result:.2f}")
                            else:
                                st.write(result)
                                
                        elif raw_response.startswith("[TEXT]"):
                            st.info(raw_response.replace("[TEXT]", "").strip())
                        else:
                            st.write(raw_response)
                            
                    except Exception as e:
                        st.error(f"Gagal memproses pertanyaan: {e}")
