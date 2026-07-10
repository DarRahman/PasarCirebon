# Pemantauan & Peramalan Harga Pangan Kabupaten Cirebon

Proyek ini menyajikan sistem pemantauan, audit kualitas, dan peramalan harga bahan pangan pokok di Kabupaten Cirebon berbasis Data Science. Dirancang sebagai dashboard siap saji untuk membantu masyarakat umum memahami dinamika harga pasar dan membantu dinas terkait dalam mendeteksi anomali pelaporan harga lapangan.

---

## Teknologi & Pustaka (Tech Stack)

Sistem dibangun menggunakan ekosistem Python modern dengan pustaka-pustaka standar industri berikut:
* **Analisis & Manipulasi Data**: `pandas` (ETL, pivoting, data wrangling) dan `numpy` (operasi numerik & kalkulasi vektor).
* **Pemodelan Machine Learning**: `xgboost` (XGBoost Regressor untuk regresi multivariat runtun waktu cepat & presisi) dan `scikit-learn` (Linear Regression untuk trend detrending).
* **Visualisasi & Dashboard**: `streamlit` (kerangka antarmuka aplikasi web) dan `plotly` (grafik interaktif runtun waktu & audit trail).
* **Scraping & Request API**: `requests` (penarik data cuaca Open-Meteo & API berita) dan `importlib` (manajemen reloading modul harian).

---

## Komponen & Arsitektur Sistem

Sistem memisahkan proses pengolahan data (ETL/Modeling) dengan antarmuka pengguna untuk menjaga efisiensi dan stabilitas:

```
[Sumber Data Web]
       │ (Kepokmas Cirebon, Open-Meteo, Google News)
       ▼
 ┌───────────┐      ┌─────────────┐      ┌───────────┐
 │   ETL &   │ ───> │  Pembersih  │ ───> │  XGBoost  │
 │ Scraping  │      │ Anomali/IQR │      │ Retraining│
 └───────────┘      └─────────────┘      └───────────┘
                                               │
                                               ▼
                                      ┌────────────────┐
                                      │ Precalculated  │
                                      │   Files/CSV    │
                                      └────────────────┘
                                               │
                                               ▼
                                      ┌────────────────┐
                                      │   Streamlit    │
                                      │ Dashboard App  │
                                      └────────────────┘
```

1. **Pipeline ETL & Automated Scraping (`update_harian.py`)**
   * Berjalan secara terjadwal otomatis (Cron Job / Windows Task Scheduler) setiap hari pada pukul **13:00 WIB**.
   * Mengunduh data harian dari portal resmi Kemendag Kepokmas Kabupaten Cirebon secara terprogram.
   * Melakukan pembersihan data otomatis, scraping berita & cuaca terbaru, retraining model, dan memperbarui database.

2. **Dashboard Interaktif (`app.py`)**
   * Aplikasi visual berbasis Streamlit untuk menyajikan data secara instan tanpa membebani performa browser dengan training ulang model di sisi pengguna.
   * Modul visual: Tren Harga Aktual, Deteksi Anomali (Audit Trail), Disparitas Pasar, Kualitas Data, dan Tanya Jawab AI (Chatbot RAG).

---

## Metodologi Pembersihan Data & Deteksi Anomali

Data mentah laporan petugas pasar sering kali memiliki anomali akibat kesalahan input (typo penulisan nominal). Pipa data otomatis menyaring anomali tersebut melalui:
* **Skala Koreksi Otomatis**: Memperbaiki harga typo yang kekurangan angka nol secara otomatis (misal: Rp 3.500 diinput Rp 350 akan dikoreksi ke batas wajar).
* **Deteksi Outlier Runtun Waktu**: Menggunakan metode statistik **Interquartile Range (IQR)** pada selisih perubahan harga harian untuk mendeteksi lonjakan tidak realistis yang disebabkan oleh kesalahan input.
* **Imputasi Gaps**: Nilai anomali dihapus dan diganti menggunakan interpolasi linier, sedangkan tanggal kosong diisi secara Forward Fill (`ffill`) untuk menjaga kontinuitas visualisasi runtun waktu.

---

## Model Peramalan: XGBoost 14 Hari (Direct Multi-Output)

Harga pangan harian pasar tradisional sangat fluktuatif dalam jangka pendek. Proyek ini mengimplementasikan pendekatan **Direct Multi-Output Forecasting** menggunakan **XGBoost Regressor**:

### Mengapa Direct Forecasting?
Alih-alih memprediksi secara rekursif (menggunakan prediksi hari esok untuk memprediksi lusa yang rentan terhadap akumulasi *error*), sistem ini melatih **14 model XGBoost yang berbeda secara independen** untuk setiap target hari prediksi ($t+1, t+2, \dots, t+14$):
* Model 1 memprediksi langsung $Y_{t+1}$
* Model 2 memprediksi langsung $Y_{t+2}$
* ...
* Model 14 memprediksi langsung $Y_{t+14}$

Setiap model menggunakan kombinasi fitur penentu harga berikut:
* **Siklus Cuaca & Hama (Open-Meteo API)**: Parameter `Curah_Hujan` dan `Suhu_Rata` harian untuk mengukur risiko gagal panen.
* **Sentimen Publik (Google News RSS)**: Pengukur sentimen berita pangan lokal (isu kelangkaan pasokan atau ketersediaan stok).
* **Kebijakan & Demand (Makan Bergizi Gratis)**: Status aktif/libur sekolah di Kabupaten Cirebon yang memengaruhi permintaan bahan pangan.
* **Siklus Musiman Tahunan**: Variabel `Hari_Dalam_Tahun` untuk mengenali musim panen raya atau paceklik berulang.
* **Hari Raya Keagamaan (HBKN)**: Variabel `Jarak_Ke_Hari_Raya` (Idulfitri, Natal, Tahun Baru) untuk mengantisipasi lonjakan permintaan ekstrem musiman.
* **Struktur Harga**: Lag historis (`Lag_1`, `Lag_2`, `Lag_7`) untuk menjaga stabilitas inersia harga pasar.

---

## Metrik Akurasi & Validasi

Sistem mengukur performa model dengan metrik statistik standar industri yang objektif dan bebas bias:
* **Rata-rata Persentase Error (MAPE)**: Rata-rata persentase penyimpangan garis prediksi dari harga aktual lapangan asli.
* **Rata-rata Selisih Harga (MAE)**: Rata-rata selisih nominal kesalahan tebakan model dalam Rupiah dari harga lapangan asli.

---

## Panduan Instalasi & Eksekusi Lokal

### 1. Dependensi
Pastikan Python 3.9+ telah terpasang. Instalasi pustaka pendukung dapat dilakukan cepat menggunakan:
```bash
pip install pandas numpy streamlit plotly xgboost torch requests beautifulsoup4 prophet
```

### 2. Langkah Pertama (Inisialisasi Data & Training)
Sebelum menjalankan dashboard untuk pertama kali, Anda **harus** menjalankan skrip ETL harian untuk membersihkan data mentah asli (`master_historis_pangan_cirebon.csv`) dan melatih model peramalan secara lokal:
```bash
python update_harian.py
```
Proses ini akan menghasilkan berkas data bersih (`master_historis_pangan_cirebon_clean.csv`) beserta detail metrik model (`validation_detail.csv`, `validation_metrics.csv`, dan `forecast_14_hari.csv`).

### 3. Menjalankan Dashboard
Setelah proses inisialisasi data selesai, jalankan perintah berikut pada direktori proyek Anda:
```bash
streamlit run app.py
```
Aplikasi dapat dibuka melalui peramban web pada alamat default: `http://localhost:8501`.

### 4. Struktur File Repositori
* `app.py`: Antarmuka visual Streamlit dan chatbot tanya jawab AI.
* `update_harian.py`: Skrip penarik data (ETL), pembersih anomali, dan pelatihan model terpisah (Direct Forecasting).
* `master_historis_pangan_cirebon.csv`: Database master harga pangan historis mentah.
* `README.md`: Dokumentasi petunjuk penggunaan.
* `master_historis_pangan_cirebon_clean.csv` *(Generated)*: Database bersih hasil pembersihan data dan penggabungan faktor eksternal.
* `forecast_14_hari.csv` *(Generated)*: Hasil peramalan 14 hari ke depan untuk semua pasar-komoditas.
* `validation_metrics.csv` & `validation_detail.csv` *(Generated)*: Hasil kalkulasi metrik evaluasi model.

---

## Kontributor

* **DarRahman** (Badar Rahman) - Lead Developer
* **Claude (AI)** - AI Assistant (Refactoring, Algoritma Direct Forecasting, dan Pembersihan Data)
