import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# ==========================================
# 1. LOAD DATASET
# ==========================================
df = pd.read_csv("dataset_prediksi_bawang_sulteng.csv")

# Pastikan tidak ada data target yang kosong
df = df.dropna(subset=["Produksi_Bawang_Ton"])

print("DATASET PREDIKSI BAWANG MERAH SULTENG")
print(f"Jumlah Data      : {len(df)}")
print(f"Jumlah Kecamatan : {df['Kecamatan'].nunique()}")
print()

# ==========================================
# 2. FITUR DAN TARGET
# ==========================================
FITUR = [
    "Curah_Hujan_mm",
    "Suhu_C",
    "Kelembapan_Persen",
    "Luas_Panen_Ha",
    "Penggunaan_Pupuk_Kg",
    "Serangan_Hama_Persen",
    "Jumlah_Petani",
    "Tahun"
]

TARGET = "Produksi_Bawang_Ton"

X = df[FITUR]
y = df[TARGET]

# ==========================================
# 3. SPLIT DATA & SCALING
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Scaling untuk SVR
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 4. INISIALISASI & PERBANDINGAN MODEL
# ==========================================
models = {
    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ),
    "SVR": SVR(
        kernel="rbf",
        C=100,
        gamma="scale"
    )
}

print("PERBANDINGAN MODEL")
print(f"{'Model':<20} {'MAE':>10} {'R2 Score':>10}")
print("-" * 45)

best_model = None
best_model_name = None
best_r2 = -999

for nama, model in models.items():

    if nama == "SVR":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"{nama:<20} {mae:>10.3f} {r2:>10.4f}")

    if r2 > best_r2:
        best_r2 = r2
        best_model = model
        best_model_name = nama

print()
print(f"Model Terbaik : {best_model_name}")
print(f"R2 Score      : {best_r2:.4f}")
print()

# ==========================================
# 5. TRAINING ULANG MODEL TERBAIK FULL DATA
# ==========================================
if best_model_name == "SVR":
    X_full = scaler.fit_transform(X)
    best_model = SVR(kernel="rbf", C=100, gamma="scale")
    best_model.fit(X_full, y)
else:
    best_model = RandomForestRegressor(n_estimators=200, random_state=42)
    best_model.fit(X, y)

# ==========================================
# 6. FUNGSI PREDIKSI MASA DEPAN
# ==========================================
def prediksi_kecamatan(nama_kecamatan, tahun_prediksi=[2026, 2027, 2028]):
    data_kec = df[df["Kecamatan"] == nama_kecamatan].sort_values("Tahun")

    if data_kec.empty:
        return None

    terakhir = data_kec.iloc[-1].copy()
    produksi_terakhir = terakhir["Produksi_Bawang_Ton"]
    hasil = []

    for tahun in tahun_prediksi:
        fitur_prediksi = {
            "Curah_Hujan_mm": terakhir["Curah_Hujan_mm"],
            "Suhu_C": terakhir["Suhu_C"],
            "Kelembapan_Persen": terakhir["Kelembapan_Persen"],
            "Luas_Panen_Ha": terakhir["Luas_Panen_Ha"] * 1.01,
            "Penggunaan_Pupuk_Kg": terakhir["Penggunaan_Pupuk_Kg"] * 1.02,
            "Serangan_Hama_Persen": terakhir["Serangan_Hama_Persen"] * 0.95,
            "Jumlah_Petani": terakhir["Jumlah_Petani"] * 1.01,
            "Tahun": tahun
        }

        X_pred = pd.DataFrame([fitur_prediksi])

        if best_model_name == "SVR":
            X_pred = scaler.transform(X_pred)

        prediksi = best_model.predict(X_pred)[0]

        batas_turun = produksi_terakhir * 0.8
        batas_naik = produksi_terakhir * 1.2

        prediksi = max(prediksi, batas_turun)
        prediksi = min(prediksi, batas_naik)

        hasil.append({
            "Tahun": tahun,
            "Prediksi Produksi (Ton)": round(prediksi, 2)
        })

        produksi_terakhir = prediksi
        terakhir["Luas_Panen_Ha"] *= 1.01
        terakhir["Penggunaan_Pupuk_Kg"] *= 1.02
        terakhir["Serangan_Hama_Persen"] *= 0.95
        terakhir["Jumlah_Petani"] *= 1.01

    return pd.DataFrame(hasil)

# ==========================================
# 7. EKSEKUSI PREDIKSI & SIMPAN HASIL
# ==========================================
print("PREDIKSI PRODUKSI BAWANG MERAH MASA DEPAN")

rows_output = []

for kecamatan in sorted(df["Kecamatan"].unique()):
    hasil = prediksi_kecamatan(kecamatan)

    if hasil is None:
        continue

    print()
    print(f"Kecamatan: {kecamatan}")

    for _, row in hasil.iterrows():
        print(f"{int(row['Tahun'])} : {row['Prediksi Produksi (Ton)']:.2f} Ton")

        rows_output.append({
            "Kecamatan": kecamatan,
            "Tahun": int(row["Tahun"]),
            "Prediksi Produksi (Ton)": row["Prediksi Produksi (Ton)"]
        })

# MENGGUNAKAN TO_CSV AGAR TIDAK ERROR OPENPYXL
df_output = pd.DataFrame(rows_output)
df_output.to_csv("hasil_prediksi_bawang_sulteng.csv", index=False)

print()
print("="*45)
print("File berhasil disimpan: hasil_prediksi_bawang_sulteng.csv")
print("="*45)

# ==========================================
# 8. PENGARUH FITUR (JIKA RANDOM FOREST)
# ==========================================
if best_model_name == "Random Forest":
    print()
    print("TINGKAT PENGARUH FITUR (FEATURE IMPORTANCES)")
    print("-" * 45)

    importance = sorted(
        zip(FITUR, best_model.feature_importances_),
        key=lambda x: x[1],
        reverse=True
    )

    for fitur, nilai in importance:
        print(f"{fitur:<25} {nilai:.4f}")
