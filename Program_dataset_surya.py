import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ==========================
# MEMBACA DATASET
# ==========================
df = pd.read_csv("dataset_prediksi_bawang_sulteng.csv")

# ==========================
# MENGUBAH DATA KATEGORI
# ==========================
encoder = LabelEncoder()
df["Kecamatan"] = encoder.fit_transform(df["Kecamatan"])

# ==========================
# MENENTUKAN FITUR DAN TARGET
# ==========================
X = df.drop("Produksi_Bawang_Ton", axis=1)
y = df["Produksi_Bawang_Ton"]

# ==========================
# MEMBAGI DATA TRAINING DAN TESTING
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# ==========================
# MEMBUAT MODEL RANDOM FOREST
# ==========================
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Melatih model
model.fit(X_train, y_train)

# Prediksi
y_pred = model.predict(X_test)

# Evaluasi model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error =", round(mae,2))
print("R2 Score =", round(r2,4))

# ==========================
# CONTOH PREDIKSI DATA BARU
# ==========================
kecamatan_baru = encoder.transform(["Dolo"])[0]

data_baru = [[
    2026,            # Tahun
    kecamatan_baru,  # Kecamatan
    2200,            # Curah hujan
    27.5,            # Suhu
    75,              # Kelembapan
    145,             # Luas panen
    850,             # Penggunaan pupuk
    10,              # Serangan hama
    80               # Jumlah petani
]]

hasil = model.predict(data_baru)

print("\nPrediksi produksi bawang tahun 2026:")
print(round(hasil[0],2), "ton")