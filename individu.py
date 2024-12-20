import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Membaca dataset dari folder
dataset_path = input("E:\Laprak\SEMESTER 3\Praktikum Kecerdasan Buatan\ttd\dataset_pola_tidur.csv")
df = pd.read_csv(dataset_path)

# Pastikan dataset memiliki kolom 'durasi_tidur' dan 'kualitas_tidur'
if 'durasi_tidur' not in df.columns or 'kualitas_tidur' not in df.columns:
    raise ValueError("Dataset harus memiliki kolom 'durasi_tidur' dan 'kualitas_tidur'.")

# Fitur dan label
X = df[['durasi_tidur']]
y = df['kualitas_tidur']

# Membagi data menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standarisasi data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Membuat model KNN
knn = KNeighborsClassifier(n_neighbors=3)

# Melatih model
knn.fit(X_train, y_train)

# Prediksi
y_pred = knn.predict(X_test)

# Evaluasi model
accuracy = accuracy_score(y_test, y_pred)
print("Akurasi:", accuracy)
print("\nLaporan Klasifikasi:\n", classification_report(y_test, y_pred))

# Contoh prediksi baru
new_data = np.array([[7], [4], [9]])  # Data baru (durasi tidur dalam jam)
new_data_scaled = scaler.transform(new_data)
new_predictions = knn.predict(new_data_scaled)
print("Prediksi Kualitas Tidur untuk data baru:", new_predictions)
