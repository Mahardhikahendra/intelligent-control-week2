import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import cv2

# === Buat dataset sederhana langsung (tanpa Kaggle) ===
# Misalnya RGB untuk beberapa warna dasar
data = {
    'ColorName': ['Red', 'Green', 'Blue', 'Yellow', 'Cyan', 'Magenta', 'Black', 'White'],
    'R': [255, 0, 0, 255, 0, 255, 0, 255],
    'G': [0, 255, 0, 255, 255, 0, 0, 255],
    'B': [0, 0, 255, 0, 255, 255, 0, 255]
}
color_data = pd.DataFrame(data)

# Pisahkan fitur dan label
X = color_data[['R', 'G', 'B']].values
y = color_data['ColorName'].values

# Normalisasi data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset untuk training dan testing
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Inisialisasi model KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Prediksi
y_pred = knn.predict(X_test)

# Evaluasi akurasi
accuracy = accuracy_score(y_test, y_pred)
print(f'Akurasi model: {accuracy * 100:.2f}%')

# === Deteksi warna dari kamera ===
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

# Ambil pixel tengah gambar
    height, width, _ = frame.shape
    pixel_center = frame[height // 2, width // 2]

    # Normalisasi sebelum prediksi
    pixel_center_scaled = scaler.transform([pixel_center])

    # Prediksi warna
    color_pred = knn.predict(pixel_center_scaled)[0]

    # Tampilkan hasil
    cv2.putText(frame, f'Color: {color_pred}', (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.circle(frame, (width // 2, height // 2), 5, (0, 0, 0), -1)

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
