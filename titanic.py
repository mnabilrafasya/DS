import pandas as pd

# Load Titanic dataset
file_path = 'C:/Users/USER/OneDrive/Documents/NetBeansProjects/Programs/Data Science/Titanic-Dataset.csv'  
titanic_data = pd.read_csv(file_path)

# Menampilkan 5 baris pertama dan info dataset
print(titanic_data.head())
print(titanic_data.info())

# Statistik deskriptif untuk data numerik
print(titanic_data.describe())


import matplotlib.pyplot as plt
import seaborn as sns

# Visualisasi distribusi umur
plt.figure(figsize=(8, 5))
sns.histplot(titanic_data['Age'], bins=20, kde=True, color='blue', alpha=0.7)
plt.title("Distribusi Usia Penumpang")
plt.xlabel("Usia")
plt.ylabel("Jumlah Penumpang")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Visualisasi jumlah penumpang yang selamat vs tidak selamat
plt.figure(figsize=(6, 4))
sns.countplot(x='Survived', hue=None, data=titanic_data, palette='Set2', legend=False)
plt.title("Distribusi Kelangsungan Hidup Penumpang")
plt.xticks([0, 1], ['Tidak Selamat', 'Selamat'])
plt.xlabel("Kelangsungan Hidup")
plt.ylabel("Jumlah Penumpang")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


#klasifikasi
from sklearn.model_selection import train_test_split

# Menghapus kolom yang tidak relevan dan menangani nilai kosong
data = titanic_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
data = data.dropna()  # Menghapus baris dengan nilai kosong

# One-hot encoding untuk variabel kategori
data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)

# Membagi dataset menjadi fitur dan target
X = data.drop('Survived', axis=1)
y = data['Survived']

# Membagi data menjadi data latih dan uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.tree import DecisionTreeClassifier

# Membuat dan melatih model
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, classification_report

# Prediksi data uji
y_pred = clf.predict(X_test)

# Menghitung akurasi dan laporan klasifikasi
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Akurasi: {accuracy * 100:.2f}%")
print("Laporan Klasifikasi:")
print(report)
