# Import library yang dibutuhkan
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Membaca dataset
file_path = 'C:/Users/USER/OneDrive/Documents/MyPrograms/Programs/Data Science/computer_games.csv'
df = pd.read_csv(file_path)

# Melihat sekilas data
print(df.head())
print(df.info())

# 1. Distribusi Genre Permainan
plt.figure(figsize=(10, 6))
genre_counts = df['Genre'].value_counts().head(10)  # Mengambil 10 genre teratas
sns.barplot(x=genre_counts.values, y=genre_counts.index, palette='viridis')
plt.title('Distribusi Genre Permainan (Top 10)')
plt.xlabel('Jumlah Game')
plt.ylabel('Genre')
plt.show()

# 2. Distribusi Game Berdasarkan Produser (Top 10)
plt.figure(figsize=(10, 6))
producer_counts = df['Producer'].value_counts().head(10)  # Mengambil 10 produser teratas
sns.barplot(x=producer_counts.values, y=producer_counts.index, palette='coolwarm')
plt.title('Distribusi Game Berdasarkan Produser (Top 10)')
plt.xlabel('Jumlah Game')
plt.ylabel('Produser')
plt.show()

# 3. Tren Rilis Game per Tahun
# Mengekstrak tahun dari kolom 'Date Released'
df['Year'] = pd.to_datetime(df['Date Released'], errors='coerce').dt.year
games_per_year = df['Year'].value_counts().sort_index()

plt.figure(figsize=(12, 6))
sns.lineplot(x=games_per_year.index, y=games_per_year.values, marker='o', color='b')
plt.title('Tren Rilis Game per Tahun')
plt.xlabel('Tahun')
plt.ylabel('Jumlah Game')
plt.grid(True)
plt.show()

# 4. Top 10 Pengembang Game Terbanyak
plt.figure(figsize=(10, 6))
dev_counts = df['Developer'].value_counts().head(10)
sns.barplot(x=dev_counts.values, y=dev_counts.index, palette='magma')
plt.title('Top 10 Pengembang Game Terbanyak')
plt.xlabel('Jumlah Game')
plt.ylabel('Pengembang')
plt.show()
