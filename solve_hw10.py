import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import os

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False

# 讀取資料
filename = 'spotify-2023.csv'
if os.path.exists(filename):
    df = pd.read_csv(filename, encoding='latin-1')
elif os.path.exists(os.path.join('HW10', filename)):
    df = pd.read_csv(os.path.join('HW10', filename), encoding='latin-1')
else:
    print("找不到 spotify-2023.csv")
    exit()

# 特徵
features = ['bpm', 'danceability_%', 'valence_%', 'energy_%', 'acousticness_%', 'instrumentalness_%', 'liveness_%', 'speechiness_%']
data = df[features]

data = data.dropna()

# 資料標準化
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

## 1. KMeans
print("正在執行 KMeans...")
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(data_scaled)

print("KMeans Inertia:", kmeans.inertia_)
print("KMeans Silhouette Score:", silhouette_score(data_scaled, kmeans_labels))

## 2. Agglomerative Clustering
print("正在執行 Agglomerative Clustering...")
agg = AgglomerativeClustering(n_clusters=3)
agg_labels = agg.fit_predict(data_scaled)

print("Agglomerative Silhouette Score:", silhouette_score(data_scaled, agg_labels))

## 3. DBSCAN
print("正在執行 DBSCAN...")
dbscan = DBSCAN(eps=2.0, min_samples=5)
dbscan_labels = dbscan.fit_predict(data_scaled)

## 計算 DBSCAN 分數
unique_labels = set(dbscan_labels)
if len(unique_labels) > 1:
    print("DBSCAN Silhouette Score:", silhouette_score(data_scaled, dbscan_labels))
    print(f"DBSCAN 分出了 {len(unique_labels) - (1 if -1 in unique_labels else 0)} 群")
else:
    print("DBSCAN 分群結果只有一類或是全為雜訊")

## PCA 降維視覺化
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=kmeans_labels, cmap='viridis', s=10)
plt.title('KMeans')

plt.subplot(1, 3, 2)
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=agg_labels, cmap='viridis', s=10)
plt.title('Agglomerative')

plt.subplot(1, 3, 3)
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=dbscan_labels, cmap='viridis', s=10)
plt.title('DBSCAN')

plt.tight_layout()
plt.savefig('HW10/clustering_results.png')
plt.show()
