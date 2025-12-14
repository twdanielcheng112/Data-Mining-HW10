import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os

# 設定中文字型
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False

# 1. 資料讀取與前處理
print("=== 資料讀取與前處理 ===")
# 嘗試讀取 HW10 資料夾中的檔案
paths_to_check = [
    'spotify-2023.csv',
    'HW10/spotify-2023.csv',
    '../HW10/spotify-2023.csv'
]

filename = None
for path in paths_to_check:
    if os.path.exists(path):
        filename = path
        break

if filename:
    df = pd.read_csv(filename, encoding='latin-1')
    print(f"成功讀取資料集: {filename}")
else:
    print("找不到資料集，請確認 spotify-2023.csv 是否存在")
    exit()

# 選取特徵與目標
# 我們嘗試預測歌曲的調性 (mode): Major (大調) vs Minor (小調)
# 或者預測是否為熱門歌曲 (streams > median)
# 這裡選擇預測 'mode'
target_col = 'mode'
feature_cols = ['bpm', 'danceability_%', 'valence_%', 'energy_%', 
                'acousticness_%', 'instrumentalness_%', 'liveness_%', 'speechiness_%']

# 移除缺失值
df_clean = df[feature_cols + [target_col]].dropna()

# 將目標欄位轉為數值 (Major=1, Minor=0)
le = LabelEncoder()
df_clean[target_col] = le.fit_transform(df_clean[target_col])
print(f"目標欄位 '{target_col}' 編碼對應: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# 儲存前處理後的資料
df_clean.to_csv('processed_spotify_data.csv', index=False)
print("已儲存前處理後的資料至 'processed_spotify_data.csv'")

# 切分訓練集與測試集
X = df_clean[feature_cols]
y = df_clean[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 特徵標準化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. 建立分類器並評估
classifiers = {
    'K-Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Gaussian NB': GaussianNB(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'SVC': SVC(kernel='rbf', random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'MLP (Neural Net)': MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
}

results = {}

print("\n=== 分類器效能評估 ===")
for name, clf in classifiers.items():
    print(f"\n正在訓練 {name}...")
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    
    print(f"--- {name} ---")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    
    # 繪製 Confusion Matrix (只存檔不顯示，避免跳出視窗)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'cm_{name.replace(" ", "_")}.png')
    plt.close()

# 3. 比較結果視覺化
plt.figure(figsize=(10, 6))
plt.bar(results.keys(), results.values(), color='skyblue')
plt.title('各分類器準確率比較 (Accuracy Comparison)')
plt.ylabel('Accuracy')
plt.ylim(0, 1.0)
for i, v in enumerate(results.values()):
    plt.text(i, v + 0.01, f'{v:.2f}', ha='center')
plt.tight_layout()
plt.savefig('classification_comparison.png')
print("\n已儲存比較圖表至 'classification_comparison.png'")
