# src/anomaly_detection_mall.py
# 无监督“欺诈=异常”检测：IsolationForest + LOF（可选 OCSVM）
# 运行：python src/anomaly_detection_mall.py

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

# ========= 配置 =========
DATA_PATH = "data/Mall_Customers.csv"     # 改成你实际路径/文件名
OUT_DIR   = "outputs"                      # 结果输出目录
CONTAM    = 0.05                           # 认为有 5% 的“可疑/异常”（可改 0.01~0.10）

USE_OCSVM = False                          # True 时会跑 One-Class SVM（慢）
RANDOM_STATE = 42
# =======================

os.makedirs(OUT_DIR, exist_ok=True)

# 1) 读取数据
df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
print(df.head())

# 2) 轻度清洗/特征准备
# 统一列名（避免空格/特殊字符）
df = df.rename(columns={
    "Annual Income (k$)": "Annual_Income_k",
    "Spending Score (1-100)": "Spending_Score"
})

# 编码 Gender -> 0/1
if "Gender" in df.columns:
    df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})

# 选择数值特征
feature_cols = [c for c in ["Age", "Annual_Income_k", "Spending_Score", "Gender"] if c in df.columns]
X = df[feature_cols].copy()

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3) 训练多个异常检测器，得到异常分
# Isolation Forest（越小越异常；我们取负的score作为“异常分”，越大=越异常）
iso = IsolationForest(
    n_estimators=300,
    contamination=CONTAM,
    random_state=RANDOM_STATE,
)
iso.fit(X_scaled)
iso_score = -iso.score_samples(X_scaled)  # 变成越大越异常

# LOF（局部离群因子，封装成分数：负值越小越异常；我们同样转成正向“异常分”）
# 注意：LOF默认不支持在新数据上 predict_proba，我们只用于当前数据评分
lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination=CONTAM,
    novelty=False
)
lof_labels = lof.fit_predict(X_scaled)      # -1 异常，1 正常
# 用负的负对数密度近似异常分；也可用 -lof.negative_outlier_factor_
lof_score = -lof.negative_outlier_factor_

# 可选：OCSVM（可能较慢）
if USE_OCSVM:
    ocs = OneClassSVM(kernel="rbf", gamma="scale", nu=CONTAM)
    ocs.fit(X_scaled)
    # decision_function 越小越异常；取负让“越大=越异常”
    ocs_score = -ocs.decision_function(X_scaled)
else:
    ocs_score = None

# 4) 分数归一化 + 简单集成（平均）
def minmax_norm(a):
    a = np.asarray(a, dtype=float)
    amin, amax = np.min(a), np.max(a)
    if amax == amin:
        return np.zeros_like(a)
    return (a - amin) / (amax - amin)

scores = [minmax_norm(iso_score), minmax_norm(lof_score)]
if ocs_score is not None:
    scores.append(minmax_norm(ocs_score))

ensemble_score = np.mean(np.vstack(scores), axis=0)  # 简单平均作为“最终异常分”

# 5) 按污染率确定阈值 -> 输出“可疑=1/正常=0”
threshold = np.quantile(ensemble_score, 1 - CONTAM)  # top CONTAM 作为异常
pred_anomaly = (ensemble_score >= threshold).astype(int)

# 6) 保存结果
out = df.copy()
out["anomaly_score"] = ensemble_score
out["is_suspect"] = pred_anomaly  # 1=可疑(“潜在欺诈”), 0=正常
out_path = os.path.join(OUT_DIR, "mall_anomaly_results.csv")
out.to_csv(out_path, index=False)

meta = {
    "features": feature_cols,
    "contamination": CONTAM,
    "threshold": float(threshold),
    "models": {
        "isolation_forest": True,
        "lof": True,
        "one_class_svm": USE_OCSVM
    }
}
with open(os.path.join(OUT_DIR, "mall_anomaly_metadata.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)

print(f"Saved results to: {out_path}")
print(out[["CustomerID"] + feature_cols + ["anomaly_score", "is_suspect"]].head(10))

# 7) 简单可视化（Income vs Spending，颜色区分是否异常）
# 只用 matplotlib，避免 seaborn
plt.figure()
colors = np.where(pred_anomaly==1, "red", "blue")
plt.scatter(out["Annual_Income_k"], out["Spending_Score"], c=colors, alpha=0.7)
plt.xlabel("Annual_Income_k")
plt.ylabel("Spending_Score")
plt.title("Anomaly Detection (red = suspect)")
plt.show()
