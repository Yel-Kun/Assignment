import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, confusion_matrix

file_path = r'/Users/rwdorji/Downloads/annual-enterprise-survey-2023-financial-year-provisional.csv'

data = pd.read_csv(file_path)

data = data.apply(lambda col: pd.factorize(col)[0] if col.dtypes == 'O' else col)

target_column = data.columns[-1]
features = data.drop(target_column, axis=1)
target = data[target_column]

scaler = StandardScaler()
scaled_data = scaler.fit_transform(features)

pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

explained_variance = pca.explained_variance_ratio_
plt.bar(range(1, len(explained_variance) + 1), explained_variance)
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance')
plt.title('Explained Variance by Components')
plt.show()

plt.scatter(pca_data[:, 0], pca_data[:, 1], c=target, cmap='viridis')
plt.title('PCA: 2D Projection of Dataset')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.colorbar(label='Target Label')
plt.show()

dbscan = DBSCAN(eps=0.5, min_samples=5)
clusters = dbscan.fit_predict(pca_data)

plt.scatter(pca_data[:, 0], pca_data[:, 1], c=clusters, cmap='viridis')
plt.title('DBSCAN Clustering on PCA Data')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.colorbar(label='Cluster')
plt.show()

valid_clusters = clusters != -1
if valid_clusters.sum() > 1:
    silhouette_avg = silhouette_score(pca_data[valid_clusters], clusters[valid_clusters])
    print(f"Silhouette Score: {silhouette_avg:.3f}")
else:
    print("Only noise detected, no valid clusters.")

data['Cluster'] = clusters
cluster_summary = data.groupby('Cluster').mean()
print("\nCluster Summary (Feature Means):")
print(cluster_summary)

cm = confusion_matrix(target, clusters)
print("\nConfusion Matrix (True Labels vs DBSCAN Clusters):")
print(cm)

dbscan = DBSCAN(eps=0.7, min_samples=5)
clusters = dbscan.fit_predict(pca_data)

plt.scatter(pca_data[:, 0], pca_data[:, 1], c=clusters, cmap='viridis')
plt.title('DBSCAN with Tuning')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.colorbar(label='Cluster')
plt.show()
