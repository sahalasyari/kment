
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Menghasilkan data sampel
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Mendefinisikan nilai-nilai K yang berbeda
k_values = [5, 6, 7]

# Plot
fig, axs = plt.subplots(1, len(k_values), figsize=(15, 5))

for i, k in enumerate(k_values):
    # Melakukan pengelompokan KMeans
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    
    # Plotting cluster
    axs[i].scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis', s=50, alpha=0.5)
    axs[i].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', s=200, marker='X')
    axs[i].set_title('K = {}'.format(k))

plt.show()
