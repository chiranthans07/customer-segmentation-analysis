#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 22:58:37 2026

@author: chiranthansateesh
"""

# Customer Segmentation using K-Means Clustering

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load dataset
data = pd.read_csv("Mall_Customers.csv")

print("Dataset Loaded Successfully")

# Display first rows
print(data.head())

# Select features for clustering
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Determine optimal clusters using Elbow Method
wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11), wcss)
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# Apply K-Means
kmeans = KMeans(n_clusters=5, random_state=42)
y_kmeans = kmeans.fit_predict(X)

# Visualization of clusters
plt.scatter(X[y_kmeans==0]['Annual Income (k$)'],
            X[y_kmeans==0]['Spending Score (1-100)'],
            s=50, c='red', label='Cluster 1')

plt.scatter(X[y_kmeans==1]['Annual Income (k$)'],
            X[y_kmeans==1]['Spending Score (1-100)'],
            s=50, c='blue', label='Cluster 2')

plt.scatter(X[y_kmeans==2]['Annual Income (k$)'],
            X[y_kmeans==2]['Spending Score (1-100)'],
            s=50, c='green', label='Cluster 3')

plt.scatter(X[y_kmeans==3]['Annual Income (k$)'],
            X[y_kmeans==3]['Spending Score (1-100)'],
            s=50, c='cyan', label='Cluster 4')

plt.scatter(X[y_kmeans==4]['Annual Income (k$)'],
            X[y_kmeans==4]['Spending Score (1-100)'],
            s=50, c='magenta', label='Cluster 5')

plt.scatter(kmeans.cluster_centers_[:,0],
            kmeans.cluster_centers_[:,1],
            s=200, c='yellow', label='Centroids')

plt.title("Customer Segments")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.legend()
plt.show()