# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 14:48:26 2024

@author: hilal
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA 
from sklearn.metrics.pairwise import cosine_similarity 
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


df = pd.read_csv('customer_segmentation_dataset.csv')
print(df.shape)
print(df.head())
print(df.columns)
# Checking null values
print(df.isnull().sum())

#Filling Missing Values
df['CREDIT_LIMIT'].fillna(df['CREDIT_LIMIT'].mean(),inplace = True)
df['MINIMUM_PAYMENTS'].fillna(df['MINIMUM_PAYMENTS'].mean(),inplace = True)

#Dropping Unnecessary Column
df.drop('CUST_ID', axis = 1, inplace = True)

#Scaling Data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

#Outlier Detection
lof_model = LocalOutlierFactor()
outlier_flags = lof_model.fit_predict(scaled_data)
df = df[outlier_flags == 1] 
scaled_data = scaled_data[outlier_flags == 1]

#Elbow Method
max_clusters = 40
inertia_values = []
for it in range(1,max_clusters):
    kmeans_model = KMeans(n_clusters = it)
    kmeans_model.fit(scaled_data)
    inertia_values.append(kmeans_model.inertia_)
    
plt.plot(inertia_values,'go-')
plt.title('Elbow Method for Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

def evaluate_clustering(labels, data):
    silhouette_avg = silhouette_score(data, labels)
    calinski_harabasz_avg = calinski_harabasz_score(data, labels)
    davies_bouldin_avg = davies_bouldin_score(data, labels)
    
    print(f"Silhouette Score: {silhouette_avg:.2f}")
    print(f"Calinski-Harabasz Index: {calinski_harabasz_avg:.2f}")
    print(f"Davies-Bouldin Index: {davies_bouldin_avg:.2f}")
    
    
#Optimal Cluster number is 6 according to Elbow Method
num_clusters = 6

#Kmeans Clustering Model
kmeans_model = KMeans(n_clusters =num_clusters)
kmeans_labels = kmeans_model.fit_predict(scaled_data) #output
print('Kmeans Evaluation')
evaluate_clustering(kmeans_labels,scaled_data)
# Agglomerative Clustering Model
agglo_model = AgglomerativeClustering(n_clusters = num_clusters)
agglo_labels = agglo_model.fit_predict(scaled_data)
print('Agglomerative Evaluation')
evaluate_clustering(agglo_labels,scaled_data)
# Combining Cluster Labels 
labels_mapping = {'kmeans': kmeans_labels,'agglomerative': agglo_labels}
clustered_data = pd.concat([df,pd.DataFrame({'cluster':kmeans_labels})],axis = 1)

#Plotting Histograms for Clusters
for feature in clustered_data.columns[:-1]:#excluding cluster column
    clusters_plot = sns.FacetGrid(clustered_data,col = 'cluster')
    clusters_plot.map(plt.hist,feature,bins = 25 )
    clusters_plot.set_titles(f"Histogram of {feature} by clusters")
    
# PCA METHOD
distance_matrix = 1-cosine_similarity(scaled_data)
pca_model = PCA(n_components = 2)
pca_result = pca_model.fit_transform(distance_matrix)
print(pca_result.shape)

x_coords,y_coords = pca_result[:,0],pca_result[:,1]

# Cluster Visulasiton
color_palette = {0:'cyan',
                 1:'magenta',
                 2: 'lime',
                 3: 'coral',
                 4: 'teal',
                 5: 'orange' }


label_descriptions = {0: 'Frequent All-Purpose Purchasers',
                      1: 'Delayed Payment Customers',
                      2: 'Installment Buyers',
                      3: 'Advance Cash Users',
                      4: 'High Expenditure Customers',
                      5: 'Low Spending Customers'}

for algo,labels in labels_mapping.items():
    results_df = pd.DataFrame({'x':x_coords,'y':y_coords,'label':labels})
    grouped_results = results_df.groupby('label')
    
    plt.figure(figsize = (14,10))
    for label,group in grouped_results:
        plt.scatter(group.x, group.y, c = color_palette[label],label = label_descriptions[label],edgecolor = 'black', s = 50)
        
        
    plt.legend()
    plt.title(f"Customer Segmentation Using {algo.capitalize()} method ")
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.show()