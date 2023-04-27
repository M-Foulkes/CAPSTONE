import pandas as pd
import numpy as np

players = pd.read_csv("ArmyData.csv")

features = ["Distance", "Sprint Dist", "Power Plays", "Energy", "Impacts", "Player Load", "Top Speed", "Distance/min", "Power Score", "Work Ratio"]

data = players[features].copy()

#1. scale the data, want every column to be treated equally
#put every column in terms of 1 to 10 to ensure no one column dominates the cluster

data = ((data- data.min())/(data.max()-data.min())) *9  + 1 # minimum needs to be greater than 0
players = players.dropna(subset=features)

#2. initialize random centroids

def initial_centroids(data, k):
    centroids = []
    for i in range(k):
        centroid = data.apply(lambda x: float(x.sample()))
        centroids.append(centroid)
    return pd.concat(centroids, axis = 1)

centroids = initial_centroids(data, 3)

#3. label each data point
#Finds the distance between data point and cluster center (Centroid)
#assign data point to cluster

def get_labels(data, centroids):
    distances = centroids.apply(lambda x: np.sqrt(((data - x) **2).sum(axis=1))) #apply function to each centroid
    return distances.idxmin(axis=1) #tells you what cluster each point belongs to

labels = get_labels(data, centroids)
labels.value_counts()


#4. update centroids
#calculate geometric mean

def update_centroids(data, labels, k):
    return data.groupby(labels).apply(lambda x: np.exp(np.log(x).mean())).T #.T switches rows and columns so that each column is a centroid and each row is a feature

from sklearn.decomposition import PCA #PCA will summarize columns into two dimensions
import matplotlib.pyplot as plt
from IPython.display import clear_output

def plot_clusters(data, labels, centroids, iteration):
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(data)
    centroids_2d = pca.transform(centroids.T)
    clear_output(wait=True)
    plt.title('Clustering Results for Wearable and Film Data')
    plt.scatter(x=data_2d[:,0], y=data_2d[:,1], c=labels)
    plt.scatter(x=centroids_2d[:,0], y=centroids_2d[:,1])
    plt.show()

max_iterations = 100
k = 3

centroids = initial_centroids(data, k)
old_centroids = pd.DataFrame()
iteration = 1

while iteration < max_iterations and not centroids.equals(old_centroids):   #the centroids.equals(old_centroids) will stop loop when centroids stop moving before max iterations
    old_centroids = centroids

    labels = get_labels(data, centroids)
    centroids = update_centroids(data, labels, k)
    iteration +=1
    
#plot_clusters(data, labels, centroids, iteration)
print("Cluster 0:")
print(players[labels ==0][["Player"] + features])
print("Cluster 1:")
print(players[labels ==1][["Player"] + features])
print("Cluster 2:") 
print(players[labels ==2][["Player"] + features])
    



#Sources: 
#https://www.youtube.com/watch?v=lX-3nGHDhQg&t=1712s

 
 


                      
