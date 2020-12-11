import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, SpectralEmbedding
from sklearn.cluster import KMeans, AffinityPropagation, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import silhouette_samples, silhouette_score


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def perform_silhouette_scoring(data):

    silhouette_scores = []
    number_of_clusters_list = range(2, 9)

    figure_1 = plt.figure()
    axes_list = []
    for x in range(8):
        axes_list.append(figure_1.add_subplot(2, 4, x+1))

    silhouette_scores = []
    for x in range(7):

        # Get the Current Axis
        current_axis = axes_list[x]

        # Get The Current Number Of Cluster
        cluster_number = number_of_clusters_list[x]

        # Perform Clustering
        labels = KMeans(n_clusters=cluster_number, n_init=20).fit_predict(data)

        # Score
        average_score = silhouette_score(data, labels)
        silhouette_scores.append(average_score)

        # Make a nice empty list
        cluster_silhouette_values_list = []
        for cluster in range(cluster_number):
            cluster_silhouette_values_list.append([])

        # Put Each Silhouet Value Into The Correct Cluster List

        # Get Individual Scores
        individual_silhouette_values = silhouette_samples(data, labels)

        number_of_datapoints = np.shape(data)[0]
        for point in range(number_of_datapoints):
            point_score = individual_silhouette_values[point]
            cluster = labels[point]
            cluster_silhouette_values_list[cluster].append(point_score)

        current_y_start = 0
        for cluster in range(cluster_number):
            cluster_scores = cluster_silhouette_values_list[cluster]
            cluster_scores.sort()
            print("Cluster scores", cluster_scores)
            cluster_size = len(cluster_scores)
            current_y_end = current_y_start + cluster_size

            y = list(range(current_y_start, current_y_end))
            x1 = np.zeros(cluster_size)
            x2 = cluster_scores
            current_axis.fill_betweenx(y, x1, x2)

            current_y_start = current_y_end

        current_axis.set_title(str(cluster_number) + " Clusters")

    axes_list[-1].set_title("Silhouette Scores")
    axes_list[-1].plot(number_of_clusters_list, silhouette_scores)
    axes_list[-1].set_xticks(number_of_clusters_list)

    plt.show()


#Load Real Data
file_location = r"C:\Users\matth\OneDrive\Documents\Martyna_Clustering_Numbers.csv"
data = np.genfromtxt(file_location, delimiter=',')

#Load Fake Data
data_save_location = r"C:\Users\matth\OneDrive\Documents\Martyna_Clustering\pseudo_data.npy"
labels_save_location = r"C:\Users\matth\OneDrive\Documents\Martyna_Clustering\pseudo_labels.npy"
data = np.load(data_save_location)

#PreProcess Data
scaled_data = StandardScaler().fit_transform(data)

#Perform Silhouette Scoring
perform_silhouette_scoring(scaled_data)

