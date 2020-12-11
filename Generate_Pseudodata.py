from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import numpy as np


def plot_data(data, labels):

    number_of_dimensions = np.shape(data)[1]

    # If 2D Use a 2D Scatter Plot
    if number_of_dimensions == 2:
        plt.scatter(x=data[:,0], y=data[:,1], c=labels, cmap='gist_rainbow')
        plt.show()

    # If 3D Use a 3D Scatter Plot
    elif number_of_dimensions == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, cmap='gist_rainbow', alpha=0.8)
        plt.show()

    #If More than 3D - Do PCA and then plot in 3D
    elif number_of_dimensions > 3:
        pca_model = PCA(n_components=3)
        data = pca_model.fit_transform(data)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, cmap='gist_rainbow', alpha=0.8)
        plt.show()


#Set Locations To Save the Pseudodata for future analysis
data_save_location = r"C:\Users\matth\OneDrive\Documents\Martyna_Clustering\pseudo_data.npy"
labels_save_location = r"C:\Users\matth\OneDrive\Documents\Martyna_Clustering\pseudo_labels.npy"

# Settings for Generating Pseduodata
number_of_samples = 100
number_of_dimensions = 22
number_of_clusters = 4
cluster_standard_deviation = 10
data, labels = make_blobs(n_samples=number_of_samples, n_features=number_of_dimensions, centers=number_of_clusters, cluster_std=cluster_standard_deviation)

#Save This Data
np.save(data_save_location, data)
np.save(labels_save_location, labels)

#Plot This Data
plot_data(data, labels)

