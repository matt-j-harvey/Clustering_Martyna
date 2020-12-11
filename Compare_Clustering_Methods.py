import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
from sklearn.mixture  import GaussianMixture, BayesianGaussianMixture
from sklearn.cluster  import KMeans, AffinityPropagation, AgglomerativeClustering, SpectralClustering, DBSCAN, OPTICS, Birch, MeanShift


def load_real_data():
    file_location = r"C:\Users\matth\OneDrive\Documents\Martyna_Clustering_Numbers.csv"
    data = np.genfromtxt(file_location, delimiter=',')
    data = StandardScaler().fit_transform(data)
    return data

def plot_data(data, labels=np.array([None])):

    number_of_datapoints = np.shape(data)[0]
    number_of_dimensions = np.shape(data)[1]

    if labels.all() == None:
        labels = np.zeros(number_of_datapoints)

    if number_of_dimensions == 2:
        plt.scatter(x=data[:,0], y=data[:,1], c=labels, cmap='gist_rainbow')
        plt.show()

    elif number_of_dimensions == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, cmap='gist_rainbow', alpha=0.8)
        plt.show()

    elif number_of_dimensions > 3:
        pca_model = PCA(n_components=3)
        data = pca_model.fit_transform(data)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, cmap='gist_rainbow', alpha=0.8)
        plt.show()


def plot_data_on_axis(axis, data, labels=np.array([None])):

    number_of_datapoints = np.shape(data)[0]
    number_of_dimensions = np.shape(data)[1]

    if labels.all() == None:
        labels = np.zeros(number_of_datapoints)

    if number_of_dimensions == 2:
        axis.scatter(x=data[:, 0], y=data[:, 1], c=labels, cmap='gist_rainbow')

    elif number_of_dimensions == 3:
        axis.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, cmap='gist_rainbow', alpha=0.8)

    elif number_of_dimensions > 3:
        pca_model = PCA(n_components=3)
        data = pca_model.fit_transform(data)
        axis.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, cmap='gist_rainbow', alpha=0.8)



def clustering_auto_determine(data):

    figure_1 = plt.figure()

    axis_1 = figure_1.add_subplot((231),  projection='3d')
    axis_2 = figure_1.add_subplot((232),  projection='3d')
    axis_3 = figure_1.add_subplot((233),  projection='3d')
    axis_4 = figure_1.add_subplot((234),  projection='3d')
    axis_5 = figure_1.add_subplot((235),  projection='3d')
    axis_6 = figure_1.add_subplot((236),  projection='3d')

    #Affinity propagation
    labels = AffinityPropagation(damping=0.5).fit_predict(data)
    plot_data_on_axis(axis_1, data, labels)
    number_of_clusters = len(set(labels))
    axis_1.set_title("Affinity Propagation: " + str(number_of_clusters))

    #OPTICS
    labels = OPTICS().fit_predict(data)
    plot_data_on_axis(axis_2, data, labels)
    number_of_clusters = len(set(labels))
    axis_2.set_title("Optics:" + str(number_of_clusters))

    #Birch
    labels = Birch().fit_predict(data)
    plot_data_on_axis(axis_3, data, labels)
    number_of_clusters = len(set(labels))
    axis_3.set_title("Birch:" + str(number_of_clusters))

    #Aggloermative
    labels = AgglomerativeClustering(distance_threshold=10, n_clusters=None).fit_predict(data)
    plot_data_on_axis(axis_4, data, labels)
    number_of_clusters = len(set(labels))
    axis_4.set_title("Agglomerative:" + str(number_of_clusters))

    #DB Scan
    labels = DBSCAN().fit_predict(data)
    plot_data_on_axis(axis_5, data, labels)
    number_of_clusters = len(set(labels))
    axis_5.set_title("DB Scan:" + str(number_of_clusters))

    #Gaussian Mixture
    labels = MeanShift().fit_predict(data)
    plot_data_on_axis(axis_6, data, labels)
    number_of_clusters = len(set(labels))
    axis_6.set_title("Mean Shift:" + str(number_of_clusters))

    plt.show()


def gaussian_mixture_model(data, cluster_range=10):

    cluster_list = []
    aic_values = []
    bic_values = []
    log_likelihood_values = []

    for number_of_clusters in range(2, cluster_range):
        model = GaussianMixture(n_components=number_of_clusters, n_init=20)
        model.fit(data)

        aic = model.aic(data)
        bic = model.bic(data)
        likelihood = model.score(data)

        aic_values.append(aic)
        bic_values.append(bic)
        log_likelihood_values.append(likelihood)

        cluster_list.append(number_of_clusters)

    print(aic_values)
    print(bic_values)
    print(log_likelihood_values)
    print(cluster_list)

    plt.title("AIC")
    plt.plot(cluster_list, aic_values)
    plt.show()

    plt.title("BIC")
    plt.plot(cluster_list, bic_values)
    plt.show()

    plt.title("Likelihood")
    plt.plot(cluster_list, log_likelihood_values)
    plt.show()




#Load Data
#data = np.load(data_save_location)


data_save_location = r"C:\Users\matth\OneDrive\Documents\Martyna_Clustering\pseudo_data.npy"
labels_save_location = r"C:\Users\matth\OneDrive\Documents\Martyna_Clustering\pseudo_labels.npy"

data = load_real_data()
plot_data(data)

#clustering_auto_determine(data)

#data = PCA(n_components=3).fit_transform(data)
#clustering_auto_determine(data)


gaussian_mixture_model(data)