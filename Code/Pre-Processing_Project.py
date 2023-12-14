import pandas as pd
import seaborn as sns
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np


class DataLoader:
    def __init__(self, file_path):
        self.df = None
        self.file_path = file_path

    def load_data(self):
        self.df = pd.read_csv(self.file_path)
        return self.df

    def display_data(self):
        print(self.df.head())


class DataVisualizer:
    @staticmethod
    def plot_bar(data, title, xlabel, ylabel):
        plt.figure(figsize=(10, 6))
        data.plot(kind='bar', color='skyblue')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    @staticmethod
    def plot_boxplot(df, x, y, title, palette='viridis'):
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=x, y=y, data=df, palette=palette)
        plt.title(title)
        plt.show()

    @staticmethod
    def plot_scatter(df, x, y, title, color='blue', marker='o'):
        plt.figure(figsize=(10, 6))
        plt.scatter(df[x], df[y], color=color, marker=marker)
        plt.title(title)
        plt.xlabel(x)
        plt.ylabel(y)
        plt.show()

    @staticmethod
    def plot_histogram(df, column, title, bins=10, color='skyblue'):
        plt.figure(figsize=(10, 6))
        df[column].plot(kind='hist', bins=bins, color=color)
        plt.title(title)
        plt.xlabel(column)
        plt.show()

    @staticmethod
    def plot_dendrogram(linkage_matrix, title):
        plt.figure(figsize=(10, 7))
        dendrogram(linkage_matrix, orientation='top', distance_sort='descending', show_leaf_counts=True)
        plt.title(title)
        plt.xlabel('Sample Index')
        plt.ylabel('Distance')
        plt.show()


class DataFilter:
    def __init__(self, df):
        self.df = df

    def filter_by_countries(self, countries):
        return self.df[self.df['country'].isin(countries)]

    def filter_top_cities_in_country(self, country, top_n=10):
        country_data = self.df[self.df['country'] == country]
        top_cities = country_data['city'].value_counts().head(top_n).index
        return country_data[country_data['city'].isin(top_cities)]

    def remove_negative_values(self, columns):
        for column in columns:
            self.df[column] = np.where(self.df[column] < 0, np.nan, self.df[column])
        return self.df

    def drop_na(self):
        self.df.dropna(inplace=True)
        return self.df

    def split_location_into_lat_lng(self):
        if 'location' in self.df.columns:
            self.df[['latitude', 'longitude']] = self.df['location'].str.extract(r"'lat': ([\d.-]+).*'lng': ([\d.-]+)")
            self.df[['latitude', 'longitude']] = self.df[['latitude', 'longitude']].apply(pd.to_numeric)
            self.df.drop('location', axis=1, inplace=True)
        return self.df


class Clustering:
    def __init__(self, df):
        self.df = df

    def perform_kmeans(self, features, k, random_state=42):
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        kmeans.fit(features)
        self.df['Cluster'] = kmeans.labels_
        return kmeans.cluster_centers_, kmeans.labels_

    def perform_agglomerative_clustering(self, features, n_clusters=None, distance_threshold=0):
        agglom = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward',
                                         distance_threshold=distance_threshold)
        agglom.fit(features)
        self.df['Cluster'] = agglom.labels_
        return agglom

    @staticmethod
    def plot_dendrogram(features, method='ward'):
        linked = linkage(features, method=method)
        plt.figure(figsize=(10, 7))
        dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Sample Index')
        plt.ylabel('Distance')
        plt.show()

    @staticmethod
    def elbow_method(features, max_k=10):
        wcss = []
        for i in range(1, max_k + 1):
            kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
            kmeans.fit(features)
            wcss.append(kmeans.inertia_)
        plt.plot(range(1, max_k + 1), wcss)
        plt.title('Elbow Method')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.show()

    @staticmethod
    def silhouette_method(features, range_n_clusters):
        for n_clusters in range_n_clusters:
            clusterer = KMeans(n_clusters=n_clusters, random_state=10)
            cluster_labels = clusterer.fit_predict(features)
            silhouette_avg = silhouette_score(features, cluster_labels)
            print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)


class Main:
    def __init__(self, file_path):
        self.data_loader = DataLoader(file_path)
        self.df = self.data_loader.load_data()
        self.visualizer = DataVisualizer()
        self.filterer = DataFilter(self.df)
        self.clusterer = Clustering(self.df)

    def run_eda(self):
        # Plotting the count of each country
        country_counts = self.df['country'].value_counts()
        self.visualizer.plot_bar(country_counts, 'Count of each Country', 'Country', 'Count')

        # Box plot to visualize the spread of 'price' and 'rating' by 'country'
        self.visualizer.plot_boxplot(self.df, 'country', 'price', 'Boxplot of Price by Country')
        self.visualizer.plot_boxplot(self.df, 'country', 'rating', 'Boxplot of Rating by Country')

        # Scatter plot for latitudes and longitudes
        self.visualizer.plot_scatter(self.df, 'latitude', 'longitude', 'Latitude vs Longitude Scatter Plot')

        # Histogram for ratings
        self.visualizer.plot_histogram(self.df, 'rating', 'Histogram of Ratings')

    def run_data_filtering(self):
        # Filtering data for specific countries
        filtered_df = self.filterer.filter_by_countries(['Canada', 'United States'])

        # Removing negative values from 'price' and 'rating'
        self.filterer.remove_negative_values(['price', 'rating'])

        # Dropping NA values
        self.filterer.drop_na()

        # Splitting location into latitude and longitude
        self.filterer.split_location_into_lat_lng()

        # Updating DataFrame after filtering
        self.df = self.filterer.df

    def run_clustering(self):
        # Selecting features for clustering
        features = self.df[['price', 'rating', 'sentiment_y', 'normalized_popularity']].values

        # Standardize features for clustering
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Elbow Method for determining the number of clusters
        self.clusterer.elbow_method(features_scaled, max_k=10)

        # Silhouette Method for evaluating clustering
        self.clusterer.silhouette_method(features_scaled, range_n_clusters=[2, 3, 4, 5])

        # Performing KMeans Clustering with an optimal number of clusters
        centers, labels = self.clusterer.perform_kmeans(features_scaled, k=4)  # k=4 is an example value

        # Performing Agglomerative Clustering
        agglom = self.clusterer.perform_agglomerative_clustering(features_scaled)

        # Plotting dendrogram for Agglomerative Clustering
        self.clusterer.plot_dendrogram(features_scaled)


if __name__ == "__main__":
    file_path = r'C:\Users\Siddharth\PycharmProjects\Datamining\full_cleaned.csv'   # This file path
    main = Main(file_path)
    main.run_eda()
    main.run_data_filtering()
    main.run_clustering()
    main.export_data()
