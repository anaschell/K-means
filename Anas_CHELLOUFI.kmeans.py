import numpy as np

class KMeans:
    def __init__(self, nombre_clusters=3, max_iteration=50):
        self.nombre_clusters = nombre_clusters
        self.max_iteration = max_iteration

    def fit(self, test_data):
        centre = test_data[np.random.choice(range(len(test_data)), self.nombre_clusters, replace=False)]
        
        for _ in range(self.max_iteration):
            distance_pt_centre = np.linalg.norm(test_data[:, np.newaxis] - centre, axis=2)
            labels = np.argmin(distance_pt_centre, axis=1)
            nv_centre = np.array([test_data[labels == i].mean(axis=0) for i in range(self.nombre_clusters)])
            centre = nv_centre
        
        self.labels = labels
        self.centre = centre
        
        return self.centre, self.labels