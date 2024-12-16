import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
from abc import ABC, abstractmethod


class Metric(nn.Module, ABC):
    """
    Base Metric class for clustering and embedding evaluations.
    """

    @staticmethod
    def to_one_hot(y: np.ndarray, dtype, device) -> torch.Tensor:
        """
        Convert a 1D numpy array to one-hot encoding.
        """
        if not isinstance(y, np.ndarray):
            raise ValueError("y must be a numpy array")
        classes = np.unique(y)
        identity = torch.eye(len(classes), dtype=dtype, device=device)
        y_onehot = torch.zeros((len(y), len(classes)), dtype=dtype, device=device)
        for i, cls in enumerate(classes):
            y_onehot[y == cls] = identity[i]
        return y_onehot

    @staticmethod
    def within_between(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute within-class and between-class variance.
        """
        x_c = torch.einsum("nd,nc->cd", x, y)  # CxD
        between = torch.norm(x_c - x_c.mean(dim=0, keepdim=True), p=2).mean()

        x_c = torch.einsum("cd,nc->nd", x_c, y)  # NxD
        within = torch.norm(x - x_c, p=2).mean()

        return within, between

    @abstractmethod
    def compute(self, *args, **kwargs):
        """
        Abstract method to compute a specific metric.
        Must be implemented in subclasses.
        """
        pass


class DaviesBouldinMetric(Metric):
    """
    Metric using the Davies-Bouldin score.
    """

    def compute(self, embeddings: np.ndarray, labels: np.ndarray) -> float:
        return 1 / davies_bouldin_score(embeddings, labels)


class NearestNeighborMetric(Metric):
    """
    Metric based on nearest neighbor mismatch ratios.
    """

    def compute(self, embeddings: np.ndarray, labels: np.ndarray, min_nn: int = 10) -> float:
        nbrs = NearestNeighbors(n_neighbors=min_nn + 1).fit(embeddings)
        _, indices = nbrs.kneighbors(embeddings)

        mismatch_ratios = np.array([
            np.mean(labels[neighbors[1:]] != labels[i]) for i, neighbors in enumerate(indices)
        ])
        return 1 - np.mean(mismatch_ratios)


class FisherMetric(Metric):
    """
    Fisher score-based metric.
    """

    def compute(self, embeddings: np.ndarray, labels: np.ndarray) -> float:
        unique_classes = np.unique(labels)
        overall_mean = np.mean(embeddings, axis=0)

        embeddings -= overall_mean

        within_scatter = np.zeros(embeddings.shape[1])
        between_scatter = np.zeros(embeddings.shape[1])

        for cls in unique_classes:
            class_data = embeddings[labels == cls]
            class_mean = np.mean(class_data, axis=0)
            n_cls = class_data.shape[0]

            class_data -= class_mean
            within_scatter += np.sum(class_data ** 2, axis=0)
            between_scatter += n_cls * class_mean ** 2

        between_scatter /= embeddings.shape[0]
        within_scatter /= embeddings.shape[0]

        return (between_scatter / within_scatter).sum()


class CompositeMetric(Metric):
    """
    Composite metric combining multiple metric computations.
    """

    def __init__(self, metrics):
        super().__init__()
        self.metrics = metrics

    def compute(self, embeddings: np.ndarray, labels: np.ndarray) -> dict:
        """
        Compute all metrics and return as a dictionary.
        """
        results = {}
        for name, metric in self.metrics.items():
            results[name] = metric.compute(embeddings, labels)
        return results

'''
# Example Usage
if __name__ == "__main__":
    # Simulate data
    embeddings = np.random.rand(100, 10)  # 100 samples with 10 features
    labels = np.random.randint(0, 3, 100)  # 3 classes

    # Initialize metrics
    metrics = {
        "DaviesBouldin": DaviesBouldinMetric(),
        "NearestNeighbor": NearestNeighborMetric(),
        "Fisher": FisherMetric(),
    }
    composite_metric = CompositeMetric(metrics)

    # Compute results
    results = composite_metric.compute(embeddings, labels)
    print("Metric Results:", results)

    # or
    print(FisherMetric().compute(embeddings,labels))
'''