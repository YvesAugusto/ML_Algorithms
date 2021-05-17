from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

class PCA:

    def __init__(self, data, n_ft):
        self.n_ft = n_ft
        self.data = data

    @property
    def cov_matrix(self):
        return np.cov(self.data.T)

    @property
    def eigen(self):
        eig_vals, eig_vecs = np.linalg.eig(self.cov_matrix)
        pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vecs))]
        pairs.sort(
            key=lambda v: v[0], reverse=True
        )
        return np.array(pairs)

    def transform(self):
        eigen = self.eigen[:self.n_ft,1][0]
        return np.array(
            self.data.dot(eigen)
        )

blobs = datasets.make_blobs(n_samples=100, n_features=3)
X, Y = blobs
X = (X-X.mean())/X.std()
X = np.array(X)
cov_matrix = np.cov(X.T)
eig_vals, eig_vecs = np.linalg.eig(cov_matrix)

pca = PCA(X,2)
P = pca.transform()
plt.scatter(X[:,0], X[:,1], c=['black'])
plt.scatter(X[:,0], P, color=['blue'])
plt.show()
