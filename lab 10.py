import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

# Load Iris dataset
iris = load_iris()
x = iris.data
y = iris.target

# Create PCA object and transform data
pca = PCA(n_components=2)
X_projected = pca.fit_transform(x)
print("Shape of data:",x.shape) 
print("shape of transformed data:",X_projected.shape) 

# Plot the results
plt.scatter(X_projected[:, 0], X_projected[:, 1], c=y, edgecolor='k', cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Iris Dataset')
plt.show()