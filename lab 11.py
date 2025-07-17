import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Create LDA object
lda = LinearDiscriminantAnalysis(n_components=2)

# Apply LDA
X_projected = lda.fit_transform(X, y)
print("shape of data:",X.shape) 
print("shape of transformed data:",X_projected.shape) 


# Plot the projected data
plt.scatter(X_projected[:, 0], X_projected[:, 1], c=y, cmap='viridis', edgecolor='k')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.title('LDA Projection of Iris Dataset')
plt.show()
