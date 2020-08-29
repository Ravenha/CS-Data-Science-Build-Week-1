from sklearn import datasets
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

iris = datasets.load_iris()
X = iris.data
y = iris.target

K = 3
model = KNeighborsClassifier(n_neighbors = K)
model.fit(X, y)

# Make predictions
print('(-2, -2, 2, 2) is class:')
print(model.predict(X, [-2, -2, 2, 2]))  # Class 0

print('(1, 5, 5, 1) is class:')
print(model.predict(X, [1, 5, 5, 1]))  # Class 1

print('(10, 10, 10, 10) is class:')
print(model.predict(X, [10, 10, 10, 10]))  # Class 2
