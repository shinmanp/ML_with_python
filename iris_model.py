from sklearn.datasets import load_iris

iris_dataset = load_iris()

# #Keys of iris_dataset
# print("Keys of iris_dataset :", iris_dataset.keys())
# Keys of iris_dataset : dict_keys(['data', 'target', 
# 	'target_names', 'DESCR', 'feature_names', 'filename'])

## Split the data into training and testing set, 
## Here 'random_state' used to suffle the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state = 0)

## Plot the data
## For plotting create DF using data in X_train
import pandas as pd
import matplotlib.pyplot as plt
iris_df = pd.DataFrame(X_train, columns = iris_dataset.feature_names)
## Create scatter matrix from the DF, color y_train
pd.plotting.scatter_matrix(iris_df, c = y_train, figsize=(15,15), marker='o', hist_kwds={'bins':20}, s=60, alpha=.8)
plt.show()

## Build model using k nearest neighbor algorithm
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train, y_train)

## Predict one data
import numpy as np
X = np.array([[5, 2.9, 1, 0.2]])
pred = knn.predict(X)
print("Predicted classification : ", iris_dataset['target_names'][pred])

## Prediction accuracy
y_pred = knn.predict(X_test)
print("Prediction Accuracy : ", np.mean(y_pred == y_test))