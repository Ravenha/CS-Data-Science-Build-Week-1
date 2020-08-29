import numpy as np
from scipy import stats
from sklearn import datasets
import matplotlib.pyplot as plt
from pandas import DataFrame


class K_Nearest_Neighbors():
    """
    Determine the classifcation of a given point based on existing data
    points
    Takes in number of neighbors to be calculated with
    """
    def __init__(self, k):
        self.k = k

    def euclidean_distance(self, vectorA, vectorB):
        """
        Helper function to calculate Euclidean distance
        """
        squared = []
        for x in range(len(vectorA)):
            n = vectorA[x] - vectorB[x]
            n2 = n ** 2
            squared.append(n2)
        distance = np.sqrt(sum(squared))
        return distance

    def fit(self, X, y):
        """
        X is a list of lists containing data values
        y is a list of classifications
        """
        for i in range(len(X)):
            # Add the classification to the list with data values
            X[i].append(y[i])
        return X

    def plot(self, X, target):
        """
        Plotting
        Two-dimensional plots ONLY
        """
        # Make sure data is two-dimensional
        if len(X[0]) == 3:
            # Working with impermanent versions of X and target
            temp_X = X.copy()
            temp_target = target.copy()

            # Add an arbitrary, fake, class value to our target so it will
            # be a different color
            temp_target = temp_target.append(-3)
            temp_X.append(temp_target)

            # Convert the list into a dataframe and plot it
            df = DataFrame(temp_X, columns=['x', 'y', 'class'])
            plot = plt.scatter(x=df['x'], y=df['y'], c=df['class']);
            return plot

        # If data is not two-dimensional
        else:
            return "Data with more than two dimensions cannot be plotted."

    def predict(self, X, target):
        """
        Target is a tuple
        """
        # Calculate Euclidean distance between target and each other point in
        # our dataset, append to list (final item)
        for point in X:
            distance = self.euclidean_distance(target, point)
            point.append(distance)

        # Sort based on distances
        sorted_data = sorted(X, key=lambda data: data[-1])
        # Isolate the closest k points
        top_x = sorted_data[0:self.k]

        array = []
        for row in top_x:
            # Determine the classifications of the top k points
            classification = row[-2]
            array.append(classification)

        object = stats.mode(array)
        # Get the mode of the classifications
        mode = object.mode[0]

        # Remove the distances so we can predict other targets
        for point in X:
            del point[-1]

        return mode


if __name__ == '__main__':
    # 2D dataset; test plotting
    X = [[1.465489372, 2.362125076],
         [3.396561688, 4.400293529],
         [1.38807019, 1.850220317],
         [3.06407232, 3.005305973],
         [7.627531214, 2.759262235],
         [5.332441248, 2.088626775],
         [6.922596716, 1.77106367],
         [8.675418651, -0.242068655],
         [7.673756466, 3.508563011]]
    y = [0, 0, 0, 0, 1, 1, 1, 1, 1]
    point = [2.7810836, 2.550537003]

    model = K_Nearest_Neighbors(3)
    model.fit(X, y)
    print(model.plot(X, point))
    print(model.predict(X, point))

    # Dataset with more than 2 dimensions
    iris = datasets.load_iris()
    X = iris.data.tolist()
    y = iris.target.tolist()

    model = K_Nearest_Neighbors(3)
    model.fit(X, y)

    # Make predictions
    print('(-2, -2, 2, 2) is class:')
    print(model.predict(X, [-2, -2, 2, 2]))  # Class 0

    print('(1, 5, 5, 1) is class:')
    print(model.predict(X, [1, 5, 5, 1]))  # Class 1

    print('(10, 10, 10, 10) is class:')
    print(model.predict(X, [10, 10, 10, 10]))  # Class 2
