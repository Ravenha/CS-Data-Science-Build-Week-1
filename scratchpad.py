import numpy as np
from scipy import stats
import seaborn as sns

#Euclidean Distance = sqrt( ((q1 - p1)^2) + ((q2 - p2)^2) + ... + ((qn - pn)^2) )
# ⟨u1−v1,u2−v2⟩

def euclidean_distance(vectorA, vectorB):
    # predict
    x = vectorA[0] - vectorB[0]
    y = vectorA[1] - vectorB[1]
    x2 = x ** 2
    y2 = y ** 2
    distance = np.sqrt(x2 + y2)

    return distance

# # Test distance function
# point = [2.7810836,2.550537003,0]
# dataset = [
# 	[1.465489372,2.362125076,0],
# 	[3.396561688,4.400293529,0],
# 	[1.38807019,1.850220317,0],
# 	[3.06407232,3.005305973,0],
# 	[7.627531214,2.759262235,1],
# 	[5.332441248,2.088626775,1],
# 	[6.922596716,1.77106367,1],
# 	[8.675418651,-0.242068655,1],
# 	[7.673756466,3.508563011,1]
#     ]


X = [
    [1.465489372,2.362125076],
    [3.396561688,4.400293529],
    [1.38807019,1.850220317],
    [3.06407232,3.005305973],
    [7.627531214,2.759262235],
    [5.332441248,2.088626775],
    [6.922596716,1.77106367],
    [8.675418651,-0.242068655],
    [7.673756466,3.508563011]
    ]
y = [0, 0, 0, 0, 1, 1, 1, 1, 1]
point = [2.7810836,2.550537003] #0

for row in dataset:
    # predict
	distance = euclidean_distance(point, row)
	row.append(distance)

#print(dataset)

def neighbors(dataset, k):
    # predict
    sorted_dataset = sorted(dataset, key=lambda data: data[3])
    top_x = sorted_dataset[0:k]
    return top_x

small_dataset = neighbors(dataset, 5)
# print(small_dataset)

def class_pred(small_dataset):
    # predict
    array = []
    for row in small_dataset:
        classification = row[2]
        array.append(classification)
        object = stats.mode(array)
    return object.mode[0]

print(class_pred(small_dataset))

def isolate_points(small_dataset):
    # For graphing
    points = []
    for row in small_dataset:
        point = row[0], row[1]
        points.append(point)
    return points

print(isolate_points(small_dataset))
