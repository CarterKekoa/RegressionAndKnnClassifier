import numpy as np
import mysklearn.mypytable as mypytable
import operator
import copy
import random

def mean(x):
    """Computes the mean of a list of values
    """
    return sum(x)/len(x)

def compute_slope_intercept(x, y):
    """
    """
    mean_x = mean(x)
    mean_y = mean(y)
    
    m = sum([(x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x))]) / sum([(x[i] - mean_x) ** 2 for i in range(len(x))])
    # y = mx + b => b = y - mx
    b = mean_y - m * mean_x
    return float(m), float(b)

def compute_euclidean_distance(v1, v2):
    """
    """
    assert len(v1) == len(v2)
    dist = np.sqrt(sum([(v1[i] - v2[i]) ** 2 for i in range(len(v1))]))
    return dist

def scale(vals, test_vals):
    """
    """
    scaled_vals_list = []
    maxs_list = []
    mins_list = []

    # for each list in list vals, get each values max and min and store in a list accordingly
    for i in range(len(vals[0])):
        maxs_list.append(max([val[i] for val in vals]))
        mins_list.append(min([val[i] for val in vals]))

    # for each list in list vals, scale each value according to the max and min to be between [0, 1]
    for row in vals:
        curr = []
        for i in range(len(row)):
            curr.append((row[i]-mins_list[i])/(maxs_list[i]-mins_list[i]))
        scaled_vals_list.append(curr)

    # for each list in list test_vals, scale each value according to the max and min to be between [0, 1]
    for row in test_vals:
        curr = []
        for i in range(len(row)):
            curr.append((row[i]-mins_list[i])/(maxs_list[i]-mins_list[i]))
        scaled_vals_list.append(curr)
    
    # returns all scaled values from the vals list, then the scaled values from the test_vals list
    return scaled_vals_list[:len(vals)], scaled_vals_list[len(vals):]

def kneighbors_prep(scaled_X_train, scaled_X_test, n_neighbors):
    """
    """
    scaled_X_train = copy.deepcopy(scaled_X_train)
    scaled_X_test = copy.deepcopy(scaled_X_test)

    # for each scaled list in scaled_X_train
    for i, instance in enumerate(scaled_X_train):
        distance = compute_euclidean_distance(instance, scaled_X_test) 
        instance.append(i)  # append the original row index
        instance.append(distance)   # append the distance
    
    
    scaled_X_train_sorted = sorted(scaled_X_train, key=operator.itemgetter(-1)) # sort the list in assending order
    top_k = scaled_X_train_sorted[:n_neighbors] # get a list of the top_k neighbors

    distances_list = []
    indices_list = []

    # for each row in the top_k list, append the distances and indices to their own lists
    for row in top_k:
        distances_list.append(row[-1])
        indices_list.append(row[-2])
    
    # return the distances and indices lists
    return distances_list, indices_list

def get_label(labels):
    """
    """
    label_types = []
    # for each value in the labels list
    for val in labels:
        # if we have not see that label type
        if val not in label_types:
            label_types.append(val) # append to list of label types
    
    count_list = [0 for label_type in label_types]

    # for value in label types
    for i, val in enumerate(label_types):
        for label in labels:
            # if the value is == to the label then incremept the count for that position
            if val == label:
                count_list[i] += 1

    max_count = 0
    label_prediction = ''
    # for value in count_list
    for i, val in enumerate(count_list):
        if val > max_count:
            label_prediction = label_types[i]
 
    return label_prediction

def get_unique(vals):
    """
    """
    unique = []
    # for values in the vals list
    for val in vals:
        if val not in unique:
            unique.append(val)
    return unique

def group_by(x_train, y_train):
    """
    """
    unique = get_unique(y_train)
    grouped = [[] for _ in unique]
    # for each value in y_train
    for i, val in enumerate(y_train):
        for j, label in enumerate(unique):
            if val == label:
                grouped[j].append(i)
    return grouped

def shuffle(X, y):
    """
    """
    for i in range(len(X)):
        rand_index = random.randrange(0, len(X)) # [0, len(X))
        X[i], X[rand_index] = X[rand_index], X[i] # this is the temporary value swap but in one line
        if y is not None:
            y[i], y[rand_index] = y[rand_index], y[i]

def get_from_folds(X_vals, y_vals, train_folds, test_folds):
    """
    """
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    # for each fold
    for row in train_folds:
        for i in row:
            X_train.append(X_vals[i])
            y_train.append(y_vals[i])

    # for each test fold
    for row in test_folds:
        for i in row:
            X_test.append(X_vals[i])
            y_test.append(y_vals[i])

    return X_train, y_train, X_test, y_test