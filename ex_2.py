import math
import sys

import numpy as np
from math import sqrt


# from sklearn.model_selection import KFold


# Convert win type column to int
def wine_type_to_int(dataset):
    for row in dataset:
        if row[11] == "W":
            row[11] = 1
        else:
            row[11] = 0
    return dataset


# Find the min and max values for each column
def dataset_minmax(dataset):
    minmax = list()
    for column_index in range(len(dataset[0]) - 1):
        col_values = [row[column_index] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        # list that every element is a tuple (min, max) of all values in that column
        minmax.append([value_min, value_max])
    return minmax


# Rescale dataset columns to the range 0-1
def minmax_normalize(dataset, minmax):
    for row in dataset:
        for i in range(len(row) - 1):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
    return dataset


def zscore_normalize(self, dataset):
    """
        Rescale dataset columns to the range 0-1
    """
    for row in range(dataset.shape[0]):
        for col in range(dataset.shape[1] - 1):
            mean = np.mean(dataset[:, col])
            standard_deviation = math.sqrt(np.var(dataset[:, col]))
            if standard_deviation != 0:
                dataset[row][col] = (dataset[row][col] - mean) / standard_deviation
    return dataset


# add ones column to X (X is a data set)
def add_ones_column(X):
    ones_col = np.ones((X.shape[0], 1))
    return np.hstack((ones_col, X))


class Knn:
    """
     class Knn to implement knn algorithm
    """

    def euclidean_distance(self, row1, row2):
        """"
            calculate the Euclidean distance between two vectors
        """
        distance = 0.0
        for i in range(len(row1)):
            distance += (row1[i] - row2[i]) ** 2
        return sqrt(distance)

    def get_neighbors(self, train_x, train_y, test_x_row, k):
        """
           Locate the most similar neighbors
           distances is a list of distances between row test and row train,
           every element is a tuple with row train values and the
           distance between this row to test row
        """
        distances = list()
        label_index = -1
        # calculate distance between test_x to every row in tarin_x
        for train_row in train_x:
            label_index += 1
            dist = self.euclidean_distance(test_x_row, train_row)
            distances.append((train_row, train_y[label_index], dist))
        # sort distances in descending  order
        distances.sort(key=lambda tup: tup[2])
        # a list of the k nearest rows in train to that row test
        k_nearest_neighbors = list()
        for i in range(k):
            k_nearest_neighbors.append((distances[i][0], distances[i][1]))
        return k_nearest_neighbors

    # Make a classification prediction with neighbors
    def predict_classification(self, train_x, train_y, test_x_row, k):
        # get k nearest neighbors
        neighbors = self.get_neighbors(train_x, train_y, test_x_row, k)
        # list of the labels of the k nearest
        output_values = [row[1] for row in neighbors]
        prediction = max(set(output_values), key=output_values.count)
        return prediction

    """ 
    def knn_train(self, data_train_x, data_train_y):
        #funcion that I used in order to choose hyper-parameters
        foldes_scores = 0 # sum up of accuracy in every fold
        folds = 7 # number of foldes to devide the data set in order to train and evaluate accuracy
        k = 7 # choose the number of closet neighbors
        kf = KFold(n_splits=folds) # devide data set into 7 folds
        for train_index, test_index in kf.split(data_train_x):
            correct_prediction = 0 # correct predictions in each fold
            X_train, X_test = data_train_x[train_index], data_train_x[test_index]
            y_train, y_test = data_train_y[train_index], data_train_y[test_index]
            # normalize the data by min max algorithm
            minmax_train = dataset_minmax(X_train)
            X_train = minmax_normalize(X_train, minmax_train)
            X_test = minmax_normalize(X_test, minmax_train)
            # evaluate the data set according to corrects predictions
            for data_test_row in range(len(X_test)):
                prediction = self.predict_classification(X_train, y_train, X_test[data_test_row], k)
                expected = y_test[data_test_row]
                if expected == prediction:
                    correct_prediction += 1
            # accuracy of single fold
            accuracy_score_fold = correct_prediction / len(X_test)
            # sum up of accuracy of all folds
            foldes_scores += accuracy_score_fold
        # average accuracy of all folds
        knn_accuracy = foldes_scores / folds
        print("k:" + str(k) + " accuracy: " + str(knn_accuracy * 100))
    """

    def knn_test(self, X_train, Y_train, X_test):
        """
            function that predict every label X_test and return list of all the predictions
        """
        k = 7
        predictions_list = list()
        for data_test_row in range(len(data_test_x)):
            minmax_train = dataset_minmax(X_train)
            X_train = minmax_normalize(X_train, minmax_train)
            X_test = minmax_normalize(X_test, minmax_train)
            prediction = self.predict_classification(X_train, Y_train, X_test[data_test_row], k)
            predictions_list.append(prediction)
        return predictions_list


class Perceptron(object):
    """
     class Perceptron to implement Perceptron algorithm
    """

    def __init__(self, no_of_inputs, iterations, learning_rate):
        '''
        The function initialized the Perceptron model.
        no_of_inputs - number of inputs to the perceptron (excluding the bias)
        iterations - number of iterations on the training data
        learning_rate - learning rate, how much the weight will change during update
        '''
        self.iterations = iterations
        self.learning_rate = learning_rate
        np.random.seed(30)  # set random seed, should not be altered!
        self.weights = np.empty(shape=(3, no_of_inputs))
        self.weights[0] = np.random.randint(2, size=no_of_inputs)
        self.weights[1] = np.random.randint(2, size=no_of_inputs)
        self.weights[2] = np.random.randint(2, size=no_of_inputs)

    def evaluate_train(self, inputs, labels):
        '''
        The function makes a predictions for the given inputs and compares
        against the labels (ground truth). It returns the accuracy.
        Accuracy = #correct_classification / #total
        '''
        correct_classification = 0
        for x, label in zip(inputs, labels):
            if np.argmax(np.dot(self.weights, x)) == label:
                correct_classification += 1
        total = np.shape(inputs)[0]
        return correct_classification / total

    def train(self, training_inputs, train_labels):
        '''
        The function train a perceptron model given training_inputs and train_labels.
        '''
        for x, label in zip(training_inputs, train_labels):
            # predict
            prediction = np.argmax(np.dot(self.weights, x))
            if prediction != label:
                # learn
                self.weights[int(label), :] = self.weights[int(label), :] + self.learning_rate * x
                self.weights[prediction, :] = self.weights[prediction, :] - self.learning_rate * x

    """ 
    def perceptron_function_train(self, data_train_x, data_train_y):
        # funcion that I used in order to choose hyper-parameters
        total_folds_accuracy = 0
        folds = 7
        kf = KFold(n_splits=folds)
        for train_index, test_index in kf.split(data_train_x):
            for iter in range(self.iterations):
                X_train, X_test = data_train_x[train_index], data_train_x[test_index]
                y_train, y_test = data_train_y[train_index], data_train_y[test_index]

                # normalize data
                minmax_train = dataset_minmax(X_train)
                X_train = minmax_normalize(X_train, minmax_train)
                X_test = minmax_normalize(X_test, minmax_train)


                # add ones column to X_train and X_test
                X_train = add_ones_column(X_train)
                X_test = add_ones_column(X_test)

                # shuffle X_train and y_train
                indices_train = np.arange(X_train.shape[0])
                np.random.shuffle(indices_train)
                X_train = X_train[indices_train]
                y_train = y_train[indices_train]
                # shuffle X_test and y_test
                indices_test = np.arange(X_test.shape[0])
                np.random.shuffle(indices_test)
                X_test = X_test[indices_test]
                y_test = y_test[indices_test]

                # train model
                self.train(X_train, y_train)
            accuracy = self.evaluate_train(X_test, y_test)
            total_folds_accuracy += accuracy
        print("perceptron test accuracy: " + str((total_folds_accuracy/folds )*100))
    """

    def perceptron_function_test(self, X_train, y_train, X_test):
        """"
            function that predict every label in X_test and return a list of all the predictions
        """
        predict = list()
        # normalize data
        minmax_train = dataset_minmax(X_train)
        X_train = minmax_normalize(X_train, minmax_train)
        X_test = minmax_normalize(X_test, minmax_train)

        # add ones column to X_train and X_test
        X_train = add_ones_column(X_train)
        X_test = add_ones_column(X_test)

        for iter in range(iterations):
            # shuffle X_train and y_train
            indices_train = np.arange(X_train.shape[0])
            np.random.shuffle(indices_train)
            X_train = X_train[indices_train]
            y_train = y_train[indices_train]

            # train model
            self.train(X_train, y_train)

        for x in X_test:
            predict.append(np.argmax(np.dot(self.weights, x)))
        return predict


class Passive_Aggressie(object):
    """
     class Passive_Aggressie to implement Passive Aggressie algorithm
    """

    def __init__(self, no_of_inputs, iterations):
        '''
        The function initialized the Perceptron model.
        no_of_inputs - number of inputs to the perceptron (excluding the bias)
        iterations - number of iterations on the training data
        learning_rate - learning rate, how much the weight will change during update
        '''
        self.iterations = iterations
        np.random.seed(30)  # set random seed, should not be altered!
        self.weights = np.empty(shape=(3, no_of_inputs))
        self.weights[0] = np.random.randint(2, size=no_of_inputs)
        self.weights[1] = np.random.randint(2, size=no_of_inputs)
        self.weights[2] = np.random.randint(2, size=no_of_inputs)

    def evaluate_train(self, inputs, labels):
        '''
        The function makes a predictions for the given inputs and compares
        against the labels (ground truth). It returns the accuracy.
        Accuracy = #correct_classification / #total
        '''
        correct_classification = 0
        for x, label in zip(inputs, labels):
            if np.argmax(np.dot(self.weights, x)) == label:
                correct_classification += 1
        total = np.shape(inputs)[0]
        return correct_classification / total

    def tao(self, w_label, w_prediction, x):
        """
            calculate tao according to the formula
        """
        loss = max(0, (1.0 - w_label @ x + w_prediction @ x))
        return loss / (2 * ((np.linalg.norm(x)) ** 2))

    def train(self, training_inputs, train_labels):
        '''
        The function train a perceptron model given training_inputs and train_labels.
        It also evaluates the model on the train set and test set after every iteration.
        '''
        for x, label in zip(training_inputs, train_labels):
            # predict
            prediction = np.argmax(np.dot(self.weights, x))
            if prediction != label:
                # learn
                tao = self.tao(self.weights[int(label), :], self.weights[prediction, :], x)
                self.weights[int(label), :] = self.weights[int(label), :] + tao * x
                self.weights[prediction, :] = self.weights[prediction, :] - tao * x

    """ 
    def  passive_aggressie_function_train(self, data_train_x, data_train_y):
        #funcion that I used in order to choose hyper-parameters
        total_folds_accuracy = 0
        folds = 7
        kf = KFold(n_splits=folds)
        for train_index, test_index in kf.split(data_train_x):
            for iter in range(self.iterations):
                X_train, X_test = data_train_x[train_index], data_train_x[test_index]
                y_train, y_test = data_train_y[train_index], data_train_y[test_index]

                # normalize data
                minmax_train = dataset_minmax(X_train)
                X_train = minmax_normalize(X_train, minmax_train)
                X_test = minmax_normalize(X_test, minmax_train)

                # add ones column to X_train and X_test
                X_train = add_ones_column(X_train)
                X_test = add_ones_column(X_test)

                # shuffle X_train and y_train
                indices_train = np.arange(X_train.shape[0])
                np.random.shuffle(indices_train)
                X_train = X_train[indices_train]
                y_train = y_train[indices_train]
                # shuffle X_test and y_test
                indices_test = np.arange(X_test.shape[0])
                np.random.shuffle(indices_test)
                X_test = X_test[indices_test]
                y_test = y_test[indices_test]

                # train model
                self.train(X_train, y_train)
            accuracy = self.evaluate_train(X_test, y_test)
            total_folds_accuracy += accuracy
        print("passive aggressie accuracy:  " + str((total_folds_accuracy/folds )*100))
    """

    def passive_aggressie_function_test(self, X_train, y_train, X_test):
        """
            function that predict every label in X_test and return a list of all the predictions.
        """
        predict = list()
        # normalize data
        minmax_train = dataset_minmax(X_train)
        X_train = minmax_normalize(X_train, minmax_train)
        X_test = minmax_normalize(X_test, minmax_train)

        # add ones column to X_train and X_test
        X_train = add_ones_column(X_train)
        X_test = add_ones_column(X_test)

        for iter in range(iterations):
            # shuffle X_train and y_train
            indices_train = np.arange(X_train.shape[0])
            np.random.shuffle(indices_train)
            X_train = X_train[indices_train]
            y_train = y_train[indices_train]

            # train model
            self.train(X_train, y_train)

        for x in X_test:
            predict.append(np.argmax(np.dot(self.weights, x)))
        return predict


if __name__ == "__main__":
    # load data from train_x.txt, train_y.txt and test_x.txt files
    data_train_x = np.genfromtxt(sys.argv[1], delimiter=",", dtype=np.float)
    data_train_y = np.genfromtxt(sys.argv[2], delimiter=',', dtype=np.float)
    data_test_x = np.genfromtxt(sys.argv[3], delimiter=',', dtype=np.float)
    # data_train_x = np.delete(data_train_x, [0, 10, 11], 1) # delete the first column
    # convert win type W to 1 and R to 0
    data_train_x = wine_type_to_int(data_train_x)
    data_test_x = wine_type_to_int(data_test_x)

    # run KNN
    knn_model = Knn()
    # knn_model.knn_train(data_train_x, data_train_y)
    knn_yhat = knn_model.knn_test(data_train_x, data_train_y, data_test_x)

    # run Perceptron
    iterations = 11
    no_of_inputs = data_train_x.shape[1] + 1
    learning_rate = 0.239
    perceptron_model = Perceptron(no_of_inputs, iterations, learning_rate)
    # perceptron_model.perceptron_function_train(data_train_x, data_train_y)
    perceptron_yhat = perceptron_model.perceptron_function_test(data_train_x, data_train_y, data_test_x)

    # run Passive Aggressie
    iterations = 89
    no_of_inputs = data_train_x.shape[1] + 1
    passive_aggressie_model = Passive_Aggressie(no_of_inputs, iterations)
    # passive_aggressie_model.passive_aggressie_function_train(data_train_x, data_train_y)
    pa_yhat = passive_aggressie_model.passive_aggressie_function_test(data_train_x, data_train_y, data_test_x)
    for i in range(len(pa_yhat)):
        print(f"knn: {int(knn_yhat[i])}, perceptron: {perceptron_yhat[i]}, pa: {pa_yhat[i]}")