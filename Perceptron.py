# Perceptron Algorithm on the Sonar Dataset
from random import seed
from random import randrange
from csv import reader
import matplotlib.pyplot as plt

# Load a CSV file
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

# Convert string column to integer
def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	folds = cross_validation_split(dataset, n_folds)
	scores = list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
                # perceptron_sgd or perceptron_bgd 
		predicted = algorithm(train_set, test_set, fold, *args)
		actual = [row[-1] for row in fold]
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
	return scores

# Make a prediction with weights
def predict(row, weights):
	activation = weights[0]
	for i in range(len(row)-1):
		activation += weights[i + 1] * row[i]
        # Heaviside step function
	return 1.0 if activation >= 0.0 else 0.0

def get_prediction_accuracy(train, weights):
    predictions = list()
    for row in train:
        prediction = predict(row, weights)
        predictions.append(prediction)
    expected_out = [row[-1] for row in train]
    accuracy = accuracy_metric(expected_out, predictions)
    return accuracy

# Estimate Perceptron weights using stochastic gradient descent
def train_weights_sgd(train, fold_test, l_rate, n_epoch):
        accuracy=[]
	weights = [0.0 for i in range(len(train[0]))]
	for epoch in range(n_epoch):
		for row in train:
			prediction = predict(row, weights)
			error = row[-1] - prediction
                        # w = w + learning_rate * (expected - predicted)
			weights[0] = weights[0] + l_rate * error
			for i in range(len(row)-1):
                                # w = w + learning_rate * (expected - predicted) * x
				weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
                accuracy.append(get_prediction_accuracy(fold_test, weights))
        accuracies.append(accuracy)
	return weights

# Estimate Perceptron weights using batch gradient descent
def train_weights_bgd(train, fold_test, l_rate, n_epoch):
        accuracy=[]
	weights = [0.0 for i in range(len(train[0]))]
        weights_update = [0.0 for i in range(len(train[0]))]
	for epoch in range(n_epoch):
		sum_error = 0.0
		for row in train:
			prediction = predict(row, weights)
			error = row[-1] - prediction
			sum_error += error**2
                        # accumulate updates across the epoch
                        weights_update[0] += l_rate * error
			for i in range(len(row)-1):
				weights_update[i + 1] += l_rate * error * row[i]
                # update the weights in a batch at the end of the epoch
		for i in range(len(weights)):
			weights[i] = weights[i] + (weights_update[i]/n_epoch)
                accuracy.append(get_prediction_accuracy(fold_test, weights))
        accuracies.append(accuracy)
	return weights

# Perceptron Algorithm With Stochastic Gradient Descent
def perceptron_sgd(train, test, fold_test, l_rate, n_epoch):
	predictions = list()
	weights = train_weights_sgd(train, fold_test, l_rate, n_epoch)
	for row in test:
		prediction = predict(row, weights)
		predictions.append(prediction)
	return(predictions)

# Perceptron Algorithm With Batch Gradient Descent
def perceptron_bgd(train, test, fold_test, l_rate, n_epoch):
	predictions = list()
        weights = train_weights_bgd(train, fold_test, l_rate, n_epoch)
	for row in test:
		prediction = predict(row, weights)
		predictions.append(prediction)
	return(predictions)

# Test the Perceptron algorithm on the sonar dataset
seed(1)
accuracies_sgd = list()
accuracies_bgd = list()
# load and prepare data
filename = 'sonar.all-data.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])-1):
	str_column_to_float(dataset, i)
# convert string class to integers
str_column_to_int(dataset, len(dataset[0])-1)
# evaluate algorithm
n_folds = 3
l_rate = 0.01
n_epoch = 500
accuracies = list()
scores = evaluate_algorithm(dataset, perceptron_sgd, n_folds, l_rate, n_epoch)
print('Scores SGD: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
scores = evaluate_algorithm(dataset, perceptron_bgd, n_folds, l_rate, n_epoch)
print('Scores BGD: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
subplot = 200 + n_folds*10 + 1
for i in range(len(accuracies)):
    plt.subplot(subplot+i)
    plt.plot(accuracies[i])
    plt.grid(True)
    if(i < n_folds):
        plt.ylabel('Accuracy SGD %d' % i)
    else:
        plt.ylabel('Accuracy BGD %d' % (i-n_folds))
    plt.xlabel('epoch number')
plt.show()
