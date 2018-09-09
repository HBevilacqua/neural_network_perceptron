# neural_network_perceptron: stochastic and batch gradient descents

Purpose: how to implement the Perceptron algorithm using stochastic or batch gradient descent from scratch with Python?<br>

The initial software is provided by the amazing tutorial "*How To Implement The Perceptron Algorithm From Scratch In Python*" by Jason Brownlee.<br>
https://machinelearningmastery.com/implement-perceptron-algorithm-scratch-python/

You should read this tuto which outlines the following steps:<br>
- Makin predictions
- Training network weights
- Modeling the sonar dataset

I add the batch gradient descent.

## Glossary

#### Perceptron
The perceptrion is a supervised algorithm used for binary classifiers. It decides whether an input vector belongs to a class or not.<br>
In neural network field, it represents one artificial neuron using the Heaviside step as transfer function.

#### Gardient
The gradient (∇f) of a scalar-valued multivariable function f(x,y,…) gathers all its partial derivatives (
∂f/∂x, ∂f/∂Y, ...) into a vector.

#### Gardient descent 
It is a first order optmization algorithm to fing the minimum of a function, generally used in ML when it is not possible to find the solutions of the equation ∂J(θ)/∂θ = 0, I mean all θ that min J(θ).

#### Training a network
Updates weights in a neural network to improve its predictions according to a dataset.

#### Classification (in ML)
Classification aims to predict a label. The outputs are class labels.

#### Dataset
Data used to train and test the network.

#### k-cross validation
It is a procedure used to estimate the skill of the model on new data.<br>
k refers to the number of groups that a given data sample is to be split into.
Sequence:
1. Shuffle the dataset randomly.
2. Split the dataset into k groups
3. For each unique group:<br>
     A. Take the group as a hold out or test data set<br>
     B. Take the remaining groups as a training data set<br>
     C. Fit a model on the training set and evaluate it on the test set<br>
     D. Retain the evaluation score and discard the model<br>
4. Summarize the skill of the model using the sample of model evaluation scores

#### Epoch
One epoch = One cycle (foward + backward) through the entire training dataset (all the rows "inputs/outputs" seen).
