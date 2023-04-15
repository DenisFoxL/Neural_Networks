import numpy as np
from sklearn.model_selection import train_test_split

# Please replace these with your ID numbers before submission.
STUDENT_IDS = [206263071, 314880477]


def add_ones_column(X):
    # add a column of ones to the data to represent the bias term
    ones_col = np.ones((X.shape[0], 1))
    return np.hstack((ones_col, X))


class AvgPerceptron(object):
    def __init__(self, n_features, max_epochs=10, learning_rate=1):
        '''
        The function initialized the Perceptron model.
        n_features - number of features used in the data (excluding the bias)
        iterations - number of iterations on the training data
        learning_rate - learning rate, how much the weight will change during update
        '''
        self.iterations = max_epochs  # number of iterations the perceptron learning algorithm will perform (given there is no early stopping condition)
        self.learning_rate = learning_rate  # the learning rate of the model
        np.random.seed(30)  # set random seed, should not be altered!
        self.bias = 0  # initializing bias as 0
        # initializing weight vector as random values between 0 and 1; folding bias into the weight vector (from here on out we will treat the bias as part of the weight vector)
        self.weights = np.concatenate(([self.bias], np.random.rand(n_features)))
        np.random.seed(30)
        self.u = np.concatenate(
            ([self.bias], np.random.rand(n_features)))  # initializing weight cache to be the same as the weight vector
        self.c = 1  # initializing total updates counter

    def predict(self, inputs) -> np.array:
        """
        The function makes a prediction for the given inputs.
        Output: -1 or 1.
        Do not use a for loop in your implementation!
        """
        predictions = np.sign(np.dot(inputs, self.weights))  # predicting by multiplying input matrix by weight vector
        return predictions

    def evaluate(self, inputs, labels) -> float:
        """
        The function makes a predictions for the given inputs and compares
        against the labels (ground truth). It returns the accuracy.
        Accuracy = #correct_classification / #total
        """
        predictions = self.predict(inputs)  # getting prediction vector (values are either 1 or -1)
        comparison = labels * predictions  # multiplying labels by predictions so that if the prediction is correct the result is 1 and if incorrect then -1
        accuracy = np.count_nonzero(comparison == 1) / len(
            comparison)  # calculating proportion of correct predictions to total predictions
        return accuracy

    def train(self, training_inputs, train_labels, verbose=True):
        '''
        The function train a perceptron model given training_inputs and train_labels.
        It also evaluates the model on the train set and test set after every iteration.
        '''
        for i in range(self.iterations):
            for x, y in zip(training_inputs, train_labels):
                if y * self.predict(x) <= 0:  # update condition
                    self.weights += self.learning_rate * y * x  # updating weight vector
                    self.u += y * self.c * self.learning_rate * x  # updating weight cache giving bigger importance to weights that survived for longer
                self.c += 1  # updating counter
            if verbose:
                train_acc = self.evaluate(training_inputs, train_labels)  # assessing accuracy at current iteration
                print(f"Iteration No.{i}, Train accuracy: {train_acc}")
        self.weights -= (self.u / self.c) #updating weights to average weights
        self.bias = self.weights[0] #extracting bias as the w0
        return self.weights, self.bias


# Load the dataset using a numpy function: np.genfromtxt
data = np.genfromtxt("hw02_dataset.csv", delimiter=',', dtype=np.float)  ### for local environment

print("print 5 rows from the data")
print(data[:5])
# print its shape
print("Data shape:", data.shape)

# Split the data into features and labels and print their shape.
# Be careful not to change the content of the data.
features = data[:, :-1]
labels = data[:, -1]
# print their shape
print("Features Shape:", features.shape)
print("Labels Shape:", labels.shape)

# Count how many samples are from class -1, and how many to class +1.
class_pos_idx = (labels == 1).sum()
class_neg_idx = (labels == -1).sum()
print("Num samples for class -1:", class_neg_idx)
print("Num samples for class +1:", class_pos_idx)

# We can now split the data into train and test.
# The train would be 80% of the total #samples.
# The rest will go for test-set.
# We can use train_test_split function from sklearn.
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=36)

# If you want, you can use the method 'add_ones_column' defined above to add a dimension
# with a fixed '1' to the data as another feature. It's the folding bias trick you've seen in class.
# ---------Uncomment the following lines:-------------
X_train = add_ones_column(X_train)
X_test = add_ones_column(X_test)
# ----------------------------------------------------

# Run the perceptron after completing the skeleton below:
n_samples, n_features = features.shape
perceptron = AvgPerceptron(n_features)
perceptron.train(X_train, y_train)
test_acc = perceptron.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc}")

print("\n\nHow is this algorithm different from the perceptron algorithm learned in class?")
your_answer = """In class, we learned about the Classical Perceptron algorithm, while here we have an Averaged 
Perceptron algorithm. The Perceptron algorithm updates weights after each misclassification and uses the final weight 
vector (the result of the training stage of the model) for prediction. The Averaged Perceptron algorithm on the other 
hand, updates weights in a similar manner but also keeps a running sum of the averaged weight vector (weight vectors 
that lasted more iterations gain a higher importance), where in the end the fitted weights are the averaged weights 
resulting in higher accuracy."""
print(your_answer)
