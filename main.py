import pandas as pd
import numpy as np

# Read in data from files
# =================================
data_set = pd.read_csv("MNIST_15_15.csv")
labels = pd.read_csv("MNIST_LABEL.csv")

# Concatenate sets to randomize samples
data_set.insert(0, 'labels', labels)

# shuffle dataframe
data_set = data_set.sample(frac=1).reset_index(drop=True)

# n == # of samples, ie, rows
# p == # of features, ie, cols
n, p = data_set.shape

# less 1 to account for labels column
p -= 1

# Encode labels as 0 and 1
y = np.zeros(n)
y[data_set.iloc[:, 0] > 5] = 1

# Separate labels
X = data_set.iloc[:, 1:-1]
X = pd.DataFrame(np.c_[np.ones(n), X])

# Normalize data
X = X/255

# Split into 10 folds for cross-validation
# Folds 0-8: training sets
# Fold 9: test sets
folds = np.array_split(X, 10)

# Split labels for cross-validation
# Folds 0-8: training sets
# Fold 9: test sets
label_folds = np.array_split(y, 10)

# Training Data
training_sets = folds[:-1]
training_labels = label_folds[:-1]

# Test Data
test_set = folds[-1]
test_labels = label_folds[-1]

# Run Linear Regression
b_opts = []
i = 0
for i in range(len(training_sets)):
    X = training_sets[i]
    y = training_labels[i]
    b_opts.append(np.dot(np.dot(np.linalg.pinv(np.dot(X.transpose(), X)), X.transpose()), y))
    i += 1

# Calculate evaluation metrics
accuracies = []
TPRs = []
FPRs = []
i = 0
for i in range(len(b_opts)):
    X_test = test_set
    y_groundtruth = test_labels.astype(int)

    b_opt = b_opts[i]
    predictions = (np.array(np.dot(X_test, b_opt)) > 0.5).astype(int)

    # Get confusion matrix sums
    true_pos = np.logical_and(predictions == 1, y_groundtruth == 1).sum()
    true_neg = np.logical_and(predictions == 0, y_groundtruth == 0).sum()
    false_pos = np.logical_and(predictions == 1, y_groundtruth == 0).sum()
    false_neg = np.logical_and(predictions == 0, y_groundtruth == 1).sum()

    # Calc metrics
    TPRs.append(true_pos / (true_pos + false_neg))
    FPRs.append(false_pos / (false_pos + true_neg))
    acc = sum(np.array(np.dot(X_test, b_opt) > 0.5) == y_groundtruth) / len(y_groundtruth)
    accuracies.append(acc)

avg_acc = sum(accuracies)
avg_acc = avg_acc / len(accuracies)

# Print Eval Metrics table
print("|===========|=======|=======|")
print("| Accuracy  | TPR   | FPR   |")
print("|===========|=======|=======|")
for i in range(len(accuracies)):
    print("|   " + str(round(accuracies[i]*100, 1)) + "%   |"
          + " " + str(round(TPRs[i]*100, 1)) + "% |"
          + " " + str(round(FPRs[i]*100, 1)) + "% |")
print("|===========|=======|=======|")
# print(accuracies)
print("|  Average Accuracy: " + str(round(avg_acc*100, 1)) + "%  |")
print("|===========|=======|=======|")
