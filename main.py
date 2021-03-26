import pandas as pd
import numpy as np

# Read in data from files
# =================================
data_set = pd.read_csv("MNIST_15_15.csv")
labels = pd.read_csv("MNIST_LABEL.csv")

# Concatenate sets to randomize samples
data_set.insert(0, 'labels', labels)

# print(data_set)

# shuffle dataframe
data_set = data_set.sample(frac=1).reset_index(drop=True)

# n == # of samples, ie, rows
# p == # of features, ie, cols
n, p = data_set.shape

# less 1 to account for labels column
p -= 1

# print("n, p: ", n, p)

# Encode labels as 0 and 1
y = np.zeros(n)
y[data_set.iloc[:, 0] > 5] = 1

# Separate labels
X = data_set.iloc[:, 1:-1]
# print(X)
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

# print(b_opts)

# Calculate evaluation metrics
accuracies = []
TPRs = []
FPRs = []
i = 0
for i in range(len(b_opts)):
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0

    X_test = test_set
    y_groundtruth = test_labels
    b_opt = b_opts[i]

    # print(np.array(np.dot(X_test, b_opt)))
    # print(X_test)

    true_pos += sum(np.array(np.dot(X_test, b_opt) > 0.5) == y_groundtruth)
    true_neg += sum(np.array(np.dot(X_test, b_opt) <= 0.5) == y_groundtruth)
    false_pos += sum(np.array(np.dot(X_test, b_opt) > 0.5) != y_groundtruth)
    false_neg += sum(np.array(np.dot(X_test, b_opt) <= 0.5) != y_groundtruth)

    TPRs.append(true_pos / (true_pos + false_neg))
    FPRs.append(false_pos / (false_pos + true_neg))
    accuracies.append(sum(np.array(np.dot(X_test, b_opt) > 0.5) == y_groundtruth) / len(y_groundtruth))

avg_acc = sum(accuracies)
avg_acc = avg_acc / len(accuracies)
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

# Calculate True Positive Rate (TPR)


# pd.set_option("display.max_rows", None, "display.max_columns", None)
# print(y)
# print(X)
