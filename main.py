import pandas as pd
import numpy as np

from datetime import datetime

# Read in data from files
# =================================
training_set = pd.read_csv("MNIST_15_15.csv")
test_set = pd.read_csv("MNIST_LABEL.csv")

# print(training_set)
# print(test_set)

