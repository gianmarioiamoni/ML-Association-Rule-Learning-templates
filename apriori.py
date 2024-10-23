import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing
#
# Data in the provided file correspond to trasactions of different customers in 1 week period of time
#
# data table doesn't contain any header row, so we specify header=None
# 
# apyori library expectes the transactions as a list 
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)
# transactionslist
transactions = []
# populate the transactions in the list as a list of list
# the second loop get all the products of each transaction
# max number of columns in a transaction is 20
# All the element in the list provided to apyori library must be strings
for i in range(0, 7501):
  transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])


# Training Apriori on the dataset
# Training the Apriori model on the dataset
#
# We use the apriori function from the apyori library
from apyori import apriori
# we take in consideration only rules that have support >= min_support
# min_confidence: we start at 80% as rule of thumbs
# min_lift: a good value is 3
# min_length, max_length: number of products on both sides of the rule
rules = apriori(transactions = transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2, max_length = 2)

# VISULISING THE RESULTS

# Displaying the first results coming directly from the output of the apriori function
# 
# we put the rules in a list and show the list
results = list(rules)
results

# Putting the results well organised into a Pandas DataFrame
def inspect (results):
  lhs: list[str] = [tuple(result[2][0][0])[0] for result in results]
  rhs: list[str] = [tuple(result[2][0][1])[0] for result in results]
  supports: list[float] = [result[1] for result in results]
  confidences: list[float] = [result[2][0][2] for result in results]
  lifts: list[float] = [result[2][0][3] for result in results]
  return list(zip(lhs, rhs, supports, confidences, lifts))
  
resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])

# Displaying the results non sorted
resultsinDataFrame

# Displaying the results sorted by descending lifts
resultsinDataFrame.nlargest(n = 10, columns = 'Lift')