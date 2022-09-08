
import numpy as np
import matplotlib.pyplot as plt
import string
from sklearn.model_selection import train_test_split

# Load the data
poem_dir = 'C:/Users/ZAKY-PC/Jupyter_proj/datasets/poems'
edgar_data = poem_dir + '/edgar_allan_poe.txt'
robert_data = poem_dir + '/robert_frost.txt'

# Collect data into list
input_files = [edgar_data, robert_data]
input_texts = []
labels = []

# loop through each input files (enumerate is used to generate label where edgar = 0, rober =1)
for label, f in enumerate(input_files):
    print(f"{f[len(poem_dir)+1:-4]} corresponds to label {label}")

    for line in open(f):  # loop each line of the file
        # rstrip to remove the last line, lower used to convert lowercase text
        line = line.rstrip().lower()
        if line:
            line = line.translate(str.maketrans(
                '', '', string.punctuation))  # remove punctuation
            input_texts.append(line)
            labels.append(label)

# Split the data into train & test
train_text, test_text, y_train, y_test = train_test_split(input_texts, labels)

# convert text into integers
idx = 1  # set current index to 1
# initialize word2idx dictionary with one entry, which maps the unknown token to 0 (for testing dataset)
word2idx = {'unk': 0}

# populate word2idx
for text in train_text:
    tokens = text.split()  # split text into tokens using split function
    for token in tokens:
        if token not in word2idx:  # check if the token is already in the word2idx dictionary
            word2idx[token] = idx  # if not insert the new idx into the token
            idx += 1  # increment idx for the new inserted token

# Convert data into integer format
train_text_int = []
test_text_int = []

for text in train_text:
    tokens = text.split()  # split text to tekenize
    # map each token to its corresponding text
    line_as_int = [word2idx[token] for token in tokens]
    # append the list of integers to our list
    train_text_int.append(line_as_int)

for text in test_text:
    tokens = text.split()
    # not every word in test set appepars in train set, so use get function to ensure the return value is 0 (unknown token)
    line_as_int = [word2idx.get(token, 0) for token in tokens]
    test_text_int.append(line_as_int)

# initialize A and pi matrices - for both classes
V = len(word2idx)  # vocabulary size

# initialize A0,pi0, A1 and pi1 to ones for each initial word and transition
A0 = np.ones((V, V))
pi0 = np.ones(V)

A1 = np.ones((V, V))
pi1 = np.ones(V)

# Populates the A and pi with the appropriate counts from the train set


def compute_counts(text_as_int, A, pi):  # text for specific class, A and pi
    for tokens in text_as_int:
        last_idx = None  # help to keep track currrent populating A or pi
        for idx in tokens:  # loop though each index in list of tokens
            if last_idx is None:  # if last_idx is none, means its the first word sentence
                pi[idx] += 1  # populate with pi (first word)
            else:
                # The last word  exist, so count a transition
                # populate with A (transition from word to next)
                A[last_idx, idx] += 1

            # update last idx
            last_idx = idx  # update the last idx to idx, so taht it has the correct value on the next iteration of the loop


compute_counts([t for t, y in zip(train_text_int, y_train)
               if y == 0], A0, pi0)  # class edgar allan poe
compute_counts([t for t, y in zip(train_text_int, y_train)
               if y == 1], A1, pi1)  # class robert frost


# normalize A and pi so they are valid probability matrices
# since is is probability, it must sum = 1. Therefore A divide by the sum
A0 /= A0.sum(axis=1, keepdims=True)
pi0 /= pi0.sum()

A1 /= A1.sum(axis=1, keepdims=True)
pi1 /= pi1.sum()

# log A and pi since we din't need the actual probs
logA0 = np.log(A0)
logpi0 = np.log(pi0)

logA1 = np.log(A1)
logpi1 = np.log(pi1)

# Calculate the priors
# count how many samples belong to class 0 and 1 in train set
count0 = sum(y == 0 for y in y_train)
count1 = sum(y == 1 for y in y_train)
total = len(y_train)  # total number of sample

# calculate the prior probabilities class 0 and 1
p0 = count0 / total
p1 = count1 / total

# take the log of p0 and p1
logp0 = np.log(p0)  # 0.33
logp1 = np.log(p1)  # 0.67
# np.round(p0,2), np.round(p1,2) # return (0.33, 0.67) -


# Build classifier to tells which Markov model to use, since we have oone for every class
class Classifier:
    def __init__(self, logAs, logpis, logpriors):
        # save as  these attribute as the object
        self.logAs = logAs
        self.logpis = logpis
        self.logpriors = logpriors
        self.K = len(logpriors)  # number of classes

    def _compute_log_likelihood(self, input_, class_):
        # retreiving logA and logpi by indexing our list by the class
        logA = self.logAs[class_]
        logpi = self.logpis[class_]

        # loop through each index in the input
        last_idx = None  # to let us know  if we are in the begining of sentence
        logprob = 0  # hold the final answer

        for idx in input_:
            if last_idx is None:  # check if last_idx is None
                # it's the first token
                # the first token of the sentence thus index logpi
                logprob += logpi[idx]
            else:
                # state transition matrix (logA)
                logprob += logA[last_idx, idx]

            # update last_idx
            last_idx = idx  # increment last_idx for the next iteration of the loop
        return logprob

    def predict(self, inputs):  # list of sequences
        # initiaze an array to store prediction
        predictions = np.zeros(len(inputs))
        # loop through each input and enumerate to get the index
        for i, input_ in enumerate(inputs):

            # Compute the posterior for each class using a list of comprehension
            # loop using index c to loop through integers from 0 to k-1
            # output is a list of posteriors one for every class
            posteriors = [self._compute_log_likelihood(input_, c) + self.logpriors[c]
                          for c in range(self.K)]

            # predict using the list of posteriors
            pred = np.argmax(posteriors)

            # store the prediction in an array of prediction at index i
            predictions[i] = pred
        return predictions


# each array must be in order  since classes are assumed to index these lists
clf = Classifier([logA0, logA1], [logpi0, logpi1], [logp0, logp1])

# Predicit at train and test dataset
Ptrain = clf.predict(train_text_int)
print(f"Train acc: {np.mean(Ptrain == y_train)}")

Ptest = clf.predict(test_text_int)
print(f"Test acc: {np.mean(Ptest == y_test)}")

# Evaluate the score using confusion matric and f1score


def evaluate_cm_f1(y, pred):
    from sklearn.metrics import confusion_matrix, f1_score
    cm = confusion_matrix(y, pred)
    print(f"Confusion matrix: ")
    print(cm)
    print(f"F1 score: {np.round(f1_score(y, pred),3)}")


print('Train set:')
evaluate_cm_f1(y_train, Ptrain)

print('Test set:')
evaluate_cm_f1(y_test, Ptest)
