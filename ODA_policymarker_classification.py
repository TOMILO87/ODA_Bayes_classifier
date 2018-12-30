import pandas as pd
import numpy as np
from nltk.corpus import stopwords

#nltk.download('stopwords')
np.random.seed(1)
pd.options.display.max_colwidth = 800 # otherwise project descriptions are cut-off (don't want them too long either)

# import CRS-data
df_2017 = pd.read_csv("CRS_2017_181229.csv", header = 0)
df_2016 = pd.read_csv("CRS_2016_181229.csv", header = 0)
df_2015 = pd.read_csv("CRS_2015_181229.csv", header = 0)
df = pd.concat([df_2017, df_2016, df_2015])

# replace NA with zero
df = df.fillna(0)

# Misc information about CRS data
headers = list(df)
table = pd.pivot_table(df, values='Value', index = ['YEAR'], aggfunc=np.sum).reset_index().rename_axis(None, axis = 1)
n = len(df.index)
print("Columns", headers)
print("Total number of contributions", n)
print("Total outcome per year", table)

# the policy objective we want to study
objective = 'Trade Development'

# get project descriptions for contributions whose disbursements > 0 and policy objective marker = 1, 2
df_12 = df.loc[df[objective].isin([1, 2]) & df['Value'] != 0][['Long Description']]
n_12 = len(df_12.index)

# get project descriptions for contributions whose disbursements > 0 and policy objective marker = 0
df_0 = df.loc[df[objective].isin([0]) & df['Value'] != 0][['Long Description']]
n_0 = len(df_0.index)

# split data into training (95%) and test samples (5%)
msk_12 = np.random.rand(n_12) < 0.95
df_train_12 = df_12[msk_12]
df_test_12 = df_12[~msk_12]
n_train_12 = len(df_train_12.index)
n_test_12 = n_12 - n_train_12
msk_0 = np.random.rand(n_0) < 0.95
df_train_0 = df_0[msk_0]
df_test_0 = df_0[~msk_0]
n_train_0 = len(df_train_0.index)
n_test_0 = n_0 - n_train_0

# import stop words to remove when comparing project descriptions
stopwords_all = set()
for i in ['english', 'spanish', 'french', 'german']:
    stopwords_all = stopwords_all.union(set(stopwords.words(i)))
stopwords_all = list(stopwords_all)

## Find key words to separate 1, 2 from 0 ##
# storage vector and parameters
key_words = [] # word to separate 1, 2 from 0 will be added to this storage vector
k_12 = 2 # number of contributions from 1, 2 to sample per comparison
k_0 = 25 # number of contributions from 0 to sample per comparison
count = 0 # keep track of number of times samples that are compared
n_key_words = 800 # number of key words we want to have likelihood for and use for classification

while len(key_words) < n_key_words:
    sample_12 = [] #reset sample
    sample_0 = []

    # convert pandas objects to list of unique words and add to sample (chosen randomly)
    check = []  # used to check that duplicate descriptions aren't sampled
    for i in range(k_12):
        rand = list(set(df_train_12.iloc[[np.random.randint(0, n_train_12)]].to_string().split()))
        while len(rand) in check:
            rand = list(set(df_train_12.iloc[[np.random.randint(0, n_train_12)]].to_string().split()))
        check.append(len(rand))
        sample_12.append(rand)
    check = []
    for i in range(k_0):
        rand = list(set(df_train_0.iloc[[np.random.randint(0, n_train_0)]].to_string().split()))
        while len(rand) in check:
            rand = list(set(df_train_0.iloc[[np.random.randint(0, n_train_0)]].to_string().split()))
        check.append(len(rand))
        sample_0.append(rand)

    # make into lower cases and remove ',','.' etcetera
    for i in range(len(sample_12)):
        sample_12[i] = [x.lower() for x in sample_12[i]]
        for j in [',', '.', '(', ')', '-', ':', "'", '=', '/']:
            sample_12[i] = [x.replace(j, '') for x in sample_12[i]]
    for i in range(len(sample_0)):
        sample_0[i] = [x.lower() for x in sample_0[i]]
        for j in [',', '.', '(', ')', '-', ':', "'", '=', '/']:
            sample_0[i] = [x.replace(j, '') for x in sample_0[i]]

    # find common words in 1, 2 sample
    common_12 = []
    for i in range(1, len(sample_12)):
        if i == 1:
            common_12 = list(set(sample_12[i-1]).intersection(set(sample_12[i])))
        else:
            common_12 = list(set(common_12).intersection(set(sample_12[i])))

    # remove words which is also in 0 sample
    for i in sample_0:
        common_12 = [x for x in common_12 if (x not in i)]

    # remove stop words or integers
    common_12 = [x for x in common_12 if x not in stopwords_all]
    try:
        common_12 = [x for x in common_12 if not (x.isdigit() or x[0] == '-' and x[1:].isdigit())]
    except IndexError:
        pass

    # add new keywords
    key_words = key_words + list(set(common_12) - set(key_words))

    count += 1

print("Comparisons done before finding key words:", count)

## Train naive Bayes classifier ##
Y_12 = [] #will contain information about presence of key words
for i in range(n_train_12):
    y_i = np.ones((1, len(key_words)))
    a = set(df_train_12.iloc[[i]].to_string().split()) # project description i:th item
    a = [x.lower() for x in a]
    for j in [',', '.', '(', ')', '-', ':', "'", '=', '/']:
        a = [x.replace(j, '') for x in a]
    no_match = list(set(key_words) - set(a))

    for j in no_match:
        y_i[0, key_words.index(j)] = 0 # if no match in project descriptions indicator variable is 0
    Y_12.append(y_i)
theta_12 = np.mean(Y_12, axis = 0)

Y_0 = [] #will contain information about presence of key words
for i in range(n_train_0):
    y_i = np.ones((1, len(key_words)))
    a = set(df_train_0.iloc[[i]].to_string().split())  # project description i
    a = [x.lower() for x in a]
    for j in [',', '.', '(', ')', '-', ':', "'", '=', '/']:
        a = [x.replace(j, '') for x in a]
    no_match = list(set(key_words) - set(a))

    for j in no_match:
        y_i[0, key_words.index(j)] = 0 # if no match in project descriptions indicator variable is 0
    Y_0.append(y_i)
theta_0 = np.mean(Y_0, axis = 0)

# for numerical stability we assign a very low probability (0.0001) if estimated probability of a word is 0
eps = 0.0001
theta_12[theta_12 == 0] = eps
theta_0[theta_0 == 0] = eps

# collect sorted key words and probabilities in 1,2 and 0
key_words_prob = {}
index = 0
for i in key_words:
    key_words_prob[i] = [theta_12[0][index]/theta_0[0][index], theta_12[0][index]*100, theta_0[0][index]*100]
    index += 1
key_words_prob = sorted(key_words_prob.items(), key=lambda kv: kv[1], reverse=True)
print('Key words and probabilities for contributions whose policy marker is 1 or 2, relative to 0 ('
      'second and third values are probabilities of each class)\n', key_words_prob)

# prior probability that a contribution has marker 1 or 2
prob_prior = n_train_12 / (n_train_12 + n_train_0)

## Test naive Bayes classifier ##

# posterior probabilities class 1, 2
score_12 = [] # keep track of correct classifications test 1, 2
PROB_POST_12 = [] # keep track of posterior probabilities test 1, 2
for i in range(n_test_12):
    y_i = np.ones((1, len(key_words)))
    a = set(df_test_12.iloc[[i]].to_string().split())  # project description i
    a = [x.lower() for x in a]
    for j in [',', '.', '(', ')', '-', ':', "'", '=', '/']:
        a = [x.replace(j, '') for x in a]
    no_match = list(set(key_words) - set(a))

    for j in no_match:
        y_i[0, key_words.index(j)] = 0

    # likelihood of marker 1, 2
    l_12 = np.multiply(np.power(theta_12, y_i), np.power((1-theta_12), (1-y_i)))
    l_12 = np.prod(l_12[0])

    # likelihood of marker 0
    l_0 = np.multiply(np.power(theta_0, y_i), np.power((1 - theta_0), (1 - y_i)))
    l_0 = np.prod(l_0[0])

    prob_post = l_12 * prob_prior/(l_12 * prob_prior + l_0 * (1-prob_prior))
    PROB_POST_12.append(prob_post)
    if prob_post > 0.5:
        score_12.append(1)
    else:
        score_12.append(0)

# posterior probabilities class 0
score_0 = [] # keep track of correct classifications test 0
PROB_POST_0 = [] # keep track of posterior probabilities test 0
for i in range(n_test_0):
    y_i = np.ones((1, len(key_words)))
    a = set(df_test_0.iloc[[i]].to_string().split())  # project description i
    a = [x.lower() for x in a]
    for j in [',', '.', '(', ')', '-', ':', "'", '=', '/']:
        a = [x.replace(j, '') for x in a]
    no_match = list(set(key_words) - set(a))

    for j in no_match:
        y_i[0, key_words.index(j)] = 0

    # likelihood of marker 1, 2
    l_12 = np.multiply(np.power(theta_12, y_i), np.power((1-theta_12), (1-y_i)))
    l_12 = np.prod(l_12[0])

    # likelihood of marker 0
    l_0 = np.multiply(np.power(theta_0, y_i), np.power((1 - theta_0), (1 - y_i)))
    l_0 = np.prod(l_0[0])

    prob_post = l_12 * prob_prior/(l_12 * prob_prior + l_0 * (1-prob_prior))
    PROB_POST_0.append(prob_post)
    if prob_post < 0.5: # we want likelihood of 1, 2 to be low
        score_0.append(1)
    else:
        score_0.append(0)

## Results ##
print("Number of train contributions whose policy marker is 1 or 2, and 0, respectively:", n_train_12, ',', n_train_0)
print("Number of test contributions whose policy marker 1 or 2, and 0, respectively:", n_test_12, ',', n_test_0)
print("Prior probability for for policy marker 1, 2:", prob_prior)
print("Average posterior probability policy marker 1 or 2 for contributions whose policy marker is 1 or 2:", np.mean(PROB_POST_12))
print("Average correct predictions for contributions whose policy marker is 1 or 2:", np.mean(score_12))
print("Average posterior probability policy marker 1 or 2 for contributions whose policy marker is 0:", np.mean(PROB_POST_0))
print("Average correct predictions for contributions whose policy marker is 0:", np.mean(score_0))
share_test_12 = n_test_12/(n_test_12+n_test_0)
print("Overall share of correct predictions:", share_test_12*np.mean(score_12) + (1-share_test_12)*np.mean(score_0))

# Get examples of correct and incorrect classifications #
correct_12 = [] # We want to be able to analyze correct classified contributions
miss_12 = [] # We want to be able to analyze miss-classified contributions
correct_0 = [] # We want to be able to analyze correct classified contributions
miss_0 = [] # We want to be able to analyze miss-classified contributions
while len(correct_12) < 3: # We only want to look at 3 random examples of each type
    index = np.random.randint(0, n_test_12)
    if score_12[index] == 1:
        correct_12.append([df_test_12.iloc[[index]].to_string().split(), PROB_POST_12[index]])
while len(miss_12) < 3: # We only want to look at 3 random examples of each type
    index = np.random.randint(0, n_test_12)
    if score_12[index] == 0:
        miss_12.append([df_test_12.iloc[[index]].to_string().split(), PROB_POST_12[index]])
while len(correct_0) < 3: # We only want to look at 3 random examples of each type
    index = np.random.randint(0, n_test_0)
    if score_0[index] == 1:
        correct_0.append([df_test_0.iloc[[index]].to_string().split(), PROB_POST_0[index]])
while len(miss_0) < 3: # We only want to look at 3 random examples of each type
    index = np.random.randint(0, n_test_0)
    if score_0[index] == 0:
        miss_0.append([df_test_0.iloc[[index]].to_string().split(), PROB_POST_0[index]])

print("\n3 examples of correct classified contributions whose policy marker is 1 or 2 and posterior probabilities")
for i in correct_12:
    print(i, "\n")
print("\n3 examples of miss-classified contributions whose policy marker is 1 or 2 and posterior probabilities")
for i in miss_12:
    print(i, "\n")
print("\n3 examples of correct classified contributions whose policy marker is 0 and and posterior probabilities")
for i in correct_0:
    print(i, "\n")
print("\n3 examples of miss-classified contributions whose policy marker is 0 and and posterior probabilities")
for i in miss_0:
    print(i, "\n")