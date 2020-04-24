# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#load pre-trained glove vectors
#embeddings_dict = {}
#with open("G:/My Drive/DublinAI/Mini Projects/nlp-getting-started/glove.6B/glove.6B.50d.txt", 'r', encoding="utf-8") as f:
#    for line in f:
#        values = line.split()
#        word = values[0]
#        vector = np.asarray(values[1:], "float32")
#        embeddings_dict[word] = vector
    
#load data
import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#tokenize data
#train.text = train.text.apply(lambda x: nltk.regexp_tokenize(x.lower(), r'\w+'))
#test.text = test.text.apply(lambda x: nltk.regexp_tokenize(x.lower(), r'\w+'))

# remove stopwords
#train.text=[w for w in train.text if w not in stopwords.words('english')]
#test.text=[w for w in test.text if w not in stopwords.words('english')]

#get average glove vector
import spacy
import numpy as np
import re 
nlp = spacy.load("en_core_web_md")

train.text = train.text.apply(lambda x: ' '.join(re.findall("[a-zA-Z!]+", x)))
train_feature = np.array([nlp(x.lower()).vector for x in train.text])

# add keyword if available
#train.keyword = train.keyword.apply(lambda x: x if pd.isnull(x) else ' '.join(re.findall("[a-zA-Z!]+", x)))
#for i in range(len(train_feature)):
#    if not pd.isnull(train.keyword[i]):
#        train_feature[i] = (train_feature[i]+nlp(train.keyword[i].lower()).vector)/2   

test.text = test.text.apply(lambda x: ' '.join(re.findall("[a-zA-Z!]+", x)))
test_feature = np.array([nlp(x.lower()).vector for x in test.text])
#test.keyword = test.keyword.apply(lambda x: x if pd.isnull(x) else ' '.join(re.findall("[a-zA-Z!]+", x)))
#for i in range(len(test_feature)):
#    if not pd.isnull(test.keyword[i]):
#        test_feature[i] = (test_feature[i]+nlp(test.keyword[i].lower()).vector)/2   

from sklearn.model_selection import cross_val_score
from sklearn import svm
clf = svm.SVC()
scores = cross_val_score(clf, train_feature, train.target, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

clf = svm.SVC().fit(train_feature, train.target)
pred = pd.DataFrame({'id':test.id,'target': clf.predict(test_feature)})
pred.to_csv('submission.csv',index=False)

# visualise data
# real_train = train[train.target==1]
# not_real_train = train[train.target==0]

# a = np.array([i for i in set(not_real_train.keyword)])
# b = np.array([sum(not_real_train.keyword == i) for i in a])
# c = np.argsort(-1*b)

# plt.rcdefaults()
# fig, ax = plt.subplots()
# y_pos = np.arange(10)

# ax.barh(y_pos, b[c[0:10]], align='center')
# ax.set_yticks(y_pos)
# ax.set_yticklabels(a[c[0:10]])
# ax.invert_yaxis()
# plt.show()

