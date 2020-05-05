#load data
import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# PRE-PROCESSING
import nltk
from nltk.corpus import stopwords
print('before: ' + train.text[7])
print('before: ' + train.text[8])
print('before: ' + train.text[15])

# remove stopwords
for i in stopwords.words('english'):
    train = train.replace(to_replace=r'\b%s\b\s'%i, value=" ",regex=True)
    test = train.replace(to_replace=r'\b%s\b\s'%i, value=" ",regex=True)
print('after stopwords: ' + train.text[7])
print('after stopwords: ' + train.text[8])
print('after stopwords: ' + train.text[15])

# remove hyperlinks
train = train.replace(to_replace=r'http\S+', value="", regex=True)
test = test.replace(to_replace=r'http\S+', value="", regex=True)
print('after hyperlinks: ' + train.text[7])
print('after HL: ' + train.text[8])
print('after HL: ' + train.text[15])

# remove mentions
train = train.replace(to_replace=r'@\S+', value="", regex=True)
test = test.replace(to_replace=r'@\S+', value="", regex=True)
train=train.replace(to_replace=r'Err:\S+', value='error ', regex=True)
test=test.replace(to_replace=r'Err:\S+', value='error ', regex=True)
print('after mentions: ' + train.text[7])
print('after m: ' + train.text[8])
print('after m: ' + train.text[15])

#replace punctuation
train = train.replace(to_replace=r'\...', value=' ', regex=True)
test = test.replace(to_replace=r'\...', value=' ', regex=True)
train = train.replace(to_replace=r'[\'\^\\,@\‘?!\.$%_:\-“’“”\#\/\*]', value='', regex=True)
test = test.replace(to_replace=r'[\'\^\\,@\‘?!\.$%_:\-“’“”\#\/\*]', value='', regex=True)
print('after punctuation: ' + train.text[7])
print('after p: ' + train.text[8])
print('after p: ' + train.text[15])

#make lowercase
train.text = train.text.apply(lambda x: x.lower())
test.text = test.text.apply(lambda x: x.lower())

#replace spaces
train=train.replace(to_replace='\n', value=' ',regex=True)
test=test.replace(to_replace='\n', value=' ',regex=True)
train = train.replace(to_replace=r'    ', value=' ', regex=True)
test = test.replace(to_replace=r'    ', value=' ', regex=True)
train = train.replace(to_replace=r'   ', value=' ', regex=True)
test = test.replace(to_replace=r'   ', value=' ', regex=True)
train = train.replace(to_replace=r'  ', value=' ', regex=True)
test = test.replace(to_replace=r'  ', value=' ', regex=True)
print('after spaces: ' + train.text[7])
print('after s: ' + train.text[8])
print('after s: ' + train.text[15])

#GET AVERAGE GLOVE VECTOR
import spacy
import numpy as np
import re 
nlp = spacy.load("en_core_web_md")

train_feature = np.array([nlp(x.lower()).vector for x in train.text])
# add keyword if available
#train.keyword = train.keyword.apply(lambda x: x if pd.isnull(x) else ' '.join(re.findall("[a-zA-Z!]+", x)))
#for i in range(len(train_feature)):
#    if not pd.isnull(train.keyword[i]):
#        train_feature[i] = (train_feature[i]+nlp(train.keyword[i].lower()).vector)/2   

test_feature = np.array([nlp(x.lower()).vector for x in test.text])
#test.keyword = test.keyword.apply(lambda x: x if pd.isnull(x) else ' '.join(re.findall("[a-zA-Z!]+", x)))
#for i in range(len(test_feature)):
#    if not pd.isnull(test.keyword[i]):
#        test_feature[i] = (test_feature[i]+nlp(test.keyword[i].lower()).vector)/2   

# TRAIN SVM
from sklearn.model_selection import cross_val_score
from sklearn import svm
clf = svm.SVC()
scores = cross_val_score(clf, train_feature, train.target, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

clf = svm.SVC().fit(train_feature, train.target)
pred = pd.DataFrame({'id':test.id,'target': clf.predict(test_feature)})
pred.to_csv('submission.csv',index=False)


