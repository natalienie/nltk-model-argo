import pickle
import nltk
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import BernoulliNB
from nltk.tokenize import word_tokenize
import random

pickle_in_d = open('pickled_algo_documents.pickle', 'rb')
documents = pickle.load(pickle_in_d)

pickle_in_f = open('pickled_algo_features.pickle', 'rb')
word_features = pickle.load(pickle_in_f)

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

featuresets = [(find_features(rev), category) for (rev, category) in documents]
random.shuffle(featuresets)
testing_set = featuresets[10000:]
training_set = featuresets[:10000]

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print('BernoulliNB_classifier accuracy percent:', (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)
save_BernoulliNB_classifier = open('pickled_BernoulliNB_classifier.pickle', 'wb')
pickle.dump(BernoulliNB_classifier, save_BernoulliNB_classifier)
save_BernoulliNB_classifier.close()
