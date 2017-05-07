import json
import sys
import os
import pickle
import sklearn
import random
import numpy
import socket
import threading
import argparse
from random import shuffle
import pandas as pd
import numpy as np
from imblearn.pipeline import Pipeline

import socketserver
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_pipeline

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from imblearn.under_sampling import RandomUnderSampler
from sklearn.naive_bayes import GaussianNB


DEBUG = False

random.seed(123)


class RF(object):   

    def learn(self, dataFile, splitRatio):
        print("Learning ...")
        json_db = "data/" + dataFile + ".rf.json"
        json_data = [json.loads(fing) for fing in open(json_db)]
        fingerprints = []
        for data in json_data:
            map = {}
            map['location'] = data['location']
            map['timestamp'] = data['timestamp']
            for fingerp in data['wifi-fingerprint']:
                map[fingerp['mac']] = fingerp['rssi']
            fingerprints.append(map)

        fingerprints = pd.DataFrame(fingerprints)
        fingerprints.fillna(-100, inplace=True)

        features = fingerprints.columns.values.tolist()
        features.remove('location')
        features.remove('timestamp')
        # features = ['34:57:60:49:dc:dc', '36:57:60:49:dc:dc', 'c8:91:f9:e7:7a:ce']
        locations = np.unique(fingerprints.location)

        X = fingerprints[features] * -1
        y = fingerprints.location

        pipe_clf = Pipeline(
            [('rnds', RandomUnderSampler()),
            ('clf', RandomForestClassifier(bootstrap=False, class_weight=None, criterion='gini', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=47, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False))
            # ('clf', KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='cityblock', metric_params=None, n_neighbors=2, p=1, weights='distance'))
            # ('clf', GaussianNB(priors=None))
            # ('clf', SVC(C=0.20000000000000001, cache_size=200, class_weight=None, coef0=0, decision_function_shape=None, degree=2, gamma=0.20000000000000001, kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False))
            ]
        )
        pipe_clf.fit(X, y)
        with open('data/' + dataFile + '.rf.pkl', 'wb') as fid:
            pickle.dump([pipe_clf, features, locations], fid)

    def classify(self, groupName, fingerpintFile):
        with open('data/' + groupName + '.rf.pkl', 'rb') as pickle_file:
            [clf, features, locations] = pickle.load(pickle_file)

        data = {}
        with open(fingerpintFile, 'r') as f_in:
            for line in f_in:
                data = json.loads(line)
        if len(data) == 0:
            return

        fingerprints = {}
        for f in features:
            fingerprints[f] = -100
        fingerprints_count=0
        for signal in data['wifi-fingerprint']:
            # Only add the mac if it exists in the learning model
            if signal['mac'] in features:
                fingerprints[signal['mac']] = signal['rssi']
                fingerprints_count += 1
        print("Features len is %d and fingerprints len is %d" % (len(features), fingerprints_count))

        prediction = clf.predict_proba(pd.DataFrame([fingerprints]))
        predictionJson = {}
        for i in range(len(prediction[0])):
            predictionJson[locations[i]] = prediction[0][i]
        print("Prediction json is %s | prediction is %s" % (predictionJson, clf.predict(pd.DataFrame([fingerprints]))))
        return predictionJson


class EchoRequestHandler(socketserver.BaseRequestHandler):

    def handle(self):
        # Echo the back to the client
        data = self.request.recv(1024)
        data = data.decode('utf-8').strip()
        print("received data:'%s'" % data)
        group = data.split('=')[0].strip()
        filename = data.split('=')[1].strip()
        payload = "error".encode('utf-8')
        if len(group) == 0:
            self.request.send(payload)
            return
        randomF = RF()
        if len(filename) == 0:
            payload = json.dumps(randomF.learn(group, 0.9)).encode('utf-8')
        else:
            payload = json.dumps(
                randomF.classify(
                    group,
                    filename +
                    ".rftemp")).encode('utf-8')
        self.request.send(payload)
        return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        help="select the port to run on")
    parser.add_argument("-g", "--group", type=str, help="select a group")
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        help="select a file with fingerprints")
    parser.add_argument("-d", "--debug", help="debug mode")
    args = parser.parse_args()
    DEBUG = args.debug
    if args.port is not None:
        socketserver.TCPServer.allow_reuse_address = True
        address = ('localhost', args.port)  # let the kernel give us a port
        server = socketserver.TCPServer(address, EchoRequestHandler)
        ip, port = server.server_address  # find out what port we were given
        server.serve_forever()
    elif args.file is not None and args.group is not None:
        randomF = RF()
        print(randomF.classify(args.group, args.file))
    elif args.group is not None:
        randomF = RF()
        print(randomF.learn(args.group, 0.5))
    else:
        print("""Usage:

To just run as TCP server:

	python3 rf.py --port 5009

To just learn:

	python3 rf.py --group GROUP

To classify

	python3 rf.py --group GROUP --file FILEWITHFINGERPRINTS
""")
