import numpy as np
from collections import defaultdict


def kfold_split(num_objects, num_folds):
    array = np.arange(num_objects)
    reset = [[0] * 2 for i in range(num_folds)]
    block = np.arange(num_objects // num_folds)
    for i in range(num_folds-1):
        reset[i][1] = array[block]
        reset[i][0] = np.delete(array, array[block])
        block += num_objects // num_folds
    block = np.arange(num_objects - num_objects // num_folds - num_objects % num_folds, num_objects)
    reset[num_folds - 1][0] = np.delete(array, array[block])
    reset[num_folds - 1][1] = array[block]
    return reset


def knn_cv_score(X_train, y_train, parameters, r2_score, folds, knn_class):
    res = dict()
    for normalizer in parameters['normalizers']:
        for n_neighbors in parameters['n_neighbors']:
            for metrics in parameters['metrics']:
                for weights in parameters['weights']:
                    average_metrics = 0.0
                    for data in folds:
                        train_data = X_train[data[0]]
                        train_mask = y_train[data[0]]
                        test_data = X_train[data[1]]
                        test_mask = y_train[data[1]]
                        scaler = normalizer[0]
                        if scaler is None:
                            scaled_train_data = train_data
                            scaled_test_data = test_data
                        else:
                            scaler.fit(train_data)
                            scaled_train_data = scaler.transform(train_data)
                            scaled_test_data = scaler.transform(test_data)
                        clf = knn_class(n_neighbors=n_neighbors, weights=weights, metric=metrics)
                        clf.fit(scaled_train_data, train_mask)
                        y_predict = clf.predict(scaled_test_data)
                        average_metrics += r2_score(test_mask, y_predict)
                    res[normalizer[1], n_neighbors, metrics, weights] = average_metrics / len(folds)
    return res
