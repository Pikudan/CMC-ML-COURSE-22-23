import numpy as np


class MinMaxScaler:
    def fit(self, data):
        self.max_val = np.max(data, axis=0)
        self.min_val = np.min(data, axis=0)

    def transform(self, data):
        X_std = (data - self.min_val) / (self.max_val - self.min_val)
        return X_std


class StandardScaler:
    def fit(self, data):
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)

    def transform(self, data):
        return (data - self.mean) / self.std
