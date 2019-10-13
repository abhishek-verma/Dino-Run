import numpy as np
import math

shouldNormalize = True

def normalize(X, axis=-1, order=2):
    if not shouldNormalize :
        return X

    l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
    l2[l2 == 0] = 1
    return X / np.expand_dims(l2, axis)

class NaiveBayes():
    def fit(self, X, y):
        X = normalize(X)
        # print(X)
        self.X, self.y = X, y
        self.classes = np.unique(y)
        self.parameters = []
        for i, c in enumerate(self.classes):
            X_where_c = X[np.where(y == c)]
            self.parameters.append([])
            for col in X_where_c.T:
                parameters = {"mean": col.mean(), "var": col.var()}
                self.parameters[i].append(parameters)

    def _calculate_likelihood(self, mean, var, x):
        eps = 1e-4
        coeff = 1.0 / math.sqrt(2.0 * math.pi * var + eps)
        exponent = math.exp(-(math.pow(x - mean, 2) / (2 * var + eps)))
        return coeff * exponent

    def _calculate_prior(self, c):
        frequency = np.mean(self.y == c)
        return frequency

    def _classify(self, sample):
        posteriors = []
        for i, c in enumerate(self.classes):
            posterior = self._calculate_prior(c)
            for feature_value, params in zip(sample, self.parameters[i]):
                likelihood = self._calculate_likelihood(params["mean"], params["var"], feature_value)
                posterior *= likelihood
            posteriors.append(posterior)
        
        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        X = normalize(X)
        y_pred = [self._classify(sample) for sample in X]
        return y_pred