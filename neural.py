import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, num_iterations=2000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        # Initialisation des poids et du biais à zéro
        num_features = X.shape[1]
        self.weights = np.zeros(num_features)
        self.bias = 0

        # Descente de gradient pour ajuster les poids et le biais
        for i in range(self.num_iterations):
            # Calcul de l'activation (prédiction) pour les exemples actuels
            z = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(z)

            # Calcul du gradient pour les poids et le biais
            dw = (1 / X.shape[0]) * np.dot(X.T, (y_pred - y))
            db = (1 / X.shape[0]) * np.sum(y_pred - y)

            # Mise à jour des poids et du biais
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            if(i == 100 or i == 1000):
                print(i, ": ",self.weights, self.bias)

    def predict(self, X):
        # Prédiction des étiquettes pour de nouvelles données
        z = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(z)
        return (y_pred >= 0.5).astype(int)