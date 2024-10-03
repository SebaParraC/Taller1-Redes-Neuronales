import numpy as np
from keras.datasets import mnist
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Load MNIST dataset using Keras
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape the data
X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)

# Normalize the data
X_train = X_train.astype('float64') / 255
X_test = X_test.astype('float64') / 255

# Apply Min-Max scaling
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Clase base para Capa
class Layer:
    def __init__(self):
        self.input = None
        self.output = None
    # computes the output Y of a layer for a given input X
    def forward_propagation(self, input):
        raise NotImplementedError
    # computes dE/dX for a given dE/dY (and update parameters if any)
    def backward_propagation(self, output_error, learning_rate):
        raise NotImplementedError
        
# Clase para capas densas (fully connected)
class FCLayer(Layer):
    def __init__(self, input_size, output_size, lambda_reg=0):
        # np.random.seed(1234)
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5
        self.lambda_reg = lambda_reg  # Se agrega parámetro coeficiente lambda de regularización L2

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)

        # Adicionamos acá el término de regularización L2 que castiga error en los pesos
        weights_error += self.lambda_reg * self.weights
def fit(self, X, y, batch_size=32):
    n_samples, n_features = X.shape
    self.weights = np.zeros(n_features)
    self.bias = 0

    for _ in range(self.n_iters):
        # Barajado aleatorio del mini-lote
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        # Procesamiento del mini-lote
        for i in range(0, n_samples, batch_size):
            # Seleccionamos el mini-lote actual
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            # Calculamos la salida del modelo
            linear_output = np.dot(X_batch, self.weights) + self.bias
            y_predicted = self._unit_step_function(linear_output)

            # Calculamos el error
            error = y_batch - y_predicted

            # Calculamos el gradiente promedio del mini-lote
            dw = np.mean(error * X_batch, axis=0)
            db = np.mean(error)

            # Realizamos el ajuste de pesos
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        # Ajuste del último mini-lote
        if i + batch_size > n_samples:
            X_batch = X_shuffled[i:]
            y_batch = y_shuffled[i:]

            # Calculamos la salida del modelo
            linear_output = np.dot(X_batch, self.weights) + self.bias
            y_predicted = self._unit_step_function(linear_output)

            # Calculamos el error
            error = y_batch - y_predicted

            # Calculamos el gradiente promedio del mini-lote
            dw = np.mean(error * X_batch, axis=0)
            db = np.mean(error)

            # Realizamos el ajuste de pesos
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

class MaxPoolingLayer(Layer):
    def __init__(self, pool_size):
        self.pool_size = pool_size

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.max(input_data.reshape(-1, self.pool_size, self.pool_size), axis=1)
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        error = np.zeros_like(self.input)
        for i in range(self.pool_size):
            for j in range(self.pool_size):
                error[:, i::self.pool_size, j::self.pool_size] = output_error
        return error

class AveragePoolingLayer(Layer):
    def __init__(self, pool_size):
        self.pool_size = pool_size

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.mean(input_data.reshape(-1, self.pool_size, self.pool_size), axis=1)
        return self.output
    def backward_propagation(self, output_error, learning_rate):
        error = np.zeros_like(self.input)
        for i in range(self.pool_size):
            for j in range(self.pool_size):
                error[:, i::self.pool_size, j::self.pool_size] = output_error / (self.pool_size ** 2)
        return error
class NoiseLayer(Layer):
    def __init__(self, noise_std):
        self.noise_std = noise_std

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = input_data + np.random.normal(0, self.noise_std, size=input_data.shape)
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        return output_error
class L1RegularizationLayer(Layer):
    def __init__(self, lambda_reg):
        self.lambda_reg = lambda_reg

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = input_data
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        error = output_error
        for i in range(self.input.shape[1]):
            if self.input[0, i] > 0:
                error[0, i] += self.lambda_reg
            elif self.input[0, i] < 0:
                error[0, i] -= self.lambda_reg
        return error
from sklearn.metrics import confusion_matrix, accuracy_score

# Necesitamos identificar cuantos nodos tiene nuestra entrada, y eso depende del tamaño de X.
entrada_dim = len(X_train[0])

# Crear instancia de Network
model = Network()

# Agregamos capas al modelo
model.add(FCLayer(entrada_dim, 128))
model.add(ActivationLayer(relu, relu_prime))
model.add(FCLayer(128, 64))
model.add(ActivationLayer(sigmoid, sigmoid_prime))
model.add(FCLayer(64, 10)) 
model.add(ActivationLayer(sigmoid, sigmoid_prime))

# Asignamos función de pérdida
model.use(bce, bce_prime)

# Entrenamos el modelo con datos de entrenamiento
model.fit(X_train, y_train, epochs=10, learning_rate=0.1)
# Usamos el modelo para predecir sobre los datos de prueba (validación)
y_hat = model.predict(X_test)

# Transformamos la salida en un vector one-hot encoded, es decir 0s y un 1. 
for i in range(len(y_hat)):
    y_hat[i] = np.argmax(y_hat[i][0])

# Reportamos los resultados del modelo
matriz_conf = confusion_matrix(y_test, y_hat)

print('MATRIZ DE CONFUSIÓN para modelo ANN')
print(matriz_conf,'\n')
print('La exactitud de testeo del modelo ANN es: {:.3f}'.format(accuracy_score(y_test,y_hat)))


