import numpy as np

class Layer:
    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

class Linear(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size)

    def forward(self, inputs):
        self.inputs = inputs
        return np.dot(inputs, self.weights.T) + self.bias

    def backward(self, grad):
        self.grad_weights = np.dot(grad.T, self.inputs)
        self.grad_bias = np.sum(grad, axis=0)
        return np.dot(grad, self.weights)

class Sigmoid(Layer):
    def forward(self, inputs):
        self.output = 1 / (1 + np.exp(-inputs))
        return self.output

    def backward(self, grad):
        return grad * self.output * (1 - self.output)

class ReLU(Layer):
    def forward(self, inputs):
        self.inputs = inputs
        return np.maximum(0, inputs)

    def backward(self, grad):
        return grad * (self.inputs > 0)

class BinaryCrossEntropyLoss(Layer):
    def forward(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets
        return -np.mean(targets * np.log(inputs) + (1 - targets) * np.log(1 - inputs))

    def backward(self):
        return (self.inputs - self.targets) / (self.inputs * (1 - self.inputs))

class MeanSquaredError(Layer):
    def forward(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets
        return np.mean((inputs - targets) ** 2) / 2

    def backward(self):
        return (self.inputs - self.targets) / self.inputs.size

class Sequential(Layer):
    def __init__(self, layers):
        self.layers = layers

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
    
    def save_weights(self, filename):
        weights = [layer.weights for layer in self.layers if hasattr(layer, 'weights')]
        biases = [layer.bias for layer in self.layers if hasattr(layer, 'bias')]
        np.savez(filename, *(weights + biases))

    def load_weights(self, filename):
        """Loads model weights from a file."""
        weights = np.load(filename)
        for layer, (w, b) in zip(self.layers, weights.items()):
            try:
                layer.weights = w
                layer.bias = b
            except AttributeError:  # Skip layers without weights
                pass

# XOR problem data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Define the neural network architecture
model = Sequential([
    Linear(2, 2),
    ReLU(),
    Linear(2, 1),
    Sigmoid()
])

# Training loop
learning_rate = 0.1
epochs = 1000
loss_fn = BinaryCrossEntropyLoss()

for epoch in range(epochs):
    # Forward pass
    outputs = model.forward(X)
    loss = loss_fn.forward(outputs, y)
    
    # Backward pass
    grad = loss_fn.backward()
    model.backward(grad)
    
    # Gradient descent
    for layer in model.layers:
        if isinstance(layer, Linear):
            layer.weights -= learning_rate * layer.grad_weights
            layer.bias -= learning_rate * layer.grad_bias
    
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss}')

# Test the trained model
predictions = model.forward(X)

# Save weights
model.save_weights("model_weights.npz")
print("Predictions after training:")
print(predictions)
