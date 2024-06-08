
import numpy as np

rng = np.random.default_rng(seed=0)

class Layer:
    def forward(self, x):
        '''
        Base class implementation of Forward layer
        '''
        raise NotImplementedError

    def backward(self, grad_output, learning_rate=None):
        '''
        Base class implementation of Backward layer
        '''
        raise NotImplementedError

class Linear(Layer):
    '''
    Implementation of Linear layer
    '''
    def __init__(self, input_size, output_size):
        '''
        Initializing parameters for linear layer 
        weights = input_size * output_size
        bias = output_size
        '''
        self.weights = rng.standard_normal((input_size, output_size))
        self.bias = np.zeros(output_size)

    def forward(self, x):
        '''
        Linear layer foward pass . Performs W^t X +b for given input X.
        '''
        self.input = x
        return np.dot(x, self.weights) + self.bias

    def backward(self, grad_output, learning_rate=0.1):
        '''
        Backward pass for linear layer . Previous layers gradient is taken as input. Returns the gradient from this layer.
        '''
        grad_input = np.dot(grad_output, self.weights.T)  
        grad_weights = np.dot(self.input.T, grad_output)
        grad_bias = np.sum(grad_output, axis=0)
        
        self.weights -= learning_rate * grad_weights
        self.bias -= learning_rate * grad_bias
        return grad_input

class Sigmoid(Layer):
    '''
    Sigmoid activatation function
    '''
    def forward(self, x):
        '''
        1/ (1+ exp(-x)) in the forward layer
        '''
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self, grad_output, learning_rate=None):
        '''
        Gradient of sigmoid is sigmoid*(1-sigmoid)
        '''
        return grad_output * self.output * (1 - self.output)

class Tanh(Layer):
    '''
    Tanh activatation function

    '''
    def forward(self, x):
        '''
        tanh(x) in the forward layer
        '''
        self.output = np.tanh(x)
        return self.output

    def backward(self, grad_output):
        '''
        1- tanh(x)**2
        '''
        return grad_output * (1 - self.output**2)



class ReLU(Layer):
    '''
    Rectified Linear unti activatation
    '''
    def forward(self, x):
        '''
        Maximum(0, num) will be given as output
        '''
        self.output = np.maximum(0, x)
        return self.output

    def backward(self, grad_output, learning_rate=None):
        '''
        Gradient of Relu is 1 for positive inputs and 0 for else
        '''
        return grad_output * (self.output>0)

class BinaryCrossEntropyLoss():
    '''
    Implementation of Binary Cross entropy
    '''

    def __init__(self, y_hat, y):
        '''
        Inputs-
          y_hat -> predictions
          y -> actual values

        '''
        self.y_hat = y_hat
        self.y = y

    def calculate_loss_gradient(self):
        '''
        Calculating loss and gradient wrt loss 
        '''
        self.output = -np.mean(self.y * np.log(self.y_hat) + (1 - self.y) * np.log(1 - self.y_hat))
        self.gradient_loss= (self.y_hat - self.y) / (self.y_hat * (1 - self.y_hat))
        return self.output, self.gradient_loss

class MeansquaredLoss():
    '''
    Implementation of Mean Squared Loss
    '''

    def __init__(self, y_hat, y):
        '''
        Inputs-
          y_hat -> predictions
          y -> actual values

        '''
        self.y_hat = y_hat
        self.y = y

    def calculate_loss_gradient(self):
        '''
        Calculating loss and gradient wrt loss 
        '''
        self.output = np.mean((self.y_hat-self.y)**2)
        self.gradient_loss= 2*(self.y_hat-self.y)/len(self.y)
        return self.output, self.gradient_loss

class Sequential(Layer):
    '''
    Wrapper layer for implementation of neural network
    '''
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        '''
        Implementation of forward pass for entire neural network
        '''
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad_output,learning_rate):
        '''
        Implementation of backward pass for entire neural network
        '''
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
        return grad_output

    def save_weights(self, filename):
        '''
        Implementation of saving trained weights
        Input- Filename 
        '''
        weights = [layer.weights for layer in self.layers if hasattr(layer, 'weights')]
        biases = [layer.bias for layer in self.layers if hasattr(layer, 'bias')]
        np.savez(filename, *weights, *biases)

    def load_weights(self, filename):
        '''
        Implementation of loading trained weights
        '''
        data = np.load(filename)
        idx = 0
        for layer in self.layers:
            if hasattr(layer, 'weights'):
                layer.weights = data[f'arr_{idx}']
                idx += 1
        for layer in self.layers:
            if hasattr(layer, 'bias'):
                layer.bias = data[f'arr_{idx}']
                idx += 1

