# NeuronsUnchained

Implementing Neural Networks from scratch and performing a classification & regression task to understand internal working on neural network. 

The library will be made up of documented classes and functions that allow users to easily construct
a neural network with an arbitrary number of layers and nodes. Through implementing
this library, we will understand more clearly the atomic components that make up a
basic neural network.

## The `Layer` Class

For the layers we create in this assignment, it is worth it to create a parent class
named `Layer` which defined the forward and backward functions that are used by all layers.
In this way, we can take advantage of polymorphism to easily compute the forward and
backward passes of the entire network.

## `Linear` Layer

A class that implements a linear layer. The class  inherits the `Layer` class
and implement both a `forward` and `backward` function.
For a given input, the forward pass is computed as

$$
f(\mathbf{x}; \mathbf{w}) = \mathbf{x} \mathbf{w}^T + \mathbf{b}.
$$

Here, $\mathbf{x} \in \mathbb{R}^{n \times d}$, $\mathbf{w} \in \mathbb{R}^{h \times d}$,
and $\mathbf{b} \in \mathbb{R}^h$,
where $n$ is the number of samples, $d$ is the number of input features, and $h$
is the number of output features.

The backward pass computes the gradient with respect to the weights and bias:

$$
\frac{d}{d\mathbf{w}} f(\mathbf{x}; \mathbf{w}) = \mathbf{x}\\
\frac{d}{d\mathbf{w}} f(\mathbf{x}; \mathbf{w}) = \mathbf{1}
$$

This is then multiplied with the gradients computed by the layer ahead of this one.

Since there may be multiple layers, it  additionally computes $\frac{df}{d\mathbf{x}}$
to complete a chain of backward passes.

## `Sigmoid` Function

A class that implements the logistic sigmoid function.
The class inherits the `Layer` class and implement both
`forward` and `backward` functions.

It is useful to store the output of forward pass of this layer
as a class member so that it may be reused when calling `backward`.

## Rectified Linear Unit (ReLU)

A class that implements the rectified linear unit.
The class inherits the `Layer` class and implement both
`forward` and `backward` functions.

## Binary Cross-Entropy Loss

A class that implements binary cross-entropy loss. This will be used when classifying the XOR problem.
The class inherits the `Layer` class and implement both
`forward` and `backward` functions.

## The `Sequential` Class

In order to create a clean interface that includes multiple layers, we create
a class that contains a list of layers which make up the network.
The `Sequential` class will contain a list of layers.
New layers can be added to it by appending them to the current list.
This class will also inherit from the `Layer` class so that it can call forward
and backward as required.

## Saving and Loading

We implement a weight saving and loading feature for a constructed network such that all
model weights can be saved to and loaded form a file. This will enable trained models to
be stored and shared. We store the weights in npz format and load them as usual for inference puroposes.
