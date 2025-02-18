import random
import numpy as np


class NeuralNetwork:
    """
        Neural Network library, contains an input layer, hidden and output.
    """

    def __init__(self, input_nodes, hidden, output):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden
        self.output_nodes = output

        # Generate random weights and biases with row, col arrays
        # numbers are generated from a uniform distribution between -1, 1

        self.biases_hidden = np.random.normal(0.0, pow(self.hidden_nodes, -0.5),
                                              (
                                                  self.hidden_nodes, 1))
        self.biases_output = np.random.normal(0.0, pow(self.output_nodes, -0.5),
                                              (
                                                  self.output_nodes, 1))

        self.weights_ih = np.random.normal(0.0, pow(self.input_nodes, -0.5),
                                           (
                                               self.hidden_nodes,
                                               self.input_nodes))
        self.weights_ho = np.random.normal(0.0, pow(self.hidden_nodes, -0.5), (
            self.output_nodes, self.hidden_nodes))

    def feedforward(self, input_array):
        # Turn the input array into a single column vector
        input_array = np.reshape(input_array, (len(input_array), 1))
        hidden_inputs = np.dot(self.weights_ih,
                               input_array) + self.biases_hidden
        hidden_layer_calculation = self.sigmoid(hidden_inputs)
        output_inputs = np.dot(self.weights_ho,
                               hidden_layer_calculation) + self.biases_output
        output_calculation = self.sigmoid(output_inputs)
        return output_calculation

    def train(self, input_data, learning_rate):
        # Restructuring the given array into a single column vector
        data = np.reshape(input_data[0], (len(input_data[0]), 1))
        target = np.reshape(input_data[1], (len(input_data[1]), 1))

        # Layer node calculation vectors
        hidden_inputs = np.dot(self.weights_ih, data) + self.biases_hidden
        hidden_layer_calculations = self.sigmoid(hidden_inputs)
        output_inputs = np.dot(self.weights_ho,
                               hidden_layer_calculations) + self.biases_output
        output_calculations = self.sigmoid(output_inputs)

        # Error calculations for each node
        output_errors = np.subtract(output_calculations, target)
        # Here, we're finding the error of the hidden layer nodes by
        # multiplying each weight leading that leads from a hidden node to
        # an output node. The transpose is necessary since in our matrix,
        # we store the weights column-wise for each hidden neuron's connections
        # but we want to dot product for that specific neuron, so we transpose
        # to get all the weights for that neuron in a row.
        hidden_errors = np.dot(self.weights_ho.T, output_errors)

        # Gradient of cost, C, approximated by delta C * delta v.
        # Here, we compute a part of the gradient, i.e. we compute the partial
        # derivatives of cost, which is the output_errors, and the
        # derivative of the sigmoid, which is s(x) * (1-s(x)) but since our
        # output calculation already applied the sigmoid, we don't reapply it.
        # gradient C is equal to partial derivative of C with respect to w
        output_gradient = output_errors * output_calculations * (
                    1 - output_calculations)
        hidden_gradient = hidden_errors * hidden_layer_calculations * (
                1 - hidden_layer_calculations)

        # Here, we finish the gradient by completing the partial derivatives
        # that rise from the chain rule/power rule.
        # A key thing to note is that delta v ~= gradient * -learning rate
        # A gradient just relates the cost to the changes in weights and biases,
        # and since we know we want the change in cost to decrease, we are able
        # to pick delta "v" such that it is -learning rate* gradient, which
        # guarantees cost decreasing since gradient^2 is >= 0. Therefore,
        # w = w + delta w = w - learning rate * gradient C where
        # gradient C represents the partial derivative of C with respect to w
        delta_w = -learning_rate * np.dot(output_gradient,
                                          hidden_layer_calculations.T)
        delta_weights_ih = -learning_rate * np.dot(hidden_gradient, data.T)

        # Modify weights and biases
        # For the biases, a thing to note is that the partial derivative of
        # the matrix input into the sigmoid, z, with respect to the biases
        # is just 1. More explicitly, z = w*a + b, dz/db = 1
        self.weights_ho += delta_w
        self.biases_output += -learning_rate * output_gradient
        self.weights_ih += delta_weights_ih
        self.biases_hidden += -learning_rate * hidden_gradient

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def evaluate(self, input_array):
        result = self.feedforward(input_array)
        result = np.reshape(result, (1, len(result)))
        return np.argmax(result[0], 0)

    def StochasticGradientDescent(self, practice_data, epochs, learning_rate):
        x = 0
        for i in range(epochs):
            if x % 1000 == 0:
                print(x)
            random.shuffle(practice_data)
            for batch in practice_data:
                self.train(batch, learning_rate)
            x += 1

    def save(self, save_names_array):
        np.save(save_names_array[0], self.weights_ih)
        np.save(save_names_array[1], self.weights_ho)
        np.save(save_names_array[2], self.biases_output)
        np.save(save_names_array[3], self.biases_hidden)

    def load(self, save_names_array):
        self.weights_ih = np.load(save_names_array[0])
        self.weights_ho = np.load(save_names_array[1])
        self.biases_output = np.load(save_names_array[2])
        self.biases_hidden = np.load(save_names_array[3])


if __name__ == "__main__":
    a = NeuralNetwork(2, 10, 1)
    # XOR gate data
    training_data = [
        [[1, 1], [0]],
        [[1, 0], [1]],
        [[0, 1], [1]],
        [[0, 0], [0]]
    ]

    # for i in range(100000):
    # if i % 1000 == 0:
    #     print(i)
    #     print(a.weights_ho)
    #     print(a.weights_ih)
    #     print(a.biases_output)
    #     print(a.biases_hidden)
    # a.train(random.choice(training_data), 0.1)

    a.StochasticGradientDescent(training_data, 50000, .5)

    print(a.feedforward([0, 1]))
    print(a.feedforward([1, 0]))
    print(a.feedforward([1, 1]))
    print(a.feedforward([0, 0]))
