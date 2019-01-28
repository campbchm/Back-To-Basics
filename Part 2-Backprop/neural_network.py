import math
import random
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def derivative(x):
    return x * (1 - x)

AND_data = [[[0, 0], [0]], 
            [[0, 1], [0]], 
            [[1, 0], [0]], 
            [[1, 1], [1]]]

OR_data = [[[0, 0], [0]], 
           [[0, 1], [1]], 
           [[1, 0], [1]], 
           [[1, 1], [1]]]

XOR_data = [[[0, 0], [0]], 
            [[0, 1], [1]], 
            [[1, 0], [1]], 
            [[1, 1], [0]]]

IF_data = [[[1, 1], [1]], 
            [[1, 0], [0]], 
            [[0, 1], [0]], 
            [[0, 0], [1]]]


class NeuralNetwork():
    def __init__(self, num_inputs, num_hidden_units, num_outputs):
        self.LEARNING_RATE = 0.5
        self.EPISODES = 5000
        self.num_inputs = num_inputs
        self.num_hidden_units = num_hidden_units
        self.num_outputs = num_outputs
        self.input_layer = Layer(num_inputs)
        self.hidden_layer = Layer(num_hidden_units)
        self.output_layer = Layer(num_outputs)
        
        self.init_weights_input_hidden()
        self.init_weights_hidden_output()
        
        self.error = 0
        
        # Data Collection
        self.error_list = []
        
        
    def init_weights_input_hidden(self):
        for hidden_neuron in self.hidden_layer.neurons:
            for in_neuron in self.input_layer.neurons:
                hidden_neuron.weights.append(random.random())
            hidden_neuron.weights.append(random.random())   # add the BIAS
    
    
    def init_weights_hidden_output(self):
        for output_neuron in self.output_layer.neurons:
            for hidden_neuron in self.hidden_layer.neurons:
                output_neuron.weights.append(random.random())
            output_neuron.weights.append(random.random())   # add the BIAS
    
    
    def feed_forward(self, inputs):
        hidden_layer_outputs = self.hidden_layer.get_layer_output(inputs)
        return self.output_layer.get_layer_output(hidden_layer_outputs)
    
        
    def back_propogate(self, inputs, output, target):
        # Calculate the error for the output neuron
        output_error = (target[0] - output[0]) * derivative(output[0])
        self.error = (output[0] - target[0])**2
        
        # Calculate the error for each of the hidden neurons
        for i in range(len(self.hidden_layer.neurons)):
            output_neuron = self.output_layer.neurons[0]
            hidden_neuron = self.hidden_layer.neurons[i]
            hidden_error = (output_neuron.weights[i] 
                            * output_error 
                            * derivative(hidden_neuron.output))
            hidden_neuron.error = hidden_error
            
        # Adjust the weights to the output neuron according to the error
        for i in range(len(self.output_layer.neurons[0].weights) -1):
            self.output_layer.neurons[0].weights[i] += (self.hidden_layer.neurons[i].output 
                                     * output_error 
                                     * self.LEARNING_RATE)
        self.output_layer.neurons[0].weights[-1] += output_error * self.LEARNING_RATE # bias
        
        # Adjust the weights going into the hidden units
        for hidden_neuron in self.hidden_layer.neurons:
            for i in range(len(hidden_neuron.weights) - 1):
                hidden_neuron.weights[i] += inputs[i] * hidden_neuron.error * self.LEARNING_RATE
            hidden_neuron.weights[-1] += hidden_neuron.error * self.LEARNING_RATE # bias
        
        
    def single_update(self, inputs, target):
        output = self.feed_forward(inputs)
        self.back_propogate(inputs, output, target)
        
        
    def train(self, data):        
        # Initial, random outputs
        print("Initial guesses by the network")
        for sample in data:
            output = self.feed_forward(sample[0])
            print("Input:", sample[0], "Guess:", output[0], "Target:", sample[1])
        
        # Train
        ("\n---Training: Current Error---")
        for i in range(self.EPISODES):
            for sample in data:
                self.single_update(sample[0], sample[1])
            if i > 16: #if i % 100 == 0 and i > 99:
                self.error_list.append(self.error)
                if i % 500 == 0:
                    print(i, ":", self.error)
        
        # Final output
        for sample in data:
            output = self.feed_forward(sample[0])
            print("Input:", sample[0], "Guess:", output[0], "Target:", sample[1])
        
                   
    def inspect(self):
        print('------')
        print('Inputs: {}'.format(self.num_inputs))
        print('------')
        print('Hidden Layer')
        self.hidden_layer.inspect()
        print('------')
        print('Output Layer')
        self.output_layer.inspect()
        print('------')
        
        
    def plot_error(self):
        fig, ax = plt.subplots()
#        xaxis = [i for i in range(100, self.EPISODES, 100)]
#        ax.plot(xaxis, self.error_list)
        ax.plot(self.error_list)
        ax.set(xlabel='Training Episodes', ylabel = 'Error',
               title = 'Error during training')
        plt.show()
        
    
        
class Layer():
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        self.neurons = []
        
        self.init_neurons()
        
        
    def init_neurons(self):
        for i in range(self.num_neurons):
            self.neurons.append(Neuron())
    
    
    def get_layer_output(self, inputs):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.calculate_output(inputs))
        return outputs
    
    
    def inspect(self):
        print('Neurons:', len(self.neurons))
        for n in range(len(self.neurons)):
            print(' Neuron', n)
            for w in range(len(self.neurons[n].weights) - 1):
                print('  Weight:', self.neurons[n].weights[w])
            print('  Bias Weight:', self.neurons[n].weights[-1])
    
        
        
class Neuron():
    def __init__(self):
        self.weights = [] # Connects Neuron to the Layer prior
        self.output = 0
        self.input = 0
        self.deltas = []
        self.error = 0
        
        
    def calculate_output(self, inputs):
        output = sigmoid(self.calculate_total_input(inputs))
        self.output = output
        return output
    
    
    def calculate_total_input(self, inputs):
        total = 0
        for i in range(len(inputs)):
            total += inputs[i] * self.weights[i]
        total_input = total + self.weights[-1]  # add the bias
        self.input = total_input
        return total_input  
    
    
if __name__ == '__main__':
    print("\nTESTING AND\n")
    ann = NeuralNetwork(2, 5, 1)
    ann.train(AND_data)
    ann.plot_error()
    print("\nTESTING OR\n")
    ann = NeuralNetwork(2, 5, 1)
    ann.train(OR_data)
    ann.plot_error()
    print("\nTESTING IF\n")
    ann = NeuralNetwork(2, 5, 1)
    ann.train(IF_data)
    ann.plot_error()
    print("\nTESTING XOR\n")
    ann = NeuralNetwork(2, 8, 1)
    ann.train(XOR_data)
    ann.plot_error()