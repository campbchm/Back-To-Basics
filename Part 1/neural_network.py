import math
import random

class NeuralNetwork():
    def __init__(self, num_inputs, num_hidden_units, num_outputs):
        self.LEARNING_RATE = 0.1
        self.num_inputs = num_inputs
        self.num_hidden_units = num_hidden_units
        self.num_outputs = num_outputs
        self.input_layer = Layer(num_inputs)
        self.hidden_layer = Layer(num_hidden_units)
        self.output_layer = Layer(num_outputs)
        
        self.init_weights_input_hidden()
        self.init_weights_hidden_output()
        
        
    def init_weights_input_hidden(self):
        for hidden_neuron in self.hidden_layer.neurons:
            for in_neuron in self.input_layer.neurons:
                hidden_neuron.weights.append(random.random())
    
    
    def init_weights_hidden_output(self):
        for output_neuron in self.output_layer.neurons:
            for hidden_neuron in self.hidden_layer.neurons:
                output_neuron.weights.append(random.random())
    
    
    def feed_forward(self, inputs):
        hidden_layer_outputs = self.hidden_layer.get_layer_output(inputs)
        return self.output_layer.get_layer_output(hidden_layer_outputs)
    
    
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
        
    
        
class Layer():
    def __init__(self, num_neurons):
        self.bias = random.random()
        self.num_neurons = num_neurons
        self.neurons = []
        
        self.init_neurons()
        
        
    def init_neurons(self):
        for i in range(self.num_neurons):
            self.neurons.append(Neuron(self.bias))
    
    
    def get_layer_output(self, inputs):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.calculate_output(inputs))
        return outputs
    
    
    def inspect(self):
        print('Neurons:', len(self.neurons))
        for n in range(len(self.neurons)):
            print(' Neuron', n)
            for w in range(len(self.neurons[n].weights)):
                print('  Weight:', self.neurons[n].weights[w])
            print('  Bias:', self.bias)
    
        
        
class Neuron():
    def __init__(self, bias):
        self.bias = bias
        self.weights = []
        
        
    def calculate_output(self, inputs):
        return self.sigmoid(self.calculate_total_input(inputs))
    
    
    def calculate_total_input(self, inputs):
        total = 0
        for i in range(len(inputs)):
            total += inputs[i] * self.weights[i]
        return total + self.bias   
    
    
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))
        