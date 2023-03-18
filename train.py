import wandb
import NeuralNet
import pandas as pd
from data_prep import data
import argparse
from NeuralNet import Neural
parser = argparse.ArgumentParser(description="Calculate the Train accuracy")

parser.add_argument('-wp' , '--wandb_project', help='Project name used to track experiments in Weights & Biases dashboard' , type=str, default='myprojectname')
parser.add_argument('-we', '--wandb_entity' , help='Wandb Entity used to track experiments in the Weights & Biases dashboard.' , type=str, default='myname')
parser.add_argument('-d', '--dataset', help='choices: ["mnist", "fashion_mnist"]', type=str, default='fashion_mnist')
parser.add_argument('-e', '--epochs', help="Number of epochs to train neural network.", type=int, default=1)
parser.add_argument('-b', '--batch_size', help="Batch size used to train neural network.", type=int, default=4)
parser.add_argument('-l','--loss', help = 'choices: ["squared_error", "cross_entropy"]' , type=str, default='cross_entropy')
parser.add_argument('-o', '--optimizer', help = 'choices: ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]', type=str, default = 'sgd')
parser.add_argument('-lr', '--learning_rate', help = 'Learning rate used to optimize model parameters', type=float, default=0.01)
parser.add_argument('-m', '--momentum', help='Momentum used by momentum and nag optimizers.',type=float, default=0.5)
parser.add_argument('-beta', '--beta', help='Beta used by rmsprop optimizer',type=float, default=0.5)
parser.add_argument('-beta1', '--beta1', help='Beta1 used by adam and nadam optimizers.',type=float, default=0.5)
parser.add_argument('-beta2', '--beta2', help='Beta2 used by adam and nadam optimizers.',type=float, default=0.5)
parser.add_argument('-eps', '--epsilon', help='Epsilon used by optimizers.',type=float, default=0.000001)
parser.add_argument('-w_d', '--weight_decay', help='Weight decay used by optimizers.',type=float, default=.0)
parser.add_argument('-w_i', '--weight_init', help = 'choices: ["random", "Xavier"]', type=str, default='random')
parser.add_argument('-nhl', '--num_layers', help='Number of hidden layers used in feedforward neural network.',type=int, default=1)
parser.add_argument('-sz', '--hidden_size', help ='Number of hidden neurons in a feedforward layer.', nargs='+', type=int, default=4, required=False)
parser.add_argument('-a', '--activation', help='choices: ["identity", "sigmoid", "tanh", "ReLU"]', type=str, default='sigmoid')
parser.add_argument('--hlayer_size', type=int, default=32)
parser.add_argument('-oa', '--output_activation', help = 'choices: ["softmax"]', type=str, default='softmax')
parser.add_argument('-oc', '--output_size', help ='Number of neurons in output layer used in feedforward neural network.', type = int, default = 10)
arguments = parser.parse_args()

x_train, x_test, x_val, y_train, y_test, y_val = data(arguments.dataset)

wandb.login() 
wandb.init(project=arguments.wandb_project,entity='harshvrma')

xsize=784
output_size=10
hidden_layer=[arguments.hidden_size] * arguments.num_layers+[10]
print(hidden_layer)
number_of_hidden_layer=3
print(arguments.num_layers)
print(arguments.activation)

obj = Neural(xsize, hidden_layer, arguments.num_layers+1, arguments.activation, arguments.weight_init, arguments.weight_decay)
optimizer=arguments.optimizer
if(optimizer=="sgd"):
        obj.optimize(x_train, y_train, x_val, y_val, arguments.optimizer,arguments.learning_rate,arguments.epochs, arguments.batch_size,arguments.loss)
elif(optimizer=="momentum"):
        obj.optimize(x_train, y_train, x_val, y_val, arguments.optimizer,arguments.learning_rate,arguments.epochs, arguments.batch_size,arguments.loss,arguments.momentum)
elif(optimizer=="nag"):
        obj.optimize(x_train, y_train, x_val, y_val, arguments.optimizer,arguments.learning_rate,arguments.epochs, arguments.batch_size,arguments.loss,arguments.momentum)
elif(optimizer=="rmsprop"):
        obj.optimize(x_train, y_train, x_val, y_val, arguments.optimizer,arguments.learning_rate,arguments.epochs, arguments.batch_size,arguments.loss,arguments.momentum,arguments.beta,arguments.epsilon)
elif(optimizer=="nadam"):
        obj.optimize(x_train, y_train, x_val, y_val, arguments.optimizer,arguments.learning_rate,arguments.epochs, arguments.batch_size,arguments.loss,arguments.momentum,arguments.beta,arguments.epsilon,arguments.beta1,arguments.beta2)
elif(optimizer=="adam"):
        obj.optimize(x_train, y_train, x_val, y_val, arguments.optimizer,arguments.learning_rate,arguments.epochs, arguments.batch_size,arguments.loss,arguments.momentum,arguments.beta,arguments.epsilon,arguments.beta1,arguments.beta2)
#print(x_test[0])
#NeuralNet = NeuralNet(arguments)
#NeuralNet.train()
