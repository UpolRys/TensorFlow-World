import tensorflow as tf 

#defining varibales is necessary beacuase they hold the parameteres. without having parameters training, updating, saving, restroing and any
#other operations cannot be performed. The defined variables in the Tensorflow are just tesnors with certain
#shapes and types. the tensors must be initialized with values to become valid. 

from tensorflow.python.framework import ops

#create three variables with three default values
weights = tf.Variable(tf.random_normal([2, 3], stddev=0.1), name="weights")
biases = tf.Variable(tf.zeros([3]), name="biases")
custom_variable = tf.Variable(tf.zeros([3]), name="custom")

#get all the variables' tensors and store them in a list
all_variables_list = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
# ops.get_collection gets the list of all defined variables from the defined graph. The name "key",
#define a specific name for each variabel on the graph

#2) Initialization 
#Initializers of the variables must be run before all other operations in the model. For an analaogy,
#we can consider the starter of the car. Instead of running an initializer, variables can be restored too from saved
#model such as a checkpoint file/ Variables acan be initialized globally, specifically, or from other variables. 

#Initializing specific Variables
#variable_list_custom is the list of variables that we want to initialize
variable_list_custom = [weights, custom_variable]

#the initializer
init_custom_op = tf.variables_initializer(var_list=all_variables_list)

#global variable initialization-- all variables can be initialized at once using following command.
#the op must be run after the model is constructed.
init_all_op = tf.global_variables_initializer()

#method 2
init_all_op = tf.variables_initializer(var_list=all_variables_list)

#initialization of a variables using other existing variables
WeightsNew = tf.Variable(weights.initialized_value(), name="WeightsNew")

#now the variable must be intializes
init_WeightsNew_op = tf.variables_initializer(var_list=[WeightsNew])

with tf.Session() as sess:
	sess.run(init_all_op)
	sess.run(init_custom_op)
	sess.run(init_WeightsNew_op)



