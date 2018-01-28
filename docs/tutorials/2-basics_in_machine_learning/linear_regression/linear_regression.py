import tensorflow as tf 


#read the file from data the .xls dile
import os
import matplotlib.pyplot as plt
import xlrd
import numpy as np

import utils

DATA_FILE = "../data/fire_theft.xls"

#step 1 read in data from the .xls file
book = xlrd.open_workbook(DATA_FILE, encoding_override="utf-8")
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
n_samples = sheet.nrows -1

#step 2 create placeholder for input X (number of fire) and label Y (number of theft)
X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")

#step3 create weight and bias, initiliazed to 0
w = tf.Variable(0.0, name="weights")
b = tf.Variable(0.0, name="bias")

#build model to predict Y
Y_predicted = X * w + b

#Step5 use the square error as the loss function
loss = tf.square(Y - Y_predicted, name = "loss")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

with tf.Session() as sess:
	#step 7: initialize the necessary varaibles, in this case, w and b
	sess.run(tf.global_variables_initializer())
	writer = tf.summary.FileWriter('.graphs/linear_reg', sess.graph)

	#step 8: train the model
	for i in range(50): #train the model 100 epochs
		total_loss = 0
		for x,y in data:
			#session runs train_op and fetch values of loss
			_, l = sess.run([optimizer, loss], feed_dict={X:x, Y:y})
			total_loss += l
		print('Epoch {0}: {1}'.format(i, total_loss/n_samples))

	writer.close()
	w, b = sess.run([w, b])


#plot the results
X, Y = data.T[0], data.T[1]
plt.plot(X,Y,'bo',label='Real data')
plt.plot(X,X*w + b, 'r', label="Predicted data")
plt.legend()
plt.show()
#step6 using gradient descent with learning rate of 0.01 to minizmize loss
