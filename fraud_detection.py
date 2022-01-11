import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import time

# Import and store dataset
credit_card_data = pd.read_csv('creditcard.csv')

# Splitting data into 4 sets
# 1. Shuffle/randomize data
# 2. One-hot encoding
# 3. Normalize
# 4. Splitting up X/y values
# 5. Convert data_frames to numpy arrays (float32)
# 6. Splitting the final data into X/y train/test

# Shuffle and randomize data
shuffled_data = credit_card_data.sample(frac=1)

# Change Class column into Class_0 ([1 0] for legit data) and Class_1 ([0 1] for fraudulent data)
one_hot_data = pd.get_dummies(shuffled_data, columns=['Class'])

# Change all values into numbers between 0 and 1
normalized_data = (one_hot_data - one_hot_data.min()) / (one_hot_data.max() - one_hot_data.min())

# Store just columns V1 through V28 in df_X and columns Class_0 and Class_1 in df_y
df_X = normalized_data.drop(['Class_0', 'Class_1'], axis=1)
df_y = normalized_data[['Class_0', 'Class_1']]

# Convert both data_frames into np arrays of float32
ar_X, ar_y = np.asarray(df_X.values, dtype='float32'), np.asarray(df_y.values, dtype='float32')

# Allocate first 80% of data into training data and remaining 20% into testing data
train_size = int(0.8 * len(ar_X))
(raw_X_train, raw_y_train) = (ar_X[:train_size], ar_y[:train_size])
(raw_X_test, raw_y_test) = (ar_X[train_size:], ar_y[train_size:])

# Gets a percent of fraud vs legit transactions (0.0017% of transactions are fraudulent)
count_legit, count_fraud = np.unique(credit_card_data['Class'], return_counts=True)[1]
fraud_ratio = float(count_fraud / (count_legit + count_fraud))
print('Percent of fraudulent transactions: ', fraud_ratio)

weighting = 1 / fraud_ratio
raw_y_train[:, 1] = raw_y_train[:, 1] * weighting

# 30 cells for the input
input_dimensions = ar_X.shape[1]
# 2 cells for the output
output_dimensions = ar_y.shape[1]
# 100 Cells for the 1st layer
num_layer_1_cells = 150
# 150 Cells for the second layer
num_layer_2_cells = 150

# We will use these as inputs to the model when it comes time to train it (assign values at run time)
X_train_node = tf.placeholder(tf.float32, [None, input_dimensions], name="X_Train")
y_train_node = tf.placeholder(tf.float32, [None, output_dimensions], name="y_Train")

# We will use these as inputs to the model once it comes time to test it
x_test_node = tf.constant(raw_X_test, name="X_Test")
y_test_node = tf.constant(raw_y_test, name="y_Test")

# First layer takes in input and passes output to 2nd layer
weight_1node = tf.Variable(tf.zeros([input_dimensions, num_layer_1_cells]), name="weight_1")
bias_1node = tf.Variable(tf.zeros([num_layer_1_cells]), name="bias_1")

# Second layer takes in input from 1st layer and passes output to 3rd layer
weight_2node = tf.Variable(tf.zeros([num_layer_1_cells, num_layer_2_cells]), name="weight_2")
bias_2node = tf.Variable(tf.zeros([num_layer_1_cells]), name="bias_2")

# Third layer takes in input from 2nd layer and outputs [1 0] or [0 1] depending on fraud vs legit
weight_3node = tf.Variable(tf.zeros([num_layer_2_cells, output_dimensions]), name="weight_2")
bias_3node = tf.Variable(tf.zeros([output_dimensions]), name="bias_3")


def network(input_tensor):
    layer1 = tf.nn.sigmoid(tf.matmul(input_tensor, weight_1node) + bias_1node)
    layer2 = tf.nn.dropout(tf.nn.sigmoid(tf.matmul(layer1, weight_2node) + bias_2node), 0.85)
    layer3 = tf.nn.softmax(tf.matmul(layer2, weight_3node) + bias_3node)
    return layer3


y_train_prediction = network(X_train_node)
y_test_prediction = network(x_test_node)

cross_entropy = tf.losses.softmax_cross_entropy(y_train_node, y_train_prediction)

optimizer = tf.train.AdamOptimizer(0.005).minimize(cross_entropy)


def calculate_accuracy(actual, predicted):
    actual = np.argmax(actual, 1)
    predicted = np.argmax(predicted, 1)
    return 100 * np.sum(np.equal(predicted, actual)) / predicted.shape[0]


num_epoch = 100

# Function to calculate the accuracy of the actual result vs the predicted result
with tf.Session() as session:
    tf.global_variables_initializer().run()
    for epoch in range(num_epoch):

        start_time = time.time()
        _, cross_entropy_score = session.run([optimizer, cross_entropy],
                                             feed_dict={X_train_node: raw_X_train, y_train_node: raw_y_train})
        if epoch % 10 == 0:
            timer = time.time() - start_time

            print('Epoch : {}'.format(epoch),
                  'Current loss :  {0:.4f}'.format(cross_entropy_score),
                  'Elapsed Time : {0: .2f}'.format(timer))

        final_y_test = y_test_node.eval()
        final_y_test_prediction = y_test_prediction.eval()
        final_accuracy = calculate_accuracy(final_y_test, final_y_test_prediction)
        print('Current accuracy is : {0:.2f}%'.format(final_accuracy))

    final_y_test = y_test_node.eval()
    final_y_test_prediction = y_test_prediction.eval()
    final_accuracy = calculate_accuracy(final_y_test, final_y_test_prediction)
    print('Current accuracy is : {0:.2f}%'.format(final_accuracy))

final_fraud_y_test = final_y_test[final_y_test[:, 1] == 1]
final_y_test_prediction = final_y_test_prediction[final_y_test[:, 1] == 1]
final_accuracy = calculate_accuracy((final_fraud_y_test, final_y_test_prediction))
print('Final fraud accuracy : {0:.2f}%' . format(final_accuracy))