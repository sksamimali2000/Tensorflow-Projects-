# MNIST Digit Classification using TensorFlow (Basic Neural Network)

## 📚 Overview
This project demonstrates a simple neural network built using **TensorFlow (1.x)** for handwritten digit classification on the famous **MNIST dataset**. The model consists of two hidden layers with ReLU activation and is trained using the Adam optimizer.

---

## ✅ Dataset
The MNIST dataset contains:
- 55,000 training images
- 10,000 testing images
- 5,000 validation images  
Each image is 28x28 pixels (flattened to a 784-dimensional vector), and the labels are one-hot encoded.

---

## ⚡ Load MNIST Data
```python
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
```


📊 Data Visualization

```Python
import numpy as np
from matplotlib import pyplot as plt

first_image = mnist.train.images[412].reshape((28, 28))
plt.imshow(first_image, cmap='gray')
plt.show()
```


⚙️ Define Model Architecture
```Python
import tensorflow as tf

n_input = 784
n_hidden_1 = 256
n_hidden_2 = 256
n_classes = 10

weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}

biases = {
    'h1': tf.Variable(tf.random_normal([n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def forward_propagation(x, weights, biases):
    layer1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['h1']))
    layer2 = tf.nn.relu(tf.add(tf.matmul(layer1, weights['h2']), biases['h2']))
    output = tf.add(tf.matmul(layer2, weights['out']), biases['out'])
    return output
```

⚡ Training Setup
```Python
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder(tf.int32, [None, n_classes])

pred = forward_propagation(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
```

🚀 Training the Model
```Python
batch_size = 100
for i in range(25):
    num_batches = int(mnist.train.num_examples / batch_size)
    total_cost = 0
    for j in range(num_batches):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        c, _ = sess.run([cost, optimizer], feed_dict={x: batch_x, y: batch_y})
        total_cost += c
    print(f"Epoch {i+1}: Total Cost = {total_cost}")
```

✅ Evaluation
```Python
predictions = tf.argmax(pred, 1)
correct_labels = tf.argmax(y, 1)
correct_predictions = tf.equal(predictions, correct_labels)

predictions, correct_predictions = sess.run(
    [predictions, correct_predictions],
    feed_dict={x: mnist.test.images, y: mnist.test.labels}
)

print(f"Correct predictions count: {correct_predictions.sum()}")
```

🎯 Conclusion
This implementation demonstrates a basic neural network trained from scratch using TensorFlow 1.x on the MNIST dataset. The network learns to classify handwritten digits into the correct classes with good accuracy after several epochs of training.

⚡ Notes
Ensure you are using TensorFlow 1.x since this code uses tf.Session() and tf.placeholder().

To run on TensorFlow 2.x, use tf.compat.v1.disable_eager_execution() at the top of the script.
