import tensorflow as tf


def initialize_parameters(layer_dims, seed):
    parameters = {}
    L = len(layer_dims)
    initializer = tf.keras.initializers.GlorotNormal(seed)
    for l in range(1, L):
        parameters['W' + str(l)] = tf.Variable(initializer(shape=[layer_dims[l], layer_dims[l-1]]), dtype=tf.float32)
        parameters['b' + str(l)] = tf.Variable(tf.zeros((layer_dims[l], 1), dtype=tf.float32))
    return parameters

def sigmoid(z):
    z = tf.cast(z, tf.float32)
    a = tf.keras.activations.sigmoid(z)
    return a

def relu(Z):
    return tf.math.maximum(0, Z)


def linear_forward(A, W, b):
    Y = tf.add(tf.matmul(W, A), b)
    cache = (A, W, b)
    return Y, cache


def linear_activation_forward(A_prev, W, b, activation):
    Z, linear_cache = linear_forward(A_prev, W, b)

    if activation == "relu":
        A = relu(Z)
    elif activation == "sigmoid":
        A = sigmoid(Z)

    activation_cache = Z
    cache = (linear_cache, activation_cache)
    return A, cache


def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters)//2

    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation='relu')
        caches.append(cache)

    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation='sigmoid')
    caches.append(cache)
    return AL, caches


def compute_cost_tf(AL, Y):
    epsilon = 1e-7 # log(AL) and log(1-AL) will break if AL is 0 or 1
    AL = tf.clip_by_value(AL, epsilon, 1 - epsilon)
    m = tf.cast(tf.shape(Y)[1], tf.float32)
    cost = -tf.reduce_sum( Y * tf.math.log(AL) + (1 - Y) * tf.math.log(1 - AL)) / m
    return cost


def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = tf.cast(tf.shape(A_prev)[1], tf.float32)
    dW = tf.matmul(dZ, A_prev, transpose_b=True) / m
    db = tf.reduce_sum(dZ, axis=1, keepdims=True) / m
    dA_prev = tf.matmul(W, dZ, transpose_a=True)
    return dA_prev, dW, db


def relu_backward(dA, activation_cache):
    Z = activation_cache
    dZ = dA * tf.cast(Z > 0, dA.dtype)
    return dZ


def sigmoid_backward(dA, activation_cache):
    Z = activation_cache
    s = sigmoid(Z)
    dZ = dA * s * (1 - s)
    return dZ


def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)  # the number of layers
    m = tf.cast(tf.shape(AL)[1], tf.float32)
    Y = tf.reshape(Y, tf.shape(AL))  # after this line, Y is the same shape as AL
    epsilon = 1e-7
    AL = tf.clip_by_value(AL, epsilon, 1 - epsilon) # This will explode if AL ≈ 0 or 1
    dAL = - (tf.math.divide(Y, AL) - tf.math.divide(1 - Y, 1 - AL))

    current_cache = caches[L - 1]
    dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dAL, current_cache, activation="sigmoid")
    grads["dA" + str(L - 1)] = dA_prev_temp
    grads["dW" + str(L)] = dW_temp
    grads["db" + str(L)] = db_temp

    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache,
                                                                    activation="relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    return grads


def update_parameters(params, grads, learning_rate):
    # parameters = copy.deepcopy(params)
    L = len(params) // 2  # number of layers in the neural network
    for l in range(1, L+1):
        params["W" + str(l)].assign_sub(learning_rate * grads["dW" + str(l)])
        params["b" + str(l)].assign_sub(learning_rate * grads["db" + str(l)])
    return params


def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False, seed = 1):
    costs = []
    parameters = initialize_parameters(layers_dims, seed)
    for i in range(0, num_iterations):
        AL, caches = L_model_forward(X, parameters)
        cost = compute_cost_tf(AL, Y)
        grads = L_model_backward(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)
        if print_cost and (i % 100 == 0 or i == num_iterations - 1):
            print("Cost after iteration {}: {}".format(i, tf.squeeze(cost)))
        if i % 100 == 0:
            costs.append(float(cost))
    return parameters, costs


def predict(X, Y, parameters):
    probas, _ = L_model_forward(X, parameters)
    # Convert probabilities to {0,1}
    p = tf.cast(probas > 0.5, tf.float32)
    if Y is not None:
        m = tf.cast(tf.shape(X)[1], tf.float32)
        accuracy = tf.reduce_sum(tf.cast(p == Y, tf.float32)) / m
        print("Accuracy:", float(accuracy))
    return p


#______________________________________________Test__________________________________________________#


X = tf.constant([
    [0, 1, 0, 1],
    [1, 0, 1, 0],
    [0, 1, 1, 0],
    [1, 0, 0, 1]
], dtype=tf.float32)

Y = tf.constant([[0.2, 0.8, 0.25, 0.75]], dtype=tf.float32)

layers_dims = [4, 3, 7, 1]

parameters, costs = L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=True, seed = 1)
print(predict(X, Y, parameters))