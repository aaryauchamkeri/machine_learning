import numpy as np


def sigmoid(fsum):
    return 1 / (1 + (np.e ** -fsum))


def predictions(weight, b, inputs):
    yhat = []
    for input in inputs:
        prediction = sigmoid(weight * input + b)
        yhat.append(prediction)
    return yhat


def derivative(w, b, inputs, outputs, bias=False):
    m = len(inputs)
    summation = 0
    for i in range(0, m):
        if not bias:
            summation += (sigmoid(w * inputs[i] + b) - outputs[i]) * inputs[i]
        else:
            summation += (sigmoid(w * inputs[i] + b) - outputs[i])

    return summation / m


def logloss(weight, b, inputs, outputs, alpha):
    while True:
        wderiv = derivative(weight, b, inputs, outputs)
        bderiv = derivative(weight, b, inputs, outputs, True)
        lw = weight
        lb = b
        weight -= alpha * wderiv
        b -= alpha * bderiv

        if abs(weight - lw) <= 0.000001 and abs(b - lb) <= 0.000001:
            break

    return weight, b


w = 0.05
b = 0.05

ip = np.array([1, 2, 3, 4, 5, 6, 7, 8])
op = np.array([0, 0, 0, 0, 1, 1, 1, 1])
alpha = 0.01

print(logloss(w, b, ip, op, alpha))
