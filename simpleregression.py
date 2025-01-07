import numpy as np

## single param linear regression


def dsetderiv(input, output, w):
    m = len(input)
    sum = 0
    for i in range(0, len(input)):
        sum += 2*input[i]*(w*input[i] - output[i])
    return sum / (2*m)


def regression(input, output, w, alpha):
    last = 0
    while abs(w - last) >= 0.000001:
        last = w
        derivative = dsetderiv(input, output, w)
        w -= alpha * derivative
    return w


learn_rate = 0.01  # alpha

w = 0.3

input = np.array([1, 2, 3, 4, 5, 6])
output = np.array([1, 2, 3, 4, 5, 6])


print(regression(input, output, w, learn_rate))
