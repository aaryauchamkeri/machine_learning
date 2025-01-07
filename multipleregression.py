import numpy as np

# simple multiple variable regression

# finding the derivative for a bias
def derivB(input, output, b, features):
    m = len(input)
    deriv = 0
    for i in range(0, len(input)):
        fsum = 0
        for j in range(len(features)):
            fsum += input[i] * features[j]
        deriv += 2*((fsum + b) - output[i])
    return deriv/(2*m)


# derivative of features
def derivFeat(input, output, b, features):
    m = len(input)
    deriv = 0
    for i in range(0, len(input)):
        fsum = 0
        for j in range(len(features)):
            fsum += input[i] * features[j]
        deriv += 2*((fsum + b) - output[i])*input[i]
    return deriv / (2 * m)


def getMaxDif(f, s):
    maxdif = float('-inf')
    for i in range(len(f)):
        maxdif = max(maxdif, abs(f[i] - s[i]))
    return maxdif


def regression(input, output, features, bias, alpha):
    featuresp = [0] * len(features)
    while getMaxDif(features, featuresp) >= 0.000001:
        tempf = [0] * len(features)
        for i in range(len(tempf)):
            d = derivFeat(input, output, bias, features)
            tempf[i] = features[i] - (alpha * d)
        bd = derivB(input, output, bias, features)
        bias -= alpha * bd
        featuresp = features
        features = tempf
    return (features, bias)


input = np.array([1,2,3,4,5,6])
output = np.array([3,4,5,6,7,8])
features = [1]
bias = 0
alpha = 0.01

print(regression(input, output, features, bias, alpha))