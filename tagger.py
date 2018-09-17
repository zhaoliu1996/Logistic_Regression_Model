import sys
import os
import csv
import math
import numpy as np
from itertools import chain
import copy

# parse the input file with specific mode, return words, features, tags and labels
def parsing(input, mode):
    # print("in parsing")
    feats = []
    labels = []
    file = list()
    with open(input, 'r') as tsv:
        # print("get file")
        for line in csv.reader(tsv, delimiter="\t"):
            if (len(line) == 0):
                continue
            feats.append(line[0])
            labels.append(line[1])
        # print("file ready")

    words = sorted(list(set(feats)))
    tags = sorted(list(set(labels)))

    features = []

    if (mode == 1):
        for word in feats:
            row = []
            row.append(words.index(word))
            features.append(row)

    else:
        words.insert(0, "BOS")
        words.append("EOS")

        for word in feats:
            row = []
            index = words.index(word)
            row.append(index-1)
            row.append(index)
            row.append(index+1)
            features.append(row)


    # print(file)
    return words, features, tags, labels

def handleData(file, mode):
    labels = []
    words = []
    if (mode == 1):
        for row in file:
            words.append(row[0])
            labels.append(row[1])

# init theta matrix according to the sizes
def getThetaMatrix(width, height):
    matrix = []
    for i in range(height):
        row = [0] * width
        matrix.append(row)
    return matrix

def updateTheta(xMatrix, valMatrix, thetaMatrix, allTags, labels, valLabels, num):
    theta = thetaMatrix
    str = ""
    for i in range(num):
        result = update(xMatrix, valMatrix, theta, allTags, labels, valLabels)
        theta = result[0]
        train_log = result[1]
        val_log = result[2]
        tempstr1 = "epoch={} likelihood(train): {}\n".format(i + 1, train_log)
        tempstr2 = "epoch={} likelihood(validation): {}\n".format(i + 1, val_log)
        str += tempstr1
        str += tempstr2
    return [theta, str]


def update(xMatrix, valMatrix, theta, allTags, labels, valLabels):
    l = len(theta[0])
    for i in range(len(labels)):
        y = labels[i]
        # x = np.array(xMatrix[i])
        x = xMatrix[i]
        grads = []

        for j in range(len(allTags)):  # comput gradiante for theta
            p = allTags[j]
            thetaRow = theta[j]
            bool = int(y == p)
            # print("x: ", x)
            # print("theta: ", theta)
            prob = getProb(x, theta, thetaRow)
            scalar = prob - bool

            grad = [0] * (l)
            for index in x:
                grad[index] = scalar
            grad[-1] = scalar

            # grad = scalar * (np.array(newX))
            # grad.tolist()
            # print("grad: ", grad)
            grads.append(grad)

        for k in range(len(allTags)):  # update all theta
            for q in x:
                theta[k][q] -= 0.5 * grads[k][q]
            theta[k][-1] -= 0.5 * grads[k][-1]
    train_log = getLog(xMatrix, theta, labels, allTags)
    val_log = getLog(valMatrix, theta, valLabels, allTags)
    # print(train_log)
    # print(val_log)

    return [theta, train_log, val_log]


# note: vecX is np.array, row in theta is np.array
def getProb(vecX, theta, thetaRow):
    total = 0
    for i in range(len(theta)):
        inner = newDot(vecX, theta[i])
        total += math.exp(inner)
    up = newDot(vecX, thetaRow)
    prob = math.exp(up) / total
    return prob


# base on e
def getLog(xMatrix, theta, labels, allTags):
    res = 0
    for i in range(len(xMatrix)):
        # vecX = np.array(xMatrix[i])
        vecX = xMatrix[i]
        y = labels[i]
        k = allTags.index(y)
        rowK = theta[k]
        inner = newDot(vecX, rowK)
        up = math.exp(inner)
        down = 0
        for row in theta:
            inside = newDot(vecX, row)
            down += math.exp(inside)
        res += math.log(up / down)
    res /= len(labels)
    res *= -1
    return res

def newDot(indices, vec):
    # print(indices)
    # print(vec)
    total = 0
    for i in indices:
        total += vec[i]
    total += vec[-1]
    return total

def test(theta, matrix, labels, labelsWithGap):
    # result = []
    erro = 0
    total = len(matrix)
    gap = 0
    str = ""
    for k in range(len(matrix)):
        # sample = np.array(matrix[k])
        sample = matrix[k]
        probs = []
        input = labelsWithGap[k + gap]

        if (input == []):
            gap += 1
            str += "\n"

        for row in theta:
            prob = newDot(sample, row)
            probs.append(prob)

        # find label with max likelihood
        arrayProbs = np.array(probs)
        indices = np.where(arrayProbs == max(probs))
        indices = indices[0].tolist()
        if (len(indices) == 1):
            index = indices[0]
            # result.append(labels[index])
            str += labels[index]
            if (labels[index] != labelsWithGap[k + gap]):
                erro += 1
        else:
            opt = []
            for i in indices:
                opt.append(labels[i])
            # result.append(min(opt))
            str += labels[index]
            if (labels[index] != labelsWithGap[k + gap]):
                erro += 1
        str += "\n"
    # print(str)
    rate = float(erro) / total
    # print(rate)
    return [str, rate]

def run(train_input, val_input, test_input, train_out, test_out, metrics, num, mode):
    # print("in run")
    # get training data
    tnwords, tnfeatures, tntags, tnlabels = parsing(train_input, mode)
    valwords, valfeatures, valtags, vallabels = parsing(val_input, mode)

    if (mode == 1):
        width = len(tnwords) + 1
    else:
        width = len(tnwords) * 3 + 1
    height = len(tntags)
    thetaMatrix = getThetaMatrix(width, height)
    res = updateTheta(tnfeatures, valfeatures, thetaMatrix, tntags,
                      tnlabels, vallabels, num)
    theta = res[0]
    likelihood = res[1]

    train_str, train_err = test(theta, tnfeatures, tntags, tnlabels)
    trainFile = open(train_out, 'w')
    trainFile.write(train_str)
    trainFile.close()
    print("a")

    testwords, testfeatures, testtags, testlabels = parsing(test_input, mode)
    test_str, test_err = test(theta, testfeatures, tntags, testlabels)
    testFile = open(test_out, 'w')
    testFile.write(test_str)
    testFile.close()

    err_train = "error(train): {}\n".format(train_err)
    err_test = "error(test): {}\n".format(test_err)
    likelihood += err_train
    likelihood += err_test
    erroFile = open(metrics, 'w')
    erroFile.write(likelihood)
    erroFile.close()




##
## Main function
if __name__ == '__main__':
    train_input = sys.argv[1]
    validation_input = sys.argv[2]
    test_input = sys.argv[3]
    train_out = sys.argv[4]
    test_out = sys.argv[5]
    metrics = sys.argv[6]
    num_epoch = int(sys.argv[7])
    feature = int(sys.argv[8])

    run(train_input, validation_input, test_input, train_out, test_out, metrics,
        num_epoch, feature)