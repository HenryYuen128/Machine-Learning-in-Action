import numpy as np
from math import log
import operator

def createDataset():
    dataSet = [[1,1,'yes'],
                        [1,1,'yes'],
                        [1,0,'no'],
                        [0,1,'no'],
                        [0,1,'no']]
    labels = ['no surfacing', 'flippers']

    return dataSet, labels

def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for sample in dataSet:
        currentLabel = sample[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
        shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob*log(prob, 2)
    return shannonEnt

def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for sample in dataSet:
        if sample[axis] == value:
            reducedSample = sample[:axis]
            reducedSample.extend(sample[axis+1:])
            retDataSet.append(reducedSample)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [sample[i] for sample in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob*calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

# 如果所有特征都已经被用于分割数据集，到叶节点时还是不止一个类别，则用投票表决决定该叶节点所属类别
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList): #取分类标签(yes or no)
        return classList[0] #如果类别完全相同则停止继续划分
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)






if __name__ == '__main__':
    dataSet, labels = createDataset()

    # print(dataSet)

    # shannonEnt = calcShannonEnt(dataSet)
    # print(shannonEnt)

    # retDataSet = splitDataSet(dataSet, 0, 1)
    # print(retDataSet)

    # bestFeature = chooseBestFeatureToSplit(dataSet)
    # print(bestFeature)
    print(dataSet[0])
    createTree(dataSet, labels)