# -*- coding = utf-8 -*-
# @FileName : kNN.py
# @Time     : 2024/05/09 16:05

from numpy import *
import operator


def createDataSet():
    group = array([[1.0, 1.1],[1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    """
    @param inX: 输入向量
    @param dataSet: 输入的训练样本集
    @param labels: 标签向量
    @param k: 用于选择最近邻居的数目
    @return: 使用欧氏距离公式计算向量点之间的距离，平方开根号
    """
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndices = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def autoNorm(dataSet):
    """
    对特征值进行归一化处理
    @param dataSet: 原输入数据集
    @return: 归一化后数据集，最大值 - 最小值，最小值
    """
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m , 1))
    return normDataSet, ranges, minVals


def img2vector(filename):
    """
    把32*32的二进制图像转换为1*1024的向量
    @param filename: 图像名称
    @return: 1*1024的向量
    """
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr)
    return returnVect


g, l = createDataSet()
print(g)
print(l)
print(classify0([0, 0], g, l, 3))
