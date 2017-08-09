#coding:utf-8
from __future__ import division
import numpy as np


def eval_metrics_smape(pred, true):
    loss = np.nan_to_num(np.fabs(pred-true)/(true+pred))
    return loss.mean()

def eval_L2(pred,true):
    loss = np.square(pred-true)
    return loss.mean()

def eval_L1(pred,true):
    loss = np.abs(pred-true)
    return loss.mean()

def error_rate(percentage_list, size_error):
    temp = 0
    for item in percentage_list:
        if item <= size_error:
            temp = temp+1
    return temp/float(len(percentage_list))

def eval_method_1(pred, true):
    #(P-T)/T
    #误差小于5的预测误差设置为0
    loss = np.abs((pred - true) / true)
    middle = np.abs(pred-true)
    loss[middle < 5] = 0  # special case
    median_error = np.median(loss)
    max_error = max(loss)
    min_error = min(loss)
    average_error = np.mean(loss)
    _5_error = error_rate(loss, 0.05)
    _20_error = error_rate(loss, 0.2)
    _50_error = error_rate(loss, 0.5)
    return median_error,max_error,min_error,average_error,_5_error,_20_error,_50_error

def eval_method_2(pred, true, average):
    #(P-T)/A
    loss = np.abs((pred - true) / average)
    #print loss
    # return loss.mean()
    median_error = np.median(loss)
    max_error = max(loss)
    min_error = min(loss)
    average_error = np.mean(loss)
    _5_error = error_rate(loss, 0.05)
    _20_error = error_rate(loss, 0.2)
    _50_error = error_rate(loss, 0.5)
    return median_error, max_error, min_error, average_error,_5_error,_20_error,_50_error

def eval_method_3(pred, true, average):
    #分段函数
    loss = []
    for a,b,c in zip(pred, true, average):
        if b< c:
            loss.append(abs(a-b)/c)
        else:
            loss.append(abs((a-b)/b))
    #print loss
    # return loss.mean()
    median_error = np.median(loss)
    max_error = max(loss)
    min_error = min(loss)
    average_error = np.mean(loss)
    _5_error = error_rate(loss, 0.05)
    _20_error = error_rate(loss, 0.2)
    _50_error = error_rate(loss, 0.5)
    return median_error, max_error, min_error, average_error,_5_error,_20_error,_50_error

def eval_method_4(pred, true, year_average):
    #绝对误差分析 P-T
    loss = np.abs(pred-true)
    median_error = np.median(loss)
    max_error = max(loss)
    min_error = min(loss)
    average_error = np.mean(loss)
    _5_error = error_rate(loss/year_average, 0.05)
    _20_error = error_rate(loss/year_average, 0.2)
    _50_error = error_rate(loss/year_average, 0.5)
    return median_error, max_error, min_error, average_error,_5_error,_20_error,_50_error


