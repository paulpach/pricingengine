#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 10:16:49 2018

@author: paul
"""

import numpy as np
from scipy.stats import expon
import matplotlib.pyplot as plt

experiments = 100



winningprice = expon(-3, 5)

def buy(price):
    win = winningprice.rvs()
    return price < win

def purchases(price):
    tot = 0
    for i in range(experiments):
        tot += buy(price)

    return tot

def revenue(price):
    return purchases(price) * price


def sampledata(samples):
    price = np.linspace(0, 30, samples);
    conversions = purchases(price)
    revenue = price * conversions
    return price, conversions, revenue

if __name__ == "__main__":
 
    fig = plt.figure()

    # generate some synthetic data
    # price is the price point tested
    # conversions is the amount of people that bought the item
    # rev is the revenue that we got from those purchases
    price, conversions, rev = sampledata(30);

    
    # show the conversions
    ax = plt.subplot(3, 1, 1)    
    plt.plot(price, conversions, "bo", lw=2, label="conversions per 100 impressions")
    plt.legend()
    

    # show the revenue
    ax = plt.subplot(3, 1, 2)
    plt.plot(price, rev, "bo", lw=2, label="revenue per 100 impressions")
    plt.legend()   


    # winning price distribution
    # a person will buy an item if his winning price is <= price
    ax = plt.subplot(3, 1, 3)    
    x = np.linspace(0, 30, 30)
    plt.plot(x, winningprice.pdf(x), label="Winning price distribution")
    plt.legend()


    plt.show()