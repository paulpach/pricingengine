#######################################
# pyGPGO examples
# integratedacq: Shows the computation of the integrated acquisition function.
#######################################

import numpy as np
from pyGPGO.surrogates.GaussianProcessMCMC import GaussianProcessMCMC
from pyGPGO.acquisition import Acquisition
from pyGPGO.covfunc import squaredExponential
from pyGPGO.GPGO import GPGO
import matplotlib.pyplot as plt


import pymc3 as pm

# price
data = np.array([
        (0.0, 0.0), 
        (1.0, 45.0), 
        (2.0, 76.0), 
        (3.0, 93.0), 
        (4.0, 108.0), 
        (5.0, 135.0), 
        (6.0, 144.0), 
        (7.0, 133.0), 
        (8.0, 104.0), 
        (9.0, 81.0), 
        (10.0, 60.0), 
        (11.0, 44.0), 
        (12.0, 36.0), 
        (13.0, 39.0), 
        (14.0, 28.0), 
        (15.0, 30.0), 
        (16.0, 16.0), 
        (17.0, 17.0), 
        (18.0, 18.0), 
        (19.0, 19.0), 
        (20.0, 20.0), 
        (21.0, 21.0), 
        (22.0, 0.0), 
        (23.0, 0.0), 
        (24.0, 0.0), 
        (25.0, 0.0), 
        (26.0, 0.0), 
        (27.0, 0.0), 
        (28.0, 0.0), 
        (29.0, 0.0), 
        (30.0, 0.0)])

# the function does not matter
# I won't evaluate it
def dummy(price):
    return price

if __name__ == '__main__':
    
    # set up the model
    sexp = squaredExponential()
    gp = GaussianProcessMCMC(sexp, step=pm.Slice)
    param = {'price': ('cont', [0, 100])}
    acq = Acquisition(mode='IntegratedExpectedImprovement')
    gpgo = GPGO(gp, acq, dummy, param)


    # incorporate already known data    
    x = data[:,0]
    y = data[:,1]
    # reshape so that we get [[1],[2],[3]...]
    reshapedx = np.expand_dims(x,1)
    gp.fit(reshapedx,y) 
    gpgo.tau = np.max(y)
    gpgo.init_evals=len(x)
    
    
    # find the next price point to test
    gpgo._optimizeAcq()
    best = gpgo.best
    
    
    print ("The next price to test is", best)
    
    
    # show sample
    fig = plt.figure()
    ax = plt.subplot(1, 1, 1)
    plt.plot(x,y, "o", label="Observed revenue")
    
    plt.axvline(x = best, label="Next price point")
    plt.legend()
    plt.show();