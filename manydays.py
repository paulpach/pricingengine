#######################################
# pyGPGO examples
# integratedacq: Shows the computation of the integrated acquisition function.
#######################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from pyGPGO.surrogates.GaussianProcessMCMC import GaussianProcessMCMC
from pyGPGO.acquisition import Acquisition
from pyGPGO.covfunc import squaredExponential
from pyGPGO.GPGO import GPGO
import data;
from itertools import islice

import pymc3 as pm

def ndprint(a, format_string ='{0:.2f}'):
    print([format_string.format(v,i) for i,v in enumerate(a)])
    
    
if __name__ == '__main__':

    
   
    # set up the model
    # the assumption is that the optimal price is somewhere between 0 and 100
    param = {'price': ('cont', [0, 100])}
    sexp = squaredExponential()
    gp = GaussianProcessMCMC(sexp, step=pm.Slice)
    acq = Acquisition(mode='IntegratedExpectedImprovement')
    gpgo = GPGO(gp, acq, data.revenue, param)


    # We have no history at all at day 0
    # but we do know that at 0,  I make no revenue,
    # so we can consider that as a valid history
    gp.fit(np.array([[0]]),np.array([0]))
    gpgo.tau = 0
    gpgo.init_evals=1
    
    
    # run 10 days,  starting from day 0
    gpgo.run(max_iter=10, resume = True)


    fig = plt.figure()
    

    # show some samples from the posterior
    ax = plt.subplot(3, 1, 1)
    plt.xlim(0,100)
    Z = np.linspace(0, 100, 100)[:, None]
    post_mean, post_var = gpgo.GP.predict(Z, return_std=True, nsamples=200)
    for i in range(200):
        plt.plot(Z.flatten(), post_mean[i], linewidth=0.4)

    #plt.plot(gpgo.GP.X.flatten(), gpgo.GP.y, 'X', label='Sampled data', markersize=10, color='red')
    
    # show tested price points
    points = zip(gpgo.GP.X.flatten(), gpgo.GP.y)
    for i, (x,y) in enumerate(points):
        if (i > 0):
            plt.text(x, y, '%d' % (i))
        
    plt.grid()
    plt.legend()


    # show where we think the optimal is
    xtest = np.linspace(0, 100, 200)[:, np.newaxis]
    a = [-gpgo._acqWrapper(np.atleast_2d(x)) for x in xtest]
    plt.subplot(3, 1, 2)
    plt.xlim(0,100)
    plt.plot(xtest, a, label='Integrated Expected Improvement')
    plt.grid()
    plt.legend()


    # show a historgram for the price points,
    # ideally we should have made a lot of offers near the optimal
    # and very few where we don't make revenue
    plt.subplot(3, 1, 3)
    plt.xlim(0,100)
    plt.hist(gpgo.GP.X.flatten(), 30, density=True, facecolor='g', alpha=0.75)
    plt.show()
    
    
    
    #gpgo.GP.posteriorPlot()
    print(gpgo.getResult())