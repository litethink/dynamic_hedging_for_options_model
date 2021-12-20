import pandas as pd
import numpy as np
import scipy.interpolate as sci
from scipy.optimize import brute
from scipy.optimize import fmin

t_list = np.array((1, 7, 14, 30, 60, 90, 180, 270, 360)) / 360.
r_list = np.array((-0.032, -0.013, -0.013, 0.007, 0.043,
                   0.083, 0.183, 0.251, 0.338)) / 100

factors = (1 + t_list * r_list)
zero_rates = 1 / t_list * np.log(factors)
tck = sci.splrep(t_list, zero_rates, k=3)  # cubic splines
tn_list = np.linspace(0.0, 1.0, 24)
ts_list = sci.splev(tn_list, tck, der=0)
de_list = sci.splev(tn_list, tck, der=1)
f = ts_list + de_list * tn_list

r0 = r_list[0]

def CIR_forward_rate(opt):
    ''' Function for forward rates in CIR85 model.

    Parameters
    ==========
    kappa_r: float
        mean-reversion factor
    theta_r: float
        long-run mean
    sigma_r: float
        volatility factor

    Returns
    =======
    forward_rate: float
        forward rate
    '''
    kappa_r, theta_r, sigma_r = opt
    t = tn_list
    g = np.sqrt(kappa_r ** 2 + 2 * sigma_r ** 2)
    sum1 = ((kappa_r * theta_r * (np.exp(g * t) - 1)) /
            (2 * g + (kappa_r + g) * (np.exp(g * t) - 1)))
    sum2 = r0 * ((4 * g ** 2 * np.exp(g * t)) /
                 (2 * g + (kappa_r + g) * (np.exp(g * t) - 1)) ** 2)
    forward_rate = sum1 + sum2
    return forward_rate

#
# Error Function
#


def CIR_error_function(opt):
    ''' Error function for CIR85 model calibration. '''
    kappa_r, theta_r, sigma_r = opt
    if 2 * kappa_r * theta_r < sigma_r ** 2:
        return 100
    if kappa_r < 0 or theta_r < 0 or sigma_r < 0.001:
        return 100
    forward_rates = CIR_forward_rate(opt)
    # import pdb
    # pdb.set_trace()
    MSE = np.sum((f - forward_rates) ** 2) / len(f)
    # print opt, MSE
    return MSE

#
# Calibration Procedure
#
def CIR_calibration():
    opt = fmin(CIR_error_function, [1.0, 0.02, 0.1],
               xtol=0.00001, ftol=0.00001,
               maxiter=300, maxfun=500)
    return opt


def gamma(kappa_r, sigma_r):
    ''' Help Function. '''
    return np.sqrt(kappa_r ** 2 + 2 * sigma_r ** 2)


def b1(alpha):
    ''' Help Function. '''
    kappa_r, theta_r, sigma_r, r0, T = alpha
    g = gamma(kappa_r, sigma_r)
    return (((2 * g * math.exp((kappa_r + g) * T / 2)) /
             (2 * g + (kappa_r + g) * (math.exp(g * T) - 1))) **
            (2 * kappa_r * theta_r / sigma_r ** 2))


def b2(alpha):
    ''' Help Function. '''
    kappa_r, theta_r, sigma_r, r0, T = alpha
    g = gamma(kappa_r, sigma_r)
    return ((2 * (math.exp(g * T) - 1)) /
            (2 * g + (kappa_r + g) * (math.exp(g * T) - 1)))


def B(alpha):
    ''' Function to value unit zero-coupon bonds in Cox-Ingersoll-Ross (1985)
    model.

    Parameters
    ==========
    r0: float
        initial short rate
    kappa_r: float
        mean-reversion factor
    theta_r: float
        long-run mean of short rate
    sigma_r: float
        volatility of short rate
    T: float
        time horizon/interval

    Returns
    =======
    zcb_value: float
        zero-coupon bond present value
    '''
    b_1 = b1(alpha)
    b_2 = b2(alpha)
    kappa_r, theta_r, sigma_r, r0, T = alpha
    return b_1 * math.exp(-b_2 * r0)
  
kappa_r, theta_r, sigma_r = CIR_calibration()

def H93_call_value(S0, K, T, r, kappa_v, theta_v, sigma_v, rho, v0):
    ''' Valuation of European call option in H93 model via Lewis (2001)
    Fourier-based approach.

    Parameters
    ==========
    S0: float
        initial stock/index level
    K: float
        strike price
    T: float
        time-to-maturity (for t=0)
    r: float
        constant risk-free short rate
    kappa_v: float
        mean-reversion factor
    theta_v: float
        long-run mean of variance
    sigma_v: float
        volatility of variance
    rho: float
        correlation between variance and stock/index level
    v0: float
        initial level of variance

    Returns
    =======
    call_value: float
        present value of European call option

    '''
    int_value = quad(lambda u: H93_int_func(u, S0, K, T, r, kappa_v,
                                            theta_v, sigma_v, rho, v0),
                     0, np.inf, limit=250)[0]
    call_value = max(0, S0 - np.exp(-r * T) * np.sqrt(S0 * K) /
                     np.pi * int_value)
    return call_value



