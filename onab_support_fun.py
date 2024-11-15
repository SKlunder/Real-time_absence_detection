# -*- coding: utf-8 -*-
# support functions for onab function

import math
import numpy as np

def RMS_exclude(sig,fs): # set any 100 ms segment to zero that has a RMS > 700 microV
    # sig: 1D signal
    # fs: sample frequency
    def rolling_rms(x, N):
        xc = np.cumsum(abs(x)**2);
        return np.sqrt((xc[N:] - xc[:-N]) / N)
    
    wind_rms = 0.1
    rms100 = rolling_rms(sig,math.ceil(wind_rms*fs))
    rmsind = rms100>700e-6 # threshold 
    if any(rmsind):
        sig_rms = np.zeros(len(sig))
    else:
        sig_rms = sig
    return sig_rms
 