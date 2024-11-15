# -*- coding: utf-8 -*-
# ONAB: ONline ABsence detection
# Assess if absence seizures are present based on single-channel EEG data. 
# Input:
# sig: the single-channel EEG signal 
# fs: sampling frequency
# par1: low-frequency threshold
# par2: high-frequency (spike) threshold
#
# Output: 
# absence: 1D file with same length as input sig. 0 if no absence is detected, 1 if an absence event is detected. 

def onab(sig, fs, par1,par2):
    import numpy as np
    import math
    from scipy import signal
    from onab_support_fun import RMS_exclude
    import pywt
    from scipy.signal import find_peaks

    
    absence = np.zeros(len(sig))
    wind_size = 1.25 # window size
    delta = 0.125 # step size
    
    # define bandpass filter
    def bandPass_filter(lowcut, highcut, fs, order=3):
        nyq = 0.5*fs
        [low, high] = (lowcut/nyq, highcut/nyq)
        b,a = signal.butter(order, [low, high], btype= 'band')
        return b,a 
    
    # coefficients of bandpass filter
    bp_b, bp_a = bandPass_filter(0.5, 25, fs)
    
    # define resample fs
    fs_resamp = 64
    
    w = 5
    freq = np.logspace(np.log10(2.5), np.log10(20), 31)
    scales = w*fs_resamp / (2*freq*np.pi)
    
    # parameter settings
    f_low = np.arange(0,12) # low frequency range corresponding to 2.5-5.5Hz
    f_spike = [26] # corresponds to 15.2 Hz
    T_low = par1
    T_spike = par2
    T_samplerate = 3 # more than this number of spikes have to be detected
    
    i = wind_size
    
    
    for n in range(1,int(math.floor(len(sig)/(fs*wind_size)-1)*wind_size/delta)): # for each window
        currentWindow = sig[int((i-wind_size)*fs):int(i*fs)]
        # bandpass filter
        sig_filt = signal.filtfilt(bp_b, bp_a, currentWindow, padtype = 'odd', 
                                      padlen=3*(max(len(bp_b),len(bp_a))-1))
        # resample
        sig_res = signal.resample_poly(sig_filt, fs_resamp, fs, axis=0, 
                                       window=('kaiser', 5.0), padtype='constant', cval=None)
        # exclude high amplitude segmemts
        sig_rms = RMS_exclude(sig_res,fs_resamp)
        
        # compute CWT
        coef, freqs = pywt.cwt(sig_rms,scales,'morl',sampling_period=1/fs_resamp,method='fft')
        wt_work = np.absolute(coef)
        
        # Detection
        LF_tr = np.sum(wt_work[f_low,:] > T_low,axis=0) # compare with low-frequency threshold
        HF_env = wt_work[f_spike,LF_tr.astype(bool)] 
        if len(HF_env) > 3 and any(HF_env): 
            pks = HF_env[find_peaks(HF_env, distance=12)[0]] # find peaks within the envelope.
            pks_tr = sum(pks>T_spike) # how many peaks above threshold?
            if pks_tr > T_samplerate: # more than 3 peaks
                absence[int((i-wind_size)*fs):int(i*fs)] = 1 # absence detected
        i = i+delta
    
    return absence
    