import numpy as np
import scipy.fft as fft
import scipy.signal as sig
import matplotlib.pyplot as plt
import time
import sys

def sim_transmission(signal, OSNR):
    #number of amplifiers
    n_amp = 128
    #noise figure
    NF = 6
    #SNR at end
    SNR = OSNR - (n_amp * NF)
    #power for last link
    power = 18 - 40 * 0.18
    lin_power = 10 ** (power/10)
    #calculate power of signal
    s_power = np.sum(np.square(abs(signal))) * timestep/ts
    #normalize signal power and multiply by linear power
    signal = 1/np.sqrt(2 * s_power) * np.sqrt(2 * lin_power * 10 ** -3) * signal
    s_power = np.sum(np.square(abs(signal))) * timestep/ts
    return signal, SNR

def OFDMtrans():
    syms = generate_syms(channels)
    inverse = fft.ifft(syms)
    #ratio of the length of the number of samples in the carrier to the length of the IFFT
    r = int(np.rint(N/channels))
    #each symbol from the ifft is tiled so that the length matches the length of the carrier 
    inverse = inverse.reshape((channels, 1))
    inverse = np.tile(inverse, (1, r))
    inverse = inverse.flatten()

    #multiplied by this to give power of 1 mW
    transmit = np.cos(2 * np.pi * fc * t) * inverse 
    plt.figure()
    plt.plot(t, transmit)
    plt.title("Optical OFDM signal in time domain")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    ##########################
    # plt.figure()
    # plt.title("Transmitted")
    # spec = fft.fft(transmit, norm = 'ortho',)
    # freq = fft.fftfreq(len(transmit),  d = timestep)
    # plt.plot(freq, abs(spec))
    # plt.axis([1.917 * 10 ** 14, 1.961 * 10 ** 14, 0, 1.1 * np.max(abs(spec))])
    ###########################
    return transmit, syms 

def OFDMrecv(signal, SNR):
    recv = signal * np.cos(2 * np.pi * fc * t)
    low_pass = sig.butter(2, rate * channels, btype = 'low', output = 'sos', fs = 1/timestep)
    recv = sig.sosfilt(low_pass, recv)
    #########################
    # plt.figure()
    # spec = fft.fft(recv, norm = 'ortho')
    # plt.title("Received")
    # freq = fft.fftfreq(len(spec), d = timestep)
    # plt.plot(freq, abs(spec))
    # plt.axis([0, rate * channels, 0, 1.1 * np.max(abs(spec))])
    #########################
    r = int(N/channels)
    det = np.zeros(channels, dtype = complex)

    #calculate the amount of noise to add based on the received signal power and the SNR
    p_rec = np.sum(np.square(abs(recv)))/(len(recv))
    p_noise = p_rec/(10 ** (SNR/10))
    real_noise = np.random.normal(0, np.sqrt(p_noise/2), len(recv))
    complex_noise = 1j*np.random.normal(0, np.sqrt(p_noise/2), len(recv))
    recv = recv + real_noise + complex_noise

    for i in range(0, channels):
        #take sample 
        det[i] = recv[r * i + int(r/2)]
    det = fft.fft(det)

    #decode bits
    for i in range(0, len(det)):
        if (det[i].real > 0):
            a = 1
        else:
            a = -1
        if(det[i].imag > 0):
            b = 1
        else:
            b = -1
        det[i] = a + b *1j
    return det

def generate_syms(channels):
    syms = np.zeros(channels, dtype = complex)
    mapping = {(0,0) : 1 + 1j, (0,1) : 1 - 1j, (1,0) : -1 - 1j, (1,1) : -1 + 1j}
    for i in range(0, channels):
        random = np.random.randint(2, size = 2)
        c_syms = mapping[tuple(random)]
        syms[i] = c_syms
    return syms

channels = 88
timestep = 1 * 10 ** -16
#centre frequency in C-band

fc = 1.917 * 10 ** 14
rate = (4.4 * 10 ** 12)/channels
ts = channels/rate
power = [250]
G = 10 ** 9
N = int(np.rint(ts/timestep))
t = np.linspace(0, ts, N)
frames = 1
p = 343
OSNR = np.sqrt(p * 10 ** -3/(6.626 * 10 ** -34 * 1.94 * 10 ** 14 * 4400 * G))
print("Input power (mW)", p)
print("OSNR is", OSNR)
errors = 0
for i in range(0, frames):
    signal, truth = OFDMtrans()
    signal, SNR = sim_transmission(signal, OSNR)
    recovered = OFDMrecv(signal, SNR)
    errors = errors + np.count_nonzero(truth - recovered)
 

print("SNR at receiver", SNR)
print("BER", errors/(frames * channels))
plt.show()

