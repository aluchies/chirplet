import numpy as np
from scipy.signal import hilbert
from scipy.fftpack import fft, fftfreq
from cmath import phase

def gaussian_chirplet(t, alpha1=1., alpha2=0., beta=2., fc=1., phi=0., tau=0.):
    """Gaussian chirplet function given input parameters

    Keyword arguments:
    t -- time vector
    alpha1 -- bandwidth factor
    alpha2 -- chirp-rate
    beta -- amplitude
    fc -- center frequency
    phi -- phase
    tau -- time of arrival

    Return values:
    y -- chirplet

    Reference: Yufeng Lu, Ramazan Demirli, Guilherme Cardosa, and Jafar Saniie,
    "A Successive Parameter estimation Algorithm for Chirplet Signal
    Decomposition," IEEE Trans. UFFC, vol 53, no 11, November 2006.

    """

    y = beta * np.exp(-alpha1 * (t-tau) ** 2 + 1j * 2 * np.pi * fc * (t - tau)
        + 1j * phi + 1j * alpha2 * (t - tau) ** 2)
    
    return y





def estimate_gaussian_chirplet_parameters(y, t):
    """Estimate Gaussian chriplet parameters for the provided signal segment.
    Parameters are estimated in the following order:

    tau, beta, phi, fc, alpha2, alpha1

    Keyword arguments:
    y -- signal segment
    t -- time vector

    Return values:
    parameters -- dictionary of parameter values
        alpha1 -- bandwidth factor
        alpha2 -- chirp-rate
        beta -- amplitude
        fc -- center frequency
        phi -- phase
        tau -- time of arrival

    """

    # Determine sampling frequency
    dt = t[1] - t[0]
    fs = 1 / dt

    # Estimate tau and beta from envelope
    env = abs(hilbert(y))
    index = np.argmax(env)
    tau = t[index]
    beta = env[index]

    # Estimate phi using tau
    analytic = hilbert(y)
    phi = phase(analytic[index] / beta)

    # Estimate center frequency by finding peak in spectrum
    S = abs(fft(y))
    freq = fftfreq(S.size) * fs
    index = np.argmax(S)
    fc = freq[index]

    # Estimate alpha2 (chirp rate)
    N = 2 ** 7
    coefs = np.zeros(N)
    for i, alpha2 in enumerate(np.linspace(0, 2, N)):
        kernel = np.real(gaussian_chirplet(t, alpha1=1.0, alpha2=alpha2,
            beta=beta, fc=fc, phi=phi, tau=tau))
        kernel = kernel / np.sqrt(np.sum(kernel ** 2))
        coefs[i] = y.dot(kernel)

    index = np.argmax(np.real(coefs))
    alpha2 = np.linspace(0, 2, N)[index]

    # Estimate alpha1 (bandwidth factor)
    N = 2 ** 7
    coefs = np.zeros(N)
    for i, alpha1 in enumerate(np.linspace(0, 2, N)):
        kernel = np.real(gaussian_chirplet(t, alpha1=alpha1,
            alpha2=alpha2, beta=beta, fc=fc, phi=phi, tau=tau))
        kernel = kernel / np.sqrt(np.sum(kernel ** 2))
        coefs[i] = y.dot(kernel)

    index = np.argmax(np.real(coefs))
    alpha1 = np.linspace(0, 2, N)[index]

    return {'alpha1':alpha1, 'alpha2':alpha2, 'beta':beta, 'fc':fc, 'phi':phi, 'tau':tau}