import numpy as np

def gaussian_chirplet(t, alpha1=1., alpha2=0., beta=2., fc=1., phi=0., tau=0.):
    """Gaussian chirplet

    Keyword arguments:
    t -- time
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