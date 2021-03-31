# V.2.5

import numpy as np
import matplotlib.pyplot as plt


# __all__ = ['TimeSampler']


class TimeSampler:
    """TimeSampler determines how and when samples will be taken from signal and noise.
    Samples timestamps for regular and irregular time signals

    Parameters
    ----------
    start_time: float/int (default 0)
                Time sampling of time series starts
    stop_time: float/int (default 10)
                Time sampling of time series stops

    """

    def __init__(self, start_time=0, stop_time=10):
        self.start_time = start_time
        self.stop_time = stop_time

    def sample_regular_time(self, num_points=None, resolution=None):
        """
        Samples regularly spaced time using the number of points or the
        resolution of the signal. Only one of the parameters is to be
        initialized. The resolution keyword argument is given priority.

        Parameters
        ----------
        num_points: int (default None)
            Number of points in time series
        resolution: float/int (default None)
            Resolution of the time series

        Returns
        -------
        numpy array
            Regularly sampled timestamps

        """
        if num_points is None and resolution is None:
            raise ValueError("One of the keyword arguments must be initialized.")
        if resolution is not None:
            time_vector = np.arange(self.start_time, self.stop_time,
                                    resolution)
            return time_vector
        else:
            time_vector = np.linspace(self.start_time, self.stop_time,
                                      num_points)
            return time_vector

    def sample_irregular_time(self, num_points=None, resolution=None,
                              keep_percentage=100):
        """
        Samples regularly spaced time using the number of points or the
        resolution of the signal. Only one of the parameters is to be
        initialized. The resolution keyword argument is given priority.

        Parameters
        ----------
        num_points: int (default None)
            Number of points in time series
        resolution: float/int (default None)
            Resolution of the time series
        keep_percentage: int(default 100)
            Percentage of points to be retained in the irregular series

        Returns
        -------
        numpy array
            Irregularly sampled timestamps

        """
        if num_points is None and resolution is None:
            raise ValueError("One of the keyword arguments must be initialized.")
        if resolution is not None:
            time_vector = np.arange(self.start_time, self.stop_time,
                                    resolution)
        else:
            time_vector = np.linspace(self.start_time, self.stop_time,
                                      num_points)
            resolution = float(self.stop_time-self.start_time)/num_points
        time_vector = self._select_random_indices(time_vector,
                                                  keep_percentage)
        return self._create_perturbations(time_vector, resolution)

    def _create_perturbations(self, time_vector, resolution):
        """
        Internal functions to create perturbations in timestamps

        Parameters
        ----------
        time_vector: numpy array
            timestamp vector

        resolution: float/int
            resolution of the time series

        Returns
        -------
        numpy array
            Irregularly sampled timestamps with perturbations

        """
        sample_perturbations = np.random.normal(loc=0.0, scale=resolution,
                                                size=len(time_vector))
        time_vector = time_vector + sample_perturbations
        return np.sort(time_vector)

    def _select_random_indices(self, time_vector, keep_percentage):
        """
        Internal functions to randomly select timestamps

        Parameters
        ----------
        time_vector: numpy array
            timestamp vector

        keep_percentage: float/int
            percentage of points retained

        Returns
        -------
        numpy array
            Irregularly sampled timestamps

        """
        num_points = len(time_vector)
        num_select_points = int(keep_percentage*num_points/100)
        index = np.sort(np.random.choice(num_points, size=num_select_points,
                                         replace=False))
        return time_vector[index]


class TimeSeries:
    """A TimeSeries object is the main interface from which to sample time series.
    You have to provide at least a signal generator; a noise generator is optional.
    It is recommended to set the sampling frequency.

    Parameters
    ----------
    signal_generator : Signal object
        signal object for time series
    noise_generator : Noise object
        noise object for time series

    """

    def __init__(self, signal_generator, noise_generator=None):
        self.signal_generator = signal_generator
        self.noise_generator = noise_generator

    def sample(self, time_vector):
        """Samples from the specified TimeSeries.

        Parameters
        ----------
        time_vector : numpy array
            Times at which to generate a sample

        Returns
        -------
        samples, signals, errors, : tuple (array, array, array)
            Returns samples, and the signals and errors they were constructed from
        """

        # Vectorize if possible
        if self.signal_generator.vectorizable and not self.noise_generator is None and self.noise_generator.vectorizable:
            signals = self.signal_generator.sample_vectorized(time_vector)
            errors = self.noise_generator.sample_vectorized(time_vector)
            samples = signals + errors
        elif self.signal_generator.vectorizable and self.noise_generator is None:
            signals = self.signal_generator.sample_vectorized(time_vector)
            errors = np.zeros(len(time_vector))
            samples = signals
        else:
            n_samples = len(time_vector)
            samples = np.zeros(n_samples)  # Signal and errors combined
            signals = np.zeros(n_samples)  # Signal samples
            errors = np.zeros(n_samples)  # Handle errors seprately
            times = np.arange(n_samples)

            # Sample iteratively, while providing access to all previously sampled steps
            for i in range(n_samples):
                # Get time
                t = time_vector[i]
                # Sample error
                if not self.noise_generator is None:
                    errors[i] = self.noise_generator.sample_next(t, samples[:i - 1], errors[:i - 1])

                # Sample signal
                signal = self.signal_generator.sample_next(t, samples[:i - 1], errors[:i - 1])
                signals[i] = signal

                # Compound signal and noise
                samples[i] = signals[i] + errors[i]

        # Return both times and samples, as well as signals and errors
        return samples, signals, errors


class BaseNoise:
    """BaseNoise class

    Signature for all noise classes.

    """

    def __init__(self):
        raise NotImplementedError

    def sample_next(self, t, samples, errors):  # We provide t for irregularly sampled timeseries
        """Samples next point based on history of samples and errors

        Parameters
        ----------
        t : int
            time
        samples : array-like
            all samples taken so far
        errors : array-like
            all errors sampled so far

        Returns
        -------
        float
            sampled error for time t

        """
        raise NotImplementedError


class BaseSignal:
    """BaseSignal class

    Signature for all signal classes.

    """

    def __init__(self):
        raise NotImplementedError

    def sample_next(self, time, samples, errors):
        """Samples next point based on history of samples and errors

        Parameters
        ----------
        time : int
            time
        samples : array-like
            all samples taken so far
        errors : array-like
            all errors sampled so far

        Returns
        -------
        float
            sampled signal for time t

        """
        raise NotImplementedError

    def sample_vectorized(self, time_vector):
        """Samples for all time points in input

        Parameters
        ----------
        time_vector : array like
            all time stamps to be sampled

        Returns
        -------
        float
            sampled signal for time t

        """
        raise NotImplementedError


class GaussianNoise(BaseNoise):
    """Gaussian noise generator.
    This class adds uncorrelated, additive white noise to your signal.

    Attributes
    ----------
    mean : float
        mean for the noise
    std : float
        standard deviation for the noise

    """

    def __init__(self, mean=0, std=1.):
        self.vectorizable = True
        self.mean = mean
        self.std = std

    def sample_next(self, t, samples, errors):
        return np.random.normal(loc=self.mean, scale=self.std, size=1)

    def sample_vectorized(self, time_vector):
        n_samples = len(time_vector)
        return np.random.normal(loc=self.mean, scale=self.std, size=n_samples)


class Sinusoidal(BaseSignal):
    """Signal generator for harmonic (sinusoidal) waves.

    Parameters
    ----------
    amplitude : number (default 1.0)
        Amplitude of the harmonic series
    frequency : number (default 1.0)
        Frequency of the harmonic series
    ftype : function (default np.sin)
        Harmonic function

    """

    def __init__(self, amplitude=1.0, frequency=1.0, ftype=np.sin):
        self.vectorizable = True
        self.amplitude = amplitude
        self.ftype = ftype
        self.frequency = frequency

    def sample_next(self, time, samples, errors):
        """Sample a single time point

        Parameters
        ----------
        time : number
            Time at which a sample was required

        Returns
        -------
        float
            sampled signal for time t

        """
        return self.amplitude * self.ftype(2*np.pi*self.frequency*time)

    def sample_vectorized(self, time_vector):
        """Sample entire series based off of time vector

        Parameters
        ----------
        time_vector : array-like
            Timestamps for signal generation

        Returns
        -------
        array-like
            sampled signal for time vector

        """
        if self.vectorizable is True:
            signal = self.amplitude * self.ftype(2*np.pi*self.frequency *
                                                 time_vector)
            return signal
        else:
            raise ValueError("Signal type not vectorizable")


class GaussianProcess(BaseSignal):
    """Gaussian Process time series sampler

    Samples time series from Gaussian Process with selected covariance function (kernel).

    Parameters
    ----------
    kernel : {'SE', 'Constant', 'Exponential', 'RQ', 'Linear', 'Matern', 'Periodic'}
        the kernel type, as described in [1]_ and [2]_, which can be:

        - `Constant`. All covariances set to `variance`
        - `Exponential`. Ornstein-Uhlenbeck kernel. Optionally, set keyword `gamma` for a gamma-exponential kernel
        - `SE`, the squared exponential.
        - `RQ`, the rational quadratic. To use this kernel, set keyword argument `alpha`
        - `Linear`. To use this kernel, set keyword arguments `c` and `offset`
        - `Matern`. To use this kernel, set keyword argument `nu`
        - `Periodic`. To use this kernel, set keyword argument `p` for the period

    mean : float
        the mean of the gaussian process
    variance : float
        the output variance of the gaussian process (sigma^2)
    lengthscale : float
            the characteristic lengthscale used to generate the covariance matrix

    References
    ----------
    .. [1] URL: http://www.cs.toronto.edu/~duvenaud/cookbook/index.html
    .. [2] Rasmussen, C.E., 2006. Gaussian processes for machine learning. URL: https://pdfs.semanticscholar.org/a9fe/ab0fe858dbde2eecff8b1f7c629cc6aff8ad.pdf

    """

    def __init__(self, kernel="SE", lengthscale=1., mean=0., variance=1., c=1., gamma=1., alpha=1., offset=0., nu=5./2, p=1.):
        self.vectorizable = True
        self.lengthscale = lengthscale
        self.mean = mean
        self.variance = variance
        self.kernel = kernel
        self.kernel_function = {"Constant": lambda x1, x2: variance,
                                "Exponential": lambda x1, x2: variance * np.exp(-np.power(np.abs(x1 - x2) / lengthscale, gamma)),
                                "SE": lambda x1, x2: variance * np.exp(- np.square(x1 - x2) / (2 * np.square(lengthscale))),
                                "RQ": lambda x1, x2: variance * np.power((1 + np.square(x1 - x2) / (2 * alpha * np.square(lengthscale))), -alpha),
                                "Linear": lambda x1, x2: variance * (x1 - c) * (x2 - c) + offset,
                                "Matern": lambda x1, x2: variance if x1 - x2 == 0. else variance * (np.power(2, 1 - nu) / scipy.special.gamma(nu)) * np.power(np.sqrt(2 * nu) * np.abs(x1 - x2) / lengthscale, nu) * scipy.special.kv(nu, np.sqrt(2 * nu) * np.abs(x1 - x2) / lengthscale),
                                "Periodic": lambda x1, x2: variance * np.exp(- 2 * np.square(np.sin(np.pi * np.abs(x1 - x2) / p))),
                                }[kernel]

    def sample_next(self, time, samples, errors):
        """Sample a single time point

        Parameters
        ----------
        time : number
            Time at which a sample was required

        Returns
        -------
        float
            sampled signal for time t

        """
        raise NotImplementedError

    def sample_vectorized(self, time_vector):
        """Sample entire series based off of time vector

        Parameters
        ----------
        time_vector : array-like
            Timestamps for signal generation

        Returns
        -------
        array-like
            sampled signal for time vector

        """
        cartesian_time = np.dstack(np.meshgrid(time_vector, time_vector)).reshape(-1, 2)
        covariance_matrix = (np.vectorize(self.kernel_function)(
            cartesian_time[:, 0], cartesian_time[:, 1])).reshape(-1, time_vector.shape[0])
        # Add small value to diagonal for numerical stability
        covariance_matrix[np.diag_indices_from(covariance_matrix)] += 1e-12
        return np.random.multivariate_normal(mean=np.full(shape=(time_vector.shape[0],), fill_value=self.mean), cov=covariance_matrix)


time_sampler = TimeSampler(stop_time=20)
print(time_sampler)
irregular_time_samples = time_sampler.sample_irregular_time(num_points=500, keep_percentage=50)
print(irregular_time_samples)
sinusoid = Sinusoidal(frequency=0.25)
print(sinusoid)
white_noise = GaussianNoise(std=0.25)
print(white_noise)
timeseries = TimeSeries(sinusoid, noise_generator=white_noise)
print(timeseries)
samples, signals, errors = timeseries.sample(irregular_time_samples)

# NOTE: plot samples for ramdomness


# plt.figure(figsize=(1300, 525))  # figma


plt.plot(irregular_time_samples, samples, marker='o')
plt.xlabel('Time')
plt.ylabel('Magnitude')
plt.title('Irregularly sampled sinusoid with noise')
plt.show()


# plt.plot(irregular_time_samples, errors, marker='o')
# plt.xlabel('Time')
# plt.ylabel('Magnitude')
# plt.title('Pseudoperiodic signal')
# plt.show()


# print(samples)
# print(signals)
# print(errors)
