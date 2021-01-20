# Abstract base class defining a generator,
# which represents a probability distribution

from abc import ABC, abstractmethod


class Generator(ABC):
    def __init__(self):

        return

    @abstractmethod
    def sample(self, n_samples, seed=None, **kwargs):
        """
        Return a random sample of the distribution with size n_samples
        """
        pass

    @abstractmethod
    def pz_estimate(self, data, grid, **kwargs):
        """
        Return redshift posteriors for the data
        """
        pass
