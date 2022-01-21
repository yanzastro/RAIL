import numpy as np
import pandas as pd
from rail.creation.degradation import Degrader


class LineConfusion(Degrader):
    """Degrader that simulates emission line confusion.

    Example: degrader = LineConfusion(true_wavelen=3727,
                                      wrong_wavelen=5007,
                                      frac_wrong=0.05)
    is a degrader that misidentifies 5% of OII lines (at 3727 angstroms)
    as OIII lines (at 5007 angstroms), which results in a larger
    spectroscopic redshift .

    Note that when selecting the galaxies for which the lines are confused,
    the degrader ignores galaxies for which this line confusion would result
    in a negative redshift, which can occur for low redshift galaxies when
    wrong_wavelen < true_wavelen.
    """

    def __init__(self, true_wavelen: float, wrong_wavelen: float, frac_wrong: float):
        """
        Parameters
        ----------
        true_wavelen : positive float
            The wavelength of the true emission line.
            Wavelength unit assumed to be the same as wrong_wavelen.
        wrong_wavelen : positive float
            The wavelength of the wrong emission line, which is being confused
            for the correct emission line.
            Wavelength unit assumed to be the same as true_wavelen.
        frac_wrong : float between zero and one
            The fraction of galaxies with confused emission lines.
        """

        # convert to floats
        true_wavelen = float(true_wavelen)
        wrong_wavelen = float(wrong_wavelen)
        frac_wrong = float(frac_wrong)

        # validate parameters
        if true_wavelen < 0:
            raise ValueError("true_wavelen must be positive")
        if wrong_wavelen < 0:
            raise ValueError("wrong_wavelen must be positive")
        if frac_wrong < 0 or frac_wrong > 1:
            raise ValueError("frac_wrong must be between 0 and 1.")

        self.true_wavelen = true_wavelen
        self.wrong_wavelen = wrong_wavelen
        self.frac_wrong = frac_wrong

    def __call__(self, data: pd.DataFrame, seed: int = None) -> pd.DataFrame:
        """
        Parameters
        ----------
        data : pd.DataFrame
            DataFrame of galaxy data to be degraded.
        seed : int, default=None
            Random seed for the degrader.

        Returns
        -------
        pd.DataFrame
            DataFrame of the degraded galaxy data.
        """

        # convert to an array for easy manipulation
        values, columns = data.values.copy(), data.columns.copy()

        # get the minimum redshift
        # if wrong_wavelen < true_wavelen, this is minimum the redshift for
        # which the confused redshift is still positive
        zmin = self.wrong_wavelen / self.true_wavelen - 1

        # select the random fraction of galaxies whose lines are confused
        rng = np.random.default_rng(seed)
        idx = rng.choice(
            np.where(values[:, 0] > zmin)[0],
            size=int(self.frac_wrong * values.shape[0]),
            replace=False,
        )

        # transform these redshifts
        values[idx, 0] = (
            1 + values[idx, 0]
        ) * self.true_wavelen / self.wrong_wavelen - 1

        # return results in a data frame
        return pd.DataFrame(values, columns=columns)


class InvRedshiftIncompleteness(Degrader):
    """Degrader that simulates incompleteness with a selection function
    inversely proportional to redshift.

    The survival probability of this selection function is
    p(z) = min(1, z_p/z),
    where z_p is the pivot redshift.
    """

    def __init__(self, pivot_redshift):
        """
        Parameters
        ----------
        pivot_redshift : positive float
            The redshift at which the incompleteness begins.
        """
        pivot_redshift = float(pivot_redshift)
        if pivot_redshift < 0:
            raise ValueError("pivot redshift must be positive.")

        self.pivot_redshift = pivot_redshift

    def __call__(self, data: pd.DataFrame, seed: int = None) -> pd.DataFrame:
        """
        Parameters
        ----------
        data : pd.DataFrame
            DataFrame of galaxy data to be degraded.
        seed : int, default=None
            Random seed for the degrader.

        Returns
        -------
        pd.DataFrame
            DataFrame of the degraded galaxy data.
        """

        # calculate survival probability for each galaxy
        survival_prob = np.clip(self.pivot_redshift / data["redshift"], 0, 1)

        # probabalistically drop galaxies from the data set
        rng = np.random.default_rng(seed)
        mask = rng.random(size=data.shape[0]) <= survival_prob

        return data[mask]


class DEEP2Selection(Degrader):
    """Degrader that models the color selections used for DEEP2 specz
    sample selection.  Cuts are originally in B-R and R-I, we will
    need to transform to g-r and r-i with some color transforms

    Selection is essentially just binary flag where we keep objects if:
    g-r <= gr_cut OR r-i >= ri_cut OR g-r <= gr_ri_slope * r-i + gr_ri_offset
    """

    def __init__(self, g_name="mag_g_lsst", r_name="mag_r_lsst",
                 i_name="mag_i_lsst", gr_cut=0.25, ri_cut=1.0,
                 gr_ri_slope=2.27, gr_ri_offset=-0.27,
                 tanh_zeropt=23.7, tanh_width=0.35):
        """
        Parameters
        ----------
        g_name : str
          name of g-filter column in dataframe
        r_name : str
          name of r-filter column in dataframe
        i_name : str
          name of i-filter column in dataframe
        gr_cut : float, default = 0.25
          color cut for g-r color
        ri_cut : float, default = 1.0
          color cut for ri- color
        gr_ri_slope : float, default = 2.27
          slope of joint r-i vs g-r cut
        gr_ri_offset : float, default = - 0.27
          offset in r-i vs g-r cut
        tanh_zeropt : float, default = 23.7
          value where random selection of imag "success" goes to zero
        tanh_width : float, default = 1.5
          scaling factor controlling speed of tanh specz "success"

        Returns
        -------
        pd.DataFrame
          Dataframe of DEEP2 mock spec-z selection
        """
        self.g_name = g_name
        self.r_name = r_name
        self.i_name = i_name
        self.gr_cut = gr_cut
        self.ri_cut = ri_cut
        self.gr_ri_slope = gr_ri_slope
        self.gr_ri_offset = gr_ri_offset
        self.tanh_zeropt = tanh_zeropt
        self.tanh_width = tanh_width

    def __call__(self, data: pd.DataFrame, seed: int) -> pd.DataFrame:
        """
        Parameters
        ----------
        data : pd.DataFrame
          Dataframe of galaxy data
        seed : int
          random int

        Returns
        -------
        pd.DataFrame
          Dataframe of DEEP2 mock spec-z selection
        """
        gr = data[self.g_name] - data[self.r_name]
        ri = data[self.r_name] - data[self.i_name]
        # mask out the color selection in gri space to model DEEP2-like cuts
        color_mask = np.logical_or(gr <= self.gr_cut,
                                   np.logical_or(ri >= self.ri_cut,
                                                 gr <= self.gr_ri_slope * ri + self.gr_ri_offset))

        # generate random numbers to model fall off in specz completeness
        rng = np.random.default_rng(seed)
        tanh_mask = rng.random(size=data.shape[0]) <= np.tanh(self.tanh_width * (self.tanh_zeropt - data[self.i_name]))

        mask = np.logical_and(color_mask, tanh_mask)

        return data[mask]
