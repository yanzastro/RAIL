"""
quick implementation of k nearest neighbor estimator
First pass will ignore photometric errors and just do
things in terms of magnitudes, we will expand in a
future update
"""

import numpy as np
import copy

from ceci.config import StageParameter as Param
from rail.estimation.estimator import Estimator, Informer

from rail.evaluation.metrics.cdeloss import CDELoss
from sklearn.neighbors import KDTree
import pandas as pd
import qp


def_bands = ['u', 'g', 'r', 'i', 'z', 'y']
refcols = [f"mag_{band}_lsst" for band in def_bands]
allcols = refcols.copy()
for band in def_bands:
    allcols.append(f"mag_{band}_lsst_err")
def_maglims = dict(mag_u_lsst=27.79,
                   mag_g_lsst=29.04,
                   mag_r_lsst=29.06,
                   mag_i_lsst=28.62,
                   mag_z_lsst=27.98,
                   mag_y_lsst=27.05)


def _computecolordata(df, ref_column_name, column_names):
    newdict = {}
    newdict['x'] = df[ref_column_name]
    nbands = len(column_names)-1
    for k in range(nbands):
        newdict[f'x{k}'] = df[column_names[k]] - df[column_names[k+1]]
    newdf = pd.DataFrame(newdict)
    coldata = newdf.to_numpy()
    return coldata


def _makepdf(dists, ids, szs, sigma):
    sigmas = np.full_like(dists, sigma)
    weights = 1./dists
    weights /= weights.sum(axis=1, keepdims=True)
    #norms = np.sum(weights, axis=1)
    means = szs[ids]
    pdfs = qp.Ensemble(qp.mixmod, data=dict(means=means, stds=sigmas, weights=weights))
    return pdfs


class Train_KNearNeighPDF(Informer):
    """Train a KNN-based estimator
    """
    name = 'Train_KNearNeighPDF'
    config_options = Informer.config_options.copy()
    config_options.update(zmin=Param(float, 0.0, msg="min z"),
                          zmax=Param(float, 3.0, msg="max_z"),
                          nzbins=Param(int, 301, msg="num z bins"),
                          trainfrac=Param(float, 0.75,
                                          msg="fraction of training data used to make tree, rest used to set best sigma"),
                          seed=Param(int, 0, msg="Random number seed for NN training"),
                          random_seed=Param(int, 87, msg="random seed for reproducibility"),
                          ref_column_name=Param(str, 'mag_i_lsst', msg="name for reference column"),
                          column_names=Param(list, refcols,
                                             msg="column names to be used in NN, *ASSUMED TO BE IN INCREASING WL ORDER!*"),
                          mag_limits=Param(dict, def_maglims, msg="1 sigma mag limits"),
                          sigma_grid_min=Param(float, 0.01, msg="minimum value of sigma for grid check"),
                          sigma_grid_max=Param(float, 0.075, msg="maximum value of sigma for grid check"),
                          ngrid_sigma=Param(int, 10, msg="number of grid points in sigma check"),
                          leaf_size=Param(int, 15, msg="min leaf size for KDTree"),
                          nneigh_min=Param(int, 3, msg="int, min number of near neighbors to use for PDF fit"),
                          nneigh_max=Param(int, 7, msg="int, max number of near neighbors to use ofr PDF fit"),
                          redshift_column_name=Param(str, 'redshift', msg="name of redshift column"))

    def __init__(self, args, comm=None):
        """ Constructor
        Do Informer specific initialization, then check on bands """
        Informer.__init__(self, args, comm=comm)

        usecols = self.config.column_names.copy()
        usecols.append(self.config.redshift_column_name)
        self.usecols = usecols
        self.zgrid = None

    def run(self):
        """
        train a KDTree on a fraction of the training data
        """
        if self.config.hdf5_groupname:
            training_data = self.get_data('input')[self.config.hdf5_groupname]
        else:  #pragma:  no cover
            training_data = self.get_data('input')
        input_df = pd.DataFrame(training_data)
        knndf = input_df[self.config.column_names]
        self.zgrid = np.linspace(self.config.zmin, self.config.zmax, self.config.nzbins)

        # replace nondetects
        # will fancy this up later with a flow to sample from truth
        for col in self.config.column_names:
            knndf.loc[np.isclose(knndf[col], 99.), col] = self.config.mag_limits[col]

        trainszs = np.array(input_df[self.config.redshift_column_name])
        colordata = _computecolordata(knndf, self.config.ref_column_name, self.config.column_names)
        nobs = colordata.shape[0]
        np.random.seed(self.config.seed)  # set seed for reproducibility
        perm = np.random.permutation(nobs)
        ntrain = round(nobs * self.config.trainfrac)
        xtrain_data = colordata[perm[:ntrain]]
        train_data = copy.deepcopy(xtrain_data)
        val_data = colordata[perm[ntrain:]]
        xtrain_sz = trainszs[perm[:ntrain]].copy()
        train_sz = np.array(copy.deepcopy(xtrain_sz))
        np.savetxt("TEMPZFILE.out", train_sz)
        val_sz = np.array(trainszs[perm[ntrain:]])
        print(f"split into {len(train_sz)} training and {len(val_sz)} validation samples")
        tmpmodel = KDTree(train_data, leaf_size=self.config.leaf_size)
        # Find best sigma and n_neigh by minimizing CDE Loss
        bestloss = 1e20
        bestsig = self.config.sigma_grid_min
        bestnn = self.config.nneigh_min
        siggrid = np.linspace(self.config.sigma_grid_min, self.config.sigma_grid_max, self.config.ngrid_sigma)
        print("finding best fit sigma and NNeigh...")
        for sig in siggrid:
            for nn in range(self.config.nneigh_min, self.config.nneigh_max+1):
                # print(f"sigma: {sig} num neigh: {nn}...")
                dists, idxs = tmpmodel.query(val_data, k=nn)
                ens = _makepdf(dists, idxs, train_sz, sig)
                cdelossobj = CDELoss(ens, self.zgrid, val_sz)
                cdeloss = cdelossobj.evaluate().statistic
                # print(f"sigma: {sig} num neigh: {nn} loss: {cdeloss}")
                if cdeloss < bestloss:
                    bestsig = sig
                    bestnn = nn
                    bestloss = cdeloss
        numneigh = bestnn
        sigma = bestsig
        print(f"\n\n\nbest fit values are sigma={sigma} and numneigh={numneigh}\n\n\n")
        # remake tree with full dataset!
        kdtree = KDTree(colordata, leaf_size=self.config.leaf_size)
        self.model = dict(kdtree=kdtree, bestsig=sigma, nneigh=numneigh, truezs=trainszs)
        self.add_data('model', self.model)



class KNearNeighPDF(Estimator):
    """KNN-based estimator
    """
    name = 'KNearNeighPDF'
    config_options = Estimator.config_options.copy()
    config_options.update(zmin=Param(float, 0.0, msg="min z"),
                          zmax=Param(float, 3.0, msg="max_z"),
                          nzbins=Param(int, 301, msg="num z bins"),
                          column_names=Param(list, refcols,
                                             msg="column names to be used in NN, *ASSUMED TO BE IN INCREASING WL ORDER!*"),
                          ref_column_name=Param(str, 'mag_i_lsst', msg="name for reference column"),
                          mag_limits=Param(dict, def_maglims, msg="1 sigma mag limits"),
                          redshift_column_name=Param(str, 'redshift', msg="name of redshift column"))

    def __init__(self, args, comm=None):
        """ Constructor:
        Do Estimator specific initialization """
        self.sigma = None
        self.numneigh = None
        self.model = None
        self.trainszs = None
        self.zgrid = None
        Estimator.__init__(self, args, comm=comm)
        usecols = self.config.column_names.copy()
        usecols.append(self.config.redshift_column_name)
        self.usecols = usecols

    def open_model(self, **kwargs):
        Estimator.open_model(self, **kwargs)
        self.sigma = self.model['bestsig']
        self.numneigh = self.model['nneigh']
        self.kdtree = self.model['kdtree']
        self.trainszs = self.model['truezs']

    def run(self):
        """
        calculate and return PDFs for each galaxy using the trained flow
        """
        # flow expects dataframe
        if self.config.hdf5_groupname:
            test_data = self.get_data('input')[self.config.hdf5_groupname]
        else:  #pragma:  no cover
            test_data = self.get_data('input')
        test_df = pd.DataFrame(test_data)
        knn_df = test_df[self.usecols]
        self.zgrid = np.linspace(self.config.zmin, self.config.zmax, self.config.nzbins)

        # replace nondetects
        for col in self.config.column_names:
            knn_df.loc[np.isclose(knn_df[col], 99.), col] = self.config.mag_limits[col]

        testcolordata = _computecolordata(knn_df, self.config.ref_column_name, self.config.column_names)
        dists, idxs = self.kdtree.query(testcolordata, k=self.numneigh)
        test_ens = _makepdf(dists, idxs, self.trainszs, self.sigma)
        zmode = test_ens.mode(grid=self.zgrid)
        test_ens.set_ancil(dict(zmode=zmode))
        self.add_data('output', test_ens)
