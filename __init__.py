from __future__ import division

import os
import numpy as np
from scipy.interpolate import UnivariateSpline as ius
from scipy import integrate
from scipy.integrate import odeint
from math import exp,log,sqrt,pi,log10
from numpy.ctypeslib import ndpointer
import ctypes as ct

from sys import platform as _platform
import copy

import timeit
from classy import Class

'''
To run the C library on Linux: gcc -shared -o clib.so clib.c -lgsl -lgslcblas -lm -std=c99 -fPIC -L/usr/local/lib/ -fopenmp
'''


cosmo = Class()

params = {'output':'mPk mTk', 'z_max_pk': 2., 'P_k_max_1/Mpc': 10., 'h' : 0.69, 'Omega_b': 0.04900, 'Omega_cdm' : 0.232,
            'A_s' : 2.25e-9, 'n_s' : 0.962, 'k_per_decade_for_pk' : 50}

# set test cosmology
cosmo.set(params)
cosmo.compute()


class rx_clusters:


    def __init__(self):

        path = os.getcwd()

        #self.cluster_data = np.loadtxt('data_codex.txt') #columns:  z  Lx0124  eLx  LAMBDACHISQ  LAMBDACHISQE

        """
        NUISANCE PARAMETERS
        --------------------
        either fixed to their best-fit values or varied by MontePython
        --------------------
        sigLx, siglam: scaling relations
        norm_:
        """
        #self.sigLx = 0.4
        #self.siglam = 0.35
        #self.norm_ = 10**14.2


        """
        COMMON VARIABLES
        --------------------
        """

        self.logMmin = 12.5
        self.logMmax = 16.1
        self.Msteps = 100
        self.zsteps = 32

        #cluster_all = self.cluster_data[self.cluster_data[:,0]>0.]
        # choose number of bins
        self.n_lambins = 8  # Lx?
        self.n_zbins = 5
        self.n_lxbins = 8
        self.n_etaObbins = 14

        self.lambins = np.array([ 30., 40., 54., 72., 96., 129., 173., 232., 310.])
        self.lxbins = np.log(np.array([1e42, 5e42, 1e43, 5e43, 1e44, 5e44, 1e45, 5e45, 1e46]))
        self.zbins = np.arange(0.1, 0.601, 0.1)
        self.etaObbins = np.round(np.logspace(np.log10(4), np.log10(1004), 15))
        # self.data_counts, self.qbins, self.zbins = np.histogram2d(np.log10(cluster_all[:,2]),cluster_all[:,0],(self.n_qbins,self.n_zbins))
        #self.data_counts, self.etaObbins, self.zbins = np.histogram2d(np.log(cluster_all[:,3]),cluster_all[:,0],(self.etaObbins,self.zbins))

        """
        SIGMA(M,z) COMPUTATION
        --------------------
        """

        time_start = timeit.default_timer()

        self.z_array = np.linspace(0.,1.1,self.zsteps)
        self.sigma_M_z_array = np.zeros((self.Msteps,self.zsteps))
        self.M_array = np.logspace(self.logMmin,self.logMmax,base=10,num=self.Msteps,endpoint=False)
        self.sigma_arrayz0 = np.zeros(self.Msteps)

        self.rho_crit = 2.7751973751261264e11; # in M_sun/h / (Mpc/h)**3
        self.R_array = (3. * self.M_array / 4. / np.pi / self.rho_crit / cosmo.Omega0_m() )**(1./3.) # Mpc / h

        #read transfer functions and wavenumbers, exclude edges to evade interpolation errors
        self.k_array = cosmo.get_transfer()['k (h/Mpc)'][1:-1] #!
        self.T_cdm = cosmo.get_transfer()['d_cdm'][1:-1] #!
        self.T_m = cosmo.get_transfer()['d_tot'][1:-1] #!
        self.T_b = cosmo.get_transfer()['d_b'][1:-1] #!
        self.p_cdm = np.zeros_like(self.k_array)

        # equation of state of DE for growth equation
        self.w0_fld = -1.
        self.wa_fld = 0.
        # solve for growth
        # start at a = 1./10. where Omega_m ~ 1
        a_array = np.linspace(1./10., 1, 1000)
        # g(a_ini) = D/a = 1, g'=0
        init = 1., 0.
        solution = odeint(self.g, init, a_array,args=(cosmo,))
        self.growspline = ius(a_array, a_array*solution[:,0]/solution[-1,0], k=5)

        """
        CALCULATING SIGMA(M,Z)
        --------------------i
        """

        #calculate P_cdm spectrum from transfer functions: P_cdm = P_m * aux**2
        aux = ((cosmo.Omega_m() - cosmo.Omega_b()) * self.T_cdm + cosmo.Omega_b() * self.T_b)/(cosmo.Omega_m() * self.T_m)
        for index, k in enumerate(self.k_array): self.p_cdm[index] = aux[index]**2 * cosmo.pk(k*cosmo.h(),0)

        # dk integrations to get mass variance for k/real space filters
        for index, aux_R in enumerate(self.R_array):
            x = self.k_array * aux_R #!
            window = 3. * (np.sin(x) - x * np.cos(x))/x**3 #!
            sigma_trpz = integrate.simps(self.p_cdm*self.k_array**2*cosmo.h()**3*window**2,self.k_array) # h**3 from k^2 dk factor
            self.sigma_arrayz0[index] = np.sqrt(sigma_trpz / 2. / np.pi**2) #!
        # sanity check
        x = self.k_array * 8.
        window = 3. * (np.sin(x) - x * np.cos(x))/x**3
        s2=integrate.simps(self.p_cdm * cosmo.h()**3 *self.k_array**2 *window**2,self.k_array)
        print 'internal sigma8:',np.sqrt(s2/2/np.pi**2)
        print 'Class sigma8:',cosmo.sigma8()

        # Decide wether to use growth function (wrong in nuCDM cosmologies) or compute sigma(M,z) explicitly
        self.use_growthfunc = True

        for index_z, aux_z in enumerate(self.z_array):
            #read transfer functions and wavenumbers, exclude edges to evade interpolation errors
            self.T_cdm = cosmo.get_transfer(aux_z)['d_cdm'][1:-1] #!
            self.T_m = cosmo.get_transfer(aux_z)['d_tot'][1:-1] #!
            self.T_b = cosmo.get_transfer(aux_z)['d_b'][1:-1] #!
            self.p_cdm = np.zeros_like(self.k_array)

            if self.use_growthfunc == True:
                self.sigma_M_z_array[:,index_z] = self.sigma_arrayz0[:] * self.growspline(1./(1.+aux_z))
            else:
                #calculate P_cdm spectrum from transfer functions: P_cdm = P_m * aux**2
                aux = ((cosmo.Omega_m() - cosmo.Omega_b()) * self.T_cdm + cosmo.Omega_b() * self.T_b)/(cosmo.Omega_m() * self.T_m)
                for index, k in enumerate(self.k_array): self.p_cdm[index] = aux[index]**2 * cosmo.pk(k*cosmo.h(),aux_z)

                # dk integrations to get mass variance for k/real space filters
                for index_R, aux_R in enumerate(self.R_array):
                    x = self.k_array * aux_R #!
                    window = 3. * (np.sin(x) - x * np.cos(x))/x**3 #!
                    sigma_trpz = integrate.simps(self.p_cdm*self.k_array**2*cosmo.h()**3*window**2,self.k_array) # h**3 from k^2 dk factor
                    self.sigma_M_z_array[index_R,index_z] = np.sqrt(sigma_trpz / 2. / np.pi**2) #!

        time_end = timeit.default_timer()

        print 'sigma-M-z calculation: ', (time_end - time_start)

        """
        CTYPES SETUP
        --------------------
        """
        # MacOS: compile clibLx.c as dylib
        # Linux: compile clibLx.c as DLL
        if _platform == "linux" or _platform == "linux2":
            OSstring = 'so'
        elif _platform == "darwin":
            OSstring = 'dylib'
        self.clibLx = ct.CDLL(path+'/clibLx.'+OSstring)

        # ctypes interface: setup functions to hand over parameters
        self.clibLx.setup_cosmo.restype = None
        self.clibLx.setup_cosmo.argtypes = (ct.c_double,ct.c_double,ct.c_double,ct.c_double,ct.c_double,ct.c_double)

        # setup current nuisance parameters
        #self.clibLx.setup_nuisance.restype = None
        #self.clibLx.setup_nuisance.argtypes = (ct.c_double,ct.c_double,ct.c_double)


        # setup Tinker splines and halo multiplicity
        self.clibLx.spline_init.restype = None
        self.clibLx.spline_init.argtypes = (ct.c_char_p,)

        #self.clibLx.setup_data.restype = None
        #self.clibLx.setup_data.argtypes = [ndpointer(ct.c_double, flags="C_CONTIGUOUS")]

        self.clibLx.setup_sigma2d_spline.restype = None
        self.clibLx.setup_sigma2d_spline.argtypes = [ndpointer(ct.c_double, flags="C_CONTIGUOUS"),ndpointer(ct.c_double, flags="C_CONTIGUOUS"),ndpointer(ct.c_double, flags="C_CONTIGUOUS")]

        self.clibLx.loglkl.restype = ct.c_double
        self.clibLx.loglkl.argtypes = (None)

        #self.clibLx.setup_interp2d.restype = None
        #self.clibLx.setup_interp2d.argtypes = (ct.c_char_p,)

        self.clibLx.setup_areaflux.restype = None
        self.clibLx.setup_areaflux.argtypes = (ct.c_char_p,)

        self.clibLx.setup_kcorr.restype = None
        self.clibLx.setup_kcorr.argtypes = (ct.c_char_p,)

        self.clibLx.setup_detecProb.restype = None
        self.clibLx.setup_detecProb.argtypes = (ct.c_char_p,)

        """
        CALLING C SETUP FUNCTIONS
        --------------------
        """
        self.clibLx.setup_cosmo(cosmo.h(), cosmo.Omega_m(), cosmo.Omega_k(), cosmo.Omega_nu, cosmo.w0_fld(), cosmo.wa_fld())
        #self.clibLx.setup_nuisance(self.sigLx, self.siglam, self.norm_)
        self.clibLx.spline_init(os.getcwd()) # setup splines for Tinker hmf and noisemap
        self.clibLx.setup_sigma2d_spline(self.M_array, self.z_array, self.sigma_M_z_array.reshape(self.zsteps * self.Msteps))
        #self.clibLx.setup_data(self.data_counts.reshape(self.n_etaObbins * self.n_zbins))
        self.clibLx.setup_areaflux(os.getcwd())
        self.clibLx.setup_kcorr(os.getcwd())
        self.clibLx.setup_detecProb(os.getcwd())

    def loglkl(self):

        """
        CALCULATING SIGMA(M,Z)
        --------------------
        """

        #calculate P_cdm spectrum from transfer functions: P_cdm = P_m * aux**2
        aux = ((cosmo.Omega_m() - cosmo.Omega_b()) * self.T_cdm + cosmo.Omega_b() * self.T_b)/(cosmo.Omega_m() * self.T_m)
        for index, k in enumerate(self.k_array): self.p_cdm[index] = aux[index]**2 * cosmo.pk(k*cosmo.h(),0)

        # dk integrations to get mass variance for k/real space filters
        for index, aux_R in enumerate(self.R_array):
            x = self.k_array * aux_R #!
            window = 3. * (np.sin(x) - x * np.cos(x))/x**3 #!
            sigma_trpz = integrate.simps(self.p_cdm*self.k_array**2*cosmo.h()**3*window**2,self.k_array) # h**3 from k^2 dk factor
            self.sigma_arrayz0[index] = np.sqrt(sigma_trpz / 2. / np.pi**2) #!
        # sanity check
        x = self.k_array * 8.
        window = 3. * (np.sin(x) - x * np.cos(x))/x**3
        s2=integrate.simps(self.p_cdm * cosmo.h()**3 *self.k_array**2 *window**2,self.k_array)
        print 'internal sigma8:',np.sqrt(s2/2/np.pi**2)
        print 'Class sigma8:',cosmo.sigma8()

        # Decide wether to use growth function (wrong in nuCDM cosmologies) or compute sigma(M,z) explicitly
        self.use_growthfunc = True

        for index_z, aux_z in enumerate(self.z_array):
            #read transfer functions and wavenumbers, exclude edges to evade interpolation errors
            self.T_cdm = cosmo.get_transfer(aux_z)['d_cdm'][1:-1] #!
            self.T_m = cosmo.get_transfer(aux_z)['d_tot'][1:-1] #!
            self.T_b = cosmo.get_transfer(aux_z)['d_b'][1:-1] #!
            self.p_cdm = np.zeros_like(self.k_array)

            if self.use_growthfunc == True:
                self.sigma_M_z_array[:,index_z] = self.sigma_arrayz0[:] * self.growspline(1./(1.+aux_z))
            else:
                #calculate P_cdm spectrum from transfer functions: P_cdm = P_m * aux**2
                aux = ((cosmo.Omega_m() - cosmo.Omega_b()) * self.T_cdm + cosmo.Omega_b() * self.T_b)/(cosmo.Omega_m() * self.T_m)
                for index, k in enumerate(self.k_array): self.p_cdm[index] = aux[index]**2 * cosmo.pk(k*cosmo.h(),aux_z)

                # dk integrations to get mass variance for k/real space filters
                for index_R, aux_R in enumerate(self.R_array):
                    x = self.k_array * aux_R #!
                    window = 3. * (np.sin(x) - x * np.cos(x))/x**3 #!
                    sigma_trpz = integrate.simps(self.p_cdm*self.k_array**2*cosmo.h()**3*window**2,self.k_array) # h**3 from k^2 dk factor
                    self.sigma_M_z_array[index_R,index_z] = np.sqrt(sigma_trpz / 2. / np.pi**2) #!

        # hand over current sigma(M,z) array, cosmology and nuisance parameters (has to be repeated in every step)
        self.clibLx.setup_sigma2d_spline(self.M_array, self.z_array, self.sigma_M_z_array.reshape(self.zsteps * self.Msteps))
        self.clibLx.setup_cosmo(cosmo.h(), cosmo.Omega_m(), cosmo.Omega_k(), cosmo.Omega_nu, cosmo.w0_fld(), cosmo.wa_fld())
        #self.clibLx.setup_nuisance(self.sigLx, self.siglam, self.norm_)
        loglkl = 0.

        """
        Poissonian Likelihood:
        ln L  =  sum_i,j [ N_ij^obs ln N_ij^theo - N_ij^theo - ln(N_ij^obs!) ]
        for bins in redshift, S/N (defined in Planck XXIV)
        """
        time_start = timeit.default_timer()
        time2 = timeit.default_timer()

        self.theory_counts = np.zeros((self.n_lambins,self.n_zbins))

        print '---- C number counts ----'
        time_start = timeit.default_timer()

        loglkl = self.clibLx.loglkl()

        print 'loglkl [c]: ', loglkl

        time_end = timeit.default_timer()

        print 'total time: ', (time_end - time_start)

        return loglkl

    def Omega_m(self,z,cosmo):
        result = ( cosmo.Omega_m()*(1.+z)**3 * (cosmo.Hubble(0)/cosmo.Hubble(z))**2 )
        return result

    # linear dark energy equation of state
    def w_de(self,a):
        result = self.w0_fld + (1. - a) * self.wa_fld
        return result

    # defined in Linder+Jenkins MNRAS 346, 573-583 (2003)
    # solved integral by assuming linear w_de scaling analytically
    def x_plus(self,a,cosmo):
        aux = 3.0 * self.wa_fld * (1. - a)
        result = cosmo.Omega_m() / (1. - cosmo.Omega_m()) * a**(3. * (self.w0_fld + self.wa_fld)) * np.exp(aux)
        return result

    # initial conditions g(a_initial) = 1, g'(a_inital) = 0
    # choose a_initial ~ 1./10. where Omega_m ~ 1
    def g(self,y,a,cosmo):
        "rescaled growth function D/a"
        y0 = y[0]
        y1 = y[1]
        y2 = -(7./2. - 3./2. * self.w_de(a)/(1+self.x_plus(a,cosmo))) * y1 / a - 3./2. * (1-self.w_de(a))/(1.+self.x_plus(a,cosmo)) * y0 / a**2
        return y1, y2




