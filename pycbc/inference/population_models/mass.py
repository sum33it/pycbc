import numpy as np
from pycbc import cosmology, conversions
import scipy
import scipy.stats

from scipy.integrate import quad, nquad
from scipy.interpolate import Rbf

from redshift import power_law_redshift as plr
from pycbc.population.population_models import sfr_grb_2008, sfr_madau_dickinson_2014, sfr_madau_fragos_2017

def powerlaw_normalized(xdata,alpha,xmin,xmax):
    """
    Parameters
    -------------
    xdata: dataset
    alpha: number or array
        Powerlaw index
    xmin: number or array
        Minimum value of the model parameter
    xmax: number or array
        Maximum value of the model parameter
    Returns
    --------
    prob: 1d array of floats
        Probability values for the data 
    """
    
    if alpha == -1:
        normalization_factor = 1./np.log(xmax/xmin)
    else:
        normalization_factor = (1+alpha)/(np.power(xmax*1.,(1+alpha)) - np.power(xmin*1.,(1+alpha)))
        
    prob = np.power(xdata,alpha)*normalization_factor
    prob *= (xdata >= xmin) & (xdata <= xmax)
        
    return np.nan_to_num(prob)

class BasePopulationModel(object):
    def __init__(self,m_min,m_max,k,R0,smax=0.99,zmax=2.0):
        self.alpha = alpha
        self.beta_q = beta_q
        self.lambda_peak = lambda_peak
        self.delta_m = delta_m
        self.mu_m = mu_m
        self.sigma_m = sigma_m
        self.m_min = m_min
        self.m_max = m_max
        self.k = k
        self.R0 = R0
        self.smax = smax
        self.zmax = zmax
        self.cached_redshift_model = None
        
        
    def _cache_redshift_model(self, zz):
        from redshift import PowerLawRedshift
        self.cached_redshift_model = PowerLawRedshift(zmax=zz)
        
    def f1(self,x):
        return np.exp(self.delta_m/x + self.delta_m/(x - self.delta_m))
    
    def p_redshift(self,z):
        try:
            return_arr = self.cached_redshift_model.prob_redshift(z,{'k':self.k})
        except (TypeError, ValueError, AttributeError):
            self._cache_redshift_model(self.zmax)
            return_arr = self.cached_redshift_model.prob_redshift(z,{'k':self.k})
        return return_arr
        
    def total_four_volume(self,analysis_time):
        try:
            tot_vol = self.cached_redshift_model.total_4volume(analysis_time,{'k':self.k})
        except (TypeError, ValueError, AttributeError):
            self._cache_redshift_model(self.zmax)
            tot_vol = self.cached_redshift_model.total_4volume(analysis_time,{'k':self.k})
        return tot_vol
    
    def smoothing_function(self,m):
        m_diff = m*1.0 - self.m_min
        sf = np.ones_like(m)
        smoothing_indx = (self.m_min <= m) & (m < self.m_min+self.delta_m)
        sf[smoothing_indx] = 1./(self.f1(m_diff[smoothing_indx])+1)
        sf *= (m >= self.m_min) 
        return sf
        
    def log_p_spin(self, sx, sy, sz):
        ''' Evaluate p(sx, sy, sz) = (1. / |s|^2) p(|s|, cos theta, phi) = 1. / (4 pi s_max |s|^2)
        where:
        - |s| = sqrt(sx^2 + sy^2 + sz^2)
    
        '''    
        s2 = sx**2 + sy**2 + sz**2 
        return - np.log(4 * np.pi) - np.log(self.smax) - np.log(s2)
    
    def log_p_isotropic_spin(self, sx, sy, sz):
        return_arr = np.zeros(len(sx))
        indx1 = np.where(sx**2+sy**2+sz**2 < self.smax)
        indx2 = np.where(sx**2+sy**2+sz**2 >= self.smax)
        return_arr[indx1] = self.log_p_spin(sx[indx1], sy[indx1], sz[indx1])
        return_arr[indx2] = -np.inf
        return return_arr

class PowerlawModel(BasePopulationModel):
    def __init__(self, alpha, beta_q, delta_m, m_min, m_max, k, R0, smax=0.99, zmax=2.0):
        self.alpha = alpha
        self.beta_q = beta_q
        self.delta_m = delta_m
        self.m_min = m_min
        self.m_max = m_max
        self.k = k
        self.R0 = R0
        self.smax = smax
        self.zmax = zmax
        self.cached_redshift_model = None
        self.cached_m1_indx_within_smoothin_function = None
        self.cached_m2_indx_within_smoothin_function = None
        self._m1i = np.linspace(2,150,1000)
        self._qi = np.linspace(0.001,1.001,500)
        self._qiv,self._m1v = np.meshgrid(self._qi, self._m1i)
        
    def p_m1(self, m1, normalize_for_smoothing=True):
        sf = self.smoothing_function(m1)
        prob_m1 = powerlaw_normalized(m1,-self.alpha, self.m_min, self.m_max)*sf
        if normalize_for_smoothing:
            prob_m1 /= self.new_norm_m1()
        return prob_m1
    
    def p_q(self, q, m1, normalize_for_smoothing=True):
        """
        We use different definition of q than in paper
        for us q=m1/m2, in the paper it is inverse
        """
        sf = self.smoothing_function(q*m1)
        prob_q = powerlaw_normalized(q,self.beta_q,self.m_min/m1,1)*sf
        prob_q *= (m1 >= self.m_min) & (m1 <= self.m_max) # Removing nans
        prob_q *= ((q <= 1) & (q>0))
        if normalize_for_smoothing:
            prob_q /= self.new_norm_q(m1)
        return np.nan_to_num(prob_q)
    
    def new_norm_m1(self):
        if self.delta_m == 0:
            return 1
        p_m1i = self.p_m1(self._m1i,normalize_for_smoothing=False)
        return np.trapz(p_m1i,self._m1i)
    
    def new_norm_q(self,m1):
        if self.delta_m == 0:
            return np.ones_like(len(m1))
        else:
            p_qv = self.p_q(self._qiv, self._m1v, normalize_for_smoothing=False)
            return np.interp(m1, self._m1i, np.trapz(p_qv,self._qiv))
    
    def log_p_m1q(self,m1,q):
        prob = np.log(self.p_m1(m1))+ np.log(self.p_q(q,m1))
        return prob
    
    def log_p_m1m2(self,m1,m2):
        prob = self.log_p_m1q(m1,m2/m1) - np.log(m1)
        return prob

    def p_m1m2(self,m1,m2):
        return np.exp(self.log_p_m1m2(m1,m2)) 
    
    def p_m1q(self,m1,q):
        return np.exp(self.log_p_m1q(m1,q))
    
    def log_p_m1m2_z(self,m1,m2,z):
        prob_m1m2z = self.log_p_m1m2(m1, m2)+np.log(self.p_redshift(z))
        return prob_m1m2z
    
    def log_p_m1m2_s1s2_z(self,m1,m2,s1x, s1y, s1z, s2x, s2y, s2z, z):
        return self.log_p_m1m2_z(m1,m2,z)+self.log_p_isotropic_spin(s1x,s1y,s1z)+self.log_p_isotropic_spin(s2x,s2y,s2z)
    
class PowerlawPlusPeak(PowerlawModel):
    def __init__(self,alpha,beta_q,lambda_peak,delta_m,mu_m,sigma_m,
                 m_min,m_max,k,R0,smax=0.99,zmax=2.0):
        self.alpha = alpha
        self.beta_q = beta_q
        self.lambda_peak = lambda_peak
        self.delta_m = delta_m
        self.mu_m = mu_m
        self.sigma_m = sigma_m
        self.m_min = m_min
        self.m_max = m_max
        self.k = k
        self.R0 = R0
        self.smax = smax
        self.zmax = zmax
        self.cached_redshift_model = None
        self._m1i = np.linspace(2,150,1000)
        self._qi = np.linspace(0.001,1.001,500)
        self._qiv,self._m1v = np.meshgrid(self._qi, self._m1i)

    def p_m1(self,m1,normalize_for_smoothing=True):
        sf = self.smoothing_function(m1)
        powerlaw_part = (1-self.lambda_peak)*powerlaw_normalized(m1,-self.alpha, self.m_min, self.m_max)
        gaussian_part = self.lambda_peak*scipy.stats.norm(self.mu_m, self.sigma_m).pdf(m1)
        prob_m1 = (powerlaw_part+gaussian_part)*sf
        if normalize_for_smoothing:
            prob_m1 /= self.new_norm_m1()
        return prob_m1
        
        
class PowerlawPlusTwoPeak(PowerlawPlusPeak):
    def __init__(self,alpha,beta_q,lambda1_peak,lambda2_peak,delta_m,
                 mu1_m,sigma1_m,mu2_m,sigma2_m,m_min, m_max, k, R0,
                 smax=0.99,zmax=2.0):
        self.alpha = alpha
        self.beta_q = beta_q
        self.lambda1_peak = lambda1_peak
        self.lambda2_peak = lambda2_peak
        self.delta_m = delta_m
        self.mu1_m = mu1_m
        self.sigma1_m = sigma1_m
        self.mu2_m = mu2_m
        self.sigma2_m = sigma2_m
        self.m_min = m_min
        self.m_max = m_max
        self.k = k
        self.R0 = R0
        self.smax = smax
        self.zmax = zmax
        self.cached_redshift_model = None
        self._m1i = np.linspace(2,150,1000)
        self._qi = np.linspace(0.001,1.001,500)
        self._qiv,self._m1v = np.meshgrid(self._qi, self._m1i)

    
    def p_m1(self, m1, normalize_for_smoothing=True):
        sf = self.smoothing_function(m1)
        powerlaw_part = (1-self.lambda1_peak-self.lambda2_peak)*powerlaw_normalized(m1,-self.alpha, self.m_min, self.m_max)
        gaussian_part1 = self.lambda1_peak*scipy.stats.norm(self.mu1_m, self.sigma1_m).pdf(m1)
        gaussian_part2 = self.lambda2_peak*scipy.stats.norm(self.mu2_m, self.sigma2_m).pdf(m1)
        prob_m1 = (powerlaw_part+gaussian_part1+gaussian_part2)*sf
        if normalize_for_smoothing:
            prob_m1 /= self.new_norm_m1()
        return prob_m1
    
    
    
