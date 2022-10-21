import numpy as np
import astropy.units as u
import astropy.cosmology as cosmo
from astropy.cosmology import Planck15


class _BaseRedshiftEvolution(object):
    """
    This is the base class for the redshift evolution model.
    It has the interpolated cache function for the
    evaluation of differential three (-comving) volume.
    Parameters
    ----------
    zmax: float
        Maximum redshift considered in the study (for normalization)
        DEFAULT: 2
    num_bins: integer
        Number of bins for a cached redshift array (upto zmax)
    """
    def __init__(self, zmax=2., num_zbins=1000):
        self.zmax = zmax
        self.num_zbins = num_zbins
        self._zi = np.linspace(1e-3,self.zmax,self.num_zbins)
        self._dv3_dzi = Planck15.differential_comoving_volume(self._zi).value*4*np.pi/1e9
        self.cached_dvc_dz = None
        self.cached_zs = None
        
    def _cache_dvc_dz(self, zz):
        """
        Initiate the cached function for differential three volume
        along with the redshift array for faster calculations.
        Input:
          zz: The redshift array
        """
        self.cached_dvc_dz =np.asarray(np.interp(zz, self._zi, self._dv3_dzi,
                                       left=0, right=0))
        self.cached_zs = zz
        
    def dV3cdz(self,redshift):
        """
        Differential comoving three volume (depends on the cosmology)
        Parameters
        ----------
          redshift: Redshift
        """
        return 4*np.pi*Planck15.differential_comoving_volume(redshift).value/1e9
    
    def dV4cdz(self, redshift, **parameters):
        """
        Function to calculate the merger rate density convolved with
        differential comoving four volume.
        Parameters
        ----------
        redshift: 1-D numpy array
            The array of redshift values
        parameters:
        """
        psi_zi = self.psi_z(redshift, **parameters)
        differential_volume = psi_zi / (1 + redshift)
        try:
            if (len(psi_zi) != len(self.cached_dvc_dz)) or (self.cached_zs != redshift) :
                raise TypeError
            differential_volume *= self.cached_dvc_dz
        except (TypeError, ValueError):
            self._cache_dvc_dz(redshift)
            differential_volume *= self.cached_dvc_dz
        return differential_volume
    
    def comoving_3volume(self, redshift):
        return Planck15.comoving_volume(redshift).value/1e9
    
    def psi_z(self, redshift, **parameters):
        return (1+redshift)**parameters['k']
    
    def normalize(self, parameters):
        """Normalization function for the probability function of redshift model
        Parameters:
        ----------------
        parameters: Dictionary
            A dictionary containing population parameters
        """
        psi_zi = self.psi_z(self._zi, **parameters)
        return np.trapz(psi_zi*self._dv3_dzi/(1+self._zi), self._zi)
        
    def total_four_volume(self, analysis_time, parameters):
        psi_zi = self.psi_z(self._zi, **parameters)
        return np.trapz(self._dv3_dzi*psi_zi/(1+self._zi), self._zi)*analysis_time
    
    def prob_redshift(self, redshift, parameters):
        return_arr = np.zeros(len(redshift))
        indx1 = np.where(redshift <= self.zmax)
        indx2 = np.where(redshift > self.zmax)
        n1 = self.normalize(parameters)
        #psi_zi = self.psi_z(redshift, **parameters)
        dvc_dzi = self.dV4cdz(redshift, **parameters)
        return_arr[indx1] = dvc_dzi[indx1]/n1
        return return_arr
    
    
class PowerLawRedshift(_BaseRedshiftEvolution):
    """
    Powerlaw evolution of the redshift
    p(z) \propto (1+z)^k
    """
    def __call__(self, redshift, parameters):
        return self.prob_redshift(redshift, parameters)
        
    def psi_z(self, redshift, **parameters):
        return (1+redshift)**parameters['k']
    
power_law_redshift = PowerLawRedshift()
