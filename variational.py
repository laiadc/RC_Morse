# @hidden
import numpy as np
from scipy.special import factorial
from scipy import linalg as LA
import scipy.sparse as sps
from scipy.linalg import eigh
from scipy.special import eval_hermite
from scipy.signal import argrelextrema
from pylab import *
from tqdm import tqdm

class eigen_state_potentialCoupledMorse2D:

  def __init__(self, De1, a1, xe1,De2, a2, xe2,xmin, xmax, n_points, G11 = 0.0005785629911337726, G12 = -8.584564281295045e-6, 
  N=10, omegax =1/np.sqrt(2), omegay=1/np.sqrt(2)):
    '''
    Class to generate data (V(x) and phi(x) ground state) for potentials of the form
    V(x) = sum_i alpha_i x^i, using the H.O basis
    Args:
      alpha_min: vector of length N, with the minimum value of the coefficients alpha
      alpha_max: vector of length N, with the maximum value of the coefficients alpha
      the values of alpha will be randomly distributed in [alpha_min, alpha_max]
    '''
    self.De1 = De1
    self.a1 = a1
    self.xe1=xe1
    self.De2 = De2
    self.a2 = a2
    self.xe2=xe2
    self.G11 = G11
    self.G12 =G12
    self.xmin = xmin
    self.xmax = xmax
    self.ymin = xmin
    self.ymax = xmax
    self.n_points = n_points
    self.N = N # Length of H.O basis
    self.omegax = omegax
    self.omegay = omegay
    nxs, nys = self.order_energy(N, omegax, omegay)
    self.nxs = nxs
    self.nys = nys
    self.memo = {}
    self.memoP = {}
    self.memoC= {}
    
  def order_energy(self,n_state, omegax, omegay):
      """
      Given omegax, omegay, we find the n_state-th excited state
      Args:
        n_state (int): Number of excited state
        omegax (float): omega in the x direction
        omegay (float): omega in the y direction
      Returns:
        (int) nx of the n_state excited state
        (int) ny of the n_state excited state
      """

      # With these omegas we create a list of the pairs (nx,ny) ordered by energy E = hbar*(omega_x*(nx+1/2) + omega_y*(ny+1/2))
      # Up to n_state (so nx,ny<= n_state), and then we order them by energy
      nxs = np.zeros((n_state+1)**2)
      nys = np.zeros((n_state+1)**2)
      Es = np.zeros((n_state+1)**2)
      i=0
      for nx in range(n_state+1):
        for ny in range(n_state+1): 
            nxs[i] = int(nx)
            nys[i] = int(ny)
            Es[i] = (omegax*(nx+1/2) + omegay*(ny + 1/2))
            i+=1
      
      idxs = np.argsort(Es)
      nxs = nxs[idxs]
      nys = nys[idxs]
      return nxs, nys


  def I_nmr(self,n,m,r):
    '''
    Calculates the value of the integral of the Hermitte polynomials
    Args:
      n (int): n of I(n,m,r)
      m (int): m of I(n,m,r)
      r (int): r of I(n,m,r)
    Returns:
      I(n,m,r)
    '''
    if r<0 or n<0 or m<0:
      return 0
    if r==0:
      if n==m:
        return np.sqrt(np.pi)*2**n*factorial(n)
      else:
        return 0
    return 1./2*self.I_nmr(n+1,m,r-1) + n*self.I_nmr(n-1,m,r-1)


  def int_P(self,n,m):
    if (n,m) in self.memoP:
      return self.memoP[(n,m)]
    I1 = -1/2*self.I_nmr(n,m,2)
    I2 = 1/2*self.I_nmr(n,m,0)
    I3 = 2*m*self.I_nmr(n,m-1,1)
    I4 = -2*m*(m-1)*self.I_nmr(n, m-2,0)
    I = I1 + I2 + I3 + I4
    self.memoP[(n,m)] = I
    return I

  

  def J(self, n, m, a):
    if (n, m, a) in self.memo:
        return self.memo[(n, m, a)]
    if m == 0:
        self.memo[(n, m, a)] = ((-1) ** n) * (a ** n)
    else:
        self.memo[(n, m, a)] = (-a * self.J(n, m - 1, a)) + (2 * n * self.J(n - 1, m - 1, a))
    return self.memo[(n, m, a)]

 
  def C_nm(self,n,m):
    '''
    Calculates the coefficient C_{nm} for the potential V(x,y) = \sum_i \sum_j alpha[i,j]x^iy^j
    Args:
      n (int): n of C_nm
      m (int): m of C_nm
      alphas (np.array): size kxk. Coefficients of the potential V(x,y)
    Returns:
      C_{nm}
    '''
    if (n,m) in self.memoC:
      return self.memoC[(n,m)]
    # Given n and m find nx, ny, mx, my
    nx = self.nxs[n]
    ny = self.nys[n]
    mx = self.nxs[m]
    my = self.nys[m]
    
    I1,I2,I3,I4,I5,I6, I_v_1, I_v_2 = 0,0,0,0,0,0,0,0

    #Iv
    if ny==my:
        I_v_1=self.De1*2.**(ny)*factorial(ny)*(self.J(nx,mx,2.*self.a1/np.sqrt(self.omegax))*np.exp(self.a1**2./self.omegax)-2.*self.J(nx,mx,self.a1/np.sqrt(self.omegax))*np.exp(self.a1**2./(4.*self.omegax)))*np.pi
    if nx==mx:
        I_v_2=self.De2*2.**(nx)*factorial(nx)*(self.J(ny,my,2.*self.a2/np.sqrt(self.omegay))*np.exp(self.a2**2./self.omegay)-2.*self.J(ny,my,self.a2/np.sqrt(self.omegay))*np.exp(self.a2**2./(4.*self.omegay)))*np.pi
    Iv = I_v_1+I_v_2
      
    Anx = 1./np.sqrt(np.sqrt(np.pi)*factorial(nx)*2.**nx)
    Any = 1./np.sqrt(np.sqrt(np.pi)*factorial(ny)*2.**ny)
    Amx = 1./np.sqrt(np.sqrt(np.pi)*factorial(mx)*2.**mx)
    Amy = 1./np.sqrt(np.sqrt(np.pi)*factorial(my)*2.**my)
    An=Anx*Any
    Am=Amx*Amy
    
    if ny==my:
        I1=self.omegax*2.**(ny)*factorial(ny)*self.int_P(nx,mx)*np.sqrt(np.pi)

    if nx==mx:
        I2=self.omegay*2.**(nx)*factorial(nx)*self.int_P(ny,my)*np.sqrt(np.pi)

    if ny==my-1 and nx==mx-1:
        I3=-4.*mx*my*2.**(nx+ny)*factorial(nx)*factorial(ny)*np.pi

    if nx==mx-1:
        I4=2.*mx*2.**(nx)*factorial(nx)*self.I_nmr(ny,my,1)*np.sqrt(np.pi)
    if ny==my-1:
        I5=2*my*2**(ny)*factorial(ny)*self.I_nmr(nx,mx,1)*np.sqrt(np.pi)

    I6=-self.I_nmr(nx,mx,1)*self.I_nmr(ny,my,1)
    C_nm=An*Am*(self.G11*(I1+I2)+self.G12*np.sqrt(self.omegax*self.omegay)*(I3+I4+I5+I6)+Iv)
    self.memoC[(n,m)] = C_nm
    return C_nm

  def find_eigen_state(self,n_state=0):
    '''
    Finds the eigen state of a potential V(x) = sum_i alpha_i x^i
    Args:
      alphas(np array): size kxk. Coefficients of the potential V(x)
      n_state (int): Number of excited state (default n_state=0, ground state)
    Returns:
      E_a (float): Energy of the ground state for potential V
      a (np.array): size N. Coefficients in the basis of the H.O potential
    '''
    N = self.N
    # 0. Generate matrix of C_nm
    C_s = np.zeros((N, N))
    for n in range(N):
        for m in range(n,N):
            C_s[n, m] = self.C_nm(n, m)
    C = C_s + C_s.T
    np.fill_diagonal(C, np.diagonal(C_s))
    # 1. Generate matrix D
    D = C + C.T
    # 2. Diagonalize matrix D
    vaps, veps = eigh(D)
    #print('Diagonalized')
    # 3. Calculate <H> for all a
    Hs = np.dot(veps.T, np.dot(C, veps))
    Hs = Hs.diagonal()
    # 4. We choose the vector which minimizes <H>
    # If n_state!=0, we choose the vector with n_state-th lowest energy
    # as an approximation of the n_state excited state 
    idxs = np.argsort(Hs)
    sel = idxs[n_state]
    a = veps[:, sel] # Final value of eigenvalues for state n_state
    E_a = Hs[sel] # Value of the energy
    return E_a, a, Hs[idxs], veps[:,idxs]

  def generate_data(self,n_state=0):
    '''
    Generates samples of potentials  with random coefficients and finds the n_state excited state for them
    Args:
      n_state (int): Number of excited state (default n_state=0, ground state)
    Returns:
      E (np.array): size n_samples. Ground energy for each V
      a (np.array): size n_samples x N. Coefficients in the H.O basis for each V
    '''
    # Find ground state for each sample
    E, a, Es, veps = self.find_eigen_state(n_state)
    return E, a, Es, veps

  def evaluate_potential(self, n_points = None):
    '''
    Given the coeefficients alphas, it evaluates the potential in V(x)
    Args:
      xmin(float): minimum value of x
      xmax (float): maximum value of x
      n_points (int): Number of points between xmin and xmax
      alpha (np.array): size n_samples x k x k. Matrix of coefficients of V(x) (each row a different potential)
    Returns:
      V(np.array): size n_samples x n_points x n_points. V(x) for every sample
      x(np.array): size n_points. Values of x and y
    '''
    if n_points is None:
      n_points = self.n_points
    x = np.arange(self.xmin, self.xmax, (self.xmax - self.xmin)/n_points)
    y = np.arange(self.xmin, self.xmax, (self.xmax - self.xmin)/n_points)      
    xv, yv = np.meshgrid(x,y)
    potential = self.De1*(1-np.exp(-self.a1*(xv - self.xe1)))**2 + self.De2*(1-np.exp(-self.a2*(yv - self.xe2)))**2#self.De1*(np.exp(-2*zx) - 2*np.exp(-zx)) + self.De2*(np.exp(-2*zy) - 2*np.exp(-zy))

    return potential, xv, yv, x, y
  
  def HO_wavefunction2(self,n, xmin, xmax, n_points, omega):
      '''
      Returns the nth eigenfunction of the harmonic oscillator in the points x
      Args:
        n (int): Energy level
        xmin(float): minimum value of x
        xmax (float): maximum value of x
        n_points (int): Number of points between xmin and xmax
      Returns:
        phi_n (np.array): size n_points. Phi_n(x)
      '''
      x = np.arange(xmin, xmax, (xmax - xmin)/n_points)
      sigma_inv = np.sqrt(omega)
      all_x = x*sigma_inv # It is a matrix of dim (num_x_points), 
      herm = eval_hermite(n, all_x) # H_n(x/sigma)
      #exp = np.exp(- all_x**2/2) # Exponential term
      sign1 = np.sign(herm)
      log_herm = np.log(np.abs(herm))
      log_exp = - all_x**2/2
      phi_n = sign1*(np.exp(log_herm + log_exp))
      #phi_n = exp*herm

      return phi_n

  def final_wavefunction(self, a, veps=None, n_state = 0, n0=0):
      '''
      Returns the final wavefunctions psi(x) = sum_i alpha_i phi_i(x) for each alpha.
      Args:
        xmin(float): minimum value of x
        xmax (float): maximum value of x
        n_points (int): Number of points between xmin and xmax
        a (np.array): size n_samples x N. Coefficients in the H.O basis for each V
      Returns:
        waves(np.array): size n_samples x n_points x n_points. psi(x,y) for each value of V (given by alpha)
      '''
      x = np.arange(self.xmin, self.xmax, (self.xmax - self.xmin)/self.n_points)
      y = np.arange(self.xmin, self.xmax, (self.xmax - self.xmin)/self.n_points)
      n_samples = 1
      a = a.reshape(1,-1)
      # Construct matrix of phi_n
      phis = np.zeros((self.N, self.n_points, self.n_points))
      ones = np.repeat(1,self.n_points)
      h = (self.xmax - self.xmin)/self.n_points
      #print('Get wavefunctions')
      for i in range(self.N):
        nx = int(self.nxs[i])
        ny = int(self.nys[i])
        phi_x = self.HO_wavefunction2(nx, self.xmin, self.xmax, self.n_points, omega = self.omegax)
        phi_y = self.HO_wavefunction2(ny, self.xmin, self.xmax, self.n_points, omega = self.omegay)
        phi_xi = np.tensordot(phi_x, ones, axes=0).T
        phi_yi = np.tensordot(phi_y, ones, axes=0)
        phi_i = phi_xi*phi_yi #np.outer(phi_x[i,:], phi_y[i,:].T)
        C = 1./np.sqrt(np.sum(phi_i*phi_i*h*h)) # Normalization constant
        phis[i,:,:] = C*phi_i

      if n_state==0:
        waves = np.zeros((n_samples, self.n_points, self.n_points))
        for n in range(n_samples):
          waves[n,:,:] = np.average(phis, axis=0, weights=a[n,:])*np.sum(a[n,:])

      else:
        size = veps.shape[-1]
        veps = veps.reshape(1, size, size)
        waves = np.zeros((n_state, n_samples, self.n_points, self.n_points))
        for state in range(n_state):
          a = veps[:,:,n0+ state]
          for n in range(n_samples):
            waves[state, n,:,:] = np.average(phis, axis=0, weights=a[n,:])*np.sum(a[n,:])

      return waves, x, phis