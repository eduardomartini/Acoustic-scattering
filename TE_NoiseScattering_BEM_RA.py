# -*- coding: utf-8 -*-

"""
################################################################################
# Numerical solution of acoustic scattering by finite perforated elastic plates
# A. V. G. Cavalieri, W. R. Wolf and J. W. Jaworski - PRSA 2016
# -----------------------------------------------------------------------------
# Develped by Msc. Cristiano Pimenta and Prof. Dr. William Wolf
# Input parameters:
# - Modal basis -> Non-dimensional frequency and vibration modes(displacements) 
# - Acoustic wavenumber
# - alphaH -> Open area fraction
# - Omega -> Vacuum bending wave Mach number
# - epsilon -> Intrinsic fluid-loading parameter
# - field -> Set True to compute the acoustic field
################################################################################
"""

# Import libraries
import sys, os
import numpy as np
from scipy.special import *
from scipy import integrate, optimize
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import glob
import argparse
import modmatrix as matrixv1


# Define the source and target points
def sources():
    x = [1.0]
    y = [.004]


    # for xc in np.arange(-3, 3, 0.1):
    #     for yc in range(-3, 3, 100):
    #         x.append(xc)
    #         y.append(yc)
    return x, y

def targets():
    x = []
    y = []

    xcenter = 1
    theta = np.linspace(0, 2.0 * np.pi, 1)
    r0 = 50.0
    x = xcenter + r0 * np.cos(theta)
    y = r0 * np.sin(theta)
    # x = x[0:-1]
    # y = y[0:-1]
    return x, y, theta

# Greens functions from a source to a target 

def R(x1, y1, x2, y2):
    """ This function computes the distance between two points"""
    dx = x2 - x1
    dy = y2 - y1
    return np.sqrt(dx**2 + dy**2)
def dRdxi(x1, y1, x2, y2):
    """ This function computes the derivative of the distance between two points"""
    dx = x2 - x1
    dy = y2 - y1
    r = np.sqrt(dx**2 + dy**2)
    return dx / r, dy / r
def d2Rdxij(x1, y1, x2, y2):
    """ This function computes the second derivative of the distance between two points"""
    dx = x2 - x1
    dy = y2 - y1
    r = np.sqrt(dx**2 + dy**2)
    return (1./r - dx*dx/r**3, -dx*dy/r**3, 1./r - dy*dy/r**3)


def monopole(k0, xt, yt, xc, yc, s):
    """ This function computs the model source monopole from a list of sources, 
    whith coordinates at (xc, yc) and (complex) intensity s, to a list of targets at (xt, yt)"""
    xt = np.asarray(xt).reshape(1,-1)
    yt = np.asarray(yt).reshape(1,-1)
    xc = np.asarray(xc).reshape(-1,1)
    yc = np.asarray(yc).reshape(-1,1)
    s = np.asarray(s)
    assert xt.shape[0] == 1
    assert yc.shape[1] == 1

    r = R(xc, yc, xt, yt)
    p   = s*( 1.0j / 4.0) * hankel1(0, k0 * r)  # Hankel of 1 order 0
    p   = np.sum(p, axis=0) 
    return p

def dipole(k0, xt, yt, xc, yc, s):
    """ This function computs the model source monopole from a list of sources, 
    whith coordinates at (xc, yc) and (complex) intensity s, to a list of targets at (xt, yt)"""
    xt = np.asarray(xt).reshape(1,-1)
    yt = np.asarray(yt).reshape(1,-1)
    xc = np.asarray(xc).reshape(-1,1)
    yc = np.asarray(yc).reshape(-1,1)
    s = np.asarray(s)
    assert xt.shape[0] == 1
    assert yc.shape[1] == 1

    r = R(xc, yc, xt, yt)
    drdx,drdy = dRdxi(xc, yc, xt, yt)
    px   = s*( 1.0j / 4.0) *( -hankel1(1,k0*r)  *drdx)
    py   = s*( 1.0j / 4.0) *( -hankel1(1,k0*r)  *drdy)
    px   = np.sum(px, axis=0) 
    py   = np.sum(py, axis=0) 
    return px,py

def quadrupole(k0, xt, yt, xc, yc, s):
    """ This function computs the model source quadrupole from a list of sources, 
    whith coordinates at (xc, yc) and (complex) intensity s, to a list of targets at (xt, yt)"""
    xt = np.asarray(xt).reshape(1,-1)
    yt = np.asarray(yt).reshape(1,-1)
    xc = np.asarray(xc).reshape(-1,1)
    yc = np.asarray(yc).reshape(-1,1)
    s = np.asarray(s)
    assert xt.shape[0] == 1
    assert yc.shape[1] == 1

    r = R(xc, yc, xt, yt)
    drdx,drdy = dRdxi(xc, yc, xt, yt)
    d2rdxx, d2rdxy, d2rdyy = d2Rdxij(xc, yc, xt, yt)
    dHdr   = -k0 * (hankel1(1,k0*r)) 
    d2Hdr2 = -(k0**2) * 0.5 * (hankel1(0,k0*r) - hankel1(2,k0*r))
    pxx = s*( 1.0j / 4.0) * ( d2Hdr2 * drdx**2   + dHdr * d2rdxx ) 
    pyx = s*( 1.0j / 4.0) * ( d2Hdr2 * drdy*drdx + dHdr * d2rdxy ) 
    pyy = s*( 1.0j / 4.0) * ( d2Hdr2 * drdy**2   + dHdr * d2rdyy ) 
    pxx = np.sum(pxx, axis=0)
    pyx = np.sum(pyx, axis=0)
    pyy = np.sum(pyy, axis=0)

    return pxx, pyx , pyy
 

def quadrupole_old(k0, xt, yt, xc, yc, s):
    """ This function computs the model source quadrupole from a list of sources, 
    whith coordinates at (xc, yc) and (complex) intensity s, to a list of targets at (xt, yt)"""
    xt = np.asarray(xt).reshape(1,-1)
    yt = np.asarray(yt).reshape(1,-1)
    xc = np.asarray(xc).reshape(-1,1)
    yc = np.asarray(yc).reshape(-1,1)
    s = np.asarray(s)
    assert xt.shape[0] == 1
    assert yc.shape[1] == 1

    dx = xt - xc
    dy = yt - yc
    
    arg = k0 * np.sqrt(dx**2 + dy**2)
    
    Ha0 = hankel1(0,arg)  # Hankel of 1 order 0
    Ha1 = hankel1(1,arg)
    Ha2 = hankel1(2,arg)
    
    arg1_y2 = - 0.5 * (Ha0 - Ha2) * ( k0**2 * dx * dy / ( dx**2 + dy**2))
    arg2_y2 =  Ha1 * k0 * dx * dy / (( dx**2 + dy**2)**(3.0/2.0))
    
    p =  s*(1j / 4.0 ) * (arg1_y2 + arg2_y2)
    p = np.sum(p, axis=0)
    return p

class gpanel:
    """This class contains all geometry information about to a panel"""
    def __init__(self,x1,y1,x2,y2):
        ''' Initializes the panel.
        Arguments
        x1, y1 --> Coordinates of the first point of the panel
        x2, y2 --> Coordinates of the first point of the panel
        '''
        self.x1, self.y1 = x1, y1
        self.x2, self.y2 = x2, y2
        self.xc, self.yc = (x1 + x2)/2, (y1 + y2)/2   # Center point of the panel
        self.length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 )   # length of the panel
        # normal -  Important to check
        self.n1 =  (y2 - y1)/self.length
        self.n2 = -(x2 - x1)/self.length


class Profil:
    """This class contains profile information."""
    def __init__(self, k0, alphaH, epsilon, omega, xs,ys,xt,yt,Npsx=400, thickness=0.002, c=1.0, eps=1e-1):
        # Geometry information
        # Npsx = 400  # number of panel created on each surface
        xs = np.asarray(xs)
        ys = np.asarray(ys)
        xt = np.asarray(xt)
        yt = np.asarray(yt)

        Nps = 2 * Npsx + 2  # Total of panels

        # c = 1.0     # Chord of the plate
        self.xs = xs
        self.ys = ys
        self.xt = xt
        self.yt = yt
        self.k0 = k0
        self.eps = eps
        self.alphaH = alphaH
        self.epsilon = epsilon
        self.omega = omega
        self.Npsx = Npsx
        self.dx = c / Npsx   

        self.xp, self.yp = np.zeros(Nps+1), np.zeros(Nps+1)

        # Coordenates points
        for i in range(Nps):
            if i <= Npsx:
                self.xp[i] = self.dx * i
                self.yp[i] = +thickness/2.0
            else:
                self.xp[i] = self.xp[Nps - i - 1]
                self.yp[i] = -thickness/2.0

        self.xp[Nps] = self.xp[0]
        self.yp[Nps] = self.yp[0]

        #defining the panels
        panels = np.empty(Nps,dtype=object)
        for i in range(Nps):
            panels[i] = gpanel(self.xp[i], self.yp[i], self.xp[i+1], self.yp[i+1])

        self.x1 = np.asarray([pi.x1 for pi in panels])
        self.y1 = np.asarray([pi.y1 for pi in panels])
        self.x2 = np.asarray([pi.x2 for pi in panels])
        self.y2 = np.asarray([pi.y2 for pi in panels])
        self.xc = np.asarray([pi.xc for pi in panels])
        self.yc = np.asarray([pi.yc for pi in panels])
        self.n1 = np.asarray([pi.n1 for pi in panels])
        self.n2 = np.asarray([pi.n2 for pi in panels])
        self.ds = np.asarray([pi.length for pi in panels])

        print("calling hgmatrix...", Nps)
        self.H , self.G  = matrixv1.mntmat.hgmatrix(k0, self.x1,self.y1,
                                                        self.x2,self.y2,
                                                        self.xc,self.yc,
                                                        self.n1,self.n2,
                                                        self.ds,Nps)
        print('H[1,1] = ', self.H[1,1], 'type = ', type(self.H[1,1]))
        print('G[1,1] = ', self.G[1,1], 'type = ', type(self.G[1,1]))
        print("Done hgmatrix...")
        """
        ==================================================================================
        ----------------- Import Structural Modal Basis information ----------------------
                                Implement poroelastic materials
        ==================================================================================
        """

        beta = np.loadtxt('modalBasis/beta.txt')   # Read the Non-dimensional frequency
        phi  = np.loadtxt('modalBasis/modes.txt')   # Read the modal basis 
        nm   = len(beta)   # define number of modes

        nplate = Nps

        # Building the matrix D
        print("calling poroelastic...")

        self.D = matrixv1.mntmat.poroelastic(nplate,k0, alphaH, epsilon, omega, self.n2, self.ds, beta, phi,Nps,nm)
        print('D[1,1] = ', self.D[1,1], 'type = ', type(self.D[1,1]))
        print("Done poroelastic.")

        # Compute LHS
        self.A = self.H - np.dot(self.G, self.D)
        # Far field matrix H, G for the target points
        print("calling hgobs...")
        self.Ht, self.Gt = matrixv1.mntmat.hgobs    (self.k0,   self.xt ,self.yt     ,
                                                                self.x1 ,self.y1     ,
                                                                self.x2 ,self.y2     ,
                                                                self.n1 ,self.n2     ,
                                                                self.ds ,len(self.xt),
                                                                Nps)
        
        self.Hs = np.empty((3, 3), dtype=object)                                                       
        self.Gs = np.empty((3, 3), dtype=object)                                                       
        for idx,dx in enumerate([-eps, 0 , eps]):
            for idy,dy in enumerate([-eps, 0 , eps]):
                self.Hs[idx,idy], self.Gs[idx,idy] = matrixv1.mntmat.hgobs(self.k0,
                                                                self.xs+dx  , self.ys+dy    ,
                                                                self.x1     , self.y1       ,
                                                                self.x2     , self.y2       ,
                                                                self.n1     , self.n2       ,
                                                                self.ds     , len(self.xs)  ,
                                                                Nps)
        print("Done hgobs.")

    def target_from_sources(self,s,sourceType='quadrupole',dx=0,dy=0):
        """ This function computes the model source quadrupole from a list of sources,
        whith coordinates at (xc, yc) and (complex) intensity s, to a list of targets at (xt, yt)"""
        # Compute the right-hand side
        if sourceType == 'monopole':
            S  = monopole(self.k0, self.xc, self.yc, self.xs+dx, self.ys-dy, s)
            Sf = monopole(self.k0, self.xt, self.yt, self.xs+dx, self.ys-dy, s)
        elif sourceType == 'quadrupole':
            S  = quadrupole(self.k0, self.xc, self.yc, self.xs+dx, self.ys-dy, s)
            Sf = quadrupole(self.k0, self.xt, self.yt, self.xs+dx, self.ys-dy, s)
        else:
            raise ValueError("Invalid source type. Choose 'monopole' or 'quadrupole'.")

        # Solve the linear system and get the source values(pressure fluctuation) for each panel
        pl = np.linalg.solve(self.A,S)
        

        pscat  = np.dot((np.dot(self.Gt , self.D) - self.Ht ) , pl)  # Scattered pressure
        pinc = -Sf   # Pressure from incident source
        ptarget =  pscat + pinc
        return ptarget

    def sources_from_target(self,t,sourceType='monopole'):
        """ This function computes the model source quadrupole from a list of sources,
        whith coordinates at (xc, yc) and (complex) intensity s, to a list of targets at (xt, yt)"""
        # Compute the right-hand side
        eps= self.eps
        T = monopole(self.k0, self.xc, self.yc, self.xt, self.yt, t)

        # Solve the linear system and get the source values(pressure fluctuation) for each panel
        pl = np.linalg.solve(self.A,T)
        
        p = np.empty((3,3), dtype=object)
        for idx,dx in enumerate([-eps, 0 , eps]):
            for idy,dy in enumerate([-eps, 0 , eps]):
                Tfin    = monopole(self.k0, self.xt, self.yt, self.xs+dx, self.ys+dy, t)
                pscat = np.dot((np.dot(self.Gs[idx,idy] , self.D) - self.Hs[idx,idy]) , pl)  # Scattered pressure
                pinc = -Tfin   # Pressure from incident source
                p[idx,idy] = pscat + pinc
        
        # return np.sum(psource*2-psourcexp-psourcexm)/eps**2
        if sourceType == 'monopole':
            return p[1,1]
        elif sourceType == 'quadrupole':
            return p
            # return (p[2,2]-p[2,0]+p[0,0]-p[0,2])/(2*eps)**2
            # return (2*p[1,1]-p[2,1]-p[0,1])/(eps)**2
            # return (2*p[1,1]-p[1,2]-p[1,0])/(eps)**2
            
            # return sum(psource*2-psourcexp-psourcexm)/eps**2, sum(psource*2-psourceyp-psourceym)/eps**2
        else:
            raise ValueError("Invalid source type. Choose 'monopole' or 'quadrupole'.")



def main(args):
    k0 = args.k0
    alphaH = args.alphaH
    omega = args.omega
    epsilon = args.epsilon
    field = args.field
    print('Epsilon used is: ' + str(epsilon))
    print('Omega used is: ' + str(omega))
    
    # Constants variables
    # R = 1.0e-3          # Radius of porous
    # kr = 4.0/np.pi  # Rayleigh conductivity
    # gamma = 0.5772156649    # Constant of Euler
    # print("alphaH/R = " + str(alphaH/R)) 

    xs,ys = sources()
    xt,yt, theta_o = targets()    
    # profil = Profil( k0, alphaH, epsilon, omega, xs,ys,xt,yt,Npsx=700, thickness=0.0001, c=1.0,eps=1e-2)

    s = [1] 
    print("Monopole source: " )
    pobs = profil.target_from_sources(s,sourceType='monopole')
    ps   = profil.sources_from_target([1],sourceType='monopole')
    print("Direct  Greens func: " + str(pobs))
    print("Adjoint Greens func: " + str(ps))
    print("Error: " + str(ps-pobs))
    
    print("Quadrupole source from monopoles: " )
    pobs = np.zeros((3,3),dtype=complex)
    for i,dx in enumerate([-profil.eps, 0 , profil.eps]):
        for j,dy in enumerate([-profil.eps, 0 , profil.eps]):
            pobs[i,j] = profil.target_from_sources(s,sourceType='monopole',dy=dy,dx=dx)[0]
    ps   = np.array(profil.sources_from_target([1],sourceType='quadrupole'),dtype=complex)
    pobsQuad = profil.target_from_sources(s,sourceType='quadrupole')[0]

    # pobs = (pobs[2,2]-pobs[2,0]+pobs[0,0]-pobs[0,2])/(2*profil.eps)**2
    # ps = (ps[2,2]-ps[2,0]+ps[0,0]-ps[0,2])/(2*profil.eps)**2

    print("Direct  Greens func (quad): " + str(pobsQuad))
    print("Direct  Greens func (mono): " + str(pobs))
    print("Adjoint Greens func (mono): " + str(ps))
    print("Error: " + str(ps-pobs))



    outputs = "results"
    if not os.path.isdir(outputs):
        os.makedirs(outputs)
    save = os.path.join(outputs, "k%0.1f_alphaH_R%d_Omega_%0.3f_res.dat" %(k0,int(alphaH/R),omega))
    f=open(save, "wt")

    for i in range(len(theta_o)):
         f.write(str(np.degrees(theta_o[i])) + " " + str(np.real(pobs[i])) + " " + str(np.imag(pobs[i])) + "\n")
    f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k0", type=float, help="Acoustic wavenumber",default=1.0)
    parser.add_argument("--alphaH", type=float, help="Open area fraction", default=0.0)
    parser.add_argument("--omega", type=float, help="Vacuum bending wave Mach number", default = 0.1)
    parser.add_argument("--epsilon", type=float, help="Intrinsic fluid-loading parameter", default = 0.0)
    parser.add_argument("--field", help="Flag to compute the field", default = False)
    
    args = parser.parse_args()

    print(args)
    main(args)
