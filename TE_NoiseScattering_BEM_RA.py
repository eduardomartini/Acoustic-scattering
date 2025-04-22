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
    theta = np.linspace(0, 2.0 * np.pi, 360)
    r0 = 50.0
    x = xcenter + r0 * np.cos(theta)
    y = r0 * np.sin(theta)
    # x = x[0:-1]
    # y = y[0:-1]
    return x, y, theta

# Greens functions from a source to a target 
def monopole(k0, xt, yt, xc, yc, s):
    """ This function computs the model source monopole from a list of sources, 
    whith coordinates at (xc, yc) and (complex) intensity s, to a list of targets at (xt, yt)"""
    
    S = []
    for i in range(len(xt)):
        p = 0 
        for j in range(len(xc)):
            dx = xt[i] - xc[j]
            dy = yt[i] - yc[j]
            
            arg = k0 * np.sqrt(dx**2 + dy**2)
            
            Ha0 = hankel1(0,arg)  
            
            p +=  s[j]*( 1.0j / 4.0) * Ha0
            
        S.append( p )
    return S

# def dipole(k0, xt, yt, xc, yc, s):
#     """ This function computs the model source dipole from a list of sources, 
#     whith coordinates at (xc, yc) and (complex) intensity s, to a list of targets at (xt, yt)"""
    
#     S = []
#     for i in range(len(xt)):
#         p = 0 
#         for j in range(len(xc)):
#             dx = xt[i] - xc[j]
#             dy = yt[i] - yc[j]
            
#             arg = k0 * np.sqrt(dx**2 + dy**2)
            
#             Ha1 = hankel1(1,arg)
            
#             p +=  s[j]*Ha1
            
#         S.append( p )
#     return S



def quadrupole(k0, xt, yt, xc, yc, s):
    """ This function computs the model source quadrupole from a list of sources, 
    whith coordinates at (xc, yc) and (complex) intensity s, to a list of targets at (xt, yt)"""
    
    S = []
    for i in range(len(xt)):
        p = 0 
        for j in range(len(xc)):
            dx = xt[i] - xc[j]
            dy = yt[i] - yc[j]
            
            arg = k0 * np.sqrt(dx**2 + dy**2)
            
            Ha0 = hankel1(0,arg)  # Hankel of 1 order 0
            Ha1 = hankel1(1,arg)
            Ha2 = hankel1(2,arg)
            
            arg1_y2 = - 0.5 * (Ha0 - Ha2) * ( k0**2 * dx * dy / ( dx**2 + dy**2))
            arg2_y2 =  Ha1 * k0 * dx * dy / (( dx**2 + dy**2)**(3.0/2.0))
            
            p +=  s[j]*(1j / 4.0 ) * (arg1_y2 + arg2_y2)
            
        S.append( p )
    return S

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
        self.n1 = (y2 - y1)/self.length
        self.n2 = -(x2 - x1)/self.length


class Profil:
    """This class contains profile information."""
    def __init__(self, k0, alphaH, epsilon, omega, xs,ys,xt,yt,Npsx=400, thickness=0.002, c=1.0):
        # Geometry information
        # Npsx = 400  # number of panel created on each surface
        Nps = 2 * Npsx + 2  # Total of panels

        # c = 1.0     # Chord of the plate
        self.xs = xs
        self.ys = ys
        self.xt = xt
        self.yt = yt
        self.k0 = k0
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

        self.x1 = [pi.x1 for pi in panels]
        self.y1 = [pi.y1 for pi in panels]
        self.x2 = [pi.x2 for pi in panels]
        self.y2 = [pi.y2 for pi in panels]
        self.xc = [pi.xc for pi in panels]
        self.yc = [pi.yc for pi in panels]
        self.n1 = [pi.n1 for pi in panels]
        self.n2 = [pi.n2 for pi in panels]
        self.ds = [pi.length for pi in panels]

        self.H , self.G  = matrixv1.mntmat.hgmatrix(k0, self.x1,self.y1,
                                                        self.x2,self.y2,
                                                        self.xc,self.yc,
                                                        self.n1,self.n2,
                                                        self.ds,Nps)
        """
        ==================================================================================
        ----------------- Import Structural Modal Basis information ----------------------
                                Implement poroelastic materials
        ==================================================================================
        """

        beta = np.loadtxt('modalBasis/beta.txt')   # Read the Non-dimensional frequency
        phi = np.loadtxt('modalBasis/modes.txt')   # Read the modal basis 
        nm = len(beta)   # define number of modes

        nplate = Nps

        # Building the matrix D
        self.D = matrixv1.mntmat.poroelastic(nplate,k0, alphaH, epsilon, omega, self.n2, self.ds, beta, phi,Nps,nm)

        # Compute LHS
        self.A = self.H - np.dot(self.G, self.D)
        # Far field matrix H, G for the target points
        self.Ht, self.Gt = matrixv1.mntmat.hgobs   (k0,xt,yt,self.x1,self.y1,self.x2,self.y2,self.n1,self.n2,self.ds,len(self.xt),Nps)
        # Far field matrix H, G for the source points
        self.Hs, self.Gs = matrixv1.mntmat.hgobs   (k0,xs,ys,self.x1,self.y1,self.x2,self.y2,self.n1,self.n2,self.ds,len(self.xs),Nps)





def main(args):
    k0 = args.k0
    alphaH = args.alphaH
    omega = args.omega
    epsilon = args.epsilon
    field = args.field
    print('Epsilon used is: ' + str(epsilon))
    print('Omega used is: ' + str(omega))
    
    # Constants variables
    R = 1.0e-3          # Radius of porous
    kr = 4.0/np.pi  # Rayleigh conductivity
    gamma = 0.5772156649    # Constant of Euler
    print("alphaH/R = " + str(alphaH/R)) 

    xs,ys = sources()
    xt,yt, theta_o = targets()    
    profil = Profil( k0, alphaH, epsilon, omega, xs,ys,xt,yt,Npsx=400, thickness=0.002, c=1.0)

    s = [1] 
    # Compute the right-hand side
    S = quadrupole(k0, profil.xc, profil.yc, xs, ys, s)

    # Solve the linear system and get the source values(pressure fluctuation) for each panel
    pl = np.linalg.solve(profil.A,S)
    
    Sf = np.asarray(quadrupole(k0, xt, yt, xs, ys, s))

    pscat  = np.dot((np.dot(profil.Gt , profil.D) - profil.Ht ) , pl)  # Scattered pressure
    pinc = -Sf   # Pressure from incident source
    pobs =  pscat + pinc

    outputs = "results"
    if not os.path.isdir(outputs):
        os.makedirs(outputs)
    save = os.path.join(outputs, "k%0.1f_alphaH_R%d_Omega_%0.3f_res.dat" %(k0,int(alphaH/R),omega))
    f=open(save, "wt")

    for i in range(len(theta_o)):
         f.write(str(np.degrees(theta_o[i])) + " " + str(np.real(pobs[i])) + " " + str(np.imag(pobs[i])) + "\n")
    f.close()
    """
    ######################################################################################## 
    # Compute the noise scattered for a rectangular mesh to obtain the field around
    # Default is false, if is desired set up True 
    ########################################################################################
    """
    if field:
        # Define the domian
        nx = 200
        ny = 200
        lx = 3.0
        ly = 1.0
        xs = -1.0
        ys = -1.0

        # x = np.zeros((nx,ny))
        # y = np.zeros((nx,ny))

        # x[0,:] = xs
        # y[:,0] = ys
        # dx = lx/(nx-1)
        # dy = ly/(ny-1)
        # for i in range(1,nx):
        #     x[i,:] = x[i-1,:] + dx
        # for j in range(1,ny):
        #     y[:,j] = y[:,j-1] + dy
        

        xx = np.linspace(xs,lx,nx)
        yy = np.linspace(ys,ly,ny)

        x,y = np.meshgrid(xx,yy)

        z1 = z[0]
        z2 = z[1]

        dpdn = np.dot(D, pl)

        pfield= np.zeros((nx,ny),dtype=complex)
        pfield = matrixv1.mntmat.field(k0,z1,z2,x1,y1,x2,y2,n1,n2,ds,dpdn,pl,x,y,nx,ny,Nps)

        field_outs = "field_outputs"
        if not os.path.isdir(field_outs):
            os.makedirs(field_outs)
        savefield = os.path.join(field_outs, "k%0.1f_alphaH_R%d_Omega_%0.3f.dat" %(k0,int(alphaH/R),omega)) 
        f=open(savefield, "wt")  
        # Format tecplot
        f.write('TITLE = "mesh" \n')
        f.write('VARIABLES = "X", "Y", "p"\n')
        f.write('ZONE I = '+ str(nx) + ', J = '+str(ny)+', F=POINT \n')
        
        for j in range(ny):
            for i in range(nx):
                f.write( "{0:0.8E} \t {1:0.8E} \t {2:0.8E} \n" .format(theta_o[i,j], pfield[i,j].real,pfield[i,j].imag))

        f.close()
    
    # Plot the Directivity
    f=plt.figure(1) 
    ax = f.add_axes([0.1, 0.1, 0.8, 0.8], projection='polar')
    ax.plot(theta_o, np.abs(pobs),linestyle='-', linewidth='2')
    f.show()
    #ax.set_rgrids([20, 40, 60,80], angle=0.)
    #ax.legend(["k = 1.0","k = 5.0","k = 10.0"], loc='lower center', prop={'size':16}, bbox_to_anchor=(1.1,-0.1))

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
