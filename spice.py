#!/usr/bin/python
#Quick and dirty macrospin flip flop implementation since Mathematica is very slow
#All quantities are in SI units
#Thomas Windbacher, 23.3.2017

#Required libraries and functionalities
import os
import io
import time
import math  as ma
import numpy as np
from numpy import linalg as la
from scipy.integrate import ode
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import pylab

#start measuring run time
start_time = time.clock()

######################################################
#Constants and parameters
######################################################
kb    =  1.38064852e-23;  # J/K
hbar  =  1.054571800e-34; # Js
qe    =  1.60217662e-19;  # As
gamma =  2.11e5; # m/(As)
mu0   =  4.*ma.pi*1.e-7;  #Vs/Am
#Temperatur
Temp = 300.; #Kelvin

#Fixed time step dt 
dt    =  1.4875e-14;   #s
#Magnetic material properties
alpha =  0.01; # 1
#Uniaxial anisotropy
K1    =  2.e5; # J/m^3
K1vec =  np.array([0., 0., 1.]);# 1
#Exchange constant
Aexch =  2.e-11; # J/m
#Magnetization saturation
Ms    = 4.e5; # A/m 

####################################################
#Parameters for spin-transfer torque calculation
####################################################

#Orientation of polarization vector (must be normalized)
s       = np.array([0.,0.,1.]);
#Aplied current in Ampere
I       = 0.; 
#Polarization
p       = 0.9; # for oxides
#Field like torque relative strength
epsilon = 0.1

#####################################################
#Geometry related calculations
#####################################################
a  = 3.e-08
b  = 3.e-08
c  = 3.e-09

VA = a*a*c
VB = VA
VQ = a*2.*a*c
V = VA;# + VB + VQ

#####################################################
#Boundary conditions 
#####################################################

#By employing spherical coordinates |m|=1 is ensured
#Starting position of magnetization
phi   = 0.2
theta = 2.
m0    = np.array([ma.cos(phi)*ma.sin(theta),ma.sin(phi)*ma.sin(theta),ma.cos(theta)])
#Start and end of simulation time
t0    = 0.
tend  = 1.e-10

#Initialization of arrays with start values
sol   = np.array([m0])
t     = np.array([t0])

#Prefactor for RHS of equation
prefactor = -1.*gamma/(1+alpha*alpha)

######################################################################
#Define functions for effective field calculations
######################################################################
def Dz (a, b, c) :
 "Demagnetization factor along z-axis"
 return 1./ma.pi*( 
        ( a*a*a + b*b*b - 2.*c*c*c ) / (3.*a*b*c) + 
        c*( ma.hypot(a,c) + ma.hypot(b,c))/(a*b)  + 
        ((a*a + b*b - 2.*c*c)*ma.sqrt(a*a + b*b + c*c))/(3.*a*b*c) -
        (ma.pow(a*a + b*b,3./2.) + ma.pow(a*a + c*c, 3./2.) + ma.pow(b*b + c*c,3./2.))/(3.*a*b*c) +
        ((b*b - c*c)*ma.log((ma.sqrt(a*a + b*b + c*c) - a)/(ma.sqrt(a*a + b*b + c*c) + a)))/(2.*b*c) + 
        ((a*a - c*c)*ma.log((ma.sqrt(a*a + b*b + c*c) - b)/(ma.sqrt(a*a + b*b + c*c) + b)))/(2.*a*c) + 
        2.*ma.atan((a*b)/(c*ma.sqrt(a*a + b*b + c*c))) + 
        b*ma.log( ( ma.sqrt(a*a + b*b) + a) / (ma.sqrt(a*a + b*b) - a) ) / (2.*c) + 
        a*ma.log( ( ma.sqrt(a*a + b*b) + b) / (ma.sqrt(a*a + b*b) - b) ) / (2.*c) +
        c*ma.log( ( ma.sqrt(a*a + c*c) - a) / (ma.sqrt(a*a + c*c) + a) ) / (2.*b) +
        c*ma.log( ( ma.sqrt(b*b + c*c) - b) / (ma.sqrt(b*b + c*c) + b) ) / (2.*a)
        );

def Dx (x, y, z) :
 "Demagnetization factor along x-axis"
 return Dz(y,z,x);

def Dy (x, y, z) :
 "Demagnetization factor along y-axis"
 return Dz(z,x,y);

def Huni (m, axis, Ms, K) :
 "Definition of the uniaxial anisotropy field"
 return  (2.*K)/(mu0*Ms)*(m*axis)*axis;

def Hdem (m, Ddem, Ms) :
 "Demagnetization field"
 return -Ms*m*Ddem;

def Hstray (m, r, Ms, V) :
 "Stray/Dipole field from neighbor boxes"
 return Ms*V*(3.*(m*r).sum()*r - (r*r).sum()*m)/ma.pow((r*r).sum(),5./2.);

def Hexch (m1, m2, r, A, Ms) :
 "Exchange field"
 return (2.*A)/(mu0*Ms)*(m1 - m2)/ma.pow(r,2.);

#Definition of thermal field -> Finocchio,J.Appl.Phys.99 doi:10.1063/1.2177049
def Htherm(T, alpha, gamma, V, Ms, dt) :
 "Thermal field according to Finocchio JAPP 99, 10.1063/1.2177049" 
 return ma.sqrt((2.*kb*T*alpha)/(gamma*V*mu0*Ms*dt*(1.+alpha*alpha)))*np.array(np.random.normal(0.,1.,3));

#Defining the effective field term
def Heff(m, Ms, N, K1vec, K1,alpha,gamma,V,T,dt):
 "Effective field containing all physical contributions"
 return Huni(m, K1vec, Ms, K1) + Hdem(m, N, Ms) + Htherm(T,alpha,gamma,V,Ms,dt) 


#Definition of the spin transfer torque terms

def gox (p, m, s) :
 "Torque angle dependence for oxides"
 return p/(2. * ( 1.+p*p*(m*s).sum() ) );

def gmet (p, m, s) :
 "Torque angle dependence for metals"
 return 1./(-4.+ma.pow(1.+p,3.)*(3.+ (m*s).sum())/(4.*ma.pow(p,3./2.)));

def tau (m, s, I, p, V, Ms, gamma, epsilon, g) :
 "Torque term"
 mxs   = np.cross(m,s);
 mxmxs = np.cross(m,mxs);
 return (hbar*I)/(Ms*V*qe)*g*( mxmxs - epsilon*mxs );

#Define right hand side of the ODE
def f(t,m,prefactor):
 "RHS of LLG-ODE"
 #Normalize m - otherwise errors accumulate and m starts to drift 
 #towards bigger and bigger >1 values
 mm= m/la.norm(m)
 heff      = Heff(mm,Ms,N,K1vec,K1,alpha,gamma,V,Temp,dt)
 precesion = np.cross(mm,heff)
 damping   = alpha*np.cross(mm,precesion)
 stt       = tau(mm,s,I,p,V,Ms,gamma,epsilon,gox(p,m,s))
 rhs       = precesion + damping + stt
 return prefactor*rhs

# Geometry dependent but constant.
# It is sufficient to calculate only once before the integration
N = np.array([Dx(a,b,c),Dy(a,b,c),Dz(a,b,c)])

###################################################################
#Functions for data manipulation and export
###################################################################




##########################################################################################
##Main section
##########################################################################################

#Set up ode solver with Runge Kutta method
#atol : float or sequence absolute tolerance for solution
#rtol : float or sequence relative tolerance for solution
#nsteps : int Maximum number of (internally defined) steps allowed during one call to the solver.
#first_step : float
#max_step : float
#safety : float Safety factor on new step selection (default 0.9)
#ifactor : float
#dfactor : float Maximum factor to increase/decrease step size by in one step
#beta : float Beta parameter for stabilised step size control.
# verbosity : int Switch for printing messages (< 0 for no messages).
r = ode(f).set_integrator('dopri5',first_step=dt,max_step=dt,nsteps=1e5,rtol=1.e-3,verbosity=True)
r.set_initial_value(m0, t0).set_f_params(prefactor)
#
while r.successful() and r.t < tend:
      r.integrate(r.t+dt)
      t = np.append(t,r.t)
      print("t: %3.6e, %3.3f%%, %8i seconds" % (r.t, r.t/tend*100., time.clock() - start_time), end="\r" )
      sol  = np.append(sol,np.array([[ r.y[0],r.y[1],r.y[2] ]]),axis=0)
print("t: %3.6e, %3.3f%%, %8i seconds" % (t[-1], t[-1]/tend*100., time.clock() - start_time), end="\n" )
##########################################################################################
##Visualization of simulation results
##########################################################################################
#mx(t)
plt.plot(t, sol[:,0],label='mx')
#my(t)
plt.plot(t, sol[:,1],label='my')
#mz(t)
plt.plot(t, sol[:,2],label='mz')
#turn on legend
plt.legend()
#set x- and y-axis labels
plt.ylabel('normalized magnetization (1)')
plt.xlabel('time (s)')
plt.grid(True)
#put it on the display
plt.show()



