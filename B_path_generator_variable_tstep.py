# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 08:29:22 2023

@author: teddy
"""

#import matplotlib.pyplot as plt
import numpy as np
#import random
import scipy.linalg as la
#from sys import argv
#import os

# %% define functions 
# do trilinear interpolation at N points in 3d space
def tri_lin_interp(pp, X0, scale=1):
    # absolute positions X
    X, Y, Z = X0[0,:], X0[1,:], X0[2,:]
    [Nx,Ny,Nz] = [np.size(pp,axis=0),np.size(pp,axis=1),np.size(pp,axis=2)]
    
    # pixel positions from coordinate positions in km
    # modulus function % impliments cyclic boundary conditions
    Xpixel, Ypixel, Zpixel = X*Nx/(Lx*scale) % Nx, Y*Ny/(Ly*scale) % Ny, Z*Nz/(Lz*scale) % Nz

    # pixel positions (rounded down)
    Xrd, Yrd, Zrd=Xpixel.astype(int), Ypixel.astype(int), Zpixel.astype(int)
    
    # pixel positions (rounded up)
    Xru, Yru, Zru=np.mod(Xrd + 1,Nx), np.mod(Yrd + 1,Ny), np.mod(Zrd + 1,Nz)
    ## Begin Trilinear interpolation.
    # Distance between two points
    Xrd = Xrd.astype(int)
    Yrd = Yrd.astype(int)
    Zrd = Zrd.astype(int)
    Xru = Xru.astype(int)
    Yru = Yru.astype(int)
    Zru = Zru.astype(int)
    
    # error in rounding down for each direction
    xd = (Xpixel-Xrd)/(Xru-Xrd)
    yd = (Ypixel-Yrd)/(Yru-Yrd)
    zd = (Zpixel-Zrd)/(Zru-Zrd)

    # data from all possible combs of rounding up/down
    ph000 = pp[Xrd, Yrd, Zrd]
    ph100 = pp[Xru, Yrd, Zrd]
    ph010 = pp[Xrd, Yru, Zrd]
    ph001 = pp[Xrd, Yrd, Zru]
    ph110 = pp[Xru, Yru, Zrd]
    ph011 = pp[Xrd, Yru, Zru]
    ph101 = pp[Xru, Yrd, Zru]
    ph111 = pp[Xru, Yru, Zru]
    
    # do linear interpolation
    ph00 = ph000*(1-xd)+ph100*xd
    ph01 = ph001*(1-xd)+ph101*xd
    ph10 = ph010*(1-xd)+ph110*xd
    ph11 = ph011*(1-xd)+ph111*xd
    
    ph0 = ph00*(1-yd)+ph10*yd
    ph1 = ph01*(1-yd)+ph11*yd
    
    ph = ph0*(1-zd)+ph1*zd
    return ph

# function to get value of B at an array of positions
# dsetField is the 3c cube of plasma
def gkyell_B_field_dynamic(positions, dsetField):

    N = np.shape(positions)[0]          # 4
    L = np.shape(positions)[1]          # 3
    
    loc = np.transpose(positions, (1,0))
    #loc = np.reshape(loc, (L, N))
    
    B_temp1 = np.zeros(np.shape(loc))
    for j in range(3):  
        index = 0+j
        pp = dsetField[:, :, :, index]  # pp is cube of plasma at current time. 'index' iterates through Bx, By, Bz
        # find B at the locations
        B_temp1[j,:] = tri_lin_interp(pp, loc)
        
    B_out = np.transpose(B_temp1, (1, 0))   # return array that is 3xN of plasma B values
    return B_out

def calc_RLEP(r): 
    # function calc_RLEP takes in array, each row of which represents a point in 3d space
    # it computes characteristic size, elongation, planarity of a s/c swarm

    # find the number of satellites as the number of rows of r
    N = np.size(r[:,0])
    if np.size(r[0,:]) == 2:
        r_tmp = np.zeros([N,3])
        r_tmp[:,:2] = r
        r = r_tmp
    # find the mesocentre rb
    rb = np.mean(r, axis=0)
    # calculate the Volumetric Tensor R
    R = np.zeros([3,3])
    for i in range(N):
        R += np.outer(r[i,:]-rb, r[i,:]-rb)/N
    # find the eigenvalues of R as value in lambdas
    temp = la.eig(R)
    lambdas = temp[0]
    # find semiaxes of quasi-ellipsoid a,b,c
    # check if eigenvalues are real
    if any(np.imag(lambdas) != np.zeros(3)):
        raise ValueError('Eigenvalue has imaginary component')
    lambdas_real = np.real(lambdas)
    #print(lambdas_real)
    [c,b,a] = np.sqrt( np.sort(lambdas_real) )
    # calculate L,E,P
    L = 2*a
    E = 1 - b/a
    P = 1 - c/b
    return [R,L,E,P]


def get_positions():
    # get_positions reads the text files containing HelioSwarm node/Hub position and time
    # data and returns them as numpy arrays
    length = np.zeros(9)
    for name in range(9):
        data = np.loadtxt("HS_config/n%s_clean.txt" % name,dtype = str)
        length[name] = np.size(data[:,4])
    L = np.min(length).astype(int)
    times = data[:L,0:4]
    positions = np.zeros([9,L,3])
    for name in range(9):
        data = np.loadtxt("HS_config/n%s_clean.txt" % name,dtype = str)    
        positions[name,:,:] = data[:L,4:7].astype(float)
    return [positions, times]

# %% Begin Script

# set random seed to make repeatable experiments
np.random.seed(0)

# parameters to adjust
N = 9                   # num spacecraft
hour = 150              # hour of helioswarm mission configuration to use for relative positional data
t_stride = 1            # time 'stride' between sucessive extracted points. stride=0 means s/c move through one static cube of plasma
t_min, t_max = 76, 300  # time indices of data cubes to use (ex. use timesteps 0 to 100)
L = 3                   # dimensions of space (3=3d data)
L_sc = 2000             # L = characteristic size of s/c config (km)
SAVE = False             # save data to .npy file or not
dt = 1/4                # time b/w timesteps (sec)
v_sc = 320              # constant velocity of spacecraft swarm (km/sec)
direction = [1, 0, 0]   # direction of s/c swarm travel [x,y,z]
rand_its = 10           # number of random locations to start s/c at

# define physical parameters of plasma cube
rhoi = 100
k0 = 0.02/rhoi
el = 0.2
Lx = 2*np.pi/k0
Ly = 2*np.pi/k0
Lz = 2*np.pi/k0/el
print('Size of plasma cube: [Lx,Ly,Lz] = ',[int(Lx), int(Ly), int(Lz)],'km')

# make list of timesteps
if t_stride != 0:
    t_inds = np.arange(t_min,t_max + 1, t_stride)  #list of all time steps
    t_steps = int(np.size(t_inds))                 #number of timesteps
else:
    t_inds = 76*np.ones([225]).astype(int) 
    t_steps = int(np.size(t_inds))         
times = np.arange(t_steps)*dt           

# extract spacecraft positional data
[positions, times_HS] = get_positions()
r_ = positions[:,hour,:] 
r_ = r_ - np.mean(r_,axis=0)            # center swarm at (0,0,0)
[R,L_shape,E,P] = calc_RLEP(r_)
r_ = (L_sc/L_shape)*(r_)                # resize swarm to have size L_sc
[R,L_shape,E,P] = calc_RLEP(r_)
print('s/c config shape: E=%.2f, P=%.2f, chi=%.2f, L=%.2f' %(E,P,np.sqrt(E**2 + P**2), L_shape))

# intialize vars
r0_save = np.zeros([L,rand_its]) 
B_sc_dynamic = np.zeros([N,L,t_steps,rand_its])
r_sc_dynamic = np.zeros([N,L,t_steps,rand_its])

# start s/c swarm at different random locations and evolve them forward in time
for it in range(rand_its):
    itz = int(it % 3)
    ity = int((it-itz)/ 3)

    # define random starting location
    r0 = np.random.rand(L)*np.array([Lx, Ly, Lz])    # random offset from (0,0,0) that s/c start at
    r0[0] = 0       # force us to start with barycenter of s/c config at x=0 positional coordinate
    r0_save[:,it] = r0
    
    # shift s/c to starting loc.
    r = r_ + r0
    
    # create list of s/c positions that evolves using direction vector and velocity defined above
    r_sc = np.tile(np.expand_dims(r, axis=(2)),(1,1,t_steps))
    t_sc_all = np.tile(np.expand_dims(times, axis=(0,1)),(N,L,1))
    V_sw = np.zeros(np.shape(t_sc_all))
    for i in range(np.size(direction)):
        V_sw[:,i,:] = (direction[i]/la.norm(direction))*t_sc_all[:,i,:]
    V_sw = v_sc*V_sw
    r_sc += V_sw
    r_sc_dynamic[:,:,:,it] = r_sc
    
# %% step through each timestep of plasma
# extract B values at positions of spacecrafts computed above

# static plasma case
if t_stride == 0:
    dsetField = np.load(r'E:\npy_files\ot3D_field_%i.npy' %t_inds[0])
    
# time evolving plasma case
for i in range(t_steps):
    if np.mod(i,int(t_steps/10)) == 0:
            print('   t_step %i of %i' %(i,t_steps))
    if t_stride != 0:
        dsetField = np.load(r'E:\npy_files\ot3D_field_%i.npy' %t_inds[i])
        
    for it in range(rand_its):
        B_sc_dynamic[:,:,i,it] = gkyell_B_field_dynamic(r_sc_dynamic[:,:,i,it], dsetField)

r_field_dynamic = np.reshape(r_field_dynamic,(Nz_timeseries,Ny_timeseries, L, t_steps, rand_its))

# %% save generated data
path = 'B_samplings/%isc/stride_%i' %(N,t_stride)
if SAVE == True:
    np.save(path + '/B_sc_tetra_%i.npy' %hour, B_sc_dynamic)        # B at s/c positions over time
    np.save(path + '/r_sc_tetra_%i.npy' %hour, r_sc_dynamic)        # s/c positions over time
    np.save(path + '/t_sc_all_tetra_%i.npy' %hour, t_sc_all)        # time vector
    np.save(path + '/r0_save_tetra_%i.npy' %hour, r0_save)          # s/c barycenters
