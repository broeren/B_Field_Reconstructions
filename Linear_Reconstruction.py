# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 08:29:22 2023

@author: teddy
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
from datetime import datetime
from itertools import combinations
from scipy.stats import wasserstein_distance

np.random.seed(0)
startTime = datetime.now()

# compute scaled size of plasma simulation based on ion gyroradius
rhoi = 100
k0 = 0.02/rhoi
el = 0.2
Lx = 2*np.pi/k0
Ly = 2*np.pi/k0
Lz = 2*np.pi/k0/el

def find_polys(loc,n):
    # find_poly creates a list of all n length combinations of numbers of 0 through N
    N = np.size(loc,0)
    temp = list(range(N))
    return list(combinations(temp,n))

def Linear3D(r,B,x0):
    # First order Linear method
    # r is 4X3 array of s/c absolute positions
    # B is 4x3 array of s/c measured B values
    # x0 is 3x1 vector (location to calculate J)
    
    # recenter problem to desired location
    r = r - np.vstack((x0,x0,x0,x0))
    
    # set constants vector of measured B values
    b = np.zeros([12,1])
    b[0:4,0], b[4:8,0], b[8:12,0] = B[:,0], B[:,1], B[:,2]
    
    # set linking matrix of positions
    H = np.zeros([4,4])
    H[:,0] = np.ones([4])
    H[:,1:4] = r
    A = np.zeros([12,12])
    A[0:4,0:4],A[4:8,4:8],A[8:12,8:12] = H,H,H
    
    # solve linear system Ax=b
    x = np.linalg.solve(A, b)
    
    # extract divergence, mag field at x0, and J
    mu_0 = 1
    CurlB = (1/mu_0)*np.array([x[10]-x[7],x[3]-x[9],x[5]-x[2]]).T
    divB = x[1] + x[6] + x[11]
    B_x0 = np.array([x[0],x[4],x[8]]).T
        
    return divB, B_x0, CurlB

def calc_RLEP(r): 
    # function calc_RLEP takes in array, each row of which represents a point in 3d space
    # it returns the Volumettric tensor (R), characteristic size (L), elongation (E), and planarity (P) 
    
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
    E = 1 - np.divide(b,a)
    P = 1 - np.divide(c,b)
    return [R,L,E,P]

# %% Configurable parameters
SAVE_figs = True
fig_path = 'figures/Linear'
B_data_path = 'sc_data'

chi_thres = 0.6     # tetrahedral shape threshold of subsets included in compuation: in range [0,np.sqrt(2)]
L_coeff = np.inf    # distance threshold: maximum distance (1 = L) from tetrahedron's barycenter
# %% load synthetic spacecraft and simulation data
B_sc = np.load(B_data_path + '/B_sc.npy')
B_field = np.load(B_data_path + '/B_field.npy')
r_sc = np.load(B_data_path + '/r_sc.npy')
r_field = np.load(B_data_path + '/r_field.npy')
t_sc = np.load(B_data_path + '/t_sc.npy' )
r0 = np.load(B_data_path + '/r0.npy')

# %% extract quantities from variables
times = np.mean(t_sc,axis=0)
B_field = np.transpose(B_field,(3,2,0,1))
r_field = np.transpose(r_field,(3,2,0,1))

# extract simulation properties
Nz_timeseries = np.shape(B_field)[0]
Ny_timeseries = np.shape(B_field)[1]
N_timeseries = Nz_timeseries*Ny_timeseries

N = np.shape(B_sc)[0]
L = np.shape(B_field)[2]
t_steps = np.shape(B_field)[3]

r_field = np.reshape(r_field,(Nz_timeseries*Ny_timeseries, L, t_steps))
B_field = np.reshape(B_field,(Nz_timeseries*Ny_timeseries, L, t_steps))
# %% print configuration shape
r_ = r_sc[:,:,0]
r_ = (r_ - np.mean(r_,axis=0))
[R,L_sc,E,P] = calc_RLEP(r_)
print('E=%.2f, P=%.2f, L=%.2f km' %(E,P, L_sc))

# %% load positional data
print('   Computing Linear recon')
Errors_list = np.zeros([Nz_timeseries, Ny_timeseries, t_steps])
B_recon_save = np.zeros([N_timeseries, L, t_steps])
for t_ind in range(t_steps):
    if np.mod(t_ind,25) == 0:
        print('        -Timestep %i of %i' %(t_ind, t_steps))
    r_sc_taylor = np.reshape(np.transpose(r_sc[:,:,t_ind],(1,0)),(3,N))
    B_sc_taylor = np.reshape(np.transpose(B_sc[:,:,t_ind],(1,0)),(3,N))
    r_recon_taylor = np.reshape(np.transpose(r_field[:,:,t_ind],(1,0)),(3,N_timeseries))
    
    # Linear reconstruction of B field 
    # find shape of all tetrahedral subsets of spacecraft
    if t_ind == 0:
        poly_indices = find_polys(r_sc_taylor.T,4)
        m = np.shape(poly_indices)[0]
        L_save = np.zeros([m])
        E_save = np.zeros([m])
        P_save = np.zeros([m])
        for j in range(m):
            # j is row of indices array to use points of 
            # (a,b,c,d)-> use s/c numbered a,b,c,d to define tetrahedron
            index = list(poly_indices[j])
            [R, L_save[j], E_save[j], P_save[j]] = calc_RLEP(r_sc_taylor[:,index].T)
        
        chi_save = np.sqrt(E_save**2 + P_save**2)
        shape_good = chi_save <= chi_thres
        print('   %i/%i tetrahedra pass the shape threshold: (E^2 + P^2)^{1/2} < %.3f' %(np.sum(shape_good),m,chi_thres))
        
    B_recon = np.zeros([N_timeseries,L])
    Dist2Bary = np.zeros([m])
    tetra_passed = np.zeros([N_timeseries])
    # loop over every reconstructed point k
    for k in range(N_timeseries):
        X0 = r_recon_taylor[:,k]
        
        # find the distance from the barycenter of each tetrahedron to each reconstructed point
        Dist2Bary = la.norm( np.tile(np.expand_dims(X0,axis=1),(1,m)) - np.mean(r_sc_taylor[:,poly_indices],axis=2), axis=0)
        
        # determine which tetrahedra are well-shaped and nearby to the chosen point k
        #ind_good = np.zeros([m]).astype(bool)
        ind_good = np.logical_and(shape_good, Dist2Bary < L_coeff*L_save)
        
        B_temp = np.zeros([m,3])
        DivB_temp = np.zeros([m])
        CurlB_temp = np.zeros([m,3])
        # loop over tetrahedra
        for j in np.where(ind_good == True)[0]:
            if ind_good[j] == True:
                # find 1st order Linear solution using this particular subset of s/c
                index = list(poly_indices[j])
                DivB_temp[j], B_temp[j,:], CurlB_temp[j,:] = Linear3D(r_sc_taylor[:,index].T,B_sc_taylor[:,index].T,X0)
                tetra_passed[k] += 1
               
        # use only these well-shaped and nearby tetrahedron in the estimate for B at this point k
        if tetra_passed[k] != 0:
            
            B_recon[k,:] = np.median(B_temp[ind_good,:],axis=0)
            
    # %% visualize B_recon
    B_field_cube = np.reshape(B_field[:,:,t_ind],(Nz_timeseries, Ny_timeseries, L))
    B_recon_cube = np.reshape(B_recon,(Nz_timeseries, Ny_timeseries, L))
    
    # %% compute error
    Errors_cube = 100* np.divide(la.norm(B_recon_cube - B_field_cube,axis=2), la.norm(B_field_cube,axis=2), out=np.ones_like(la.norm(B_field_cube,axis=2)), where = la.norm(B_field_cube,axis=2)!=0)
    Errors_list[:,:,t_ind] = Errors_cube
    B_recon_save[:,:,t_ind] = B_recon
    
Errors_list[Errors_list == 0] = np.nan
# %% visualize B_recon example (x-y plane)
dx = np.median(np.diff(r_field[:,0,:]))
dy = np.median(np.diff(r_field[:,1,:],axis=0))
dz = np.median(np.diff(np.unique(r_field[:,2,0])))

min_ind, max_ind = 0, t_steps
t_ind_save = np.arange(min_ind, max_ind)
r_field_cube = np.reshape(r_field,(Nz_timeseries, Ny_timeseries, L, t_steps))
B_field_cube = np.reshape(B_field,(Nz_timeseries, Ny_timeseries, L, t_steps))
B_recon_cube = np.reshape(B_recon_save,(Nz_timeseries, Ny_timeseries, L, max_ind-min_ind))
r_field_cube_plot = r_field_cube - np.tile(np.expand_dims(r0,axis=(0,1,3)),(Nz_timeseries,Ny_timeseries,1,t_steps))
r_sc_plot = r_sc - np.tile(np.expand_dims(r0,axis=(0,2)),(N,1,t_steps))
 
# %% compute relative Wasserstein distance (within distance L) of Bx for every time slice and every iteration
radii_yz = np.sqrt( (r_field_cube[:,:,1,:] - r0[1])**2 + (r_field_cube[:,:,2,:] - r0[2])**2)

# difference between recon and simulation
Wx = np.zeros([ max_ind-min_ind])

# difference between simulation and constant value
Wx0 = np.zeros([ max_ind-min_ind])
    
for i in range(max_ind-min_ind):
    w1 = radii_yz[:,:,min_ind+i] < L_sc
    
    d1x = B_recon_cube[:,:,0,i]
    d2x = B_field_cube[:,:,0,min_ind+i]
    d0x = np.mean(B_field_cube[:,:,0,min_ind+i]) + np.zeros(np.shape(B_field_cube[:,:,0,min_ind+i]))
    Wx[i] = wasserstein_distance(d1x[w1], d2x[w1])
    Wx0[i] = wasserstein_distance(d0x[w1], d2x[w1])

print('\nRelative Wasserstein Distance of Bx components:')
print('    median W_d(B_x) = %.3f' %(np.median(Wx/Wx0)))
print('    10-90%% W_d(B_x) in [%.3f,%.3f]' %(np.percentile(Wx/Wx0,10), np.percentile(Wx/Wx0,90)))

# %% plot the Bx component of the reconstruction vs simulation
B_component = 1         # plot the x component
x_slice = int(t_steps/2)
z_slice = int(Nz_timeseries/2)
maxs = np.min(r_field,axis=(0,2)) - r0
mins = np.max(r_field,axis=(0,2)) - r0
x_plot = np.linspace(mins[0],maxs[0], np.shape(B_field_cube)[3])
y_plot = np.linspace(mins[1],maxs[1], np.shape(B_field_cube)[1])
z_plot = np.linspace(mins[2],maxs[2], np.shape(B_field_cube)[0])

fig, ax = plt.subplots(2,1,sharex=True, sharey=True, figsize=(7,5))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
ax[0].set_title('$B_x$ of Simulated Field', fontsize=18)
#im0 = ax[0].pcolormesh(x_plot, y_plot, B_field_cube[z_slice,:,B_component,:], cmap='bwr',shading='nearest', vmin=-.1, vmax=.1)
im0 = ax[0].pcolormesh(x_plot, y_plot, B_field_cube[z_slice,:,B_component,:], cmap='bwr',shading='nearest', vmin=-.1, vmax=.1)
ax[0].scatter(r_sc[:,0,:] - r0[0], r_sc[:,1,:] - r0[1], s=0.5, c='k', alpha=0.2)
ax[0].set_ylabel('$y$ km', fontsize=16)
ax[0].set_xlim(np.min(r_sc,axis=(0,2))[0],np.max(r_sc,axis=(0,2))[0])
ax[0].set_ylim(-2000,2000)

ax[1].set_title('$B_x$ of Linear Reconstructed Field', fontsize=18)
ax[1].pcolormesh(x_plot, y_plot, B_recon_cube[z_slice,:,B_component,:], cmap='bwr',shading='nearest', vmin=-.1, vmax=.1)
ax[1].scatter(r_sc[:,0,:] - r0[0], r_sc[:,1,:] - r0[1], s=0.5, c='k', alpha=0.2)
ax[1].set_ylabel('$y$ km', fontsize=16)
ax[1].set_xlabel('$x$ km', fontsize=16)
ax[1].set_ylim(-2000,2000)

for a in ax.flat:
    a.minorticks_on()
    a.set_aspect('equal')

cb_ax = fig.add_axes([0.92, 0.19, 0.02, 0.6])
cbar = fig.colorbar(im0, cax=cb_ax)
cbar.set_label('$B_y$', rotation=0, fontsize=20)
if SAVE_figs == True:
    plt.savefig(fig_path + '/By_recon_xy.png', format='png',dpi = 600)

# %% visualize B_recon example (xy plane)
x_res, y_res, z_res = 15, 6, 4
y_inds = slice(0,Ny_timeseries,y_res)
x_inds = slice(min_ind,max_ind,x_res)
x_inds2 = slice(0, max_ind - min_ind , x_res)

minx, maxx = np.min(r_sc_plot[:,0,x_inds]), np.max(r_sc_plot[:,0,x_inds])
miny, maxy = -2*L_sc, 2*L_sc
minz, maxz = -2*L_sc, 2*L_sc

SCALE = 1
a = 3
fig = plt.figure(figsize=(8,5))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
L2 = plt.quiver(r_field_cube_plot[z_slice,y_inds,0,x_inds], r_field_cube_plot[z_slice,y_inds,1,x_inds], B_recon_cube[z_slice,y_inds,0,x_inds2], B_recon_cube[z_slice,y_inds,1,x_inds2],scale=SCALE, headaxislength=3, headwidth=3, headlength=3, facecolor='red',alpha=0.5)
L3 = plt.quiver(r_field_cube_plot[z_slice,y_inds,0,x_inds], r_field_cube_plot[z_slice,y_inds,1,x_inds], B_field_cube[z_slice,y_inds,0,x_inds], B_field_cube[z_slice,y_inds,1,x_inds],scale=SCALE, headaxislength=3, headwidth=3, headlength=3, facecolor='blue',alpha=1)
plt.scatter(r_sc_plot[:,0,:], r_sc_plot[:,1,:], s=2, color='w', marker='o',alpha=0.7, edgecolors={'k'})
plt.axis([minx, maxx, miny, maxy])
plt.xlabel(r'$x$ (km)' ,fontsize=24)
plt.ylabel(r'$y$ (km)',fontsize=24)
plt.grid(b=True, which='major', color='#999999', linestyle='-', alpha=0.3)
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.1)
plt.quiverkey(L2, 0.30, 1.02, SCALE/20, r'Reconstruction',labelpos ='E', fontproperties={'size': 16})
plt.quiverkey(L3, 0.67, 1.02, SCALE/20, r'Simulation',labelpos ='E' , fontproperties={'size': 16})
plt.title('\n')
plt.tight_layout()
if SAVE_figs == True:
    plt.savefig(fig_path + '/Brecon_xy.png', format='png',dpi = 600)
    
# %% visualize B_recon example (yz plane)
y_inds = slice(0,Ny_timeseries,y_res)
z_inds = slice(0,Nz_timeseries,z_res)

SCALE = 4
a = 3
fig = plt.figure(figsize=(8,5))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
L2 = plt.quiver(r_field_cube_plot[z_inds,y_inds,1,x_slice], r_field_cube_plot[z_inds,y_inds,2,x_slice], B_recon_cube[z_inds,y_inds,1,x_slice- min_ind], B_recon_cube[z_inds,y_inds,2,x_slice- min_ind],scale=SCALE, headaxislength=3, headwidth=3, headlength=3, facecolor='red',alpha=0.5)
L3 = plt.quiver(r_field_cube_plot[z_inds,y_inds,1,x_slice], r_field_cube_plot[z_inds,y_inds,2,x_slice], B_field_cube[z_inds,y_inds,1,x_slice], B_field_cube[z_inds,y_inds,2,x_slice],scale=SCALE, headaxislength=3, headwidth=3, headlength=3, facecolor='blue',alpha=1)
plt.scatter(r_sc_plot[:,1,x_slice], r_sc_plot[:,2,x_slice], s=40, color='w', marker='o',alpha=0.7, edgecolors={'k'})
plt.axis([miny, maxy, minz, maxz])
plt.xlabel(r'$y$ (km)' ,fontsize=24)
plt.ylabel(r'$z$ (km)',fontsize=24)
plt.grid(b=True, which='major', color='#999999', linestyle='-', alpha=0.3)
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.1)
plt.quiverkey(L2, 0.30, 1.02, SCALE/20, r'Reconstruction',labelpos ='E', fontproperties={'size': 16})
plt.quiverkey(L3, 0.67, 1.02, SCALE/20, r'Simulation',labelpos ='E' , fontproperties={'size': 16})
plt.title('\n')
plt.tight_layout()
if SAVE_figs == True:
    plt.savefig(fig_path + '/Brecon_yz.png', format='png',dpi = 600)
  
# %% plot error in reconstruction in xy plane
r_field_cube = np.reshape(r_field,(Nz_timeseries, Ny_timeseries, L, t_steps))
x = r_field_cube[z_slice,:,0,:] - r0[0]
y = r_field_cube[z_slice,:,1,:] - r0[1]
level_boundaries = np.linspace(0, 10, 11)
color = Errors_list[z_slice,:,:]

plt.figure(figsize=(10,6))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.contourf(x, y, color, level_boundaries)
cbar = plt.colorbar()
cbar.set_label('Error \%', rotation=90, fontsize=16)
plt.scatter(r_sc[:,0,:], r_sc[:,1,:] - r0[1], color='black',s=3,alpha=0.4)
plt.title('$N=%i$ Linear Reconstruction: %ix%ix%i pts\n $E=%.2f$, $P=%.2f$, $\chi=%.2f$, $L=%.1f$km' %(N,t_steps,Ny_timeseries,Nz_timeseries,E,P,np.sqrt(E**2 + P**2),L_sc),fontsize=20, wrap=True)
plt.xlabel(r'$x$ (km)',fontsize=20)
plt.ylabel(r'$y$ (km)',fontsize=20)
plt.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.25)
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.05)
plt.tight_layout()
if SAVE_figs == True:
    plt.savefig(fig_path + '/Brecon_error.png', format='png',dpi = 600)
    
# %% plot error in reconstruction in yz plane
z_slice = int((Nz_timeseries-1)/2)
r_field_cube = np.reshape(r_field,(Nz_timeseries, Ny_timeseries, L, t_steps))
y = r_field_cube[:,:,1,0] - r0[1]
z = r_field_cube[:,:,2,0] - r0[2]
level_boundaries = np.linspace(0, 10, 11)
color = np.nanmean(Errors_list[:,:,:], axis=2)

plt.figure(figsize=(10,6))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.contourf(z, y,  color, level_boundaries)
cbar = plt.colorbar()
cbar.set_label('Error \%', rotation=90, fontsize=16)
plt.scatter(r_sc[:,2,:] - r0[2],r_sc[:,1,:] - r0[1],  color='black',s=3,alpha=0.4)
#plt.title('$N=%i$ Linear Reconstruction: %ix%ix%i pts\n $E=%.2f$, $P=%.2f$, $\chi=%.2f$, $L=%.1f$km' %(N,t_steps,Ny_timeseries,Nz_timeseries,E,P,np.sqrt(E**2 + P**2),L_sc),fontsize=20, wrap=True)
plt.title('Linear Reconstruction Error',fontsize=20, wrap=True)

plt.xlabel(r'$x$ (km)',fontsize=20)
plt.ylabel(r'$y$ (km)',fontsize=20)
plt.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.25)
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.05)
plt.tight_layout()
if SAVE_figs == True:
    plt.savefig(fig_path + '/Brecon_error.png', format='png',dpi = 600)

# %% volume of reconstruction
print('\nPoint-Wise Error Volume Fraction:')
print('   Error < 20%%: %.3f' %(np.sum(Errors_list < 20)/np.sum(Errors_list > 0)))
print('   Error < 15%%: %.3f' %(np.sum(Errors_list < 15)/np.sum(Errors_list > 0)))
print('   Error < 10%%: %.3f' %(np.sum(Errors_list < 10)/np.sum(Errors_list > 0)))
print('   Error < 5%%: %.3f' %(np.sum(Errors_list < 5)/np.sum(Errors_list > 0)))
print('   Error < 1%%: %.3f' %(np.sum(Errors_list < 1)/np.sum(Errors_list > 0)))

# %% plot volume fraction as a function of error %
pts = 1000
thres_save = np.logspace(-1,2,pts)
tot = np.zeros(pts)
for i in range(pts):
    tot[i] = np.sum(Errors_list < thres_save[i])/np.sum(Errors_list > 0)

plt.figure()
plt.plot(thres_save, tot, c='b')
plt.xscale('log')
plt.ylim(0,1)
plt.xlim(1,100)
plt.xlabel(r'Error \%',fontsize=20)
plt.ylabel(r'Fraction of Pts',fontsize=20)
plt.title('Fraction of Volume Reconstructed' )
plt.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.25)
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.1)
if SAVE_figs == True:
    plt.savefig(fig_path + '/Error_frac.png', format='png',dpi = 600)

# %% print simulation time
endTime = datetime.now()
print('Time to execute:',endTime-startTime)