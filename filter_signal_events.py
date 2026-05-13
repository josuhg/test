# %%
import pandas as pd
import numpy as np
from subprocess import run
import os
import sys
from math import sin, cos, atan2, sqrt, pi, exp

# %%
mixing = sys.argv[1] # 'Um42'
channel = sys.argv[2] # 'to_numumu'
mode = sys.argv[3] # 'neutrino'
parent = sys.argv[4] # 'NM_kaon+'

# %%
um42_grid = [10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 24, 25, 26, 27, 28, 30, 31, 32, 33, 37, 43, 50, 58, 67, 78, 90, 104, 105, 107, 120, 134, 136, 139, 161, 187, 210, 212, 216, 233, 244, 246, 250, 290, 300, 310, 328, 329, 330, 335, 349, 358, 360, 369, 384, 388, 391, 395, 428, 432, 467, 474, 510, 519, 558, 568, 609, 623, 666, 682, 727, 748, 774, 776, 795, 819, 868, 880, 882, 897, 949, 983, 1018, 1020, 1037, 1058, 1077, 1117, 1132, 1180, 1237, 1293, 1352, 1410, 1416, 1477, 1490, 1499, 1552, 1583, 1587, 1614, 1675, 1676, 1700, 1745, 1763, 1769, 1843, 1862]

# %%
nombres = ['M_weight', 'N_weight', 'MN', 'USQUARED', 'L_lab_N', 'x_0', 'y_0', 'z_0', 'thetaN', 'phiN', 'x_f', 'y_f', 'z_f', 'M_width_CM', 'BR_M', 'N_total_width_CM', 'BR_N', 'time_N', 'time_nu', 'PoT_f']

# %%
alpha = 0.101
beta = 0

z_DUNE = 574
d_hall_detector = 2.43
h = 3 # 2 for the PRISM we remove 1m of active volume 
l = 5 # 4 for the PRISM we remove 1m of active volume 
w = 7 # 6 for the PRISM we remove 1m of active volume 
dOA_X = 0
dOA_Y = 0



z_DUNE_LArTPC = z_DUNE + d_hall_detector/np.cos(alpha)                     # total distance from the target to detector [in m]
z_DUNE_LArTPC2 = z_DUNE_LArTPC + l/np.cos(alpha)/2                         # total distance from the target to the center of detector [in m]
XMIN = dOA_X-w/2                                                           # m
XMAX = dOA_X+w/2                                                           # m
YMIN = dOA_Y-h/2                                                           # m
YMAX = dOA_Y+h/2                                                           # m
ZMIN = z_DUNE_LArTPC2-l/2                                                     # m
ZMAX = z_DUNE_LArTPC2+l/2                                                     # m

# def xhall(x, z):
#     return np.cos(beta) * x - np.sin(beta) * (z - z_DUNE_LArTPC2)
# def yhall(x, y, z):
#     return np.cos(alpha) * y - np.sin(alpha) * ( (z - z_DUNE_LArTPC2) * np.cos(beta) + np.sin(beta) * x)
# def zhall(x, y, z):
#     return np.sin(alpha) * y + np.cos(alpha) * ( (z - z_DUNE_LArTPC2) * np.cos(beta) + np.sin(beta) * x)

# %%
XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX

# %%
XMAX - XMIN - w, YMAX - YMIN - h, ZMAX - ZMIN - l

# %%
def compute_weight(Lambda, D, d):
    weight = exp( -D/Lambda )*(1-exp( -d/Lambda ))
    return weight

def rotation_y(alpha):
    """Rotation matrix around Y axis by angle alpha (radians)."""
    return np.array([
        [1, 0, 0],
        [0, cos(alpha), sin(alpha)],
        [0,-sin(alpha), cos(alpha)]
    ])

def HNL_direction(th, ph):
    """
    Particle direction vector u(th, ph) with physics convention:
    th: polar angle from +Z (radians)
    ph: azimuth around Z (radians)
    Returns a unit vector.
    """
    return np.array([
        sin(th) * sin(ph),
        sin(th) * cos(ph),
        cos(th)
    ])

def detector_intersection_and_distance(p0, u, box_min, box_max, d, theta):
    """
    Intersect a ray p(t)=p0+t*u with a rotated-translated axis-aligned box.
    Returns entry/exit points in XYZ and traveled distances from p0.

    Parameters
    ----------
    p0 : (3,) array_like
        Ray origin in XYZ.
    u : (3,) array_like
        Ray direction in XYZ (ideally unit-length).
    box_min, box_max : (3,) array_like
        Min/max corners of the box in local X'Y'Z'.
    d : (3,) array_like
        Translation of the box in XYZ.
    theta : float
        Rotation angle (radians) about global Y axis.

    Returns
    -------
    (p_enter, p_exit, dist_enter, dist_exit) or None
        Entry/exit points in XYZ and distances from p0 along the ray.
        If no intersection, returns None.
    """
    R = rotation_y(theta)
    R_T = R.T

    # Transform ray into local coordinates
    p0_prime = R_T @ (p0 - d)
    u_prime = R_T @ u

    # If u is not unit, we'll scale distances appropriately
    u_norm = np.linalg.norm(u)
    if u_norm == 0:
        return None

    tmin, tmax = -np.inf, np.inf

    for i in range(3):
        ui = u_prime[i]
        pi = p0_prime[i]
        if abs(ui) < 1e-12:
            # Parallel to slab: must be within bounds on this axis
            if pi < box_min[i] or pi > box_max[i]:
                return None
        else:
            t1 = (box_min[i] - pi) / ui
            t2 = (box_max[i] - pi) / ui
            t_near, t_far = min(t1, t2), max(t1, t2)
            tmin = max(tmin, t_near)
            tmax = min(tmax, t_far)
            if tmin > tmax:
                return None

    if tmax < 0:
        return None  # Entire box behind the ray origin

    # Entry is the first non-negative intersection along the ray
    t_enter = max(tmin, 0.0)
    t_exit = tmax

    p_enter_prime = p0_prime + t_enter * u_prime
    p_exit_prime  = p0_prime + t_exit  * u_prime

    # Back to global XYZ
    p_enter = d + R @ p_enter_prime
    p_exit  = d + R @ p_exit_prime

    # Distances from origin along the ray:
    # If u is unit, distance = t. Otherwise, scale by ||u||.
    dist_enter = t_enter * u_norm
    dist_exit  = t_exit  * u_norm

    return p_enter, p_exit, p_enter_prime, p_exit_prime, dist_enter, dist_exit

def random_decay_point_and_distance(p_enter, p_exit):
    """
    Choose a random point between p_enter and p_exit.
    Returns the point and the distance from p_enter.
    """
    # Random fraction between 0 and 1
    t = np.random.rand()
    # Point along the segment
    p_rand = p_enter + t * (p_exit - p_enter)
    # Distance from p_enter
    #dist = np.linalg.norm(p_rand - p_enter)
    dist = np.linalg.norm(p_exit - p_enter)
    return p_rand, dist

InvGeVtoMeters = 0.197e-15
def LengthLab_N4(GammaCM, USquared, EN, MN):
    # we divide gammaCM by U^2 since the total width has been computed with madgraph for U = 10^-3
    GammaCM_rescaled = GammaCM * USquared / ((1e-3)**2)     # in GeV
    tauCM = 1 / GammaCM_rescaled                            # in GeV^-1
    LengthCM = InvGeVtoMeters * tauCM                       # in m
    pN = np.sqrt(EN**2 - MN**2)                                # in GeV 
    LengthLab = pN / MN * LengthCM                          # in m
    return LengthLab

# %%
box_min = np.array([XMIN, YMIN, -l/2])
box_max = np.array([XMAX, YMAX, l/2])
# Box transform
dispB = np.array([0, 0, z_DUNE_LArTPC2])

# %%
# mixing = 'Um42'
# channel = 'to_numumu'
# mode = 'neutrino'
# parent = 'NM_kaon+'

if mode == 'neutrino':
    N = 'n'
else:
    if mode == 'antineutrino':
        N = 'n~'

channel_dic = {'to_numumu': '3body', 'to_mupi': '2body'}

if channel_dic[channel] == '3body':
    paso = 5
else:
    if channel_dic[channel] == '2body':
        paso = 4

#mass = 310
for mass in um42_grid:

    file = '/Users/ific/Desktop/HNL_DUNE_ND_signal_bkg/signal/%s/%s/events-%sdecay-%sMeV-1e-06-from%s.dat'%(mixing, channel, N, mass, parent)
    zipped_file = file + '.gz'
    if os.path.exists(zipped_file):
        run(['gunzip', zipped_file])
    #else:
        #print(f"File {zipped_file} does not exist.")
    if os.path.exists(file):
        df = pd.read_csv(file, delim_whitespace=True, names=nombres, skiprows=2)
        df_final = pd.DataFrame(columns=nombres+['N_weight_final','E', 'dist_to_enter', 'dist_in_det', 'x_final', 'y_final', 'z_final'])
        print(f'Processing file with {mass} MeV HNL')
        for i in range(0,len(df),paso):
            evento = i

            x_0_test = df['x_0'].iloc[evento]
            y_0_test = df['y_0'].iloc[evento]
            z_0_test = df['z_0'].iloc[evento]
            thetaN_test = df['thetaN'].iloc[evento]
            phiN_test = df['phiN'].iloc[evento]  
            N_total_width_CM_test = df['N_total_width_CM'].iloc[evento]
            if channel_dic[channel] == '3body':
                EN_test = df['N_weight'].iloc[evento+1] + df['N_weight'].iloc[evento+2] + df['N_weight'].iloc[evento+3]
            else:
                if channel_dic[channel] == '2body':
                    EN_test = df['N_weight'].iloc[evento+1] + df['N_weight'].iloc[evento+2]
            MN_test = df['MN'].iloc[evento]
            L_lab_N_test = df['L_lab_N'].iloc[evento]

            p0 = np.array([x_0_test, y_0_test, z_0_test])
                
            hnl_direct = HNL_direction(thetaN_test, phiN_test)
            result = detector_intersection_and_distance(p0, hnl_direct, box_min, box_max, dispB, alpha)
            if result is not None:
                p_enter, p_exit, p_enter_prime, p_exit_prime, dist_to_enter, dist_exit = result
                pf, dist_in_det = random_decay_point_and_distance(p_enter, p_exit)
                #L_lab_N = LengthLab_N4(N_total_width_CM_test, 1e-6, EN_test, MN_test)
                N_weight = compute_weight(L_lab_N_test, dist_to_enter, dist_in_det)
                #print(f"Event {evento}:")
                #print(f"  Entry point (global XYZ): {p_enter}")
                #print(f"  Exit point (global XYZ): {p_exit}")
                #print(f"  Entry point (local X'Y'Z'): {p_enter_prime}")
                #print(f"  Exit point (local X'Y'Z'): {p_exit_prime}")
                #print(f"  Distance from origin to entry: {dist_to_enter:.2f} m")
                #print(f"  Distance from origin to exit: {dist_exit:.2f} m")
                #print(f"  Random decay point in detector: {pf}")
                #print(f"  Distance traveled in detector: {dist_in_det:.2f} m")
                #print(f"  Computed weight for this event: {N_weight:.3e}")
                # Append to df_final
                row = df.iloc[evento].copy()
                row['N_weight_final'] = N_weight
                row['E'] = EN_test
                row['dist_to_enter'] = dist_to_enter
                row['dist_in_det'] = dist_in_det
                row['x_final'] = pf[0]
                row['y_final'] = pf[1]
                row['z_final'] = pf[2]
                df_final = pd.concat([df_final, row.to_frame().T], ignore_index=True)
                if channel_dic[channel] == '3body':
                    df_final = pd.concat([df_final, df.iloc[evento+1:evento+4]], ignore_index=True)  # Add the next 3 rows for the same event
                else:
                    if channel_dic[channel] == '2body':
                        df_final = pd.concat([df_final, df.iloc[evento+1:evento+3]], ignore_index=True)  # Add the next 2 rows for the same event
            #else:
                #print(f"Event {evento}: No intersection with the detector.")
        out_file = '/Users/ific/Desktop/HNL_DUNE_ND_signal_bkg/signal/%s/%s/filter/events-%sdecay-%sMeV-1e-06-from%s.dat'%(mixing, channel, N, mass, parent)
        df_final.to_csv(out_file, index=False, sep='\t')


