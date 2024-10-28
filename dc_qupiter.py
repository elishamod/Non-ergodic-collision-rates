# -*- coding: utf-8 -*-
"""
Plot the critical diameter as function of Qupiter's mass and SME.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, cm, colors

e = 0.03    # planetesimals' orbit eccentricity
r = 1.0    # planetesimals' orbit radius in au
m_star = 0.1    # central star mass in sun masses
d_earth = 2 * 6400    # earth diameter in km

min_dc, max_dc = 3.0, 4.8
#*qlims) color_levels = 10**np.linspace(min_dc, max_dc, 1 + 6 * (max_dc - min_dc))
color_levels = 10**np.linspace(min_dc, max_dc, 19)

nm, na = 300, 400
mlims = [-5, -2]
alims = [-2, 2]
m_eps = 0.02
# m = M_BH / (1e6 * M_sun)
m_vec = 10.0 ** np.linspace(mlims[0] - m_eps, mlims[1] + m_eps, nm)
a_vec = 10.0 ** np.linspace(alims[0], alims[1], na)
m, a = np.meshgrid(m_vec, a_vec)

e_fac = np.sqrt(e)  # the factor eccentriciry gives to dc


def fmt(x, pos=0):
    aa, bb = '{:.1e}'.format(x).split('e')
    bb = int(bb)
    if aa == '1.0':
        return r'$10^{{{}}}$'.format(bb)
    else:
        return r'${} \times 10^{{{}}}$'.format(aa, bb)


def fmt2(x, pos=0):
    if x == 1.0:
        return ''
    if float(int(x)) == x:
        return str(int(x))
    return str(x)
    

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rc('axes', labelsize=16)
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.rc('legend', fontsize=14)
plt.rc('axes', titlesize=18)


fig = plt.figure()
# critical d from Qupiter's pull
dc = 1.4e8 * e_fac * np.sqrt(m) * np.minimum(a, a**(-1.5) * r**2.5)
# critical d from GR
dc = np.maximum(dc, 2e4 * e_fac * np.sqrt(r * m_star))
# stop from exceeding limits
dc = np.maximum(dc, 10**min_dc)
dc = np.minimum(dc, 10**max_dc)

# plot the colormap of D_c
cs = plt.contourf(m, a, dc, locator=ticker.LogLocator(), cmap=cm.PuBu_r, \
                norm=colors.LogNorm(vmin=dc.min(), vmax=dc.max()), \
                levels=color_levels)
cbar = fig.colorbar(cs, format=ticker.FuncFormatter(fmt))

# find and plot the line of D_c = d_earth
m_earth = m_vec[np.argmin(np.abs(dc - d_earth), axis=1)]
plt.plot(m_earth, a_vec, color='red', label='$D_c=2R_\oplus$')

# stylize the figure
plt.grid(True)
plt.legend()
plt.xlim(10 ** mlims[0], 10 ** mlims[1])
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$M_Q/M_\star$')
plt.ylabel(r'$a_Q$ [au]')
plt.title('$D_c$ [km]: e=' + str(e) + 
          ', r=' + fmt2(r) + 'au, $M_\star$=' + fmt2(m_star) + '$M_\odot$')
fig.set_size_inches(8, 6)
plt.tight_layout()
plt.show()

