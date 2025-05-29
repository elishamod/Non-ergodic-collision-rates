# -*- coding: utf-8 -*-
"""
Plot the ratio of t_ref to t_dep in SMBH and IMBH scenarios.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, cm, colors
import argparse

alpha = 1.75
m_star = 1.0  # in Msun
m_B = 1.0  # in Msun
dfacs = [10 ** k for k in range(4)]
nm, nq = 300, 400
mlims = [-3, 1]
qlims = [-7, -2]
d_sun_pc = 4.5e-8  # in pc
plot_single = True
plot_binary = True

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--save', action='store_true', help='save the figures')
args = parser.parse_args()
save_flag = args.save


def accum_mass(q, m):
    '''
    Parameters:
        q : Orbit radius in pc.
        m : SMBH mass in 10^6 Msun.
        d : Collision diameter in dsun.
    Returns
        Accumulated mass in 10^6 Msun.
    '''
    return m * m ** (-0.5 * (3 - alpha)) * q ** (3 - alpha)


def t_ref_GR(q, m, d):
    '''
    Parameters:
        q : Orbit radius in pc.
        m : SMBH mass in 10^6 Msun.
        d : Collision diameter in dsun.
    Returns
        GR refresh time in terms of orbital period T.
    '''
    return 0.45 * d / m


def t_ref_mass(q, m, d):
    '''
    Parameters:
        q : Orbit radius in pc.
        m : SMBH mass in 10^6 Msun.
        d : Collision diameter in dsun.
    Returns
        Mass refresh time in terms of orbital period T.
    '''
    d_pc = d * d_sun_pc
    return (d_pc / q) * (m / accum_mass(q, m))


def t_ref(q, m, d):
    '''
    Parameters:
        q : Orbit radius in pc.
        m : SMBH mass in 10^6 Msun.
        d : Collision diameter in dsun.
    Returns
        Total refresh time in terms of orbital period T.
    '''
    return (t_ref_mass(q, m, d) ** -1 + t_ref_GR(q, m, d) ** -1) ** -1


def t_dep(q, m, d):
    '''
    Parameters:
        q : Orbit radius in pc.
        m : SMBH mass in 10^6 Msun.
        d : Collision diameter in dsun.
    Returns
        Depletion time in terms of orbital period T.
    '''
    d_pc = d * d_sun_pc
    return q / d_pc


def d_eff(q, m, d0):
    '''
    Parameters:
        q : Orbit radius in pc.
        m : SMBH mass in 10^6 Msun.
        d0 : Star diameter in dsun.
    Returns
        Effective collision diamter, after gravitational focusing, in dsun.
    '''
    f = 1e2 * m_star * m ** -1 * q * d0 ** -1
    return d0 * np.sqrt(1 + f)


def ratio_GR(q, m, d0, focusing=True):
    '''
    Parameters:
        q : Orbit radius in pc.
        m : SMBH mass in 10^6 Msun.
        d0 : Star diameter in dsun.
    Returns
        Ratio between GR refresh time and depletion time.
    '''
    if focusing:
        d = d_eff(q, m, d0)
    else:
        d = d0
    return t_ref_GR(q, m, d) / t_dep(q, m, d)


def ratio_mass(q, m, d0, focusing=True):
    '''
    Parameters:
        q : Orbit radius in pc.
        m : SMBH mass in 10^6 Msun.
        d0 : Star diameter in dsun.
    Returns
        Ratio between mass refresh time and depletion time.
    '''
    if focusing:
        d = d_eff(q, m, d0)
    else:
        d = d0
    return t_ref_mass(q, m, d) / t_dep(q, m, d)


def ratio(q, m, d0, focusing=True):
    '''
    Parameters:
        q : Orbit radius in pc.
        m : SMBH mass in 10^6 Msun.
        d0 : Star diameter in dsun.
    Returns
        Ratio between refresh time and depletion time.
    '''
    if focusing:
        d = d_eff(q, m, d0)
    else:
        d = d0
    return t_ref(q, m, d) / t_dep(q, m, d)


def tidal_disruption_radius(m, d0):
    '''
    Parameters:
        m : SMBH mass in 10^6 Msun.
        d0 : Star diameter in dsun.
    Returns
        Tidal disruption radius (in pc) - under which the star will break apart.
    '''
    R_star_pc = d0 * d_sun_pc / 2
    return 1e2 * R_star_pc * m ** (1 / 3) * m_star ** (-1 / 3)


def max_binary_separation(m_B, q, m):
    '''
    Parameters:
        m_B: binary mass in Msun.
        q: Orbit radius in pc.
        m: SMBH mass in 10^6 Msun.
    Returns
        Maximual separtion of a binary (in pc) due to SMBH's tidal disruption.
    '''
    return 1e-2 * q * (m_B / m) ** (1 / 3)


def find_ind_where(x, val, **kwargs):
    return np.argmin(abs(x - val), **kwargs)


m_vec = 10.0 ** np.linspace(mlims[0], mlims[1], nm)
q_vec = 10.0 ** np.linspace(qlims[0] - 0.1, qlims[1] + 0.1, nq)
m, q = np.meshgrid(m_vec, q_vec)
# R_BH = 9.57e-8 pc * m
R_BH = 9.57e-8 * m


def fmt(x, pos):
    a, b = '{:.1e}'.format(x).split('e')
    b = int(b)
    if a == '1.0':
        return r'$10^{{{}}}$'.format(b)
    else:
        return r'${} \times 10^{{{}}}$'.format(a, b)


plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rc('axes', labelsize=16)
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.rc('legend', fontsize=14)
plt.rc('axes', titlesize=18)


def plot_func(fig, ax, q, m, d0, ext_ratios=(-4, 4), d_for_TDE=-1, focusing=True, do_legend=True, binary=False):
    x_vec_fac = 1e6
    if d_for_TDE == -1:
        d_for_TDE = d0
    min_ratio, max_ratio = ext_ratios[0], ext_ratios[1]
    mass_rat = ratio_mass(q, m, d0, focusing=focusing)
    GR_rat = ratio_GR(q, m, d0, focusing=focusing)
    rat = ratio(q, m, d0, focusing=focusing)
    rat = np.maximum(np.minimum(rat, 10 ** max_ratio), 10 ** min_ratio)
    cs = ax.contourf(m * x_vec_fac, q, rat, locator=ticker.LogLocator(), cmap=cm.PuBu_r,
                     norm=colors.LogNorm(vmin=rat.min(), vmax=rat.max()),
                     levels=10 ** np.linspace(min_ratio, max_ratio, 1 + 3 * (max_ratio - min_ratio)))
    cbar = fig.colorbar(cs, format=ticker.FuncFormatter(fmt), ax=ax)
    ax.plot(m_vec * x_vec_fac, R_BH[0, :], 'k', linewidth=2, label='$R_{BH}$')
    q_TDE = tidal_disruption_radius(m_vec, d_for_TDE)
    ax.plot(m_vec * x_vec_fac, q_TDE, 'r', linewidth=2, label='$R_{TDE}$')
    qind_mass_rat = find_ind_where(mass_rat, 1, axis=0)
    qind_GR_rat = find_ind_where(GR_rat, 1, axis=0)
    ax.plot(m_vec * x_vec_fac, q_vec[qind_mass_rat], 'g', linewidth=2, label='$mass:t_{ref}=t_{dep}$')
    ax.plot(m_vec * x_vec_fac, q_vec[qind_GR_rat], 'y', linewidth=2, label='$GR:t_{ref}=t_{dep}$')
    if binary:
        q_top = q_vec[qind_mass_rat]
        q_bot = np.minimum(np.maximum(q_vec[qind_GR_rat], q_TDE), q_top)
    else:
        q_top = np.minimum(q_vec[qind_GR_rat], q_vec[qind_mass_rat])
        q_bot = np.minimum(q_TDE, q_top)
    ax.fill_between(m_vec * x_vec_fac, q_bot, q_top, alpha=0.25, color='y')
    ax.grid(True)
    ax.set_xscale('log')
    ax.set_yscale('log')
    # ax.set_xlabel(r'$M_\bullet/10^6M_\odot$')
    ax.set_xlabel(r'$M_\bullet/M_\odot$')
    ax.set_ylabel('r [pc]')
    if do_legend:
        ax.legend()


# Create the stellar collisions figure
if plot_single:
    fig, axs = plt.subplots(2, 2)
    for k in range(len(dfacs)):
        d0 = dfacs[k]
        ax = axs[k // 2, k % 2]
        plot_func(fig, ax, q, m, d0, ext_ratios=(-4, 4), do_legend=(k == 0))
        ax.set_title('$t_{ref}/t_{dep}$: D = ' + str(2 * d0) + r'R$_\odot$')
        ax.set_ylim(10 ** qlims[0], 10 ** qlims[1])
    fig.set_size_inches(12, 9)
    plt.tight_layout()
    if save_flag:
        fig.savefig('figs/t_ratio_SMBH_const_d.pdf', transparent=True)

# Create the binary disruptions figure
if plot_binary:
    d_max = max_binary_separation(m_B, q, m) / d_sun_pc
    fig, ax = plt.subplots(1, 1)
    plot_func(fig, ax, q, m, d_max, ext_ratios=(-2, 2), d_for_TDE=1.0, focusing=False, binary=True)
    ax.set_title('$t_{ref}/t_{dep}$: Maximal binary separation')
    ax.set_ylim(10 ** qlims[0], 10 ** qlims[1])
    fig.set_size_inches(8, 6)
    plt.tight_layout()
    if save_flag:
        fig.savefig('figs/t_ratio_SMBH_binary_dmax.pdf', transparent=True)

plt.show()
