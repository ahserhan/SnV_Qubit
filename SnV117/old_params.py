"""
Parameters for SnV117 center in diamond (for use with QuTiP Hamiltonian).

All parameters are for the 117SnV- defect center.
"""

import numpy as np


# Spin values
S = 1/2  # Electron spin
Sn = 1/2  # Nuclear spin (117Sn)

# Orbital magnetic field susceptibility
q = 0.171  # [] ground state orbital magnetic field susceptibility
q_exc = 0.073  # [] excited state orbital magnetic field susceptibility

# Spin-orbit coupling
L = 830.0  # [GHz] spin-orbit coupling ground state
L_exc = 3000.0  # [GHz] spin-orbit coupling excited state

# Strain susceptibilities
# From DFT (Guo et al. arxiv:2307.11916 (2023) https://arxiv.org/abs/2307.11916)
d = 0.787e6  # [GHz/strain]
d_err = 0.1e6  # [GHz/strain]
f = -0.562e6  # [GHz/strain]
f_err = 0.1e6  # [GHz/strain]

# Strain susceptibility matrix [representation, m, n]
# Two representations (egx and egy)
P = np.array([
    [[ d,     0,   f/2],
     [ 0,    -d,     0],
     [f/2,    0,     0]],
    [[ 0,    -d,     0],
     [-d,     0,   f/2],
     [ 0,   f/2,     0]]
])

# Coherence coefficient
chi = 6.15725529e-30  # [s^2] coherence phonon cross-section parameter
chi_err = np.array([
    4.75762948e-30,  # lower bound
    6.15725529e-30,  # nominal
    7.88529044e-30   # upper bound
])

# Hyperfine Properties
#=====================
# Ratio of electron to proton mass
rmep = 5.44617021e-4

# Ratio of nuclear/electron gyromagnetic ratio (assuming g~2 for electrons)
rg = -2.00208 * rmep / 2

# Ground state hyperfine parameters (from DFT)
Aiso_gnd = 661.9  # [MHz] isotropic hyperfine coupling
Add_gnd = -5.53   # [MHz] anisotropic hyperfine coupling

# Asher: I noted a typo in the original code. The Add_gnd should be divided by 2.
Apar_gnd = (Aiso_gnd + Add_gnd) / 1000.0  # [GHz] parallel hyperfine coupling
Aperp_gnd = (Aiso_gnd - Add_gnd / 2) / 1000.0  # [GHz] perpendicular hyperfine coupling

# Excited state hyperfine parameters (from DFT) [GHz]
Apar_exc = 672.39 / 1000.0   # [GHz] parallel hyperfine coupling
Aperp_exc = 295.82 / 1000.0  # [GHz] perpendicular hyperfine coupling

# Diamond Properties
#===================
# Diamond stiffness parameters (GPa) at 10 K 
# (Migliori et al. J. Appl. Phys. (2008); https://doi.org/10.1063/1.2975190)
a = 1079.26  # C_11 = C_22 = C_33
b = 126.73   # C_12 = C_13 = C_23
c = 578.16   # C_44 = C_55 = C_66; all others are zero

# Build stiffness tensor C_diamond[i,j,k,l] in bulk coordinate system X=100, Y=010, Z=001
C_diamond = np.zeros((3, 3, 3, 3))
C_diamond[0, 0, 0, 0] = a
C_diamond[1, 1, 1, 1] = a
C_diamond[2, 2, 2, 2] = a
C_diamond[0, 0, 1, 1] = b
C_diamond[0, 0, 2, 2] = b
C_diamond[1, 1, 2, 2] = b
C_diamond[1, 2, 1, 2] = c / 2
C_diamond[2, 0, 2, 0] = c / 2
C_diamond[0, 1, 0, 1] = c / 2

# Enforce index symmetry
C_diamond = [[[[max(C_diamond[i, j, k, l], C_diamond[i, j, l, k], 
                      C_diamond[j, i, k, l], C_diamond[j, i, l, k],
                      C_diamond[k, l, i, j], C_diamond[l, k, i, j], 
                      C_diamond[k, l, j, i], C_diamond[l, k, j, i])
                for i in range(3)] for j in range(3)] 
               for k in range(3)] for l in range(3)]
C_diamond = np.array(C_diamond)  # [GPa] Stiffness tensor

# Diamond density
rho_diamond = 3.501  # [g/cm^3]

# Electron Parameters
#====================
be = 9.2740100783e-24  # [J/T] Bohr magneton
ge = 2.0023  # electron g factor
h = 6.626e-34  # [J*s] Planck constant

# Unit Conversions
#=================
bohr_to_angstrom = 0.529177  # [bohr/angstrom]
T_to_GHz = ge * be / h / 1e9  # [GHz/T] Conversion from T to GHz
h_GHz = 6.626e-25  # [J/GHz] Planck constant

