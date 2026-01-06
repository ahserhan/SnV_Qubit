"""
Hamiltonian model for SnV117 center in diamond using QuTiP.

This module provides the Hamiltonian for SnV117 (spin 1/2 electron, spin 1/2 nuclear)
with the following interactions:
- SOC (Spin-Orbit Coupling)
- IOC (Iso-Orbital Coupling)
- Strain (Jahn-Teller)
- Magnetic field on electron
- Magnetic field on nucleus
- Magnetic field on orbital degree of freedom
- Hyperfine coupling (spin 1/2 * spin 1/2)
"""

import numpy as np
import qutip as qt


def create_hamiltonian_nuclear(Sn=1/2):
    """
    Create a Hamiltonian for SnV117 center in diamond using QuTiP.
    
    SnV117 has:
    - Electron spin S = 1/2
    - Nuclear spin Sn = 1/2 (default)
    - Orbital degree of freedom (2D)
    
    Tensor product structure: orbital ⊗ electron ⊗ nuclear
    
    Parameters:
        Sn: Nuclear spin (default 1/2 for SnV117)
    
    Returns:
        tuple: (H, Href, p, J2)
            - H: Total Hamiltonian function
            - Href: Reference Hamiltonian (SOC + strain only)
            - p: Dipole moment operators [px, py, pz]
            - J2: Total angular momentum squared operator
    """
    # Electron spin (S = 1/2)
    S = 1/2
    X = qt.jmat(S, 'x')
    Y = qt.jmat(S, 'y')
    Z = qt.jmat(S, 'z')
    I = qt.qeye(int(2*S + 1))  # Identity matrix (2x2, used for both orbital and electron spaces)
    
    # Nuclear spin operators
    Xn = qt.jmat(Sn, 'x')
    Yn = qt.jmat(Sn, 'y')
    Zn = qt.jmat(Sn, 'z')
    In = qt.qeye(int(2*Sn + 1))
    
    # Total angular momentum squared operator
    # J^2 = (S_e*(S_e+1) + S_n*(S_n+1))*I_orb ⊗ I_e ⊗ I_n + 2*I_orb ⊗ (S_e ⊗ S_n)
    # Structure: orbital ⊗ electron ⊗ nuclear
    J2 = ((S*(S + 1) + Sn*(Sn + 1)) * qt.tensor(I, I, In) + 
          2 * (qt.tensor(I, X, Xn) + qt.tensor(I, Y, Yn) + qt.tensor(I, Z, Zn)))
    
    # Magnetic field on electron
    # Structure: orbital ⊗ electron ⊗ nuclear
    Hbxe = lambda bx: bx * qt.tensor(I, X, In)
    Hbye = lambda by: by * qt.tensor(I, Y, In)
    Hbze = lambda bz: bz * qt.tensor(I, Z, In)
    
    # Magnetic field on nucleus (rg = ratio of nuclear/electron gyromagnetic ratios)
    Hbxn = lambda bx, rg: rg * bx * qt.tensor(I, I, Xn)
    Hbyn = lambda by, rg: rg * by * qt.tensor(I, I, Yn)
    Hbzn = lambda bz, rg: rg * bz * qt.tensor(I, I, Zn)
    
    # Magnetic field on orbital degree of freedom
    Hbzo = lambda bz, q: q * bz * qt.tensor(Z, I, In)
    
    # Hyperfine coupling
    Hhf = lambda Aperp, Apar: (Aperp * qt.tensor(I, X, Xn) + 
                                Aperp * qt.tensor(I, Y, Yn) + 
                                Apar * qt.tensor(I, Z, Zn))
    
    # SOC (Spin-Orbit Coupling)
    # Factor of 2 since each Z has a factor of 1/2
    Hsoc = lambda L: 2 * L * qt.tensor(Z, Z, In)
    
    # IOC (Iso-Orbital Coupling, also called upsilon)
    # Factor of 2 since each Z has a factor of 1/2
    Hioc = lambda u: 2 * u * qt.tensor(Z, I, Zn)
    
    # Strain/Jahn-Teller terms
    Hegx = lambda alpha: -2 * alpha * qt.tensor(X, I, In)
    Hegy = lambda beta: -2 * beta * qt.tensor(Y, I, In)
    
    # Dipole moment operators [px, py, pz]
    # Transformation matrix from egx/egy basis to eg+/eg- basis
    T = np.array([[-1, -1j], [1, -1j]]).T / np.sqrt(2)
    
    # Operators in egx/egy basis (acting on orbital space, using spin-1/2 operators as 2x2 matrices)
    p_egx_egy = [
        2 * Z,   # Z => -X
        -2 * X,  # X => -Y
        I        # I => I
    ]
    
    # Transform to eg+/eg- basis
    # Structure matches: np.einsum('ji,jk,kl->il', T.conj(), x, T)
    p_eg = []
    for op in p_egx_egy:
        op_array = op.full()  # Convert Qobj to numpy array
        # T.conj().T @ op_array @ T (equivalent to einsum('ji,jk,kl->il', T.conj(), op_array, T))
        op_transformed = T.conj().T @ op_array @ T
        p_eg.append(qt.Qobj(op_transformed))
    
    # Tensor with electron and nuclear spaces
    # Structure: orbital ⊗ electron ⊗ nuclear
    p = [qt.tensor(p_op, I, In) for p_op in p_eg]
    p = np.array(p)  # Shape: (3, dim, dim) for [px, py, pz]
    
    # Reference Hamiltonian (SOC + strain only)
    Href = lambda L, alpha, beta: Hsoc(L) + Hegx(alpha) + Hegy(beta)
    
    # Total Hamiltonian
    def H(bx, by, bz, rg, q, Aperp, Apar, L, alpha, beta, upsilon=0):
        """
        Total Hamiltonian for SnV117.
        
        Parameters:
            bx, by, bz: Magnetic field components (GHz)
            rg: Ratio of nuclear/electron gyromagnetic ratios
            q: Orbital magnetic field susceptibility
            Aperp: Perpendicular hyperfine coupling (GHz)
            Apar: Parallel hyperfine coupling (GHz)
            L: Spin-orbit coupling strength (GHz)
            alpha: Strain parameter in x direction (GHz)
            beta: Strain parameter in y direction (GHz)
            upsilon: Iso-orbital coupling strength (GHz), default=0
            
        Returns:
            Qobj: Hamiltonian operator
        """
        return (Href(L, alpha, beta) + 
                (Hbxe(bx) + Hbxn(bx, rg)) + 
                (Hbye(by) + Hbyn(by, rg)) + 
                (Hbze(bz) + Hbzn(bz, rg) + Hbzo(bz, q)) + 
                Hhf(Aperp, Apar) + 
                Hioc(upsilon))
    
    return H, Href, p, J2


def PLE_transitions(Sn, B, theta, phi, rg, q, Aperp, Apar, L, alpha, beta, q_exc, Aperp_exc, Apar_exc, L_exc, alpha_exc, beta_exc, eta):
    """
    Calculate PLE transition intensities for a sweep of B-field strength using QuTiP.
    
    Parameters match the numpy version for compatibility.
    """
    H, Href, p, J2 = create_hamiltonian_nuclear(Sn)
    Ns = int(2*Sn + 1)
    
    # Convert B to array and expand for broadcasting (matching original code)
    B = np.array(B)
    if B.ndim == 0:
        B = B.reshape(1)
    B = np.einsum('k,klm->klm', B, np.ones((B.size, 4*Ns, 4*Ns)))
    bz = B * np.cos(theta)
    bx = B * np.sin(theta) * np.cos(phi)
    by = B * np.sin(theta) * np.cos(phi)  # Matching original code (line 298)
    
    # Solve ground-state Hamiltonian
    # Convert Qobj to numpy arrays for eigenvalue calculations
    Hplot_list = []
    for i in range(B.shape[0]):
        H_qobj = H(bx[i, 0, 0], by[i, 0, 0], bz[i, 0, 0], rg, q, Aperp, Apar, L, alpha, beta)
        Hplot_list.append(H_qobj.full())
    Hplot = np.array(Hplot_list)
    
    Hplot_ref_qobj = Href(L, alpha, 0)
    Hplot_ref = Hplot_ref_qobj.full()
    
    # Calculate eigenvalues and eigenvectors
    E = np.array([np.linalg.eigh(Hplot[i])[0] for i in range(B.shape[0])])
    U = np.array([np.linalg.eigh(Hplot[i])[1] for i in range(B.shape[0])])
    Eref, Uref = np.linalg.eigh(Hplot_ref)
    
    # Convert J2 to numpy array
    J2_array = J2.full()
    
    # Calculate alignment
    alignment = np.einsum('...ji,jk,...kl->...il', U.conj(), J2_array, U)
    alignment = np.real(np.einsum('...ii->...i', alignment))
    
    # Solve excited-state Hamiltonian
    Hplot_exc_list = []
    for i in range(B.shape[0]):
        H_exc_qobj = H(bx[i, 0, 0], by[i, 0, 0], bz[i, 0, 0], rg, q_exc, Aperp_exc, Apar_exc, L_exc, alpha_exc, beta_exc)
        Hplot_exc_list.append(H_exc_qobj.full())
    Hplot_exc = np.array(Hplot_exc_list)
    
    Hplot_ref_exc_qobj = Href(L_exc, alpha_exc, 0)
    Hplot_ref_exc = Hplot_ref_exc_qobj.full()
    
    E_exc = np.array([np.linalg.eigh(Hplot_exc[i])[0] for i in range(B.shape[0])])
    U_exc = np.array([np.linalg.eigh(Hplot_exc[i])[1] for i in range(B.shape[0])])
    Eref_exc, _ = np.linalg.eigh(Hplot_ref_exc)
    
    # Calculate alignment for excited state
    alignment_exc = np.einsum('...ji,jk,...kl->...il', U_exc.conj(), J2_array, U_exc)
    alignment_exc = np.real(np.einsum('...ii->...i', alignment_exc))
    
    # Convert p to numpy arrays
    p_array = np.array([p[i].full() for i in range(3)])
    
    # Transition dipole moments
    transition = np.einsum('...ji,mjk,...kl->...mil', U_exc.conj(), p_array, U)
    transition = np.einsum('...ijk,i->...jk', np.abs(transition)**2, eta)
    
    return E, Eref, U, alignment, E_exc, Eref_exc, U_exc, alignment_exc, transition


def PLE_spectrum(Sn, intensity, lw, f_meas, B, theta, phi, rg, q, Aperp, Apar, L, alpha, beta, q_exc, Aperp_exc, Apar_exc, L_exc, alpha_exc, beta_exc, eta):
    """
    Calculate PLE spectrum as sum of Lorentzians with same linewidth, 
    intensity modulated by transition intensity using QuTiP.
    """
    peak = lambda f, f0, a, sigma: a * (sigma/2)**2 / ((f - f0)**2 + (sigma/2)**2)  # Lorentzian
    Ns = int(2*Sn + 1)
    
    E, Eref, _, _, E_exc, Eref_exc, _, _, transition = PLE_transitions(
        Sn, B, theta, phi, rg, q, Aperp, Apar, L, alpha, beta, 
        q_exc, Aperp_exc, Apar_exc, L_exc, alpha_exc, beta_exc, eta
    )
    
    # Calculate PLE spectrum
    # Handle case where B might be scalar
    B_size = E.shape[0] if E.ndim > 1 else 1
    PLE = [[[peak(f_meas, (E_exc[j, l] - Eref_exc[0]) - (E[j, k] - Eref[0]), 
                   transition[j, l, k], lw) 
             for l in range(2*Ns)] 
            for k in range(2*Ns)] 
           for j in range(B_size)]
    PLE = intensity * np.array(PLE).sum((1, 2))
    return PLE
