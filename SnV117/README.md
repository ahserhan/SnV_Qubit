# SnV117 QuTiP Model

This repository contains a QuTiP-based implementation of the Hamiltonian model for SnV117 centers in diamond, including magneto-optical spectrum simulations.

## Files

- `hamiltonian_qutip.py`: QuTiP-based Hamiltonian implementation for SnV117 including:
  - Spin-Orbit Coupling (SOC)
  - Iso-Orbital Coupling (IOC)
  - Jahn-Teller strain
  - Magnetic field effects on electron, nucleus, and orbital degrees of freedom
  - Hyperfine coupling (spin 1/2 × spin 1/2)

- `parameters_qutip.py`: Physical parameters specific to SnV117 centers

- `SnV117_spectrum_qutip.ipynb`: Jupyter notebook demonstrating PLE spectrum calculations with magnetic field sweeps

## Requirements

- QuTiP
- NumPy
- Matplotlib

## Usage

Run the Jupyter notebook to see PLE spectrum simulations with various magnetic field orientations.
