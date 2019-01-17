# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""
A quantum-chemical study of CO dissociation mechanism on low-index Miller 
planes of ϴ-Fe3C 

Authors: Robin J.P. Broos, Bart Klumpers, Bart Zijlstra, 
         Ivo A.W. Filot, and Emiel J.M. Hensen

Catalysis Today

Notes:
    We recommend to use Spyder (see: https://www.spyder-ide.org/) to run
    these scripts. Please note that these scripts have been written
    for Python version 3.
"""

import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt

def main():
    """
    Calculate reaction rate as function of temperature and plot the results
    """
    
    # list of labels for the surfaces
    surfaces = ['100','010','110','011','001','101','0-11',
                '1-11','111','Co(11-21)']
    
    # list of kinetic parameters for each of the surfaces
    # note that the energies are given in kJ/mol, these
    # are later on converted to J/mol in the script
    eads_co_list = [180, 169, 204, 205, 185, 214, 155, 174, 167, 175]
    eads_h2_list = [110, 111, 130, 123, 85, 161, 88, 140, 169, 100]
    eact_hco_form_f_list = [60, 59, 243, 256, 100, 240, 72, 155, 125, 99]
    eact_hco_form_b_list = [13, 6, 143, 146, 50, 46, 26, 37, 42, 1]
    eact_hco_diss_f_list = [88, 71, 66, 98, 54, 46, 28, 1, 43, 61]
    
    # set the temperature and pressure
    pt = 1.0    # bar
    T = 500     # Kelvin
    
    # collect kinetic results
    rates = []
    for eads_co, eads_h2, eact_hco_form_f, eact_hco_form_b, eact_hco_diss in \
        zip(eads_co_list, eads_h2_list, eact_hco_form_f_list, 
            eact_hco_form_b_list, eact_hco_diss_f_list):
        # perform time integration
        t,y = solve_odes(T, eads_co*1e3, eads_h2*1e3, eact_hco_form_f*1e3, 
                         eact_hco_form_b*1e3, eact_hco_diss*1e3, pt)
        
        # uncomment to show time-integration
        # verify these plots to check that the steady-state solution
        # is reached (i.e. increase integration time if not)
        #plt.semilogx(t,y)
        #plt.show()
        
        # obtain final rate of CO dissociation step
        rate = calc_k_arr(T, 1e13,  eact_hco_diss*1e3) * y[-1,2]
        
        # append to rates array
        rates.append(rate)
        
    plt.bar(np.arange(0, len(rates)), rates)
    plt.yscale('log')
    plt.xticks(np.arange(0, len(rates)), surfaces, rotation='vertical')
    plt.xlabel('Surface')
    plt.ylabel('log(r)')
    plt.title('HCO mechanism')
    plt.show()
    
    for label,r in zip(surfaces, rates):
        print("%s: %12.6e" % (label,r))

def solve_odes(T, eads_co, eads_h2, eact_hco_form_f, 
               eact_hco_form_b, eact_hco_diss, pt):
    """
    Time-integrate chemo-kinetic system
    
    T                   - Temperature in K
    eads_co             - Adsorption energy of CO in J/mol
    eads_h2             - Adsorption energy of H2 in J/mol
    eact_hco_form_f     - Forward barrier for HCO formation
    eact_hco_form_b     - Reverse barrier for HCO formation
    eact_hco_diss       - Forward barrier for HCO dissociation
    pt                  - Total pressure in bar
    """
    # initial conditions
    y0 = [0, 0, 0, 1]
    t0 = 0
    t1 = 1e7                     # total integration time
    pa = 2.0/3.0 * pt * 1e5      # pressure of CO in Pa
    pb = 2.0/3.0 * pt * 1e5      # pressure of H2 in Pa

    # construct ODE solver
    r = ode(dydt).set_integrator('vode', method='bdf', 
           atol=1e-12, rtol=1e-12, nsteps=1000, with_jacobian=True)
    r.set_initial_value(y0, t0).set_f_params([T, pa, pb, eads_co, eads_h2, 
                       eact_hco_form_f, eact_hco_form_b, eact_hco_diss])
    
    # integrate on a logaritmic scale
    xx = np.linspace(-12.0, np.log10(t1), int((np.log10(t1) + 12.0) * 10))
    yy = []
    tt = []
    for x in xx:
        tnew = 10.0**x
        tt.append(tnew)
        yy.append(r.integrate(tnew))
        
    return tt, np.matrix(yy)

def dydt(t, y, params):
    """
    Set of ordinary differential equations
    """
    T =  params[0]
    pa = params[1]
    pb = params[2]
    eads_co = params[3]
    eads_h2 = params[4]
    eact_hco_form_f = params[5]
    eact_hco_form_b = params[6]
    eact_hco_diss = params[7]

    dydt = np.zeros(4)
    
    ma = 28 * 1.66054e-27
    mb =  2 * 1.66054e-27

    # calculate all reaction rate constants    
    k_ads_1 = calc_kads(T, pa, 1e-20, ma)
    k_des_1 = calc_kdes(T, 1e-20, ma, 1, 2.8, eads_co)
    k_ads_2 = calc_kads(T, pb, 1e-20, mb)
    k_des_2 = calc_kdes(T, 1e-20, mb, 2, 2.08, eads_h2)   
    kf_3 = calc_k_arr(T, 1e13, eact_hco_form_f)
    kb_3 = calc_k_arr(T, 1e13, eact_hco_form_b)
    kf_4 = calc_k_arr(T, 1e13, eact_hco_diss)

    # collect similar terms in new variables    
    r1f = k_ads_1 * y[3]
    r1b = k_des_1 * y[0]
    r2f = k_ads_2 * y[3]**2
    r2b = k_des_2 * y[1]**2   
    r3f = kf_3 * y[0] * y[1]
    r3b = kb_3 * y[2] * y[3]    
    r4f = kf_4 * y[2] * y[3]
    
    dydt[0] = r1f - r1b - r3f + r3b
    dydt[1] = 2.0 * r2f - 2.0 * r2b - r3f + r3b
    dydt[2] = r3f - r3b - r4f 
    dydt[3] = -r1f + r1b - 2.0 * r2f + 2.0 * r2b + r3f - r3b + r4f 
    
    return dydt

def calc_k_arr(T, nu, Eact):
    """
    Calculate reaction rate constant for a surface reaction
    
    T       - Temperature in K
    nu      - Pre-exponential factor in s^-1
    Eact    - Activation energy in J/mol
    """
    R = 8.3144598 # gas constant
    return nu * np.exp(-Eact / (R * T))

def calc_kads(T, P, A, m):
    """
    Reaction rate constant for adsorption
    
    T           - Temperature in K
    P           - Pressure in Pa
    A           - Surface area in m^2
    m           - Mass of reactant in kg
    """
    kb = 1.38064852E-23 # boltzmann constant
    return P*A / np.sqrt(2 * np.pi * m * kb * T)

def calc_kdes(T, A, m, sigma, theta_rot, Edes):
    """
    Reaction rate constant for desorption
    
    T           - Temperature in K
    A           - Surface area in m^2
    m           - Mass of reactant in kg
    sigma       - Symmetry number
    theta_rot   - Rotational temperature in K
    Edes        - Desorption energy in J/mol
    """
    kb = 1.38064852e-23 # boltzmann constant
    h = 6.62607004e-34  # planck constant
    R = 8.3144598       # gas constant
    return kb * T**3 / h**3 * A * (2 * np.pi * m * kb) / \
        (sigma * theta_rot) * np.exp(-Edes / (R*T))

if __name__ == '__main__':
    main()