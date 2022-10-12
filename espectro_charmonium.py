import timeit
import matplotlib.pyplot as pl
import numpy as np
from scipy import integrate
    
start = timeit.default_timer()

##############################################################################

# CONSTANTES
m_c = 1.67 # masa charm
    
##############################################################################
    
# PARÁMETROS
h = .01 # paso de integración
H = 20 # altura de corte de las soluciones
l = 0 # momento angular
    
##############################################################################
    
#  FUNCIONES AUXILIARES    
def f(w, r, lamb, g_in, m_in):
    a_0 = 2 / (m_c * g_in**2)
    u = w[0]
    v = w[1]
    fu = v
    fv = ((l * (l+1) / (r**2)) * u - (2 / (3*np.pi*r)) * 
          np.exp(-m_in*a_0*r) * u + lamb**2 *u)
    return np.array([fu,fv], float)

def formato_contour(levels, data, title):
    pl.rc('text', usetex=True)
    pl.rc('font', family='serif')
    pl.contourf(G, M, data, 20, levels=levels, cmap='viridis')
    pl.title(title,fontsize=16);
    pl.xlabel('constante de acoplamiento, $g$', fontsize=16)
    pl.ylabel('masa, $m$(GeV)', fontsize=16)
    pl.tight_layout()
    cb = pl.colorbar()
    cb.ax.tick_params(labelsize=14)
    pl.xticks(fontsize=16); pl.yticks(fontsize=16)
    pl.minorticks_on()
    pl.tick_params(which='major', length=10)
    pl.tick_params(which='minor', length=5)
    pl.locator_params(axis='y', nbins=6); pl.locator_params(axis='x', nbins=5)

##############################################################################

# ESPECTRO SIN CORRECCIONES

# función que resuelve la ec. radial a una energía dada (RK4)
def solve(energy, g_in, m_in):
    upoints = []
    vpoints = []
    rpoints = []
    r = 1e-3
    w = np.array([0,1.0], float)
    upoints.append(w[0])
    vpoints.append(w[1])
    rpoints.append(r)
    while abs(upoints[-1]) <= H:
        k1 = h * f(w, r, energy, g_in, m_in)
        k2 = h * f(w+0.5*k1, r+0.5*h, energy, g_in, m_in)
        k3 = h * f(w+0.5*k2, r+0.5*h, energy, g_in, m_in)
        k4 = h * f(w+k3, r+h, energy, g_in, m_in)
        w += (k1 + 2*k2 + 2*k3 + k4) / 6
        upoints.append(w[0])
        vpoints.append(w[1])
        r += h
        rpoints.append(r)
    return rpoints,upoints
    
# barrido de g y m
G = np.arange(2., 4., .1)
M = np.arange(.2, .5, .01)

data_sc = np.zeros([len(M), len(G)])
data_c = np.zeros([len(M), len(G)])
data_spl = np.zeros([len(M), len(G)])
data_err = np.zeros([len(M), len(G)])
data_vel = np.zeros([len(M), len(G)])
for i_m,m in enumerate(M):
    for i_g,g in enumerate(G):
        print(m,g)
        E_I = m_c * g**4 / 4
        lamb_init = .25 # busco en el rango 0-.25
        vlamb = np.arange(lamb_init, .01, -.01)
        
        # veo dónde hay cambio de dirección en la divergencia
        U_end = [solve(lamb,g,m)[1][-1] for lamb in vlamb] # último valor de U
        limits = [(1 if u > 0 else 0) for u in U_end]
        
        for i in range(len(limits)-1):
            if limits[i+1] != limits[i]:
                upper = vlamb[i]
                lower = vlamb[i+1]
                change = limits[i+1] - limits[i]
                break
        else:
            print('No hay estados ligados')
            data_c[i_m, i_g] = 100
            data_sc[i_m, i_g] = 100
            data_spl[i_m, i_g] = 100
            data_err[i_m, i_g] = 100
            data_vel[i_m, i_g] = 100
            continue # vamos al siguiente (m,g)
                
        # cálculo de energías y estados propios
        while True:
            lamb = .5 * (upper+lower)
            P,U = solve(lamb, g, m)
            cut = 0
            convergence_count = 0
            for k in range(len(U)):
                if abs(U[k]) < 1e-2:
                    convergence_count += 1
                else:
                    convergence_count = 0
                if convergence_count == 50:
                    cut = k
                    break # convergió
            else: # no convergió
                if U[-1] > 0 and change == -1:
                    upper = lamb
                elif U[-1] < 0 and change == 1:
                    upper = lamb
                else:
                    lower = lamb
                continue # siguiente iteración del while
            break # la función converge y salgo del loop
                
        Eb = -lamb**2 * E_I # energía de ligadura
        E = Eb + 2*m_c
        print('Energía sin correcciones:', E)
        print('Energía ligadura:', -lamb**2 * E_I)
                     
        # normalizo la función de onda

        a0 = 2 / (m_c * g**2)
        R_cut = [x / (y * a0) for x, y in zip(U[1:cut], P[1:cut])]
        r_coord = P[1:cut]
        norm = integrate.simps(
            [a0**3 * x**2 * y**2 for x,y in zip(r_coord,R_cut)], r_coord)
        func_onda = [x / np.sqrt(norm) for x in R_cut]
            
##############################################################################
        # CORRECCIONES AL ESPECTRO
        
        # Función de onda en el origen
        func_onda_0_sq = m_c*g**2*a0/(12*np.pi**2) \
        * integrate.simps(
            [y**2*np.exp(-m*a0*x)*(m*a0*x + 1) 
                 for x, y in zip(r_coord,func_onda)],
            r_coord)
        
        # Término cinético p^4
        c1 = -1/(4*m_c)
        mvalue1 = c1 * (Eb**2 + (2 * g**2 * Eb * a0**2) / (3*np.pi)
            * integrate.simps(
                [x*y**2*np.exp(-m*a0*x) for x, y in zip(r_coord,func_onda)],
                r_coord)
            + (g**4 * a0) / (9*np.pi)
            * integrate.simps(
                [y**2*np.exp(-2*m*a0*x) for x,y in zip(r_coord,func_onda)],
                r_coord)
            )
        Ec = E + mvalue1
        print('corrección 1:', mvalue1)
            
        # Término m^2 f(r)
        c2 = -a0**2 * g**2 * m**2/(12 * np.pi * m_c**2)
        mvalue2 = c2 * integrate.simps(
            [x*y**2*np.exp(-m*a0*x) for x, y in zip(r_coord,func_onda)],
            r_coord)
        Ec += mvalue2
        print('corrección 2:', mvalue2)

        # Término Darwin
        c3 = g**2 / (3 * m_c**2)
        mvalue3 = c3 * func_onda_0_sq
        Ec += mvalue3
        print('corrección 3:', mvalue3)

        # Término spin-orbit
        J = 0
        if l == 1:
            c4 = -g**2/(4 * np.pi * m_c**2)
        else: 
            c4 = 0
        mvalue4 = c4 * (J*(J+1) - 4) * integrate.simps(
            [y**2 / x * (m*a0*x + 1) * np.exp(-m * a0 * x) 
                for x, y in zip(r_coord,func_onda)],
            r_coord)
        Ec += mvalue4
        print('corrección 4:', mvalue4)

        # Término momentos
        c5 = 4 * g**2 / (3 * np.pi * m_c**2 * m**2)
        d_func_onda = np.gradient(func_onda, r_coord)
        dd_func_onda = np.gradient(d_func_onda, r_coord)
        mvalue5 = c5 * (m_c * Eb * integrate.simps(
            [y**2 / x * (1 - np.exp(-m*a0*x)*(m*a0*x + 1)) 
                for x, y in zip(r_coord,func_onda)],
            r_coord)
            + g**2/(3*np.pi*a0) * integrate.simps(
            [y**2 * np.exp(-m*a0*x) / x**2 *
                (1 - np.exp(-m*a0*x) * (m*a0*x + 1)) 
                for x, y in zip(r_coord,func_onda)],
            r_coord)
            - 3/ a0**2 * integrate.simps(
            [y/x*ddy*(1 - np.exp(-m*a0*x)*(m**2*a0**2*x**2/3 + m*a0*x + 1)) 
                for x, y, ddy in zip(r_coord,func_onda,dd_func_onda)],
            r_coord))
        Ec += mvalue5
        print('corrección 5:', mvalue5)

        # Término spin-spin (A)
        S = 0
        c6 = -a0**2 * g**2 * m**2/(6 * np.pi * m_c**2)
        mvalue6 = c6 * (S*(S+1) - 3/2) * integrate.simps(
            [x * y**2 * np.exp(-m*a0*x) for x, y in zip(r_coord,func_onda)],
            r_coord)
        Ec += mvalue6            
        print('corrección 6:', mvalue6)
        # Término spin-spin delta
        if l == 0:
            c7 = 4 * g**2/(9 * m_c**2)
        else:
            c7 = 0
        mvalue7 = c7 * (S*(S+1) - 3/2) * func_onda_0_sq
        Ec += mvalue7
        print('corrección 7:', mvalue7)
        #Término spin-spin (B)
        J = 0
        c8 = - g**4 / (12 * np.pi * m_c**2)
        mvalue8 = c8 * ( 2*(S*(S+1) - 3/2) * integrate.simps(
            [y**2 / x * np.exp(-m*a0*x) * (m*a0*x + 1)
                for x, y in zip(r_coord,func_onda)],
            r_coord)
            - (4 /((2*l + 3)*(2*l-1)) * (S*(S+1)*l*(l+1) 
            - 3/4 * (J*(J+1) - S*(S+1) - l*(l+1)) - 3/4*(J*(J+1) - S*(S+1) 
            - l*(l+1))**2) + 2*(S*(S+1) - 3/2)) *integrate.simps(
                [y**2 / x * np.exp(-m*a0*x) * (m**2*a0**2*x**2/3 + m*a0*x + 1)
                    for x, y in zip(r_coord,func_onda)],
                r_coord)
            )
        Ec += mvalue8
        print('corrección 8:', mvalue8)
            
        print('Energía con correcciones:', Ec)
                
        # Splitting    
        D = - 4 * g**2 / 3 * 4 * (2*m**2*a0**2 /(48 * m_c**2 * np.pi) 
            * integrate.simps(
                [x*y**2*np.exp(-m*a0*x) for x, y in zip(r_coord,func_onda)],
                r_coord) 
            - 1/(6*m_c**2) * func_onda_0_sq)
        print('Splitting:', D)
                
        # Error cuadrático        
        Err = np.sqrt((Ec-2.98)**2 + (D-.157)**2)
        print('Error:', Err)
        
        # Velocidad
        v = 4  / m_c * (Eb + g**2*a0**2/(3*np.pi) 
            * integrate.simps(
                [x*y**2*np.exp(-m*a0*x) for x, y in zip(r_coord,func_onda)],
                r_coord)
            )
        print('Velocidad:', v)
           
        # guardo los resultados
        data_c[i_m, i_g] = Ec
        data_sc[i_m, i_g] = E
        data_spl[i_m, i_g] = D
        data_err[i_m, i_g] = Err
        data_vel[i_m, i_g] = v
    
# Gráfica con correcciones
pl.figure(2)
formato_contour(
    levels=[0, .05, .1, .15, .2, .25, .3, .35, .4, .45, .5],
    data=abs(data_c - 2.98),
    title='$|M_\mathrm{num}-M_\mathrm{exp}|$ con correcciones')
pl.savefig('l0_con_correcciones.png',dpi=300, bbox_inches = 'tight')
        
# Gráfica sin correcciones
pl.figure(3)
formato_contour(
    levels=[0, .05, .1, .15, .2, .25, .3, .35, .4, .45, .5],
    data=abs(data_sc - 2.98),
    title='$|M_\mathrm{num}-M_\mathrm{exp}|$ sin correcciones')
pl.savefig('l0_sin_correcciones.png',dpi=300, bbox_inches = 'tight')
            
# Gráfica splitting
pl.figure(4)
formato_contour(
    levels=[0, .05, .1, .15, .2, .25, .3, .35, .4, .45, .5],
    data=abs(data_spl - .157),
    title='Splitting (valor de referencia 0.157)')
pl.savefig('l0_splitting.png',dpi=300, bbox_inches = 'tight')

# Gráfica error
pl.figure(5)
formato_contour(
    levels=[0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.],
    data=data_err,
    title='Error')
pl.savefig('l0_error.png',dpi=300, bbox_inches = 'tight')

# Gráfica velocidad
pl.figure(6)
formato_contour(
    levels=[0, .1, .2, .3, .4, .5, .6, .7, .8],
    data=data_vel,
    title=r'Velocidad $\langle p^2/\mu^2 \rangle$')
pl.savefig('l0_velocidad.png',dpi=300, bbox_inches = 'tight')

##############################################################################
    
stop = timeit.default_timer()

print('Time:', stop - start)