import sys
import numpy as np
import pandas as pd
from operator import attrgetter
from numpy.random import RandomState
from scipy.spatial import distance_matrix

"""Funciones externas al ACS"""

def generar_solucion(n, prng):
    """ Retorna una representacion vectorial pseudoaleatorea de un problema.

    Parameters
    ----------
    n : int
        Largo del vector a obtener.
    prng : RandomState
        Generador de numeros pseudoaleatoreos.
    Returns
    -------
    list
        Un vector de cero a n desordenado.
    """

    solucion = list(range(n))
    prng.shuffle(solucion)
    return solucion


def calcular_costo(dm, solucion):
    """ Retorna el costo total de recorrer un grafo.

    Parameters
    ----------
    dm : DataFrame
        Matriz con los costos entre dos nodos del grafo.
    solucion : list
        Vector con el orden a recorrer el grafo.
    Returns
    -------
    float
        Costo de recorrer la solucion y volver al nodo de origen.
    """

    costo = 0.0
    for i in range(0, len(solucion) - 1):
        costo += dm[solucion[i]][solucion[i + 1]]
    costo += dm[solucion[-1]][solucion[0]]
    return costo


def seleccion_ruleta(poblacion, probabilidades, n, prng):
    """ Retorna una cantidad n de individuos desde una poblacion en forma aleatorea.

    Parameters
    ----------
    poblacion : list
        Elementos de la poblacion.
    probabilidades : list
        Probabilidades respectivas a los elementos de poblacion.
    n : int
        Numero de individuos a obtener desde la ruleta.
    prng : RandomState
        Generador de numeros pseudoaleatoreos.
    Returns
    -------
    list
        Listado de individuos.
    """
    seleccionados = []
    for _ in range(n):
        r = prng.rand()
        for (i, individuo) in enumerate(poblacion):
            if r <= probabilidades[i]:
                seleccionados.append(individuo)
                break
    return seleccionados

def leer_archivo_tsp(path):
    """ Funcion que retorna las coordenadas desde un archivo .tsp

    Parameters
    ----------
    path : String
        Direccion relativa del archivo TSP
    Returns
    -------
    list
        Lista con las coordenadas
    """
    data = []
    with open(path) as f:
        for line in f.readlines()[6:-1]:
            _, *b = line.split()
            data.append((float(i) for i in b))
    return data




"""Creacion de Ant Colony System"""
class ACS():
    
    """Variables a sintonizar:

        Archivo de entrada.
        Valor semilla generador valores randómicos.
        Tamaño de la colonia o número de hormigas.
        Condición de término o número de iteraciones.
        Factor de evaporación de la feromona (α).
        El peso del valor de la heurística (β).
        Valor de probabilidad límite (q0).
    """
    def __init__(self, n, alpha, beta, rho, max_it, dm, eta, prng):
        self.n = n
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.max_it = max_it
        self.dm = dm
        self.eta = eta
        self.prng = prng #prng Pseudorandom number generator
        # Solucion inicial y feromonas.
        self.ruta_global = generar_solucion(len(dm), prng)
        self.costo_global = calcular_costo(dm, self.ruta_global)
        self.tau_cero = np.reciprocal(self.costo_global)
        #self.tau_cero = 0.01
        self.tau = np.full(dm.shape, self.tau_cero)
        # Hormigas.
        self.hormigas = []
        self.mejor_hormiga = None
        
        
    def _init_hormigas(self):
        """ Asigna randomicamente una y solo una hormiga en cualquiera de los vertices de la solucion. """
        self.hormigas = []
        muestra = self.prng.choice(self.ruta_global, self.n, replace=False)
        for nodo_inicial in muestra:
            nk = list(self.ruta_global)
            nk.pop(nk.index(nodo_inicial))
            nueva_hormiga = Hormiga(self, nk, nodo_inicial)
            self.hormigas.append(nueva_hormiga)

class Hormiga():

    def __init__(self, acs, nk, nodo_inicial):
        self.acs = acs
        self.nk = nk
        self.nodo_inicial = nodo_inicial
        # Ruta de hormiga
        self.actual = nodo_inicial
        self.proximo = 0
        self.ruta = [nodo_inicial]
        self.costo = 0

"""Inicializacion de variables de entrada"""
if __name__ == '__main__':
    np.seterr(divide='ignore')#ignora la division por cero
    # python acs.py 24 0.1 2.5 0.9 500 berlin52.tsp 123
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
        alpha = float(sys.argv[2])
        beta = float(sys.argv[3])
        rho = float(sys.argv[4])
        max_it = int(sys.argv[5])
        tsp = leer_archivo_tsp(sys.argv[6])
        # Si no contiene semilla se realizara de forma aleatorea.
        if len(sys.argv) > 7:
            prng = RandomState(int(sys.argv[7]))#Inicializa en estado radomico la semilla
        else:
            prng = RandomState()
    # python acs.py
    else:
        n = 10
        alpha = 0.1
        beta = 2.5
        rho = 0.9
        max_it = 100
        tsp = leer_archivo_tsp("berlin52.tsp")
        prng = RandomState()
    # Coordenadas, Matriz de Distancia y Heuristica
    coord = pd.DataFrame(tsp, columns=['x_coord', 'y_coord'])
    dm = pd.DataFrame(distance_matrix(coord.values, coord.values))
    eta = np.where(dm != 0, np.reciprocal(dm), 0)
    acs = ACS(n, alpha, beta, rho, max_it, dm, eta, prng)
    acs.run()
    print(acs)

