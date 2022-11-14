import sys
import numpy as np
import pandas as pd
from operator import attrgetter
from numpy.random import RandomState
from scipy.spatial import distance_matrix

"""Inicio Funciones externas al ACS"""

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


def calcular_costo(df, solucion):
    """ Retorna el costo total de recorrer un grafo.

    Parameters
    ----------
    df : DataFrame
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
        costo += df[solucion[i]][solucion[i + 1]]
    costo += df[solucion[-1]][solucion[0]]
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
        Factor de evaporación de la feromona.
        El peso del valor de la heurística.
        Valor de probabilidad límite.
    """
    def __init__(self, tamaño_colonia, feromona, heuristica, probabilidad, max_iteraciones, df, nu_ij, semilla):
        self.tamaño_colonia = tamaño_colonia
        self.feromona = feromona
        self.heuristica = heuristica
        self.probabilidad = probabilidad
        self.max_iteraciones = max_iteraciones
        self.df = df
        self.nu_ij = nu_ij
        self.semilla = semilla #semilla Pseudorandom number generator
        # Solucion inicial y feromonas.
        self.ruta_global = generar_solucion(len(df), semilla)
        self.costo_global = calcular_costo(df, self.ruta_global)
        self.tau_cero = np.reciprocal(self.costo_global)
        #self.tau_cero = 0.01
        self.tau = np.full(df.shape, self.tau_cero)
        # Hormigas.
        self.hormigas = []
        self.mejor_hormiga = None
        
    def _init_hormigas(self):
        """ Inicializa una hormiga randomicamente en cualquiera de los vertices de la solucion. """
        self.hormigas = []
        try:
            muestra = self.semilla.choice(self.ruta_global, self.tamaño_colonia, replace=False)
        except ValueError:
            muestra = self.semilla.choice(self.ruta_global, self.tamaño_colonia, replace=True)
        for nodo_inicial in muestra:
            nk = list(self.ruta_global)
            nk.pop(nk.index(nodo_inicial))
            nueva_hormiga = Hormiga(self, nk, nodo_inicial)
            self.hormigas.append(nueva_hormiga)

    def _actualizacion_local(self, hormiga):
        """ Agrega feromonas solamente en el tramo seleccionado (Ecuacion 4). """
        a = (1 - self.feromona) * self.tau[hormiga.actual][hormiga.proximo]
        b = self.feromona * self.tau_cero
        self.tau[hormiga.actual][hormiga.proximo] = a + b

    def _actualizar_solucion(self):
        """ Actualiza si hay una nueva mejor solucion global. """
        if self.mejor_hormiga.costo < self.costo_global:
            self.ruta_global = self.mejor_hormiga.ruta
            self.costo_global = self.mejor_hormiga.costo
            print("Nuevo costo global:", self.mejor_hormiga.costo)

    def _actualizacion_global(self):
        """ Evapora todas las feromonas y agrega a las de la mejor ruta. """
        # Evaporacion a todas las feromonas.
        self.tau = np.multiply(self.tau, (1 - self.feromona))
        # Agrega feromonas en la ruta seleccionada.
        ruta = self.mejor_hormiga.ruta
        delta = np.reciprocal(self.mejor_hormiga.costo)
        for i in range(1, len(self.ruta_global) - 1):
            self.tau[ruta[i - 1]][ruta[i]] += self.feromona * delta

    def run(self):
        """ Se ejecutara hasta que se cumple la condicion de termino. """
        for it in range(self.max_iteraciones):
            # Asignacion de hormigas.
            self._init_hormigas()
            # Para cada vertice del grafo.
            for _ in range(len(self.df) - 1):
                # Seleccionar el proximo segmento en el grafo.
                for hormiga in self.hormigas:
                    hormiga.run()
                # Actualizacion local.
                for hormiga in self.hormigas:
                    self._actualizacion_local(hormiga)
                    hormiga._avanzar()
            # Sumar el retorno a el nodo inicial y obtener la mejor hormiga.
            for hormiga in self.hormigas:
                hormiga.costo += self.df[hormiga.nodo_inicial][hormiga.actual]
            self.mejor_hormiga = min(self.hormigas, key=attrgetter('costo'))
            # Actualizar la solucion global de ser necesario.
            self._actualizar_solucion()
            # Actualizacion global.
            self._actualizacion_global()
            print("Iteracion:", it)

    def __str__(self):
        """ Retorna el resultado final. """
        return f'Distancia: {self.costo_global}\nSolucion: {self.ruta_global}'

"""Clase Hormiga"""

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
    
    def _explotar(self):
        """ (Primera ecuacion). """
        maximo = 0
        for vecino in self.nk:
            tau = self.acs.tau[self.actual][vecino]
            nb = self.acs.nu_ij[self.actual][vecino] ** self.acs.heuristica
            if maximo < tau * nb:
                maximo = tau * nb
                self.proximo = vecino

    def _explorar(self):
        """ (Segunda ecuacion). """
        valores = []
        for vecino in self.nk:
            tau = self.acs.tau[self.actual][vecino]
            nb = self.acs.nu_ij[self.actual][vecino] ** self.acs.heuristica
            valores.append(tau * nb)
        suma_valores = sum(valores)
        relativo = [x / suma_valores for x in valores]
        probabilidades = [sum(relativo[:i + 1]) for i in range(len(relativo))]
        seleccion = seleccion_ruleta(self.nk, probabilidades, 1, self.acs.semilla)
        self.proximo = seleccion[0]

    def _avanzar(self):
        """ Permite a la hormiga avanzar en el grafo. """
        self.costo += self.acs.df[self.actual][self.proximo]
        self.ruta.append(self.proximo)
        self.nk.pop(self.nk.index(self.proximo))
        self.actual = self.proximo
        self.proximo = 0

    def run(self):
        """ Regla de transicion, decide en que forma se eligira el proximo nodo. """
        r = self.acs.semilla.rand()
        if r <= self.acs.probabilidad:
            self._explotar()
        else:
            self._explorar()
            

"""Inicializacion de variables de entrada"""

"""Variables a sintonizar
        Archivo de entrada. tsp
        Valor semilla generador valores randómicos. semilla
        Tamaño de la colonia o número de hormigas. tamaño_colonia
        Condición de término o número de iteraciones. max_iteraciones
        Factor de evaporación de la feromona (α). feromona
        El peso del valor de la heurística (β). heuristica
        Valor de probabilidad límite (q0). probabilidad
    """
if __name__ == '__main__':
    np.seterr(divide='ignore')#ignora la division por cero
    # python acs.py 24 0.1 2.5 0.9 500 berlin52.tsp 123
    if len(sys.argv) > 1:
        tamaño_colonia = int(sys.argv[1])
        feromona = float(sys.argv[2])
        heuristica = float(sys.argv[3])
        probabilidad = float(sys.argv[4])
        max_iteraciones = int(sys.argv[5])
        archivo_entrada = leer_archivo_tsp(sys.argv[6])
        # Si no contiene semilla se realizara de forma aleatorea.
        if len(sys.argv) > 7:
            semilla = RandomState(int(sys.argv[7]))#Inicializa en estado radomico la semilla
        else:
            semilla = RandomState()
    # python acs.py
    else:
        tamaño_colonia = 10
        feromona = 0.1
        heuristica = 2.5
        probabilidad = 0.9
        max_iteraciones = 100
        archivo_entrada = leer_archivo_tsp("berlin52.tsp")
        semilla = RandomState()
        
    # Coordenadas, Matriz de Distancia y Heuristica
    coord = pd.DataFrame(archivo_entrada, columns=['x_coord', 'y_coord'])
    df = pd.DataFrame(distance_matrix(coord.values, coord.values))
    nu_ij = np.where(df != 0, np.reciprocal(df), 0)
    acs = ACS(tamaño_colonia, feromona, heuristica, probabilidad, max_iteraciones, df, nu_ij, semilla)
    acs.run()
    print(acs)

