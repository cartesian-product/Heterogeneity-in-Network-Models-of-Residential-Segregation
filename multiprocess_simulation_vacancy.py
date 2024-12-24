import networkx as nx
from matplotlib import pyplot as plt
import numpy as np
import pycxsimulator
import itertools
import dill

grid_lw = 32

class Person:
    def __init__(self, tolerance_threshold: float) -> None:
        self.color = np.random.choice([0,1])
        self.tolerance_threshold = tolerance_threshold


class Result:
    def __init__(self) -> None:
        self.graph = nx.Graph()
        self.parameters = []
        self.attribute_assortativity = []
        self.n_connected_components = []
        self.avg_cc_size = []
        self.std_cc_size = []        

def payoff(node, target):
    global G
    x = 0 ### local similarity
    occupied_neighbors = 0
    similar_neighbors = 0
    tolerance_threshold = G.nodes[node]['object'].tolerance_threshold
    score = 0
    for nbr in G.neighbors(target):
        if G.nodes[nbr]['object'] != None:
            if G.nodes[nbr]['object'].color == G.nodes[node]['object'].color:
                similar_neighbors += 1
            occupied_neighbors += 1
    if occupied_neighbors > 0:
        x = similar_neighbors / occupied_neighbors
    else:
        x = 1
    if (1-x) <= tolerance_threshold:
        score = 1
    else:
        score = 0
    return score

def set_tolerance_threshold(method='xie_zhou',tolerance_threshold=None):
    if method == 'xie_zhou':
        group = np.random.random()
        if group < .1047:
            tolerance_threshold = np.random.uniform(0.0,0.07)
        elif group < (.1047 + .1810):
            tolerance_threshold = np.random.uniform(0.07,0.21)
        elif group < (.1047 + .1810 + .2673):
            tolerance_threshold = np.random.uniform(0.21,0.36)
        elif group < (.1047 + .1810 + .2673 + .1386):
            tolerance_threshold = np.random.uniform(0.36,0.57)
        elif group < (.1047 + .1810 + .2673 + .1386 + .2659):
            tolerance_threshold = np.random.uniform(0.57,1.01)
        else:
            tolerance_threshold = None
        
    elif method == 'schelling' and tolerance_threshold == None:
        tolerance_threshold = .3961503279960033
    else: tolerance_threshold = tolerance_threshold
    
    return tolerance_threshold

def non_guttman_transfer_probabilities(node, candidate_vacancies):
    global G
    transfer_probabilities = []
    d = 0
    for v in candidate_vacancies:
        similar_neighbors = 0
        occupied_neighbors = 0
        for nbr in G.neighbors(v):
            if G.nodes[nbr]['object'] != None:
                if G.nodes[nbr]['object'].color == G.nodes[node]['object'].color:
                    similar_neighbors += 1
                occupied_neighbors += 1
        if occupied_neighbors > 0:
            x = similar_neighbors / occupied_neighbors
        else:
            x = 1
        d += np.exp(13*x - 17.9*x**2)
        transfer_probabilities.append(np.exp(13*x - 17.9*x**2))
    transfer_probabilities = [p / d for p in transfer_probabilities]
    return transfer_probabilities

def calculate_mixing():
    global G
    subgraph_nodes = []
    for node in G.nodes:
        if G.nodes[node]['object'] != None:
            subgraph_nodes.append(node)
    subgraph = nx.subgraph(G,subgraph_nodes)
    connected_components = list(nx.connected_components(subgraph))
    total_size = 0
    weighted_assortativity = 0
    for c in connected_components:
        component_size = len(list(c))
        total_size += component_size
        component_graph = nx.subgraph(G,list(c))
        if len(set([component_graph.nodes[c]['color'] for c in list(component_graph.nodes)])) == 1:
            weighted_assortativity += 1 * component_size
        else:
            weighted_assortativity += (nx.attribute_assortativity_coefficient(component_graph,'color') * component_size)
    graph_assortativity_coefficient = weighted_assortativity / total_size
    number_of_connected_components = len(list(connected_components))
    average_component_size = np.mean([len(c) for c in list(connected_components)])
    std_component_size = np.std([len(c) for c in list(connected_components)])
    return [graph_assortativity_coefficient, number_of_connected_components, average_component_size, std_component_size]

def update_result(change=True):
    global result, mixing_metrics
    if change:
        mixing_metrics = calculate_mixing()
    result.attribute_assortativity.append(mixing_metrics[0])
    result.n_connected_components.append(mixing_metrics[1])
    result.avg_cc_size.append(mixing_metrics[2])
    result.std_cc_size.append(mixing_metrics[3])


def initialize(excess_housing = 0, topology_modifier = 0, method='xie_zhou', tolerance_threshold=None):
    global G, pos, vacancies
    G = nx.grid_2d_graph(grid_lw,grid_lw, periodic=False)
    pos = dict((n,n) for n in G.nodes)
    vacancies = []
    if excess_housing > 0:
        for node in G.nodes:
            if np.random.random() < excess_housing:
                G.nodes[node]['object'] = None
                G.nodes[node]['color'] = 2
                vacancies.append(node)
            else:
                G.nodes[node]['object'] = Person(tolerance_threshold=set_tolerance_threshold(method,tolerance_threshold))
                G.nodes[node]['color'] = G.nodes[node]['object'].color
    else:
        for node in G.nodes:
            G.nodes[node]['object'] = Person(tolerance_threshold=set_tolerance_threshold(method, tolerance_threshold))
            G.nodes[node]['color'] = G.nodes[node]['object'].color
    if topology_modifier > 0:
        oG = G.copy()
        for _ in range(topology_modifier):
            origin = list(G.nodes)[np.random.choice(len(list(G.nodes)))]
            first = list(oG.neighbors(origin))
            neighborhood = []
            for node in first:
                neighborhood.append(node)
                for nbr in list(oG.neighbors(node)):
                    if nbr not in neighborhood:
                        neighborhood.append(nbr)
            for pair in itertools.product(neighborhood, neighborhood):
                if pair[0] != pair[1]:
                    G.add_edge(pair[0], pair[1]) 

def observe():
    global G, pos
    plt.cla()
    color_map = []
    for node in G.nodes:
        if G.nodes[node]['object'] == None:
            color_map.append('grey')
        elif G.nodes[node]['object'].color == 0:
            color_map.append('red')
        else:
            color_map.append('blue')
    nx.draw(G, pos = pos, node_size=30, node_color= color_map, with_labels=False)


def update():
    global G, vacancies, terminate, skip
    candidate_nodes = []
    for n in G.nodes:
        if G.nodes[n]['object'] != None:
            if G.nodes[n]['object'].tolerance_threshold == None:
                candidate_nodes.append(n)
            elif payoff(n,n) == 0:
                candidate_nodes.append(n)
    if len(candidate_nodes) == 0:
        print("No more candidate nodes.")
        terminate = True
        return None
    node_a = candidate_nodes[np.random.choice(len(candidate_nodes))]
    object_a = G.nodes[node_a]['object']
    if object_a == None:
        return None
    candidate_vacancies = []
    if object_a.tolerance_threshold == None:
        candidate_vacancies_probabilities = non_guttman_transfer_probabilities(node_a, vacancies)
        node_b = vacancies[np.random.choice(len(vacancies), p=candidate_vacancies_probabilities)]
    else:
        for v in vacancies:
            if payoff(node_a,v) == 1:
                candidate_vacancies.append(v)
        if len(candidate_vacancies) == 0:
            skip = True
            return None
        node_b = candidate_vacancies[np.random.choice(len(candidate_vacancies))]
    G.nodes[node_a]['object'] = None
    G.nodes[node_a]['color'] = 2
    G.nodes[node_b]['object'] = object_a
    G.nodes[node_b]['color'] = object_a.color
    vacancies.remove(node_b)
    vacancies.append(node_a)

def mp_simulate(n):
    global G, result, terminate, skip
    iters = 100
    excess_housing_fraction = 0.30
    results = []
    for k in range(iters):
        terminate = False
        method = "schelling"
        topology_modifier = n * 32
        initialize(topology_modifier = topology_modifier, excess_housing=excess_housing_fraction, method=method)
        result = Result()
        result.parameters = [f"Topology Modifier: {topology_modifier}", f"Excess Housing Fraction: {excess_housing_fraction}", f"Method: {method}"]
        update_result()
        for j in range(4000):
            skip = False
            update()
            if terminate == True:
                update_result(change=False)
                break
            elif skip == True:
                update_result(change=False)
            else:
                update_result()
        result.graph = G
        results.append(result)
    
    with open(f"sch_results_vac{n}.pkl", "wb") as dill_file:
        dill.dump(results, dill_file)

    results = []
    for k in range(iters):
        terminate = False
        method = "xie_zhou"
        topology_modifier = n * 32
        initialize(topology_modifier = topology_modifier, excess_housing=excess_housing_fraction, method=method)
        result = Result()
        result.parameters = [f"Topology Modifier: {topology_modifier}", f"Excess Housing Fraction: {excess_housing_fraction}", f"Method: {method}"]
        update_result()
        for j in range(4000):
            skip = False
            update()
            if terminate == True:
                update_result(change=False)
                break
            elif skip == True:
                update_result(change=False)
            else:
                update_result()
        result.graph = G
        results.append(result)
    
    with open(f"xz_results_vac{n}.pkl", "wb") as dill_file:
        dill.dump(results, dill_file)