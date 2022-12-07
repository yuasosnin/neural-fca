from typing import *

from copy import copy
from dataclasses import dataclass
import networkx as nx

from fcapy.lattice.formal_concept import FormalConcept
from fcapy.poset import POSet
from fcapy.visualizer.line_layouts import calc_levels

import torch
import torch.nn as nn
from sparselinear import SparseLinear

    
class ConceptNetwork(nn.Sequential):
    def __init__(self, lattice, nonlinearity: Callable = nn.ReLU):
        attributes = tuple(lattice[lattice.bottom].intent)
        self._poset = self._poset_from_best_concepts(lattice, attributes)
        self._spec = self._get_layer_specifications(self._poset)
        layers = self._get_layers_from_connections(self._spec, nonlinearity)
        super().__init__(*layers)
    
    @dataclass
    class _POSetElement:
        intent: FrozenSet[Union[int, str]]
        children: FrozenSet[int]

        def __init__(self, intent, children=None):
            self.intent = frozenset(intent)
            if children and children is not None:
                self.children = children
            else:
                self.children = frozenset()

        def __eq__(self, other) -> bool:
            return self.intent == other.intent

        def __le__(self, other) -> bool:
            return (self.intent & other.intent) == other.intent

        def __hash__(self):
            return hash(self.intent)

    def _poset_from_best_concepts(
            self, best_concepts: List[FormalConcept], attributes: Tuple[str]) -> POSet:
        poset = POSet(best_concepts)

        concepts = {self._POSetElement(c.intent) for c in poset}
        single_concepts = {self._POSetElement({i}) for i in attributes}
        output = {self._POSetElement(attributes)}

        poset = POSet(concepts | single_concepts | output)
        for element in range(len(poset)):
            poset[element].children = poset.children(element)
        return poset
    
    def _get_bipartite_connections(self, current_level, next_level):
        next_level_ = copy(next_level)
        connections = []
        for i, element in enumerate(next_level):
            for j in self._poset.parents(element):
                connections.append((current_level.index(j), i))
                self._poset[j].children -= set(next_level)

        for j, element in enumerate(current_level):
            if self._poset[element].children:
                connections.append((j, len(next_level_)))
                next_level_.append(element)
        
        return connections, next_level_
    
    def _get_layer_specifications(self, poset):
        _, level_table = calc_levels(poset)
        current_level = level_table[0]
        spec = []
        for i in range(1, len(level_table)):
            connections, next_level = self._get_bipartite_connections(current_level, level_table[i])
            in_features = len(current_level)
            out_features = len(next_level)
            spec.append((in_features, out_features, connections))
            current_level = next_level
        return spec
    
    def networkx_graph(self):
        pos = {}
        edges = {}
        total_nodes = 0
        all_connections = []
        for n, (in_features, out_features, connections) in enumerate(self._spec):
            connections_ = [(a+total_nodes, b+total_nodes+in_features) for (a,b) in connections]
            pos |= {i+total_nodes: [(i+1)/(in_features+1), -n] for i in range(in_features)}
            pos |= {i+total_nodes+in_features: [(i+1)/(out_features+1), -n-1] for i in range(out_features)}
            total_nodes += in_features
            all_connections.extend(connections_)
            
            weights = get_weights(self[n*2])
            edges |= dict(zip(connections_, weights))
            
        return nx.Graph(all_connections), pos, {frozenset(k):v.item() for k,v in edges.items()}
    
    def _get_layers_from_connections(self, spec, nonlinearity):
        linear_layers = []
        for in_features, out_features, connections in spec:
            connections = [(b,a) for (a,b) in connections]
            connectivity = torch.tensor(connections, dtype=torch.long).T
            layer = SparseLinear(in_features, out_features, connectivity=connectivity)
            linear_layers.append(layer)

        layers = [layer for ll in linear_layers for layer in [ll, nonlinearity()]][:-2] + [nn.Linear(in_features, 1), nn.Sigmoid()]
        return layers
    
def get_weights(x):
    if isinstance(x, SparseLinear):
        w = x.weight.values()
    else:
        w = x.weight.data[0]
    return w