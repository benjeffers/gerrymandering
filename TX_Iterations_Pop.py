from scipy.io import loadmat 
from scipy import sparse
import numpy as np
import time
import pandas as pd
import networkx as nx
import random
from Plan import Plan

num_districts = 36


#weight the edges of G randomly
def weight_edges_rand(G):
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = random.uniform(0, 1)

    return G

# create the spanning tree and get the adjacency matrix
def spanning_tree(G):
    min_span_tree = nx.minimum_spanning_tree(G)
    span_adj = nx.to_numpy_array(min_span_tree)
    span_edge_nodes = list(min_span_tree.edges())

    return span_adj, np.array(span_edge_nodes)

def cut_graph(weighted_G):
    span_adj, span_edge_nodes = spanning_tree(weighted_G)
    possible_cuts = [i for i in range(0, 8941)]
    cut_edges = [0]*35

    for i in range(0, num_districts-1):
        r = random.randint(0, len(possible_cuts))
        remove_edge = possible_cuts[r]
        cut_edges[i] = remove_edge
        possible_cuts.remove(remove_edge)

    for i in range(0, num_districts-1):
        r_vertex_1 = span_edge_nodes[cut_edges[i]][0]
        r_vertex_2 = span_edge_nodes[cut_edges[i]][1]
        span_adj[r_vertex_1][r_vertex_2] = 0
        span_adj[r_vertex_2][r_vertex_1] = 0

    return span_adj

#creates a plan given a graph and the list of VTD populations
def create_initial_plan(G, vert_pop):

    districts = []
    size = []
    pop = []

    weighted_G = weight_edges_rand(G)
    
    cut_adjacency = cut_graph(weighted_G)
    split_graph = nx.from_numpy_matrix(cut_adjacency)

    connected_comps = nx.connected_components(split_graph)
    for comp in connected_comps:
        districts.append(list(comp))

    for i in range(0, num_districts):
        pop_total = 0
        dist_size = 0
        for dist in districts[i]:
            pop_total += vert_pop[dist]
            dist_size += 1
        pop.append(pop_total)
        size.append(dist_size)


    return Plan(districts, size, pop)

def modify_by_spantree_bestcut_pop(subgraph, sub_pop):
    weighted_subgraph = weight_edges_rand(subgraph)
    span_adj, span_edge_nodes = spanning_tree(weighted_subgraph)

    temp_adj = span_adj

    num_nodes = len(span_adj[0])
    num_edges = num_nodes - 1

    possible_cuts = np.array([i for i in range(0, num_edges)])

    removed_edges = []
    comp_length = np.array([])
    comp_pop = np.array([])


    while len(possible_cuts):
        
        r = random.randint(0, len(possible_cuts))
        
        remove_edge = possible_cuts[r]
        removed_edges.append(remove_edge)

       

        remove_idx = np.where(possible_cuts == remove_edge)[0][0]
        
        possible_cuts = np.delete(possible_cuts, remove_idx)
        
        r_vertex_1 = span_edge_nodes[remove_edge][0]
        r_vertex_2 = span_edge_nodes[remove_edge][1]

       

        temp_adj[r_vertex_1][r_vertex_2] = 0
        temp_adj[r_vertex_2][r_vertex_1] = 0

        x = np.zeros(num_nodes)
        x[r_vertex_1] = 1
       
        conn_bool = True
        while conn_bool:
            y = x
            x = np.matmul(temp_adj, x) + x

            if np.array_equal(x, y):
                conn_bool = False
        
        comp_1 = np.where(x > 0)[0]
        comp_2 = np.where(x == 0)[0]


        np.append(comp_pop, [sum(sub_pop[comp_1]), sum(sub_pop[comp_2])])
        np.append(comp_length, [len(comp_1), len(comp_2)])

        if sum(sub_pop[comp_1]) == sum(sub_pop[comp_2]):
            done = True
            possible_cuts = np.array([])

        if sum(sub_pop[comp_1]) > sum(sub_pop[comp_2]):
            node_1 = np.in1d(span_edge_nodes[:,0], comp_2).astype(int)
            node_2 = np.in1d(span_edge_nodes[:,1], comp_2).astype(int)
            edges = np.where(node_1 == node_2, 1, 0)
            edges_comp_2 = np.where(edges == 1)
            possible_cuts = np.setdiff1d(possible_cuts, edges_comp_2)
        
        if sum(sub_pop[comp_2]) > sum(sub_pop[comp_1]):
            node_1 = np.in1d(span_edge_nodes[:,0], comp_1).astype(int)
            node_2 = np.in1d(span_edge_nodes[:,1], comp_1).astype(int)
            edges = np.where(node_1 == node_2, 1, 0)
            edges_comp_1 = np.where(edges == 1)
            possible_cuts = np.setdiff1d(possible_cuts, edges_comp_1)

    pop_diff = abs(comp_pop[0,:] - comp_pop[1,:])
    m = min(pop_diff)
    idx = np.where(pop_diff == m)[0][0]

    remove_edge = removed_edges[idx]

    r_vertex_1 = span_edge_nodes[remove_edge][0]
    r_vertex_2 = span_edge_nodes[remove_edge][1]
    span_adj[r_vertex_1][r_vertex_2] = 0
    span_adj[r_vertex_2][r_vertex_1] = 0


    x = np.zeros(num_nodes)
    x[r_vertex_1] = 1

    conn_bool = True
    while conn_bool:
        y = x
        x = np.matmul(temp_adj, x) + x

        if np.array_equal(x, y):
            conn_bool = False
        
    comp_1 = np.where(x > 0)[0]
    comp_2 = np.where(x == 0)[0]

    return comp_1, comp_2
    
    

#load in the voter data
TX_data = pd.read_parquet('TX_data.parquet.gzip')
#load in the adjacency matrix to make the graph
adj = pd.read_parquet('TX_adj.parquet.gzip').to_numpy()


#create graph from adjacency matrix
G = nx.from_numpy_matrix(adj)
edge_list = list(G.edges())

#get list of populations in VTDs
vert_pop = list(TX_data['TOTPOP'])
total_votes = sum(vert_pop)


#People Per Representative (PPR)
PPR = total_votes/num_districts
slack = 0.0010
lower_pop = PPR - slack*PPR
upper_pop = PPR + slack*PPR

print('Expected pop is {:.2f}'.format(PPR))
print('lower pop with {:.2f} slack is {:.2f}'.format(slack*100, lower_pop))
print('upper pop with {:.2f} slack is {:.2f}'.format(slack*100, upper_pop))


plan = create_initial_plan(G, vert_pop)

keep = plan.pop
gap = max(plan.pop) - min(plan.pop)
print("Gap: " + str(gap))

temp_plan = plan

initial = plan.pop # so that we can chekc later if the botes changed

rep = 0
tries = []
gap = 50000000

while rep < 10000 and gap > slack*PPR:
    rep = rep + 1
    connected_bool = False

    while not connected_bool:
        rand_dists = random.sample(range(0, num_districts), 2) #we do this so that the two numbers are not the same
        district_1 = rand_dists[0]
        district_2 = rand_dists[1]
        is_connected = np.sign(sum(sum(adj[np.ix_(plan.districts[district_1], plan.districts[district_2])])))
        if not is_connected:
            connected_bool = True

    subgraph_idx = plan.districts[district_1] + plan.districts[district_2]
    
    sub_adj = adj[np.ix_(subgraph_idx, subgraph_idx)]

    subgraph = nx.from_numpy_array(sub_adj)
    subgraph.remove_edges_from(nx.selfloop_edges(subgraph))
    sub_pop = np.array([vert_pop[index] for index in subgraph_idx])
    target_pop = sum(sub_pop)/2

    print("sub pop: " + str(sub_pop))

    tries = 0
    max_tries = 50
    check = False
    equal_plans = False 
    save_d1 = plan.districts[district_1]
    save_d2 = plan.districts[district_2]



    
    modify_by_spantree_bestcut_pop(subgraph, sub_pop)
    rep = 20000
