import pandas as pd
import networkx as nx


def _token_to_int_pair(v):
    v1, v2 = v.split(' ')
    return int(v1), int(v2)


def run():
    df_labels = pd.read_csv('../datasets/karate/labels.txt')

    edges = []
    with open('../datasets/karate/edges.txt') as f:
        for line in f:
            edges.extend([_token_to_int_pair(t.strip('[]'))
                          for t in line.strip().split('] [')])

    vertex_to_label_map = {}
    for _, row in df_labels.iterrows():
        vertex_to_label_map[row['vertext_id']] = row['label']

    vertex_ids = set()
    for edge in edges:
        vertex_ids.update(edge)

    node_data = []
    for vertex_id in vertex_ids:
        node_data.append((vertex_id, {'label': vertex_to_label_map[vertex_id]}))

    graph = nx.Graph()
    graph.add_nodes_from(node_data)
    for edge in edges:
        graph.add_edge(*edge)

    print(list(graph.nodes))
    print(list(graph.edges))


if __name__ == '__main__':
    run()
