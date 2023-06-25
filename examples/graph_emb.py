import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from karateclub.node_embedding.neighbourhood.deepwalk import DeepWalk


G = nx.random_tree(40)
# nx.draw(G)
# G = nx.complete_graph(5)
# nx.draw(G)
nx.draw_networkx(G, with_labels=True)
plt.show()

deepwalk_alg = DeepWalk(dimensions=2)
deepwalk_alg.fit(G)
graph_embeding = deepwalk_alg.get_embedding()
plt.scatter(graph_embeding[:, 0], graph_embeding[:, 1])
plt.show()
