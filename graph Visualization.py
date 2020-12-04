import networkx as nx
import matplotlib.pyplot as plt
import community

g = nx.read_edgelist('./dataset/facebook_combined.txt',create_using = nx.Graph(), nodetype = int)
print(nx.info(g))

sp = nx.spring_layout(g)
nx.draw_networkx(g, pos = sp, with_labels = False,alpha = 1.0,node_size=35)
plt.show()

