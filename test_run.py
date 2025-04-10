# First, read the benchmark of a certain dataset by specifying the name. 
# The nine supported datasets are: cora, citeseer, pubmed, cs, physics, photo, computers, arxiv, and proteins.
# For example, for the Cora dataset:
from nas_bench_graph.readbench import light_read
datasets = ['cora', 'citeseer', 'pubmed', 'cs', 'physics', 'photo', 'computers', 'arxiv', 'proteins']
for dataset in datasets:
    bench = light_read(dataset)
    print(f"The number of architectures (dataset: {dataset:<9}): {len(bench):>6}")


###########
# How to define a network by configuration
###########
dataset = 'cora'
bench = light_read(dataset)
# Define an architecture
from nas_bench_graph.architecture import Arch
network = Arch([0, 1, 2, 1], ['gcn', 'gin', 'fc', 'cheb']) 
# 0 means the inital computing node is connected to the input node
# 1 means the next computing node is connected to the first computing node
# 2 means the next computing node is connected to the second computing node 
# 1 means there is another computing node connected to the first computing node
print("arch.valid_hash(): ", network.valid_hash())


info = bench[network.valid_hash()]
info['valid_perf']   # validation performance
info['perf']         # test performance
info['latency']      # latency
info['para']         # number of parameters
print(info)


###########
# How to define a network by index
###########
from nas_bench_graph.architecture import all_archs
index = 123
network = all_archs()[index]
info = bench[network.valid_hash()]
print(info)