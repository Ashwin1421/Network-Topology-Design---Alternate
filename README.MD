# Network Topology Design - Heuristic Based Approach.

A heuristic approach towards finding minimum cost network topologies for different network sizes. 
For finding minimum costs, there are two different approaches.
One uses a random approach and tries to find local minima in the search space. 
Other approach is much more defined and finds the optimal solution such that any alteration to network parameters 
leads to a network cost more than what is already achieved, therefore it achieves an optimal solution.

## Getting Started

On GitHub, navigate to the main page of the repository.
Under the repository name, click **Clone or download.**

### Prerequisites

python 3.6
networkx
matplotlib
numpy

### Installing

After cloning the repository, simply launch a local terminal window and traverse to your local directory.
(Command Prompt in case of Windows/ Terminal in case of Linux). 
Run the main program by executing the following commands :-

Usage : Main.py -N 20 -a
Options:
1. -N :- Number of nodes in network.
2. -a :- Generate Network topology using 1st algorithm.
2. -b :- Generate Network topology using 2nd algorithm.
```
python Main.py -N 16 -b
```
Run the program for 16 nodes and algorithm B.

```
python Main.py -N 30 -a
```
Run the program for 30 nodes and algorithm A.


## Author

**Ashwin Joshi**
**ashwin.joshi12@live.com**
