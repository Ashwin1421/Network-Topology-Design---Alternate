from sys import argv
from math import sqrt, pow, sin, cos, radians
from time import localtime, strftime, time
from decimal import getcontext, Decimal
import networkx as nx, numpy as np
from matplotlib import pyplot as plt, pylab


class SolutionOne:
    def __init__(self, N, d):
        self.N = int(N)
        self.diameter = d
        if self.N % 2 == 0:
            self.degree = 3
        else:
            self.degree = 4
        self.solution_set = []
        self.hops = []

    def node_dist(self, n1, n2):
        """
        Computes the geometric distance between two sets of coordinates,
        using the geometric coordinate distance formula.
        d = sqrt( (x2-x1)^2 + (y2-y1)^2 )

        :param n1: Tuple (x1,y1) coordinates of node n1.
        :param n2: Tuple (x2,y2) coordinates of node n2.
        :return: Coordinate Distance between n1 and n2.
        """
        x1, y1 = n1
        x2, y2 = n2

        return sqrt(pow((x2 - x1), 2) + pow((y2 - y1), 2))

    def get_total_dist(self):
        """
        Adding up all individual link distances.
        A link distance is just distance between nodes u and v of and edge e = (u,v).
        :return: Total cumulative sum of all individual link distances.
        """
        sum = 0
        for n1, n2, d in self.graph.edges(data=True):
            sum += self.node_dist(self.graph.node[n1]['pos'], self.graph.node[n2]['pos'])
            d['weight'] = self.node_dist(self.graph.node[n1]['pos'], self.graph.node[n2]['pos'])

        return sum

    def gen_graph(self, d):

        """
        A graph generating routine.
        :param d: Minimum degree (for every node) required to build the graph.
        :return: True if the generated graph satisfies all the given constraints, else False.
        """
        self.degree_sequence = [d] * self.N
        for i in range(d, self.N, d):
            self.degree_sequence[i] += d + 1

        if nx.is_valid_degree_sequence(self.degree_sequence):
            self.graph = nx.random_degree_sequence_graph(self.degree_sequence)
            self.pos = nx.random_layout(self.graph)
            nx.set_node_attributes(self.graph, 'pos', self.pos)
            self.paths = nx.shortest_path_length(self.graph)
            for n1 in self.graph.nodes():
                for n2 in self.graph.nodes():
                    if n1 != n2:
                        self.hops.append(self.paths[n1][n2])

            node_degrees = [self.graph.degree(n) for n in self.graph.nodes()]
            if max(self.hops) <= self.diameter and (min(node_degrees) <= self.degree <= max(node_degrees)):
                return True
            else:
                return False


    def save_graph(self, graph, file_name, pos):
        """
        This method plots the generated graph and all its components.
        The plot is saved to current directory as a '.png' file.
        :param graph: Input graph object.
        :param file_name: Input file object.
        :param pos: Input graph.node position dictionary.
        :return: Total graph cost.
        """
        # initialze Figure
        plt.figure(figsize=(25, 25), dpi=100)
        plt.axis('off')
        fig = plt.figure(1)
        fig.suptitle('Random Layout Graph', fontsize=12)
        nx.draw_networkx_nodes(graph, pos, label='Total Nodes = ' + str(self.N))
        nx.draw_networkx_edges(graph, pos, label='Total Edges = ' + str(self.graph.size()))
        nx.draw_networkx_labels(graph, pos)

        self.total_dist = self.get_total_dist()

        edge_labels = dict()

        for u, v, d in self.graph.edges(data=True):
            edge_labels[(u, v)] = d['weight']

        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)

        cut = 1.20
        xmax = cut * max(xx for xx, yy in pos.values())
        ymax = cut * max(yy for xx, yy in pos.values())
        plt.xlim(0, xmax)
        plt.ylim(0, ymax)

        plt.plot([], [], ' ', label='Total Cost = ' + str(self.total_dist))

        plt.legend(numpoints=2)
        plt.savefig(file_name, bbox_inches="tight", pad_inches=0.1)
        pylab.close()
        del fig
        return self.total_dist

    def run(self):
        """
        A routine to run the entire algorithm for a large search space.
        :return: Minimum network cost.
        """
        count = 1
        while True:
            count += 1
            if self.gen_graph(self.degree):

                f = 'Solution_A_' + str(self.N) + '_' + strftime("%d-%m-%Y_%H-%M-%S", localtime()) + '.png'
                self.solution_set.append(self.save_graph(self.graph, f, self.pos))
                break
            else:
                continue

        return min(self.solution_set), count


class SolutionTwo:
    def __init__(self, N, d):
        self.N = int(N)
        self.diameter = d
        self.degree = 3
        self.pos = dict()
        self.graph = nx.Graph()
        self.precision = 4

    def node_dist(self, n1, n2):
        """
        Computes the geometric distance between two sets of coordinates,
        using the geometric coordinate distance formula.
        d = sqrt( (x2-x1)^2 + (y2-y1)^2 )

        :param n1: Tuple (x1,y1) coordinates of node n1.
        :param n2: Tuple (x2,y2) coordinates of node n2.
        :return: Coordinate Distance between n1 and n2.
        """
        x1, y1 = n1
        x2, y2 = n2

        return sqrt(pow((x2 - x1), 2) + pow((y2 - y1), 2))

    def get_total_dist(self):
        """
        Add up all the individual link distances.
        :return: Total network cost.
        """
        sum = 0
        for u, v, d in self.graph.edges(data=True):
            sum += d['weight']

        return sum

    def getXY(self, radius, Cx, Cy, angle):
        """
        x = cx + r * cos(a)
        y = cy + r * sin(a)

        :param radius: Radius of unit circle, i.e 1.
        :param Cx: x coordinate of center of circle.
        :param Cy: y coordinate of center of circle.
        :param angle: Angle from 0 to 360 in degrees.
        :return: Point (x,y) on the circumference of the circle.
        """

        return round(float(Cx + radius * Decimal(cos(radians(angle)))), self.precision), round(
            float(Cy + radius * Decimal(sin(radians(angle)))), self.precision)

    def gen_node_pos(self):
        """
        Routine to generate all node positions on the circumference of
        a unit circle, by keeping the 1st node at the center of the circle.
        :return: None.
        """
        cx = 0
        cy = 0
        rad = 1
        step = 360 / (self.N - 1)
        self.pos[0] = (0, 0)
        for n, a in enumerate(np.arange(0, 360, step)):
            self.pos[n + 1] = self.getXY(rad, cx, cy, a)

    def gen_graph(self):

        """
        Generate the graph using the previously generated node positions.
        :return: True if all the given constraints are satisfied, else False.
        """
        for n in range(self.N):
            self.graph.add_node(n, pos=self.pos[n])

        for n in range(1, self.N):
            self.graph.add_edge(0, n, weight=self.node_dist(self.pos[0], self.pos[n]))

        for n in range(1, self.N - 1):
            self.graph.add_edge(n, n + 1, weight=self.node_dist(self.pos[n], self.pos[n + 1]))

        self.graph.add_edge(self.N - 1, 1, weight=self.node_dist(self.pos[self.N - 1], self.pos[1]))

        self.shortest_path = nx.shortest_path_length(self.graph)
        self.hops = []
        for u in self.graph.nodes():
            for v in self.graph.nodes():
                if u != v:
                    self.hops.append(self.shortest_path[u][v])

        node_degrees = [self.graph.degree(n) for n in self.graph.nodes()]
        if max(self.hops) <= self.diameter and (min(node_degrees) <= 3 <= max(node_degrees)):
            return True
        else:
            return False

    def save_graph(self, graph, file_name, pos):
        """
        Save the generated graph as '.png' figure.
        :param graph: Input graph object.
        :param file_name: Input file object.
        :param pos: Input graph.node position dictionary.
        :return: Minimum network cost.
        """
        # initialze Figure
        plt.figure(figsize=(25, 25), dpi=100)
        plt.axis('off')
        fig = plt.figure(1)
        fig.suptitle('Circular Layout Graph', fontsize=12)
        nx.draw_networkx_nodes(graph, pos, label='Total Nodes = ' + str(self.N))
        nx.draw_networkx_edges(graph, pos, label='Total Edges = ' + str(self.graph.size()))
        nx.draw_networkx_labels(graph, pos)

        self.total_dist = self.get_total_dist()

        edge_labels = nx.get_edge_attributes(graph, 'weight')
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)

        cut = 1.20
        xmax = cut * max(xx for xx, yy in pos.values())
        ymax = cut * max(yy for xx, yy in pos.values())
        plt.xlim(-1.5, xmax)
        plt.ylim(-1.5, ymax)

        plt.plot([], [], ' ', label='Total Cost = ' + str(self.total_dist))

        plt.legend(numpoints=2)
        plt.savefig(file_name, bbox_inches="tight", pad_inches=0.1)
        pylab.close()
        del fig
        return self.total_dist

    def run(self):
        """
        Runs algorithm B with all necessary parameters.
        :return: Minimum network cost.
        """
        self.gen_node_pos()
        if self.gen_graph():
            f = 'Solution_B_' + str(self.N) + '_' + strftime("%d-%m-%Y_%H-%M-%S", localtime()) + '.png'
            return self.save_graph(self.graph, f, nx.get_node_attributes(self.graph, 'pos'))


class Main:
    """
    Main class which calls the 2 solutions with all necessary parameters.
    """
    def __init__(self, N, algorithm):
        self.N = int(N)
        self.diameter = 4
        self.A = algorithm

    def run(self):
        """
        Run selected algorithm with passed number of nodes.
        :return: Cost returned by selected algorithm.
        """

        if self.A == '-a':
            cost, runs = SolutionOne(self.N, self.diameter).run()
            print('Total Network Cost =', cost)
            print('Solved in %d tries.' % runs)
            return cost
        if self.A == '-b':
            cost = SolutionTwo(self.N, self.diameter).run()
            print('Total Network Cost =', cost)
            return cost


if __name__ == '__main__':

    start_time = time()
    output_data = []

    if len(argv) != 4:
        print('Usage : Main.py -N 20 -a')
        print('Options:')
        print('1. -N :- Number of nodes in network.')
        print('2. -a :- Generate Network topology using 1st algorithm.')
        print('2. -b :- Generate Network topology using 2nd algorithm.')
    elif len(argv) == 4:
        N = int(argv[2])
        a = argv[3]
        if a != '-a' and a != '-b':
            print('Invalid option %s.' % a)
            exit(1)
        else:
            Main(N, a).run()
            print('Took %s seconds to complete.' % (time() - start_time))
