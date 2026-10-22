from typing import List, Dict, Any, Set, Optional, Generator


class Graph:
    def __init__(self):
        self.vertices: Set[Any] = set()
        self.graph: Dict[Any, List[Any]] = {}

    def add_vertex(self, u: Any):
        self.vertices.add(u)
        if u not in self.graph:
            self.graph[u] = []

    def add_edge(self, u: Any, v: Any):
        self.add_vertex(u)
        self.add_vertex(v)
        self.graph[u].append(v)

    def get_sccs(self) -> List[List[Any]]:
        """
        Tarjan's algorithm for finding SCCs.
        """
        index_counter = [0]
        stack = []
        lowlink = {}
        index = {}
        result = []

        def connect(node):
            index[node] = index_counter[0]
            lowlink[node] = index_counter[0]
            index_counter[0] += 1
            stack.append(node)

            try:
                successors = self.graph.get(node, [])
                for successor in successors:
                    if successor not in index:
                        connect(successor)
                        lowlink[node] = min(lowlink[node], lowlink[successor])
                    elif successor in stack:
                        lowlink[node] = min(lowlink[node], index[successor])
            except RecursionError:
                # Fallback or handle deep recursion if needed
                pass

            if lowlink[node] == index[node]:
                connected_component = []
                while True:
                    successor = stack.pop()
                    connected_component.append(successor)
                    if successor == node:
                        break
                result.append(connected_component)

        for node in self.vertices:
            if node not in index:
                connect(node)

        return result

    def topological_sort(self) -> List[Any]:
        """
        Returns vertices in topological order (u before v if u->v).
        Note: This is only valid for DAGs. For cyclic graphs, this returns *some* order 
        compatible with the condensation DAG if used on SCCs, but here we run it on the graph itself?
        No, the stack estimator runs it on the SCC DAG.
        So this function just needs to support DAGs.
        """
        visited = set()
        stack = []

        def visit(node):
            visited.add(node)
            for neighbor in self.graph.get(node, []):
                if neighbor not in visited:
                    visit(neighbor)
            stack.append(node)

        for node in self.vertices:
            if node not in visited:
                visit(node)

        # Stack has reverse topological order (children first).
        # We want u before v (parents first).
        return stack[::-1]
