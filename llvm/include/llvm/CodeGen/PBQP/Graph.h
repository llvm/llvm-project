//===-------------------- Graph.h - PBQP Graph ------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// PBQP Graph class.
//
//===----------------------------------------------------------------------===//


#ifndef LLVM_CODEGEN_PBQP_GRAPH_H
#define LLVM_CODEGEN_PBQP_GRAPH_H

#include "Math.h"

#include <list>
#include <map>
#include <llvm/ADT/ilist.h>

namespace PBQP {

  /// PBQP Graph class.
  /// Instances of this class describe PBQP problems.
  class Graph {
  private:

    // ----- TYPEDEFS -----
    class NodeEntry;
    class EdgeEntry;

    typedef llvm::ilist<NodeEntry> NodeList;
    typedef llvm::ilist<EdgeEntry> EdgeList;

  public:

    typedef NodeEntry* NodeItr;
    typedef const NodeEntry* ConstNodeItr;

    typedef EdgeEntry* EdgeItr;
    typedef const EdgeEntry* ConstEdgeItr;

  private:

    typedef std::list<EdgeItr> AdjEdgeList;
  
  public:

    typedef AdjEdgeList::iterator AdjEdgeItr;

  private:

    class NodeEntry : public llvm::ilist_node<NodeEntry> {
      friend struct llvm::ilist_sentinel_traits<NodeEntry>;
    private:
      Vector costs;      
      AdjEdgeList adjEdges;
      unsigned degree;
      void *data;
      NodeEntry() : costs(0, 0) {}
    public:
      NodeEntry(const Vector &costs) : costs(costs), degree(0) {}
      Vector& getCosts() { return costs; }
      const Vector& getCosts() const { return costs; }
      unsigned getDegree() const { return degree; }
      AdjEdgeItr edgesBegin() { return adjEdges.begin(); }
      AdjEdgeItr edgesEnd() { return adjEdges.end(); }
      AdjEdgeItr addEdge(EdgeItr e) {
        ++degree;
        return adjEdges.insert(adjEdges.end(), e);
      }
      void removeEdge(AdjEdgeItr ae) {
        --degree;
        adjEdges.erase(ae);
      }
      void setData(void *data) { this->data = data; }
      void* getData() { return data; }
    };

    class EdgeEntry : public llvm::ilist_node<EdgeEntry> {
      friend struct llvm::ilist_sentinel_traits<EdgeEntry>;
    private:
      NodeItr node1, node2;
      Matrix costs;
      AdjEdgeItr node1AEItr, node2AEItr;
      void *data;
      EdgeEntry() : costs(0, 0, 0) {}
    public:
      EdgeEntry(NodeItr node1, NodeItr node2, const Matrix &costs)
        : node1(node1), node2(node2), costs(costs) {}
      NodeItr getNode1() const { return node1; }
      NodeItr getNode2() const { return node2; }
      Matrix& getCosts() { return costs; }
      const Matrix& getCosts() const { return costs; }
      void setNode1AEItr(AdjEdgeItr ae) { node1AEItr = ae; }
      AdjEdgeItr getNode1AEItr() { return node1AEItr; }
      void setNode2AEItr(AdjEdgeItr ae) { node2AEItr = ae; }
      AdjEdgeItr getNode2AEItr() { return node2AEItr; }
      void setData(void *data) { this->data = data; }
      void *getData() { return data; }
    };

    // ----- MEMBERS -----

    NodeList nodes;
    unsigned numNodes;

    EdgeList edges;
    unsigned numEdges;

    // ----- INTERNAL METHODS -----

    NodeEntry& getNode(NodeItr nItr) { return *nItr; }
    const NodeEntry& getNode(ConstNodeItr nItr) const { return *nItr; }

    EdgeEntry& getEdge(EdgeItr eItr) { return *eItr; }
    const EdgeEntry& getEdge(ConstEdgeItr eItr) const { return *eItr; }

    NodeItr addConstructedNode(const NodeEntry &n) {
      ++numNodes;
      return nodes.insert(nodes.end(), n);
    }

    EdgeItr addConstructedEdge(const EdgeEntry &e) {
      assert(findEdge(e.getNode1(), e.getNode2()) == edges.end() &&
             "Attempt to add duplicate edge.");
      ++numEdges;
      EdgeItr edgeItr = edges.insert(edges.end(), e);
      EdgeEntry &ne = getEdge(edgeItr);
      NodeEntry &n1 = getNode(ne.getNode1());
      NodeEntry &n2 = getNode(ne.getNode2());
      // Sanity check on matrix dimensions:
      assert((n1.getCosts().getLength() == ne.getCosts().getRows()) &&
             (n2.getCosts().getLength() == ne.getCosts().getCols()) &&
             "Edge cost dimensions do not match node costs dimensions.");
      ne.setNode1AEItr(n1.addEdge(edgeItr));
      ne.setNode2AEItr(n2.addEdge(edgeItr));
      return edgeItr;
    }

    inline void copyFrom(const Graph &other);
  public:

    /// \brief Construct an empty PBQP graph.
    Graph() : numNodes(0), numEdges(0) {}

    /// \brief Copy construct this graph from "other". Note: Does not copy node
    ///        and edge data, only graph structure and costs.
    /// @param other Source graph to copy from.
    Graph(const Graph &other) : numNodes(0), numEdges(0) {
      copyFrom(other);
    }

    /// \brief Make this graph a copy of "other". Note: Does not copy node and
    ///        edge data, only graph structure and costs.
    /// @param other The graph to copy from.
    /// @return A reference to this graph.
    ///
    /// This will clear the current graph, erasing any nodes and edges added,
    /// before copying from other.
    Graph& operator=(const Graph &other) {
      clear();      
      copyFrom(other);
      return *this;
    }

    /// \brief Add a node with the given costs.
    /// @param costs Cost vector for the new node.
    /// @return Node iterator for the added node.
    NodeItr addNode(const Vector &costs) {
      return addConstructedNode(NodeEntry(costs));
    }

    /// \brief Add an edge between the given nodes with the given costs.
    /// @param n1Itr First node.
    /// @param n2Itr Second node.
    /// @return Edge iterator for the added edge.
    EdgeItr addEdge(Graph::NodeItr n1Itr, Graph::NodeItr n2Itr,
                    const Matrix &costs) {
      assert(getNodeCosts(n1Itr).getLength() == costs.getRows() &&
             getNodeCosts(n2Itr).getLength() == costs.getCols() &&
             "Matrix dimensions mismatch.");
      return addConstructedEdge(EdgeEntry(n1Itr, n2Itr, costs)); 
    }

    /// \brief Get the number of nodes in the graph.
    /// @return Number of nodes in the graph.
    unsigned getNumNodes() const { return numNodes; }

    /// \brief Get the number of edges in the graph.
    /// @return Number of edges in the graph.
    unsigned getNumEdges() const { return numEdges; }

    /// \brief Get a node's cost vector.
    /// @param nItr Node iterator.
    /// @return Node cost vector.
    Vector& getNodeCosts(NodeItr nItr) { return getNode(nItr).getCosts(); }

    /// \brief Get a node's cost vector (const version).
    /// @param nItr Node iterator.
    /// @return Node cost vector.
    const Vector& getNodeCosts(ConstNodeItr nItr) const {
      return getNode(nItr).getCosts();
    }

    /// \brief Set a node's data pointer.
    /// @param nItr Node iterator.
    /// @param data Pointer to node data.
    ///
    /// Typically used by a PBQP solver to attach data to aid in solution.
    void setNodeData(NodeItr nItr, void *data) { getNode(nItr).setData(data); }

    /// \brief Get the node's data pointer.
    /// @param nItr Node iterator.
    /// @return Pointer to node data.
    void* getNodeData(NodeItr nItr) { return getNode(nItr).getData(); }
    
    /// \brief Get an edge's cost matrix.
    /// @param eItr Edge iterator.
    /// @return Edge cost matrix.
    Matrix& getEdgeCosts(EdgeItr eItr) { return getEdge(eItr).getCosts(); }

    /// \brief Get an edge's cost matrix (const version).
    /// @param eItr Edge iterator.
    /// @return Edge cost matrix.
    const Matrix& getEdgeCosts(ConstEdgeItr eItr) const {
      return getEdge(eItr).getCosts();
    }

    /// \brief Set an edge's data pointer.
    /// @param eItr Edge iterator.
    /// @param data Pointer to edge data.
    ///
    /// Typically used by a PBQP solver to attach data to aid in solution.
    void setEdgeData(EdgeItr eItr, void *data) { getEdge(eItr).setData(data); }

    /// \brief Get an edge's data pointer.
    /// @param eItr Edge iterator.
    /// @return Pointer to edge data. 
    void* getEdgeData(EdgeItr eItr) { return getEdge(eItr).getData(); }

    /// \brief Get a node's degree.
    /// @param nItr Node iterator.
    /// @return The degree of the node.
    unsigned getNodeDegree(NodeItr nItr) const {
      return getNode(nItr).getDegree();
    }

    /// \brief Begin iterator for node set.
    NodeItr nodesBegin() { return nodes.begin(); }

    /// \brief Begin const iterator for node set.
    ConstNodeItr nodesBegin() const { return nodes.begin(); }

    /// \brief End iterator for node set.
    NodeItr nodesEnd() { return nodes.end(); }

    /// \brief End const iterator for node set.
    ConstNodeItr nodesEnd() const { return nodes.end(); }

    /// \brief Begin iterator for edge set.
    EdgeItr edgesBegin() { return edges.begin(); }

    /// \brief End iterator for edge set.
    EdgeItr edgesEnd() { return edges.end(); }

    /// \brief Get begin iterator for adjacent edge set.
    /// @param nItr Node iterator.
    /// @return Begin iterator for the set of edges connected to the given node.
    AdjEdgeItr adjEdgesBegin(NodeItr nItr) {
      return getNode(nItr).edgesBegin();
    }

    /// \brief Get end iterator for adjacent edge set.
    /// @param nItr Node iterator.
    /// @return End iterator for the set of edges connected to the given node.
    AdjEdgeItr adjEdgesEnd(NodeItr nItr) {
      return getNode(nItr).edgesEnd();
    }

    /// \brief Get the first node connected to this edge.
    /// @param eItr Edge iterator.
    /// @return The first node connected to the given edge. 
    NodeItr getEdgeNode1(EdgeItr eItr) {
      return getEdge(eItr).getNode1();
    }

    /// \brief Get the second node connected to this edge.
    /// @param eItr Edge iterator.
    /// @return The second node connected to the given edge. 
    NodeItr getEdgeNode2(EdgeItr eItr) {
      return getEdge(eItr).getNode2();
    } 

    /// \brief Get the "other" node connected to this edge.
    /// @param eItr Edge iterator.
    /// @param nItr Node iterator for the "given" node.
    /// @return The iterator for the "other" node connected to this edge. 
    NodeItr getEdgeOtherNode(EdgeItr eItr, NodeItr nItr) {
      EdgeEntry &e = getEdge(eItr);
      if (e.getNode1() == nItr) {
        return e.getNode2();
      } // else
      return e.getNode1();
    }

    /// \brief Get the edge connecting two nodes.
    /// @param n1Itr First node iterator.
    /// @param n2Itr Second node iterator.
    /// @return An iterator for edge (n1Itr, n2Itr) if such an edge exists,
    ///         otherwise returns edgesEnd(). 
    EdgeItr findEdge(NodeItr n1Itr, NodeItr n2Itr) {
      for (AdjEdgeItr aeItr = adjEdgesBegin(n1Itr), aeEnd = adjEdgesEnd(n1Itr);
         aeItr != aeEnd; ++aeItr) {
        if ((getEdgeNode1(*aeItr) == n2Itr) ||
            (getEdgeNode2(*aeItr) == n2Itr)) {
          return *aeItr;
        }
      }
      return edges.end();
    }

    /// \brief Remove a node from the graph.
    /// @param nItr Node iterator.
    void removeNode(NodeItr nItr) {
      NodeEntry &n = getNode(nItr);
      for (AdjEdgeItr itr = n.edgesBegin(), end = n.edgesEnd(); itr != end;) {
        EdgeItr eItr = *itr;
        ++itr;
        removeEdge(eItr); 
      }
      nodes.erase(nItr);
      --numNodes;
    }

    /// \brief Remove an edge from the graph.
    /// @param eItr Edge iterator.
    void removeEdge(EdgeItr eItr) {
      EdgeEntry &e = getEdge(eItr);
      NodeEntry &n1 = getNode(e.getNode1());
      NodeEntry &n2 = getNode(e.getNode2());
      n1.removeEdge(e.getNode1AEItr());
      n2.removeEdge(e.getNode2AEItr());
      edges.erase(eItr);
      --numEdges;
    }

    /// \brief Remove all nodes and edges from the graph.
    void clear() {
      nodes.clear();
      edges.clear();
      numNodes = numEdges = 0;
    }

    /// \brief Dump a graph to an output stream.
    template <typename OStream>
    void dump(OStream &os) {
      os << getNumNodes() << " " << getNumEdges() << "\n";

      for (NodeItr nodeItr = nodesBegin(), nodeEnd = nodesEnd();
           nodeItr != nodeEnd; ++nodeItr) {
        const Vector& v = getNodeCosts(nodeItr);
        os << "\n" << v.getLength() << "\n";
        assert(v.getLength() != 0 && "Empty vector in graph.");
        os << v[0];
        for (unsigned i = 1; i < v.getLength(); ++i) {
          os << " " << v[i];
        }
        os << "\n";
      }

      for (EdgeItr edgeItr = edgesBegin(), edgeEnd = edgesEnd();
           edgeItr != edgeEnd; ++edgeItr) {
        unsigned n1 = std::distance(nodesBegin(), getEdgeNode1(edgeItr));
        unsigned n2 = std::distance(nodesBegin(), getEdgeNode2(edgeItr));
        assert(n1 != n2 && "PBQP graphs shound not have self-edges.");
        const Matrix& m = getEdgeCosts(edgeItr);
        os << "\n" << n1 << " " << n2 << "\n"
           << m.getRows() << " " << m.getCols() << "\n";
        assert(m.getRows() != 0 && "No rows in matrix.");
        assert(m.getCols() != 0 && "No cols in matrix.");
        for (unsigned i = 0; i < m.getRows(); ++i) {
          os << m[i][0];
          for (unsigned j = 1; j < m.getCols(); ++j) {
            os << " " << m[i][j];
          }
          os << "\n";
        }
      }
    }

    /// \brief Print a representation of this graph in DOT format.
    /// @param os Output stream to print on.
    template <typename OStream>
    void printDot(OStream &os) {
    
      os << "graph {\n";

      for (NodeItr nodeItr = nodesBegin(), nodeEnd = nodesEnd();
           nodeItr != nodeEnd; ++nodeItr) {

        os << "  node" << nodeItr << " [ label=\""
           << nodeItr << ": " << getNodeCosts(nodeItr) << "\" ]\n";
      }

      os << "  edge [ len=" << getNumNodes() << " ]\n";

      for (EdgeItr edgeItr = edgesBegin(), edgeEnd = edgesEnd();
           edgeItr != edgeEnd; ++edgeItr) {

        os << "  node" << getEdgeNode1(edgeItr)
           << " -- node" << getEdgeNode2(edgeItr)
           << " [ label=\"";

        const Matrix &edgeCosts = getEdgeCosts(edgeItr);

        for (unsigned i = 0; i < edgeCosts.getRows(); ++i) {
          os << edgeCosts.getRowAsVector(i) << "\\n";
        }
        os << "\" ]\n";
      }
      os << "}\n";
    }

  };

  class NodeItrComparator {
  public:
    bool operator()(Graph::NodeItr n1, Graph::NodeItr n2) const {
      return &*n1 < &*n2;
    }

    bool operator()(Graph::ConstNodeItr n1, Graph::ConstNodeItr n2) const {
      return &*n1 < &*n2;
    }
  };

  class EdgeItrCompartor {
  public:
    bool operator()(Graph::EdgeItr e1, Graph::EdgeItr e2) const {
      return &*e1 < &*e2;
    }

    bool operator()(Graph::ConstEdgeItr e1, Graph::ConstEdgeItr e2) const {
      return &*e1 < &*e2;
    }
  };

  void Graph::copyFrom(const Graph &other) {
    std::map<Graph::ConstNodeItr, Graph::NodeItr,
             NodeItrComparator> nodeMap;

     for (Graph::ConstNodeItr nItr = other.nodesBegin(),
                             nEnd = other.nodesEnd();
         nItr != nEnd; ++nItr) {
      nodeMap[nItr] = addNode(other.getNodeCosts(nItr));
    }
      
  }

}

#endif // LLVM_CODEGEN_PBQP_GRAPH_HPP
