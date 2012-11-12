//===----- llvm/unittest/ADT/SCCIteratorTest.cpp - SCCIterator tests ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <limits.h>
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/SCCIterator.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace llvm {

/// Graph<N> - A graph with N nodes.  Note that N can be at most 8.
template <unsigned N>
class Graph {
private:
  // Disable copying.
  Graph(const Graph&);
  Graph& operator=(const Graph&);

  static void ValidateIndex(unsigned Idx) {
    assert(Idx < N && "Invalid node index!");
  }
public:

  /// NodeSubset - A subset of the graph's nodes.
  class NodeSubset {
    typedef unsigned char BitVector; // Where the limitation N <= 8 comes from.
    BitVector Elements;
    NodeSubset(BitVector e) : Elements(e) {}
  public:
    /// NodeSubset - Default constructor, creates an empty subset.
    NodeSubset() : Elements(0) {
      assert(N <= sizeof(BitVector)*CHAR_BIT && "Graph too big!");
    }
    /// NodeSubset - Copy constructor.
    NodeSubset(const NodeSubset &other) : Elements(other.Elements) {}

    /// Comparison operators.
    bool operator==(const NodeSubset &other) const {
      return other.Elements == this->Elements;
    }
    bool operator!=(const NodeSubset &other) const {
      return !(*this == other);
    }

    /// AddNode - Add the node with the given index to the subset.
    void AddNode(unsigned Idx) {
      ValidateIndex(Idx);
      Elements |= 1U << Idx;
    }

    /// DeleteNode - Remove the node with the given index from the subset.
    void DeleteNode(unsigned Idx) {
      ValidateIndex(Idx);
      Elements &= ~(1U << Idx);
    }

    /// count - Return true if the node with the given index is in the subset.
    bool count(unsigned Idx) {
      ValidateIndex(Idx);
      return (Elements & (1U << Idx)) != 0;
    }

    /// isEmpty - Return true if this is the empty set.
    bool isEmpty() const {
      return Elements == 0;
    }

    /// isSubsetOf - Return true if this set is a subset of the given one.
    bool isSubsetOf(const NodeSubset &other) const {
      return (this->Elements | other.Elements) == other.Elements;
    }

    /// Complement - Return the complement of this subset.
    NodeSubset Complement() const {
      return ~(unsigned)this->Elements & ((1U << N) - 1);
    }

    /// Join - Return the union of this subset and the given one.
    NodeSubset Join(const NodeSubset &other) const {
      return this->Elements | other.Elements;
    }

    /// Meet - Return the intersection of this subset and the given one.
    NodeSubset Meet(const NodeSubset &other) const {
      return this->Elements & other.Elements;
    }
  };

  /// NodeType - Node index and set of children of the node.
  typedef std::pair<unsigned, NodeSubset> NodeType;

private:
  /// Nodes - The list of nodes for this graph.
  NodeType Nodes[N];
public:

  /// Graph - Default constructor.  Creates an empty graph.
  Graph() {
    // Let each node know which node it is.  This allows us to find the start of
    // the Nodes array given a pointer to any element of it.
    for (unsigned i = 0; i != N; ++i)
      Nodes[i].first = i;
  }

  /// AddEdge - Add an edge from the node with index FromIdx to the node with
  /// index ToIdx.
  void AddEdge(unsigned FromIdx, unsigned ToIdx) {
    ValidateIndex(FromIdx);
    Nodes[FromIdx].second.AddNode(ToIdx);
  }

  /// DeleteEdge - Remove the edge (if any) from the node with index FromIdx to
  /// the node with index ToIdx.
  void DeleteEdge(unsigned FromIdx, unsigned ToIdx) {
    ValidateIndex(FromIdx);
    Nodes[FromIdx].second.DeleteNode(ToIdx);
  }

  /// AccessNode - Get a pointer to the node with the given index.
  NodeType *AccessNode(unsigned Idx) const {
    ValidateIndex(Idx);
    // The constant cast is needed when working with GraphTraits, which insists
    // on taking a constant Graph.
    return const_cast<NodeType *>(&Nodes[Idx]);
  }

  /// NodesReachableFrom - Return the set of all nodes reachable from the given
  /// node.
  NodeSubset NodesReachableFrom(unsigned Idx) const {
    // This algorithm doesn't scale, but that doesn't matter given the small
    // size of our graphs.
    NodeSubset Reachable;

    // The initial node is reachable.
    Reachable.AddNode(Idx);
    do {
      NodeSubset Previous(Reachable);

      // Add in all nodes which are children of a reachable node.
      for (unsigned i = 0; i != N; ++i)
        if (Previous.count(i))
          Reachable = Reachable.Join(Nodes[i].second);

      // If nothing changed then we have found all reachable nodes.
      if (Reachable == Previous)
        return Reachable;

      // Rinse and repeat.
    } while (1);
  }

  /// ChildIterator - Visit all children of a node.
  class ChildIterator {
    friend class Graph;

    /// FirstNode - Pointer to first node in the graph's Nodes array.
    NodeType *FirstNode;
    /// Children - Set of nodes which are children of this one and that haven't
    /// yet been visited.
    NodeSubset Children;

    ChildIterator(); // Disable default constructor.
  protected:
    ChildIterator(NodeType *F, NodeSubset C) : FirstNode(F), Children(C) {}

  public:
    /// ChildIterator - Copy constructor.
    ChildIterator(const ChildIterator& other) : FirstNode(other.FirstNode),
      Children(other.Children) {}

    /// Comparison operators.
    bool operator==(const ChildIterator &other) const {
      return other.FirstNode == this->FirstNode &&
        other.Children == this->Children;
    }
    bool operator!=(const ChildIterator &other) const {
      return !(*this == other);
    }

    /// Prefix increment operator.
    ChildIterator& operator++() {
      // Find the next unvisited child node.
      for (unsigned i = 0; i != N; ++i)
        if (Children.count(i)) {
          // Remove that child - it has been visited.  This is the increment!
          Children.DeleteNode(i);
          return *this;
        }
      assert(false && "Incrementing end iterator!");
      return *this; // Avoid compiler warnings.
    }

    /// Postfix increment operator.
    ChildIterator operator++(int) {
      ChildIterator Result(*this);
      ++(*this);
      return Result;
    }

    /// Dereference operator.
    NodeType *operator*() {
      // Find the next unvisited child node.
      for (unsigned i = 0; i != N; ++i)
        if (Children.count(i))
          // Return a pointer to it.
          return FirstNode + i;
      assert(false && "Dereferencing end iterator!");
      return 0; // Avoid compiler warning.
    }
  };

  /// child_begin - Return an iterator pointing to the first child of the given
  /// node.
  static ChildIterator child_begin(NodeType *Parent) {
    return ChildIterator(Parent - Parent->first, Parent->second);
  }

  /// child_end - Return the end iterator for children of the given node.
  static ChildIterator child_end(NodeType *Parent) {
    return ChildIterator(Parent - Parent->first, NodeSubset());
  }
};

template <unsigned N>
struct GraphTraits<Graph<N> > {
  typedef typename Graph<N>::NodeType NodeType;
  typedef typename Graph<N>::ChildIterator ChildIteratorType;

 static inline NodeType *getEntryNode(const Graph<N> &G) { return G.AccessNode(0); }
 static inline ChildIteratorType child_begin(NodeType *Node) {
   return Graph<N>::child_begin(Node);
 }
 static inline ChildIteratorType child_end(NodeType *Node) {
   return Graph<N>::child_end(Node);
 }
};

TEST(SCCIteratorTest, AllSmallGraphs) {
  // Test SCC computation against every graph with NUM_NODES nodes or less.
  // Since SCC considers every node to have an implicit self-edge, we only
  // create graphs for which every node has a self-edge.
#define NUM_NODES 4
#define NUM_GRAPHS (NUM_NODES * (NUM_NODES - 1))
  typedef Graph<NUM_NODES> GT;

  /// Enumerate all graphs using NUM_GRAPHS bits.
  assert(NUM_GRAPHS < sizeof(unsigned) * CHAR_BIT && "Too many graphs!");
  for (unsigned GraphDescriptor = 0; GraphDescriptor < (1U << NUM_GRAPHS);
       ++GraphDescriptor) {
    GT G;

    // Add edges as specified by the descriptor.
    unsigned DescriptorCopy = GraphDescriptor;
    for (unsigned i = 0; i != NUM_NODES; ++i)
      for (unsigned j = 0; j != NUM_NODES; ++j) {
        // Always add a self-edge.
        if (i == j) {
          G.AddEdge(i, j);
          continue;
        }
        if (DescriptorCopy & 1)
          G.AddEdge(i, j);
        DescriptorCopy >>= 1;
      }

    // Test the SCC logic on this graph.

    /// NodesInSomeSCC - Those nodes which are in some SCC.
    GT::NodeSubset NodesInSomeSCC;

    for (scc_iterator<GT> I = scc_begin(G), E = scc_end(G); I != E; ++I) {
      std::vector<GT::NodeType*> &SCC = *I;

      // Get the nodes in this SCC as a NodeSubset rather than a vector.
      GT::NodeSubset NodesInThisSCC;
      for (unsigned i = 0, e = SCC.size(); i != e; ++i)
        NodesInThisSCC.AddNode(SCC[i]->first);

      // There should be at least one node in every SCC.
      EXPECT_FALSE(NodesInThisSCC.isEmpty());

      // Check that every node in the SCC is reachable from every other node in
      // the SCC.
      for (unsigned i = 0; i != NUM_NODES; ++i)
        if (NodesInThisSCC.count(i))
          EXPECT_TRUE(NodesInThisSCC.isSubsetOf(G.NodesReachableFrom(i)));

      // OK, now that we now that every node in the SCC is reachable from every
      // other, this means that the set of nodes reachable from any node in the
      // SCC is the same as the set of nodes reachable from every node in the
      // SCC.  Check that for every node N not in the SCC but reachable from the
      // SCC, no element of the SCC is reachable from N.
      for (unsigned i = 0; i != NUM_NODES; ++i)
        if (NodesInThisSCC.count(i)) {
          GT::NodeSubset NodesReachableFromSCC = G.NodesReachableFrom(i);
          GT::NodeSubset ReachableButNotInSCC =
            NodesReachableFromSCC.Meet(NodesInThisSCC.Complement());

          for (unsigned j = 0; j != NUM_NODES; ++j)
            if (ReachableButNotInSCC.count(j))
              EXPECT_TRUE(G.NodesReachableFrom(j).Meet(NodesInThisSCC).isEmpty());

          // The result must be the same for all other nodes in this SCC, so
          // there is no point in checking them.
          break;
        }

      // This is indeed a SCC: a maximal set of nodes for which each node is
      // reachable from every other.

      // Check that we didn't already see this SCC.
      EXPECT_TRUE(NodesInSomeSCC.Meet(NodesInThisSCC).isEmpty());

      NodesInSomeSCC = NodesInSomeSCC.Join(NodesInThisSCC);

      // Check a property that is specific to the LLVM SCC iterator and
      // guaranteed by it: if a node in SCC S1 has an edge to a node in
      // SCC S2, then S1 is visited *after* S2.  This means that the set
      // of nodes reachable from this SCC must be contained either in the
      // union of this SCC and all previously visited SCC's.

      for (unsigned i = 0; i != NUM_NODES; ++i)
        if (NodesInThisSCC.count(i)) {
          GT::NodeSubset NodesReachableFromSCC = G.NodesReachableFrom(i);
          EXPECT_TRUE(NodesReachableFromSCC.isSubsetOf(NodesInSomeSCC));
          // The result must be the same for all other nodes in this SCC, so
          // there is no point in checking them.
          break;
        }
    }

    // Finally, check that the nodes in some SCC are exactly those that are
    // reachable from the initial node.
    EXPECT_EQ(NodesInSomeSCC, G.NodesReachableFrom(0));
  }
}

}
