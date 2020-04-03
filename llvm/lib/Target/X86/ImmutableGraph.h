//==========-- ImmutableGraph.h - A fast DAG implementation ---------=========//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Description: ImmutableGraph is a fast DAG implementation that cannot be
/// modified, except by creating a new ImmutableGraph. ImmutableGraph is
/// implemented as two arrays: one containing nodes, and one containing edges.
/// The advantages to this implementation are two-fold:
/// 1. Iteration and traversal operations should experience terrific caching
///    performance.
/// 2. Set representations and operations on nodes and edges become
///    extraordinarily efficient. For instance, a set of edges is implemented as
///    a bit vector, wherein each bit corresponds to one edge in the edge
///    array. This implies a lower bound of 64x spacial improvement over, e.g.,
///    an llvm::DenseSet or llvm::SmallSet. It also means that
///    insert/erase/contains operations complete in negligible constant time:
///    insert and erase require one load and one store, and contains requires
///    just one load.
///
//===----------------------------------------------------------------------===//

#ifndef IMMUTABLEGRAPH_H
#define IMMUTABLEGRAPH_H

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <iterator>
#include <utility>
#include <vector>

namespace llvm {

template <typename _NodeValueT, typename _EdgeValueT, typename _SizeT = int>
class ImmutableGraph {
  using Traits = GraphTraits<ImmutableGraph<_NodeValueT, _EdgeValueT> *>;
  template <typename> friend class ImmutableGraphBuilder;

public:
  using NodeValueT = _NodeValueT;
  using EdgeValueT = _EdgeValueT;
  using size_type = _SizeT;
  class Node;
  class Edge {
    friend class ImmutableGraph;
    template <typename> friend class ImmutableGraphBuilder;
    friend Traits;

    Node *__dest;
    EdgeValueT __value;

  public:
    EdgeValueT &value() { return __value; }
  };
  class Node {
    friend class ImmutableGraph;
    template <typename> friend class ImmutableGraphBuilder;
    friend Traits;

    Edge *__edges;
    NodeValueT __value;

  public:
    NodeValueT &value() { return __value; }
  };

protected:
  ImmutableGraph(Node *Nodes, size_type NodesSize, Edge *Edges,
                 size_type EdgesSize)
      : __nodes{Nodes}, __nodes_size{NodesSize}, __edges{Edges},
        __edges_size{EdgesSize} {}
  ImmutableGraph(const ImmutableGraph &) = delete;
  ImmutableGraph(ImmutableGraph &&) = delete;
  ImmutableGraph &operator=(const ImmutableGraph &) = delete;
  ImmutableGraph &operator=(ImmutableGraph &&) = delete;

public:
  ~ImmutableGraph() {
    delete[] __edges;
    delete[] __nodes;
  }

  Node *nodes_begin() const { return __nodes; }
  Node *nodes_end() const { return __nodes + __nodes_size; }
  Edge *edges_begin() const { return __edges; }
  Edge *edges_end() const { return __edges + __edges_size; }
  size_type nodes_size() const { return __nodes_size; }
  size_type edges_size() const { return __edges_size; }
  bool empty() const { return __nodes_size == 0; }

  class NodeSet {
    friend class iterator;

    const ImmutableGraph &__g;
    BitVector __v;

  public:
    NodeSet(const ImmutableGraph &G, bool ContainsAll = false)
        : __g{G}, __v{static_cast<unsigned>(__g.nodes_size()), ContainsAll} {}
    bool insert(Node *N) {
      size_type Idx = std::distance(__g.nodes_begin(), N);
      bool AlreadyExists = __v.test(Idx);
      __v.set(Idx);
      return !AlreadyExists;
    }
    void erase(Node *N) {
      size_type Idx = std::distance(__g.nodes_begin(), N);
      __v.reset(Idx);
    }
    bool contains(Node *N) const {
      size_type Idx = std::distance(__g.nodes_begin(), N);
      return __v.test(Idx);
    }
    void clear() { __v.reset(); }
    size_type empty() const { return __v.none(); }
    /// Return the number of elements in the set
    size_type count() const { return __v.count(); }
    /// Return the size of the set's domain
    size_type size() const { return __v.size(); }
    /// Set union
    NodeSet &operator|=(const NodeSet &RHS) {
      assert(&this->__g == &RHS.__g);
      __v |= RHS.__v;
      return *this;
    }
    /// Set intersection
    NodeSet &operator&=(const NodeSet &RHS) {
      assert(&this->__g == &RHS.__g);
      __v &= RHS.__v;
      return *this;
    }
    /// Set disjoint union
    NodeSet &operator^=(const NodeSet &RHS) {
      assert(&this->__g == &RHS.__g);
      __v ^= RHS.__v;
      return *this;
    }

    using index_iterator = typename BitVector::const_set_bits_iterator;
    index_iterator index_begin() const { return __v.set_bits_begin(); }
    index_iterator index_end() const { return __v.set_bits_end(); }
    void set(size_type Idx) { __v.set(Idx); }
    void reset(size_type Idx) { __v.reset(Idx); }

    class iterator {
      const NodeSet &__set;
      size_type __current;

      void advance() {
        assert(__current != -1);
        __current = __set.__v.find_next(__current);
      }

    public:
      iterator(const NodeSet &Set, size_type Begin)
          : __set{Set}, __current{Begin} {}
      iterator operator++(int) {
        iterator Tmp = *this;
        advance();
        return Tmp;
      }
      iterator &operator++() {
        advance();
        return *this;
      }
      Node *operator*() const {
        assert(__current != -1);
        return __set.__g.nodes_begin() + __current;
      }
      bool operator==(const iterator &other) const {
        assert(&this->__set == &other.__set);
        return this->__current == other.__current;
      }
      bool operator!=(const iterator &other) const { return !(*this == other); }
    };

    iterator begin() const { return iterator{*this, __v.find_first()}; }
    iterator end() const { return iterator{*this, -1}; }
  };

  class EdgeSet {
    const ImmutableGraph &__g;
    BitVector __v;

  public:
    EdgeSet(const ImmutableGraph &G, bool ContainsAll = false)
        : __g{G}, __v{static_cast<unsigned>(__g.edges_size()), ContainsAll} {}
    bool insert(Edge *E) {
      size_type Idx = std::distance(__g.edges_begin(), E);
      bool AlreadyExists = __v.test(Idx);
      __v.set(Idx);
      return !AlreadyExists;
    }
    void erase(Edge *E) {
      size_type Idx = std::distance(__g.edges_begin(), E);
      __v.reset(Idx);
    }
    bool contains(Edge *E) const {
      size_type Idx = std::distance(__g.edges_begin(), E);
      return __v.test(Idx);
    }
    void clear() { __v.reset(); }
    bool empty() const { return __v.none(); }
    /// Return the number of elements in the set
    size_type count() const { return __v.count(); }
    /// Return the size of the set's domain
    size_type size() const { return __v.size(); }
    /// Set union
    EdgeSet &operator|=(const EdgeSet &RHS) {
      assert(&this->__g == &RHS.__g);
      __v |= RHS.__v;
      return *this;
    }
    /// Set intersection
    EdgeSet &operator&=(const EdgeSet &RHS) {
      assert(&this->__g == &RHS.__g);
      __v &= RHS.__v;
      return *this;
    }
    /// Set disjoint union
    EdgeSet &operator^=(const EdgeSet &RHS) {
      assert(&this->__g == &RHS.__g);
      __v ^= RHS.__v;
      return *this;
    }

    using index_iterator = typename BitVector::const_set_bits_iterator;
    index_iterator index_begin() const { return __v.set_bits_begin(); }
    index_iterator index_end() const { return __v.set_bits_end(); }
    void set(size_type Idx) { __v.set(Idx); }
    void reset(size_type Idx) { __v.reset(Idx); }

    class iterator {
      const EdgeSet &__set;
      size_type __current;

      void advance() {
        assert(__current != -1);
        __current = __set.__v.find_next(__current);
      }

    public:
      iterator(const EdgeSet &Set, size_type Begin)
          : __set{Set}, __current{Begin} {}
      iterator operator++(int) {
        iterator Tmp = *this;
        advance();
        return Tmp;
      }
      iterator &operator++() {
        advance();
        return *this;
      }
      Edge *operator*() const {
        assert(__current != -1);
        return __set.__g.edges_begin() + __current;
      }
      bool operator==(const iterator &other) const {
        assert(&this->__set == &other.__set);
        return this->__current == other.__current;
      }
      bool operator!=(const iterator &other) const { return !(*this == other); }
    };

    iterator begin() const { return iterator{*this, __v.find_first()}; }
    iterator end() const { return iterator{*this, -1}; }
  };

private:
  Node *__nodes;
  size_type __nodes_size;
  Edge *__edges;
  size_type __edges_size;
};

template <typename GraphT> class ImmutableGraphBuilder {
  using NodeValueT = typename GraphT::NodeValueT;
  using EdgeValueT = typename GraphT::EdgeValueT;
  static_assert(
      std::is_base_of<ImmutableGraph<NodeValueT, EdgeValueT>, GraphT>::value,
      "Template argument to ImmutableGraphBuilder must derive from "
      "ImmutableGraph<>");
  using size_type = typename GraphT::size_type;
  using NodeSet = typename GraphT::NodeSet;
  using Node = typename GraphT::Node;
  using EdgeSet = typename GraphT::EdgeSet;
  using Edge = typename GraphT::Edge;
  using BuilderEdge = std::pair<EdgeValueT, size_type>;
  using EdgeList = std::vector<BuilderEdge>;
  using BuilderVertex = std::pair<NodeValueT, EdgeList>;
  using VertexVec = std::vector<BuilderVertex>;

public:
  using NodeRef = size_type;

  NodeRef addVertex(const NodeValueT &V) {
    auto I = __adj_list.emplace(__adj_list.end(), V, EdgeList{});
    return std::distance(__adj_list.begin(), I);
  }

  void addEdge(const EdgeValueT &E, NodeRef From, NodeRef To) {
    __adj_list[From].second.emplace_back(E, To);
  }

  bool empty() const { return __adj_list.empty(); }

  template <typename... ArgT> GraphT *get(ArgT &&... Args) {
    size_type VertexSize = __adj_list.size(), EdgeSize = 0;
    for (const auto &V : __adj_list) {
      EdgeSize += V.second.size();
    }
    auto *VertexArray = new Node[VertexSize + 1 /* terminator node */];
    auto *EdgeArray = new Edge[EdgeSize];
    size_type VI = 0, EI = 0;
    for (; VI < static_cast<size_type>(__adj_list.size()); ++VI) {
      VertexArray[VI].__value = std::move(__adj_list[VI].first);
      VertexArray[VI].__edges = &EdgeArray[EI];
      auto NumEdges = static_cast<size_type>(__adj_list[VI].second.size());
      if (NumEdges > 0) {
        for (size_type VEI = 0; VEI < NumEdges; ++VEI, ++EI) {
          auto &E = __adj_list[VI].second[VEI];
          EdgeArray[EI].__value = std::move(E.first);
          EdgeArray[EI].__dest = VertexArray + E.second;
        }
      }
    }
    assert(VI == VertexSize && EI == EdgeSize && "Gadget graph malformed");
    VertexArray[VI].__edges = EdgeArray + EdgeSize; // terminator node
    return new GraphT{VertexArray, VertexSize, EdgeArray, EdgeSize,
                      std::forward<ArgT>(Args)...};
  }

  template <typename... ArgT>
  static GraphT *trim(const GraphT &G, const NodeSet &TrimNodes,
                      const EdgeSet &TrimEdges, ArgT &&... Args) {
    size_type NewVertexSize = TrimNodes.size() - TrimNodes.count();
    size_type NewEdgeSize = TrimEdges.size() - TrimEdges.count();
    auto *NewVertexArray = new Node[NewVertexSize + 1 /* terminator node */];
    auto *NewEdgeArray = new Edge[NewEdgeSize];
    size_type TrimmedNodesSoFar = 0,
              *TrimmedNodes = new size_type[TrimNodes.size()];
    for (size_type I = 0; I < TrimNodes.size(); ++I) {
      TrimmedNodes[I] = TrimmedNodesSoFar;
      if (TrimNodes.contains(G.nodes_begin() + I))
        ++TrimmedNodesSoFar;
    }
    size_type VertexI = 0, EdgeI = 0;
    for (Node *NI = G.nodes_begin(), *NE = G.nodes_end(); NI != NE; ++NI) {
      if (TrimNodes.contains(NI))
        continue;
      size_type NewNumEdges =
          static_cast<int>((NI + 1)->__edges - NI->__edges) > 0
              ? std::count_if(
                    NI->__edges, (NI + 1)->__edges,
                    [&TrimEdges](Edge &E) { return !TrimEdges.contains(&E); })
              : 0;
      NewVertexArray[VertexI].__value = NI->__value;
      NewVertexArray[VertexI].__edges = &NewEdgeArray[EdgeI];
      if (NewNumEdges > 0) {
        for (Edge *EI = NI->__edges, *EE = (NI + 1)->__edges; EI != EE; ++EI) {
          if (TrimEdges.contains(EI))
            continue;
          NewEdgeArray[EdgeI].__value = EI->__value;
          size_type DestIdx = std::distance(G.nodes_begin(), EI->__dest);
          size_type NewIdx = DestIdx - TrimmedNodes[DestIdx];
          assert(NewIdx < NewVertexSize);
          NewEdgeArray[EdgeI].__dest = NewVertexArray + NewIdx;
          ++EdgeI;
        }
      }
      ++VertexI;
    }
    delete[] TrimmedNodes;
    assert(VertexI == NewVertexSize && EdgeI == NewEdgeSize &&
           "Gadget graph malformed");
    NewVertexArray[VertexI].__edges = NewEdgeArray + NewEdgeSize;
    return new GraphT{NewVertexArray, NewVertexSize, NewEdgeArray, NewEdgeSize,
                      std::forward<ArgT>(Args)...};
  }

private:
  VertexVec __adj_list;
};

template <typename NodeValueT, typename EdgeValueT>
struct GraphTraits<ImmutableGraph<NodeValueT, EdgeValueT> *> {
  using GraphT = ImmutableGraph<NodeValueT, EdgeValueT>;
  using NodeRef = typename GraphT::Node *;
  using EdgeRef = typename GraphT::Edge &;

  static NodeRef edge_dest(EdgeRef E) { return E.__dest; }
  using ChildIteratorType =
      mapped_iterator<typename GraphT::Edge *, decltype(&edge_dest)>;

  static NodeRef getEntryNode(GraphT *G) { return G->nodes_begin(); }
  static ChildIteratorType child_begin(NodeRef N) {
    return {N->__edges, &edge_dest};
  }
  static ChildIteratorType child_end(NodeRef N) {
    return {(N + 1)->__edges, &edge_dest};
  }

  static NodeRef getNode(typename GraphT::Node &N) { return NodeRef{&N}; }
  using nodes_iterator =
      mapped_iterator<typename GraphT::Node *, decltype(&getNode)>;
  static nodes_iterator nodes_begin(GraphT *G) {
    return {G->nodes_begin(), &getNode};
  }
  static nodes_iterator nodes_end(GraphT *G) {
    return {G->nodes_end(), &getNode};
  }

  using ChildEdgeIteratorType = typename GraphT::Edge *;

  static ChildEdgeIteratorType child_edge_begin(NodeRef N) {
    return N->__edges;
  }
  static ChildEdgeIteratorType child_edge_end(NodeRef N) {
    return (N + 1)->__edges;
  }
  static typename GraphT::size_type size(GraphT *G) { return G->nodes_size(); }
};

} // end namespace llvm

#endif // IMMUTABLEGRAPH_H
