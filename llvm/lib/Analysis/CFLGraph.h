//======- CFLGraph.h - Abstract stratified sets implementation. --------======//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This file defines CFLGraph, an auxiliary data structure used by CFL-based
/// alias analysis.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_CFLGRAPH_H
#define LLVM_ANALYSIS_CFLGRAPH_H

#include "AliasAnalysisSummary.h"
#include "llvm/ADT/STLExtras.h"

namespace llvm {
namespace cflaa {
/// Edges can be one of four "weights" -- each weight must have an inverse
/// weight (Assign has Assign; Reference has Dereference).
enum class EdgeType {
  /// The weight assigned when assigning from or to a value. For example, in:
  /// %b = getelementptr %a, 0
  /// ...The relationships are %b assign %a, and %a assign %b. This used to be
  /// two edges, but having a distinction bought us nothing.
  Assign,

  /// The edge used when we have an edge going from some handle to a Value.
  /// Examples of this include:
  /// %b = load %a              (%b Dereference %a)
  /// %b = extractelement %a, 0 (%a Dereference %b)
  Dereference,

  /// The edge used when our edge goes from a value to a handle that may have
  /// contained it at some point. Examples:
  /// %b = load %a              (%a Reference %b)
  /// %b = extractelement %a, 0 (%b Reference %a)
  Reference
};

/// \brief The Program Expression Graph (PEG) of CFL analysis
/// CFLGraph is auxiliary data structure used by CFL-based alias analysis to
/// describe flow-insensitive pointer-related behaviors. Given an LLVM function,
/// the main purpose of this graph is to abstract away unrelated facts and
/// translate the rest into a form that can be easily digested by CFL analyses.
class CFLGraph {
  typedef Value *Node;

  struct Edge {
    EdgeType Type;
    Node Other;
  };

  typedef std::vector<Edge> EdgeList;

  struct NodeInfo {
    EdgeList Edges;
    AliasAttrs Attr;
  };

  typedef DenseMap<Node, NodeInfo> NodeMap;
  NodeMap NodeImpls;

  // Gets the inverse of a given EdgeType.
  static EdgeType flipWeight(EdgeType Initial) {
    switch (Initial) {
    case EdgeType::Assign:
      return EdgeType::Assign;
    case EdgeType::Dereference:
      return EdgeType::Reference;
    case EdgeType::Reference:
      return EdgeType::Dereference;
    }
    llvm_unreachable("Incomplete coverage of EdgeType enum");
  }

  const NodeInfo *getNode(Node N) const {
    auto Itr = NodeImpls.find(N);
    if (Itr == NodeImpls.end())
      return nullptr;
    return &Itr->second;
  }
  NodeInfo *getNode(Node N) {
    auto Itr = NodeImpls.find(N);
    if (Itr == NodeImpls.end())
      return nullptr;
    return &Itr->second;
  }

  static Node nodeDeref(const NodeMap::value_type &P) { return P.first; }
  typedef std::pointer_to_unary_function<const NodeMap::value_type &, Node>
      NodeDerefFun;

public:
  typedef EdgeList::const_iterator const_edge_iterator;
  typedef mapped_iterator<NodeMap::const_iterator, NodeDerefFun>
      const_node_iterator;

  bool addNode(Node N) {
    return NodeImpls
        .insert(std::make_pair(N, NodeInfo{EdgeList(), getAttrNone()}))
        .second;
  }

  void addAttr(Node N, AliasAttrs Attr) {
    auto *Info = getNode(N);
    assert(Info != nullptr);
    Info->Attr |= Attr;
  }

  void addEdge(Node From, Node To, EdgeType Type) {
    auto *FromInfo = getNode(From);
    assert(FromInfo != nullptr);
    auto *ToInfo = getNode(To);
    assert(ToInfo != nullptr);

    FromInfo->Edges.push_back(Edge{Type, To});
    ToInfo->Edges.push_back(Edge{flipWeight(Type), From});
  }

  AliasAttrs attrFor(Node N) const {
    auto *Info = getNode(N);
    assert(Info != nullptr);
    return Info->Attr;
  }

  iterator_range<const_edge_iterator> edgesFor(Node N) const {
    auto *Info = getNode(N);
    assert(Info != nullptr);
    auto &Edges = Info->Edges;
    return make_range(Edges.begin(), Edges.end());
  }

  iterator_range<const_node_iterator> nodes() const {
    return make_range<const_node_iterator>(
        map_iterator(NodeImpls.begin(), NodeDerefFun(nodeDeref)),
        map_iterator(NodeImpls.end(), NodeDerefFun(nodeDeref)));
  }

  bool empty() const { return NodeImpls.empty(); }
  std::size_t size() const { return NodeImpls.size(); }
};
}
}

#endif
