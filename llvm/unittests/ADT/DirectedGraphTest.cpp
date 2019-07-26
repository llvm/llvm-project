//===- llvm/unittest/ADT/DirectedGraphTest.cpp ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines concrete derivations of the directed-graph base classes
// for testing purposes.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/DirectedGraph.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "gtest/gtest.h"

namespace llvm {

//===--------------------------------------------------------------------===//
// Derived nodes, edges and graph types based on DirectedGraph.
//===--------------------------------------------------------------------===//

class DGTestNode;
class DGTestEdge;
using DGTestNodeBase = DGNode<DGTestNode, DGTestEdge>;
using DGTestEdgeBase = DGEdge<DGTestNode, DGTestEdge>;
using DGTestBase = DirectedGraph<DGTestNode, DGTestEdge>;

class DGTestNode : public DGTestNodeBase {
public:
  DGTestNode() = default;
};

class DGTestEdge : public DGTestEdgeBase {
public:
  DGTestEdge() = delete;
  DGTestEdge(DGTestNode &N) : DGTestEdgeBase(N) {}
};

class DGTestGraph : public DGTestBase {
public:
  DGTestGraph() = default;
  ~DGTestGraph(){};
};

using EdgeListTy = SmallVector<DGTestEdge *, 2>;

//===--------------------------------------------------------------------===//
// GraphTraits specializations for the DGTest
//===--------------------------------------------------------------------===//

template <> struct GraphTraits<DGTestNode *> {
  using NodeRef = DGTestNode *;

  static DGTestNode *DGTestGetTargetNode(DGEdge<DGTestNode, DGTestEdge> *P) {
    return &P->getTargetNode();
  }

  // Provide a mapped iterator so that the GraphTrait-based implementations can
  // find the target nodes without having to explicitly go through the edges.
  using ChildIteratorType =
      mapped_iterator<DGTestNode::iterator, decltype(&DGTestGetTargetNode)>;
  using ChildEdgeIteratorType = DGTestNode::iterator;

  static NodeRef getEntryNode(NodeRef N) { return N; }
  static ChildIteratorType child_begin(NodeRef N) {
    return ChildIteratorType(N->begin(), &DGTestGetTargetNode);
  }
  static ChildIteratorType child_end(NodeRef N) {
    return ChildIteratorType(N->end(), &DGTestGetTargetNode);
  }

  static ChildEdgeIteratorType child_edge_begin(NodeRef N) {
    return N->begin();
  }
  static ChildEdgeIteratorType child_edge_end(NodeRef N) { return N->end(); }
};

template <>
struct GraphTraits<DGTestGraph *> : public GraphTraits<DGTestNode *> {
  using nodes_iterator = DGTestGraph::iterator;
  static NodeRef getEntryNode(DGTestGraph *DG) { return *DG->begin(); }
  static nodes_iterator nodes_begin(DGTestGraph *DG) { return DG->begin(); }
  static nodes_iterator nodes_end(DGTestGraph *DG) { return DG->end(); }
};

//===--------------------------------------------------------------------===//
// Test various modification and query functions.
//===--------------------------------------------------------------------===//

TEST(DirectedGraphTest, AddAndConnectNodes) {
  DGTestGraph DG;
  DGTestNode N1, N2, N3;
  DGTestEdge E1(N1), E2(N2), E3(N3);

  // Check that new nodes can be added successfully.
  EXPECT_TRUE(DG.addNode(N1));
  EXPECT_TRUE(DG.addNode(N2));
  EXPECT_TRUE(DG.addNode(N3));

  // Check that duplicate nodes are not added to the graph.
  EXPECT_FALSE(DG.addNode(N1));

  // Check that nodes can be connected using valid edges with no errors.
  EXPECT_TRUE(DG.connect(N1, N2, E2));
  EXPECT_TRUE(DG.connect(N2, N3, E3));
  EXPECT_TRUE(DG.connect(N3, N1, E1));

  // The graph looks like this now:
  //
  // +---------------+
  // v               |
  // N1 -> N2 -> N3 -+

  // Check that already connected nodes with the given edge are not connected
  // again (ie. edges are between nodes are not duplicated).
  EXPECT_FALSE(DG.connect(N3, N1, E1));

  // Check that there are 3 nodes in the graph.
  EXPECT_TRUE(DG.size() == 3);

  // Check that the added nodes can be found in the graph.
  EXPECT_NE(DG.findNode(N3), DG.end());

  // Check that nodes that are not part of the graph are not found.
  DGTestNode N4;
  EXPECT_EQ(DG.findNode(N4), DG.end());

  // Check that findIncommingEdgesToNode works correctly.
  EdgeListTy EL;
  EXPECT_TRUE(DG.findIncomingEdgesToNode(N1, EL));
  EXPECT_TRUE(EL.size() == 1);
  EXPECT_EQ(*EL[0], E1);
}

TEST(DirectedGraphTest, AddRemoveEdge) {
  DGTestGraph DG;
  DGTestNode N1, N2, N3;
  DGTestEdge E1(N1), E2(N2), E3(N3);
  DG.addNode(N1);
  DG.addNode(N2);
  DG.addNode(N3);
  DG.connect(N1, N2, E2);
  DG.connect(N2, N3, E3);
  DG.connect(N3, N1, E1);

  // The graph looks like this now:
  //
  // +---------------+
  // v               |
  // N1 -> N2 -> N3 -+

  // Check that there are 3 nodes in the graph.
  EXPECT_TRUE(DG.size() == 3);

  // Check that the target nodes of the edges are correct.
  EXPECT_EQ(E1.getTargetNode(), N1);
  EXPECT_EQ(E2.getTargetNode(), N2);
  EXPECT_EQ(E3.getTargetNode(), N3);

  // Remove the edge from N1 to N2.
  N1.removeEdge(E2);

  // The graph looks like this now:
  //
  // N2 -> N3 -> N1

  // Check that there are no incoming edges to N2.
  EdgeListTy EL;
  EXPECT_FALSE(DG.findIncomingEdgesToNode(N2, EL));
  EXPECT_TRUE(EL.empty());

  // Put the edge from N1 to N2 back in place.
  N1.addEdge(E2);

  // Check that E2 is the only incoming edge to N2.
  EL.clear();
  EXPECT_TRUE(DG.findIncomingEdgesToNode(N2, EL));
  EXPECT_EQ(*EL[0], E2);
}

TEST(DirectedGraphTest, hasEdgeTo) {
  DGTestGraph DG;
  DGTestNode N1, N2, N3;
  DGTestEdge E1(N1), E2(N2), E3(N3), E4(N1);
  DG.addNode(N1);
  DG.addNode(N2);
  DG.addNode(N3);
  DG.connect(N1, N2, E2);
  DG.connect(N2, N3, E3);
  DG.connect(N3, N1, E1);
  DG.connect(N2, N1, E4);

  // The graph looks like this now:
  //
  // +-----+
  // v     |
  // N1 -> N2 -> N3
  // ^           |
  // +-----------+

  EXPECT_TRUE(N2.hasEdgeTo(N1));
  EXPECT_TRUE(N3.hasEdgeTo(N1));
}

TEST(DirectedGraphTest, AddRemoveNode) {
  DGTestGraph DG;
  DGTestNode N1, N2, N3;
  DGTestEdge E1(N1), E2(N2), E3(N3);
  DG.addNode(N1);
  DG.addNode(N2);
  DG.addNode(N3);
  DG.connect(N1, N2, E2);
  DG.connect(N2, N3, E3);
  DG.connect(N3, N1, E1);

  // The graph looks like this now:
  //
  // +---------------+
  // v               |
  // N1 -> N2 -> N3 -+

  // Check that there are 3 nodes in the graph.
  EXPECT_TRUE(DG.size() == 3);

  // Check that a node in the graph can be removed, but not more than once.
  EXPECT_TRUE(DG.removeNode(N1));
  EXPECT_EQ(DG.findNode(N1), DG.end());
  EXPECT_FALSE(DG.removeNode(N1));

  // The graph looks like this now:
  //
  // N2 -> N3

  // Check that there are 2 nodes in the graph and only N2 is connected to N3.
  EXPECT_TRUE(DG.size() == 2);
  EXPECT_TRUE(N3.getEdges().empty());
  EdgeListTy EL;
  EXPECT_FALSE(DG.findIncomingEdgesToNode(N2, EL));
  EXPECT_TRUE(EL.empty());
}

TEST(DirectedGraphTest, SCC) {

  DGTestGraph DG;
  DGTestNode N1, N2, N3, N4;
  DGTestEdge E1(N1), E2(N2), E3(N3), E4(N4);
  DG.addNode(N1);
  DG.addNode(N2);
  DG.addNode(N3);
  DG.addNode(N4);
  DG.connect(N1, N2, E2);
  DG.connect(N2, N3, E3);
  DG.connect(N3, N1, E1);
  DG.connect(N3, N4, E4);

  // The graph looks like this now:
  //
  // +---------------+
  // v               |
  // N1 -> N2 -> N3 -+    N4
  //             |        ^
  //             +--------+

  // Test that there are two SCCs:
  // 1. {N1, N2, N3}
  // 2. {N4}
  using NodeListTy = SmallPtrSet<DGTestNode *, 3>;
  SmallVector<NodeListTy, 4> ListOfSCCs;
  for (auto &SCC : make_range(scc_begin(&DG), scc_end(&DG)))
    ListOfSCCs.push_back(NodeListTy(SCC.begin(), SCC.end()));

  EXPECT_TRUE(ListOfSCCs.size() == 2);

  for (auto &SCC : ListOfSCCs) {
    if (SCC.size() > 1)
      continue;
    EXPECT_TRUE(SCC.size() == 1);
    EXPECT_TRUE(SCC.count(&N4) == 1);
  }
  for (auto &SCC : ListOfSCCs) {
    if (SCC.size() <= 1)
      continue;
    EXPECT_TRUE(SCC.size() == 3);
    EXPECT_TRUE(SCC.count(&N1) == 1);
    EXPECT_TRUE(SCC.count(&N2) == 1);
    EXPECT_TRUE(SCC.count(&N3) == 1);
    EXPECT_TRUE(SCC.count(&N4) == 0);
  }
}

} // namespace llvm
