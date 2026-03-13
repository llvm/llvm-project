//===- unittests/Support/GenericDomTreeTest.cpp - GenericDomTree.h tests --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/GenericDomTree.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/Support/DataTypes.h"
#include "gtest/gtest.h"
using namespace llvm;

namespace {

// Very simple (fake) graph structure to test dominator tree on.
struct NumberedGraph;

struct NumberedNode {
  NumberedGraph *Parent;
  unsigned Number;

  NumberedNode(NumberedGraph *Parent, unsigned Number)
      : Parent(Parent), Number(Number) {}

  NumberedGraph *getParent() const { return Parent; }
};

struct NumberedGraph {
  SmallVector<std::unique_ptr<NumberedNode>> Nodes;
  unsigned NumberEpoch = 0;

  NumberedNode *addNode() {
    unsigned Num = Nodes.size();
    return Nodes.emplace_back(std::make_unique<NumberedNode>(this, Num)).get();
  }
};
} // namespace

namespace llvm {
template <> struct GraphTraits<NumberedNode *> {
  using NodeRef = NumberedNode *;
  static unsigned getNumber(NumberedNode *Node) { return Node->Number; }
};

template <> struct GraphTraits<const NumberedNode *> {
  using NodeRef = NumberedNode *;
  static unsigned getNumber(const NumberedNode *Node) { return Node->Number; }
};

template <> struct GraphTraits<NumberedGraph *> {
  using NodeRef = NumberedNode *;
  static unsigned getMaxNumber(NumberedGraph *G) { return G->Nodes.size(); }
  static unsigned getNumberEpoch(NumberedGraph *G) { return G->NumberEpoch; }
};

namespace DomTreeBuilder {
// Dummy specialization. Only needed so that we can call recalculate(), which
// sets DT.Parent -- but we can't access DT.Parent here.
template <> void Calculate(DomTreeBase<NumberedNode> &DT) {}
} // end namespace DomTreeBuilder
} // end namespace llvm

namespace {

TEST(GenericDomTree, BlockNumbers) {
  NumberedGraph G;
  NumberedNode *N1 = G.addNode();
  NumberedNode *N2 = G.addNode();

  DomTreeBase<NumberedNode> DT;
  DT.recalculate(G); // only sets parent
  // Construct fake domtree: node 0 dominates all other nodes
  DT.setNewRoot(N1);
  DT.addNewBlock(N2, N1);

  // Roundtrip is correct
  for (auto &N : G.Nodes)
    EXPECT_EQ(DT.getNode(N.get())->getBlock(), N.get());
  // If we manually change a number, we should get a different node.
  ASSERT_EQ(N1->Number, 0u);
  ASSERT_EQ(N2->Number, 1u);
  N1->Number = 1;
  EXPECT_EQ(DT.getNode(N1)->getBlock(), N2);
  EXPECT_EQ(DT.getNode(N2)->getBlock(), N2);
  N2->Number = 0;
  EXPECT_EQ(DT.getNode(N2)->getBlock(), N1);

  // Renumer blocks, should fix node domtree-internal node map
  DT.updateBlockNumbers();
  for (auto &N : G.Nodes)
    EXPECT_EQ(DT.getNode(N.get())->getBlock(), N.get());

  // Adding a new node with a higher number is no problem
  NumberedNode *N3 = G.addNode();
  EXPECT_EQ(DT.getNode(N3), nullptr);
  // ... even if it exceeds getMaxNumber()
  NumberedNode *N4 = G.addNode();
  N4->Number = 1000;
  EXPECT_EQ(DT.getNode(N4), nullptr);

  DT.addNewBlock(N3, N1);
  DT.addNewBlock(N4, N1);
  for (auto &N : G.Nodes)
    EXPECT_EQ(DT.getNode(N.get())->getBlock(), N.get());
}

} // namespace
