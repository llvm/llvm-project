//===- PostOrderIteratorTest.cpp - PostOrderIterator unit tests -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "gtest/gtest.h"
#include "TestGraph.h"

using namespace llvm;

namespace {

// Whether we're able to compile
TEST(PostOrderIteratorTest, Compiles) {
  typedef SmallPtrSet<void *, 4> ExtSetTy;

  // Tests that template specializations are kept up to date
  void *Null = nullptr;
  po_iterator_storage<std::set<void *>, false> PIS;
  PIS.insertEdge(Optional<void *>(), Null);
  ExtSetTy Ext;
  po_iterator_storage<ExtSetTy, true> PISExt(Ext);
  PIS.insertEdge(Optional<void *>(), Null);

  // Test above, but going through po_iterator (which inherits from template
  // base)
  BasicBlock *NullBB = nullptr;
  auto PI = po_end(NullBB);
  PI.insertEdge(Optional<BasicBlock *>(), NullBB);
  auto PIExt = po_ext_end(NullBB, Ext);
  PIExt.insertEdge(Optional<BasicBlock *>(), NullBB);
}

// Test post-order and reverse post-order traversals for simple graph type.
TEST(PostOrderIteratorTest, PostOrderAndReversePostOrderTraverrsal) {
  Graph<6> G;
  G.AddEdge(0, 1);
  G.AddEdge(0, 2);
  G.AddEdge(0, 3);
  G.AddEdge(1, 4);
  G.AddEdge(2, 5);
  G.AddEdge(5, 2);
  G.AddEdge(2, 4);
  G.AddEdge(1, 4);

  SmallVector<int> FromIterator;
  for (auto N : post_order(G))
    FromIterator.push_back(N->first);
  EXPECT_EQ(6u, FromIterator.size());
  EXPECT_EQ(4, FromIterator[0]);
  EXPECT_EQ(1, FromIterator[1]);
  EXPECT_EQ(5, FromIterator[2]);
  EXPECT_EQ(2, FromIterator[3]);
  EXPECT_EQ(3, FromIterator[4]);
  EXPECT_EQ(0, FromIterator[5]);
  FromIterator.clear();

  ReversePostOrderTraversal<Graph<6>> RPOT(G);
  for (auto N : RPOT)
    FromIterator.push_back(N->first);
  EXPECT_EQ(6u, FromIterator.size());
  EXPECT_EQ(0, FromIterator[0]);
  EXPECT_EQ(3, FromIterator[1]);
  EXPECT_EQ(2, FromIterator[2]);
  EXPECT_EQ(5, FromIterator[3]);
  EXPECT_EQ(1, FromIterator[4]);
  EXPECT_EQ(4, FromIterator[5]);
}
}
