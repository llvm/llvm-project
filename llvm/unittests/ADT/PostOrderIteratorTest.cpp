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

#include <array>
#include <iterator>
#include <type_traits>

#include <cstddef>

using namespace llvm;

namespace {

// Whether we're able to compile
TEST(PostOrderIteratorTest, Compiles) {
  typedef SmallPtrSet<void *, 4> ExtSetTy;

  // Tests that template specializations are kept up to date
  void *Null = nullptr;
  po_iterator_storage<std::set<void *>, false> PIS;
  PIS.insertEdge(std::optional<void *>(), Null);
  ExtSetTy Ext;
  po_iterator_storage<ExtSetTy, true> PISExt(Ext);
  PIS.insertEdge(std::optional<void *>(), Null);

  // Test above, but going through po_iterator (which inherits from template
  // base)
  BasicBlock *NullBB = nullptr;
  auto PI = po_end(NullBB);
  PI.insertEdge(std::optional<BasicBlock *>(), NullBB);
  auto PIExt = po_ext_end(NullBB, Ext);
  PIExt.insertEdge(std::optional<BasicBlock *>(), NullBB);
}

static_assert(
    std::is_convertible_v<decltype(*std::declval<po_iterator<Graph<3>>>()),
                          typename po_iterator<Graph<3>>::reference>);

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

// po_iterator should be (at-least) a forward-iterator
static_assert(std::is_base_of_v<std::forward_iterator_tag,
                                po_iterator<Graph<4>>::iterator_category>);

// po_ext_iterator cannot provide multi-pass guarantee, therefore its only
// an input-iterator
static_assert(std::is_same_v<po_ext_iterator<Graph<4>>::iterator_category,
                             std::input_iterator_tag>);

TEST(PostOrderIteratorTest, MultiPassSafeWithInternalSet) {
  Graph<4> G;
  G.AddEdge(0, 1);
  G.AddEdge(1, 2);
  G.AddEdge(1, 3);

  std::array<decltype(G)::NodeType *, 4> NodesFirstPass, NodesSecondPass;

  auto B = po_begin(G), E = po_end(G);

  std::size_t I = 0;
  for (auto It = B; It != E; ++It)
    NodesFirstPass[I++] = *It;

  I = 0;
  for (auto It = B; It != E; ++It)
    NodesSecondPass[I++] = *It;

  EXPECT_EQ(NodesFirstPass, NodesSecondPass);
}
}
