//=== llvm/unittest/ADT/BreadthFirstIteratorTest.cpp - BFS iterator tests -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/BreadthFirstIterator.h"
#include "TestGraph.h"
#include "gtest/gtest.h"

#include <array>
#include <iterator>
#include <type_traits>

#include <cstddef>

using namespace llvm;

namespace llvm {

TEST(BreadthFristIteratorTest, Basic) {
  typedef bf_iterator<Graph<4>> BFIter;

  Graph<4> G;
  G.AddEdge(0, 1);
  G.AddEdge(0, 2);
  G.AddEdge(1, 3);

  auto It = BFIter::begin(G);
  auto End = BFIter::end(G);
  EXPECT_EQ(It.getLevel(), 0U);
  EXPECT_EQ(*It, G.AccessNode(0));
  ++It;
  EXPECT_EQ(It.getLevel(), 1U);
  EXPECT_EQ(*It, G.AccessNode(1));
  ++It;
  EXPECT_EQ(It.getLevel(), 1U);
  EXPECT_EQ(*It, G.AccessNode(2));
  ++It;
  EXPECT_EQ(It.getLevel(), 2U);
  EXPECT_EQ(*It, G.AccessNode(3));
  ++It;
  EXPECT_EQ(It, End);
}

TEST(BreadthFristIteratorTest, Cycle) {
  typedef bf_iterator<Graph<4>> BFIter;

  Graph<4> G;
  G.AddEdge(0, 1);
  G.AddEdge(1, 0);
  G.AddEdge(1, 2);
  G.AddEdge(2, 1);
  G.AddEdge(2, 1);
  G.AddEdge(2, 3);
  G.AddEdge(3, 2);
  G.AddEdge(3, 1);
  G.AddEdge(3, 0);

  auto It = BFIter::begin(G);
  auto End = BFIter::end(G);
  EXPECT_EQ(It.getLevel(), 0U);
  EXPECT_EQ(*It, G.AccessNode(0));
  ++It;
  EXPECT_EQ(It.getLevel(), 1U);
  EXPECT_EQ(*It, G.AccessNode(1));
  ++It;
  EXPECT_EQ(It.getLevel(), 2U);
  EXPECT_EQ(*It, G.AccessNode(2));
  ++It;
  EXPECT_EQ(It.getLevel(), 3U);
  EXPECT_EQ(*It, G.AccessNode(3));
  ++It;
  EXPECT_EQ(It, End);
}

static_assert(
    std::is_convertible_v<decltype(*std::declval<bf_iterator<Graph<3>>>()),
                          typename bf_iterator<Graph<3>>::reference>);

// bf_iterator should be (at-least) a forward-iterator
static_assert(std::is_base_of_v<std::forward_iterator_tag,
                                bf_iterator<Graph<4>>::iterator_category>);

TEST(BreadthFristIteratorTest, MultiPassSafeWithInternalSet) {
  Graph<4> G;
  G.AddEdge(0, 1);
  G.AddEdge(1, 2);
  G.AddEdge(1, 3);

  std::array<decltype(G)::NodeType *, 4> NodesFirstPass, NodesSecondPass;

  auto B = bf_begin(G), E = bf_end(G);

  std::size_t I = 0;
  for (auto It = B; It != E; ++It)
    NodesFirstPass[I++] = *It;

  I = 0;
  for (auto It = B; It != E; ++It)
    NodesSecondPass[I++] = *It;

  EXPECT_EQ(NodesFirstPass, NodesSecondPass);
}

} // end namespace llvm
