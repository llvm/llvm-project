//=== llvm/unittest/ADT/BreadthFirstIteratorTest.cpp - BFS iterator tests -===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/BreadthFirstIterator.h"
#include "TestGraph.h"
#include "gtest/gtest.h"

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

} // end namespace llvm
