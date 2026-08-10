//===- OutlinedHashTreeTest.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CGData/OutlinedHashTree.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(OutlinedHashTreeTest, Empty) {
  OutlinedHashTree HashTree;
  EXPECT_TRUE(HashTree.empty());
  // The header node is always present.
  EXPECT_EQ(HashTree.size(), 1u);
  EXPECT_EQ(HashTree.depth(), 0u);
}

TEST(OutlinedHashTreeTest, Insert) {
  OutlinedHashTree HashTree;
  HashTree.insert({{1, 2, 3}, 1});
  // The node count is 4 (including the root node).
  EXPECT_EQ(HashTree.size(), 4u);
  // The terminal count is 1.
  EXPECT_EQ(HashTree.size(/*GetTerminalCountOnly=*/true), 1u);
  // The depth is 3.
  EXPECT_EQ(HashTree.depth(), 3u);

  HashTree.clear();
  EXPECT_TRUE(HashTree.empty());

  HashTree.insert({{1, 2, 3}, 1});
  HashTree.insert({{1, 2, 4}, 2});
  // The nodes of 1 and 2 are shared with the same prefix.
  // The nodes are root, 1, 2, 3 and 4, whose counts are 5.
  EXPECT_EQ(HashTree.size(), 5u);
}

TEST(OutlinedHashTreeTest, Find) {
  OutlinedHashTree HashTree;
  HashTree.insert({{1, 2, 3}, 1});
  HashTree.insert({{1, 2, 3}, 2});

  // The node count does not change as the same sequences are added.
  EXPECT_EQ(HashTree.size(), 4u);
  // The terminal counts are accumulated from two same sequences.
  EXPECT_TRUE(HashTree.find({1, 2, 3}));
  EXPECT_EQ(HashTree.find({1, 2, 3}).value(), 3u);
  EXPECT_FALSE(HashTree.find({1, 2}));
}

TEST(OutlinedHashTreeTest, Merge) {
  // Build HashTree1 inserting 2 sequences.
  OutlinedHashTree HashTree1;

  HashTree1.insert({{1, 2}, 20});
  HashTree1.insert({{1, 4}, 30});

  // Build HashTree2 and HashTree3 for each
  OutlinedHashTree HashTree2;
  HashTree2.insert({{1, 2}, 20});
  OutlinedHashTree HashTree3;
  HashTree3.insert({{1, 4}, 30});

  // Merge HashTree3 into HashTree2.
  HashTree2.merge(&HashTree3);

  // Compare HashTree1 and HashTree2.
  EXPECT_EQ(HashTree1.size(), HashTree2.size());
  EXPECT_EQ(HashTree1.depth(), HashTree2.depth());
  EXPECT_EQ(HashTree1.find({1, 2}), HashTree2.find({1, 2}));
  EXPECT_EQ(HashTree1.find({1, 4}), HashTree2.find({1, 4}));
  EXPECT_EQ(HashTree1.find({1, 3}), HashTree2.find({1, 3}));
}

} // end namespace
