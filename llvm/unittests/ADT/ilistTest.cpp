//===- llvm/unittest/ADT/APInt.cpp - APInt unit tests ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ilist.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ilist_node.h"
#include "gtest/gtest.h"
#include <ostream>

using namespace llvm;

namespace {

struct Node : ilist_node<Node> {
  int Value;

  Node() {}
  Node(int Value) : Value(Value) {}
  Node(const Node&) = default;
  ~Node() { Value = -1; }
};

TEST(ilistTest, Basic) {
  ilist<Node> List;
  List.push_back(Node(1));
  EXPECT_EQ(1, List.back().Value);
  EXPECT_EQ(nullptr, List.getPrevNode(List.back()));
  EXPECT_EQ(nullptr, List.getNextNode(List.back()));

  List.push_back(Node(2));
  EXPECT_EQ(2, List.back().Value);
  EXPECT_EQ(2, List.getNextNode(List.front())->Value);
  EXPECT_EQ(1, List.getPrevNode(List.back())->Value);

  const ilist<Node> &ConstList = List;
  EXPECT_EQ(2, ConstList.back().Value);
  EXPECT_EQ(2, ConstList.getNextNode(ConstList.front())->Value);
  EXPECT_EQ(1, ConstList.getPrevNode(ConstList.back())->Value);
}

TEST(ilistTest, SpliceOne) {
  ilist<Node> List;
  List.push_back(1);

  // The single-element splice operation supports noops.
  List.splice(List.begin(), List, List.begin());
  EXPECT_EQ(1u, List.size());
  EXPECT_EQ(1, List.front().Value);
  EXPECT_TRUE(std::next(List.begin()) == List.end());

  // Altenative noop. Move the first element behind itself.
  List.push_back(2);
  List.push_back(3);
  List.splice(std::next(List.begin()), List, List.begin());
  EXPECT_EQ(3u, List.size());
  EXPECT_EQ(1, List.front().Value);
  EXPECT_EQ(2, std::next(List.begin())->Value);
  EXPECT_EQ(3, List.back().Value);
}

TEST(ilistTest, UnsafeClear) {
  ilist<Node> List;

  // Before even allocating a sentinel.
  List.clearAndLeakNodesUnsafely();
  EXPECT_EQ(0u, List.size());

  // Empty list with sentinel.
  ilist<Node>::iterator E = List.end();
  List.clearAndLeakNodesUnsafely();
  EXPECT_EQ(0u, List.size());
  // The sentinel shouldn't change.
  EXPECT_TRUE(E == List.end());

  // List with contents.
  List.push_back(1);
  ASSERT_EQ(1u, List.size());
  Node *N = &*List.begin();
  EXPECT_EQ(1, N->Value);
  List.clearAndLeakNodesUnsafely();
  EXPECT_EQ(0u, List.size());
  ASSERT_EQ(1, N->Value);
  delete N;

  // List is still functional.
  List.push_back(5);
  List.push_back(6);
  ASSERT_EQ(2u, List.size());
  EXPECT_EQ(5, List.front().Value);
  EXPECT_EQ(6, List.back().Value);
}

}
