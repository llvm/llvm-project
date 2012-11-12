//===- llvm/unittest/ADT/APInt.cpp - APInt unit tests ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <ostream>
#include "gtest/gtest.h"
#include "llvm/ADT/ilist.h"
#include "llvm/ADT/ilist_node.h"

using namespace llvm;

namespace {

struct Node : ilist_node<Node> {
  int Value;

  Node() {}
  Node(int _Value) : Value(_Value) {}
};

TEST(ilistTest, Basic) {
  ilist<Node> List;
  List.push_back(Node(1));
  EXPECT_EQ(1, List.back().Value);
  EXPECT_EQ(0, List.back().getPrevNode());
  EXPECT_EQ(0, List.back().getNextNode());

  List.push_back(Node(2));
  EXPECT_EQ(2, List.back().Value);
  EXPECT_EQ(2, List.front().getNextNode()->Value);
  EXPECT_EQ(1, List.back().getPrevNode()->Value);

  const ilist<Node> &ConstList = List;
  EXPECT_EQ(2, ConstList.back().Value);
  EXPECT_EQ(2, ConstList.front().getNextNode()->Value);
  EXPECT_EQ(1, ConstList.back().getPrevNode()->Value);
}

}
