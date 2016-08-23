//===- unittests/ADT/IListNodeBaseTest.cpp - ilist_node_base unit tests ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ilist_node.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(IListNodeBaseTest, DefaultConstructor) {
  ilist_node_base A;
  EXPECT_EQ(nullptr, A.getPrev());
  EXPECT_EQ(nullptr, A.getNext());
  EXPECT_FALSE(A.isKnownSentinel());
}

TEST(IListNodeBaseTest, setPrevAndNext) {
  ilist_node_base A, B, C;
  A.setPrev(&B);
  EXPECT_EQ(&B, A.getPrev());
  EXPECT_EQ(nullptr, A.getNext());
  EXPECT_EQ(nullptr, B.getPrev());
  EXPECT_EQ(nullptr, B.getNext());
  EXPECT_EQ(nullptr, C.getPrev());
  EXPECT_EQ(nullptr, C.getNext());

  A.setNext(&C);
  EXPECT_EQ(&B, A.getPrev());
  EXPECT_EQ(&C, A.getNext());
  EXPECT_EQ(nullptr, B.getPrev());
  EXPECT_EQ(nullptr, B.getNext());
  EXPECT_EQ(nullptr, C.getPrev());
  EXPECT_EQ(nullptr, C.getNext());
}

TEST(IListNodeBaseTest, isKnownSentinel) {
  ilist_node_base A, B;
  EXPECT_FALSE(A.isKnownSentinel());
  A.setPrev(&B);
  A.setNext(&B);
  EXPECT_EQ(&B, A.getPrev());
  EXPECT_EQ(&B, A.getNext());
  A.initializeSentinel();
#ifdef LLVM_ENABLE_ABI_BREAKING_CHECKS
  EXPECT_TRUE(A.isKnownSentinel());
#else
  EXPECT_FALSE(A.isKnownSentinel());
#endif
  EXPECT_EQ(&B, A.getPrev());
  EXPECT_EQ(&B, A.getNext());
}

} // end namespace
