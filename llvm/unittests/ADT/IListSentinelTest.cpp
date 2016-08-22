//===- unittests/ADT/IListSentinelTest.cpp - ilist_sentinel unit tests ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ilist.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

class Node : public ilist_node<Node> {};

TEST(IListSentinelTest, DefaultConstructor) {
  ilist_sentinel<Node> S;
  EXPECT_EQ(&S, ilist_node_access::getPrev(S));
  EXPECT_EQ(&S, ilist_node_access::getNext(S));
#ifdef LLVM_ENABLE_ABI_BREAKING_CHECKS
  EXPECT_TRUE(S.isKnownSentinel());
#else
  EXPECT_FALSE(S.isKnownSentinel());
#endif
}

TEST(IListSentinelTest, NormalNodeIsNotKnownSentinel) {
  Node N;
  EXPECT_EQ(nullptr, ilist_node_access::getPrev(N));
  EXPECT_EQ(nullptr, ilist_node_access::getNext(N));
  EXPECT_FALSE(N.isKnownSentinel());
}

} // end namespace
