//===- unittests/ADT/IListNodeBaseTest.cpp - ilist_node_base unit tests ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ilist_node_base.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

typedef ilist_node_base<false> RawNode;
typedef ilist_node_base<true> TrackingNode;

TEST(IListNodeBaseTest, DefaultConstructor) {
  RawNode A;
  EXPECT_EQ(nullptr, A.getPrev());
  EXPECT_EQ(nullptr, A.getNext());
  EXPECT_FALSE(A.isKnownSentinel());

  TrackingNode TA;
  EXPECT_EQ(nullptr, TA.getPrev());
  EXPECT_EQ(nullptr, TA.getNext());
  EXPECT_FALSE(TA.isKnownSentinel());
  EXPECT_FALSE(TA.isSentinel());
}

TEST(IListNodeBaseTest, setPrevAndNext) {
  RawNode A, B, C;
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

  TrackingNode TA, TB, TC;
  TA.setPrev(&TB);
  EXPECT_EQ(&TB, TA.getPrev());
  EXPECT_EQ(nullptr, TA.getNext());
  EXPECT_EQ(nullptr, TB.getPrev());
  EXPECT_EQ(nullptr, TB.getNext());
  EXPECT_EQ(nullptr, TC.getPrev());
  EXPECT_EQ(nullptr, TC.getNext());

  TA.setNext(&TC);
  EXPECT_EQ(&TB, TA.getPrev());
  EXPECT_EQ(&TC, TA.getNext());
  EXPECT_EQ(nullptr, TB.getPrev());
  EXPECT_EQ(nullptr, TB.getNext());
  EXPECT_EQ(nullptr, TC.getPrev());
  EXPECT_EQ(nullptr, TC.getNext());
}

TEST(IListNodeBaseTest, isKnownSentinel) {
  // Without sentinel tracking.
  RawNode A, B;
  EXPECT_FALSE(A.isKnownSentinel());
  A.setPrev(&B);
  A.setNext(&B);
  EXPECT_EQ(&B, A.getPrev());
  EXPECT_EQ(&B, A.getNext());
  EXPECT_FALSE(A.isKnownSentinel());
  A.initializeSentinel();
  EXPECT_FALSE(A.isKnownSentinel());
  EXPECT_EQ(&B, A.getPrev());
  EXPECT_EQ(&B, A.getNext());

  // With sentinel tracking.
  TrackingNode TA, TB;
  EXPECT_FALSE(TA.isKnownSentinel());
  EXPECT_FALSE(TA.isSentinel());
  TA.setPrev(&TB);
  TA.setNext(&TB);
  EXPECT_EQ(&TB, TA.getPrev());
  EXPECT_EQ(&TB, TA.getNext());
  EXPECT_FALSE(TA.isKnownSentinel());
  EXPECT_FALSE(TA.isSentinel());
  TA.initializeSentinel();
  EXPECT_TRUE(TA.isKnownSentinel());
  EXPECT_TRUE(TA.isSentinel());
  EXPECT_EQ(&TB, TA.getPrev());
  EXPECT_EQ(&TB, TA.getNext());
}

} // end namespace
