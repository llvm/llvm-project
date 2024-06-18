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

class Parent {};

typedef ilist_node_base<false, void> RawNode;
typedef ilist_node_base<true, void> TrackingNode;
typedef ilist_node_base<false, Parent*> ParentNode;
typedef ilist_node_base<true, Parent*> ParentTrackingNode;

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

  ParentNode PA;
  EXPECT_EQ(nullptr, PA.getPrev());
  EXPECT_EQ(nullptr, PA.getNext());
  EXPECT_EQ(nullptr, PA.getNodeBaseParent());
  EXPECT_FALSE(PA.isKnownSentinel());

  ParentTrackingNode PTA;
  EXPECT_EQ(nullptr, PTA.getPrev());
  EXPECT_EQ(nullptr, PTA.getNext());
  EXPECT_EQ(nullptr, PTA.getNodeBaseParent());
  EXPECT_FALSE(PTA.isKnownSentinel());
  EXPECT_FALSE(PTA.isSentinel());
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

  ParentNode PA, PB, PC;
  PA.setPrev(&PB);
  EXPECT_EQ(&PB, PA.getPrev());
  EXPECT_EQ(nullptr, PA.getNext());
  EXPECT_EQ(nullptr, PB.getPrev());
  EXPECT_EQ(nullptr, PB.getNext());
  EXPECT_EQ(nullptr, PC.getPrev());
  EXPECT_EQ(nullptr, PC.getNext());

  PA.setNext(&PC);
  EXPECT_EQ(&PB, PA.getPrev());
  EXPECT_EQ(&PC, PA.getNext());
  EXPECT_EQ(nullptr, PB.getPrev());
  EXPECT_EQ(nullptr, PB.getNext());
  EXPECT_EQ(nullptr, PC.getPrev());
  EXPECT_EQ(nullptr, PC.getNext());

  ParentTrackingNode PTA, PTB, PTC;
  PTA.setPrev(&PTB);
  EXPECT_EQ(&PTB, PTA.getPrev());
  EXPECT_EQ(nullptr, PTA.getNext());
  EXPECT_EQ(nullptr, PTB.getPrev());
  EXPECT_EQ(nullptr, PTB.getNext());
  EXPECT_EQ(nullptr, PTC.getPrev());
  EXPECT_EQ(nullptr, PTC.getNext());

  PTA.setNext(&PTC);
  EXPECT_EQ(&PTB, PTA.getPrev());
  EXPECT_EQ(&PTC, PTA.getNext());
  EXPECT_EQ(nullptr, PTB.getPrev());
  EXPECT_EQ(nullptr, PTB.getNext());
  EXPECT_EQ(nullptr, PTC.getPrev());
  EXPECT_EQ(nullptr, PTC.getNext());

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

  // Without sentinel tracking (with Parent).
  ParentNode PA, PB;
  EXPECT_FALSE(PA.isKnownSentinel());
  PA.setPrev(&PB);
  PA.setNext(&PB);
  EXPECT_EQ(&PB, PA.getPrev());
  EXPECT_EQ(&PB, PA.getNext());
  EXPECT_FALSE(PA.isKnownSentinel());
  PA.initializeSentinel();
  EXPECT_FALSE(PA.isKnownSentinel());
  EXPECT_EQ(&PB, PA.getPrev());
  EXPECT_EQ(&PB, PA.getNext());

  // With sentinel tracking (with Parent).
  ParentTrackingNode PTA, PTB;
  EXPECT_FALSE(PTA.isKnownSentinel());
  EXPECT_FALSE(PTA.isSentinel());
  PTA.setPrev(&PTB);
  PTA.setNext(&PTB);
  EXPECT_EQ(&PTB, PTA.getPrev());
  EXPECT_EQ(&PTB, PTA.getNext());
  EXPECT_FALSE(PTA.isKnownSentinel());
  EXPECT_FALSE(PTA.isSentinel());
  PTA.initializeSentinel();
  EXPECT_TRUE(PTA.isKnownSentinel());
  EXPECT_TRUE(PTA.isSentinel());
  EXPECT_EQ(&PTB, PTA.getPrev());
  EXPECT_EQ(&PTB, PTA.getNext());
}

TEST(IListNodeBaseTest, setNodeBaseParent) {
  Parent Par;
  ParentNode PA;
  EXPECT_EQ(nullptr, PA.getNodeBaseParent());
  PA.setNodeBaseParent(&Par);
  EXPECT_EQ(&Par, PA.getNodeBaseParent());

  ParentTrackingNode PTA;
  EXPECT_EQ(nullptr, PTA.getNodeBaseParent());
  PTA.setNodeBaseParent(&Par);
  EXPECT_EQ(&Par, PTA.getNodeBaseParent());
}

} // end namespace
