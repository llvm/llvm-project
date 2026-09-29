//===- DWARFVerifierTest.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/DWARF/DWARFVerifier.h"
#include "llvm/DebugInfo/DWARF/DWARFAddressRange.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

using DieRangeInfo = DWARFVerifier::DieRangeInfo;

// Helper to create a DieRangeInfo with a single range [Lo, Hi).
DieRangeInfo makeRI(uint64_t Lo, uint64_t Hi) {
  return DieRangeInfo({{Lo, Hi}});
}

TEST(DWARFVerifierTest, InsertNoOverlap) {
  DieRangeInfo Parent;
  // Insert three non-overlapping ranges.
  EXPECT_EQ(Parent.insert(makeRI(100, 200)), Parent.Children.end());
  EXPECT_EQ(Parent.insert(makeRI(300, 400)), Parent.Children.end());
  EXPECT_EQ(Parent.insert(makeRI(500, 600)), Parent.Children.end());
  EXPECT_EQ(Parent.Children.size(), 3u);
}

TEST(DWARFVerifierTest, InsertOverlapWithPredecessor) {
  DieRangeInfo Parent;
  EXPECT_EQ(Parent.insert(makeRI(100, 300)), Parent.Children.end());
  // Overlaps with [100, 300).
  auto It = Parent.insert(makeRI(200, 400));
  EXPECT_NE(It, Parent.Children.end());
  // The conflicting child should be the first one.
  EXPECT_EQ(It->Ranges.front().LowPC, 100u);
}

TEST(DWARFVerifierTest, InsertOverlapWithSuccessor) {
  DieRangeInfo Parent;
  EXPECT_EQ(Parent.insert(makeRI(300, 500)), Parent.Children.end());
  // Insert before, overlapping with [300, 500).
  auto It = Parent.insert(makeRI(100, 400));
  EXPECT_NE(It, Parent.Children.end());
  EXPECT_EQ(It->Ranges.front().LowPC, 300u);
}

TEST(DWARFVerifierTest, InsertEmptyRange) {
  DieRangeInfo Parent;
  DieRangeInfo Empty;
  EXPECT_EQ(Parent.insert(Empty), Parent.Children.end());
  EXPECT_EQ(Parent.Children.size(), 0u);
}

TEST(DWARFVerifierTest, InsertAdjacentRanges) {
  DieRangeInfo Parent;
  // Adjacent but not overlapping: [100,200) and [200,300).
  EXPECT_EQ(Parent.insert(makeRI(100, 200)), Parent.Children.end());
  EXPECT_EQ(Parent.insert(makeRI(200, 300)), Parent.Children.end());
  EXPECT_EQ(Parent.Children.size(), 2u);
}

TEST(DWARFVerifierTest, InsertLongRangeOverlapsSuccessor) {
  DieRangeInfo Parent;
  EXPECT_EQ(Parent.insert(makeRI(100, 200)), Parent.Children.end());
  EXPECT_EQ(Parent.insert(makeRI(300, 400)), Parent.Children.end());
  EXPECT_EQ(Parent.insert(makeRI(500, 600)), Parent.Children.end());
  // Insert a long range that overlaps multiple children.
  auto It = Parent.insert(makeRI(150, 550));
  EXPECT_NE(It, Parent.Children.end());
}

TEST(DWARFVerifierTest, InsertBetweenNonOverlapping) {
  DieRangeInfo Parent;
  EXPECT_EQ(Parent.insert(makeRI(100, 200)), Parent.Children.end());
  EXPECT_EQ(Parent.insert(makeRI(500, 600)), Parent.Children.end());
  // Fits in the gap.
  EXPECT_EQ(Parent.insert(makeRI(300, 400)), Parent.Children.end());
  EXPECT_EQ(Parent.Children.size(), 3u);
}

TEST(DWARFVerifierTest, InsertOverlapByOneAddress) {
  DieRangeInfo Parent;
  EXPECT_EQ(Parent.insert(makeRI(100, 200)), Parent.Children.end());
  // Overlaps by a single address.
  auto It = Parent.insert(makeRI(199, 300));
  EXPECT_NE(It, Parent.Children.end());
}

TEST(DWARFVerifierTest, InsertReverseOrder) {
  DieRangeInfo Parent;
  // Insert in reverse address order.
  EXPECT_EQ(Parent.insert(makeRI(500, 600)), Parent.Children.end());
  EXPECT_EQ(Parent.insert(makeRI(300, 400)), Parent.Children.end());
  EXPECT_EQ(Parent.insert(makeRI(100, 200)), Parent.Children.end());
  EXPECT_EQ(Parent.Children.size(), 3u);
}

TEST(DWARFVerifierTest, InsertReverseOrderWithOverlap) {
  DieRangeInfo Parent;
  EXPECT_EQ(Parent.insert(makeRI(500, 600)), Parent.Children.end());
  EXPECT_EQ(Parent.insert(makeRI(300, 400)), Parent.Children.end());
  // Overlaps with [300, 400).
  auto It = Parent.insert(makeRI(350, 450));
  EXPECT_NE(It, Parent.Children.end());
}

TEST(DWARFVerifierTest, InsertRandomOrderNoOverlap) {
  DieRangeInfo Parent;
  // Insert in shuffled order.
  EXPECT_EQ(Parent.insert(makeRI(500, 600)), Parent.Children.end());
  EXPECT_EQ(Parent.insert(makeRI(100, 200)), Parent.Children.end());
  EXPECT_EQ(Parent.insert(makeRI(700, 800)), Parent.Children.end());
  EXPECT_EQ(Parent.insert(makeRI(300, 400)), Parent.Children.end());
  EXPECT_EQ(Parent.Children.size(), 4u);
}

} // namespace
