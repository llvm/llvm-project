//===-- RangeTest.cpp ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/RangeMap.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace lldb_private;

using RangeDataVectorT = RangeDataVector<uint32_t, uint32_t, uint32_t>;
using EntryT = RangeDataVectorT::Entry;

static testing::Matcher<const EntryT *> EntryIs(uint32_t ID) {
  return testing::Pointee(testing::Field(&EntryT::data, ID));
}

TEST(RangeDataVector, FindEntryThatContains) {
  RangeDataVectorT Map;
  uint32_t NextID = 0;
  Map.Append(EntryT(0, 10, NextID++));
  Map.Append(EntryT(10, 10, NextID++));
  Map.Append(EntryT(20, 10, NextID++));
  Map.Sort();

  EXPECT_THAT(Map.FindEntryThatContains(0), EntryIs(0));
  EXPECT_THAT(Map.FindEntryThatContains(9), EntryIs(0));
  EXPECT_THAT(Map.FindEntryThatContains(10), EntryIs(1));
  EXPECT_THAT(Map.FindEntryThatContains(19), EntryIs(1));
  EXPECT_THAT(Map.FindEntryThatContains(20), EntryIs(2));
  EXPECT_THAT(Map.FindEntryThatContains(29), EntryIs(2));
  EXPECT_THAT(Map.FindEntryThatContains(30), nullptr);
}

TEST(RangeDataVector, FindEntryThatContains_Overlap) {
  RangeDataVectorT Map;
  uint32_t NextID = 0;
  Map.Append(EntryT(0, 40, NextID++));
  Map.Append(EntryT(10, 20, NextID++));
  Map.Append(EntryT(20, 10, NextID++));
  Map.Sort();

  // With overlapping intervals, the intention seems to be to return the first
  // interval which contains the address.
  EXPECT_THAT(Map.FindEntryThatContains(25), EntryIs(0));

  // However, this does not always succeed.
  // TODO: This should probably return the range (0, 40) as well.
  EXPECT_THAT(Map.FindEntryThatContains(35), nullptr);
}
