//===-- VMRangeTest.cpp -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include <limits>

#include "lldb/Utility/VMRange.h"

using namespace lldb_private;

namespace lldb_private {
void PrintTo(const VMRange &v, std::ostream *os) {
  (*os) << "VMRange(" << v.GetBaseAddress() << ", " << v.GetEndAddress() << ")";
}
} // namespace lldb_private

TEST(VMRange, IsValid) {
  VMRange range;
  EXPECT_FALSE(range.IsValid());

  range.Reset(0x1, 0x100);
  EXPECT_TRUE(range.IsValid());

  range.Reset(0x1, 0x1);
  EXPECT_FALSE(range.IsValid());
}

TEST(VMRange, Clear) {
  VMRange range(0x100, 0x200);
  EXPECT_NE(VMRange(), range);
  range.Clear();
  EXPECT_EQ(VMRange(), range);
}

TEST(VMRange, Comparison) {
  VMRange range1(0x100, 0x200);
  VMRange range2(0x100, 0x200);
  EXPECT_EQ(range1, range2);

  EXPECT_NE(VMRange(0x100, 0x1ff), range1);
  EXPECT_NE(VMRange(0x100, 0x201), range1);
  EXPECT_NE(VMRange(0x0ff, 0x200), range1);
  EXPECT_NE(VMRange(0x101, 0x200), range1);

  range2.Clear();
  EXPECT_NE(range1, range2);
}

TEST(VMRange, Reset) {
  VMRange range(0x100, 0x200);
  EXPECT_FALSE(VMRange(0x200, 0x200) == range);
  range.Reset(0x200, 0x200);
  EXPECT_TRUE(VMRange(0x200, 0x200) == range);
}

TEST(VMRange, SetEndAddress) {
  VMRange range(0x100, 0x200);

  range.SetEndAddress(0xFF);
  EXPECT_EQ(0U, range.GetByteSize());
  EXPECT_FALSE(range.IsValid());

  range.SetEndAddress(0x101);
  EXPECT_EQ(1U, range.GetByteSize());
  EXPECT_TRUE(range.IsValid());
}

TEST(VMRange, ContainsAddr) {
  VMRange range(0x100, 0x200);

  EXPECT_FALSE(range.Contains(0x00));
  EXPECT_FALSE(range.Contains(0xFF));
  EXPECT_TRUE(range.Contains(0x100));
  EXPECT_TRUE(range.Contains(0x101));
  EXPECT_TRUE(range.Contains(0x1FF));
  EXPECT_FALSE(range.Contains(0x200));
  EXPECT_FALSE(range.Contains(0x201));
  EXPECT_FALSE(range.Contains(0xFFF));
  EXPECT_FALSE(range.Contains(std::numeric_limits<lldb::addr_t>::max()));
}

TEST(VMRange, ContainsRange) {
  VMRange range(0x100, 0x200);

  EXPECT_FALSE(range.Contains(VMRange(0x0, 0x0)));

  EXPECT_FALSE(range.Contains(VMRange(0x0, 0x100)));
  EXPECT_FALSE(range.Contains(VMRange(0x0, 0x101)));
  EXPECT_TRUE(range.Contains(VMRange(0x100, 0x105)));
  EXPECT_TRUE(range.Contains(VMRange(0x101, 0x105)));
  EXPECT_TRUE(range.Contains(VMRange(0x100, 0x1FF)));
  EXPECT_TRUE(range.Contains(VMRange(0x105, 0x200)));
  EXPECT_FALSE(range.Contains(VMRange(0x105, 0x201)));
  EXPECT_FALSE(range.Contains(VMRange(0x200, 0x201)));
  EXPECT_TRUE(range.Contains(VMRange(0x100, 0x200)));
  EXPECT_FALSE(
      range.Contains(VMRange(0x105, std::numeric_limits<lldb::addr_t>::max())));

  // Empty range.
  EXPECT_TRUE(range.Contains(VMRange(0x100, 0x100)));

  range.Clear();
  EXPECT_FALSE(range.Contains(VMRange(0x0, 0x0)));
}

TEST(VMRange, Ordering) {
  VMRange range1(0x44, 0x200);
  VMRange range2(0x100, 0x1FF);
  VMRange range3(0x100, 0x200);

  EXPECT_LE(range1, range1);
  EXPECT_GE(range1, range1);

  EXPECT_LT(range1, range2);
  EXPECT_LT(range2, range3);

  EXPECT_GT(range2, range1);
  EXPECT_GT(range3, range2);

  // Ensure that < and > are always false when comparing ranges with themselves.
  EXPECT_FALSE(range1 < range1);
  EXPECT_FALSE(range2 < range2);
  EXPECT_FALSE(range3 < range3);

  EXPECT_FALSE(range1 > range1);
  EXPECT_FALSE(range2 > range2);
  EXPECT_FALSE(range3 > range3);
}

TEST(VMRange, CollectionContains) {
  VMRange::collection collection = {VMRange(0x100, 0x105),
                                    VMRange(0x108, 0x110)};

  EXPECT_FALSE(VMRange::ContainsValue(collection, 0xFF));
  EXPECT_TRUE(VMRange::ContainsValue(collection, 0x100));
  EXPECT_FALSE(VMRange::ContainsValue(collection, 0x105));
  EXPECT_TRUE(VMRange::ContainsValue(collection, 0x109));

  EXPECT_TRUE(VMRange::ContainsRange(collection, VMRange(0x100, 0x104)));
  EXPECT_TRUE(VMRange::ContainsRange(collection, VMRange(0x108, 0x100)));
  EXPECT_FALSE(VMRange::ContainsRange(collection, VMRange(0xFF, 0x100)));

  // TODO: Implement and test ContainsRange with values that span multiple
  // ranges in the collection.
}
