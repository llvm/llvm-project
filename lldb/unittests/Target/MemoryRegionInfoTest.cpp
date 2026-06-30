//===-- MemoryRegionInfoTest.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/MemoryRegionInfo.h"
#include "lldb/Target/MemoryRegionInfoCache.h"
#include "lldb/Utility/RangeMap.h"
#include "llvm/Support/FormatVariadic.h"
#include "gtest/gtest.h"

using namespace lldb_private;

TEST(MemoryRegionInfoTest, Formatv) {
  EXPECT_EQ("yes", llvm::formatv("{0}", eLazyBoolYes).str());
  EXPECT_EQ("no", llvm::formatv("{0}", eLazyBoolNo).str());
  EXPECT_EQ("don't know", llvm::formatv("{0}", eLazyBoolDontKnow).str());
}

TEST(MemoryRegionInfoTest, CacheErasing) {
  MemoryRegionInfoCache cache;
  cache.AddRegion(MemoryRegionInfo({0x1000, 0x100}, eLazyBoolYes, eLazyBoolYes,
                                   eLazyBoolYes, eLazyBoolNo, eLazyBoolNo,
                                   ConstString("First entry")));
  cache.AddRegion(MemoryRegionInfo({0x2000, 0x100}, eLazyBoolYes, eLazyBoolYes,
                                   eLazyBoolYes, eLazyBoolNo, eLazyBoolNo,
                                   ConstString("Second entry")));
  cache.AddRegion(MemoryRegionInfo({0x2100, 0x100}, eLazyBoolYes, eLazyBoolYes,
                                   eLazyBoolYes, eLazyBoolNo, eLazyBoolNo,
                                   ConstString("Third entry")));
  cache.AddRegion(MemoryRegionInfo({0x2200, 0x100}, eLazyBoolYes, eLazyBoolYes,
                                   eLazyBoolYes, eLazyBoolNo, eLazyBoolNo,
                                   ConstString("Fourth entry")));
  cache.AddRegion(MemoryRegionInfo({0x2300, 0x100}, eLazyBoolYes, eLazyBoolYes,
                                   eLazyBoolYes, eLazyBoolNo, eLazyBoolNo,
                                   ConstString("Fifth entry")));
  cache.AddRegion(MemoryRegionInfo({0x5000, 0x100}, eLazyBoolYes, eLazyBoolYes,
                                   eLazyBoolYes, eLazyBoolNo, eLazyBoolNo,
                                   ConstString("Sixth entry")));
  cache.AddRegion(MemoryRegionInfo({0x6000, 0x100}, eLazyBoolYes, eLazyBoolYes,
                                   eLazyBoolYes, eLazyBoolNo, eLazyBoolNo,
                                   ConstString("Seventh entry")));

  std::optional<MemoryRegionInfo> ri = cache.GetMemoryRegion(0x6000);
  ASSERT_TRUE(ri);

  // Erase the last entry.
  ASSERT_EQ(cache.GetSize(), 7U);
  cache.EraseContaining(0x6000);
  ASSERT_EQ(cache.GetSize(), 6U);
  std::optional<MemoryRegionInfo> erased_ri = cache.GetMemoryRegion(0x6000);
  ASSERT_FALSE(erased_ri);
  cache.EraseContaining(0x6000); // no-op
  ASSERT_EQ(cache.GetSize(), 6U);

  // Erase the last entry & beyond.
  cache.EraseRange(0x5000, 0x5000000);
  ASSERT_EQ(cache.GetSize(), 5U);
  erased_ri = cache.GetMemoryRegion(0x5000);
  ASSERT_FALSE(erased_ri);

  // Erase from before the first entry, through the first entry.
  cache.EraseRange(0x500, 0xb01);
  ASSERT_EQ(cache.GetSize(), 4U);
  erased_ri = cache.GetMemoryRegion(0x1000);
  ASSERT_FALSE(erased_ri);

  // Erase the second and third entries.
  cache.EraseRange(0x2000, 0x101);
  ASSERT_EQ(cache.GetSize(), 2U);
  erased_ri = cache.GetMemoryRegion(0x2000);
  ASSERT_FALSE(erased_ri);
  erased_ri = cache.GetMemoryRegion(0x2100);
  ASSERT_FALSE(erased_ri);

  ri = cache.GetMemoryRegion(0x2200); // Fourth entry still available.
  ASSERT_TRUE(ri);
  ri = cache.GetMemoryRegion(0x2300); // Fifth entry still available.
  ASSERT_TRUE(ri);

  // Erase some entries that don't exist.
  cache.EraseRange(0x2000, 0x101);
  ASSERT_EQ(cache.GetSize(), 2U);
  cache.EraseContaining(0x2400);
  ASSERT_EQ(cache.GetSize(), 2U);

  cache.AddRegion(MemoryRegionInfo({0x0, 0x1000}, eLazyBoolYes, eLazyBoolYes,
                                   eLazyBoolYes, eLazyBoolNo, eLazyBoolNo,
                                   ConstString("Zeroth entry")));

  // Do an Erase that overflows, confirm entry at 0 is not erased.
  cache.EraseContaining(LLDB_INVALID_ADDRESS);
  ASSERT_EQ(cache.GetSize(), 3U);
  ri = cache.GetMemoryRegion(0x0);
  ASSERT_TRUE(ri);
}
