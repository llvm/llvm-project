//=== - unittest/Support/OptimizedStructLayoutTest.cpp - Layout tests -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/OptimizedStructLayout.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

class LayoutTest {
  struct Field {
    uint64_t Size;
    Align Alignment;
    uint64_t ForcedOffset;
    uint64_t ExpectedOffset;
  };

  SmallVector<Field, 16> Fields;
  bool Verified = false;

public:
  LayoutTest() {}
  LayoutTest(const LayoutTest &) = delete;
  LayoutTest &operator=(const LayoutTest &) = delete;
  ~LayoutTest() { assert(Verified); }

  LayoutTest &flexible(uint64_t Size, uint64_t Alignment,
                       uint64_t ExpectedOffset) {
    Fields.push_back({Size, Align(Alignment),
                      OptimizedStructLayoutField::FlexibleOffset, ExpectedOffset});
    return *this;
  }

  LayoutTest &fixed(uint64_t Size, uint64_t Alignment, uint64_t Offset) {
    Fields.push_back({Size, Align(Alignment), Offset, Offset});
    return *this;
  }

  void verify(uint64_t ExpectedSize, uint64_t ExpectedAlignment) {
    SmallVector<OptimizedStructLayoutField, 8> LayoutFields;
    LayoutFields.reserve(Fields.size());
    for (auto &F : Fields)
      LayoutFields.emplace_back(&F, F.Size, F.Alignment, F.ForcedOffset);

    auto SizeAndAlign = performOptimizedStructLayout(LayoutFields);

    EXPECT_EQ(SizeAndAlign.first, ExpectedSize);
    EXPECT_EQ(SizeAndAlign.second, Align(ExpectedAlignment));

    for (auto &LF : LayoutFields) {
      auto &F = *static_cast<const Field *>(LF.Id);
      EXPECT_EQ(LF.Offset, F.ExpectedOffset);
    }

    Verified = true;
  }
};

}

TEST(OptimizedStructLayoutTest, Basic) {
  LayoutTest()
    .flexible(12, 4, 8)
    .flexible(8,  8, 0)
    .flexible(4,  4, 20)
    .verify(24, 8);
}

TEST(OptimizedStructLayoutTest, OddSize) {
  LayoutTest()
    .flexible(8,  8, 16)
    .flexible(4,  4, 12)
    .flexible(1,  1, 10)
    .flexible(10, 8, 0)
    .verify(24, 8);
}

TEST(OptimizedStructLayoutTest, Gaps) {
  LayoutTest()
    .fixed(4, 4, 8)
    .fixed(4, 4, 16)
    .flexible(4, 4, 0)
    .flexible(4, 4, 4)
    .flexible(4, 4, 12)
    .flexible(4, 4, 20)
    .verify(24, 4);
}

TEST(OptimizedStructLayoutTest, Greed) {
  // The greedy algorithm doesn't find the optimal layout here, which
  // would be to put the 5-byte field at the end.
  LayoutTest()
    .fixed(4, 4, 8)
    .flexible(5, 4, 0)
    .flexible(4, 4, 12)
    .flexible(4, 4, 16)
    .flexible(4, 4, 20)
    .verify(24, 4);
}

TEST(OptimizedStructLayoutTest, Jagged) {
  LayoutTest()
    .flexible(1, 2, 18)
    .flexible(13, 8, 0)
    .flexible(3, 2, 14)
    .verify(19, 8);
}

TEST(OptimizedStructLayoutTest, GardenPath) {
  // The 4-byte-aligned field is our highest priority, but the less-aligned
  // fields keep leaving the end offset mis-aligned.
  LayoutTest()
    .fixed(7, 4, 0)
    .flexible(4, 4, 44)
    .flexible(6, 1, 7)
    .flexible(5, 1, 13)
    .flexible(7, 2, 18)
    .flexible(4, 1, 25)
    .flexible(4, 1, 29)
    .flexible(1, 1, 33)
    .flexible(4, 2, 34)
    .flexible(4, 2, 38)
    .flexible(2, 2, 42)
    .flexible(2, 2, 48)
    .verify(50, 4);
}

// PR 51131
TEST(OptimizedStructLayoutTest, HighAlignment) {
  // Handle the case where a flexible field has such a high alignment
  // requirement that aligning LastEnd to it gives an offset past the
  // end of the gap before the next fixed-alignment field.
  LayoutTest()
    .fixed(8, 8, 0)
    .fixed(8, 8, 8)
    .fixed(64, 64, 64)
    .flexible(1, 1, 16)
    .flexible(1, 1, 17)
    .flexible(4, 128, 128)
    .flexible(1, 1, 18)
    .flexible(1, 1, 19)
    .verify(132, 128);
}
