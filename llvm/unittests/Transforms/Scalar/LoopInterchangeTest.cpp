//===- LoopInterchangeTest.cpp - LoopInterchange helper unit tests --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Numeric boundaries of LoopInterchangeUtils.h: normalized INT64_MIN,
// byte-range containment overflow, and lossless wide unsigned comparisons.
//
//===----------------------------------------------------------------------===//

#include "../lib/Transforms/Scalar/LoopInterchangeUtils.h"
#include "llvm/ADT/APInt.h"
#include "gtest/gtest.h"
#include <limits>

using namespace llvm;
using namespace llvm::loop_interchange_utils;

namespace {

TEST(LoopInterchangeUtilsTest, CheckedAbsRejectsInt64Min) {
  // The row stride must fit in int64_t, but |INT64_MIN| is 2^63.
  EXPECT_FALSE(
      checkedAbsToUnsigned(std::numeric_limits<int64_t>::min()).has_value());

  // Ordinary magnitudes, including zero and both extreme representable values.
  auto Zero = checkedAbsToUnsigned(0);
  ASSERT_TRUE(Zero.has_value());
  EXPECT_EQ(*Zero, 0u);

  auto Pos = checkedAbsToUnsigned(1335);
  ASSERT_TRUE(Pos.has_value());
  EXPECT_EQ(*Pos, 1335u);

  auto Neg = checkedAbsToUnsigned(-1335);
  ASSERT_TRUE(Neg.has_value());
  EXPECT_EQ(*Neg, 1335u);

  auto Max = checkedAbsToUnsigned(std::numeric_limits<int64_t>::max());
  ASSERT_TRUE(Max.has_value());
  EXPECT_EQ(*Max, 9223372036854775807ULL);

  auto NegMax = checkedAbsToUnsigned(-std::numeric_limits<int64_t>::max());
  ASSERT_TRUE(NegMax.has_value());
  EXPECT_EQ(*NegMax, 9223372036854775807ULL);
}

TEST(LoopInterchangeUtilsTest, FixedByteRangeWithinObject) {
  // A [4 x [4 x double]] object is 128 bytes; element start offsets run
  // 0, 8, ..., 120 and each element occupies 8 bytes.
  constexpr uint64_t ElemSize = 8;
  constexpr uint64_t ObjSize = 128;

  // In-bounds range: last element starts at byte 120, last byte 127 < 128.
  EXPECT_TRUE(isFixedByteRangeWithinObject(0, 120, ElemSize, ObjSize));

  // Exact upper boundary: last byte == ObjectSize - 1 is still inside.
  EXPECT_TRUE(isFixedByteRangeWithinObject(120, 120, ElemSize, ObjSize));

  // One byte outside: last byte == ObjectSize is rejected.
  EXPECT_FALSE(isFixedByteRangeWithinObject(0, 121, ElemSize, ObjSize));
  EXPECT_FALSE(isFixedByteRangeWithinObject(121, 121, ElemSize, ObjSize));

  // A negative element start is rejected (out of object, not wrapped).
  EXPECT_FALSE(isFixedByteRangeWithinObject(-8, 120, ElemSize, ObjSize));

  // A reversed range (last start before first) is rejected.
  EXPECT_FALSE(isFixedByteRangeWithinObject(120, 0, ElemSize, ObjSize));

  // A zero-sized element occupies no valid byte range.
  EXPECT_FALSE(isFixedByteRangeWithinObject(0, 8, 0, ObjSize));

  // Adding ElementSize - 1 to the last element start must not wrap a huge
  // positive last start into a small in-bounds last byte.
  EXPECT_FALSE(isFixedByteRangeWithinObject(
      std::numeric_limits<int64_t>::max(), std::numeric_limits<int64_t>::max(),
      ElemSize, std::numeric_limits<uint64_t>::max()));

  // An element size too large to be a signed byte offset is rejected.
  EXPECT_FALSE(isFixedByteRangeWithinObject(
      0, 0, static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) + 1,
      ObjSize));
}

TEST(LoopInterchangeUtilsTest, ZeroExtendUnsignedCompareIsLossless) {
  // Ordinary same-width unsigned comparisons.
  EXPECT_TRUE(unsignedLEWithZeroExtend(APInt(64, 1335), APInt(64, 1335)));
  EXPECT_TRUE(unsignedLEWithZeroExtend(APInt(64, 1334), APInt(64, 1335)));
  EXPECT_FALSE(unsignedLEWithZeroExtend(APInt(64, 1336), APInt(64, 1335)));

  // A wide i129 trip whose value needs bit 64 (2^64) must be compared at full
  // width, never truncated toward the i64 bound's width.
  APInt WideTrip = APInt::getOneBitSet(129, 64); // 2^64
  APInt NarrowW(64, 1335);
  std::pair<APInt, APInt> Widened = zeroExtendToCommonWidth(WideTrip, NarrowW);
  EXPECT_EQ(Widened.first.getBitWidth(), 129u);
  EXPECT_EQ(Widened.second.getBitWidth(), 129u);
  // High bits are preserved exactly: the widened trip is still 2^64 and the
  // widened bound is still 1335 (no truncation in either direction).
  EXPECT_EQ(Widened.first, APInt::getOneBitSet(129, 64));
  EXPECT_EQ(Widened.second, APInt(129, 1335));
  // Truncating 2^64 to i64 would wrongly make it u<= 1335.
  EXPECT_FALSE(unsignedLEWithZeroExtend(WideTrip, NarrowW));

  // Other direction: a small i64 trip is u<= a huge i129 bound.
  APInt HugeW = APInt::getOneBitSet(129, 100); // 2^100
  EXPECT_TRUE(unsignedLEWithZeroExtend(APInt(64, 4), HugeW));
  EXPECT_TRUE(unsignedLEWithZeroExtend(HugeW, HugeW));
  EXPECT_FALSE(unsignedLEWithZeroExtend(HugeW, APInt(64, 4)));

  // Bit 63 is a value bit for this unsigned comparison. This distinguishes
  // zero extension from sign extension and ule from sle.
  APInt HighBit64 = APInt::getOneBitSet(64, 63);
  EXPECT_FALSE(unsignedLEWithZeroExtend(HighBit64, NarrowW));
  std::pair<APInt, APInt> HighBitWidened =
      zeroExtendToCommonWidth(HighBit64, APInt(129, 0));
  EXPECT_EQ(HighBitWidened.first, APInt::getOneBitSet(129, 63));
  EXPECT_EQ(HighBitWidened.second, APInt(129, 0));
}

TEST(LoopInterchangeUtilsTest, ByteOffsetProofDiscriminatesHigh32) {
  // The bound_offset_high32 test case in outer-epilogue-fission-bounds.ll uses
  // the fixed epilogue offsets 0x10 and 0x100000010, whose low 32 bits agree
  // while their high 32 bits differ. These checks cover the full-width
  // arithmetic of the helpers.
  constexpr int64_t Lo = 0x10;        // 16
  constexpr int64_t Hi = 0x100000010; // 4294967312, low 32 bits == 0x10

  // The two offsets are distinct 64-bit values whose 32-bit truncations
  // collide.
  EXPECT_NE(Lo, Hi);
  EXPECT_EQ(static_cast<uint32_t>(Lo), static_cast<uint32_t>(Hi));

  // Full-width magnitudes keep these two offsets distinct.
  auto AbsLo = checkedAbsToUnsigned(Lo);
  auto AbsHi = checkedAbsToUnsigned(Hi);
  ASSERT_TRUE(AbsLo.has_value());
  ASSERT_TRUE(AbsHi.has_value());
  EXPECT_EQ(*AbsLo, 0x10ULL);
  EXPECT_EQ(*AbsHi, 0x100000010ULL);
  EXPECT_NE(*AbsLo, *AbsHi);

  // Both 8-byte accesses fit in 0x100000018 bytes. Hi starts the last element.
  constexpr uint64_t ElemSize = 8;
  constexpr uint64_t FixtureObj = 0x100000018;
  constexpr uint64_t ShrunkObj = 0x100000010;
  EXPECT_TRUE(isFixedByteRangeWithinObject(Lo, Lo, ElemSize, FixtureObj));
  EXPECT_TRUE(isFixedByteRangeWithinObject(Hi, Hi, ElemSize, FixtureObj));
  // Only Lo fits in 0x100000010 bytes. Truncating Hi to 32 bits would wrongly
  // accept it as 0x10.
  EXPECT_TRUE(isFixedByteRangeWithinObject(Lo, Lo, ElemSize, ShrunkObj));
  EXPECT_FALSE(isFixedByteRangeWithinObject(Hi, Hi, ElemSize, ShrunkObj));

  // The shared APInt comparison helper orders the two offsets at full width:
  // 0x10 u<= 0x100000010 but not the reverse. A truncation to 32 bits would
  // make both directions trivially true.
  APInt LoAP(64, static_cast<uint64_t>(Lo));
  APInt HiAP(64, static_cast<uint64_t>(Hi));
  EXPECT_TRUE(unsignedLEWithZeroExtend(LoAP, HiAP));
  EXPECT_FALSE(unsignedLEWithZeroExtend(HiAP, LoAP));
}

} // namespace
