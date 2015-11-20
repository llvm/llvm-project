//===- unittests/Support/MathExtrasTest.cpp - math utils tests ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "llvm/Support/MathExtras.h"

using namespace llvm;

namespace {

TEST(MathExtras, countTrailingZeros) {
  uint8_t Z8 = 0;
  uint16_t Z16 = 0;
  uint32_t Z32 = 0;
  uint64_t Z64 = 0;
  EXPECT_EQ(8u, countTrailingZeros(Z8));
  EXPECT_EQ(16u, countTrailingZeros(Z16));
  EXPECT_EQ(32u, countTrailingZeros(Z32));
  EXPECT_EQ(64u, countTrailingZeros(Z64));

  uint8_t NZ8 = 42;
  uint16_t NZ16 = 42;
  uint32_t NZ32 = 42;
  uint64_t NZ64 = 42;
  EXPECT_EQ(1u, countTrailingZeros(NZ8));
  EXPECT_EQ(1u, countTrailingZeros(NZ16));
  EXPECT_EQ(1u, countTrailingZeros(NZ32));
  EXPECT_EQ(1u, countTrailingZeros(NZ64));
}

TEST(MathExtras, countLeadingZeros) {
  uint8_t Z8 = 0;
  uint16_t Z16 = 0;
  uint32_t Z32 = 0;
  uint64_t Z64 = 0;
  EXPECT_EQ(8u, countLeadingZeros(Z8));
  EXPECT_EQ(16u, countLeadingZeros(Z16));
  EXPECT_EQ(32u, countLeadingZeros(Z32));
  EXPECT_EQ(64u, countLeadingZeros(Z64));

  uint8_t NZ8 = 42;
  uint16_t NZ16 = 42;
  uint32_t NZ32 = 42;
  uint64_t NZ64 = 42;
  EXPECT_EQ(2u, countLeadingZeros(NZ8));
  EXPECT_EQ(10u, countLeadingZeros(NZ16));
  EXPECT_EQ(26u, countLeadingZeros(NZ32));
  EXPECT_EQ(58u, countLeadingZeros(NZ64));

  EXPECT_EQ(8u, countLeadingZeros(0x00F000FFu));
  EXPECT_EQ(8u, countLeadingZeros(0x00F12345u));
  for (unsigned i = 0; i <= 30; ++i) {
    EXPECT_EQ(31 - i, countLeadingZeros(1u << i));
  }

  EXPECT_EQ(8u, countLeadingZeros(0x00F1234500F12345ULL));
  EXPECT_EQ(1u, countLeadingZeros(1ULL << 62));
  for (unsigned i = 0; i <= 62; ++i) {
    EXPECT_EQ(63 - i, countLeadingZeros(1ULL << i));
  }
}

TEST(MathExtras, findFirstSet) {
  uint8_t Z8 = 0;
  uint16_t Z16 = 0;
  uint32_t Z32 = 0;
  uint64_t Z64 = 0;
  EXPECT_EQ(0xFFULL, findFirstSet(Z8));
  EXPECT_EQ(0xFFFFULL, findFirstSet(Z16));
  EXPECT_EQ(0xFFFFFFFFULL, findFirstSet(Z32));
  EXPECT_EQ(0xFFFFFFFFFFFFFFFFULL, findFirstSet(Z64));

  uint8_t NZ8 = 42;
  uint16_t NZ16 = 42;
  uint32_t NZ32 = 42;
  uint64_t NZ64 = 42;
  EXPECT_EQ(1u, findFirstSet(NZ8));
  EXPECT_EQ(1u, findFirstSet(NZ16));
  EXPECT_EQ(1u, findFirstSet(NZ32));
  EXPECT_EQ(1u, findFirstSet(NZ64));
}

TEST(MathExtras, findLastSet) {
  uint8_t Z8 = 0;
  uint16_t Z16 = 0;
  uint32_t Z32 = 0;
  uint64_t Z64 = 0;
  EXPECT_EQ(0xFFULL, findLastSet(Z8));
  EXPECT_EQ(0xFFFFULL, findLastSet(Z16));
  EXPECT_EQ(0xFFFFFFFFULL, findLastSet(Z32));
  EXPECT_EQ(0xFFFFFFFFFFFFFFFFULL, findLastSet(Z64));

  uint8_t NZ8 = 42;
  uint16_t NZ16 = 42;
  uint32_t NZ32 = 42;
  uint64_t NZ64 = 42;
  EXPECT_EQ(5u, findLastSet(NZ8));
  EXPECT_EQ(5u, findLastSet(NZ16));
  EXPECT_EQ(5u, findLastSet(NZ32));
  EXPECT_EQ(5u, findLastSet(NZ64));
}

TEST(MathExtras, reverseBits) {
  uint8_t NZ8 = 42;
  uint16_t NZ16 = 42;
  uint32_t NZ32 = 42;
  uint64_t NZ64 = 42;
  EXPECT_EQ(0x54ULL, reverseBits(NZ8));
  EXPECT_EQ(0x5400ULL, reverseBits(NZ16));
  EXPECT_EQ(0x54000000ULL, reverseBits(NZ32));
  EXPECT_EQ(0x5400000000000000ULL, reverseBits(NZ64));
}

TEST(MathExtras, isPowerOf2_32) {
  EXPECT_TRUE(isPowerOf2_32(1 << 6));
  EXPECT_TRUE(isPowerOf2_32(1 << 12));
  EXPECT_FALSE(isPowerOf2_32((1 << 19) + 3));
  EXPECT_FALSE(isPowerOf2_32(0xABCDEF0));
}

TEST(MathExtras, isPowerOf2_64) {
  EXPECT_TRUE(isPowerOf2_64(1LL << 46));
  EXPECT_TRUE(isPowerOf2_64(1LL << 12));
  EXPECT_FALSE(isPowerOf2_64((1LL << 53) + 3));
  EXPECT_FALSE(isPowerOf2_64(0xABCDEF0ABCDEF0LL));
}

TEST(MathExtras, ByteSwap_32) {
  EXPECT_EQ(0x44332211u, ByteSwap_32(0x11223344));
  EXPECT_EQ(0xDDCCBBAAu, ByteSwap_32(0xAABBCCDD));
}

TEST(MathExtras, ByteSwap_64) {
  EXPECT_EQ(0x8877665544332211ULL, ByteSwap_64(0x1122334455667788LL));
  EXPECT_EQ(0x1100FFEEDDCCBBAAULL, ByteSwap_64(0xAABBCCDDEEFF0011LL));
}

TEST(MathExtras, countLeadingOnes) {
  for (int i = 30; i >= 0; --i) {
    // Start with all ones and unset some bit.
    EXPECT_EQ(31u - i, countLeadingOnes(0xFFFFFFFF ^ (1 << i)));
  }
  for (int i = 62; i >= 0; --i) {
    // Start with all ones and unset some bit.
    EXPECT_EQ(63u - i, countLeadingOnes(0xFFFFFFFFFFFFFFFFULL ^ (1LL << i)));
  }
  for (int i = 30; i >= 0; --i) {
    // Start with all ones and unset some bit.
    EXPECT_EQ(31u - i, countLeadingOnes(0xFFFFFFFF ^ (1 << i)));
  }
}

TEST(MathExtras, FloatBits) {
  static const float kValue = 5632.34f;
  EXPECT_FLOAT_EQ(kValue, BitsToFloat(FloatToBits(kValue)));
}

TEST(MathExtras, DoubleBits) {
  static const double kValue = 87987234.983498;
  EXPECT_FLOAT_EQ(kValue, BitsToDouble(DoubleToBits(kValue)));
}

TEST(MathExtras, MinAlign) {
  EXPECT_EQ(1u, MinAlign(2, 3));
  EXPECT_EQ(2u, MinAlign(2, 4));
  EXPECT_EQ(1u, MinAlign(17, 64));
  EXPECT_EQ(256u, MinAlign(256, 512));
}

TEST(MathExtras, NextPowerOf2) {
  EXPECT_EQ(4u, NextPowerOf2(3));
  EXPECT_EQ(16u, NextPowerOf2(15));
  EXPECT_EQ(256u, NextPowerOf2(128));
}

TEST(MathExtras, RoundUpToAlignment) {
  EXPECT_EQ(8u, RoundUpToAlignment(5, 8));
  EXPECT_EQ(24u, RoundUpToAlignment(17, 8));
  EXPECT_EQ(0u, RoundUpToAlignment(~0LL, 8));

  EXPECT_EQ(7u, RoundUpToAlignment(5, 8, 7));
  EXPECT_EQ(17u, RoundUpToAlignment(17, 8, 1));
  EXPECT_EQ(3u, RoundUpToAlignment(~0LL, 8, 3));
  EXPECT_EQ(552u, RoundUpToAlignment(321, 255, 42));
}

template<typename T>
void SaturatingAddTestHelper()
{
  const T Max = std::numeric_limits<T>::max();
  EXPECT_EQ(T(3), SaturatingAdd(T(1), T(2)));
  EXPECT_EQ(Max, SaturatingAdd(Max, T(1)));
  EXPECT_EQ(Max, SaturatingAdd(T(1), Max));
  EXPECT_EQ(Max, SaturatingAdd(Max, Max));
}

TEST(MathExtras, SaturatingAdd) {
  SaturatingAddTestHelper<uint8_t>();
  SaturatingAddTestHelper<uint16_t>();
  SaturatingAddTestHelper<uint32_t>();
  SaturatingAddTestHelper<uint64_t>();
}

}
