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

TEST(MathExtras, CountLeadingZeros_32) {
  EXPECT_EQ(8u, CountLeadingZeros_32(0x00F000FF));
  EXPECT_EQ(8u, CountLeadingZeros_32(0x00F12345));
  for (unsigned i = 0; i <= 30; ++i) {
    EXPECT_EQ(31 - i, CountLeadingZeros_32(1 << i));
  }
}

TEST(MathExtras, CountLeadingZeros_64) {
  EXPECT_EQ(8u, CountLeadingZeros_64(0x00F1234500F12345LL));
  EXPECT_EQ(1u, CountLeadingZeros_64(1LL << 62));
  for (unsigned i = 0; i <= 62; ++i) {
    EXPECT_EQ(63 - i, CountLeadingZeros_64(1LL << i));
  }
}

TEST(MathExtras, CountLeadingOnes_32) {
  for (int i = 30; i >= 0; --i) {
    // Start with all ones and unset some bit.
    EXPECT_EQ(31u - i, CountLeadingOnes_32(0xFFFFFFFF ^ (1 << i)));
  }
}

TEST(MathExtras, CountLeadingOnes_64) {
  for (int i = 62; i >= 0; --i) {
    // Start with all ones and unset some bit.
    EXPECT_EQ(63u - i, CountLeadingOnes_64(0xFFFFFFFFFFFFFFFFLL ^ (1LL << i)));
  }
  for (int i = 30; i >= 0; --i) {
    // Start with all ones and unset some bit.
    EXPECT_EQ(31u - i, CountLeadingOnes_32(0xFFFFFFFF ^ (1 << i)));
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
}

}
