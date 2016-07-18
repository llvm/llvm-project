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

TEST(MathExtras, isIntN) {
  EXPECT_TRUE(isIntN(16, 32767));
  EXPECT_FALSE(isIntN(16, 32768));
}

TEST(MathExtras, isUIntN) {
  EXPECT_TRUE(isUIntN(16, 65535));
  EXPECT_FALSE(isUIntN(16, 65536));
  EXPECT_TRUE(isUIntN(1, 0));
  EXPECT_TRUE(isUIntN(6, 63));
}

TEST(MathExtras, maxIntN) {
  EXPECT_EQ(32767, maxIntN(16));
  EXPECT_EQ(2147483647, maxIntN(32));
  EXPECT_EQ(std::numeric_limits<int32_t>::max(), maxIntN(32));
  EXPECT_EQ(std::numeric_limits<int64_t>::max(), maxIntN(64));
}

TEST(MathExtras, minIntN) {
  EXPECT_EQ(-32768LL, minIntN(16));
  EXPECT_EQ(-64LL, minIntN(7));
  EXPECT_EQ(std::numeric_limits<int32_t>::min(), minIntN(32));
  EXPECT_EQ(std::numeric_limits<int64_t>::min(), minIntN(64));
}

TEST(MathExtras, maxUIntN) {
  EXPECT_EQ(0xffffULL, maxUIntN(16));
  EXPECT_EQ(0xffffffffULL, maxUIntN(32));
  EXPECT_EQ(0xffffffffffffffffULL, maxUIntN(64));
  EXPECT_EQ(1ULL, maxUIntN(1));
  EXPECT_EQ(0x0fULL, maxUIntN(4));
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
  EXPECT_DOUBLE_EQ(kValue, BitsToDouble(DoubleToBits(kValue)));
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

TEST(MathExtras, alignTo) {
  EXPECT_EQ(8u, alignTo(5, 8));
  EXPECT_EQ(24u, alignTo(17, 8));
  EXPECT_EQ(0u, alignTo(~0LL, 8));

  EXPECT_EQ(7u, alignTo(5, 8, 7));
  EXPECT_EQ(17u, alignTo(17, 8, 1));
  EXPECT_EQ(3u, alignTo(~0LL, 8, 3));
  EXPECT_EQ(552u, alignTo(321, 255, 42));
}

template<typename T>
void SaturatingAddTestHelper()
{
  const T Max = std::numeric_limits<T>::max();
  bool ResultOverflowed;

  EXPECT_EQ(T(3), SaturatingAdd(T(1), T(2)));
  EXPECT_EQ(T(3), SaturatingAdd(T(1), T(2), &ResultOverflowed));
  EXPECT_FALSE(ResultOverflowed);

  EXPECT_EQ(Max, SaturatingAdd(Max, T(1)));
  EXPECT_EQ(Max, SaturatingAdd(Max, T(1), &ResultOverflowed));
  EXPECT_TRUE(ResultOverflowed);

  EXPECT_EQ(Max, SaturatingAdd(T(1), T(Max - 1)));
  EXPECT_EQ(Max, SaturatingAdd(T(1), T(Max - 1), &ResultOverflowed));
  EXPECT_FALSE(ResultOverflowed);

  EXPECT_EQ(Max, SaturatingAdd(T(1), Max));
  EXPECT_EQ(Max, SaturatingAdd(T(1), Max, &ResultOverflowed));
  EXPECT_TRUE(ResultOverflowed);

  EXPECT_EQ(Max, SaturatingAdd(Max, Max));
  EXPECT_EQ(Max, SaturatingAdd(Max, Max, &ResultOverflowed));
  EXPECT_TRUE(ResultOverflowed);
}

TEST(MathExtras, SaturatingAdd) {
  SaturatingAddTestHelper<uint8_t>();
  SaturatingAddTestHelper<uint16_t>();
  SaturatingAddTestHelper<uint32_t>();
  SaturatingAddTestHelper<uint64_t>();
}

template<typename T>
void SaturatingMultiplyTestHelper()
{
  const T Max = std::numeric_limits<T>::max();
  bool ResultOverflowed;

  // Test basic multiplication.
  EXPECT_EQ(T(6), SaturatingMultiply(T(2), T(3)));
  EXPECT_EQ(T(6), SaturatingMultiply(T(2), T(3), &ResultOverflowed));
  EXPECT_FALSE(ResultOverflowed);

  EXPECT_EQ(T(6), SaturatingMultiply(T(3), T(2)));
  EXPECT_EQ(T(6), SaturatingMultiply(T(3), T(2), &ResultOverflowed));
  EXPECT_FALSE(ResultOverflowed);

  // Test multiplication by zero.
  EXPECT_EQ(T(0), SaturatingMultiply(T(0), T(0)));
  EXPECT_EQ(T(0), SaturatingMultiply(T(0), T(0), &ResultOverflowed));
  EXPECT_FALSE(ResultOverflowed);

  EXPECT_EQ(T(0), SaturatingMultiply(T(1), T(0)));
  EXPECT_EQ(T(0), SaturatingMultiply(T(1), T(0), &ResultOverflowed));
  EXPECT_FALSE(ResultOverflowed);

  EXPECT_EQ(T(0), SaturatingMultiply(T(0), T(1)));
  EXPECT_EQ(T(0), SaturatingMultiply(T(0), T(1), &ResultOverflowed));
  EXPECT_FALSE(ResultOverflowed);

  EXPECT_EQ(T(0), SaturatingMultiply(Max, T(0)));
  EXPECT_EQ(T(0), SaturatingMultiply(Max, T(0), &ResultOverflowed));
  EXPECT_FALSE(ResultOverflowed);

  EXPECT_EQ(T(0), SaturatingMultiply(T(0), Max));
  EXPECT_EQ(T(0), SaturatingMultiply(T(0), Max, &ResultOverflowed));
  EXPECT_FALSE(ResultOverflowed);

  // Test multiplication by maximum value.
  EXPECT_EQ(Max, SaturatingMultiply(Max, T(2)));
  EXPECT_EQ(Max, SaturatingMultiply(Max, T(2), &ResultOverflowed));
  EXPECT_TRUE(ResultOverflowed);

  EXPECT_EQ(Max, SaturatingMultiply(T(2), Max));
  EXPECT_EQ(Max, SaturatingMultiply(T(2), Max, &ResultOverflowed));
  EXPECT_TRUE(ResultOverflowed);

  EXPECT_EQ(Max, SaturatingMultiply(Max, Max));
  EXPECT_EQ(Max, SaturatingMultiply(Max, Max, &ResultOverflowed));
  EXPECT_TRUE(ResultOverflowed);

  // Test interesting boundary conditions for algorithm -
  // ((1 << A) - 1) * ((1 << B) + K) for K in [-1, 0, 1]
  // and A + B == std::numeric_limits<T>::digits.
  // We expect overflow iff A > B and K = 1.
  const int Digits = std::numeric_limits<T>::digits;
  for (int A = 1, B = Digits - 1; B >= 1; ++A, --B) {
    for (int K = -1; K <= 1; ++K) {
      T X = (T(1) << A) - T(1);
      T Y = (T(1) << B) + K;
      bool OverflowExpected = A > B && K == 1;

      if(OverflowExpected) {
        EXPECT_EQ(Max, SaturatingMultiply(X, Y));
        EXPECT_EQ(Max, SaturatingMultiply(X, Y, &ResultOverflowed));
        EXPECT_TRUE(ResultOverflowed);
      } else {
        EXPECT_EQ(X * Y, SaturatingMultiply(X, Y));
        EXPECT_EQ(X * Y, SaturatingMultiply(X, Y, &ResultOverflowed));
        EXPECT_FALSE(ResultOverflowed);
      }
    }
  }
}

TEST(MathExtras, SaturatingMultiply) {
  SaturatingMultiplyTestHelper<uint8_t>();
  SaturatingMultiplyTestHelper<uint16_t>();
  SaturatingMultiplyTestHelper<uint32_t>();
  SaturatingMultiplyTestHelper<uint64_t>();
}

template<typename T>
void SaturatingMultiplyAddTestHelper()
{
  const T Max = std::numeric_limits<T>::max();
  bool ResultOverflowed;

  // Test basic multiply-add.
  EXPECT_EQ(T(16), SaturatingMultiplyAdd(T(2), T(3), T(10)));
  EXPECT_EQ(T(16), SaturatingMultiplyAdd(T(2), T(3), T(10), &ResultOverflowed));
  EXPECT_FALSE(ResultOverflowed);

  // Test multiply overflows, add doesn't overflow
  EXPECT_EQ(Max, SaturatingMultiplyAdd(Max, Max, T(0), &ResultOverflowed));
  EXPECT_TRUE(ResultOverflowed);

  // Test multiply doesn't overflow, add overflows
  EXPECT_EQ(Max, SaturatingMultiplyAdd(T(1), T(1), Max, &ResultOverflowed));
  EXPECT_TRUE(ResultOverflowed);

  // Test multiply-add with Max as operand
  EXPECT_EQ(Max, SaturatingMultiplyAdd(T(1), T(1), Max, &ResultOverflowed));
  EXPECT_TRUE(ResultOverflowed);

  EXPECT_EQ(Max, SaturatingMultiplyAdd(T(1), Max, T(1), &ResultOverflowed));
  EXPECT_TRUE(ResultOverflowed);

  EXPECT_EQ(Max, SaturatingMultiplyAdd(Max, Max, T(1), &ResultOverflowed));
  EXPECT_TRUE(ResultOverflowed);

  EXPECT_EQ(Max, SaturatingMultiplyAdd(Max, Max, Max, &ResultOverflowed));
  EXPECT_TRUE(ResultOverflowed);

  // Test multiply-add with 0 as operand
  EXPECT_EQ(T(1), SaturatingMultiplyAdd(T(1), T(1), T(0), &ResultOverflowed));
  EXPECT_FALSE(ResultOverflowed);

  EXPECT_EQ(T(1), SaturatingMultiplyAdd(T(1), T(0), T(1), &ResultOverflowed));
  EXPECT_FALSE(ResultOverflowed);

  EXPECT_EQ(T(1), SaturatingMultiplyAdd(T(0), T(0), T(1), &ResultOverflowed));
  EXPECT_FALSE(ResultOverflowed);

  EXPECT_EQ(T(0), SaturatingMultiplyAdd(T(0), T(0), T(0), &ResultOverflowed));
  EXPECT_FALSE(ResultOverflowed);

}

TEST(MathExtras, SaturatingMultiplyAdd) {
  SaturatingMultiplyAddTestHelper<uint8_t>();
  SaturatingMultiplyAddTestHelper<uint16_t>();
  SaturatingMultiplyAddTestHelper<uint32_t>();
  SaturatingMultiplyAddTestHelper<uint64_t>();
}

TEST(MathExtras, IsShiftedUInt) {
  EXPECT_TRUE((isShiftedUInt<1, 0>(0)));
  EXPECT_TRUE((isShiftedUInt<1, 0>(1)));
  EXPECT_FALSE((isShiftedUInt<1, 0>(2)));
  EXPECT_FALSE((isShiftedUInt<1, 0>(3)));
  EXPECT_FALSE((isShiftedUInt<1, 0>(0x8000000000000000)));
  EXPECT_TRUE((isShiftedUInt<1, 63>(0x8000000000000000)));
  EXPECT_TRUE((isShiftedUInt<2, 62>(0xC000000000000000)));
  EXPECT_FALSE((isShiftedUInt<2, 62>(0xE000000000000000)));

  // 0x201 is ten bits long and has a 1 in the MSB and LSB.
  EXPECT_TRUE((isShiftedUInt<10, 5>(uint64_t(0x201) << 5)));
  EXPECT_FALSE((isShiftedUInt<10, 5>(uint64_t(0x201) << 4)));
  EXPECT_FALSE((isShiftedUInt<10, 5>(uint64_t(0x201) << 6)));
}

TEST(MathExtras, IsShiftedInt) {
  EXPECT_TRUE((isShiftedInt<1, 0>(0)));
  EXPECT_TRUE((isShiftedInt<1, 0>(-1)));
  EXPECT_FALSE((isShiftedInt<1, 0>(2)));
  EXPECT_FALSE((isShiftedInt<1, 0>(3)));
  EXPECT_FALSE((isShiftedInt<1, 0>(0x8000000000000000)));
  EXPECT_TRUE((isShiftedInt<1, 63>(0x8000000000000000)));
  EXPECT_TRUE((isShiftedInt<2, 62>(0xC000000000000000)));
  EXPECT_FALSE((isShiftedInt<2, 62>(0xE000000000000000)));

  // 0x201 is ten bits long and has a 1 in the MSB and LSB.
  EXPECT_TRUE((isShiftedInt<11, 5>(int64_t(0x201) << 5)));
  EXPECT_FALSE((isShiftedInt<11, 5>(int64_t(0x201) << 3)));
  EXPECT_FALSE((isShiftedInt<11, 5>(int64_t(0x201) << 6)));
  EXPECT_TRUE((isShiftedInt<11, 5>(-(int64_t(0x201) << 5))));
  EXPECT_FALSE((isShiftedInt<11, 5>(-(int64_t(0x201) << 3))));
  EXPECT_FALSE((isShiftedInt<11, 5>(-(int64_t(0x201) << 6))));

  EXPECT_TRUE((isShiftedInt<6, 10>(-(int64_t(1) << 15))));
  EXPECT_FALSE((isShiftedInt<6, 10>(int64_t(1) << 15)));
}

} // namespace
