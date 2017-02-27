//===- llvm/unittest/ADT/APInt.cpp - APInt unit tests ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "gtest/gtest.h"
#include <array>

using namespace llvm;

namespace {

TEST(APIntTest, ValueInit) {
  APInt Zero = APInt();
  EXPECT_TRUE(!Zero);
  EXPECT_TRUE(!Zero.zext(64));
  EXPECT_TRUE(!Zero.sext(64));
}

// Test that APInt shift left works when bitwidth > 64 and shiftamt == 0
TEST(APIntTest, ShiftLeftByZero) {
  APInt One = APInt::getNullValue(65) + 1;
  APInt Shl = One.shl(0);
  EXPECT_TRUE(Shl[0]);
  EXPECT_FALSE(Shl[1]);
}

TEST(APIntTest, i64_ArithmeticRightShiftNegative) {
  const APInt neg_one(64, static_cast<uint64_t>(-1), true);
  EXPECT_EQ(neg_one, neg_one.ashr(7));
}

TEST(APIntTest, i128_NegativeCount) {
  APInt Minus3(128, static_cast<uint64_t>(-3), true);
  EXPECT_EQ(126u, Minus3.countLeadingOnes());
  EXPECT_EQ(-3, Minus3.getSExtValue());

  APInt Minus1(128, static_cast<uint64_t>(-1), true);
  EXPECT_EQ(0u, Minus1.countLeadingZeros());
  EXPECT_EQ(128u, Minus1.countLeadingOnes());
  EXPECT_EQ(128u, Minus1.getActiveBits());
  EXPECT_EQ(0u, Minus1.countTrailingZeros());
  EXPECT_EQ(128u, Minus1.countTrailingOnes());
  EXPECT_EQ(128u, Minus1.countPopulation());
  EXPECT_EQ(-1, Minus1.getSExtValue());
}

TEST(APIntTest, i33_Count) {
  APInt i33minus2(33, static_cast<uint64_t>(-2), true);
  EXPECT_EQ(0u, i33minus2.countLeadingZeros());
  EXPECT_EQ(32u, i33minus2.countLeadingOnes());
  EXPECT_EQ(33u, i33minus2.getActiveBits());
  EXPECT_EQ(1u, i33minus2.countTrailingZeros());
  EXPECT_EQ(32u, i33minus2.countPopulation());
  EXPECT_EQ(-2, i33minus2.getSExtValue());
  EXPECT_EQ(((uint64_t)-2)&((1ull<<33) -1), i33minus2.getZExtValue());
}

TEST(APIntTest, i61_Count) {
  APInt i61(61, 1 << 15);
  EXPECT_EQ(45u, i61.countLeadingZeros());
  EXPECT_EQ(0u, i61.countLeadingOnes());
  EXPECT_EQ(16u, i61.getActiveBits());
  EXPECT_EQ(15u, i61.countTrailingZeros());
  EXPECT_EQ(1u, i61.countPopulation());
  EXPECT_EQ(static_cast<int64_t>(1 << 15), i61.getSExtValue());
  EXPECT_EQ(static_cast<uint64_t>(1 << 15), i61.getZExtValue());

  i61.setBits(8, 19);
  EXPECT_EQ(42u, i61.countLeadingZeros());
  EXPECT_EQ(0u, i61.countLeadingOnes());
  EXPECT_EQ(19u, i61.getActiveBits());
  EXPECT_EQ(8u, i61.countTrailingZeros());
  EXPECT_EQ(11u, i61.countPopulation());
  EXPECT_EQ(static_cast<int64_t>((1 << 19) - (1 << 8)), i61.getSExtValue());
  EXPECT_EQ(static_cast<uint64_t>((1 << 19) - (1 << 8)), i61.getZExtValue());
}

TEST(APIntTest, i65_Count) {
  APInt i65(65, 0, true);
  EXPECT_EQ(65u, i65.countLeadingZeros());
  EXPECT_EQ(0u, i65.countLeadingOnes());
  EXPECT_EQ(0u, i65.getActiveBits());
  EXPECT_EQ(1u, i65.getActiveWords());
  EXPECT_EQ(65u, i65.countTrailingZeros());
  EXPECT_EQ(0u, i65.countPopulation());

  APInt i65minus(65, 0, true);
  i65minus.setBit(64);
  EXPECT_EQ(0u, i65minus.countLeadingZeros());
  EXPECT_EQ(1u, i65minus.countLeadingOnes());
  EXPECT_EQ(65u, i65minus.getActiveBits());
  EXPECT_EQ(64u, i65minus.countTrailingZeros());
  EXPECT_EQ(1u, i65minus.countPopulation());
}

TEST(APIntTest, i128_PositiveCount) {
  APInt u128max = APInt::getAllOnesValue(128);
  EXPECT_EQ(128u, u128max.countLeadingOnes());
  EXPECT_EQ(0u, u128max.countLeadingZeros());
  EXPECT_EQ(128u, u128max.getActiveBits());
  EXPECT_EQ(0u, u128max.countTrailingZeros());
  EXPECT_EQ(128u, u128max.countTrailingOnes());
  EXPECT_EQ(128u, u128max.countPopulation());

  APInt u64max(128, static_cast<uint64_t>(-1), false);
  EXPECT_EQ(64u, u64max.countLeadingZeros());
  EXPECT_EQ(0u, u64max.countLeadingOnes());
  EXPECT_EQ(64u, u64max.getActiveBits());
  EXPECT_EQ(0u, u64max.countTrailingZeros());
  EXPECT_EQ(64u, u64max.countTrailingOnes());
  EXPECT_EQ(64u, u64max.countPopulation());
  EXPECT_EQ((uint64_t)~0ull, u64max.getZExtValue());

  APInt zero(128, 0, true);
  EXPECT_EQ(128u, zero.countLeadingZeros());
  EXPECT_EQ(0u, zero.countLeadingOnes());
  EXPECT_EQ(0u, zero.getActiveBits());
  EXPECT_EQ(128u, zero.countTrailingZeros());
  EXPECT_EQ(0u, zero.countTrailingOnes());
  EXPECT_EQ(0u, zero.countPopulation());
  EXPECT_EQ(0u, zero.getSExtValue());
  EXPECT_EQ(0u, zero.getZExtValue());

  APInt one(128, 1, true);
  EXPECT_EQ(127u, one.countLeadingZeros());
  EXPECT_EQ(0u, one.countLeadingOnes());
  EXPECT_EQ(1u, one.getActiveBits());
  EXPECT_EQ(0u, one.countTrailingZeros());
  EXPECT_EQ(1u, one.countTrailingOnes());
  EXPECT_EQ(1u, one.countPopulation());
  EXPECT_EQ(1, one.getSExtValue());
  EXPECT_EQ(1u, one.getZExtValue());

  APInt s128(128, 2, true);
  EXPECT_EQ(126u, s128.countLeadingZeros());
  EXPECT_EQ(0u, s128.countLeadingOnes());
  EXPECT_EQ(2u, s128.getActiveBits());
  EXPECT_EQ(1u, s128.countTrailingZeros());
  EXPECT_EQ(0u, s128.countTrailingOnes());
  EXPECT_EQ(1u, s128.countPopulation());
  EXPECT_EQ(2, s128.getSExtValue());
  EXPECT_EQ(2u, s128.getZExtValue());

  // NOP Test
  s128.setBits(42, 42);
  EXPECT_EQ(126u, s128.countLeadingZeros());
  EXPECT_EQ(0u, s128.countLeadingOnes());
  EXPECT_EQ(2u, s128.getActiveBits());
  EXPECT_EQ(1u, s128.countTrailingZeros());
  EXPECT_EQ(0u, s128.countTrailingOnes());
  EXPECT_EQ(1u, s128.countPopulation());
  EXPECT_EQ(2, s128.getSExtValue());
  EXPECT_EQ(2u, s128.getZExtValue());

  s128.setBits(3, 32);
  EXPECT_EQ(96u, s128.countLeadingZeros());
  EXPECT_EQ(0u, s128.countLeadingOnes());
  EXPECT_EQ(32u, s128.getActiveBits());
  EXPECT_EQ(33u, s128.getMinSignedBits());
  EXPECT_EQ(1u, s128.countTrailingZeros());
  EXPECT_EQ(0u, s128.countTrailingOnes());
  EXPECT_EQ(30u, s128.countPopulation());
  EXPECT_EQ(static_cast<uint32_t>((~0u << 3) | 2), s128.getZExtValue());

  s128.setBits(62, 128);
  EXPECT_EQ(0u, s128.countLeadingZeros());
  EXPECT_EQ(66u, s128.countLeadingOnes());
  EXPECT_EQ(128u, s128.getActiveBits());
  EXPECT_EQ(63u, s128.getMinSignedBits());
  EXPECT_EQ(1u, s128.countTrailingZeros());
  EXPECT_EQ(0u, s128.countTrailingOnes());
  EXPECT_EQ(96u, s128.countPopulation());
  EXPECT_EQ(static_cast<int64_t>((3ull << 62) |
                                 static_cast<uint32_t>((~0u << 3) | 2)),
            s128.getSExtValue());
}

TEST(APIntTest, i256) {
  APInt s256(256, 15, true);
  EXPECT_EQ(252u, s256.countLeadingZeros());
  EXPECT_EQ(0u, s256.countLeadingOnes());
  EXPECT_EQ(4u, s256.getActiveBits());
  EXPECT_EQ(0u, s256.countTrailingZeros());
  EXPECT_EQ(4u, s256.countTrailingOnes());
  EXPECT_EQ(4u, s256.countPopulation());
  EXPECT_EQ(15, s256.getSExtValue());
  EXPECT_EQ(15u, s256.getZExtValue());

  s256.setBits(62, 66);
  EXPECT_EQ(190u, s256.countLeadingZeros());
  EXPECT_EQ(0u, s256.countLeadingOnes());
  EXPECT_EQ(66u, s256.getActiveBits());
  EXPECT_EQ(67u, s256.getMinSignedBits());
  EXPECT_EQ(0u, s256.countTrailingZeros());
  EXPECT_EQ(4u, s256.countTrailingOnes());
  EXPECT_EQ(8u, s256.countPopulation());

  s256.setBits(60, 256);
  EXPECT_EQ(0u, s256.countLeadingZeros());
  EXPECT_EQ(196u, s256.countLeadingOnes());
  EXPECT_EQ(256u, s256.getActiveBits());
  EXPECT_EQ(61u, s256.getMinSignedBits());
  EXPECT_EQ(0u, s256.countTrailingZeros());
  EXPECT_EQ(4u, s256.countTrailingOnes());
  EXPECT_EQ(200u, s256.countPopulation());
  EXPECT_EQ(static_cast<int64_t>((~0ull << 60) | 15), s256.getSExtValue());
}

TEST(APIntTest, i1) {
  const APInt neg_two(1, static_cast<uint64_t>(-2), true);
  const APInt neg_one(1, static_cast<uint64_t>(-1), true);
  const APInt zero(1, 0);
  const APInt one(1, 1);
  const APInt two(1, 2);

  EXPECT_EQ(0, neg_two.getSExtValue());
  EXPECT_EQ(-1, neg_one.getSExtValue());
  EXPECT_EQ(1u, neg_one.getZExtValue());
  EXPECT_EQ(0u, zero.getZExtValue());
  EXPECT_EQ(-1, one.getSExtValue());
  EXPECT_EQ(1u, one.getZExtValue());
  EXPECT_EQ(0u, two.getZExtValue());
  EXPECT_EQ(0, two.getSExtValue());

  // Basic equalities for 1-bit values.
  EXPECT_EQ(zero, two);
  EXPECT_EQ(zero, neg_two);
  EXPECT_EQ(one, neg_one);
  EXPECT_EQ(two, neg_two);

  // Min/max signed values.
  EXPECT_TRUE(zero.isMaxSignedValue());
  EXPECT_FALSE(one.isMaxSignedValue());
  EXPECT_FALSE(zero.isMinSignedValue());
  EXPECT_TRUE(one.isMinSignedValue());

  // Additions.
  EXPECT_EQ(two, one + one);
  EXPECT_EQ(zero, neg_one + one);
  EXPECT_EQ(neg_two, neg_one + neg_one);

  // Subtractions.
  EXPECT_EQ(neg_two, neg_one - one);
  EXPECT_EQ(two, one - neg_one);
  EXPECT_EQ(zero, one - one);

  // And
  EXPECT_EQ(zero, zero & zero);
  EXPECT_EQ(zero, one & zero);
  EXPECT_EQ(zero, zero & one);
  EXPECT_EQ(one, one & one);
  EXPECT_EQ(zero, zero & zero);
  EXPECT_EQ(zero, neg_one & zero);
  EXPECT_EQ(zero, zero & neg_one);
  EXPECT_EQ(neg_one, neg_one & neg_one);

  // Or
  EXPECT_EQ(zero, zero | zero);
  EXPECT_EQ(one, one | zero);
  EXPECT_EQ(one, zero | one);
  EXPECT_EQ(one, one | one);
  EXPECT_EQ(zero, zero | zero);
  EXPECT_EQ(neg_one, neg_one | zero);
  EXPECT_EQ(neg_one, zero | neg_one);
  EXPECT_EQ(neg_one, neg_one | neg_one);

  // Xor
  EXPECT_EQ(zero, zero ^ zero);
  EXPECT_EQ(one, one ^ zero);
  EXPECT_EQ(one, zero ^ one);
  EXPECT_EQ(zero, one ^ one);
  EXPECT_EQ(zero, zero ^ zero);
  EXPECT_EQ(neg_one, neg_one ^ zero);
  EXPECT_EQ(neg_one, zero ^ neg_one);
  EXPECT_EQ(zero, neg_one ^ neg_one);

  // Shifts.
  EXPECT_EQ(zero, one << one);
  EXPECT_EQ(one, one << zero);
  EXPECT_EQ(zero, one.shl(1));
  EXPECT_EQ(one, one.shl(0));
  EXPECT_EQ(zero, one.lshr(1));
  EXPECT_EQ(zero, one.ashr(1));

  // Rotates.
  EXPECT_EQ(one, one.rotl(0));
  EXPECT_EQ(one, one.rotl(1));
  EXPECT_EQ(one, one.rotr(0));
  EXPECT_EQ(one, one.rotr(1));

  // Multiplies.
  EXPECT_EQ(neg_one, neg_one * one);
  EXPECT_EQ(neg_one, one * neg_one);
  EXPECT_EQ(one, neg_one * neg_one);
  EXPECT_EQ(one, one * one);

  // Divides.
  EXPECT_EQ(neg_one, one.sdiv(neg_one));
  EXPECT_EQ(neg_one, neg_one.sdiv(one));
  EXPECT_EQ(one, neg_one.sdiv(neg_one));
  EXPECT_EQ(one, one.sdiv(one));

  EXPECT_EQ(neg_one, one.udiv(neg_one));
  EXPECT_EQ(neg_one, neg_one.udiv(one));
  EXPECT_EQ(one, neg_one.udiv(neg_one));
  EXPECT_EQ(one, one.udiv(one));

  // Remainders.
  EXPECT_EQ(zero, neg_one.srem(one));
  EXPECT_EQ(zero, neg_one.urem(one));
  EXPECT_EQ(zero, one.srem(neg_one));

  // sdivrem
  {
  APInt q(8, 0);
  APInt r(8, 0);
  APInt one(8, 1);
  APInt two(8, 2);
  APInt nine(8, 9);
  APInt four(8, 4);

  EXPECT_EQ(nine.srem(two), one);
  EXPECT_EQ(nine.srem(-two), one);
  EXPECT_EQ((-nine).srem(two), -one);
  EXPECT_EQ((-nine).srem(-two), -one);

  APInt::sdivrem(nine, two, q, r);
  EXPECT_EQ(four, q);
  EXPECT_EQ(one, r);
  APInt::sdivrem(-nine, two, q, r);
  EXPECT_EQ(-four, q);
  EXPECT_EQ(-one, r);
  APInt::sdivrem(nine, -two, q, r);
  EXPECT_EQ(-four, q);
  EXPECT_EQ(one, r);
  APInt::sdivrem(-nine, -two, q, r);
  EXPECT_EQ(four, q);
  EXPECT_EQ(-one, r);
  }
}

TEST(APIntTest, compare) {
  std::array<APInt, 5> testVals{{
    APInt{16, 2},
    APInt{16, 1},
    APInt{16, 0},
    APInt{16, (uint64_t)-1, true},
    APInt{16, (uint64_t)-2, true},
  }};

  for (auto &arg1 : testVals)
    for (auto &arg2 : testVals) {
      auto uv1 = arg1.getZExtValue();
      auto uv2 = arg2.getZExtValue();
      auto sv1 = arg1.getSExtValue();
      auto sv2 = arg2.getSExtValue();

      EXPECT_EQ(uv1 <  uv2, arg1.ult(arg2));
      EXPECT_EQ(uv1 <= uv2, arg1.ule(arg2));
      EXPECT_EQ(uv1 >  uv2, arg1.ugt(arg2));
      EXPECT_EQ(uv1 >= uv2, arg1.uge(arg2));

      EXPECT_EQ(sv1 <  sv2, arg1.slt(arg2));
      EXPECT_EQ(sv1 <= sv2, arg1.sle(arg2));
      EXPECT_EQ(sv1 >  sv2, arg1.sgt(arg2));
      EXPECT_EQ(sv1 >= sv2, arg1.sge(arg2));

      EXPECT_EQ(uv1 <  uv2, arg1.ult(uv2));
      EXPECT_EQ(uv1 <= uv2, arg1.ule(uv2));
      EXPECT_EQ(uv1 >  uv2, arg1.ugt(uv2));
      EXPECT_EQ(uv1 >= uv2, arg1.uge(uv2));

      EXPECT_EQ(sv1 <  sv2, arg1.slt(sv2));
      EXPECT_EQ(sv1 <= sv2, arg1.sle(sv2));
      EXPECT_EQ(sv1 >  sv2, arg1.sgt(sv2));
      EXPECT_EQ(sv1 >= sv2, arg1.sge(sv2));
    }
}

TEST(APIntTest, compareWithRawIntegers) {
  EXPECT_TRUE(!APInt(8, 1).uge(256));
  EXPECT_TRUE(!APInt(8, 1).ugt(256));
  EXPECT_TRUE( APInt(8, 1).ule(256));
  EXPECT_TRUE( APInt(8, 1).ult(256));
  EXPECT_TRUE(!APInt(8, 1).sge(256));
  EXPECT_TRUE(!APInt(8, 1).sgt(256));
  EXPECT_TRUE( APInt(8, 1).sle(256));
  EXPECT_TRUE( APInt(8, 1).slt(256));
  EXPECT_TRUE(!(APInt(8, 0) == 256));
  EXPECT_TRUE(  APInt(8, 0) != 256);
  EXPECT_TRUE(!(APInt(8, 1) == 256));
  EXPECT_TRUE(  APInt(8, 1) != 256);

  auto uint64max = UINT64_MAX;
  auto int64max  = INT64_MAX;
  auto int64min  = INT64_MIN;

  auto u64 = APInt{128, uint64max};
  auto s64 = APInt{128, static_cast<uint64_t>(int64max), true};
  auto big = u64 + 1;

  EXPECT_TRUE( u64.uge(uint64max));
  EXPECT_TRUE(!u64.ugt(uint64max));
  EXPECT_TRUE( u64.ule(uint64max));
  EXPECT_TRUE(!u64.ult(uint64max));
  EXPECT_TRUE( u64.sge(int64max));
  EXPECT_TRUE( u64.sgt(int64max));
  EXPECT_TRUE(!u64.sle(int64max));
  EXPECT_TRUE(!u64.slt(int64max));
  EXPECT_TRUE( u64.sge(int64min));
  EXPECT_TRUE( u64.sgt(int64min));
  EXPECT_TRUE(!u64.sle(int64min));
  EXPECT_TRUE(!u64.slt(int64min));

  EXPECT_TRUE(u64 == uint64max);
  EXPECT_TRUE(u64 != int64max);
  EXPECT_TRUE(u64 != int64min);

  EXPECT_TRUE(!s64.uge(uint64max));
  EXPECT_TRUE(!s64.ugt(uint64max));
  EXPECT_TRUE( s64.ule(uint64max));
  EXPECT_TRUE( s64.ult(uint64max));
  EXPECT_TRUE( s64.sge(int64max));
  EXPECT_TRUE(!s64.sgt(int64max));
  EXPECT_TRUE( s64.sle(int64max));
  EXPECT_TRUE(!s64.slt(int64max));
  EXPECT_TRUE( s64.sge(int64min));
  EXPECT_TRUE( s64.sgt(int64min));
  EXPECT_TRUE(!s64.sle(int64min));
  EXPECT_TRUE(!s64.slt(int64min));

  EXPECT_TRUE(s64 != uint64max);
  EXPECT_TRUE(s64 == int64max);
  EXPECT_TRUE(s64 != int64min);

  EXPECT_TRUE( big.uge(uint64max));
  EXPECT_TRUE( big.ugt(uint64max));
  EXPECT_TRUE(!big.ule(uint64max));
  EXPECT_TRUE(!big.ult(uint64max));
  EXPECT_TRUE( big.sge(int64max));
  EXPECT_TRUE( big.sgt(int64max));
  EXPECT_TRUE(!big.sle(int64max));
  EXPECT_TRUE(!big.slt(int64max));
  EXPECT_TRUE( big.sge(int64min));
  EXPECT_TRUE( big.sgt(int64min));
  EXPECT_TRUE(!big.sle(int64min));
  EXPECT_TRUE(!big.slt(int64min));

  EXPECT_TRUE(big != uint64max);
  EXPECT_TRUE(big != int64max);
  EXPECT_TRUE(big != int64min);
}

TEST(APIntTest, compareWithInt64Min) {
  int64_t edge = INT64_MIN;
  int64_t edgeP1 = edge + 1;
  int64_t edgeM1 = INT64_MAX;
  auto a = APInt{64, static_cast<uint64_t>(edge), true};

  EXPECT_TRUE(!a.slt(edge));
  EXPECT_TRUE( a.sle(edge));
  EXPECT_TRUE(!a.sgt(edge));
  EXPECT_TRUE( a.sge(edge));
  EXPECT_TRUE( a.slt(edgeP1));
  EXPECT_TRUE( a.sle(edgeP1));
  EXPECT_TRUE(!a.sgt(edgeP1));
  EXPECT_TRUE(!a.sge(edgeP1));
  EXPECT_TRUE( a.slt(edgeM1));
  EXPECT_TRUE( a.sle(edgeM1));
  EXPECT_TRUE(!a.sgt(edgeM1));
  EXPECT_TRUE(!a.sge(edgeM1));
}

TEST(APIntTest, compareWithHalfInt64Max) {
  uint64_t edge = 0x4000000000000000;
  uint64_t edgeP1 = edge + 1;
  uint64_t edgeM1 = edge - 1;
  auto a = APInt{64, edge};

  EXPECT_TRUE(!a.ult(edge));
  EXPECT_TRUE( a.ule(edge));
  EXPECT_TRUE(!a.ugt(edge));
  EXPECT_TRUE( a.uge(edge));
  EXPECT_TRUE( a.ult(edgeP1));
  EXPECT_TRUE( a.ule(edgeP1));
  EXPECT_TRUE(!a.ugt(edgeP1));
  EXPECT_TRUE(!a.uge(edgeP1));
  EXPECT_TRUE(!a.ult(edgeM1));
  EXPECT_TRUE(!a.ule(edgeM1));
  EXPECT_TRUE( a.ugt(edgeM1));
  EXPECT_TRUE( a.uge(edgeM1));

  EXPECT_TRUE(!a.slt(edge));
  EXPECT_TRUE( a.sle(edge));
  EXPECT_TRUE(!a.sgt(edge));
  EXPECT_TRUE( a.sge(edge));
  EXPECT_TRUE( a.slt(edgeP1));
  EXPECT_TRUE( a.sle(edgeP1));
  EXPECT_TRUE(!a.sgt(edgeP1));
  EXPECT_TRUE(!a.sge(edgeP1));
  EXPECT_TRUE(!a.slt(edgeM1));
  EXPECT_TRUE(!a.sle(edgeM1));
  EXPECT_TRUE( a.sgt(edgeM1));
  EXPECT_TRUE( a.sge(edgeM1));
}

TEST(APIntTest, compareLargeIntegers) {
  // Make sure all the combinations of signed comparisons work with big ints.
  auto One = APInt{128, static_cast<uint64_t>(1), true};
  auto Two = APInt{128, static_cast<uint64_t>(2), true};
  auto MinusOne = APInt{128, static_cast<uint64_t>(-1), true};
  auto MinusTwo = APInt{128, static_cast<uint64_t>(-2), true};

  EXPECT_TRUE(!One.slt(One));
  EXPECT_TRUE(!Two.slt(One));
  EXPECT_TRUE(MinusOne.slt(One));
  EXPECT_TRUE(MinusTwo.slt(One));

  EXPECT_TRUE(One.slt(Two));
  EXPECT_TRUE(!Two.slt(Two));
  EXPECT_TRUE(MinusOne.slt(Two));
  EXPECT_TRUE(MinusTwo.slt(Two));

  EXPECT_TRUE(!One.slt(MinusOne));
  EXPECT_TRUE(!Two.slt(MinusOne));
  EXPECT_TRUE(!MinusOne.slt(MinusOne));
  EXPECT_TRUE(MinusTwo.slt(MinusOne));

  EXPECT_TRUE(!One.slt(MinusTwo));
  EXPECT_TRUE(!Two.slt(MinusTwo));
  EXPECT_TRUE(!MinusOne.slt(MinusTwo));
  EXPECT_TRUE(!MinusTwo.slt(MinusTwo));
}

TEST(APIntTest, binaryOpsWithRawIntegers) {
  // Single word check.
  uint64_t E1 = 0x2CA7F46BF6569915ULL;
  APInt A1(64, E1);

  EXPECT_EQ(A1 & E1, E1);
  EXPECT_EQ(A1 & 0, 0);
  EXPECT_EQ(A1 & 1, 1);
  EXPECT_EQ(A1 & 5, 5);
  EXPECT_EQ(A1 & UINT64_MAX, E1);

  EXPECT_EQ(A1 | E1, E1);
  EXPECT_EQ(A1 | 0, E1);
  EXPECT_EQ(A1 | 1, E1);
  EXPECT_EQ(A1 | 2, E1 | 2);
  EXPECT_EQ(A1 | UINT64_MAX, UINT64_MAX);

  EXPECT_EQ(A1 ^ E1, 0);
  EXPECT_EQ(A1 ^ 0, E1);
  EXPECT_EQ(A1 ^ 1, E1 ^ 1);
  EXPECT_EQ(A1 ^ 7, E1 ^ 7);
  EXPECT_EQ(A1 ^ UINT64_MAX, ~E1);

  // Multiword check.
  uint64_t N = 0xEB6EB136591CBA21ULL;
  integerPart E2[4] = {
    N,
    0x7B9358BD6A33F10AULL,
    0x7E7FFA5EADD8846ULL,
    0x305F341CA00B613DULL
  };
  APInt A2(integerPartWidth*4, E2);

  EXPECT_EQ(A2 & N, N);
  EXPECT_EQ(A2 & 0, 0);
  EXPECT_EQ(A2 & 1, 1);
  EXPECT_EQ(A2 & 5, 1);
  EXPECT_EQ(A2 & UINT64_MAX, N);

  EXPECT_EQ(A2 | N, A2);
  EXPECT_EQ(A2 | 0, A2);
  EXPECT_EQ(A2 | 1, A2);
  EXPECT_EQ(A2 | 2, A2 + 2);
  EXPECT_EQ(A2 | UINT64_MAX, A2 - N + UINT64_MAX);

  EXPECT_EQ(A2 ^ N, A2 - N);
  EXPECT_EQ(A2 ^ 0, A2);
  EXPECT_EQ(A2 ^ 1, A2 - 1);
  EXPECT_EQ(A2 ^ 7, A2 + 5);
  EXPECT_EQ(A2 ^ UINT64_MAX, A2 - N + ~N);
}

TEST(APIntTest, rvalue_arithmetic) {
  // Test all combinations of lvalue/rvalue lhs/rhs of add/sub

  // Lamdba to return an APInt by value, but also provide the raw value of the
  // allocated data.
  auto getRValue = [](const char *HexString, uint64_t const *&RawData) {
    APInt V(129, HexString, 16);
    RawData = V.getRawData();
    return V;
  };

  APInt One(129, "1", 16);
  APInt Two(129, "2", 16);
  APInt Three(129, "3", 16);
  APInt MinusOne = -One;

  const uint64_t *RawDataL = nullptr;
  const uint64_t *RawDataR = nullptr;

  {
    // 1 + 1 = 2
    APInt AddLL = One + One;
    EXPECT_EQ(AddLL, Two);

    APInt AddLR = One + getRValue("1", RawDataR);
    EXPECT_EQ(AddLR, Two);
    EXPECT_EQ(AddLR.getRawData(), RawDataR);

    APInt AddRL = getRValue("1", RawDataL) + One;
    EXPECT_EQ(AddRL, Two);
    EXPECT_EQ(AddRL.getRawData(), RawDataL);

    APInt AddRR = getRValue("1", RawDataL) + getRValue("1", RawDataR);
    EXPECT_EQ(AddRR, Two);
    EXPECT_EQ(AddRR.getRawData(), RawDataR);

    // LValue's and constants
    APInt AddLK = One + 1;
    EXPECT_EQ(AddLK, Two);

    APInt AddKL = 1 + One;
    EXPECT_EQ(AddKL, Two);

    // RValue's and constants
    APInt AddRK = getRValue("1", RawDataL) + 1;
    EXPECT_EQ(AddRK, Two);
    EXPECT_EQ(AddRK.getRawData(), RawDataL);

    APInt AddKR = 1 + getRValue("1", RawDataR);
    EXPECT_EQ(AddKR, Two);
    EXPECT_EQ(AddKR.getRawData(), RawDataR);
  }

  {
    // 0x0,FFFF...FFFF + 0x2 = 0x100...0001
    APInt AllOnes(129, "0FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF", 16);
    APInt HighOneLowOne(129, "100000000000000000000000000000001", 16);

    APInt AddLL = AllOnes + Two;
    EXPECT_EQ(AddLL, HighOneLowOne);

    APInt AddLR = AllOnes + getRValue("2", RawDataR);
    EXPECT_EQ(AddLR, HighOneLowOne);
    EXPECT_EQ(AddLR.getRawData(), RawDataR);

    APInt AddRL = getRValue("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF", RawDataL) + Two;
    EXPECT_EQ(AddRL, HighOneLowOne);
    EXPECT_EQ(AddRL.getRawData(), RawDataL);

    APInt AddRR = getRValue("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF", RawDataL) +
                  getRValue("2", RawDataR);
    EXPECT_EQ(AddRR, HighOneLowOne);
    EXPECT_EQ(AddRR.getRawData(), RawDataR);

    // LValue's and constants
    APInt AddLK = AllOnes + 2;
    EXPECT_EQ(AddLK, HighOneLowOne);

    APInt AddKL = 2 + AllOnes;
    EXPECT_EQ(AddKL, HighOneLowOne);

    // RValue's and constants
    APInt AddRK = getRValue("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF", RawDataL) + 2;
    EXPECT_EQ(AddRK, HighOneLowOne);
    EXPECT_EQ(AddRK.getRawData(), RawDataL);

    APInt AddKR = 2 + getRValue("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF", RawDataR);
    EXPECT_EQ(AddKR, HighOneLowOne);
    EXPECT_EQ(AddKR.getRawData(), RawDataR);
  }

  {
    // 2 - 1 = 1
    APInt SubLL = Two - One;
    EXPECT_EQ(SubLL, One);

    APInt SubLR = Two - getRValue("1", RawDataR);
    EXPECT_EQ(SubLR, One);
    EXPECT_EQ(SubLR.getRawData(), RawDataR);

    APInt SubRL = getRValue("2", RawDataL) - One;
    EXPECT_EQ(SubRL, One);
    EXPECT_EQ(SubRL.getRawData(), RawDataL);

    APInt SubRR = getRValue("2", RawDataL) - getRValue("1", RawDataR);
    EXPECT_EQ(SubRR, One);
    EXPECT_EQ(SubRR.getRawData(), RawDataR);

    // LValue's and constants
    APInt SubLK = Two - 1;
    EXPECT_EQ(SubLK, One);

    APInt SubKL = 2 - One;
    EXPECT_EQ(SubKL, One);

    // RValue's and constants
    APInt SubRK = getRValue("2", RawDataL) - 1;
    EXPECT_EQ(SubRK, One);
    EXPECT_EQ(SubRK.getRawData(), RawDataL);

    APInt SubKR = 2 - getRValue("1", RawDataR);
    EXPECT_EQ(SubKR, One);
    EXPECT_EQ(SubKR.getRawData(), RawDataR);
  }

  {
    // 0x100...0001 - 0x0,FFFF...FFFF = 0x2
    APInt AllOnes(129, "0FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF", 16);
    APInt HighOneLowOne(129, "100000000000000000000000000000001", 16);

    APInt SubLL = HighOneLowOne - AllOnes;
    EXPECT_EQ(SubLL, Two);

    APInt SubLR = HighOneLowOne -
                  getRValue("0FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF", RawDataR);
    EXPECT_EQ(SubLR, Two);
    EXPECT_EQ(SubLR.getRawData(), RawDataR);

    APInt SubRL = getRValue("100000000000000000000000000000001", RawDataL) -
                  AllOnes;
    EXPECT_EQ(SubRL, Two);
    EXPECT_EQ(SubRL.getRawData(), RawDataL);

    APInt SubRR = getRValue("100000000000000000000000000000001", RawDataL) -
                  getRValue("0FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF", RawDataR);
    EXPECT_EQ(SubRR, Two);
    EXPECT_EQ(SubRR.getRawData(), RawDataR);

    // LValue's and constants
    // 0x100...0001 - 0x2 = 0x0,FFFF...FFFF
    APInt SubLK = HighOneLowOne - 2;
    EXPECT_EQ(SubLK, AllOnes);

    // 2 - (-1) = 3
    APInt SubKL = 2 - MinusOne;
    EXPECT_EQ(SubKL, Three);

    // RValue's and constants
    // 0x100...0001 - 0x2 = 0x0,FFFF...FFFF
    APInt SubRK = getRValue("100000000000000000000000000000001", RawDataL) - 2;
    EXPECT_EQ(SubRK, AllOnes);
    EXPECT_EQ(SubRK.getRawData(), RawDataL);

    APInt SubKR = 2 - getRValue("1FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF", RawDataR);
    EXPECT_EQ(SubKR, Three);
    EXPECT_EQ(SubKR.getRawData(), RawDataR);
  }
}


// Tests different div/rem varaints using scheme (a * b + c) / a
void testDiv(APInt a, APInt b, APInt c) {
  ASSERT_TRUE(a.uge(b)); // Must: a >= b
  ASSERT_TRUE(a.ugt(c)); // Must: a > c

  auto p = a * b + c;

  auto q = p.udiv(a);
  auto r = p.urem(a);
  EXPECT_EQ(b, q);
  EXPECT_EQ(c, r);
  APInt::udivrem(p, a, q, r);
  EXPECT_EQ(b, q);
  EXPECT_EQ(c, r);
  q = p.sdiv(a);
  r = p.srem(a);
  EXPECT_EQ(b, q);
  EXPECT_EQ(c, r);
  APInt::sdivrem(p, a, q, r);
  EXPECT_EQ(b, q);
  EXPECT_EQ(c, r);

  if (b.ugt(c)) { // Test also symmetric case
    q = p.udiv(b);
    r = p.urem(b);
    EXPECT_EQ(a, q);
    EXPECT_EQ(c, r);
    APInt::udivrem(p, b, q, r);
    EXPECT_EQ(a, q);
    EXPECT_EQ(c, r);
    q = p.sdiv(b);
    r = p.srem(b);
    EXPECT_EQ(a, q);
    EXPECT_EQ(c, r);
    APInt::sdivrem(p, b, q, r);
    EXPECT_EQ(a, q);
    EXPECT_EQ(c, r);
  }
}

TEST(APIntTest, divrem_big1) {
  // Tests KnuthDiv rare step D6
  testDiv({256, "1ffffffffffffffff", 16},
          {256, "1ffffffffffffffff", 16},
          {256, 0});
}

TEST(APIntTest, divrem_big2) {
  // Tests KnuthDiv rare step D6
  testDiv({1024,                       "112233ceff"
                 "cecece000000ffffffffffffffffffff"
                 "ffffffffffffffffffffffffffffffff"
                 "ffffffffffffffffffffffffffffffff"
                 "ffffffffffffffffffffffffffffff33", 16},
          {1024,           "111111ffffffffffffffff"
                 "ffffffffffffffffffffffffffffffff"
                 "fffffffffffffffffffffffffffffccf"
                 "ffffffffffffffffffffffffffffff00", 16},
          {1024, 7919});
}

TEST(APIntTest, divrem_big3) {
  // Tests KnuthDiv case without shift
  testDiv({256, "80000001ffffffffffffffff", 16},
          {256, "ffffffffffffff0000000", 16},
          {256, 4219});
}

TEST(APIntTest, divrem_big4) {
  // Tests heap allocation in divide() enfoced by huge numbers
  testDiv(APInt{4096, 5}.shl(2001),
          APInt{4096, 1}.shl(2000),
          APInt{4096, 4219*13});
}

TEST(APIntTest, divrem_big5) {
  // Tests one word divisor case of divide()
  testDiv(APInt{1024, 19}.shl(811),
          APInt{1024, 4356013}, // one word
          APInt{1024, 1});
}

TEST(APIntTest, divrem_big6) {
  // Tests some rare "borrow" cases in D4 step
  testDiv(APInt{512, "ffffffffffffffff00000000000000000000000001", 16},
          APInt{512, "10000000000000001000000000000001", 16},
          APInt{512, "10000000000000000000000000000000", 16});
}

TEST(APIntTest, divrem_big7) {
  // Yet another test for KnuthDiv rare step D6.
  testDiv({224, "800000008000000200000005", 16},
          {224, "fffffffd", 16},
          {224, "80000000800000010000000f", 16});
}

TEST(APIntTest, fromString) {
  EXPECT_EQ(APInt(32, 0), APInt(32,   "0", 2));
  EXPECT_EQ(APInt(32, 1), APInt(32,   "1", 2));
  EXPECT_EQ(APInt(32, 2), APInt(32,  "10", 2));
  EXPECT_EQ(APInt(32, 3), APInt(32,  "11", 2));
  EXPECT_EQ(APInt(32, 4), APInt(32, "100", 2));

  EXPECT_EQ(APInt(32, 0), APInt(32,   "+0", 2));
  EXPECT_EQ(APInt(32, 1), APInt(32,   "+1", 2));
  EXPECT_EQ(APInt(32, 2), APInt(32,  "+10", 2));
  EXPECT_EQ(APInt(32, 3), APInt(32,  "+11", 2));
  EXPECT_EQ(APInt(32, 4), APInt(32, "+100", 2));

  EXPECT_EQ(APInt(32, uint64_t(-0LL)), APInt(32,   "-0", 2));
  EXPECT_EQ(APInt(32, uint64_t(-1LL)), APInt(32,   "-1", 2));
  EXPECT_EQ(APInt(32, uint64_t(-2LL)), APInt(32,  "-10", 2));
  EXPECT_EQ(APInt(32, uint64_t(-3LL)), APInt(32,  "-11", 2));
  EXPECT_EQ(APInt(32, uint64_t(-4LL)), APInt(32, "-100", 2));


  EXPECT_EQ(APInt(32,  0), APInt(32,  "0",  8));
  EXPECT_EQ(APInt(32,  1), APInt(32,  "1",  8));
  EXPECT_EQ(APInt(32,  7), APInt(32,  "7",  8));
  EXPECT_EQ(APInt(32,  8), APInt(32,  "10", 8));
  EXPECT_EQ(APInt(32, 15), APInt(32,  "17", 8));
  EXPECT_EQ(APInt(32, 16), APInt(32,  "20", 8));

  EXPECT_EQ(APInt(32,  +0), APInt(32,  "+0",  8));
  EXPECT_EQ(APInt(32,  +1), APInt(32,  "+1",  8));
  EXPECT_EQ(APInt(32,  +7), APInt(32,  "+7",  8));
  EXPECT_EQ(APInt(32,  +8), APInt(32,  "+10", 8));
  EXPECT_EQ(APInt(32, +15), APInt(32,  "+17", 8));
  EXPECT_EQ(APInt(32, +16), APInt(32,  "+20", 8));

  EXPECT_EQ(APInt(32,  uint64_t(-0LL)), APInt(32,  "-0",  8));
  EXPECT_EQ(APInt(32,  uint64_t(-1LL)), APInt(32,  "-1",  8));
  EXPECT_EQ(APInt(32,  uint64_t(-7LL)), APInt(32,  "-7",  8));
  EXPECT_EQ(APInt(32,  uint64_t(-8LL)), APInt(32,  "-10", 8));
  EXPECT_EQ(APInt(32, uint64_t(-15LL)), APInt(32,  "-17", 8));
  EXPECT_EQ(APInt(32, uint64_t(-16LL)), APInt(32,  "-20", 8));


  EXPECT_EQ(APInt(32,  0), APInt(32,  "0", 10));
  EXPECT_EQ(APInt(32,  1), APInt(32,  "1", 10));
  EXPECT_EQ(APInt(32,  9), APInt(32,  "9", 10));
  EXPECT_EQ(APInt(32, 10), APInt(32, "10", 10));
  EXPECT_EQ(APInt(32, 19), APInt(32, "19", 10));
  EXPECT_EQ(APInt(32, 20), APInt(32, "20", 10));

  EXPECT_EQ(APInt(32,  uint64_t(-0LL)), APInt(32,  "-0", 10));
  EXPECT_EQ(APInt(32,  uint64_t(-1LL)), APInt(32,  "-1", 10));
  EXPECT_EQ(APInt(32,  uint64_t(-9LL)), APInt(32,  "-9", 10));
  EXPECT_EQ(APInt(32, uint64_t(-10LL)), APInt(32, "-10", 10));
  EXPECT_EQ(APInt(32, uint64_t(-19LL)), APInt(32, "-19", 10));
  EXPECT_EQ(APInt(32, uint64_t(-20LL)), APInt(32, "-20", 10));


  EXPECT_EQ(APInt(32,  0), APInt(32,  "0", 16));
  EXPECT_EQ(APInt(32,  1), APInt(32,  "1", 16));
  EXPECT_EQ(APInt(32, 15), APInt(32,  "F", 16));
  EXPECT_EQ(APInt(32, 16), APInt(32, "10", 16));
  EXPECT_EQ(APInt(32, 31), APInt(32, "1F", 16));
  EXPECT_EQ(APInt(32, 32), APInt(32, "20", 16));

  EXPECT_EQ(APInt(32,  uint64_t(-0LL)), APInt(32,  "-0", 16));
  EXPECT_EQ(APInt(32,  uint64_t(-1LL)), APInt(32,  "-1", 16));
  EXPECT_EQ(APInt(32, uint64_t(-15LL)), APInt(32,  "-F", 16));
  EXPECT_EQ(APInt(32, uint64_t(-16LL)), APInt(32, "-10", 16));
  EXPECT_EQ(APInt(32, uint64_t(-31LL)), APInt(32, "-1F", 16));
  EXPECT_EQ(APInt(32, uint64_t(-32LL)), APInt(32, "-20", 16));

  EXPECT_EQ(APInt(32,  0), APInt(32,  "0", 36));
  EXPECT_EQ(APInt(32,  1), APInt(32,  "1", 36));
  EXPECT_EQ(APInt(32, 35), APInt(32,  "Z", 36));
  EXPECT_EQ(APInt(32, 36), APInt(32, "10", 36));
  EXPECT_EQ(APInt(32, 71), APInt(32, "1Z", 36));
  EXPECT_EQ(APInt(32, 72), APInt(32, "20", 36));
  
  EXPECT_EQ(APInt(32,  uint64_t(-0LL)), APInt(32,  "-0", 36));
  EXPECT_EQ(APInt(32,  uint64_t(-1LL)), APInt(32,  "-1", 36));
  EXPECT_EQ(APInt(32, uint64_t(-35LL)), APInt(32,  "-Z", 36));
  EXPECT_EQ(APInt(32, uint64_t(-36LL)), APInt(32, "-10", 36));
  EXPECT_EQ(APInt(32, uint64_t(-71LL)), APInt(32, "-1Z", 36));
  EXPECT_EQ(APInt(32, uint64_t(-72LL)), APInt(32, "-20", 36));
}

TEST(APIntTest, FromArray) {
  EXPECT_EQ(APInt(32, uint64_t(1)), APInt(32, ArrayRef<uint64_t>(1)));
}

TEST(APIntTest, StringBitsNeeded2) {
  EXPECT_EQ(1U, APInt::getBitsNeeded(  "0", 2));
  EXPECT_EQ(1U, APInt::getBitsNeeded(  "1", 2));
  EXPECT_EQ(2U, APInt::getBitsNeeded( "10", 2));
  EXPECT_EQ(2U, APInt::getBitsNeeded( "11", 2));
  EXPECT_EQ(3U, APInt::getBitsNeeded("100", 2));

  EXPECT_EQ(1U, APInt::getBitsNeeded(  "+0", 2));
  EXPECT_EQ(1U, APInt::getBitsNeeded(  "+1", 2));
  EXPECT_EQ(2U, APInt::getBitsNeeded( "+10", 2));
  EXPECT_EQ(2U, APInt::getBitsNeeded( "+11", 2));
  EXPECT_EQ(3U, APInt::getBitsNeeded("+100", 2));

  EXPECT_EQ(2U, APInt::getBitsNeeded(  "-0", 2));
  EXPECT_EQ(2U, APInt::getBitsNeeded(  "-1", 2));
  EXPECT_EQ(3U, APInt::getBitsNeeded( "-10", 2));
  EXPECT_EQ(3U, APInt::getBitsNeeded( "-11", 2));
  EXPECT_EQ(4U, APInt::getBitsNeeded("-100", 2));
}

TEST(APIntTest, StringBitsNeeded8) {
  EXPECT_EQ(3U, APInt::getBitsNeeded( "0", 8));
  EXPECT_EQ(3U, APInt::getBitsNeeded( "7", 8));
  EXPECT_EQ(6U, APInt::getBitsNeeded("10", 8));
  EXPECT_EQ(6U, APInt::getBitsNeeded("17", 8));
  EXPECT_EQ(6U, APInt::getBitsNeeded("20", 8));

  EXPECT_EQ(3U, APInt::getBitsNeeded( "+0", 8));
  EXPECT_EQ(3U, APInt::getBitsNeeded( "+7", 8));
  EXPECT_EQ(6U, APInt::getBitsNeeded("+10", 8));
  EXPECT_EQ(6U, APInt::getBitsNeeded("+17", 8));
  EXPECT_EQ(6U, APInt::getBitsNeeded("+20", 8));

  EXPECT_EQ(4U, APInt::getBitsNeeded( "-0", 8));
  EXPECT_EQ(4U, APInt::getBitsNeeded( "-7", 8));
  EXPECT_EQ(7U, APInt::getBitsNeeded("-10", 8));
  EXPECT_EQ(7U, APInt::getBitsNeeded("-17", 8));
  EXPECT_EQ(7U, APInt::getBitsNeeded("-20", 8));
}

TEST(APIntTest, StringBitsNeeded10) {
  EXPECT_EQ(1U, APInt::getBitsNeeded( "0", 10));
  EXPECT_EQ(2U, APInt::getBitsNeeded( "3", 10));
  EXPECT_EQ(4U, APInt::getBitsNeeded( "9", 10));
  EXPECT_EQ(4U, APInt::getBitsNeeded("10", 10));
  EXPECT_EQ(5U, APInt::getBitsNeeded("19", 10));
  EXPECT_EQ(5U, APInt::getBitsNeeded("20", 10));

  EXPECT_EQ(1U, APInt::getBitsNeeded( "+0", 10));
  EXPECT_EQ(4U, APInt::getBitsNeeded( "+9", 10));
  EXPECT_EQ(4U, APInt::getBitsNeeded("+10", 10));
  EXPECT_EQ(5U, APInt::getBitsNeeded("+19", 10));
  EXPECT_EQ(5U, APInt::getBitsNeeded("+20", 10));

  EXPECT_EQ(2U, APInt::getBitsNeeded( "-0", 10));
  EXPECT_EQ(5U, APInt::getBitsNeeded( "-9", 10));
  EXPECT_EQ(5U, APInt::getBitsNeeded("-10", 10));
  EXPECT_EQ(6U, APInt::getBitsNeeded("-19", 10));
  EXPECT_EQ(6U, APInt::getBitsNeeded("-20", 10));
}

TEST(APIntTest, StringBitsNeeded16) {
  EXPECT_EQ(4U, APInt::getBitsNeeded( "0", 16));
  EXPECT_EQ(4U, APInt::getBitsNeeded( "F", 16));
  EXPECT_EQ(8U, APInt::getBitsNeeded("10", 16));
  EXPECT_EQ(8U, APInt::getBitsNeeded("1F", 16));
  EXPECT_EQ(8U, APInt::getBitsNeeded("20", 16));

  EXPECT_EQ(4U, APInt::getBitsNeeded( "+0", 16));
  EXPECT_EQ(4U, APInt::getBitsNeeded( "+F", 16));
  EXPECT_EQ(8U, APInt::getBitsNeeded("+10", 16));
  EXPECT_EQ(8U, APInt::getBitsNeeded("+1F", 16));
  EXPECT_EQ(8U, APInt::getBitsNeeded("+20", 16));

  EXPECT_EQ(5U, APInt::getBitsNeeded( "-0", 16));
  EXPECT_EQ(5U, APInt::getBitsNeeded( "-F", 16));
  EXPECT_EQ(9U, APInt::getBitsNeeded("-10", 16));
  EXPECT_EQ(9U, APInt::getBitsNeeded("-1F", 16));
  EXPECT_EQ(9U, APInt::getBitsNeeded("-20", 16));
}

TEST(APIntTest, toString) {
  SmallString<16> S;
  bool isSigned;

  APInt(8, 0).toString(S, 2, true, true);
  EXPECT_EQ(S.str().str(), "0b0");
  S.clear();
  APInt(8, 0).toString(S, 8, true, true);
  EXPECT_EQ(S.str().str(), "00");
  S.clear();
  APInt(8, 0).toString(S, 10, true, true);
  EXPECT_EQ(S.str().str(), "0");
  S.clear();
  APInt(8, 0).toString(S, 16, true, true);
  EXPECT_EQ(S.str().str(), "0x0");
  S.clear();
  APInt(8, 0).toString(S, 36, true, false);
  EXPECT_EQ(S.str().str(), "0");
  S.clear();

  isSigned = false;
  APInt(8, 255, isSigned).toString(S, 2, isSigned, true);
  EXPECT_EQ(S.str().str(), "0b11111111");
  S.clear();
  APInt(8, 255, isSigned).toString(S, 8, isSigned, true);
  EXPECT_EQ(S.str().str(), "0377");
  S.clear();
  APInt(8, 255, isSigned).toString(S, 10, isSigned, true);
  EXPECT_EQ(S.str().str(), "255");
  S.clear();
  APInt(8, 255, isSigned).toString(S, 16, isSigned, true);
  EXPECT_EQ(S.str().str(), "0xFF");
  S.clear();
  APInt(8, 255, isSigned).toString(S, 36, isSigned, false);
  EXPECT_EQ(S.str().str(), "73");
  S.clear();

  isSigned = true;
  APInt(8, 255, isSigned).toString(S, 2, isSigned, true);
  EXPECT_EQ(S.str().str(), "-0b1");
  S.clear();
  APInt(8, 255, isSigned).toString(S, 8, isSigned, true);
  EXPECT_EQ(S.str().str(), "-01");
  S.clear();
  APInt(8, 255, isSigned).toString(S, 10, isSigned, true);
  EXPECT_EQ(S.str().str(), "-1");
  S.clear();
  APInt(8, 255, isSigned).toString(S, 16, isSigned, true);
  EXPECT_EQ(S.str().str(), "-0x1");
  S.clear();
  APInt(8, 255, isSigned).toString(S, 36, isSigned, false);
  EXPECT_EQ(S.str().str(), "-1");
  S.clear();
}

TEST(APIntTest, Log2) {
  EXPECT_EQ(APInt(15, 7).logBase2(), 2U);
  EXPECT_EQ(APInt(15, 7).ceilLogBase2(), 3U);
  EXPECT_EQ(APInt(15, 7).exactLogBase2(), -1);
  EXPECT_EQ(APInt(15, 8).logBase2(), 3U);
  EXPECT_EQ(APInt(15, 8).ceilLogBase2(), 3U);
  EXPECT_EQ(APInt(15, 8).exactLogBase2(), 3);
  EXPECT_EQ(APInt(15, 9).logBase2(), 3U);
  EXPECT_EQ(APInt(15, 9).ceilLogBase2(), 4U);
  EXPECT_EQ(APInt(15, 9).exactLogBase2(), -1);
}

TEST(APIntTest, magic) {
  EXPECT_EQ(APInt(32, 3).magic().m, APInt(32, "55555556", 16));
  EXPECT_EQ(APInt(32, 3).magic().s, 0U);
  EXPECT_EQ(APInt(32, 5).magic().m, APInt(32, "66666667", 16));
  EXPECT_EQ(APInt(32, 5).magic().s, 1U);
  EXPECT_EQ(APInt(32, 7).magic().m, APInt(32, "92492493", 16));
  EXPECT_EQ(APInt(32, 7).magic().s, 2U);
}

TEST(APIntTest, magicu) {
  EXPECT_EQ(APInt(32, 3).magicu().m, APInt(32, "AAAAAAAB", 16));
  EXPECT_EQ(APInt(32, 3).magicu().s, 1U);
  EXPECT_EQ(APInt(32, 5).magicu().m, APInt(32, "CCCCCCCD", 16));
  EXPECT_EQ(APInt(32, 5).magicu().s, 2U);
  EXPECT_EQ(APInt(32, 7).magicu().m, APInt(32, "24924925", 16));
  EXPECT_EQ(APInt(32, 7).magicu().s, 3U);
  EXPECT_EQ(APInt(64, 25).magicu(1).m, APInt(64, "A3D70A3D70A3D70B", 16));
  EXPECT_EQ(APInt(64, 25).magicu(1).s, 4U);
}

#ifdef GTEST_HAS_DEATH_TEST
#ifndef NDEBUG
TEST(APIntTest, StringDeath) {
  EXPECT_DEATH(APInt(0, "", 0), "Bitwidth too small");
  EXPECT_DEATH(APInt(32, "", 0), "Invalid string length");
  EXPECT_DEATH(APInt(32, "0", 0), "Radix should be 2, 8, 10, 16, or 36!");
  EXPECT_DEATH(APInt(32, "", 10), "Invalid string length");
  EXPECT_DEATH(APInt(32, "-", 10), "String is only a sign, needs a value.");
  EXPECT_DEATH(APInt(1, "1234", 10), "Insufficient bit width");
  EXPECT_DEATH(APInt(32, "\0", 10), "Invalid string length");
  EXPECT_DEATH(APInt(32, StringRef("1\02", 3), 10), "Invalid character in digit string");
  EXPECT_DEATH(APInt(32, "1L", 10), "Invalid character in digit string");
}
#endif
#endif

TEST(APIntTest, mul_clear) {
  APInt ValA(65, -1ULL);
  APInt ValB(65, 4);
  APInt ValC(65, 0);
  ValC = ValA * ValB;
  ValA *= ValB;
  EXPECT_EQ(ValA.toString(10, false), ValC.toString(10, false));
}

TEST(APIntTest, Rotate) {
  EXPECT_EQ(APInt(8, 1),  APInt(8, 1).rotl(0));
  EXPECT_EQ(APInt(8, 2),  APInt(8, 1).rotl(1));
  EXPECT_EQ(APInt(8, 4),  APInt(8, 1).rotl(2));
  EXPECT_EQ(APInt(8, 16), APInt(8, 1).rotl(4));
  EXPECT_EQ(APInt(8, 1),  APInt(8, 1).rotl(8));

  EXPECT_EQ(APInt(8, 16), APInt(8, 16).rotl(0));
  EXPECT_EQ(APInt(8, 32), APInt(8, 16).rotl(1));
  EXPECT_EQ(APInt(8, 64), APInt(8, 16).rotl(2));
  EXPECT_EQ(APInt(8, 1),  APInt(8, 16).rotl(4));
  EXPECT_EQ(APInt(8, 16), APInt(8, 16).rotl(8));

  EXPECT_EQ(APInt(32, 2), APInt(32, 1).rotl(33));
  EXPECT_EQ(APInt(32, 2), APInt(32, 1).rotl(APInt(32, 33)));

  EXPECT_EQ(APInt(32, 2), APInt(32, 1).rotl(33));
  EXPECT_EQ(APInt(32, 2), APInt(32, 1).rotl(APInt(32, 33)));
  EXPECT_EQ(APInt(32, 2), APInt(32, 1).rotl(APInt(33, 33)));
  EXPECT_EQ(APInt(32, (1 << 8)), APInt(32, 1).rotl(APInt(32, 40)));
  EXPECT_EQ(APInt(32, (1 << 30)), APInt(32, 1).rotl(APInt(31, 30)));
  EXPECT_EQ(APInt(32, (1 << 31)), APInt(32, 1).rotl(APInt(31, 31)));

  EXPECT_EQ(APInt(32, 1), APInt(32, 1).rotl(APInt(1, 0)));
  EXPECT_EQ(APInt(32, 2), APInt(32, 1).rotl(APInt(1, 1)));

  EXPECT_EQ(APInt(32, 16), APInt(32, 1).rotl(APInt(3, 4)));

  EXPECT_EQ(APInt(32, 1), APInt(32, 1).rotl(APInt(64, 64)));
  EXPECT_EQ(APInt(32, 2), APInt(32, 1).rotl(APInt(64, 65)));

  EXPECT_EQ(APInt(7, 24), APInt(7, 3).rotl(APInt(7, 3)));
  EXPECT_EQ(APInt(7, 24), APInt(7, 3).rotl(APInt(7, 10)));
  EXPECT_EQ(APInt(7, 24), APInt(7, 3).rotl(APInt(5, 10)));
  EXPECT_EQ(APInt(7, 6), APInt(7, 3).rotl(APInt(12, 120)));

  EXPECT_EQ(APInt(8, 16), APInt(8, 16).rotr(0));
  EXPECT_EQ(APInt(8, 8),  APInt(8, 16).rotr(1));
  EXPECT_EQ(APInt(8, 4),  APInt(8, 16).rotr(2));
  EXPECT_EQ(APInt(8, 1),  APInt(8, 16).rotr(4));
  EXPECT_EQ(APInt(8, 16), APInt(8, 16).rotr(8));

  EXPECT_EQ(APInt(8, 1),   APInt(8, 1).rotr(0));
  EXPECT_EQ(APInt(8, 128), APInt(8, 1).rotr(1));
  EXPECT_EQ(APInt(8, 64),  APInt(8, 1).rotr(2));
  EXPECT_EQ(APInt(8, 16),  APInt(8, 1).rotr(4));
  EXPECT_EQ(APInt(8, 1),   APInt(8, 1).rotr(8));

  EXPECT_EQ(APInt(32, (1 << 31)), APInt(32, 1).rotr(33));
  EXPECT_EQ(APInt(32, (1 << 31)), APInt(32, 1).rotr(APInt(32, 33)));

  EXPECT_EQ(APInt(32, (1 << 31)), APInt(32, 1).rotr(33));
  EXPECT_EQ(APInt(32, (1 << 31)), APInt(32, 1).rotr(APInt(32, 33)));
  EXPECT_EQ(APInt(32, (1 << 31)), APInt(32, 1).rotr(APInt(33, 33)));
  EXPECT_EQ(APInt(32, (1 << 24)), APInt(32, 1).rotr(APInt(32, 40)));

  EXPECT_EQ(APInt(32, (1 << 2)), APInt(32, 1).rotr(APInt(31, 30)));
  EXPECT_EQ(APInt(32, (1 << 1)), APInt(32, 1).rotr(APInt(31, 31)));

  EXPECT_EQ(APInt(32, 1), APInt(32, 1).rotr(APInt(1, 0)));
  EXPECT_EQ(APInt(32, (1 << 31)), APInt(32, 1).rotr(APInt(1, 1)));

  EXPECT_EQ(APInt(32, (1 << 28)), APInt(32, 1).rotr(APInt(3, 4)));

  EXPECT_EQ(APInt(32, 1), APInt(32, 1).rotr(APInt(64, 64)));
  EXPECT_EQ(APInt(32, (1 << 31)), APInt(32, 1).rotr(APInt(64, 65)));

  EXPECT_EQ(APInt(7, 48), APInt(7, 3).rotr(APInt(7, 3)));
  EXPECT_EQ(APInt(7, 48), APInt(7, 3).rotr(APInt(7, 10)));
  EXPECT_EQ(APInt(7, 48), APInt(7, 3).rotr(APInt(5, 10)));
  EXPECT_EQ(APInt(7, 65), APInt(7, 3).rotr(APInt(12, 120)));

  APInt Big(256, "00004000800000000000000000003fff8000000000000003", 16);
  APInt Rot(256, "3fff80000000000000030000000000000000000040008000", 16);
  EXPECT_EQ(Rot, Big.rotr(144));

  EXPECT_EQ(APInt(32, 8), APInt(32, 1).rotl(Big));
  EXPECT_EQ(APInt(32, (1 << 29)), APInt(32, 1).rotr(Big));
}

TEST(APIntTest, Splat) {
  APInt ValA(8, 0x01);
  EXPECT_EQ(ValA, APInt::getSplat(8, ValA));
  EXPECT_EQ(APInt(64, 0x0101010101010101ULL), APInt::getSplat(64, ValA));

  APInt ValB(3, 5);
  EXPECT_EQ(APInt(4, 0xD), APInt::getSplat(4, ValB));
  EXPECT_EQ(APInt(15, 0xDB6D), APInt::getSplat(15, ValB));
}

TEST(APIntTest, tcDecrement) {
  // Test single word decrement.

  // No out borrow.
  {
    integerPart singleWord = ~integerPart(0) << (integerPartWidth - 1);
    integerPart carry = APInt::tcDecrement(&singleWord, 1);
    EXPECT_EQ(carry, integerPart(0));
    EXPECT_EQ(singleWord, ~integerPart(0) >> 1);
  }

  // With out borrow.
  {
    integerPart singleWord = 0;
    integerPart carry = APInt::tcDecrement(&singleWord, 1);
    EXPECT_EQ(carry, integerPart(1));
    EXPECT_EQ(singleWord, ~integerPart(0));
  }

  // Test multiword decrement.

  // No across word borrow, no out borrow.
  {
    integerPart test[4] = {0x1, 0x1, 0x1, 0x1};
    integerPart expected[4] = {0x0, 0x1, 0x1, 0x1};
    APInt::tcDecrement(test, 4);
    EXPECT_EQ(APInt::tcCompare(test, expected, 4), 0);
  }

  // 1 across word borrow, no out borrow.
  {
    integerPart test[4] = {0x0, 0xF, 0x1, 0x1};
    integerPart expected[4] = {~integerPart(0), 0xE, 0x1, 0x1};
    integerPart carry = APInt::tcDecrement(test, 4);
    EXPECT_EQ(carry, integerPart(0));
    EXPECT_EQ(APInt::tcCompare(test, expected, 4), 0);
  }

  // 2 across word borrow, no out borrow.
  {
    integerPart test[4] = {0x0, 0x0, 0xC, 0x1};
    integerPart expected[4] = {~integerPart(0), ~integerPart(0), 0xB, 0x1};
    integerPart carry = APInt::tcDecrement(test, 4);
    EXPECT_EQ(carry, integerPart(0));
    EXPECT_EQ(APInt::tcCompare(test, expected, 4), 0);
  }

  // 3 across word borrow, no out borrow.
  {
    integerPart test[4] = {0x0, 0x0, 0x0, 0x1};
    integerPart expected[4] = {~integerPart(0), ~integerPart(0), ~integerPart(0), 0x0};
    integerPart carry = APInt::tcDecrement(test, 4);
    EXPECT_EQ(carry, integerPart(0));
    EXPECT_EQ(APInt::tcCompare(test, expected, 4), 0);
  }

  // 3 across word borrow, with out borrow.
  {
    integerPart test[4] = {0x0, 0x0, 0x0, 0x0};
    integerPart expected[4] = {~integerPart(0), ~integerPart(0), ~integerPart(0), ~integerPart(0)};
    integerPart carry = APInt::tcDecrement(test, 4);
    EXPECT_EQ(carry, integerPart(1));
    EXPECT_EQ(APInt::tcCompare(test, expected, 4), 0);
  }
}

TEST(APIntTest, arrayAccess) {
  // Single word check.
  uint64_t E1 = 0x2CA7F46BF6569915ULL;
  APInt A1(64, E1);
  for (unsigned i = 0, e = 64; i < e; ++i) {
    EXPECT_EQ(bool(E1 & (1ULL << i)),
              A1[i]);
  }

  // Multiword check.
  integerPart E2[4] = {
    0xEB6EB136591CBA21ULL,
    0x7B9358BD6A33F10AULL,
    0x7E7FFA5EADD8846ULL,
    0x305F341CA00B613DULL
  };
  APInt A2(integerPartWidth*4, E2);
  for (unsigned i = 0; i < 4; ++i) {
    for (unsigned j = 0; j < integerPartWidth; ++j) {
      EXPECT_EQ(bool(E2[i] & (1ULL << j)),
                A2[i*integerPartWidth + j]);
    }
  }
}

TEST(APIntTest, LargeAPIntConstruction) {
  // Check that we can properly construct very large APInt. It is very
  // unlikely that people will ever do this, but it is a legal input,
  // so we should not crash on it.
  APInt A9(UINT32_MAX, 0);
  EXPECT_FALSE(A9.getBoolValue());
}

TEST(APIntTest, nearestLogBase2) {
  // Single word check.

  // Test round up.
  uint64_t I1 = 0x1800001;
  APInt A1(64, I1);
  EXPECT_EQ(A1.nearestLogBase2(), A1.ceilLogBase2());

  // Test round down.
  uint64_t I2 = 0x1000011;
  APInt A2(64, I2);
  EXPECT_EQ(A2.nearestLogBase2(), A2.logBase2());

  // Test ties round up.
  uint64_t I3 = 0x1800000;
  APInt A3(64, I3);
  EXPECT_EQ(A3.nearestLogBase2(), A3.ceilLogBase2());

  // Multiple word check.

  // Test round up.
  integerPart I4[4] = {0x0, 0xF, 0x18, 0x0};
  APInt A4(integerPartWidth*4, I4);
  EXPECT_EQ(A4.nearestLogBase2(), A4.ceilLogBase2());

  // Test round down.
  integerPart I5[4] = {0x0, 0xF, 0x10, 0x0};
  APInt A5(integerPartWidth*4, I5);
  EXPECT_EQ(A5.nearestLogBase2(), A5.logBase2());

  // Test ties round up.
  uint64_t I6[4] = {0x0, 0x0, 0x0, 0x18};
  APInt A6(integerPartWidth*4, I6);
  EXPECT_EQ(A6.nearestLogBase2(), A6.ceilLogBase2());

  // Test BitWidth == 1 special cases.
  APInt A7(1, 1);
  EXPECT_EQ(A7.nearestLogBase2(), 0ULL);
  APInt A8(1, 0);
  EXPECT_EQ(A8.nearestLogBase2(), UINT32_MAX);

  // Test the zero case when we have a bit width large enough such
  // that the bit width is larger than UINT32_MAX-1.
  APInt A9(UINT32_MAX, 0);
  EXPECT_EQ(A9.nearestLogBase2(), UINT32_MAX);
}

TEST(APIntTest, IsSplat) {
  APInt A(32, 0x01010101);
  EXPECT_FALSE(A.isSplat(1));
  EXPECT_FALSE(A.isSplat(2));
  EXPECT_FALSE(A.isSplat(4));
  EXPECT_TRUE(A.isSplat(8));
  EXPECT_TRUE(A.isSplat(16));
  EXPECT_TRUE(A.isSplat(32));

  APInt B(24, 0xAAAAAA);
  EXPECT_FALSE(B.isSplat(1));
  EXPECT_TRUE(B.isSplat(2));
  EXPECT_TRUE(B.isSplat(4));
  EXPECT_TRUE(B.isSplat(8));
  EXPECT_TRUE(B.isSplat(24));

  APInt C(24, 0xABAAAB);
  EXPECT_FALSE(C.isSplat(1));
  EXPECT_FALSE(C.isSplat(2));
  EXPECT_FALSE(C.isSplat(4));
  EXPECT_FALSE(C.isSplat(8));
  EXPECT_TRUE(C.isSplat(24));

  APInt D(32, 0xABBAABBA);
  EXPECT_FALSE(D.isSplat(1));
  EXPECT_FALSE(D.isSplat(2));
  EXPECT_FALSE(D.isSplat(4));
  EXPECT_FALSE(D.isSplat(8));
  EXPECT_TRUE(D.isSplat(16));
  EXPECT_TRUE(D.isSplat(32));

  APInt E(32, 0);
  EXPECT_TRUE(E.isSplat(1));
  EXPECT_TRUE(E.isSplat(2));
  EXPECT_TRUE(E.isSplat(4));
  EXPECT_TRUE(E.isSplat(8));
  EXPECT_TRUE(E.isSplat(16));
  EXPECT_TRUE(E.isSplat(32));
}

TEST(APIntTest, isMask) {
  EXPECT_FALSE(APIntOps::isMask(APInt(32, 0x01010101)));
  EXPECT_FALSE(APIntOps::isMask(APInt(32, 0xf0000000)));
  EXPECT_FALSE(APIntOps::isMask(APInt(32, 0xffff0000)));
  EXPECT_FALSE(APIntOps::isMask(APInt(32, 0xff << 1)));

  for (int N : { 1, 2, 3, 4, 7, 8, 16, 32, 64, 127, 128, 129, 256 }) {
    EXPECT_FALSE(APIntOps::isMask(APInt(N, 0)));

    APInt One(N, 1);
    for (int I = 1; I <= N; ++I) {
      APInt MaskVal = One.shl(I) - 1;
      EXPECT_TRUE(APIntOps::isMask(MaskVal));
    }
  }
}

#if defined(__clang__)
// Disable the pragma warning from versions of Clang without -Wself-move
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-pragmas"
// Disable the warning that triggers on exactly what is being tested.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wself-move"
#endif
TEST(APIntTest, SelfMoveAssignment) {
  APInt X(32, 0xdeadbeef);
  X = std::move(X);
  EXPECT_EQ(32u, X.getBitWidth());
  EXPECT_EQ(0xdeadbeefULL, X.getLimitedValue());

  uint64_t Bits[] = {0xdeadbeefdeadbeefULL, 0xdeadbeefdeadbeefULL};
  APInt Y(128, Bits);
  Y = std::move(Y);
  EXPECT_EQ(128u, Y.getBitWidth());
  EXPECT_EQ(~0ULL, Y.getLimitedValue());
  const uint64_t *Raw = Y.getRawData();
  EXPECT_EQ(2u, Y.getNumWords());
  EXPECT_EQ(0xdeadbeefdeadbeefULL, Raw[0]);
  EXPECT_EQ(0xdeadbeefdeadbeefULL, Raw[1]);
}
#if defined(__clang__)
#pragma clang diagnostic pop
#pragma clang diagnostic pop
#endif
}

TEST(APIntTest, reverseBits) {
  EXPECT_EQ(1, APInt(1, 1).reverseBits());
  EXPECT_EQ(0, APInt(1, 0).reverseBits());

  EXPECT_EQ(3, APInt(2, 3).reverseBits());
  EXPECT_EQ(3, APInt(2, 3).reverseBits());

  EXPECT_EQ(0xb, APInt(4, 0xd).reverseBits());
  EXPECT_EQ(0xd, APInt(4, 0xb).reverseBits());
  EXPECT_EQ(0xf, APInt(4, 0xf).reverseBits());

  EXPECT_EQ(0x30, APInt(7, 0x6).reverseBits());
  EXPECT_EQ(0x5a, APInt(7, 0x2d).reverseBits());

  EXPECT_EQ(0x0f, APInt(8, 0xf0).reverseBits());
  EXPECT_EQ(0xf0, APInt(8, 0x0f).reverseBits());

  EXPECT_EQ(0x0f0f, APInt(16, 0xf0f0).reverseBits());
  EXPECT_EQ(0xf0f0, APInt(16, 0x0f0f).reverseBits());

  EXPECT_EQ(0x0f0f0f0f, APInt(32, 0xf0f0f0f0).reverseBits());
  EXPECT_EQ(0xf0f0f0f0, APInt(32, 0x0f0f0f0f).reverseBits());

  EXPECT_EQ(0x402880a0 >> 1, APInt(31, 0x05011402).reverseBits());

  EXPECT_EQ(0x0f0f0f0f, APInt(32, 0xf0f0f0f0).reverseBits());
  EXPECT_EQ(0xf0f0f0f0, APInt(32, 0x0f0f0f0f).reverseBits());

  EXPECT_EQ(0x0f0f0f0f0f0f0f0f, APInt(64, 0xf0f0f0f0f0f0f0f0).reverseBits());
  EXPECT_EQ(0xf0f0f0f0f0f0f0f0, APInt(64, 0x0f0f0f0f0f0f0f0f).reverseBits());

  for (unsigned N : { 1, 8, 16, 24, 31, 32, 33,
                      63, 64, 65, 127, 128, 257, 1024 }) {
    for (unsigned I = 0; I < N; ++I) {
      APInt X = APInt::getOneBitSet(N, I);
      APInt Y = APInt::getOneBitSet(N, N - (I + 1));
      EXPECT_EQ(Y, X.reverseBits());
      EXPECT_EQ(X, Y.reverseBits());
    }
  }
}

TEST(APIntTest, extractBits) {
  APInt i32(32, 0x1234567);
  EXPECT_EQ(0x3456, i32.extractBits(16, 4));

  APInt i257(257, 0xFFFFFFFFFF0000FFull, true);
  EXPECT_EQ(0xFFu, i257.extractBits(16, 0));
  EXPECT_EQ((0xFFu >> 1), i257.extractBits(16, 1));
  EXPECT_EQ(-1, i257.extractBits(32, 64).getSExtValue());
  EXPECT_EQ(-1, i257.extractBits(128, 128).getSExtValue());
  EXPECT_EQ(-1, i257.extractBits(66, 191).getSExtValue());
  EXPECT_EQ(static_cast<int64_t>(0xFFFFFFFFFF80007Full),
            i257.extractBits(128, 1).getSExtValue());
  EXPECT_EQ(static_cast<int64_t>(0xFFFFFFFFFF80007Full),
            i257.extractBits(129, 1).getSExtValue());
}

TEST(APIntTest, getLowBitsSet) {
  APInt i128lo64 = APInt::getLowBitsSet(128, 64);
  EXPECT_EQ(0u, i128lo64.countLeadingOnes());
  EXPECT_EQ(64u, i128lo64.countLeadingZeros());
  EXPECT_EQ(64u, i128lo64.getActiveBits());
  EXPECT_EQ(0u, i128lo64.countTrailingZeros());
  EXPECT_EQ(64u, i128lo64.countTrailingOnes());
  EXPECT_EQ(64u, i128lo64.countPopulation());
}

TEST(APIntTest, getHighBitsSet) {
  APInt i64hi32 = APInt::getHighBitsSet(64, 32);
  EXPECT_EQ(32u, i64hi32.countLeadingOnes());
  EXPECT_EQ(0u, i64hi32.countLeadingZeros());
  EXPECT_EQ(64u, i64hi32.getActiveBits());
  EXPECT_EQ(32u, i64hi32.countTrailingZeros());
  EXPECT_EQ(0u, i64hi32.countTrailingOnes());
  EXPECT_EQ(32u, i64hi32.countPopulation());
}
