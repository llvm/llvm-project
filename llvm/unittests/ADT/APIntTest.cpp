//===- llvm/unittest/ADT/APInt.cpp - APInt unit tests ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <ostream>
#include "gtest/gtest.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallString.h"

using namespace llvm;

namespace {

// Test that APInt shift left works when bitwidth > 64 and shiftamt == 0
TEST(APIntTest, ShiftLeftByZero) {
  APInt One = APInt::getNullValue(65) + 1;
  APInt Shl = One.shl(0);
  EXPECT_EQ(true, Shl[0]);
  EXPECT_EQ(false, Shl[1]);
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

TEST(APIntTest, i65_Count) {
  APInt i65minus(65, 0, true);
  i65minus.set(64);
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

  // Additions.
  EXPECT_EQ(two, one + one);
  EXPECT_EQ(zero, neg_one + one);
  EXPECT_EQ(neg_two, neg_one + neg_one);

  // Subtractions.
  EXPECT_EQ(neg_two, neg_one - one);
  EXPECT_EQ(two, one - neg_one);
  EXPECT_EQ(zero, one - one);

  // Shifts.
  EXPECT_EQ(zero, one << one);
  EXPECT_EQ(one, one << zero);
  EXPECT_EQ(zero, one.shl(1));
  EXPECT_EQ(one, one.shl(0));
  EXPECT_EQ(zero, one.lshr(1));
  EXPECT_EQ(zero, one.ashr(1));

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

#ifdef GTEST_HAS_DEATH_TEST
#ifndef NDEBUG
TEST(APIntTest, StringDeath) {
  EXPECT_DEATH(APInt(0, "", 0), "Bitwidth too small");
  EXPECT_DEATH(APInt(32, "", 0), "Invalid string length");
  EXPECT_DEATH(APInt(32, "0", 0), "Radix should be 2, 8, 10, or 16!");
  EXPECT_DEATH(APInt(32, "", 10), "Invalid string length");
  EXPECT_DEATH(APInt(32, "-", 10), "String is only a sign, needs a value.");
  EXPECT_DEATH(APInt(1, "1234", 10), "Insufficient bit width");
  EXPECT_DEATH(APInt(32, "\0", 10), "Invalid string length");
  EXPECT_DEATH(APInt(32, StringRef("1\02", 3), 10), "Invalid character in digit string");
  EXPECT_DEATH(APInt(32, "1L", 10), "Invalid character in digit string");
}
#endif
#endif

}
