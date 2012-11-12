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
  EXPECT_TRUE(Shl[0]);
  EXPECT_FALSE(Shl[1]);
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

// XFAIL this test on FreeBSD where the system gcc-4.2.1 seems to miscompile it.
#if defined(__llvm__) || !defined(__FreeBSD__)

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

#endif

TEST(APIntTest, i65_Count) {
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

  APInt Big(256, "00004000800000000000000000003fff8000000000000000", 16);
  APInt Rot(256, "3fff80000000000000000000000000000000000040008000", 16);
  EXPECT_EQ(Rot, Big.rotr(144));
}

}
