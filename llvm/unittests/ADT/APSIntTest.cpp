//===- llvm/unittest/ADT/APSIntTest.cpp - APSInt unit tests ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/APSInt.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(APSIntTest, MoveTest) {
  APSInt A(32, true);
  EXPECT_TRUE(A.isUnsigned());

  APSInt B(128, false);
  A = B;
  EXPECT_FALSE(A.isUnsigned());

  APSInt C(B);
  EXPECT_FALSE(C.isUnsigned());

  APInt Wide(256, 0);
  const uint64_t *Bits = Wide.getRawData();
  APSInt D(std::move(Wide));
  EXPECT_TRUE(D.isUnsigned());
  EXPECT_EQ(Bits, D.getRawData()); // Verify that "Wide" was really moved.

  A = APSInt(64, true);
  EXPECT_TRUE(A.isUnsigned());

  Wide = APInt(128, 1);
  Bits = Wide.getRawData();
  A = std::move(Wide);
  EXPECT_TRUE(A.isUnsigned());
  EXPECT_EQ(Bits, A.getRawData()); // Verify that "Wide" was really moved.
}

TEST(APSIntTest, get) {
  EXPECT_TRUE(APSInt::get(7).isSigned());
  EXPECT_EQ(64u, APSInt::get(7).getBitWidth());
  EXPECT_EQ(7u, APSInt::get(7).getZExtValue());
  EXPECT_EQ(7, APSInt::get(7).getSExtValue());
  EXPECT_TRUE(APSInt::get(-7).isSigned());
  EXPECT_EQ(64u, APSInt::get(-7).getBitWidth());
  EXPECT_EQ(-7, APSInt::get(-7).getSExtValue());
  EXPECT_EQ(UINT64_C(0) - 7, APSInt::get(-7).getZExtValue());
}

TEST(APSIntTest, getUnsigned) {
  EXPECT_TRUE(APSInt::getUnsigned(7).isUnsigned());
  EXPECT_EQ(64u, APSInt::getUnsigned(7).getBitWidth());
  EXPECT_EQ(7u, APSInt::getUnsigned(7).getZExtValue());
  EXPECT_EQ(7, APSInt::getUnsigned(7).getSExtValue());
  EXPECT_TRUE(APSInt::getUnsigned(-7).isUnsigned());
  EXPECT_EQ(64u, APSInt::getUnsigned(-7).getBitWidth());
  EXPECT_EQ(-7, APSInt::getUnsigned(-7).getSExtValue());
  EXPECT_EQ(UINT64_C(0) - 7, APSInt::getUnsigned(-7).getZExtValue());
}

TEST(APSIntTest, isRepresentableByInt64) {
  ASSERT_TRUE(APSInt(APInt(3, 7), true).isRepresentableByInt64());
  ASSERT_TRUE(APSInt(APInt(128, 7), true).isRepresentableByInt64());
  ASSERT_TRUE(APSInt(APInt(128, 7), false).isRepresentableByInt64());
  ASSERT_TRUE(APSInt(APInt(64, -1), false).isRepresentableByInt64());
  ASSERT_FALSE(APSInt(APInt(64, (uint64_t)-1), true).isRepresentableByInt64());
}
TEST(APSIntTest, getExtValue) {
  EXPECT_TRUE(APSInt(APInt(3, 7), true).isUnsigned());
  EXPECT_TRUE(APSInt(APInt(3, 7), false).isSigned());
  EXPECT_TRUE(APSInt(APInt(4, 7), true).isUnsigned());
  EXPECT_TRUE(APSInt(APInt(4, 7), false).isSigned());
  EXPECT_TRUE(APSInt(APInt(4, -7), true).isUnsigned());
  EXPECT_TRUE(APSInt(APInt(4, -7), false).isSigned());
  EXPECT_EQ(7, APSInt(APInt(3, 7), true).getExtValue());
  EXPECT_EQ(-1, APSInt(APInt(3, 7), false).getExtValue());
  EXPECT_EQ(7, APSInt(APInt(4, 7), true).getExtValue());
  EXPECT_EQ(7, APSInt(APInt(4, 7), false).getExtValue());
  EXPECT_EQ(9, APSInt(APInt(4, -7), true).getExtValue());
  EXPECT_EQ(-7, APSInt(APInt(4, -7), false).getExtValue());
}
TEST(APSIntTest, tryExtValue) {
  ASSERT_EQ(-7, APSInt(APInt(64, -7), false).tryExtValue().value_or(42));
  ASSERT_EQ(42, APSInt(APInt(128, -7), false).tryExtValue().value_or(42));
  ASSERT_EQ(-1,
            APSInt(APInt::getAllOnes(128), false).tryExtValue().value_or(42));
  ASSERT_EQ(42, APSInt(APInt(64, -7), true).tryExtValue().value_or(42));
  ASSERT_EQ(1, APSInt(APInt(128, 1), true).tryExtValue().value_or(42));
  ASSERT_EQ(42,
            APSInt(APInt::getAllOnes(128), true).tryExtValue().value_or(42));
}

TEST(APSIntTest, compareValues) {
  auto U = [](uint64_t V) { return APSInt::getUnsigned(V); };
  auto S = [](int64_t V) { return APSInt::get(V); };

  // Bit-width matches and is-signed.
  EXPECT_TRUE(APSInt::compareValues(S(7), S(8)) < 0);
  EXPECT_TRUE(APSInt::compareValues(S(8), S(7)) > 0);
  EXPECT_TRUE(APSInt::compareValues(S(7), S(7)) == 0);
  EXPECT_TRUE(APSInt::compareValues(S(-7), S(8)) < 0);
  EXPECT_TRUE(APSInt::compareValues(S(8), S(-7)) > 0);
  EXPECT_TRUE(APSInt::compareValues(S(-7), S(-7)) == 0);
  EXPECT_TRUE(APSInt::compareValues(S(-7), S(-8)) > 0);
  EXPECT_TRUE(APSInt::compareValues(S(-8), S(-7)) < 0);
  EXPECT_TRUE(APSInt::compareValues(S(-7), S(-7)) == 0);

  // Bit-width matches and not is-signed.
  EXPECT_TRUE(APSInt::compareValues(U(7), U(8)) < 0);
  EXPECT_TRUE(APSInt::compareValues(U(8), U(7)) > 0);
  EXPECT_TRUE(APSInt::compareValues(U(7), U(7)) == 0);

  // Bit-width matches and mixed signs.
  EXPECT_TRUE(APSInt::compareValues(U(7), S(8)) < 0);
  EXPECT_TRUE(APSInt::compareValues(U(8), S(7)) > 0);
  EXPECT_TRUE(APSInt::compareValues(U(7), S(7)) == 0);
  EXPECT_TRUE(APSInt::compareValues(U(8), S(-7)) > 0);

  // Bit-width mismatch and is-signed.
  EXPECT_TRUE(APSInt::compareValues(S(7).trunc(32), S(8)) < 0);
  EXPECT_TRUE(APSInt::compareValues(S(8).trunc(32), S(7)) > 0);
  EXPECT_TRUE(APSInt::compareValues(S(7).trunc(32), S(7)) == 0);
  EXPECT_TRUE(APSInt::compareValues(S(-7).trunc(32), S(8)) < 0);
  EXPECT_TRUE(APSInt::compareValues(S(8).trunc(32), S(-7)) > 0);
  EXPECT_TRUE(APSInt::compareValues(S(-7).trunc(32), S(-7)) == 0);
  EXPECT_TRUE(APSInt::compareValues(S(-7).trunc(32), S(-8)) > 0);
  EXPECT_TRUE(APSInt::compareValues(S(-8).trunc(32), S(-7)) < 0);
  EXPECT_TRUE(APSInt::compareValues(S(-7).trunc(32), S(-7)) == 0);
  EXPECT_TRUE(APSInt::compareValues(S(7), S(8).trunc(32)) < 0);
  EXPECT_TRUE(APSInt::compareValues(S(8), S(7).trunc(32)) > 0);
  EXPECT_TRUE(APSInt::compareValues(S(7), S(7).trunc(32)) == 0);
  EXPECT_TRUE(APSInt::compareValues(S(-7), S(8).trunc(32)) < 0);
  EXPECT_TRUE(APSInt::compareValues(S(8), S(-7).trunc(32)) > 0);
  EXPECT_TRUE(APSInt::compareValues(S(-7), S(-7).trunc(32)) == 0);
  EXPECT_TRUE(APSInt::compareValues(S(-7), S(-8).trunc(32)) > 0);
  EXPECT_TRUE(APSInt::compareValues(S(-8), S(-7).trunc(32)) < 0);
  EXPECT_TRUE(APSInt::compareValues(S(-7), S(-7).trunc(32)) == 0);

  // Bit-width mismatch and not is-signed.
  EXPECT_TRUE(APSInt::compareValues(U(7), U(8).trunc(32)) < 0);
  EXPECT_TRUE(APSInt::compareValues(U(8), U(7).trunc(32)) > 0);
  EXPECT_TRUE(APSInt::compareValues(U(7), U(7).trunc(32)) == 0);
  EXPECT_TRUE(APSInt::compareValues(U(7).trunc(32), U(8)) < 0);
  EXPECT_TRUE(APSInt::compareValues(U(8).trunc(32), U(7)) > 0);
  EXPECT_TRUE(APSInt::compareValues(U(7).trunc(32), U(7)) == 0);

  // Bit-width mismatch and mixed signs.
  EXPECT_TRUE(APSInt::compareValues(U(7).trunc(32), S(8)) < 0);
  EXPECT_TRUE(APSInt::compareValues(U(8).trunc(32), S(7)) > 0);
  EXPECT_TRUE(APSInt::compareValues(U(7).trunc(32), S(7)) == 0);
  EXPECT_TRUE(APSInt::compareValues(U(8).trunc(32), S(-7)) > 0);
  EXPECT_TRUE(APSInt::compareValues(U(7), S(8).trunc(32)) < 0);
  EXPECT_TRUE(APSInt::compareValues(U(8), S(7).trunc(32)) > 0);
  EXPECT_TRUE(APSInt::compareValues(U(7), S(7).trunc(32)) == 0);
  EXPECT_TRUE(APSInt::compareValues(U(8), S(-7).trunc(32)) > 0);
}

TEST(APSIntTest, FromString) {
  EXPECT_EQ(APSInt("1").getExtValue(), 1);
  EXPECT_EQ(APSInt("-1").getExtValue(), -1);
  EXPECT_EQ(APSInt("0").getExtValue(), 0);
  EXPECT_EQ(APSInt("56789").getExtValue(), 56789);
  EXPECT_EQ(APSInt("-1234").getExtValue(), -1234);
}

TEST(APSIntTest, FromStringBitWidth) {
  EXPECT_EQ(APSInt("0").getBitWidth(), 1U);
  EXPECT_EQ(APSInt("000").getBitWidth(), 1U);
  EXPECT_EQ(APSInt("1").getBitWidth(), 1U);
  EXPECT_EQ(APSInt("2").getBitWidth(), 2U);
  EXPECT_EQ(APSInt("3").getBitWidth(), 2U);
  EXPECT_EQ(APSInt("003").getBitWidth(), 2U);
  EXPECT_EQ(APSInt("15").getBitWidth(), 4U);
  EXPECT_EQ(APSInt("16").getBitWidth(), 5U);
  EXPECT_EQ(APSInt("17").getBitWidth(), 5U);

  EXPECT_EQ(APSInt("-0").getBitWidth(), 1U);
  EXPECT_EQ(APSInt("-000").getBitWidth(), 1U);
  EXPECT_EQ(APSInt("-1").getBitWidth(), 1U);
  EXPECT_EQ(APSInt("-2").getBitWidth(), 2U);
  EXPECT_EQ(APSInt("-3").getBitWidth(), 3U);
  EXPECT_EQ(APSInt("-003").getBitWidth(), 3U);
  EXPECT_EQ(APSInt("-5").getBitWidth(), 4U);
  EXPECT_EQ(APSInt("-15").getBitWidth(), 5U);
  EXPECT_EQ(APSInt("-16").getBitWidth(), 5U);
  EXPECT_EQ(APSInt("-17").getBitWidth(), 6U);
}

#if defined(GTEST_HAS_DEATH_TEST) && !defined(NDEBUG)

TEST(APSIntTest, StringDeath) {
  EXPECT_DEATH((void)APSInt(""), "Invalid string length");
  EXPECT_DEATH((void)APSInt("1a"), "Invalid character in digit string");
}

#endif

TEST(APSIntTest, SignedHighBit) {
  APSInt False(APInt(1, 0), false);
  APSInt True(APInt(1, 1), false);
  APSInt CharMin(APInt(8, 0), false);
  APSInt CharSmall(APInt(8, 0x13), false);
  APSInt CharBoundaryUnder(APInt(8, 0x7f), false);
  APSInt CharBoundaryOver(APInt(8, 0x80), false);
  APSInt CharLarge(APInt(8, 0xd9), false);
  APSInt CharMax(APInt(8, 0xff), false);

  EXPECT_FALSE(False.isNegative());
  EXPECT_TRUE(False.isNonNegative());
  EXPECT_FALSE(False.isStrictlyPositive());

  EXPECT_TRUE(True.isNegative());
  EXPECT_FALSE(True.isNonNegative());
  EXPECT_FALSE(True.isStrictlyPositive());

  EXPECT_FALSE(CharMin.isNegative());
  EXPECT_TRUE(CharMin.isNonNegative());
  EXPECT_FALSE(CharMin.isStrictlyPositive());

  EXPECT_FALSE(CharSmall.isNegative());
  EXPECT_TRUE(CharSmall.isNonNegative());
  EXPECT_TRUE(CharSmall.isStrictlyPositive());

  EXPECT_FALSE(CharBoundaryUnder.isNegative());
  EXPECT_TRUE(CharBoundaryUnder.isNonNegative());
  EXPECT_TRUE(CharBoundaryUnder.isStrictlyPositive());

  EXPECT_TRUE(CharBoundaryOver.isNegative());
  EXPECT_FALSE(CharBoundaryOver.isNonNegative());
  EXPECT_FALSE(CharBoundaryOver.isStrictlyPositive());

  EXPECT_TRUE(CharLarge.isNegative());
  EXPECT_FALSE(CharLarge.isNonNegative());
  EXPECT_FALSE(CharLarge.isStrictlyPositive());

  EXPECT_TRUE(CharMax.isNegative());
  EXPECT_FALSE(CharMax.isNonNegative());
  EXPECT_FALSE(CharMax.isStrictlyPositive());
}

TEST(APSIntTest, UnsignedHighBit) {
  APSInt False(APInt(1, 0));
  APSInt True(APInt(1, 1));
  APSInt CharMin(APInt(8, 0));
  APSInt CharSmall(APInt(8, 0x13));
  APSInt CharBoundaryUnder(APInt(8, 0x7f));
  APSInt CharBoundaryOver(APInt(8, 0x80));
  APSInt CharLarge(APInt(8, 0xd9));
  APSInt CharMax(APInt(8, 0xff));

  EXPECT_FALSE(False.isNegative());
  EXPECT_TRUE(False.isNonNegative());
  EXPECT_FALSE(False.isStrictlyPositive());

  EXPECT_FALSE(True.isNegative());
  EXPECT_TRUE(True.isNonNegative());
  EXPECT_TRUE(True.isStrictlyPositive());

  EXPECT_FALSE(CharMin.isNegative());
  EXPECT_TRUE(CharMin.isNonNegative());
  EXPECT_FALSE(CharMin.isStrictlyPositive());

  EXPECT_FALSE(CharSmall.isNegative());
  EXPECT_TRUE(CharSmall.isNonNegative());
  EXPECT_TRUE(CharSmall.isStrictlyPositive());

  EXPECT_FALSE(CharBoundaryUnder.isNegative());
  EXPECT_TRUE(CharBoundaryUnder.isNonNegative());
  EXPECT_TRUE(CharBoundaryUnder.isStrictlyPositive());

  EXPECT_FALSE(CharBoundaryOver.isNegative());
  EXPECT_TRUE(CharBoundaryOver.isNonNegative());
  EXPECT_TRUE(CharBoundaryOver.isStrictlyPositive());

  EXPECT_FALSE(CharLarge.isNegative());
  EXPECT_TRUE(CharLarge.isNonNegative());
  EXPECT_TRUE(CharLarge.isStrictlyPositive());

  EXPECT_FALSE(CharMax.isNegative());
  EXPECT_TRUE(CharMax.isNonNegative());
  EXPECT_TRUE(CharMax.isStrictlyPositive());
}

} // end anonymous namespace
