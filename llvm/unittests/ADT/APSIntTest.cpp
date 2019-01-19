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

#if defined(GTEST_HAS_DEATH_TEST) && !defined(NDEBUG)

TEST(APSIntTest, StringDeath) {
  EXPECT_DEATH(APSInt(""), "Invalid string length");
  EXPECT_DEATH(APSInt("1a"), "Invalid character in digit string");
}

#endif

} // end anonymous namespace
