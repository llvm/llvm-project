//===- MPIntTest.cpp - Tests for MPInt ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/DynamicAPInt.h"
#include "llvm/ADT/SlowDynamicAPInt.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {
// googletest boilerplate to run the same tests with both MPInt and SlowMPInt.
template <typename> class IntTest : public testing::Test {};
using TypeList = testing::Types<DynamicAPInt, detail::SlowDynamicAPInt>;

// This is for pretty-printing the test name with the name of the class in use.
class TypeNames {
public:
  template <typename T>
  static std::string GetName(int) { // NOLINT; gtest mandates this name.
    if (std::is_same<T, DynamicAPInt>())
      return "MPInt";
    if (std::is_same<T, detail::SlowDynamicAPInt>())
      return "SlowMPInt";
    llvm_unreachable("Unknown class!");
  }
};
TYPED_TEST_SUITE(IntTest, TypeList, TypeNames);

TYPED_TEST(IntTest, ops) {
  TypeParam Two(2), Five(5), Seven(7), Ten(10);
  EXPECT_EQ(Five + Five, Ten);
  EXPECT_EQ(Five * Five, 2 * Ten + Five);
  EXPECT_EQ(Five * Five, 3 * Ten - Five);
  EXPECT_EQ(Five * Two, Ten);
  EXPECT_EQ(Five / Two, Two);
  EXPECT_EQ(Five % Two, Two / Two);

  EXPECT_EQ(-Ten % Seven, -10 % 7);
  EXPECT_EQ(Ten % -Seven, 10 % -7);
  EXPECT_EQ(-Ten % -Seven, -10 % -7);
  EXPECT_EQ(Ten % Seven, 10 % 7);

  EXPECT_EQ(-Ten / Seven, -10 / 7);
  EXPECT_EQ(Ten / -Seven, 10 / -7);
  EXPECT_EQ(-Ten / -Seven, -10 / -7);
  EXPECT_EQ(Ten / Seven, 10 / 7);

  TypeParam X = Ten;
  X += Five;
  EXPECT_EQ(X, 15);
  X *= Two;
  EXPECT_EQ(X, 30);
  X /= Seven;
  EXPECT_EQ(X, 4);
  X -= Two * 10;
  EXPECT_EQ(X, -16);
  X *= 2 * Two;
  EXPECT_EQ(X, -64);
  X /= Two / -2;
  EXPECT_EQ(X, 64);

  EXPECT_LE(Ten, Ten);
  EXPECT_GE(Ten, Ten);
  EXPECT_EQ(Ten, Ten);
  EXPECT_FALSE(Ten != Ten);
  EXPECT_FALSE(Ten < Ten);
  EXPECT_FALSE(Ten > Ten);
  EXPECT_LT(Five, Ten);
  EXPECT_GT(Ten, Five);
}

TYPED_TEST(IntTest, ops64Overloads) {
  TypeParam Two(2), Five(5), Seven(7), Ten(10);
  EXPECT_EQ(Five + 5, Ten);
  EXPECT_EQ(Five + 5, 5 + Five);
  EXPECT_EQ(Five * 5, 2 * Ten + 5);
  EXPECT_EQ(Five * 5, 3 * Ten - 5);
  EXPECT_EQ(Five * Two, Ten);
  EXPECT_EQ(5 / Two, 2);
  EXPECT_EQ(Five / 2, 2);
  EXPECT_EQ(2 % Two, 0);
  EXPECT_EQ(2 - Two, 0);
  EXPECT_EQ(2 % Two, Two % 2);

  TypeParam X = Ten;
  X += 5;
  EXPECT_EQ(X, 15);
  X *= 2;
  EXPECT_EQ(X, 30);
  X /= 7;
  EXPECT_EQ(X, 4);
  X -= 20;
  EXPECT_EQ(X, -16);
  X *= 4;
  EXPECT_EQ(X, -64);
  X /= -1;
  EXPECT_EQ(X, 64);

  EXPECT_LE(Ten, 10);
  EXPECT_GE(Ten, 10);
  EXPECT_EQ(Ten, 10);
  EXPECT_FALSE(Ten != 10);
  EXPECT_FALSE(Ten < 10);
  EXPECT_FALSE(Ten > 10);
  EXPECT_LT(Five, 10);
  EXPECT_GT(Ten, 5);

  EXPECT_LE(10, Ten);
  EXPECT_GE(10, Ten);
  EXPECT_EQ(10, Ten);
  EXPECT_FALSE(10 != Ten);
  EXPECT_FALSE(10 < Ten);
  EXPECT_FALSE(10 > Ten);
  EXPECT_LT(5, Ten);
  EXPECT_GT(10, Five);
}

TYPED_TEST(IntTest, overflows) {
  TypeParam X(1ll << 60);
  EXPECT_EQ((X * X - X * X * X * X) / (X * X * X), 1 - (1ll << 60));
  TypeParam Y(1ll << 62);
  EXPECT_EQ((Y + Y + Y + Y + Y + Y) / Y, 6);
  EXPECT_EQ(-(2 * (-Y)), 2 * Y); // -(-2^63) overflow.
  X *= X;
  EXPECT_EQ(X, (Y * Y) / 16);
  Y += Y;
  Y += Y;
  Y += Y;
  Y /= 8;
  EXPECT_EQ(Y, 1ll << 62);

  TypeParam Min(std::numeric_limits<int64_t>::min());
  TypeParam One(1);
  EXPECT_EQ(floorDiv(Min, -One), -Min);
  EXPECT_EQ(ceilDiv(Min, -One), -Min);
  EXPECT_EQ(abs(Min), -Min);

  TypeParam Z = Min;
  Z /= -1;
  EXPECT_EQ(Z, -Min);
  TypeParam W(Min);
  --W;
  EXPECT_EQ(W, TypeParam(Min) - 1);
  TypeParam U(Min);
  U -= 1;
  EXPECT_EQ(U, W);

  TypeParam Max(std::numeric_limits<int64_t>::max());
  TypeParam V = Max;
  ++V;
  EXPECT_EQ(V, Max + 1);
  TypeParam T = Max;
  T += 1;
  EXPECT_EQ(T, V);
}

TYPED_TEST(IntTest, floorCeilModAbsLcmGcd) {
  TypeParam X(1ll << 50), One(1), Two(2), Three(3);

  // Run on small values and large values.
  for (const TypeParam &Y : {X, X * X}) {
    EXPECT_EQ(floorDiv(3 * Y, Three), Y);
    EXPECT_EQ(ceilDiv(3 * Y, Three), Y);
    EXPECT_EQ(floorDiv(3 * Y - 1, Three), Y - 1);
    EXPECT_EQ(ceilDiv(3 * Y - 1, Three), Y);
    EXPECT_EQ(floorDiv(3 * Y - 2, Three), Y - 1);
    EXPECT_EQ(ceilDiv(3 * Y - 2, Three), Y);

    EXPECT_EQ(mod(3 * Y, Three), 0);
    EXPECT_EQ(mod(3 * Y + 1, Three), One);
    EXPECT_EQ(mod(3 * Y + 2, Three), Two);

    EXPECT_EQ(floorDiv(3 * Y, Y), 3);
    EXPECT_EQ(ceilDiv(3 * Y, Y), 3);
    EXPECT_EQ(floorDiv(3 * Y - 1, Y), 2);
    EXPECT_EQ(ceilDiv(3 * Y - 1, Y), 3);
    EXPECT_EQ(floorDiv(3 * Y - 2, Y), 2);
    EXPECT_EQ(ceilDiv(3 * Y - 2, Y), 3);

    EXPECT_EQ(mod(3 * Y, Y), 0);
    EXPECT_EQ(mod(3 * Y + 1, Y), 1);
    EXPECT_EQ(mod(3 * Y + 2, Y), 2);

    EXPECT_EQ(abs(Y), Y);
    EXPECT_EQ(abs(-Y), Y);

    EXPECT_EQ(gcd(3 * Y, Three), Three);
    EXPECT_EQ(lcm(Y, Three), 3 * Y);
    EXPECT_EQ(gcd(2 * Y, 3 * Y), Y);
    EXPECT_EQ(lcm(2 * Y, 3 * Y), 6 * Y);
    EXPECT_EQ(gcd(15 * Y, 6 * Y), 3 * Y);
    EXPECT_EQ(lcm(15 * Y, 6 * Y), 30 * Y);
  }
}
} // namespace
