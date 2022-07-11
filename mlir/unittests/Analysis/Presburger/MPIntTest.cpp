//===- MPIntTest.cpp - Tests for MPInt ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/MPInt.h"
#include "mlir/Analysis/Presburger/SlowMPInt.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace mlir;
using namespace presburger;

// googletest boilerplate to run the same tests with both MPInt and SlowMPInt.
template <typename>
class IntTest : public testing::Test {};
using TypeList = testing::Types<MPInt, detail::SlowMPInt>;
// This is for pretty-printing the test name with the name of the class in use.
class TypeNames {
public:
  template <typename T>
  static std::string GetName(int) { // NOLINT; gtest mandates this name.
    if (std::is_same<T, MPInt>())
      return "MPInt";
    if (std::is_same<T, detail::SlowMPInt>())
      return "SlowMPInt";
    llvm_unreachable("Unknown class!");
  }
};
TYPED_TEST_SUITE(IntTest, TypeList, TypeNames);

TYPED_TEST(IntTest, ops) {
  TypeParam two(2), five(5), seven(7), ten(10);
  EXPECT_EQ(five + five, ten);
  EXPECT_EQ(five * five, 2 * ten + five);
  EXPECT_EQ(five * five, 3 * ten - five);
  EXPECT_EQ(five * two, ten);
  EXPECT_EQ(five / two, two);
  EXPECT_EQ(five % two, two / two);

  EXPECT_EQ(-ten % seven, -10 % 7);
  EXPECT_EQ(ten % -seven, 10 % -7);
  EXPECT_EQ(-ten % -seven, -10 % -7);
  EXPECT_EQ(ten % seven, 10 % 7);

  EXPECT_EQ(-ten / seven, -10 / 7);
  EXPECT_EQ(ten / -seven, 10 / -7);
  EXPECT_EQ(-ten / -seven, -10 / -7);
  EXPECT_EQ(ten / seven, 10 / 7);

  TypeParam x = ten;
  x += five;
  EXPECT_EQ(x, 15);
  x *= two;
  EXPECT_EQ(x, 30);
  x /= seven;
  EXPECT_EQ(x, 4);
  x -= two * 10;
  EXPECT_EQ(x, -16);
  x *= 2 * two;
  EXPECT_EQ(x, -64);
  x /= two / -2;
  EXPECT_EQ(x, 64);

  EXPECT_LE(ten, ten);
  EXPECT_GE(ten, ten);
  EXPECT_EQ(ten, ten);
  EXPECT_FALSE(ten != ten);
  EXPECT_FALSE(ten < ten);
  EXPECT_FALSE(ten > ten);
  EXPECT_LT(five, ten);
  EXPECT_GT(ten, five);
}

TYPED_TEST(IntTest, ops64Overloads) {
  TypeParam two(2), five(5), seven(7), ten(10);
  EXPECT_EQ(five + 5, ten);
  EXPECT_EQ(five + 5, 5 + five);
  EXPECT_EQ(five * 5, 2 * ten + 5);
  EXPECT_EQ(five * 5, 3 * ten - 5);
  EXPECT_EQ(five * two, ten);
  EXPECT_EQ(5 / two, 2);
  EXPECT_EQ(five / 2, 2);
  EXPECT_EQ(2 % two, 0);
  EXPECT_EQ(2 - two, 0);
  EXPECT_EQ(2 % two, two % 2);

  TypeParam x = ten;
  x += 5;
  EXPECT_EQ(x, 15);
  x *= 2;
  EXPECT_EQ(x, 30);
  x /= 7;
  EXPECT_EQ(x, 4);
  x -= 20;
  EXPECT_EQ(x, -16);
  x *= 4;
  EXPECT_EQ(x, -64);
  x /= -1;
  EXPECT_EQ(x, 64);

  EXPECT_LE(ten, 10);
  EXPECT_GE(ten, 10);
  EXPECT_EQ(ten, 10);
  EXPECT_FALSE(ten != 10);
  EXPECT_FALSE(ten < 10);
  EXPECT_FALSE(ten > 10);
  EXPECT_LT(five, 10);
  EXPECT_GT(ten, 5);

  EXPECT_LE(10, ten);
  EXPECT_GE(10, ten);
  EXPECT_EQ(10, ten);
  EXPECT_FALSE(10 != ten);
  EXPECT_FALSE(10 < ten);
  EXPECT_FALSE(10 > ten);
  EXPECT_LT(5, ten);
  EXPECT_GT(10, five);
}

TYPED_TEST(IntTest, overflows) {
  TypeParam x(1ll << 60);
  EXPECT_EQ((x * x - x * x * x * x) / (x * x * x), 1 - (1ll << 60));
  TypeParam y(1ll << 62);
  EXPECT_EQ((y + y + y + y + y + y) / y, 6);
  EXPECT_EQ(-(2 * (-y)), 2 * y); // -(-2^63) overflow.
  x *= x;
  EXPECT_EQ(x, (y * y) / 16);
  y += y;
  y += y;
  y += y;
  y /= 8;
  EXPECT_EQ(y, 1ll << 62);
  int64_t min = std::numeric_limits<int64_t>::min();
  int64_t max = std::numeric_limits<int64_t>::max();
  TypeParam z(min);
  z /= -1;
  EXPECT_EQ(z, -TypeParam(min));
  TypeParam w(min);
  --w;
  EXPECT_EQ(w, TypeParam(min) - 1);
  TypeParam u(min);
  u -= 1;
  EXPECT_EQ(u, w);
  TypeParam v(max);
  ++v;
  EXPECT_EQ(v, TypeParam(max) + 1);
  TypeParam t(max);
  t += 1;
  EXPECT_EQ(t, v);
}
