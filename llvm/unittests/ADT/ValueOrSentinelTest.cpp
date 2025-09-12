//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ValueOrSentinel.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(ValueOrSentinelTest, Basic) {
  // Default constructor should equal sentinel.
  ValueOrSentinelIntMax<int> Value;
  EXPECT_FALSE(Value.has_value());
  EXPECT_FALSE(bool(Value));

  // Assignment operator.
  Value = 1000;
  EXPECT_TRUE(Value.has_value());

  // .value(), operator*, implicit constructor, explicit conversion
  EXPECT_EQ(Value, 1000);
  EXPECT_EQ(Value.value(), 1000);
  EXPECT_EQ(*Value, 1000);
  EXPECT_EQ(int(Value), 1000);

  // .clear() should set value to sentinel
  Value.clear();
  EXPECT_FALSE(Value.has_value());
  EXPECT_FALSE(bool(Value));

  // construction from value, comparison operators
  ValueOrSentinelIntMax<int> OtherValue(99);
  EXPECT_TRUE(OtherValue.has_value());
  EXPECT_TRUE(bool(OtherValue));
  EXPECT_EQ(OtherValue, 99);
  EXPECT_NE(Value, OtherValue);

  Value = OtherValue;
  EXPECT_EQ(Value, OtherValue);
}

TEST(ValueOrSentinelTest, PointerType) {
  ValueOrSentinel<int *, nullptr> Value;
  EXPECT_FALSE(Value.has_value());

  int A = 10;
  Value = &A;
  EXPECT_TRUE(Value.has_value());

  EXPECT_EQ(*Value.value(), 10);

  Value.clear();
  EXPECT_FALSE(Value.has_value());
}

} // end anonymous namespace
