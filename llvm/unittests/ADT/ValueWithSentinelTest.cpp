//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ValueWithSentinel.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(ValueWithSentinelTest, Basic) {
  ValueWithSentinelNumericMax<int> Value;
  EXPECT_FALSE(Value.has_value());

  Value = 1000;
  EXPECT_TRUE(Value.has_value());

  EXPECT_EQ(Value, 1000);
  EXPECT_EQ(Value.value(), 1000);

  Value.clear();
  EXPECT_FALSE(Value.has_value());

  ValueWithSentinelNumericMax<int> OtherValue(99);
  EXPECT_TRUE(OtherValue.has_value());
  EXPECT_NE(Value, OtherValue);

  Value = OtherValue;
  EXPECT_EQ(Value, OtherValue);
}

} // end anonymous namespace
