//===-- SBStructuredDataTest.cpp ------------------------===----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===/

#include "gtest/gtest.h"

#include "lldb/API/SBStringList.h"
#include "lldb/API/SBStructuredData.h"

#include <cstring>
#include <string>

using namespace lldb;

class SBStructuredDataTest : public testing::Test {};

TEST_F(SBStructuredDataTest, NullImpl) {
  SBStructuredData data(nullptr);
  EXPECT_EQ(data.GetType(), eStructuredDataTypeInvalid);
  EXPECT_EQ(data.GetSize(), 0ul);
  SBStringList keys;
  EXPECT_FALSE(data.GetKeys(keys));
  EXPECT_EQ(data.GetValueForKey("key").GetType(), eStructuredDataTypeInvalid);
  EXPECT_EQ(data.GetItemAtIndex(0).GetType(), eStructuredDataTypeInvalid);
  EXPECT_EQ(data.GetIntegerValue(UINT64_MAX), UINT64_MAX);
  EXPECT_EQ(data.GetFloatValue(DBL_MAX), DBL_MAX);
  EXPECT_TRUE(data.GetBooleanValue(true));
  EXPECT_FALSE(data.GetBooleanValue(false));
  char dst[1];
  EXPECT_EQ(data.GetStringValue(dst, sizeof(dst)), 0ul);
}
