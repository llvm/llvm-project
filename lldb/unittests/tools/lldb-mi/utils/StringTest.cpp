//===-- StringTest.cpp ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MIUtilString.h"
#include "gtest/gtest.h"

TEST(StringTest, ConstructFromNullptr) {
  auto str = CMIUtilString(nullptr);
  EXPECT_TRUE(str.empty());
  EXPECT_NE(nullptr, str.c_str());
  EXPECT_EQ(CMIUtilString(""), str);
}

TEST(StringTest, AssignNullptr) {
  CMIUtilString str;
  str = nullptr;
  EXPECT_TRUE(str.empty());
  EXPECT_NE(nullptr, str.c_str());
  EXPECT_EQ(CMIUtilString(""), str);
}

TEST(StringTest, IsAllValidAlphaAndNumeric) {
  EXPECT_TRUE(CMIUtilString::IsAllValidAlphaAndNumeric("123abc"));
  EXPECT_FALSE(CMIUtilString::IsAllValidAlphaAndNumeric(""));
  EXPECT_FALSE(CMIUtilString::IsAllValidAlphaAndNumeric(nullptr));
}

