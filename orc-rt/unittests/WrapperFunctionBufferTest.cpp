//===-- WrapperFunctionBufferTest.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Test WrapperFunctionBuffer APIs.
//
//===----------------------------------------------------------------------===//

#include "orc-rt-c/WrapperFunction.h"
#include "orc-rt/WrapperFunction.h"
#include "gtest/gtest.h"

using namespace orc_rt;

namespace {
constexpr const char *TestString = "test string";
} // end anonymous namespace

TEST(WrapperFunctionUtilsTest, DefaultWrapperFunctionBuffer) {
  WrapperFunctionBuffer B;
  EXPECT_TRUE(B.empty());
  EXPECT_EQ(B.size(), 0U);
  EXPECT_EQ(B.getOutOfBandError(), nullptr);
}

TEST(WrapperFunctionUtilsTest, WrapperFunctionBufferFromCStruct) {
  orc_rt_WrapperFunctionBuffer CB =
      orc_rt_CreateWrapperFunctionBufferFromString(TestString);
  WrapperFunctionBuffer B(CB);
  EXPECT_EQ(B.size(), strlen(TestString) + 1);
  EXPECT_TRUE(strcmp(B.data(), TestString) == 0);
  EXPECT_FALSE(B.empty());
  EXPECT_EQ(B.getOutOfBandError(), nullptr);
}

TEST(WrapperFunctionUtilsTest, WrapperFunctionBufferFromRange) {
  auto B = WrapperFunctionBuffer::copyFrom(TestString, strlen(TestString) + 1);
  EXPECT_EQ(B.size(), strlen(TestString) + 1);
  EXPECT_TRUE(strcmp(B.data(), TestString) == 0);
  EXPECT_FALSE(B.empty());
  EXPECT_EQ(B.getOutOfBandError(), nullptr);
}

TEST(WrapperFunctionUtilsTest, WrapperFunctionBufferFromCString) {
  auto B = WrapperFunctionBuffer::copyFrom(TestString);
  EXPECT_EQ(B.size(), strlen(TestString) + 1);
  EXPECT_TRUE(strcmp(B.data(), TestString) == 0);
  EXPECT_FALSE(B.empty());
  EXPECT_EQ(B.getOutOfBandError(), nullptr);
}

TEST(WrapperFunctionUtilsTest, WrapperFunctionBufferFromOutOfBandError) {
  auto B = WrapperFunctionBuffer::createOutOfBandError(TestString);
  EXPECT_FALSE(B.empty());
  EXPECT_TRUE(strcmp(B.getOutOfBandError(), TestString) == 0);
}
