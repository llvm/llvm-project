//===----- WrapperFunctionUtilsTest.cpp - Test Wrapper-Function utils -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/Shared/WrapperFunctionUtils.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::orc::shared;

namespace {
constexpr const char *TestString = "test string";
} // end anonymous namespace

TEST(WrapperFunctionUtilsTest, DefaultWrapperFunctionResult) {
  WrapperFunctionResult R;
  EXPECT_TRUE(R.empty());
  EXPECT_EQ(R.size(), 0U);
  EXPECT_EQ(R.getOutOfBandError(), nullptr);
}

TEST(WrapperFunctionUtilsTest, WrapperFunctionResultFromRange) {
  auto R = WrapperFunctionResult::copyFrom(TestString, strlen(TestString) + 1);
  EXPECT_EQ(R.size(), strlen(TestString) + 1);
  EXPECT_TRUE(strcmp(R.data(), TestString) == 0);
  EXPECT_FALSE(R.empty());
  EXPECT_EQ(R.getOutOfBandError(), nullptr);
}

TEST(WrapperFunctionUtilsTest, WrapperFunctionResultFromCString) {
  auto R = WrapperFunctionResult::copyFrom(TestString);
  EXPECT_EQ(R.size(), strlen(TestString) + 1);
  EXPECT_TRUE(strcmp(R.data(), TestString) == 0);
  EXPECT_FALSE(R.empty());
  EXPECT_EQ(R.getOutOfBandError(), nullptr);
}

TEST(WrapperFunctionUtilsTest, WrapperFunctionResultFromStdString) {
  auto R = WrapperFunctionResult::copyFrom(std::string(TestString));
  EXPECT_EQ(R.size(), strlen(TestString) + 1);
  EXPECT_TRUE(strcmp(R.data(), TestString) == 0);
  EXPECT_FALSE(R.empty());
  EXPECT_EQ(R.getOutOfBandError(), nullptr);
}

TEST(WrapperFunctionUtilsTest, WrapperFunctionResultFromOutOfBandError) {
  auto R = WrapperFunctionResult::createOutOfBandError(TestString);
  EXPECT_FALSE(R.empty());
  EXPECT_TRUE(strcmp(R.getOutOfBandError(), TestString) == 0);
}

static void voidNoop() {}

static WrapperFunctionResult voidNoopWrapper(const char *ArgData,
                                             size_t ArgSize) {
  return WrapperFunction<void()>::handle(ArgData, ArgSize, voidNoop);
}

static WrapperFunctionResult addWrapper(const char *ArgData, size_t ArgSize) {
  return WrapperFunction<int32_t(int32_t, int32_t)>::handle(
      ArgData, ArgSize, [](int32_t X, int32_t Y) -> int32_t { return X + Y; });
}

TEST(WrapperFunctionUtilsTest, WrapperFunctionCallVoidNoopAndHandle) {
  EXPECT_FALSE(!!WrapperFunction<void()>::call(voidNoopWrapper));
}

TEST(WrapperFunctionUtilsTest, WrapperFunctionCallAndHandle) {
  int32_t Result;
  EXPECT_FALSE(!!WrapperFunction<int32_t(int32_t, int32_t)>::call(
      addWrapper, Result, 1, 2));
  EXPECT_EQ(Result, (int32_t)3);
}
