//===----- WrapperFunctionUtilsTest.cpp - Test Wrapper-Function utils -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/Shared/WrapperFunctionUtils.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

#include <future>

using namespace llvm;
using namespace llvm::orc;
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

class AddClass {
public:
  AddClass(int32_t X) : X(X) {}
  int32_t addMethod(int32_t Y) { return X + Y; }
private:
  int32_t X;
};

static WrapperFunctionResult voidNoopWrapper(const char *ArgData,
                                             size_t ArgSize) {
  return WrapperFunction<void()>::handle(ArgData, ArgSize, voidNoop);
}

static WrapperFunctionResult addWrapper(const char *ArgData, size_t ArgSize) {
  return WrapperFunction<int32_t(int32_t, int32_t)>::handle(
      ArgData, ArgSize, [](int32_t X, int32_t Y) -> int32_t { return X + Y; });
}

static WrapperFunctionResult addMethodWrapper(const char *ArgData,
                                              size_t ArgSize) {
  return WrapperFunction<int32_t(SPSExecutorAddr, int32_t)>::handle(
      ArgData, ArgSize, makeMethodWrapperHandler(&AddClass::addMethod));
}

TEST(WrapperFunctionUtilsTest, WrapperFunctionCallAndHandleVoid) {
  EXPECT_FALSE(!!WrapperFunction<void()>::call(voidNoopWrapper));
}

TEST(WrapperFunctionUtilsTest, WrapperFunctionCallAndHandleRet) {
  int32_t Result;
  EXPECT_FALSE(!!WrapperFunction<int32_t(int32_t, int32_t)>::call(
      addWrapper, Result, 1, 2));
  EXPECT_EQ(Result, (int32_t)3);
}

TEST(WrapperFunctionUtilsTest, WrapperFunctionMethodCallAndHandleRet) {
  int32_t Result;
  AddClass AddObj(1);
  EXPECT_FALSE(!!WrapperFunction<int32_t(SPSExecutorAddr, int32_t)>::call(
      addMethodWrapper, Result, ExecutorAddr::fromPtr(&AddObj), 2));
  EXPECT_EQ(Result, (int32_t)3);
}

static void voidNoopAsync(unique_function<void(SPSEmpty)> SendResult) {
  SendResult(SPSEmpty());
}

static WrapperFunctionResult voidNoopAsyncWrapper(const char *ArgData,
                                                  size_t ArgSize) {
  std::promise<WrapperFunctionResult> RP;
  auto RF = RP.get_future();

  WrapperFunction<void()>::handleAsync(
      ArgData, ArgSize, voidNoopAsync,
      [&](WrapperFunctionResult R) { RP.set_value(std::move(R)); });

  return RF.get();
}

static WrapperFunctionResult addAsyncWrapper(const char *ArgData,
                                             size_t ArgSize) {
  std::promise<WrapperFunctionResult> RP;
  auto RF = RP.get_future();

  WrapperFunction<int32_t(int32_t, int32_t)>::handleAsync(
      ArgData, ArgSize,
      [](unique_function<void(int32_t)> SendResult, int32_t X, int32_t Y) {
        SendResult(X + Y);
      },
      [&](WrapperFunctionResult R) { RP.set_value(std::move(R)); });
  return RF.get();
}

TEST(WrapperFunctionUtilsTest, WrapperFunctionCallAndHandleAsyncVoid) {
  EXPECT_FALSE(!!WrapperFunction<void()>::call(voidNoopAsyncWrapper));
}

TEST(WrapperFunctionUtilsTest, WrapperFunctionCallAndHandleAsyncRet) {
  int32_t Result;
  EXPECT_FALSE(!!WrapperFunction<int32_t(int32_t, int32_t)>::call(
      addAsyncWrapper, Result, 1, 2));
  EXPECT_EQ(Result, (int32_t)3);
}

static WrapperFunctionResult failingWrapper(const char *ArgData,
                                            size_t ArgSize) {
  return WrapperFunctionResult::createOutOfBandError("failed");
}

void asyncFailingWrapperCaller(unique_function<void(WrapperFunctionResult)> F,
                               const char *ArgData, size_t ArgSize) {
  F(failingWrapper(ArgData, ArgSize));
}

TEST(WrapperFunctionUtilsTest, WrapperFunctionCallFailingAsync) {
  WrapperFunction<void()>::callAsync(asyncFailingWrapperCaller, [](Error Err) {
    EXPECT_THAT_ERROR(std::move(Err), Failed());
  });
}
