//===- CallSPSCITest.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for the function-call SPS Controller Interface.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/sps-ci/CallSPSCI.h"
#include "orc-rt/SPSWrapperFunction.h"

#include "DirectCaller.h"
#include "gtest/gtest.h"

#include <optional>
#include <string>
#include <vector>

using namespace orc_rt;

namespace {

class CallSPSCITest : public ::testing::Test {
protected:
  void SetUp() override { cantFail(sps_ci::addCall(CI)); }

  DirectCaller caller(const char *Name) {
    return DirectCaller(nullptr, reinterpret_cast<orc_rt_WrapperFunction>(
                                     const_cast<void *>(CI.at(Name))));
  }

  SimpleSymbolTable CI;
};

TEST_F(CallSPSCITest, Registration) {
  EXPECT_TRUE(CI.count("orc_rt_ci_sps_call_void_void"));
  EXPECT_TRUE(CI.count("orc_rt_ci_sps_call_main"));
}

static int CallVoidVoidCount = 0;
static void callVoidVoidFn() { ++CallVoidVoidCount; }

TEST_F(CallSPSCITest, CallVoidVoid) {
  using SPSSig = void(SPSExecutorAddr);
  SPSWrapperFunction<SPSSig>::call(
      caller("orc_rt_ci_sps_call_void_void"),
      [](Error Err) { cantFail(std::move(Err)); },
      reinterpret_cast<void *>(callVoidVoidFn));
  EXPECT_EQ(CallVoidVoidCount, 1);
}

static int CallMainArgC = -1;
static std::vector<std::string> CallMainArgV;
static bool CallMainArgVIsNullTerminated = false;
static int callMainFn(int argc, char *argv[]) {
  CallMainArgC = argc;
  for (int I = 0; I < argc; ++I)
    CallMainArgV.push_back(argv[I]);
  CallMainArgVIsNullTerminated = (argv[argc] == nullptr);
  return 42;
}

TEST_F(CallSPSCITest, CallMain) {
  using SPSSig = int64_t(SPSExecutorAddr, SPSSequence<SPSString>);
  std::optional<Expected<int64_t>> Result;
  std::vector<std::string> Args = {"prog", "arg1", "arg2"};
  SPSWrapperFunction<SPSSig>::call(
      caller("orc_rt_ci_sps_call_main"),
      [&](Expected<int64_t> R) { Result = std::move(R); },
      reinterpret_cast<void *>(callMainFn), Args);

  ASSERT_TRUE(Result.has_value());
  ASSERT_TRUE(!!*Result) << toString(Result->takeError());
  EXPECT_EQ(**Result, 42);

  EXPECT_EQ(CallMainArgC, 3)
      << "argc should equal the number of program arguments, "
         "not including the null terminator";
  ASSERT_EQ(CallMainArgV.size(), 3U);
  EXPECT_EQ(CallMainArgV[0], "prog");
  EXPECT_EQ(CallMainArgV[1], "arg1");
  EXPECT_EQ(CallMainArgV[2], "arg2");
  EXPECT_TRUE(CallMainArgVIsNullTerminated)
      << "argv[argc] must be a null pointer per the C standard";
}

static int CallMainEmptyArgVArgC = -1;
static bool CallMainEmptyArgVIsNullTerminated = false;
static int callMainEmptyArgVFn(int argc, char *argv[]) {
  CallMainEmptyArgVArgC = argc;
  CallMainEmptyArgVIsNullTerminated = (argv[argc] == nullptr);
  return 42;
}

TEST_F(CallSPSCITest, CallMainEmptyArgV) {
  using SPSSig = int64_t(SPSExecutorAddr, SPSSequence<SPSString>);
  std::optional<Expected<int64_t>> Result;
  std::vector<std::string> Args;
  SPSWrapperFunction<SPSSig>::call(
      caller("orc_rt_ci_sps_call_main"),
      [&](Expected<int64_t> R) { Result = std::move(R); },
      reinterpret_cast<void *>(callMainEmptyArgVFn), Args);

  ASSERT_TRUE(Result.has_value());
  ASSERT_TRUE(!!*Result) << toString(Result->takeError());
  EXPECT_EQ(**Result, 42);
  EXPECT_EQ(CallMainEmptyArgVArgC, 0);
  EXPECT_TRUE(CallMainEmptyArgVIsNullTerminated);
}

} // namespace
