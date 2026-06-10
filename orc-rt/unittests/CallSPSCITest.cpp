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

static int CallMainArgc = -1;
static std::vector<std::string> CallMainArgv;
static bool CallMainArgvIsNullTerminated = false;
static int callMainFn(int Argc, char *Argv[]) {
  CallMainArgc = Argc;
  for (int I = 0; I < Argc; ++I)
    CallMainArgv.emplace_back(Argv[I]);
  // Per the C standard, argv[argc] shall be a null pointer.
  CallMainArgvIsNullTerminated = (Argv[Argc] == nullptr);
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

  EXPECT_EQ(CallMainArgc, 3)
      << "argc should equal the number of program arguments, "
         "not including the null terminator";
  ASSERT_EQ(CallMainArgv.size(), 3U);
  EXPECT_EQ(CallMainArgv[0], "prog");
  EXPECT_EQ(CallMainArgv[1], "arg1");
  EXPECT_EQ(CallMainArgv[2], "arg2");
  EXPECT_TRUE(CallMainArgvIsNullTerminated)
      << "argv[argc] must be a null pointer per the C standard";
}

static int CallMainEmptyArgvArgc = -1;
static bool CallMainEmptyArgvIsNullTerminated = false;
static int callMainEmptyArgvFn(int Argc, char *Argv[]) {
  CallMainEmptyArgvArgc = Argc;
  CallMainEmptyArgvIsNullTerminated = (Argv[Argc] == nullptr);
  return 42;
}

TEST_F(CallSPSCITest, CallMainEmptyArgv) {
  using SPSSig = int64_t(SPSExecutorAddr, SPSSequence<SPSString>);
  std::optional<Expected<int64_t>> Result;
  std::vector<std::string> Args;
  SPSWrapperFunction<SPSSig>::call(
      caller("orc_rt_ci_sps_call_main"),
      [&](Expected<int64_t> R) { Result = std::move(R); },
      reinterpret_cast<void *>(callMainEmptyArgvFn), Args);

  ASSERT_TRUE(Result.has_value());
  ASSERT_TRUE(!!*Result) << toString(Result->takeError());
  EXPECT_EQ(**Result, 42);
  EXPECT_EQ(CallMainEmptyArgvArgc, 0);
  EXPECT_TRUE(CallMainEmptyArgvIsNullTerminated);
}

} // namespace
