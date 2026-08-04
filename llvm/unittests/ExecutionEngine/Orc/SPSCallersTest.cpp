//===- SPSCallersTest.cpp - Test SPS call wrappers ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/AbsoluteSymbols.h"
#include "llvm/ExecutionEngine/Orc/RTBridge/SPS/Calls.h"
#include "llvm/ExecutionEngine/Orc/SelfExecutorProcessControl.h"
#include "llvm/ExecutionEngine/Orc/Shared/WrapperFunctionUtils.h"
#include "llvm/Support/MSVCErrorWorkarounds.h"
#include "llvm/Testing/Support/Error.h"

#include <cstring>
#include <future>
#include <string>
#include <vector>

#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::orc;
using namespace llvm::orc::shared;
using llvm::orc::rt::sps::MainCaller;
// Test "main" function. Returns argc plus the length of the first element of
// argv (if argv is non-empty). Does not inspect argv entries beyond the first.
static int testMain(int argc, char *argv[]) {
  int Result = argc;
  if (argc > 0)
    Result += static_cast<int>(std::strlen(argv[0]));
  return Result;
}

// Executor-side "call-main" wrapper. Decodes the main-function address and the
// argument vector, then invokes the main function with a C-style (argc, argv).
static CWrapperFunctionBuffer callMainWrapper(const char *ArgData,
                                              size_t ArgSize) {
  return WrapperFunction<int64_t(SPSExecutorAddr, SPSSequence<SPSString>)>::
      handle(ArgData, ArgSize,
             [](ExecutorAddr MainFnAddr,
                std::vector<std::string> Args) -> int64_t {
               std::vector<char *> ArgV;
               ArgV.reserve(Args.size() + 1);
               for (auto &Arg : Args)
                 ArgV.push_back(Arg.data());
               ArgV.push_back(nullptr);
               auto *Main = MainFnAddr.toPtr<int(int, char **)>();
               return Main(static_cast<int>(Args.size()), ArgV.data());
             })
          .release();
}

TEST(SPSCallersTest, CallMainSyncViaDirectConstruction) {
  ExecutionSession ES(cantFail(SelfExecutorProcessControl::Create()));

  MainCaller CallMain(ES, ExecutorAddr::fromPtr(callMainWrapper));
  ExecutorAddr MainAddr = ExecutorAddr::fromPtr(testMain);

  std::vector<std::string> Args;
  Expected<int64_t> R0 = CallMain(MainAddr, Args);
  ASSERT_THAT_EXPECTED(R0, Succeeded());
  EXPECT_EQ(*R0, 0); // argc == 0, no argv[0].

  Args = {"hello"};
  Expected<int64_t> R1 = CallMain(MainAddr, Args);
  ASSERT_THAT_EXPECTED(R1, Succeeded());
  EXPECT_EQ(*R1, 1 + 5); // argc == 1, strlen("hello") == 5.

  Args = {"a", "bb"};
  Expected<int64_t> R2 = CallMain(MainAddr, Args);
  ASSERT_THAT_EXPECTED(R2, Succeeded());
  EXPECT_EQ(*R2, 2 + 1); // argc == 2, strlen("a") == 1.

  cantFail(ES.endSession());
}

TEST(SPSCallersTest, CallMainAsyncViaCallOperator) {
  ExecutionSession ES(cantFail(SelfExecutorProcessControl::Create()));

  MainCaller CallMain(ES, ExecutorAddr::fromPtr(callMainWrapper));

  std::vector<std::string> Args = {"foo", "bar"};
  std::promise<MSVCPExpected<int64_t>> P;
  auto F = P.get_future();
  CallMain([&](Expected<int64_t> R) { P.set_value(std::move(R)); },
           ExecutorAddr::fromPtr(testMain), Args);

  Expected<int64_t> R = F.get();
  ASSERT_THAT_EXPECTED(R, Succeeded());
  EXPECT_EQ(*R, 2 + 3); // argc == 2, strlen("foo") == 3.

  cantFail(ES.endSession());
}

TEST(SPSCallersTest, CallMainThroughRTInterface) {
  ExecutionSession ES(cantFail(SelfExecutorProcessControl::Create()));

  MainCaller CallMain(ES, ExecutorAddr::fromPtr(callMainWrapper));

  // Drive the caller through the runtime-agnostic rt::MainCaller interface to
  // exercise the virtual dispatch path (and to guard the interface's public
  // accessibility).
  rt::MainCaller &Base = CallMain;
  ExecutorAddr MainAddr = ExecutorAddr::fromPtr(testMain);

  // Synchronous call operator (inherited from rt::MainCaller).
  std::vector<std::string> Args = {"hello"};
  Expected<int64_t> RSync = Base(MainAddr, Args);
  ASSERT_THAT_EXPECTED(RSync, Succeeded());
  EXPECT_EQ(*RSync, 1 + 5); // argc == 1, strlen("hello") == 5.

  // Asynchronous call operator (virtual).
  std::promise<MSVCPExpected<int64_t>> P;
  auto F = P.get_future();
  Base([&](Expected<int64_t> R) { P.set_value(std::move(R)); }, MainAddr, Args);
  Expected<int64_t> RAsync = F.get();
  ASSERT_THAT_EXPECTED(RAsync, Succeeded());
  EXPECT_EQ(*RAsync, 1 + 5);

  cantFail(ES.endSession());
}

TEST(SPSCallersTest, CreateLooksUpCallMainInBootstrapJD) {
  ExecutionSession ES(cantFail(SelfExecutorProcessControl::Create()));

  // Register the call-main wrapper in the bootstrap JITDylib under the name
  // MainCaller::Create looks for.
  auto &BootstrapJD = ES.getBootstrapJITDylib();
  cantFail(BootstrapJD.define(absoluteSymbols(
      {{ES.intern(MainCaller::CIName),
        {ExecutorAddr::fromPtr(callMainWrapper), JITSymbolFlags::Exported}}})));

  Expected<MainCaller> CallMain = MainCaller::Create(ES);
  ASSERT_THAT_EXPECTED(CallMain, Succeeded());

  std::vector<std::string> Args = {"x", "y", "z"};
  Expected<int64_t> R = (*CallMain)(ExecutorAddr::fromPtr(testMain), Args);
  ASSERT_THAT_EXPECTED(R, Succeeded());
  EXPECT_EQ(*R, 3 + 1); // argc == 3, strlen("x") == 1.

  cantFail(ES.endSession());
}
