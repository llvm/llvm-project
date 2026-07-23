//===----------- CallSPSViaEPC.cpp - Test CallSPSViaEPC.h APIs ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/CallSPSViaEPC.h"
#include "llvm/ExecutionEngine/Orc/SelfExecutorProcessControl.h"

#include "llvm/Testing/Support/Error.h"

#include <future>

#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::orc;
using namespace llvm::orc::shared;

static CWrapperFunctionBuffer voidWrapper(const char *ArgData, size_t ArgSize) {
  return WrapperFunction<void()>::handle(ArgData, ArgSize, []() {}).release();
}

static CWrapperFunctionBuffer mainWrapper(const char *ArgData, size_t ArgSize) {
  return WrapperFunction<int32_t(SPSSequence<SPSString>)>::handle(
             ArgData, ArgSize,
             [](std::vector<std::string> Args) -> int32_t {
               return Args.size();
             })
      .release();
}

TEST(CallSPSViaEPCTest, CallVoidViaCallerAsync) {
  auto EPC = cantFail(SelfExecutorProcessControl::Create());
  SPSEPCCaller<void()> C(*EPC);

  Error Err = Error::success();
  {
    ErrorAsOutParameter _(Err);
    C([&](Error E) { Err = std::move(E); },
      ExecutorSymbolDef::fromPtr(voidWrapper));
  }
  EXPECT_THAT_ERROR(std::move(Err), Succeeded());
}

TEST(CallSPSViaEPCTest, CallVoidViaCallerSync) {
  auto EPC = cantFail(SelfExecutorProcessControl::Create());
  SPSEPCCaller<void()> C(*EPC);

  Error Err =
      C(std::promise<MSVCPError>(), ExecutorSymbolDef::fromPtr(voidWrapper));
  EXPECT_THAT_ERROR(std::move(Err), Succeeded());
}

TEST(CallSPSViaEPCTest, CallMainViaCallerAsync) {
  auto EPC = cantFail(SelfExecutorProcessControl::Create());
  SPSEPCCaller<int32_t(SPSSequence<SPSString>)> C(*EPC);
  std::vector<std::string> Args;

  std::optional<Expected<int32_t>> R1;
  C([&](Expected<int32_t> R) { R1 = std::move(R); },
    ExecutorSymbolDef::fromPtr(mainWrapper), Args);
  ASSERT_THAT_EXPECTED(*R1, Succeeded());
  EXPECT_EQ(**R1, 0);

  Args.push_back("foo");
  std::optional<Expected<int32_t>> R2;
  C([&](Expected<int32_t> R) { R2 = std::move(R); },
    ExecutorSymbolDef::fromPtr(mainWrapper), Args);
  ASSERT_THAT_EXPECTED(*R2, Succeeded());
  EXPECT_EQ(**R2, 1);

  Args.push_back("foo");
  std::optional<Expected<int32_t>> R3;
  C([&](Expected<int32_t> R) { R3 = std::move(R); },
    ExecutorSymbolDef::fromPtr(mainWrapper), Args);
  ASSERT_THAT_EXPECTED(*R3, Succeeded());
  EXPECT_EQ(**R3, 2);

  Args.clear();
  std::optional<Expected<int32_t>> R4;
  C([&](Expected<int32_t> R) { R4 = std::move(R); },
    ExecutorSymbolDef::fromPtr(mainWrapper), Args);
  ASSERT_THAT_EXPECTED(*R4, Succeeded());
  EXPECT_EQ(**R4, 0);
}

TEST(CallSPSViaEPCTest, CallMainViaGenericCallAsync) {
  auto EPC = cantFail(SelfExecutorProcessControl::Create());
  SPSEPCCall<int32_t(SPSSequence<SPSString>)> C(
      *EPC, ExecutorSymbolDef::fromPtr(mainWrapper));
  std::vector<std::string> Args;

  std::optional<Expected<int32_t>> R1;
  C([&](Expected<int32_t> R) { R1 = std::move(R); }, Args);
  ASSERT_THAT_EXPECTED(*R1, Succeeded());
  EXPECT_EQ(**R1, 0);

  Args.push_back("foo");
  std::optional<Expected<int32_t>> R2;
  C([&](Expected<int32_t> R) { R2 = std::move(R); }, Args);
  ASSERT_THAT_EXPECTED(*R2, Succeeded());
  EXPECT_EQ(**R2, 1);

  Args.push_back("foo");
  std::optional<Expected<int32_t>> R3;
  C([&](Expected<int32_t> R) { R3 = std::move(R); }, Args);
  ASSERT_THAT_EXPECTED(*R3, Succeeded());
  EXPECT_EQ(**R3, 2);

  Args.clear();
  std::optional<Expected<int32_t>> R4;
  C([&](Expected<int32_t> R) { R4 = std::move(R); }, Args);
  ASSERT_THAT_EXPECTED(*R4, Succeeded());
  EXPECT_EQ(**R4, 0);
}

TEST(CallSPSViaEPCTest, CallMainViaCallerSync) {
  auto EPC = cantFail(SelfExecutorProcessControl::Create());
  SPSEPCCaller<int32_t(SPSSequence<SPSString>)> C(*EPC);
  std::vector<std::string> Args;

  Expected<int32_t> R1 = C(std::promise<MSVCPExpected<int32_t>>(),
                           ExecutorSymbolDef::fromPtr(mainWrapper), Args);
  ASSERT_THAT_EXPECTED(R1, Succeeded());
  EXPECT_EQ(*R1, 0);

  Args.push_back("foo");
  Expected<int32_t> R2 = C(std::promise<MSVCPExpected<int32_t>>(),
                           ExecutorSymbolDef::fromPtr(mainWrapper), Args);
  ASSERT_THAT_EXPECTED(R2, Succeeded());
  EXPECT_EQ(*R2, 1);

  Args.push_back("foo");
  Expected<int32_t> R3 = C(std::promise<MSVCPExpected<int32_t>>(),
                           ExecutorSymbolDef::fromPtr(mainWrapper), Args);
  ASSERT_THAT_EXPECTED(R3, Succeeded());
  EXPECT_EQ(*R3, 2);

  Args.clear();
  Expected<int32_t> R4 = C(std::promise<MSVCPExpected<int32_t>>(),
                           ExecutorSymbolDef::fromPtr(mainWrapper), Args);
  ASSERT_THAT_EXPECTED(R4, Succeeded());
  EXPECT_EQ(*R4, 0);
}

TEST(CallSPSViaEPCTest, CallMainViaGenericCallSync) {
  auto EPC = cantFail(SelfExecutorProcessControl::Create());
  SPSEPCCall<int32_t(SPSSequence<SPSString>)> C(
      *EPC, ExecutorSymbolDef::fromPtr(mainWrapper));
  std::vector<std::string> Args;

  Expected<int32_t> R1 = C(std::promise<MSVCPExpected<int32_t>>(), Args);
  ASSERT_THAT_EXPECTED(R1, Succeeded());
  EXPECT_EQ(*R1, 0);

  Args.push_back("foo");
  Expected<int32_t> R2 = C(std::promise<MSVCPExpected<int32_t>>(), Args);
  ASSERT_THAT_EXPECTED(R2, Succeeded());
  EXPECT_EQ(*R2, 1);

  Args.push_back("foo");
  Expected<int32_t> R3 = C(std::promise<MSVCPExpected<int32_t>>(), Args);
  ASSERT_THAT_EXPECTED(R3, Succeeded());
  EXPECT_EQ(*R3, 2);

  Args.clear();
  Expected<int32_t> R4 = C(std::promise<MSVCPExpected<int32_t>>(), Args);
  ASSERT_THAT_EXPECTED(R4, Succeeded());
  EXPECT_EQ(*R4, 0);
}
