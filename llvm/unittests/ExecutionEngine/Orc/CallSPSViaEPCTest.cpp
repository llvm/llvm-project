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

#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::orc;
using namespace llvm::orc::shared;

static CWrapperFunctionResult mainWrapper(const char *ArgData, size_t ArgSize) {
  return WrapperFunction<int32_t(SPSSequence<SPSString>)>::handle(
             ArgData, ArgSize,
             [](std::vector<std::string> Args) -> int32_t {
               return Args.size();
             })
      .release();
}

TEST(CallSPSViaEPCTest, CallMainViaCaller) {
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

TEST(CallSPSViaEPCTest, CallMainViaGenericCall) {
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
