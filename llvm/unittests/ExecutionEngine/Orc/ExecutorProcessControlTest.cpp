//===- ExecutorProcessControlTest.cpp - Test ExecutorProcessControl utils -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/ExecutorProcessControl.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/Support/MSVCErrorWorkarounds.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

#include <future>

using namespace llvm;
using namespace llvm::orc;
using namespace llvm::orc::shared;

static llvm::orc::shared::detail::CWrapperFunctionResult
addWrapper(const char *ArgData, size_t ArgSize) {
  return WrapperFunction<int32_t(int32_t, int32_t)>::handle(
             ArgData, ArgSize, [](int32_t X, int32_t Y) { return X + Y; })
      .release();
}

static void addAsyncWrapper(unique_function<void(int32_t)> SendResult,
                            int32_t X, int32_t Y) {
  SendResult(X + Y);
}

TEST(ExecutorProcessControl, RunWrapperTemplate) {
  auto EPC = cantFail(
      SelfExecutorProcessControl::Create(std::make_shared<SymbolStringPool>()));

  int32_t Result;
  EXPECT_THAT_ERROR(EPC->runSPSWrapper<int32_t(int32_t, int32_t)>(
                        pointerToJITTargetAddress(addWrapper), Result, 2, 3),
                    Succeeded());
  EXPECT_EQ(Result, 5);
}

TEST(ExecutorProcessControl, RunWrapperAsyncTemplate) {
  auto EPC = cantFail(
      SelfExecutorProcessControl::Create(std::make_shared<SymbolStringPool>()));

  std::promise<MSVCPExpected<int32_t>> RP;
  using Sig = int32_t(int32_t, int32_t);
  EPC->runSPSWrapperAsync<Sig>(
      [&](Error SerializationErr, int32_t R) {
        if (SerializationErr)
          RP.set_value(std::move(SerializationErr));
        RP.set_value(std::move(R));
      },
      pointerToJITTargetAddress(addWrapper), 2, 3);
  Expected<int32_t> Result = RP.get_future().get();
  EXPECT_THAT_EXPECTED(Result, HasValue(5));
}

TEST(ExecutorProcessControl, RegisterAsyncHandlerAndRun) {

  constexpr JITTargetAddress AddAsyncTagAddr = 0x01;

  auto EPC = cantFail(
      SelfExecutorProcessControl::Create(std::make_shared<SymbolStringPool>()));
  ExecutionSession ES(EPC->getSymbolStringPool());
  auto &JD = ES.createBareJITDylib("JD");

  auto AddAsyncTag = ES.intern("addAsync_tag");
  cantFail(JD.define(absoluteSymbols(
      {{AddAsyncTag,
        JITEvaluatedSymbol(AddAsyncTagAddr, JITSymbolFlags::Exported)}})));

  ExecutorProcessControl::WrapperFunctionAssociationMap Associations;

  Associations[AddAsyncTag] =
      EPC->wrapAsyncWithSPS<int32_t(int32_t, int32_t)>(addAsyncWrapper);

  cantFail(EPC->associateJITSideWrapperFunctions(JD, std::move(Associations)));

  std::promise<int32_t> RP;
  auto RF = RP.get_future();

  using ArgSerialization = SPSArgList<int32_t, int32_t>;
  size_t ArgBufferSize = ArgSerialization::size(1, 2);
  WrapperFunctionResult ArgBuffer;
  char *ArgBufferData =
      WrapperFunctionResult::allocate(ArgBuffer, ArgBufferSize);
  SPSOutputBuffer OB(ArgBufferData, ArgBufferSize);
  EXPECT_TRUE(ArgSerialization::serialize(OB, 1, 2));

  EPC->runJITSideWrapperFunction(
      [&](WrapperFunctionResult ResultBuffer) {
        int32_t Result;
        SPSInputBuffer IB(ResultBuffer.data(), ResultBuffer.size());
        EXPECT_TRUE(SPSArgList<int32_t>::deserialize(IB, Result));
        RP.set_value(Result);
      },
      AddAsyncTagAddr, ArrayRef<char>(ArgBuffer.data(), ArgBuffer.size()));

  EXPECT_EQ(RF.get(), (int32_t)3);

  cantFail(ES.endSession());
}
