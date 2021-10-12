//===- ExecutionSessionWrapperFunctionCallsTest.cpp -- Test wrapper calls -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/ExecutorProcessControl.h"
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

static llvm::orc::shared::detail::CWrapperFunctionResult
voidWrapper(const char *ArgData, size_t ArgSize) {
  return WrapperFunction<void()>::handle(ArgData, ArgSize, []() {}).release();
}

TEST(ExecutionSessionWrapperFunctionCalls, RunWrapperTemplate) {
  ExecutionSession ES(cantFail(SelfExecutorProcessControl::Create()));

  int32_t Result;
  EXPECT_THAT_ERROR(ES.callSPSWrapper<int32_t(int32_t, int32_t)>(
                        ExecutorAddr::fromPtr(addWrapper), Result, 2, 3),
                    Succeeded());
  EXPECT_EQ(Result, 5);
  cantFail(ES.endSession());
}

TEST(ExecutionSessionWrapperFunctionCalls, RunVoidWrapperAsyncTemplate) {
  ExecutionSession ES(cantFail(SelfExecutorProcessControl::Create()));

  std::promise<MSVCPError> RP;
  ES.callSPSWrapperAsync<void()>(ExecutorAddr::fromPtr(voidWrapper),
                                 [&](Error SerializationErr) {
                                   RP.set_value(std::move(SerializationErr));
                                 });
  Error Err = RP.get_future().get();
  EXPECT_THAT_ERROR(std::move(Err), Succeeded());
  cantFail(ES.endSession());
}

TEST(ExecutionSessionWrapperFunctionCalls, RunNonVoidWrapperAsyncTemplate) {
  ExecutionSession ES(cantFail(SelfExecutorProcessControl::Create()));

  std::promise<MSVCPExpected<int32_t>> RP;
  ES.callSPSWrapperAsync<int32_t(int32_t, int32_t)>(
      ExecutorAddr::fromPtr(addWrapper),
      [&](Error SerializationErr, int32_t R) {
        if (SerializationErr)
          RP.set_value(std::move(SerializationErr));
        RP.set_value(std::move(R));
      },
      2, 3);
  Expected<int32_t> Result = RP.get_future().get();
  EXPECT_THAT_EXPECTED(Result, HasValue(5));
  cantFail(ES.endSession());
}

TEST(ExecutionSessionWrapperFunctionCalls, RegisterAsyncHandlerAndRun) {

  constexpr JITTargetAddress AddAsyncTagAddr = 0x01;

  ExecutionSession ES(cantFail(SelfExecutorProcessControl::Create()));
  auto &JD = ES.createBareJITDylib("JD");

  auto AddAsyncTag = ES.intern("addAsync_tag");
  cantFail(JD.define(absoluteSymbols(
      {{AddAsyncTag,
        JITEvaluatedSymbol(AddAsyncTagAddr, JITSymbolFlags::Exported)}})));

  ExecutionSession::JITDispatchHandlerAssociationMap Associations;

  Associations[AddAsyncTag] =
      ES.wrapAsyncWithSPS<int32_t(int32_t, int32_t)>(addAsyncWrapper);

  cantFail(ES.registerJITDispatchHandlers(JD, std::move(Associations)));

  std::promise<int32_t> RP;
  auto RF = RP.get_future();

  using ArgSerialization = SPSArgList<int32_t, int32_t>;
  size_t ArgBufferSize = ArgSerialization::size(1, 2);
  auto ArgBuffer = WrapperFunctionResult::allocate(ArgBufferSize);
  SPSOutputBuffer OB(ArgBuffer.data(), ArgBuffer.size());
  EXPECT_TRUE(ArgSerialization::serialize(OB, 1, 2));

  ES.runJITDispatchHandler(
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
