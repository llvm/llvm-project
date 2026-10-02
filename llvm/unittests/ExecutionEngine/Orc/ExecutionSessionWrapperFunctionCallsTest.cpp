//===- ExecutionSessionWrapperFunctionCallsTest.cpp -- Test wrapper calls -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/AbsoluteSymbols.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/ExecutorProcessControl.h"
#include "llvm/ExecutionEngine/Orc/SelfExecutorProcessControl.h"
#include "gtest/gtest.h"

#include <future>

using namespace llvm;
using namespace llvm::orc;
using namespace llvm::orc::shared;

static void addAsyncWrapper(unique_function<void(int32_t)> SendResult,
                            int32_t X, int32_t Y) {
  SendResult(X + Y);
}

TEST(ExecutionSessionWrapperFunctionCalls, RegisterAsyncHandlerAndRun) {

  constexpr ExecutorAddr AddAsyncTagAddr(0x01);

  ExecutionSession ES(cantFail(SelfExecutorProcessControl::Create()));
  auto &JD = ES.createBareJITDylib("JD");

  auto AddAsyncTag = ES.intern("addAsync_tag");
  cantFail(JD.define(absoluteSymbols(
      {{AddAsyncTag, {AddAsyncTagAddr, JITSymbolFlags::Exported}}})));

  ExecutionSession::JITDispatchHandlerAssociationMap Associations;

  Associations[AddAsyncTag] =
      ES.wrapAsyncWithSPS<int32_t(int32_t, int32_t)>(addAsyncWrapper);

  cantFail(ES.registerJITDispatchHandlers(JD, std::move(Associations)));

  std::promise<int32_t> RP;
  auto RF = RP.get_future();

  using ArgSerialization = SPSArgList<int32_t, int32_t>;
  size_t ArgBufferSize = ArgSerialization::size(1, 2);
  auto ArgBuffer = WrapperFunctionBuffer::allocate(ArgBufferSize);
  SPSOutputBuffer OB(ArgBuffer.data(), ArgBuffer.size());
  EXPECT_TRUE(ArgSerialization::serialize(OB, 1, 2));

  ES.runJITDispatchHandler(
      [&](WrapperFunctionBuffer ResultBuffer) {
        int32_t Result;
        SPSInputBuffer IB(ResultBuffer.data(), ResultBuffer.size());
        EXPECT_TRUE(SPSArgList<int32_t>::deserialize(IB, Result));
        RP.set_value(Result);
      },
      AddAsyncTagAddr, std::move(ArgBuffer));

  EXPECT_EQ(RF.get(), (int32_t)3);

  cantFail(ES.endSession());
}
