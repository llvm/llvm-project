//===---------------- SimpleExecutorMemoryManagerTest.cpp -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/TargetProcess/SimpleExecutorMemoryManager.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

#include <limits>
#include <vector>

using namespace llvm;
using namespace llvm::orc;
using namespace llvm::orc::shared;
using namespace llvm::orc::rt_bootstrap;

namespace {

orc::shared::CWrapperFunctionResult incrementWrapper(const char *ArgData,
                                                     size_t ArgSize) {
  return WrapperFunction<SPSError(SPSExecutorAddr)>::handle(
             ArgData, ArgSize,
             [](ExecutorAddr A) -> Error {
               *A.toPtr<int *>() += 1;
               return Error::success();
             })
      .release();
}

TEST(SimpleExecutorMemoryManagerTest, AllocFinalizeFree) {
  SimpleExecutorMemoryManager MemMgr;

  constexpr unsigned AllocSize = 16384;
  auto Mem = MemMgr.allocate(AllocSize);
  EXPECT_THAT_ERROR(Mem.takeError(), Succeeded());

  std::string HW = "Hello, world!";

  int FinalizeCounter = 0;
  int DeallocateCounter = 0;

  tpctypes::FinalizeRequest FR;
  FR.Segments.push_back(
      tpctypes::SegFinalizeRequest{tpctypes::WPF_Read | tpctypes::WPF_Write,
                                   *Mem,
                                   AllocSize,
                                   {HW.data(), HW.size() + 1}});
  FR.Actions.push_back(
      {/* Finalize: */
       cantFail(WrapperFunctionCall::Create<SPSArgList<SPSExecutorAddr>>(
           ExecutorAddr::fromPtr(incrementWrapper),
           ExecutorAddr::fromPtr(&FinalizeCounter))),
       /*  Deallocate: */
       cantFail(WrapperFunctionCall::Create<SPSArgList<SPSExecutorAddr>>(
           ExecutorAddr::fromPtr(incrementWrapper),
           ExecutorAddr::fromPtr(&DeallocateCounter)))});

  EXPECT_EQ(FinalizeCounter, 0);
  EXPECT_EQ(DeallocateCounter, 0);

  auto FinalizeErr = MemMgr.finalize(FR);
  EXPECT_THAT_ERROR(std::move(FinalizeErr), Succeeded());

  EXPECT_EQ(FinalizeCounter, 1);
  EXPECT_EQ(DeallocateCounter, 0);

  EXPECT_EQ(HW, std::string(Mem->toPtr<const char *>()));

  auto DeallocateErr = MemMgr.deallocate({*Mem});
  EXPECT_THAT_ERROR(std::move(DeallocateErr), Succeeded());

  EXPECT_EQ(FinalizeCounter, 1);
  EXPECT_EQ(DeallocateCounter, 1);
}

} // namespace
