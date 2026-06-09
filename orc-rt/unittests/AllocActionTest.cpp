//===- AllocActionTest.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for orc-rt's AllocAction.h APIs.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/AllocAction.h"
#include "orc-rt/ExecutorAddress.h"
#include "orc-rt/SPSAllocAction.h"

#include "AllocActionTestUtils.h"
#include "gtest/gtest.h"

using namespace orc_rt;

TEST(AllocActionTest, DefaultConstruct) {
  AllocAction AA;
  EXPECT_FALSE(AA);
}

static orc_rt_WrapperFunctionBuffer noopAction(const char *ArgData,
                                               size_t ArgSize) {
  return WrapperFunctionBuffer().release();
}

TEST(AllocActionTest, ConstructWithAction) {
  AllocAction AA(noopAction, WrapperFunctionBuffer());
  EXPECT_TRUE(AA);
}

// Increments int via pointer.
static orc_rt_WrapperFunctionBuffer
increment_sps_allocaction(const char *ArgData, size_t ArgSize) {
  return SPSAllocActionFunction<SPSExecutorAddr>::handle(
             ArgData, ArgSize,
             [](ExecutorAddr IntPtr) {
               *IntPtr.toPtr<int *>() += 1;
               return WrapperFunctionBuffer();
             })
      .release();
}

// Increments int via pointer.
static orc_rt_WrapperFunctionBuffer
decrement_sps_allocaction(const char *ArgData, size_t ArgSize) {
  return SPSAllocActionFunction<SPSExecutorAddr>::handle(
             ArgData, ArgSize,
             [](ExecutorAddr IntPtr) {
               *IntPtr.toPtr<int *>() -= 1;
               return WrapperFunctionBuffer();
             })
      .release();
}

template <typename T>
static WrapperFunctionBuffer makeExecutorAddrBuffer(T *P) {
  return *spsSerialize<SPSArgList<SPSExecutorAddr>>(ExecutorAddr::fromPtr(P));
}

TEST(AllocActionTest, RunBasicAction) {
  int Val = 0;
  AllocAction IncVal(increment_sps_allocaction, makeExecutorAddrBuffer(&Val));
  EXPECT_TRUE(IncVal);
  auto B = IncVal();
  EXPECT_TRUE(B.empty());
  EXPECT_EQ(Val, 1);
}

TEST(AllocActionTest, RunFinalizationActionsComplete) {
  int Val = 0;

  std::vector<AllocActionPair> InitialActions;

  auto MakeAAOnVal = [&](AllocActionFn Fn) {
    return *MakeAllocAction<SPSExecutorAddr>::from(Fn,
                                                   ExecutorAddr::fromPtr(&Val));
  };
  InitialActions.push_back({MakeAAOnVal(increment_sps_allocaction),
                            MakeAAOnVal(decrement_sps_allocaction)});
  InitialActions.push_back({MakeAAOnVal(increment_sps_allocaction),
                            MakeAAOnVal(decrement_sps_allocaction)});

  auto DeallocActions = cantFail(runFinalizeActions(std::move(InitialActions)));

  EXPECT_EQ(Val, 2);

  runDeallocActions(std::move(DeallocActions));

  EXPECT_EQ(Val, 0);
}

static orc_rt_WrapperFunctionBuffer fail_sps_allocaction(const char *ArgData,
                                                         size_t ArgSize) {
  return WrapperFunctionBuffer::createOutOfBandError("failed").release();
}

TEST(AllocActionTest, RunFinalizeActionsFail) {
  int Val = 0;

  std::vector<AllocActionPair> InitialActions;

  auto MakeAAOnVal = [&](AllocActionFn Fn) {
    return *MakeAllocAction<SPSExecutorAddr>::from(Fn,
                                                   ExecutorAddr::fromPtr(&Val));
  };
  InitialActions.push_back({MakeAAOnVal(increment_sps_allocaction),
                            MakeAAOnVal(decrement_sps_allocaction)});
  InitialActions.push_back({*MakeAllocAction<>::from(fail_sps_allocaction),
                            MakeAAOnVal(decrement_sps_allocaction)});

  auto DeallocActions = runFinalizeActions(std::move(InitialActions));

  if (DeallocActions) {
    ADD_FAILURE() << "Failed to report error from runFinalizeActions";
    return;
  }

  EXPECT_EQ(toString(DeallocActions.takeError()), std::string("failed"));

  // Check that we ran the decrement corresponding to the first increment.
  EXPECT_EQ(Val, 0);
}

TEST(AllocActionTest, RunFinalizeActionsNullFinalize) {
  int Val = 0;

  std::vector<AllocActionPair> InitialActions;

  auto MakeAAOnVal = [&](AllocActionFn Fn) {
    return *MakeAllocAction<SPSExecutorAddr>::from(Fn,
                                                   ExecutorAddr::fromPtr(&Val));
  };
  InitialActions.push_back({MakeAAOnVal(increment_sps_allocaction),
                            MakeAAOnVal(decrement_sps_allocaction)});
  InitialActions.push_back({*MakeAllocAction<>::from(nullptr),
                            MakeAAOnVal(decrement_sps_allocaction)});

  auto DeallocActions = cantFail(runFinalizeActions(std::move(InitialActions)));

  // Both dealloc actions should be included in the returned list, despite one
  // of them having a null finalize action.
  EXPECT_EQ(DeallocActions.size(), 2U);

  runDeallocActions(std::move(DeallocActions));

  EXPECT_EQ(Val, -1);
}

TEST(AllocActionTest, RunFinalizeActionsNullDealloc) {
  int Val = 0;

  std::vector<AllocActionPair> InitialActions;

  auto MakeAAOnVal = [&](AllocActionFn Fn) {
    return *MakeAllocAction<SPSExecutorAddr>::from(Fn,
                                                   ExecutorAddr::fromPtr(&Val));
  };
  InitialActions.push_back({MakeAAOnVal(increment_sps_allocaction),
                            MakeAAOnVal(decrement_sps_allocaction)});
  InitialActions.push_back({MakeAAOnVal(increment_sps_allocaction),
                            *MakeAllocAction<>::from(nullptr)});

  auto DeallocActions = cantFail(runFinalizeActions(std::move(InitialActions)));

  // Null dealloc actions should be filtered out of the returned list.
  EXPECT_EQ(DeallocActions.size(), 1U);

  runDeallocActions(std::move(DeallocActions));

  EXPECT_EQ(Val, 1);
}
