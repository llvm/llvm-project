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
// These tests exercise the AllocAction layer directly, using a small bespoke
// (de)serializer that exchanges raw int* values via memcpy. SPS-layer tests
// live in SPSAllocActionTest.cpp.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/AllocAction.h"

#include "gtest/gtest.h"

#include "CommonTestUtils.h"

#include <cstring>

using namespace orc_rt;

namespace {

// A minimal AllocActionFunction (de)serializer pair that exchanges a single
// int* via memcpy. Used to drive AllocActionFunction::handle without pulling
// in SPS.
struct IntPtrDeserializer {
  bool deserialize(const char *ArgData, size_t ArgSize,
                   std::tuple<int *> &Args) {
    if (ArgSize != sizeof(int *))
      return false;
    memcpy(&std::get<0>(Args), ArgData, sizeof(int *));
    return true;
  }
};

struct IdentitySerializer {
  static WrapperFunctionBuffer serialize(WrapperFunctionBuffer B) { return B; }
};

WrapperFunctionBuffer makeIntPtrArgBuffer(int *P) {
  auto B = WrapperFunctionBuffer::allocate(sizeof(int *));
  memcpy(B.data(), &P, sizeof(int *));
  return B;
}

} // namespace

static orc_rt_WrapperFunctionBuffer noopAction(const char *ArgData,
                                               size_t ArgSize) {
  return WrapperFunctionBuffer().release();
}

// Increments an int via pointer.
static orc_rt_WrapperFunctionBuffer
increment_int_ptr_action(const char *ArgData, size_t ArgSize) {
  return AllocActionFunction::handle(ArgData, ArgSize, IntPtrDeserializer(),
                                     IdentitySerializer(),
                                     [](int *P) {
                                       ++*P;
                                       return WrapperFunctionBuffer();
                                     })
      .release();
}

// Decrements an int via pointer.
static orc_rt_WrapperFunctionBuffer
decrement_int_ptr_action(const char *ArgData, size_t ArgSize) {
  return AllocActionFunction::handle(ArgData, ArgSize, IntPtrDeserializer(),
                                     IdentitySerializer(),
                                     [](int *P) {
                                       --*P;
                                       return WrapperFunctionBuffer();
                                     })
      .release();
}

// Always returns an out-of-band error.
static orc_rt_WrapperFunctionBuffer fail_action(const char *ArgData,
                                                size_t ArgSize) {
  return WrapperFunctionBuffer::createOutOfBandError("failed").release();
}

// Always returns an out-of-band error with a distinct message. Used to tell
// finalize-path and dealloc-path failures apart in error-reporting tests.
static orc_rt_WrapperFunctionBuffer fail_action_2(const char *ArgData,
                                                  size_t ArgSize) {
  return WrapperFunctionBuffer::createOutOfBandError("failed_2").release();
}

TEST(AllocActionTest, DefaultConstruct) {
  AllocAction AA;
  EXPECT_FALSE(AA);
}

TEST(AllocActionTest, ConstructWithAction) {
  AllocAction AA(noopAction, WrapperFunctionBuffer());
  EXPECT_TRUE(AA);
}

TEST(AllocActionTest, RunBasicAction) {
  int Val = 0;
  AllocAction IncVal(increment_int_ptr_action, makeIntPtrArgBuffer(&Val));
  EXPECT_TRUE(IncVal);
  auto B = IncVal();
  EXPECT_TRUE(B.empty());
  EXPECT_EQ(Val, 1);
}

TEST(AllocActionTest, RunFinalizationActionsComplete) {
  int Val = 0;

  std::vector<AllocActionPair> InitialActions;

  auto MakeAAOnVal = [&](AllocActionFn Fn) {
    return AllocAction(Fn, makeIntPtrArgBuffer(&Val));
  };
  InitialActions.push_back({MakeAAOnVal(increment_int_ptr_action),
                            MakeAAOnVal(decrement_int_ptr_action)});
  InitialActions.push_back({MakeAAOnVal(increment_int_ptr_action),
                            MakeAAOnVal(decrement_int_ptr_action)});

  auto DeallocActions =
      cantFail(runFinalizeActions(std::move(InitialActions), noErrors));

  EXPECT_EQ(Val, 2);

  runDeallocActions(std::move(DeallocActions), noErrors);

  EXPECT_EQ(Val, 0);
}

TEST(AllocActionTest, RunFinalizeActionsFail) {
  int SucceedingPairVal = 0;
  int FailingPairDeallocVal = 0;
  int AfterFailurePairVal = 0;

  std::vector<AllocActionPair> InitialActions;

  auto MakeAA = [&](AllocActionFn Fn, int *P) {
    return AllocAction(Fn, makeIntPtrArgBuffer(P));
  };
  // First pair's finalize and dealloc actions both increment SucceedingPairVal.
  InitialActions.push_back(
      {MakeAA(increment_int_ptr_action, &SucceedingPairVal),
       MakeAA(increment_int_ptr_action, &SucceedingPairVal)});
  // Second pair's finalize fails, so its dealloc action must not run.
  InitialActions.push_back(
      {AllocAction(fail_action, WrapperFunctionBuffer()), // finalize fails
       MakeAA(increment_int_ptr_action, &FailingPairDeallocVal)});
  // Third pair: finalize actions past the failure point must not run.
  InitialActions.push_back(
      {MakeAA(increment_int_ptr_action, &AfterFailurePairVal),
       MakeAA(increment_int_ptr_action, &AfterFailurePairVal)});

  auto DeallocActions = runFinalizeActions(std::move(InitialActions), noErrors);

  if (DeallocActions) {
    ADD_FAILURE() << "Failed to report error from runFinalizeActions";
    return;
  }

  EXPECT_EQ(toString(DeallocActions.takeError()), std::string("failed"));

  // First pair fully ran: +1 from its finalize action, +1 from its dealloc
  // action during cleanup.
  EXPECT_EQ(SucceedingPairVal, 2);

  // Second pair's finalize failed, so its dealloc action must not run.
  EXPECT_EQ(FailingPairDeallocVal, 0);

  // Third pair's finalize is past the failure point, so it must not run.
  EXPECT_EQ(AfterFailurePairVal, 0);
}

TEST(AllocActionTest, RunFinalizeActionsNullFinalize) {
  int Val = 0;

  std::vector<AllocActionPair> InitialActions;

  auto MakeAAOnVal = [&](AllocActionFn Fn) {
    return AllocAction(Fn, makeIntPtrArgBuffer(&Val));
  };
  InitialActions.push_back({MakeAAOnVal(increment_int_ptr_action),
                            MakeAAOnVal(decrement_int_ptr_action)});
  InitialActions.push_back({AllocAction(nullptr, WrapperFunctionBuffer()),
                            MakeAAOnVal(decrement_int_ptr_action)});

  auto DeallocActions =
      cantFail(runFinalizeActions(std::move(InitialActions), noErrors));

  // Both dealloc actions should be included in the returned list, despite one
  // of them having a null finalize action.
  EXPECT_EQ(DeallocActions.size(), 2U);

  runDeallocActions(std::move(DeallocActions), noErrors);

  EXPECT_EQ(Val, -1);
}

TEST(AllocActionTest, RunFinalizeActionsNullDealloc) {
  int Val = 0;

  std::vector<AllocActionPair> InitialActions;

  auto MakeAAOnVal = [&](AllocActionFn Fn) {
    return AllocAction(Fn, makeIntPtrArgBuffer(&Val));
  };
  InitialActions.push_back({MakeAAOnVal(increment_int_ptr_action),
                            MakeAAOnVal(decrement_int_ptr_action)});
  InitialActions.push_back({MakeAAOnVal(increment_int_ptr_action),
                            AllocAction(nullptr, WrapperFunctionBuffer())});

  auto DeallocActions =
      cantFail(runFinalizeActions(std::move(InitialActions), noErrors));

  // Null dealloc actions should be filtered out of the returned list.
  EXPECT_EQ(DeallocActions.size(), 1U);

  runDeallocActions(std::move(DeallocActions), noErrors);

  EXPECT_EQ(Val, 1);
}

TEST(AllocActionTest, RunDeallocActionsReportsError) {
  int Val = 0;

  std::vector<AllocAction> DeallocActions;
  // First dealloc action runs second (dealloc runs in reverse), but must still
  // run despite the failure reported below.
  DeallocActions.push_back(
      AllocAction(increment_int_ptr_action, makeIntPtrArgBuffer(&Val)));
  // Runs first: fails and should be reported.
  DeallocActions.push_back(AllocAction(fail_action, WrapperFunctionBuffer()));

  std::vector<std::string> ErrMsgs;
  runDeallocActions(std::move(DeallocActions), AccumulateErrors(ErrMsgs));

  ASSERT_EQ(ErrMsgs.size(), 1U);
  EXPECT_EQ(ErrMsgs[0], "failed");

  // The non-failing dealloc action still ran.
  EXPECT_EQ(Val, 1);
}

TEST(AllocActionTest, RunDeallocActionsReportsAllErrors) {
  std::vector<AllocAction> DeallocActions;
  DeallocActions.push_back(AllocAction(fail_action, WrapperFunctionBuffer()));
  DeallocActions.push_back(AllocAction(fail_action_2, WrapperFunctionBuffer()));

  std::vector<std::string> ErrMsgs;
  runDeallocActions(std::move(DeallocActions), AccumulateErrors(ErrMsgs));

  // Both failures reported; dealloc runs in reverse, so fail_action_2 first.
  ASSERT_EQ(ErrMsgs.size(), 2U);
  EXPECT_EQ(ErrMsgs[0], "failed_2");
  EXPECT_EQ(ErrMsgs[1], "failed");
}

TEST(AllocActionTest, RunFinalizeActionsFailReportsCleanupErrors) {
  int Val = 0;

  std::vector<AllocActionPair> InitialActions;
  // Finalize succeeds; its dealloc decrements Val during cleanup. Proves
  // cleanup continues past the failure of the pair below.
  InitialActions.push_back(
      {AllocAction(increment_int_ptr_action, makeIntPtrArgBuffer(&Val)),
       AllocAction(decrement_int_ptr_action, makeIntPtrArgBuffer(&Val))});
  // Finalize succeeds; its dealloc fails during the cleanup triggered below.
  InitialActions.push_back(
      {AllocAction(increment_int_ptr_action, makeIntPtrArgBuffer(&Val)),
       AllocAction(fail_action, WrapperFunctionBuffer())});
  // Finalize fails, triggering cleanup of the accumulated dealloc actions.
  InitialActions.push_back({AllocAction(fail_action_2, WrapperFunctionBuffer()),
                            AllocAction(nullptr, WrapperFunctionBuffer())});

  std::vector<std::string> ErrMsgs;
  auto DeallocActions =
      runFinalizeActions(std::move(InitialActions), AccumulateErrors(ErrMsgs));

  // The finalize failure is the returned error...
  ASSERT_FALSE(DeallocActions);
  EXPECT_EQ(toString(DeallocActions.takeError()), std::string("failed_2"));

  // ...while the cleanup dealloc failure is reported out-of-band.
  ASSERT_EQ(ErrMsgs.size(), 1U);
  EXPECT_EQ(ErrMsgs[0], "failed");

  // Both finalize actions ran (Val += 2) and both cleanup dealloc actions ran:
  // the failing one left Val untouched, the surviving one decremented it. That
  // the decrement took effect proves cleanup continued past the failure.
  EXPECT_EQ(Val, 1);
}
