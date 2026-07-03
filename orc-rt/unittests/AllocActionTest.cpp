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

} // anonymous namespace

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

  auto DeallocActions = cantFail(runFinalizeActions(std::move(InitialActions)));

  EXPECT_EQ(Val, 2);

  runDeallocActions(std::move(DeallocActions));

  EXPECT_EQ(Val, 0);
}

TEST(AllocActionTest, RunFinalizeActionsFail) {
  int Val = 0;

  std::vector<AllocActionPair> InitialActions;

  auto MakeAAOnVal = [&](AllocActionFn Fn) {
    return AllocAction(Fn, makeIntPtrArgBuffer(&Val));
  };
  InitialActions.push_back({MakeAAOnVal(increment_int_ptr_action),
                            MakeAAOnVal(decrement_int_ptr_action)});
  InitialActions.push_back({AllocAction(fail_action, WrapperFunctionBuffer()),
                            MakeAAOnVal(decrement_int_ptr_action)});

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
    return AllocAction(Fn, makeIntPtrArgBuffer(&Val));
  };
  InitialActions.push_back({MakeAAOnVal(increment_int_ptr_action),
                            MakeAAOnVal(decrement_int_ptr_action)});
  InitialActions.push_back({AllocAction(nullptr, WrapperFunctionBuffer()),
                            MakeAAOnVal(decrement_int_ptr_action)});

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
    return AllocAction(Fn, makeIntPtrArgBuffer(&Val));
  };
  InitialActions.push_back({MakeAAOnVal(increment_int_ptr_action),
                            MakeAAOnVal(decrement_int_ptr_action)});
  InitialActions.push_back({MakeAAOnVal(increment_int_ptr_action),
                            AllocAction(nullptr, WrapperFunctionBuffer())});

  auto DeallocActions = cantFail(runFinalizeActions(std::move(InitialActions)));

  // Null dealloc actions should be filtered out of the returned list.
  EXPECT_EQ(DeallocActions.size(), 1U);

  runDeallocActions(std::move(DeallocActions));

  EXPECT_EQ(Val, 1);
}
