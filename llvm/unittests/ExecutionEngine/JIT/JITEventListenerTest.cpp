//===- JITEventListenerTest.cpp - Unit tests for JITEventListeners --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/JITEventListener.h"

#include "llvm/LLVMContext.h"
#include "llvm/Instructions.h"
#include "llvm/Module.h"
#include "llvm/TypeBuilder.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/CodeGen/MachineCodeInfo.h"
#include "llvm/ExecutionEngine/JIT.h"
#include "llvm/Support/TargetSelect.h"
#include "gtest/gtest.h"
#include <vector>

using namespace llvm;

int dummy;

namespace {

struct FunctionEmittedEvent {
  // Indices are local to the RecordingJITEventListener, since the
  // JITEventListener interface makes no guarantees about the order of
  // calls between Listeners.
  unsigned Index;
  const Function *F;
  void *Code;
  size_t Size;
  JITEvent_EmittedFunctionDetails Details;
};
struct FunctionFreedEvent {
  unsigned Index;
  void *Code;
};

struct RecordingJITEventListener : public JITEventListener {
  std::vector<FunctionEmittedEvent> EmittedEvents;
  std::vector<FunctionFreedEvent> FreedEvents;

  unsigned NextIndex;

  RecordingJITEventListener() : NextIndex(0) {}

  virtual void NotifyFunctionEmitted(const Function &F,
                                     void *Code, size_t Size,
                                     const EmittedFunctionDetails &Details) {
    FunctionEmittedEvent Event = {NextIndex++, &F, Code, Size, Details};
    EmittedEvents.push_back(Event);
  }

  virtual void NotifyFreeingMachineCode(void *OldPtr) {
    FunctionFreedEvent Event = {NextIndex++, OldPtr};
    FreedEvents.push_back(Event);
  }
};

class JITEventListenerTest : public testing::Test {
 protected:
  JITEventListenerTest()
      : M(new Module("module", getGlobalContext())),
        EE(EngineBuilder(M)
           .setEngineKind(EngineKind::JIT)
           .create()) {
  }

  Module *M;
  const OwningPtr<ExecutionEngine> EE;
};

Function *buildFunction(Module *M) {
  Function *Result = Function::Create(
      TypeBuilder<int32_t(int32_t), false>::get(getGlobalContext()),
      GlobalValue::ExternalLinkage, "id", M);
  Value *Arg = Result->arg_begin();
  BasicBlock *BB = BasicBlock::Create(M->getContext(), "entry", Result);
  ReturnInst::Create(M->getContext(), Arg, BB);
  return Result;
}

// Tests that a single JITEventListener follows JIT events accurately.
TEST_F(JITEventListenerTest, Simple) {
  RecordingJITEventListener Listener;
  EE->RegisterJITEventListener(&Listener);
  Function *F1 = buildFunction(M);
  Function *F2 = buildFunction(M);

  void *F1_addr = EE->getPointerToFunction(F1);
  void *F2_addr = EE->getPointerToFunction(F2);
  EE->getPointerToFunction(F1);  // Should do nothing.
  EE->freeMachineCodeForFunction(F1);
  EE->freeMachineCodeForFunction(F2);

  ASSERT_EQ(2U, Listener.EmittedEvents.size());
  ASSERT_EQ(2U, Listener.FreedEvents.size());

  EXPECT_EQ(0U, Listener.EmittedEvents[0].Index);
  EXPECT_EQ(F1, Listener.EmittedEvents[0].F);
  EXPECT_EQ(F1_addr, Listener.EmittedEvents[0].Code);
  EXPECT_LT(0U, Listener.EmittedEvents[0].Size)
      << "We don't know how big the function will be, but it had better"
      << " contain some bytes.";

  EXPECT_EQ(1U, Listener.EmittedEvents[1].Index);
  EXPECT_EQ(F2, Listener.EmittedEvents[1].F);
  EXPECT_EQ(F2_addr, Listener.EmittedEvents[1].Code);
  EXPECT_LT(0U, Listener.EmittedEvents[1].Size)
      << "We don't know how big the function will be, but it had better"
      << " contain some bytes.";

  EXPECT_EQ(2U, Listener.FreedEvents[0].Index);
  EXPECT_EQ(F1_addr, Listener.FreedEvents[0].Code);

  EXPECT_EQ(3U, Listener.FreedEvents[1].Index);
  EXPECT_EQ(F2_addr, Listener.FreedEvents[1].Code);

  F1->eraseFromParent();
  F2->eraseFromParent();
}

// Tests that a single JITEventListener follows JIT events accurately.
TEST_F(JITEventListenerTest, MultipleListenersDontInterfere) {
  RecordingJITEventListener Listener1;
  RecordingJITEventListener Listener2;
  RecordingJITEventListener Listener3;
  Function *F1 = buildFunction(M);
  Function *F2 = buildFunction(M);

  EE->RegisterJITEventListener(&Listener1);
  EE->RegisterJITEventListener(&Listener2);
  void *F1_addr = EE->getPointerToFunction(F1);
  EE->RegisterJITEventListener(&Listener3);
  EE->UnregisterJITEventListener(&Listener1);
  void *F2_addr = EE->getPointerToFunction(F2);
  EE->UnregisterJITEventListener(&Listener2);
  EE->UnregisterJITEventListener(&Listener3);
  EE->freeMachineCodeForFunction(F1);
  EE->RegisterJITEventListener(&Listener2);
  EE->RegisterJITEventListener(&Listener3);
  EE->RegisterJITEventListener(&Listener1);
  EE->freeMachineCodeForFunction(F2);
  EE->UnregisterJITEventListener(&Listener1);
  EE->UnregisterJITEventListener(&Listener2);
  EE->UnregisterJITEventListener(&Listener3);

  // Listener 1.
  ASSERT_EQ(1U, Listener1.EmittedEvents.size());
  ASSERT_EQ(1U, Listener1.FreedEvents.size());

  EXPECT_EQ(0U, Listener1.EmittedEvents[0].Index);
  EXPECT_EQ(F1, Listener1.EmittedEvents[0].F);
  EXPECT_EQ(F1_addr, Listener1.EmittedEvents[0].Code);
  EXPECT_LT(0U, Listener1.EmittedEvents[0].Size)
      << "We don't know how big the function will be, but it had better"
      << " contain some bytes.";

  EXPECT_EQ(1U, Listener1.FreedEvents[0].Index);
  EXPECT_EQ(F2_addr, Listener1.FreedEvents[0].Code);

  // Listener 2.
  ASSERT_EQ(2U, Listener2.EmittedEvents.size());
  ASSERT_EQ(1U, Listener2.FreedEvents.size());

  EXPECT_EQ(0U, Listener2.EmittedEvents[0].Index);
  EXPECT_EQ(F1, Listener2.EmittedEvents[0].F);
  EXPECT_EQ(F1_addr, Listener2.EmittedEvents[0].Code);
  EXPECT_LT(0U, Listener2.EmittedEvents[0].Size)
      << "We don't know how big the function will be, but it had better"
      << " contain some bytes.";

  EXPECT_EQ(1U, Listener2.EmittedEvents[1].Index);
  EXPECT_EQ(F2, Listener2.EmittedEvents[1].F);
  EXPECT_EQ(F2_addr, Listener2.EmittedEvents[1].Code);
  EXPECT_LT(0U, Listener2.EmittedEvents[1].Size)
      << "We don't know how big the function will be, but it had better"
      << " contain some bytes.";

  EXPECT_EQ(2U, Listener2.FreedEvents[0].Index);
  EXPECT_EQ(F2_addr, Listener2.FreedEvents[0].Code);

  // Listener 3.
  ASSERT_EQ(1U, Listener3.EmittedEvents.size());
  ASSERT_EQ(1U, Listener3.FreedEvents.size());

  EXPECT_EQ(0U, Listener3.EmittedEvents[0].Index);
  EXPECT_EQ(F2, Listener3.EmittedEvents[0].F);
  EXPECT_EQ(F2_addr, Listener3.EmittedEvents[0].Code);
  EXPECT_LT(0U, Listener3.EmittedEvents[0].Size)
      << "We don't know how big the function will be, but it had better"
      << " contain some bytes.";

  EXPECT_EQ(1U, Listener3.FreedEvents[0].Index);
  EXPECT_EQ(F2_addr, Listener3.FreedEvents[0].Code);

  F1->eraseFromParent();
  F2->eraseFromParent();
}

TEST_F(JITEventListenerTest, MatchesMachineCodeInfo) {
  RecordingJITEventListener Listener;
  MachineCodeInfo MCI;
  Function *F = buildFunction(M);

  EE->RegisterJITEventListener(&Listener);
  EE->runJITOnFunction(F, &MCI);
  void *F_addr = EE->getPointerToFunction(F);
  EE->freeMachineCodeForFunction(F);

  ASSERT_EQ(1U, Listener.EmittedEvents.size());
  ASSERT_EQ(1U, Listener.FreedEvents.size());

  EXPECT_EQ(0U, Listener.EmittedEvents[0].Index);
  EXPECT_EQ(F, Listener.EmittedEvents[0].F);
  EXPECT_EQ(F_addr, Listener.EmittedEvents[0].Code);
  EXPECT_EQ(MCI.address(), Listener.EmittedEvents[0].Code);
  EXPECT_EQ(MCI.size(), Listener.EmittedEvents[0].Size);

  EXPECT_EQ(1U, Listener.FreedEvents[0].Index);
  EXPECT_EQ(F_addr, Listener.FreedEvents[0].Code);
}

class JITEnvironment : public testing::Environment {
  virtual void SetUp() {
    // Required to create a JIT.
    InitializeNativeTarget();
  }
};
testing::Environment* const jit_env =
  testing::AddGlobalTestEnvironment(new JITEnvironment);

}  // anonymous namespace
