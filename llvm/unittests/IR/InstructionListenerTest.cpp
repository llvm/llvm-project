//===- InstructionListenerTest.cpp - per-Function instruction listener ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/InstructionListener.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

class TrackingListener : public InstructionListener {
  DenseSet<const Value *> &UniformValuesRef;

  static void onRemoved(InstructionListener *Self, Instruction *I) {
    static_cast<TrackingListener *>(Self)->UniformValuesRef.erase(I);
  }

  static void onRAUW(InstructionListener *Self, Instruction *Old, Value *New) {
    TrackingListener *This = static_cast<TrackingListener *>(Self);
    This->UniformValuesRef.erase(Old);
  }

public:
  TrackingListener(Function &F, DenseSet<const Value *> &UniformValues)
      : InstructionListener(F, &onRemoved, &onRAUW),
        UniformValuesRef(UniformValues) {}
};

static Function *createFunction(Module &M, const char *Name) {
  Type *I32Ty = Type::getInt32Ty(M.getContext());
  FunctionType *FTy = FunctionType::get(I32Ty, {I32Ty, I32Ty}, false);
  return Function::Create(FTy, GlobalValue::ExternalLinkage, Name, &M);
}

TEST(InstructionListenerTest, EraseFromParent) {
  LLVMContext C;
  Module M("test", C);
  Function *F = createFunction(M, "f");
  BasicBlock *BB = BasicBlock::Create(C, "entry", F);
  IRBuilder<> Builder(BB);

  Instruction *Add =
      cast<Instruction>(Builder.CreateAdd(F->getArg(0), F->getArg(1)));
  Builder.CreateRet(F->getArg(0));

  DenseSet<const Value *> UniformValues;
  UniformValues.insert(Add);

  TrackingListener Listener(*F, UniformValues);

  EXPECT_TRUE(UniformValues.contains(Add));
  Add->eraseFromParent();
  EXPECT_FALSE(UniformValues.contains(Add));
}

TEST(InstructionListenerTest, RemoveAndDelete) {
  LLVMContext C;
  Module M("test", C);
  Function *F = createFunction(M, "f");
  BasicBlock *BB = BasicBlock::Create(C, "entry", F);
  IRBuilder<> Builder(BB);

  Instruction *Add =
      cast<Instruction>(Builder.CreateAdd(F->getArg(0), F->getArg(1)));
  Builder.CreateRet(F->getArg(0));

  DenseSet<const Value *> UniformValues;
  UniformValues.insert(Add);

  TrackingListener Listener(*F, UniformValues);

  EXPECT_TRUE(UniformValues.contains(Add));
  Add->removeFromParent();
  EXPECT_FALSE(UniformValues.contains(Add));
  Add->deleteValue();
}

TEST(InstructionListenerTest, BasicBlockErasure) {
  LLVMContext C;
  Module M("test", C);
  Function *F = createFunction(M, "f");
  BasicBlock *Entry = BasicBlock::Create(C, "entry", F);
  BasicBlock *Dead = BasicBlock::Create(C, "dead", F);
  IRBuilder<> Builder(Entry);
  Builder.CreateRet(F->getArg(0));

  Builder.SetInsertPoint(Dead);
  Instruction *Add =
      cast<Instruction>(Builder.CreateAdd(F->getArg(0), F->getArg(1)));
  Builder.CreateUnreachable();

  DenseSet<const Value *> UniformValues;
  UniformValues.insert(Add);

  TrackingListener Listener(*F, UniformValues);

  EXPECT_TRUE(UniformValues.contains(Add));
  Dead->eraseFromParent();
  EXPECT_FALSE(UniformValues.contains(Add));
}

TEST(InstructionListenerTest, ListenerScopeRAII) {
  LLVMContext C;
  Module M("test", C);
  Function *F = createFunction(M, "f");
  BasicBlock *BB = BasicBlock::Create(C, "entry", F);
  IRBuilder<> Builder(BB);
  Builder.CreateRet(F->getArg(0));

  EXPECT_FALSE(F->hasInstructionListeners());
  {
    DenseSet<const Value *> UniformValues;
    TrackingListener Listener(*F, UniformValues);
    EXPECT_TRUE(F->hasInstructionListeners());
  }
  EXPECT_FALSE(F->hasInstructionListeners());
}

TEST(InstructionListenerTest, MultipleListeners) {
  LLVMContext C;
  Module M("test", C);
  Function *F = createFunction(M, "f");
  BasicBlock *BB = BasicBlock::Create(C, "entry", F);
  IRBuilder<> Builder(BB);

  Instruction *Add =
      cast<Instruction>(Builder.CreateAdd(F->getArg(0), F->getArg(1)));
  Builder.CreateRet(F->getArg(0));

  DenseSet<const Value *> UniformValues1;
  DenseSet<const Value *> UniformValues2;
  UniformValues1.insert(Add);
  UniformValues2.insert(Add);

  TrackingListener L1(*F, UniformValues1);
  TrackingListener L2(*F, UniformValues2);

  EXPECT_TRUE(UniformValues1.contains(Add));
  EXPECT_TRUE(UniformValues2.contains(Add));
  Add->eraseFromParent();
  EXPECT_FALSE(UniformValues1.contains(Add));
  EXPECT_FALSE(UniformValues2.contains(Add));
}

TEST(InstructionListenerTest, PerFunctionIsolation) {
  LLVMContext C;
  Module M("test", C);
  Function *F1 = createFunction(M, "f1");
  Function *F2 = createFunction(M, "f2");

  BasicBlock *BB1 = BasicBlock::Create(C, "entry", F1);
  BasicBlock *BB2 = BasicBlock::Create(C, "entry", F2);

  IRBuilder<> B1(BB1);
  Instruction *Add1 =
      cast<Instruction>(B1.CreateAdd(F1->getArg(0), F1->getArg(1)));
  B1.CreateRet(F1->getArg(0));

  IRBuilder<> B2(BB2);
  Instruction *Add2 =
      cast<Instruction>(B2.CreateAdd(F2->getArg(0), F2->getArg(1)));
  B2.CreateRet(F2->getArg(0));

  DenseSet<const Value *> UniformValues1;
  DenseSet<const Value *> UniformValues2;
  UniformValues1.insert(Add1);
  UniformValues2.insert(Add2);

  TrackingListener L1(*F1, UniformValues1);
  TrackingListener L2(*F2, UniformValues2);

  // Erasing from F1 should not affect F2's listener.
  Add1->eraseFromParent();
  EXPECT_FALSE(UniformValues1.contains(Add1));
  EXPECT_TRUE(UniformValues2.contains(Add2));

  Add2->eraseFromParent();
  EXPECT_FALSE(UniformValues2.contains(Add2));
}

TEST(InstructionListenerTest, RAUWBasic) {
  LLVMContext C;
  Module M("test", C);
  Function *F = createFunction(M, "f");
  BasicBlock *BB = BasicBlock::Create(C, "entry", F);
  IRBuilder<> Builder(BB);

  Instruction *Add =
      cast<Instruction>(Builder.CreateAdd(F->getArg(0), F->getArg(1)));
  Builder.CreateRet(Add);

  DenseSet<const Value *> UniformValues;
  UniformValues.insert(Add);

  TrackingListener Listener(*F, UniformValues);

  EXPECT_TRUE(UniformValues.contains(Add));
  Add->replaceAllUsesWith(F->getArg(0));
  EXPECT_FALSE(UniformValues.contains(Add));
}

TEST(InstructionListenerTest, RAUWReceivesNewValue) {
  LLVMContext C;
  Module M("test", C);
  Function *F = createFunction(M, "f");
  BasicBlock *BB = BasicBlock::Create(C, "entry", F);
  IRBuilder<> Builder(BB);

  Instruction *Add =
      cast<Instruction>(Builder.CreateAdd(F->getArg(0), F->getArg(1)));
  Builder.CreateRet(Add);

  Value *ReceivedOld = nullptr;
  Value *ReceivedNew = nullptr;

  class CapturingListener : public InstructionListener {
    Value *&OldRef;
    Value *&NewRef;

    static void onRemoved(InstructionListener *, Instruction *) {}
    static void onRAUW(InstructionListener *Self, Instruction *Old,
                       Value *New) {
      CapturingListener *This = static_cast<CapturingListener *>(Self);
      This->OldRef = Old;
      This->NewRef = New;
    }

  public:
    CapturingListener(Function &F, Value *&Old, Value *&New)
        : InstructionListener(F, &onRemoved, &onRAUW), OldRef(Old),
          NewRef(New) {}
  };

  CapturingListener Listener(*F, ReceivedOld, ReceivedNew);

  Value *Replacement = F->getArg(0);
  Add->replaceAllUsesWith(Replacement);

  EXPECT_EQ(ReceivedOld, Add);
  EXPECT_EQ(ReceivedNew, Replacement);
}

TEST(InstructionListenerTest, RAUWWithoutCallback) {
  LLVMContext C;
  Module M("test", C);
  Function *F = createFunction(M, "f");
  BasicBlock *BB = BasicBlock::Create(C, "entry", F);
  IRBuilder<> Builder(BB);

  Instruction *Add =
      cast<Instruction>(Builder.CreateAdd(F->getArg(0), F->getArg(1)));
  Builder.CreateRet(Add);

  DenseSet<const Value *> UniformValues;
  UniformValues.insert(Add);

  class DeletionOnlyListener : public InstructionListener {
    DenseSet<const Value *> &Ref;

    static void onRemoved(InstructionListener *Self, Instruction *I) {
      static_cast<DeletionOnlyListener *>(Self)->Ref.erase(I);
    }

  public:
    DeletionOnlyListener(Function &F, DenseSet<const Value *> &S)
        : InstructionListener(F, &onRemoved), Ref(S) {}
  };

  DeletionOnlyListener Listener(*F, UniformValues);

  // RAUW should not crash even without a RAUW callback.
  EXPECT_TRUE(UniformValues.contains(Add));
  Add->replaceAllUsesWith(F->getArg(0));
  // Set not updated since no RAUW callback — only deletion updates it.
  EXPECT_TRUE(UniformValues.contains(Add));
}

TEST(InstructionListenerTest, RAUWPerFunctionIsolation) {
  LLVMContext C;
  Module M("test", C);
  Function *F1 = createFunction(M, "f1");
  Function *F2 = createFunction(M, "f2");

  BasicBlock *BB1 = BasicBlock::Create(C, "entry", F1);
  BasicBlock *BB2 = BasicBlock::Create(C, "entry", F2);

  IRBuilder<> B1(BB1);
  Instruction *Add1 =
      cast<Instruction>(B1.CreateAdd(F1->getArg(0), F1->getArg(1)));
  B1.CreateRet(Add1);

  IRBuilder<> B2(BB2);
  Instruction *Add2 =
      cast<Instruction>(B2.CreateAdd(F2->getArg(0), F2->getArg(1)));
  B2.CreateRet(Add2);

  DenseSet<const Value *> UniformValues1;
  DenseSet<const Value *> UniformValues2;
  UniformValues1.insert(Add1);
  UniformValues2.insert(Add2);

  TrackingListener L1(*F1, UniformValues1);
  TrackingListener L2(*F2, UniformValues2);

  // RAUW in F1 should not affect F2's listener.
  Add1->replaceAllUsesWith(F1->getArg(0));
  EXPECT_FALSE(UniformValues1.contains(Add1));
  EXPECT_TRUE(UniformValues2.contains(Add2));
}

/// Demonstrates replacing per-value CallbackVH with a single listener.
/// This mirrors the pattern used by LazyValueInfo (LVIValueHandle) and
/// similar analyses that cache per-value results in a DenseMap.
TEST(InstructionListenerTest, CacheInvalidationPattern) {
  LLVMContext C;
  Module M("test", C);
  Function *F = createFunction(M, "f");
  BasicBlock *BB = BasicBlock::Create(C, "entry", F);
  IRBuilder<> Builder(BB);

  Instruction *Add1 =
      cast<Instruction>(Builder.CreateAdd(F->getArg(0), F->getArg(1)));
  Instruction *Add2 = cast<Instruction>(Builder.CreateAdd(F->getArg(0), Add1));
  Builder.CreateRet(F->getArg(0));

  // Simulate an analysis cache: maps Instruction* -> some cached result.
  DenseMap<const Value *, int> Cache;
  Cache[Add1] = 42;
  Cache[Add2] = 99;

  class CacheListener : public InstructionListener {
    DenseMap<const Value *, int> &CacheRef;

    static void onRemoved(InstructionListener *Self, Instruction *I) {
      static_cast<CacheListener *>(Self)->CacheRef.erase(I);
    }

    static void onRAUW(InstructionListener *Self, Instruction *Old, Value *) {
      static_cast<CacheListener *>(Self)->CacheRef.erase(Old);
    }

  public:
    CacheListener(Function &F, DenseMap<const Value *, int> &C)
        : InstructionListener(F, &onRemoved, &onRAUW), CacheRef(C) {}
  };

  CacheListener Listener(*F, Cache);

  EXPECT_EQ(Cache.size(), 2u);

  // RAUW Add1 with arg0 — cache entry for Add1 is invalidated.
  Add1->replaceAllUsesWith(F->getArg(0));
  EXPECT_EQ(Cache.size(), 1u);
  EXPECT_FALSE(Cache.contains(Add1));
  EXPECT_TRUE(Cache.contains(Add2));

  // Erase Add1 — already removed from cache by RAUW, no effect.
  Add1->eraseFromParent();
  EXPECT_EQ(Cache.size(), 1u);

  // Erase Add2 — triggers deletion callback, removes from cache.
  Add2->eraseFromParent();
  EXPECT_TRUE(Cache.empty());
}

TEST(InstructionListenerTest, FunctionErasedBeforeListener) {
  LLVMContext C;
  Module M("test", C);
  Function *F = createFunction(M, "f");
  BasicBlock *BB = BasicBlock::Create(C, "entry", F);
  IRBuilder<> Builder(BB);
  Builder.CreateRet(F->getArg(0));

  DenseSet<const Value *> UniformValues;
  TrackingListener Listener(*F, UniformValues);

  EXPECT_TRUE(F->hasInstructionListeners());
  F->eraseFromParent();
  // Listener outlives the Function. Its destructor must not crash.
}

} // end anonymous namespace
