//===- MemorySSA.cpp - Unit tests for MemorySSA ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/MemorySSAUpdater.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace llvm;

const static char DLString[] = "e-i64:64-f80:128-n8:16:32:64-S128";

/// There's a lot of common setup between these tests. This fixture helps reduce
/// that. Tests should mock up a function, store it in F, and then call
/// setupAnalyses().
class MemorySSATest : public testing::Test {
protected:
  // N.B. Many of these members depend on each other (e.g. the Module depends on
  // the Context, etc.). So, order matters here (and in TestAnalyses).
  LLVMContext C;
  Module M;
  IRBuilder<> B;
  DataLayout DL;
  TargetLibraryInfoImpl TLII;
  TargetLibraryInfo TLI;
  Function *F;

  // Things that we need to build after the function is created.
  struct TestAnalyses {
    DominatorTree DT;
    AssumptionCache AC;
    AAResults AA;
    BasicAAResult BAA;
    // We need to defer MSSA construction until AA is *entirely* set up, which
    // requires calling addAAResult. Hence, we just use a pointer here.
    std::unique_ptr<MemorySSA> MSSA;
    MemorySSAWalker *Walker;

    TestAnalyses(MemorySSATest &Test)
        : DT(*Test.F), AC(*Test.F), AA(Test.TLI),
          BAA(Test.DL, *Test.F, Test.TLI, AC, &DT) {
      AA.addAAResult(BAA);
      MSSA = std::make_unique<MemorySSA>(*Test.F, &AA, &DT);
      Walker = MSSA->getWalker();
    }
  };

  std::unique_ptr<TestAnalyses> Analyses;

  void setupAnalyses() {
    assert(F);
    Analyses.reset(new TestAnalyses(*this));
  }

public:
  MemorySSATest()
      : M("MemorySSATest", C), B(C), DL(DLString), TLI(TLII), F(nullptr) {}
};

TEST_F(MemorySSATest, CreateALoad) {
  // We create a diamond where there is a store on one side, and then after
  // building MemorySSA, create a load after the merge point, and use it to test
  // updating by creating an access for the load.
  F = Function::Create(FunctionType::get(B.getVoidTy(), {B.getPtrTy()}, false),
                       GlobalValue::ExternalLinkage, "F", &M);
  BasicBlock *Entry(BasicBlock::Create(C, "", F));
  BasicBlock *Left(BasicBlock::Create(C, "", F));
  BasicBlock *Right(BasicBlock::Create(C, "", F));
  BasicBlock *Merge(BasicBlock::Create(C, "", F));
  B.SetInsertPoint(Entry);
  B.CreateCondBr(B.getTrue(), Left, Right);
  B.SetInsertPoint(Left);
  Argument *PointerArg = &*F->arg_begin();
  B.CreateStore(B.getInt8(16), PointerArg);
  BranchInst::Create(Merge, Left);
  BranchInst::Create(Merge, Right);

  setupAnalyses();
  MemorySSA &MSSA = *Analyses->MSSA;
  MemorySSAUpdater Updater(&MSSA);
  // Add the load
  B.SetInsertPoint(Merge);
  LoadInst *LoadInst = B.CreateLoad(B.getInt8Ty(), PointerArg);

  // MemoryPHI should already exist.
  MemoryPhi *MP = MSSA.getMemoryAccess(Merge);
  EXPECT_NE(MP, nullptr);

  // Create the load memory acccess
  MemoryUse *LoadAccess = cast<MemoryUse>(Updater.createMemoryAccessInBB(
      LoadInst, MP, Merge, MemorySSA::Beginning));
  MemoryAccess *DefiningAccess = LoadAccess->getDefiningAccess();
  EXPECT_TRUE(isa<MemoryPhi>(DefiningAccess));
  MSSA.verifyMemorySSA();
}
TEST_F(MemorySSATest, CreateLoadsAndStoreUpdater) {
  // We create a diamond, then build memoryssa with no memory accesses, and
  // incrementally update it by inserting a store in the, entry, a load in the
  // merge point, then a store in the branch, another load in the merge point,
  // and then a store in the entry.
  F = Function::Create(FunctionType::get(B.getVoidTy(), {B.getPtrTy()}, false),
                       GlobalValue::ExternalLinkage, "F", &M);
  BasicBlock *Entry(BasicBlock::Create(C, "", F));
  BasicBlock *Left(BasicBlock::Create(C, "", F));
  BasicBlock *Right(BasicBlock::Create(C, "", F));
  BasicBlock *Merge(BasicBlock::Create(C, "", F));
  B.SetInsertPoint(Entry);
  B.CreateCondBr(B.getTrue(), Left, Right);
  B.SetInsertPoint(Left, Left->begin());
  Argument *PointerArg = &*F->arg_begin();
  B.SetInsertPoint(Left);
  B.CreateBr(Merge);
  B.SetInsertPoint(Right);
  B.CreateBr(Merge);

  setupAnalyses();
  MemorySSA &MSSA = *Analyses->MSSA;
  MemorySSAUpdater Updater(&MSSA);
  // Add the store
  B.SetInsertPoint(Entry, Entry->begin());
  StoreInst *EntryStore = B.CreateStore(B.getInt8(16), PointerArg);
  MemoryAccess *EntryStoreAccess = Updater.createMemoryAccessInBB(
      EntryStore, nullptr, Entry, MemorySSA::Beginning);
  Updater.insertDef(cast<MemoryDef>(EntryStoreAccess));

  // Add the load
  B.SetInsertPoint(Merge, Merge->begin());
  LoadInst *FirstLoad = B.CreateLoad(B.getInt8Ty(), PointerArg);

  // MemoryPHI should not already exist.
  MemoryPhi *MP = MSSA.getMemoryAccess(Merge);
  EXPECT_EQ(MP, nullptr);

  // Create the load memory access
  MemoryUse *FirstLoadAccess = cast<MemoryUse>(Updater.createMemoryAccessInBB(
      FirstLoad, nullptr, Merge, MemorySSA::Beginning));
  Updater.insertUse(FirstLoadAccess);
  // Should just have a load using the entry access, because it should discover
  // the phi is trivial
  EXPECT_EQ(FirstLoadAccess->getDefiningAccess(), EntryStoreAccess);

  // Create a store on the left
  // Add the store
  B.SetInsertPoint(Left, Left->begin());
  StoreInst *LeftStore = B.CreateStore(B.getInt8(16), PointerArg);
  MemoryAccess *LeftStoreAccess = Updater.createMemoryAccessInBB(
      LeftStore, nullptr, Left, MemorySSA::Beginning);
  Updater.insertDef(cast<MemoryDef>(LeftStoreAccess), false);

  // MemoryPHI should exist after adding LeftStore.
  MP = MSSA.getMemoryAccess(Merge);
  EXPECT_NE(MP, nullptr);

  // Add the second load
  B.SetInsertPoint(Merge, Merge->begin());
  LoadInst *SecondLoad = B.CreateLoad(B.getInt8Ty(), PointerArg);

  // Create the load memory access
  MemoryUse *SecondLoadAccess = cast<MemoryUse>(Updater.createMemoryAccessInBB(
      SecondLoad, nullptr, Merge, MemorySSA::Beginning));
  Updater.insertUse(SecondLoadAccess);
  // Now the load should be a phi of the entry store and the left store
  MemoryPhi *MergePhi =
      dyn_cast<MemoryPhi>(SecondLoadAccess->getDefiningAccess());
  EXPECT_NE(MergePhi, nullptr);
  EXPECT_EQ(MergePhi->getIncomingValue(0), EntryStoreAccess);
  EXPECT_EQ(MergePhi->getIncomingValue(1), LeftStoreAccess);
  // Now create a store below the existing one in the entry
  B.SetInsertPoint(Entry, --Entry->end());
  StoreInst *SecondEntryStore = B.CreateStore(B.getInt8(16), PointerArg);
  MemoryAccess *SecondEntryStoreAccess = Updater.createMemoryAccessInBB(
      SecondEntryStore, nullptr, Entry, MemorySSA::End);
  // Insert it twice just to test renaming
  Updater.insertDef(cast<MemoryDef>(SecondEntryStoreAccess), false);
  EXPECT_NE(FirstLoadAccess->getDefiningAccess(), MergePhi);
  Updater.insertDef(cast<MemoryDef>(SecondEntryStoreAccess), true);
  EXPECT_EQ(FirstLoadAccess->getDefiningAccess(), MergePhi);
  // and make sure the phi below it got updated, despite being blocks away
  MergePhi = dyn_cast<MemoryPhi>(SecondLoadAccess->getDefiningAccess());
  EXPECT_NE(MergePhi, nullptr);
  EXPECT_EQ(MergePhi->getIncomingValue(0), SecondEntryStoreAccess);
  EXPECT_EQ(MergePhi->getIncomingValue(1), LeftStoreAccess);
  MSSA.verifyMemorySSA();
}

TEST_F(MemorySSATest, CreateALoadUpdater) {
  // We create a diamond, then build memoryssa with no memory accesses, and
  // incrementally update it by inserting a store in one of the branches, and a
  // load in the merge point
  F = Function::Create(FunctionType::get(B.getVoidTy(), {B.getPtrTy()}, false),
                       GlobalValue::ExternalLinkage, "F", &M);
  BasicBlock *Entry(BasicBlock::Create(C, "", F));
  BasicBlock *Left(BasicBlock::Create(C, "", F));
  BasicBlock *Right(BasicBlock::Create(C, "", F));
  BasicBlock *Merge(BasicBlock::Create(C, "", F));
  B.SetInsertPoint(Entry);
  B.CreateCondBr(B.getTrue(), Left, Right);
  B.SetInsertPoint(Left, Left->begin());
  Argument *PointerArg = &*F->arg_begin();
  B.SetInsertPoint(Left);
  B.CreateBr(Merge);
  B.SetInsertPoint(Right);
  B.CreateBr(Merge);

  setupAnalyses();
  MemorySSA &MSSA = *Analyses->MSSA;
  MemorySSAUpdater Updater(&MSSA);
  B.SetInsertPoint(Left, Left->begin());
  // Add the store
  StoreInst *SI = B.CreateStore(B.getInt8(16), PointerArg);
  MemoryAccess *StoreAccess =
      Updater.createMemoryAccessInBB(SI, nullptr, Left, MemorySSA::Beginning);
  Updater.insertDef(cast<MemoryDef>(StoreAccess));

  // MemoryPHI should be created when inserting the def
  MemoryPhi *MP = MSSA.getMemoryAccess(Merge);
  EXPECT_NE(MP, nullptr);

  // Add the load
  B.SetInsertPoint(Merge, Merge->begin());
  LoadInst *LoadInst = B.CreateLoad(B.getInt8Ty(), PointerArg);

  // Create the load memory acccess
  MemoryUse *LoadAccess = cast<MemoryUse>(Updater.createMemoryAccessInBB(
      LoadInst, nullptr, Merge, MemorySSA::Beginning));
  Updater.insertUse(LoadAccess);
  MemoryAccess *DefiningAccess = LoadAccess->getDefiningAccess();
  EXPECT_TRUE(isa<MemoryPhi>(DefiningAccess));
  MSSA.verifyMemorySSA();
}

TEST_F(MemorySSATest, SinkLoad) {
  F = Function::Create(FunctionType::get(B.getVoidTy(), {B.getPtrTy()}, false),
                       GlobalValue::ExternalLinkage, "F", &M);
  BasicBlock *Entry(BasicBlock::Create(C, "", F));
  BasicBlock *Left(BasicBlock::Create(C, "", F));
  BasicBlock *Right(BasicBlock::Create(C, "", F));
  BasicBlock *Merge(BasicBlock::Create(C, "", F));
  B.SetInsertPoint(Entry);
  B.CreateCondBr(B.getTrue(), Left, Right);
  B.SetInsertPoint(Left, Left->begin());
  Argument *PointerArg = &*F->arg_begin();
  B.SetInsertPoint(Left);
  B.CreateBr(Merge);
  B.SetInsertPoint(Right);
  B.CreateBr(Merge);

  // Load in left block
  B.SetInsertPoint(Left, Left->begin());
  LoadInst *LoadInst1 = B.CreateLoad(B.getInt8Ty(), PointerArg);
  // Store in merge block
  B.SetInsertPoint(Merge, Merge->begin());
  B.CreateStore(B.getInt8(16), PointerArg);

  setupAnalyses();
  MemorySSA &MSSA = *Analyses->MSSA;
  MemorySSAUpdater Updater(&MSSA);

  // Mimic sinking of a load:
  // - clone load
  // - insert in "exit" block
  // - insert in mssa
  // - remove from original block

  LoadInst *LoadInstClone = cast<LoadInst>(LoadInst1->clone());
  LoadInstClone->insertInto(Merge, Merge->begin());
  MemoryAccess * NewLoadAccess =
      Updater.createMemoryAccessInBB(LoadInstClone, nullptr,
                                     LoadInstClone->getParent(),
                                     MemorySSA::Beginning);
  Updater.insertUse(cast<MemoryUse>(NewLoadAccess));
  MSSA.verifyMemorySSA();
  Updater.removeMemoryAccess(MSSA.getMemoryAccess(LoadInst1));
  MSSA.verifyMemorySSA();
}

TEST_F(MemorySSATest, MoveAStore) {
  // We create a diamond where there is a in the entry, a store on one side, and
  // a load at the end.  After building MemorySSA, we test updating by moving
  // the store from the side block to the entry block. This destroys the old
  // access.
  F = Function::Create(FunctionType::get(B.getVoidTy(), {B.getPtrTy()}, false),
                       GlobalValue::ExternalLinkage, "F", &M);
  BasicBlock *Entry(BasicBlock::Create(C, "", F));
  BasicBlock *Left(BasicBlock::Create(C, "", F));
  BasicBlock *Right(BasicBlock::Create(C, "", F));
  BasicBlock *Merge(BasicBlock::Create(C, "", F));
  B.SetInsertPoint(Entry);
  Argument *PointerArg = &*F->arg_begin();
  StoreInst *EntryStore = B.CreateStore(B.getInt8(16), PointerArg);
  B.CreateCondBr(B.getTrue(), Left, Right);
  B.SetInsertPoint(Left);
  StoreInst *SideStore = B.CreateStore(B.getInt8(16), PointerArg);
  BranchInst::Create(Merge, Left);
  BranchInst::Create(Merge, Right);
  B.SetInsertPoint(Merge);
  B.CreateLoad(B.getInt8Ty(), PointerArg);
  setupAnalyses();
  MemorySSA &MSSA = *Analyses->MSSA;
  MemorySSAUpdater Updater(&MSSA);
  // Move the store
  SideStore->moveBefore(Entry->getTerminator());
  MemoryAccess *EntryStoreAccess = MSSA.getMemoryAccess(EntryStore);
  MemoryAccess *SideStoreAccess = MSSA.getMemoryAccess(SideStore);
  MemoryAccess *NewStoreAccess = Updater.createMemoryAccessAfter(
      SideStore, EntryStoreAccess, EntryStoreAccess);
  EntryStoreAccess->replaceAllUsesWith(NewStoreAccess);
  Updater.removeMemoryAccess(SideStoreAccess);
  MSSA.verifyMemorySSA();
}

TEST_F(MemorySSATest, MoveAStoreUpdater) {
  // We create a diamond where there is a in the entry, a store on one side, and
  // a load at the end.  After building MemorySSA, we test updating by moving
  // the store from the side block to the entry block.  This destroys the old
  // access.
  F = Function::Create(FunctionType::get(B.getVoidTy(), {B.getPtrTy()}, false),
                       GlobalValue::ExternalLinkage, "F", &M);
  BasicBlock *Entry(BasicBlock::Create(C, "", F));
  BasicBlock *Left(BasicBlock::Create(C, "", F));
  BasicBlock *Right(BasicBlock::Create(C, "", F));
  BasicBlock *Merge(BasicBlock::Create(C, "", F));
  B.SetInsertPoint(Entry);
  Argument *PointerArg = &*F->arg_begin();
  StoreInst *EntryStore = B.CreateStore(B.getInt8(16), PointerArg);
  B.CreateCondBr(B.getTrue(), Left, Right);
  B.SetInsertPoint(Left);
  auto *SideStore = B.CreateStore(B.getInt8(16), PointerArg);
  BranchInst::Create(Merge, Left);
  BranchInst::Create(Merge, Right);
  B.SetInsertPoint(Merge);
  auto *MergeLoad = B.CreateLoad(B.getInt8Ty(), PointerArg);
  setupAnalyses();
  MemorySSA &MSSA = *Analyses->MSSA;
  MemorySSAUpdater Updater(&MSSA);

  // Move the store
  SideStore->moveBefore(Entry->getTerminator());
  auto *EntryStoreAccess = MSSA.getMemoryAccess(EntryStore);
  auto *SideStoreAccess = MSSA.getMemoryAccess(SideStore);
  auto *NewStoreAccess = Updater.createMemoryAccessAfter(
      SideStore, EntryStoreAccess, EntryStoreAccess);
  // Before, the load will point to a phi of the EntryStore and SideStore.
  auto *LoadAccess = cast<MemoryUse>(MSSA.getMemoryAccess(MergeLoad));
  EXPECT_TRUE(isa<MemoryPhi>(LoadAccess->getDefiningAccess()));
  MemoryPhi *MergePhi = cast<MemoryPhi>(LoadAccess->getDefiningAccess());
  EXPECT_EQ(MergePhi->getIncomingValue(1), EntryStoreAccess);
  EXPECT_EQ(MergePhi->getIncomingValue(0), SideStoreAccess);
  Updater.removeMemoryAccess(SideStoreAccess);
  Updater.insertDef(cast<MemoryDef>(NewStoreAccess));
  // After it's a phi of the new side store access.
  EXPECT_EQ(MergePhi->getIncomingValue(0), NewStoreAccess);
  EXPECT_EQ(MergePhi->getIncomingValue(1), NewStoreAccess);
  MSSA.verifyMemorySSA();
}

TEST_F(MemorySSATest, MoveAStoreUpdaterMove) {
  // We create a diamond where there is a in the entry, a store on one side, and
  // a load at the end.  After building MemorySSA, we test updating by moving
  // the store from the side block to the entry block.  This does not destroy
  // the old access.
  F = Function::Create(FunctionType::get(B.getVoidTy(), {B.getPtrTy()}, false),
                       GlobalValue::ExternalLinkage, "F", &M);
  BasicBlock *Entry(BasicBlock::Create(C, "", F));
  BasicBlock *Left(BasicBlock::Create(C, "", F));
  BasicBlock *Right(BasicBlock::Create(C, "", F));
  BasicBlock *Merge(BasicBlock::Create(C, "", F));
  B.SetInsertPoint(Entry);
  Argument *PointerArg = &*F->arg_begin();
  StoreInst *EntryStore = B.CreateStore(B.getInt8(16), PointerArg);
  B.CreateCondBr(B.getTrue(), Left, Right);
  B.SetInsertPoint(Left);
  auto *SideStore = B.CreateStore(B.getInt8(16), PointerArg);
  BranchInst::Create(Merge, Left);
  BranchInst::Create(Merge, Right);
  B.SetInsertPoint(Merge);
  auto *MergeLoad = B.CreateLoad(B.getInt8Ty(), PointerArg);
  setupAnalyses();
  MemorySSA &MSSA = *Analyses->MSSA;
  MemorySSAUpdater Updater(&MSSA);

  // Move the store
  auto *EntryStoreAccess = MSSA.getMemoryAccess(EntryStore);
  auto *SideStoreAccess = MSSA.getMemoryAccess(SideStore);
  // Before, the load will point to a phi of the EntryStore and SideStore.
  auto *LoadAccess = cast<MemoryUse>(MSSA.getMemoryAccess(MergeLoad));
  EXPECT_TRUE(isa<MemoryPhi>(LoadAccess->getDefiningAccess()));
  MemoryPhi *MergePhi = cast<MemoryPhi>(LoadAccess->getDefiningAccess());
  EXPECT_EQ(MergePhi->getIncomingValue(1), EntryStoreAccess);
  EXPECT_EQ(MergePhi->getIncomingValue(0), SideStoreAccess);
  SideStore->moveBefore(*EntryStore->getParent(), ++EntryStore->getIterator());
  Updater.moveAfter(SideStoreAccess, EntryStoreAccess);
  // After, it's a phi of the side store.
  EXPECT_EQ(MergePhi->getIncomingValue(0), SideStoreAccess);
  EXPECT_EQ(MergePhi->getIncomingValue(1), SideStoreAccess);

  MSSA.verifyMemorySSA();
}

TEST_F(MemorySSATest, MoveAStoreAllAround) {
  // We create a diamond where there is a in the entry, a store on one side, and
  // a load at the end.  After building MemorySSA, we test updating by moving
  // the store from the side block to the entry block, then to the other side
  // block, then to before the load.  This does not destroy the old access.
  F = Function::Create(FunctionType::get(B.getVoidTy(), {B.getPtrTy()}, false),
                       GlobalValue::ExternalLinkage, "F", &M);
  BasicBlock *Entry(BasicBlock::Create(C, "", F));
  BasicBlock *Left(BasicBlock::Create(C, "", F));
  BasicBlock *Right(BasicBlock::Create(C, "", F));
  BasicBlock *Merge(BasicBlock::Create(C, "", F));
  B.SetInsertPoint(Entry);
  Argument *PointerArg = &*F->arg_begin();
  StoreInst *EntryStore = B.CreateStore(B.getInt8(16), PointerArg);
  B.CreateCondBr(B.getTrue(), Left, Right);
  B.SetInsertPoint(Left);
  auto *SideStore = B.CreateStore(B.getInt8(16), PointerArg);
  BranchInst::Create(Merge, Left);
  BranchInst::Create(Merge, Right);
  B.SetInsertPoint(Merge);
  auto *MergeLoad = B.CreateLoad(B.getInt8Ty(), PointerArg);
  setupAnalyses();
  MemorySSA &MSSA = *Analyses->MSSA;
  MemorySSAUpdater Updater(&MSSA);

  // Move the store
  auto *EntryStoreAccess = MSSA.getMemoryAccess(EntryStore);
  auto *SideStoreAccess = MSSA.getMemoryAccess(SideStore);
  // Before, the load will point to a phi of the EntryStore and SideStore.
  auto *LoadAccess = cast<MemoryUse>(MSSA.getMemoryAccess(MergeLoad));
  EXPECT_TRUE(isa<MemoryPhi>(LoadAccess->getDefiningAccess()));
  MemoryPhi *MergePhi = cast<MemoryPhi>(LoadAccess->getDefiningAccess());
  EXPECT_EQ(MergePhi->getIncomingValue(1), EntryStoreAccess);
  EXPECT_EQ(MergePhi->getIncomingValue(0), SideStoreAccess);
  // Move the store before the entry store
  SideStore->moveBefore(*EntryStore->getParent(), EntryStore->getIterator());
  Updater.moveBefore(SideStoreAccess, EntryStoreAccess);
  // After, it's a phi of the entry store.
  EXPECT_EQ(MergePhi->getIncomingValue(0), EntryStoreAccess);
  EXPECT_EQ(MergePhi->getIncomingValue(1), EntryStoreAccess);
  MSSA.verifyMemorySSA();
  // Now move the store to the right branch
  SideStore->moveBefore(*Right, Right->begin());
  Updater.moveToPlace(SideStoreAccess, Right, MemorySSA::Beginning);
  MSSA.verifyMemorySSA();
  EXPECT_EQ(MergePhi->getIncomingValue(0), EntryStoreAccess);
  EXPECT_EQ(MergePhi->getIncomingValue(1), SideStoreAccess);
  // Now move it before the load
  SideStore->moveBefore(MergeLoad);
  Updater.moveBefore(SideStoreAccess, LoadAccess);
  EXPECT_EQ(MergePhi->getIncomingValue(0), EntryStoreAccess);
  EXPECT_EQ(MergePhi->getIncomingValue(1), EntryStoreAccess);
  MSSA.verifyMemorySSA();
}

TEST_F(MemorySSATest, RemoveAPhi) {
  // We create a diamond where there is a store on one side, and then a load
  // after the merge point.  This enables us to test a bunch of different
  // removal cases.
  F = Function::Create(FunctionType::get(B.getVoidTy(), {B.getPtrTy()}, false),
                       GlobalValue::ExternalLinkage, "F", &M);
  BasicBlock *Entry(BasicBlock::Create(C, "", F));
  BasicBlock *Left(BasicBlock::Create(C, "", F));
  BasicBlock *Right(BasicBlock::Create(C, "", F));
  BasicBlock *Merge(BasicBlock::Create(C, "", F));
  B.SetInsertPoint(Entry);
  B.CreateCondBr(B.getTrue(), Left, Right);
  B.SetInsertPoint(Left);
  Argument *PointerArg = &*F->arg_begin();
  StoreInst *StoreInst = B.CreateStore(B.getInt8(16), PointerArg);
  BranchInst::Create(Merge, Left);
  BranchInst::Create(Merge, Right);
  B.SetInsertPoint(Merge);
  LoadInst *LoadInst = B.CreateLoad(B.getInt8Ty(), PointerArg);

  setupAnalyses();
  MemorySSA &MSSA = *Analyses->MSSA;
  MemorySSAUpdater Updater(&MSSA);

  // Before, the load will be a use of a phi<store, liveonentry>.
  MemoryUse *LoadAccess = cast<MemoryUse>(MSSA.getMemoryAccess(LoadInst));
  MemoryDef *StoreAccess = cast<MemoryDef>(MSSA.getMemoryAccess(StoreInst));
  MemoryAccess *DefiningAccess = LoadAccess->getDefiningAccess();
  EXPECT_TRUE(isa<MemoryPhi>(DefiningAccess));
  // Kill the store
  Updater.removeMemoryAccess(StoreAccess);
  MemoryPhi *MP = cast<MemoryPhi>(DefiningAccess);
  // Verify the phi ended up as liveonentry, liveonentry
  for (auto &Op : MP->incoming_values())
    EXPECT_TRUE(MSSA.isLiveOnEntryDef(cast<MemoryAccess>(Op.get())));
  // Replace the phi uses with the live on entry def
  MP->replaceAllUsesWith(MSSA.getLiveOnEntryDef());
  // Verify the load is now defined by liveOnEntryDef
  EXPECT_TRUE(MSSA.isLiveOnEntryDef(LoadAccess->getDefiningAccess()));
  // Remove the PHI
  Updater.removeMemoryAccess(MP);
  MSSA.verifyMemorySSA();
}

TEST_F(MemorySSATest, RemoveMemoryAccess) {
  // We create a diamond where there is a store on one side, and then a load
  // after the merge point.  This enables us to test a bunch of different
  // removal cases.
  F = Function::Create(FunctionType::get(B.getVoidTy(), {B.getPtrTy()}, false),
                       GlobalValue::ExternalLinkage, "F", &M);
  BasicBlock *Entry(BasicBlock::Create(C, "", F));
  BasicBlock *Left(BasicBlock::Create(C, "", F));
  BasicBlock *Right(BasicBlock::Create(C, "", F));
  BasicBlock *Merge(BasicBlock::Create(C, "", F));
  B.SetInsertPoint(Entry);
  B.CreateCondBr(B.getTrue(), Left, Right);
  B.SetInsertPoint(Left);
  Argument *PointerArg = &*F->arg_begin();
  StoreInst *StoreInst = B.CreateStore(B.getInt8(16), PointerArg);
  BranchInst::Create(Merge, Left);
  BranchInst::Create(Merge, Right);
  B.SetInsertPoint(Merge);
  LoadInst *LoadInst = B.CreateLoad(B.getInt8Ty(), PointerArg);

  setupAnalyses();
  MemorySSA &MSSA = *Analyses->MSSA;
  MemorySSAWalker *Walker = Analyses->Walker;
  MemorySSAUpdater Updater(&MSSA);

  // Before, the load will be a use of a phi<store, liveonentry>. It should be
  // the same after.
  MemoryUse *LoadAccess = cast<MemoryUse>(MSSA.getMemoryAccess(LoadInst));
  MemoryDef *StoreAccess = cast<MemoryDef>(MSSA.getMemoryAccess(StoreInst));
  MemoryAccess *DefiningAccess = LoadAccess->getDefiningAccess();
  EXPECT_TRUE(isa<MemoryPhi>(DefiningAccess));
  // The load is currently clobbered by one of the phi arguments, so the walker
  // should determine the clobbering access as the phi.
  EXPECT_EQ(DefiningAccess, Walker->getClobberingMemoryAccess(LoadInst));
  Updater.removeMemoryAccess(StoreAccess);
  MSSA.verifyMemorySSA();
  // After the removeaccess, let's see if we got the right accesses
  // The load should still point to the phi ...
  EXPECT_EQ(DefiningAccess, LoadAccess->getDefiningAccess());
  // but we should now get live on entry for the clobbering definition of the
  // load, since it will walk past the phi node since every argument is the
  // same.
  // XXX: This currently requires either removing the phi or resetting optimized
  // on the load

  EXPECT_FALSE(
      MSSA.isLiveOnEntryDef(Walker->getClobberingMemoryAccess(LoadInst)));
  // If we reset optimized, we get live on entry.
  LoadAccess->resetOptimized();
  EXPECT_TRUE(
      MSSA.isLiveOnEntryDef(Walker->getClobberingMemoryAccess(LoadInst)));
  // The phi should now be a two entry phi with two live on entry defs.
  for (const auto &Op : DefiningAccess->operands()) {
    MemoryAccess *Operand = cast<MemoryAccess>(&*Op);
    EXPECT_TRUE(MSSA.isLiveOnEntryDef(Operand));
  }

  // Now we try to remove the single valued phi
  Updater.removeMemoryAccess(DefiningAccess);
  MSSA.verifyMemorySSA();
  // Now the load should be a load of live on entry.
  EXPECT_TRUE(MSSA.isLiveOnEntryDef(LoadAccess->getDefiningAccess()));
}

// We had a bug with caching where the walker would report MemoryDef#3's clobber
// (below) was MemoryDef#1.
//
// define void @F(i8*) {
//   %A = alloca i8, i8 1
// ; 1 = MemoryDef(liveOnEntry)
//   store i8 0, i8* %A
// ; 2 = MemoryDef(1)
//   store i8 1, i8* %A
// ; 3 = MemoryDef(2)
//   store i8 2, i8* %A
// }
TEST_F(MemorySSATest, TestTripleStore) {
  F = Function::Create(FunctionType::get(B.getVoidTy(), {}, false),
                       GlobalValue::ExternalLinkage, "F", &M);
  B.SetInsertPoint(BasicBlock::Create(C, "", F));
  Type *Int8 = Type::getInt8Ty(C);
  Value *Alloca = B.CreateAlloca(Int8, ConstantInt::get(Int8, 1), "A");
  StoreInst *S1 = B.CreateStore(ConstantInt::get(Int8, 0), Alloca);
  StoreInst *S2 = B.CreateStore(ConstantInt::get(Int8, 1), Alloca);
  StoreInst *S3 = B.CreateStore(ConstantInt::get(Int8, 2), Alloca);

  setupAnalyses();
  MemorySSA &MSSA = *Analyses->MSSA;
  MemorySSAWalker *Walker = Analyses->Walker;

  unsigned I = 0;
  for (StoreInst *V : {S1, S2, S3}) {
    // Everything should be clobbered by its defining access
    MemoryAccess *DefiningAccess = MSSA.getMemoryAccess(V)->getDefiningAccess();
    MemoryAccess *WalkerClobber = Walker->getClobberingMemoryAccess(V);
    EXPECT_EQ(DefiningAccess, WalkerClobber)
        << "Store " << I << " doesn't have the correct clobbering access";
    // EXPECT_EQ expands such that if we increment I above, it won't get
    // incremented except when we try to print the error message.
    ++I;
  }
}

// ...And fixing the above bug made it obvious that, when walking, MemorySSA's
// walker was caching the initial node it walked. This was fine (albeit
// mostly redundant) unless the initial node being walked is a clobber for the
// query. In that case, we'd cache that the node clobbered itself.
TEST_F(MemorySSATest, TestStoreAndLoad) {
  F = Function::Create(FunctionType::get(B.getVoidTy(), {}, false),
                       GlobalValue::ExternalLinkage, "F", &M);
  B.SetInsertPoint(BasicBlock::Create(C, "", F));
  Type *Int8 = Type::getInt8Ty(C);
  Value *Alloca = B.CreateAlloca(Int8, ConstantInt::get(Int8, 1), "A");
  Instruction *SI = B.CreateStore(ConstantInt::get(Int8, 0), Alloca);
  Instruction *LI = B.CreateLoad(Int8, Alloca);

  setupAnalyses();
  MemorySSA &MSSA = *Analyses->MSSA;
  MemorySSAWalker *Walker = Analyses->Walker;

  MemoryAccess *LoadClobber = Walker->getClobberingMemoryAccess(LI);
  EXPECT_EQ(LoadClobber, MSSA.getMemoryAccess(SI));
  EXPECT_TRUE(MSSA.isLiveOnEntryDef(Walker->getClobberingMemoryAccess(SI)));
}

// Another bug (related to the above two fixes): It was noted that, given the
// following code:
// ; 1 = MemoryDef(liveOnEntry)
// store i8 0, i8* %1
//
// ...A query to getClobberingMemoryAccess(MemoryAccess*, MemoryLocation) would
// hand back the store (correctly). A later call to
// getClobberingMemoryAccess(const Instruction*) would also hand back the store
// (incorrectly; it should return liveOnEntry).
//
// This test checks that repeated calls to either function returns what they're
// meant to.
TEST_F(MemorySSATest, TestStoreDoubleQuery) {
  F = Function::Create(FunctionType::get(B.getVoidTy(), {}, false),
                       GlobalValue::ExternalLinkage, "F", &M);
  B.SetInsertPoint(BasicBlock::Create(C, "", F));
  Type *Int8 = Type::getInt8Ty(C);
  Value *Alloca = B.CreateAlloca(Int8, ConstantInt::get(Int8, 1), "A");
  StoreInst *SI = B.CreateStore(ConstantInt::get(Int8, 0), Alloca);

  setupAnalyses();
  MemorySSA &MSSA = *Analyses->MSSA;
  MemorySSAWalker *Walker = Analyses->Walker;

  MemoryAccess *StoreAccess = MSSA.getMemoryAccess(SI);
  MemoryLocation StoreLoc = MemoryLocation::get(SI);
  MemoryAccess *Clobber =
      Walker->getClobberingMemoryAccess(StoreAccess, StoreLoc);
  MemoryAccess *LiveOnEntry = Walker->getClobberingMemoryAccess(SI);

  EXPECT_EQ(Clobber, StoreAccess);
  EXPECT_TRUE(MSSA.isLiveOnEntryDef(LiveOnEntry));

  // Try again (with entries in the cache already) for good measure...
  Clobber = Walker->getClobberingMemoryAccess(StoreAccess, StoreLoc);
  LiveOnEntry = Walker->getClobberingMemoryAccess(SI);
  EXPECT_EQ(Clobber, StoreAccess);
  EXPECT_TRUE(MSSA.isLiveOnEntryDef(LiveOnEntry));
}

// Bug: During phi optimization, the walker wouldn't cache to the proper result
// in the farthest-walked BB.
//
// Specifically, it would assume that whatever we walked to was a clobber.
// "Whatever we walked to" isn't a clobber if we hit a cache entry.
//
// ...So, we need a test case that looks like:
//    A
//   / \
//  B   |
//   \ /
//    C
//
// Where, when we try to optimize a thing in 'C', a blocker is found in 'B'.
// The walk must determine that the blocker exists by using cache entries *while
// walking* 'B'.
TEST_F(MemorySSATest, PartialWalkerCacheWithPhis) {
  F = Function::Create(FunctionType::get(B.getVoidTy(), {}, false),
                       GlobalValue::ExternalLinkage, "F", &M);
  B.SetInsertPoint(BasicBlock::Create(C, "A", F));
  Type *Int8 = Type::getInt8Ty(C);
  Constant *One = ConstantInt::get(Int8, 1);
  Constant *Zero = ConstantInt::get(Int8, 0);
  Value *AllocA = B.CreateAlloca(Int8, One, "a");
  Value *AllocB = B.CreateAlloca(Int8, One, "b");
  BasicBlock *IfThen = BasicBlock::Create(C, "B", F);
  BasicBlock *IfEnd = BasicBlock::Create(C, "C", F);

  B.CreateCondBr(UndefValue::get(Type::getInt1Ty(C)), IfThen, IfEnd);

  B.SetInsertPoint(IfThen);
  Instruction *FirstStore = B.CreateStore(Zero, AllocA);
  B.CreateStore(Zero, AllocB);
  Instruction *ALoad0 = B.CreateLoad(Int8, AllocA, "");
  Instruction *BStore = B.CreateStore(Zero, AllocB);
  // Due to use optimization/etc. we make a store to A, which is removed after
  // we build MSSA. This helps keep the test case simple-ish.
  Instruction *KillStore = B.CreateStore(Zero, AllocA);
  Instruction *ALoad = B.CreateLoad(Int8, AllocA, "");
  B.CreateBr(IfEnd);

  B.SetInsertPoint(IfEnd);
  Instruction *BelowPhi = B.CreateStore(Zero, AllocA);

  setupAnalyses();
  MemorySSA &MSSA = *Analyses->MSSA;
  MemorySSAWalker *Walker = Analyses->Walker;
  MemorySSAUpdater Updater(&MSSA);

  // Kill `KillStore`; it exists solely so that the load after it won't be
  // optimized to FirstStore.
  Updater.removeMemoryAccess(MSSA.getMemoryAccess(KillStore));
  KillStore->eraseFromParent();
  auto *ALoadMA = cast<MemoryUse>(MSSA.getMemoryAccess(ALoad));
  EXPECT_EQ(ALoadMA->getDefiningAccess(), MSSA.getMemoryAccess(BStore));

  // Populate the cache for the store to AllocB directly after FirstStore. It
  // should point to something in block B (so something in D can't be optimized
  // to it).
  MemoryAccess *Load0Clobber = Walker->getClobberingMemoryAccess(ALoad0);
  EXPECT_EQ(MSSA.getMemoryAccess(FirstStore), Load0Clobber);

  // If the bug exists, this will introduce a bad cache entry for %a on BStore.
  // It will point to the store to %b after FirstStore. This only happens during
  // phi optimization.
  MemoryAccess *BottomClobber = Walker->getClobberingMemoryAccess(BelowPhi);
  MemoryAccess *Phi = MSSA.getMemoryAccess(IfEnd);
  EXPECT_EQ(BottomClobber, Phi);

  // This query will first check the cache for {%a, BStore}. It should point to
  // FirstStore, not to the store after FirstStore.
  MemoryAccess *UseClobber = Walker->getClobberingMemoryAccess(ALoad);
  EXPECT_EQ(UseClobber, MSSA.getMemoryAccess(FirstStore));
}

// Test that our walker properly handles loads with the invariant group
// attribute. It's a bit hacky, since we add the invariant attribute *after*
// building MSSA. Otherwise, the use optimizer will optimize it for us, which
// isn't what we want.
// FIXME: It may be easier/cleaner to just add an 'optimize uses?' flag to MSSA.
TEST_F(MemorySSATest, WalkerInvariantLoadOpt) {
  F = Function::Create(FunctionType::get(B.getVoidTy(), {}, false),
                       GlobalValue::ExternalLinkage, "F", &M);
  B.SetInsertPoint(BasicBlock::Create(C, "", F));
  Type *Int8 = Type::getInt8Ty(C);
  Constant *One = ConstantInt::get(Int8, 1);
  Value *AllocA = B.CreateAlloca(Int8, One, "");

  Instruction *Store = B.CreateStore(One, AllocA);
  Instruction *Load = B.CreateLoad(Int8, AllocA);

  setupAnalyses();
  MemorySSA &MSSA = *Analyses->MSSA;
  MemorySSAWalker *Walker = Analyses->Walker;

  auto *LoadMA = cast<MemoryUse>(MSSA.getMemoryAccess(Load));
  auto *StoreMA = cast<MemoryDef>(MSSA.getMemoryAccess(Store));
  EXPECT_EQ(LoadMA->getDefiningAccess(), StoreMA);

  // ...At the time of writing, no cache should exist for LoadMA. Be a bit
  // flexible to future changes.
  Walker->invalidateInfo(LoadMA);
  Load->setMetadata(LLVMContext::MD_invariant_load, MDNode::get(C, {}));

  MemoryAccess *LoadClobber = Walker->getClobberingMemoryAccess(LoadMA);
  EXPECT_EQ(LoadClobber, MSSA.getLiveOnEntryDef());
}

// Test loads get reoptimized properly by the walker.
TEST_F(MemorySSATest, WalkerReopt) {
  F = Function::Create(FunctionType::get(B.getVoidTy(), {}, false),
                       GlobalValue::ExternalLinkage, "F", &M);
  B.SetInsertPoint(BasicBlock::Create(C, "", F));
  Type *Int8 = Type::getInt8Ty(C);
  Value *AllocaA = B.CreateAlloca(Int8, ConstantInt::get(Int8, 1), "A");
  Instruction *SIA = B.CreateStore(ConstantInt::get(Int8, 0), AllocaA);
  Value *AllocaB = B.CreateAlloca(Int8, ConstantInt::get(Int8, 1), "B");
  Instruction *SIB = B.CreateStore(ConstantInt::get(Int8, 0), AllocaB);
  Instruction *LIA = B.CreateLoad(Int8, AllocaA);

  setupAnalyses();
  MemorySSA &MSSA = *Analyses->MSSA;
  MemorySSAWalker *Walker = Analyses->Walker;
  MemorySSAUpdater Updater(&MSSA);

  MemoryAccess *LoadClobber = Walker->getClobberingMemoryAccess(LIA);
  MemoryUse *LoadAccess = cast<MemoryUse>(MSSA.getMemoryAccess(LIA));
  EXPECT_EQ(LoadClobber, MSSA.getMemoryAccess(SIA));
  EXPECT_TRUE(MSSA.isLiveOnEntryDef(Walker->getClobberingMemoryAccess(SIA)));
  Updater.removeMemoryAccess(LoadAccess);

  // Create the load memory access pointing to an unoptimized place.
  MemoryUse *NewLoadAccess = cast<MemoryUse>(Updater.createMemoryAccessInBB(
      LIA, MSSA.getMemoryAccess(SIB), LIA->getParent(), MemorySSA::End));
  // This should it cause it to be optimized
  EXPECT_EQ(Walker->getClobberingMemoryAccess(NewLoadAccess), LoadClobber);
  EXPECT_EQ(NewLoadAccess->getDefiningAccess(), LoadClobber);
}

// Test out MemorySSAUpdater::moveBefore
TEST_F(MemorySSATest, MoveAboveMemoryDef) {
  F = Function::Create(FunctionType::get(B.getVoidTy(), {}, false),
                       GlobalValue::ExternalLinkage, "F", &M);
  B.SetInsertPoint(BasicBlock::Create(C, "", F));

  Type *Int8 = Type::getInt8Ty(C);
  Value *A = B.CreateAlloca(Int8, ConstantInt::get(Int8, 1), "A");
  Value *B_ = B.CreateAlloca(Int8, ConstantInt::get(Int8, 1), "B");
  Value *C = B.CreateAlloca(Int8, ConstantInt::get(Int8, 1), "C");

  StoreInst *StoreA0 = B.CreateStore(ConstantInt::get(Int8, 0), A);
  StoreInst *StoreB = B.CreateStore(ConstantInt::get(Int8, 0), B_);
  LoadInst *LoadB = B.CreateLoad(Int8, B_);
  StoreInst *StoreA1 = B.CreateStore(ConstantInt::get(Int8, 4), A);
  StoreInst *StoreC = B.CreateStore(ConstantInt::get(Int8, 4), C);
  StoreInst *StoreA2 = B.CreateStore(ConstantInt::get(Int8, 4), A);
  LoadInst *LoadC = B.CreateLoad(Int8, C);

  setupAnalyses();
  MemorySSA &MSSA = *Analyses->MSSA;
  MemorySSAWalker &Walker = *Analyses->Walker;

  MemorySSAUpdater Updater(&MSSA);
  StoreC->moveBefore(StoreB);
  Updater.moveBefore(cast<MemoryDef>(MSSA.getMemoryAccess(StoreC)),
                     cast<MemoryDef>(MSSA.getMemoryAccess(StoreB)));

  MSSA.verifyMemorySSA();

  EXPECT_EQ(MSSA.getMemoryAccess(StoreB)->getDefiningAccess(),
            MSSA.getMemoryAccess(StoreC));
  EXPECT_EQ(MSSA.getMemoryAccess(StoreC)->getDefiningAccess(),
            MSSA.getMemoryAccess(StoreA0));
  EXPECT_EQ(MSSA.getMemoryAccess(StoreA2)->getDefiningAccess(),
            MSSA.getMemoryAccess(StoreA1));
  EXPECT_EQ(Walker.getClobberingMemoryAccess(LoadB),
            MSSA.getMemoryAccess(StoreB));
  EXPECT_EQ(Walker.getClobberingMemoryAccess(LoadC),
            MSSA.getMemoryAccess(StoreC));

  // exercise block numbering
  EXPECT_TRUE(MSSA.locallyDominates(MSSA.getMemoryAccess(StoreC),
                                    MSSA.getMemoryAccess(StoreB)));
  EXPECT_TRUE(MSSA.locallyDominates(MSSA.getMemoryAccess(StoreA1),
                                    MSSA.getMemoryAccess(StoreA2)));
}

TEST_F(MemorySSATest, Irreducible) {
  // Create the equivalent of
  // x = something
  // if (...)
  //    goto second_loop_entry
  // while (...) {
  // second_loop_entry:
  // }
  // use(x)

  SmallVector<PHINode *, 8> Inserted;
  IRBuilder<> B(C);
  F = Function::Create(FunctionType::get(B.getVoidTy(), {B.getPtrTy()}, false),
                       GlobalValue::ExternalLinkage, "F", &M);

  // Make blocks
  BasicBlock *IfBB = BasicBlock::Create(C, "if", F);
  BasicBlock *LoopStartBB = BasicBlock::Create(C, "loopstart", F);
  BasicBlock *LoopMainBB = BasicBlock::Create(C, "loopmain", F);
  BasicBlock *AfterLoopBB = BasicBlock::Create(C, "afterloop", F);
  B.SetInsertPoint(IfBB);
  B.CreateCondBr(B.getTrue(), LoopMainBB, LoopStartBB);
  B.SetInsertPoint(LoopStartBB);
  B.CreateBr(LoopMainBB);
  B.SetInsertPoint(LoopMainBB);
  B.CreateCondBr(B.getTrue(), LoopStartBB, AfterLoopBB);
  B.SetInsertPoint(AfterLoopBB);
  Argument *FirstArg = &*F->arg_begin();
  setupAnalyses();
  MemorySSA &MSSA = *Analyses->MSSA;
  MemorySSAUpdater Updater(&MSSA);
  // Create the load memory acccess
  LoadInst *LoadInst = B.CreateLoad(B.getInt8Ty(), FirstArg);
  MemoryUse *LoadAccess = cast<MemoryUse>(Updater.createMemoryAccessInBB(
      LoadInst, nullptr, AfterLoopBB, MemorySSA::Beginning));
  Updater.insertUse(LoadAccess);
  MSSA.verifyMemorySSA();
}

TEST_F(MemorySSATest, MoveToBeforeLiveOnEntryInvalidatesCache) {
  // Create:
  //   %1 = alloca i8
  //   ; 1 = MemoryDef(liveOnEntry)
  //   store i8 0, i8* %1
  //   ; 2 = MemoryDef(1)
  //   store i8 0, i8* %1
  //
  // ...And be sure that MSSA's caching doesn't give us `1` for the clobber of
  // `2` after `1` is removed.
  IRBuilder<> B(C);
  F = Function::Create(FunctionType::get(B.getVoidTy(), {B.getPtrTy()}, false),
                       GlobalValue::ExternalLinkage, "F", &M);

  BasicBlock *Entry = BasicBlock::Create(C, "if", F);
  B.SetInsertPoint(Entry);

  Value *A = B.CreateAlloca(B.getInt8Ty());
  StoreInst *StoreA = B.CreateStore(B.getInt8(0), A);
  StoreInst *StoreB = B.CreateStore(B.getInt8(0), A);

  setupAnalyses();

  MemorySSA &MSSA = *Analyses->MSSA;

  auto *DefA = cast<MemoryDef>(MSSA.getMemoryAccess(StoreA));
  auto *DefB = cast<MemoryDef>(MSSA.getMemoryAccess(StoreB));

  MemoryAccess *BClobber = MSSA.getWalker()->getClobberingMemoryAccess(DefB);
  ASSERT_EQ(DefA, BClobber);

  MemorySSAUpdater(&MSSA).removeMemoryAccess(DefA);
  StoreA->eraseFromParent();

  EXPECT_EQ(DefB->getDefiningAccess(), MSSA.getLiveOnEntryDef());

  EXPECT_EQ(MSSA.getWalker()->getClobberingMemoryAccess(DefB),
            MSSA.getLiveOnEntryDef())
      << "(DefA = " << DefA << ")";
}

TEST_F(MemorySSATest, RemovingDefInvalidatesCache) {
  // Create:
  //   %x = alloca i8
  //   %y = alloca i8
  //   ; 1 = MemoryDef(liveOnEntry)
  //   store i8 0, i8* %x
  //   ; 2 = MemoryDef(1)
  //   store i8 0, i8* %y
  //   ; 3 = MemoryDef(2)
  //   store i8 0, i8* %x
  //
  // And be sure that MSSA's caching handles the removal of def `1`
  // appropriately.
  IRBuilder<> B(C);
  F = Function::Create(FunctionType::get(B.getVoidTy(), {B.getPtrTy()}, false),
                       GlobalValue::ExternalLinkage, "F", &M);

  BasicBlock *Entry = BasicBlock::Create(C, "if", F);
  B.SetInsertPoint(Entry);

  Value *X = B.CreateAlloca(B.getInt8Ty());
  Value *Y = B.CreateAlloca(B.getInt8Ty());
  StoreInst *StoreX1 = B.CreateStore(B.getInt8(0), X);
  StoreInst *StoreY = B.CreateStore(B.getInt8(0), Y);
  StoreInst *StoreX2 = B.CreateStore(B.getInt8(0), X);

  setupAnalyses();

  MemorySSA &MSSA = *Analyses->MSSA;

  auto *DefX1 = cast<MemoryDef>(MSSA.getMemoryAccess(StoreX1));
  auto *DefY = cast<MemoryDef>(MSSA.getMemoryAccess(StoreY));
  auto *DefX2 = cast<MemoryDef>(MSSA.getMemoryAccess(StoreX2));

  EXPECT_EQ(DefX2->getDefiningAccess(), DefY);
  MemoryAccess *X2Clobber = MSSA.getWalker()->getClobberingMemoryAccess(DefX2);
  ASSERT_EQ(DefX1, X2Clobber);

  MemorySSAUpdater(&MSSA).removeMemoryAccess(DefX1);
  StoreX1->eraseFromParent();

  EXPECT_EQ(DefX2->getDefiningAccess(), DefY);
  EXPECT_EQ(MSSA.getWalker()->getClobberingMemoryAccess(DefX2),
            MSSA.getLiveOnEntryDef())
      << "(DefX1 = " << DefX1 << ")";
}

// Test Must alias for optimized defs.
TEST_F(MemorySSATest, TestStoreMustAlias) {
  F = Function::Create(FunctionType::get(B.getVoidTy(), {}, false),
                       GlobalValue::ExternalLinkage, "F", &M);
  B.SetInsertPoint(BasicBlock::Create(C, "", F));
  Type *Int8 = Type::getInt8Ty(C);
  Value *AllocaA = B.CreateAlloca(Int8, ConstantInt::get(Int8, 1), "A");
  Value *AllocaB = B.CreateAlloca(Int8, ConstantInt::get(Int8, 1), "B");
  StoreInst *SA1 = B.CreateStore(ConstantInt::get(Int8, 1), AllocaA);
  StoreInst *SB1 = B.CreateStore(ConstantInt::get(Int8, 1), AllocaB);
  StoreInst *SA2 = B.CreateStore(ConstantInt::get(Int8, 2), AllocaA);
  StoreInst *SB2 = B.CreateStore(ConstantInt::get(Int8, 2), AllocaB);
  StoreInst *SA3 = B.CreateStore(ConstantInt::get(Int8, 3), AllocaA);
  StoreInst *SB3 = B.CreateStore(ConstantInt::get(Int8, 3), AllocaB);

  setupAnalyses();
  MemorySSA &MSSA = *Analyses->MSSA;
  MemorySSAWalker *Walker = Analyses->Walker;

  unsigned I = 0;
  for (StoreInst *V : {SA1, SB1, SA2, SB2, SA3, SB3}) {
    MemoryDef *MemDef = dyn_cast_or_null<MemoryDef>(MSSA.getMemoryAccess(V));
    EXPECT_EQ(MemDef->isOptimized(), false)
        << "Store " << I << " is optimized from the start?";
    if (V == SA1)
      Walker->getClobberingMemoryAccess(V);
    else {
      MemoryAccess *Def = MemDef->getDefiningAccess();
      MemoryAccess *Clob = Walker->getClobberingMemoryAccess(V);
      EXPECT_NE(Def, Clob) << "Store " << I
                           << " has Defining Access equal to Clobbering Access";
    }
    EXPECT_EQ(MemDef->isOptimized(), true)
        << "Store " << I << " was not optimized";
    // EXPECT_EQ expands such that if we increment I above, it won't get
    // incremented except when we try to print the error message.
    ++I;
  }
}

// Test May alias for optimized defs.
TEST_F(MemorySSATest, TestStoreMayAlias) {
  F = Function::Create(
      FunctionType::get(B.getVoidTy(), {B.getPtrTy(), B.getPtrTy()}, false),
      GlobalValue::ExternalLinkage, "F", &M);
  B.SetInsertPoint(BasicBlock::Create(C, "", F));
  Type *Int8 = Type::getInt8Ty(C);
  auto *ArgIt = F->arg_begin();
  Argument *PointerA = &*ArgIt;
  Argument *PointerB = &*(++ArgIt);
  Value *AllocaC = B.CreateAlloca(Int8, ConstantInt::get(Int8, 1), "C");
  // Store into arg1, must alias because it's LOE => must
  StoreInst *SA1 = B.CreateStore(ConstantInt::get(Int8, 0), PointerA);
  // Store into arg2, may alias store to arg1 => may
  StoreInst *SB1 = B.CreateStore(ConstantInt::get(Int8, 1), PointerB);
  // Store into aloca, no alias with args, so must alias LOE => must
  StoreInst *SC1 = B.CreateStore(ConstantInt::get(Int8, 2), AllocaC);
  // Store into arg1, may alias store to arg2 => may
  StoreInst *SA2 = B.CreateStore(ConstantInt::get(Int8, 3), PointerA);
  // Store into arg2, may alias store to arg1 => may
  StoreInst *SB2 = B.CreateStore(ConstantInt::get(Int8, 4), PointerB);
  // Store into aloca, no alias with args, so must alias SC1 => must
  StoreInst *SC2 = B.CreateStore(ConstantInt::get(Int8, 5), AllocaC);
  // Store into arg2, must alias store to arg2 => must
  StoreInst *SB3 = B.CreateStore(ConstantInt::get(Int8, 6), PointerB);
  std::initializer_list<StoreInst *> Sts = {SA1, SB1, SC1, SA2, SB2, SC2, SB3};

  setupAnalyses();
  MemorySSA &MSSA = *Analyses->MSSA;
  MemorySSAWalker *Walker = Analyses->Walker;

  unsigned I = 0;
  for (StoreInst *V : Sts) {
    MemoryDef *MemDef = dyn_cast_or_null<MemoryDef>(MSSA.getMemoryAccess(V));
    EXPECT_EQ(MemDef->isOptimized(), false)
        << "Store " << I << " is optimized from the start?";
    ++I;
  }

  for (StoreInst *V : Sts)
    Walker->getClobberingMemoryAccess(V);

  I = 0;
  for (StoreInst *V : Sts) {
    MemoryDef *MemDef = dyn_cast_or_null<MemoryDef>(MSSA.getMemoryAccess(V));
    EXPECT_EQ(MemDef->isOptimized(), true)
        << "Store " << I << " was not optimized";
    // EXPECT_EQ expands such that if we increment I above, it won't get
    // incremented except when we try to print the error message.
    ++I;
  }
}

TEST_F(MemorySSATest, LifetimeMarkersAreClobbers) {
  // Example code:
  // define void @a(i8* %foo) {
  //   %bar = getelementptr i8, i8* %foo, i64 1
  //   %baz = getelementptr i8, i8* %foo, i64 2
  //   store i8 0, i8* %foo
  //   store i8 0, i8* %bar
  //   call void @llvm.lifetime.end.p0i8(i64 3, i8* %foo)
  //   call void @llvm.lifetime.start.p0i8(i64 3, i8* %foo)
  //   store i8 0, i8* %foo
  //   store i8 0, i8* %bar
  //   call void @llvm.memset.p0i8(i8* %baz, i8 0, i64 1)
  //   ret void
  // }
  //
  // Patterns like this are possible after inlining; the stores to %foo and %bar
  // should both be clobbered by the lifetime.start call if they're dominated by
  // it.

  IRBuilder<> B(C);
  F = Function::Create(FunctionType::get(B.getVoidTy(), {B.getPtrTy()}, false),
                       GlobalValue::ExternalLinkage, "F", &M);

  // Make blocks
  BasicBlock *Entry = BasicBlock::Create(C, "entry", F);

  B.SetInsertPoint(Entry);
  Value *Foo = &*F->arg_begin();

  Value *Bar = B.CreatePtrAdd(Foo, B.getInt64(1), "bar");
  Value *Baz = B.CreatePtrAdd(Foo, B.getInt64(2), "baz");

  B.CreateStore(B.getInt8(0), Foo);
  B.CreateStore(B.getInt8(0), Bar);

  auto GetLifetimeIntrinsic = [&](Intrinsic::ID ID) {
    return Intrinsic::getDeclaration(&M, ID, {Foo->getType()});
  };

  B.CreateCall(GetLifetimeIntrinsic(Intrinsic::lifetime_end),
               {B.getInt64(3), Foo});
  Instruction *LifetimeStart = B.CreateCall(
      GetLifetimeIntrinsic(Intrinsic::lifetime_start), {B.getInt64(3), Foo});

  Instruction *FooStore = B.CreateStore(B.getInt8(0), Foo);
  Instruction *BarStore = B.CreateStore(B.getInt8(0), Bar);
  Instruction *BazMemSet = B.CreateMemSet(Baz, B.getInt8(0), 1, Align(1));

  setupAnalyses();
  MemorySSA &MSSA = *Analyses->MSSA;

  MemoryAccess *LifetimeStartAccess = MSSA.getMemoryAccess(LifetimeStart);
  ASSERT_NE(LifetimeStartAccess, nullptr);

  MemoryAccess *FooAccess = MSSA.getMemoryAccess(FooStore);
  ASSERT_NE(FooAccess, nullptr);

  MemoryAccess *BarAccess = MSSA.getMemoryAccess(BarStore);
  ASSERT_NE(BarAccess, nullptr);

  MemoryAccess *BazAccess = MSSA.getMemoryAccess(BazMemSet);
  ASSERT_NE(BazAccess, nullptr);

  MemoryAccess *FooClobber =
      MSSA.getWalker()->getClobberingMemoryAccess(FooAccess);
  EXPECT_EQ(FooClobber, LifetimeStartAccess);

  MemoryAccess *BarClobber =
      MSSA.getWalker()->getClobberingMemoryAccess(BarAccess);
  EXPECT_EQ(BarClobber, LifetimeStartAccess);

  MemoryAccess *BazClobber =
      MSSA.getWalker()->getClobberingMemoryAccess(BazAccess);
  EXPECT_EQ(BazClobber, LifetimeStartAccess);

  MemoryAccess *LifetimeStartClobber =
      MSSA.getWalker()->getClobberingMemoryAccess(
          LifetimeStartAccess, MemoryLocation::getAfter(Foo));
  EXPECT_EQ(LifetimeStartClobber, LifetimeStartAccess);
}

TEST_F(MemorySSATest, DefOptimizationsAreInvalidatedOnMoving) {
  IRBuilder<> B(C);
  F = Function::Create(FunctionType::get(B.getVoidTy(), {B.getInt1Ty()}, false),
                       GlobalValue::ExternalLinkage, "F", &M);

  // Make a CFG like
  //     entry
  //      / \
  //     a   b
  //      \ /
  //       c
  //
  // Put a def in A and a def in B, move the def from A -> B, observe as the
  // optimization is invalidated.
  BasicBlock *Entry = BasicBlock::Create(C, "entry", F);
  BasicBlock *BlockA = BasicBlock::Create(C, "a", F);
  BasicBlock *BlockB = BasicBlock::Create(C, "b", F);
  BasicBlock *BlockC = BasicBlock::Create(C, "c", F);

  B.SetInsertPoint(Entry);
  Type *Int8 = Type::getInt8Ty(C);
  Value *Alloca = B.CreateAlloca(Int8, ConstantInt::get(Int8, 1), "alloc");
  StoreInst *StoreEntry = B.CreateStore(B.getInt8(0), Alloca);
  B.CreateCondBr(B.getTrue(), BlockA, BlockB);

  B.SetInsertPoint(BlockA);
  StoreInst *StoreA = B.CreateStore(B.getInt8(1), Alloca);
  B.CreateBr(BlockC);

  B.SetInsertPoint(BlockB);
  StoreInst *StoreB = B.CreateStore(B.getInt8(2), Alloca);
  B.CreateBr(BlockC);

  B.SetInsertPoint(BlockC);
  B.CreateUnreachable();

  setupAnalyses();
  MemorySSA &MSSA = *Analyses->MSSA;

  auto *AccessEntry = cast<MemoryDef>(MSSA.getMemoryAccess(StoreEntry));
  auto *StoreAEntry = cast<MemoryDef>(MSSA.getMemoryAccess(StoreA));
  auto *StoreBEntry = cast<MemoryDef>(MSSA.getMemoryAccess(StoreB));

  ASSERT_EQ(MSSA.getWalker()->getClobberingMemoryAccess(StoreAEntry),
            AccessEntry);
  ASSERT_TRUE(StoreAEntry->isOptimized());

  ASSERT_EQ(MSSA.getWalker()->getClobberingMemoryAccess(StoreBEntry),
            AccessEntry);
  ASSERT_TRUE(StoreBEntry->isOptimized());

  // Note that if we did InsertionPlace::Beginning, we don't go out of our way
  // to invalidate the cache for StoreBEntry. If the user wants to actually do
  // moves like these, it's up to them to ensure that nearby cache entries are
  // correctly invalidated (which, in general, requires walking all instructions
  // that the moved instruction dominates. So we probably shouldn't be doing
  // moves like this in general. Still, works as a test-case. ;) )
  MemorySSAUpdater(&MSSA).moveToPlace(StoreAEntry, BlockB,
                                      MemorySSA::InsertionPlace::End);
  ASSERT_FALSE(StoreAEntry->isOptimized());
  ASSERT_EQ(MSSA.getWalker()->getClobberingMemoryAccess(StoreAEntry),
            StoreBEntry);
}

TEST_F(MemorySSATest, TestOptimizedDefsAreProperUses) {
  F = Function::Create(
      FunctionType::get(B.getVoidTy(), {B.getPtrTy(), B.getPtrTy()}, false),
      GlobalValue::ExternalLinkage, "F", &M);
  B.SetInsertPoint(BasicBlock::Create(C, "", F));
  Type *Int8 = Type::getInt8Ty(C);
  Value *AllocA = B.CreateAlloca(Int8, ConstantInt::get(Int8, 1), "A");
  Value *AllocB = B.CreateAlloca(Int8, ConstantInt::get(Int8, 1), "B");

  StoreInst *StoreA = B.CreateStore(ConstantInt::get(Int8, 0), AllocA);
  StoreInst *StoreB = B.CreateStore(ConstantInt::get(Int8, 1), AllocB);
  StoreInst *StoreA2 = B.CreateStore(ConstantInt::get(Int8, 2), AllocA);

  setupAnalyses();
  MemorySSA &MSSA = *Analyses->MSSA;
  MemorySSAWalker *Walker = Analyses->Walker;

  // If these don't hold, there's no chance of the test result being useful.
  ASSERT_EQ(Walker->getClobberingMemoryAccess(StoreA),
            MSSA.getLiveOnEntryDef());
  ASSERT_EQ(Walker->getClobberingMemoryAccess(StoreB),
            MSSA.getLiveOnEntryDef());
  auto *StoreAAccess = cast<MemoryDef>(MSSA.getMemoryAccess(StoreA));
  auto *StoreA2Access = cast<MemoryDef>(MSSA.getMemoryAccess(StoreA2));
  ASSERT_EQ(Walker->getClobberingMemoryAccess(StoreA2), StoreAAccess);
  ASSERT_EQ(StoreA2Access->getOptimized(), StoreAAccess);

  auto *StoreBAccess = cast<MemoryDef>(MSSA.getMemoryAccess(StoreB));
  ASSERT_LT(StoreAAccess->getID(), StoreBAccess->getID());
  ASSERT_LT(StoreBAccess->getID(), StoreA2Access->getID());

  auto SortVecByID = [](std::vector<const MemoryDef *> &Defs) {
    llvm::sort(Defs, [](const MemoryDef *LHS, const MemoryDef *RHS) {
      return LHS->getID() < RHS->getID();
    });
  };

  auto SortedUserList = [&](const MemoryDef *MD) {
    std::vector<const MemoryDef *> Result;
    transform(MD->users(), std::back_inserter(Result),
              [](const User *U) { return cast<MemoryDef>(U); });
    SortVecByID(Result);
    return Result;
  };

  // Use std::vectors, since they have nice pretty-printing if the test fails.
  // Parens are necessary because EXPECT_EQ is a macro, and we have commas in
  // our init lists...
  EXPECT_EQ(SortedUserList(StoreAAccess),
            (std::vector<const MemoryDef *>{StoreBAccess, StoreA2Access}));

  EXPECT_EQ(SortedUserList(StoreBAccess),
            std::vector<const MemoryDef *>{StoreA2Access});

  // StoreAAccess should be present twice, since it uses liveOnEntry for both
  // its defining and optimized accesses. This is a bit awkward, and is not
  // relied upon anywhere at the moment. If this is painful, we can fix it.
  EXPECT_EQ(SortedUserList(cast<MemoryDef>(MSSA.getLiveOnEntryDef())),
            (std::vector<const MemoryDef *>{StoreAAccess, StoreAAccess,
                                            StoreBAccess}));
}

//   entry
//     |
//   header
//    / \
// body  |
//    \ /
//    exit
// header:
//  ; 1 = MemoryDef(liveOnEntry)
// body:
//  ; 2 = MemoryDef(1)
// exit:
//  ; 3 = MemoryPhi({body, 2}, {header, 1})
//  ; 4 = MemoryDef(3); optimized to 3, cannot optimize thorugh phi.
//  Insert edge: entry -> exit, check mssa Update is correct.
TEST_F(MemorySSATest, TestAddedEdgeToBlockWithPhiNotOpt) {
  F = Function::Create(FunctionType::get(B.getVoidTy(), {B.getPtrTy()}, false),
                       GlobalValue::ExternalLinkage, "F", &M);
  Argument *PointerArg = &*F->arg_begin();
  BasicBlock *Entry(BasicBlock::Create(C, "entry", F));
  BasicBlock *Header(BasicBlock::Create(C, "header", F));
  BasicBlock *Body(BasicBlock::Create(C, "body", F));
  BasicBlock *Exit(BasicBlock::Create(C, "exit", F));
  B.SetInsertPoint(Entry);
  BranchInst::Create(Header, Entry);
  B.SetInsertPoint(Header);
  B.CreateStore(B.getInt8(16), PointerArg);
  B.CreateCondBr(B.getTrue(), Exit, Body);
  B.SetInsertPoint(Body);
  B.CreateStore(B.getInt8(16), PointerArg);
  BranchInst::Create(Exit, Body);
  B.SetInsertPoint(Exit);
  StoreInst *S1 = B.CreateStore(B.getInt8(16), PointerArg);

  setupAnalyses();
  MemorySSA &MSSA = *Analyses->MSSA;
  MemorySSAWalker *Walker = Analyses->Walker;
  std::unique_ptr<MemorySSAUpdater> MSSAU =
      std::make_unique<MemorySSAUpdater>(&MSSA);

  MemoryPhi *Phi = MSSA.getMemoryAccess(Exit);
  EXPECT_EQ(Phi, Walker->getClobberingMemoryAccess(S1));

  // Alter CFG, add edge: entry -> exit
  Entry->getTerminator()->eraseFromParent();
  B.SetInsertPoint(Entry);
  B.CreateCondBr(B.getTrue(), Header, Exit);
  SmallVector<CFGUpdate, 1> Updates;
  Updates.push_back({cfg::UpdateKind::Insert, Entry, Exit});
  Analyses->DT.applyUpdates(Updates);
  MSSAU->applyInsertUpdates(Updates, Analyses->DT);
  EXPECT_EQ(Phi, Walker->getClobberingMemoryAccess(S1));
}

//   entry
//     |
//   header
//    / \
// body  |
//    \ /
//    exit
// header:
//  ; 1 = MemoryDef(liveOnEntry)
// body:
//  ; 2 = MemoryDef(1)
// exit:
//  ; 3 = MemoryPhi({body, 2}, {header, 1})
//  ; 4 = MemoryDef(3); optimize this to 1 now, added edge should invalidate
//  the optimized access.
//  Insert edge: entry -> exit, check mssa Update is correct.
TEST_F(MemorySSATest, TestAddedEdgeToBlockWithPhiOpt) {
  F = Function::Create(FunctionType::get(B.getVoidTy(), {B.getPtrTy()}, false),
                       GlobalValue::ExternalLinkage, "F", &M);
  Argument *PointerArg = &*F->arg_begin();
  Type *Int8 = Type::getInt8Ty(C);
  BasicBlock *Entry(BasicBlock::Create(C, "entry", F));
  BasicBlock *Header(BasicBlock::Create(C, "header", F));
  BasicBlock *Body(BasicBlock::Create(C, "body", F));
  BasicBlock *Exit(BasicBlock::Create(C, "exit", F));

  B.SetInsertPoint(Entry);
  Value *Alloca = B.CreateAlloca(Int8, ConstantInt::get(Int8, 1), "A");
  BranchInst::Create(Header, Entry);

  B.SetInsertPoint(Header);
  StoreInst *S1 = B.CreateStore(B.getInt8(16), PointerArg);
  B.CreateCondBr(B.getTrue(), Exit, Body);

  B.SetInsertPoint(Body);
  B.CreateStore(ConstantInt::get(Int8, 0), Alloca);
  BranchInst::Create(Exit, Body);

  B.SetInsertPoint(Exit);
  StoreInst *S2 = B.CreateStore(B.getInt8(16), PointerArg);

  setupAnalyses();
  MemorySSA &MSSA = *Analyses->MSSA;
  MemorySSAWalker *Walker = Analyses->Walker;
  std::unique_ptr<MemorySSAUpdater> MSSAU =
      std::make_unique<MemorySSAUpdater>(&MSSA);

  MemoryDef *DefS1 = cast<MemoryDef>(MSSA.getMemoryAccess(S1));
  EXPECT_EQ(DefS1, Walker->getClobberingMemoryAccess(S2));

  // Alter CFG, add edge: entry -> exit
  Entry->getTerminator()->eraseFromParent();
  B.SetInsertPoint(Entry);
  B.CreateCondBr(B.getTrue(), Header, Exit);
  SmallVector<CFGUpdate, 1> Updates;
  Updates.push_back({cfg::UpdateKind::Insert, Entry, Exit});
  Analyses->DT.applyUpdates(Updates);
  MSSAU->applyInsertUpdates(Updates, Analyses->DT);

  MemoryPhi *Phi = MSSA.getMemoryAccess(Exit);
  EXPECT_EQ(Phi, Walker->getClobberingMemoryAccess(S2));
}

//   entry
//    /  |
//   a   |
//  / \  |
//  b c  f
//  \ /  |
//   d   |
//    \ /
//     e
// f:
//  ; 1 = MemoryDef(liveOnEntry)
// e:
//  ; 2 = MemoryPhi({d, liveOnEntry}, {f, 1})
//
// Insert edge: f -> c, check update is correct.
// After update:
// f:
//  ; 1 = MemoryDef(liveOnEntry)
// c:
//  ; 3 = MemoryPhi({a, liveOnEntry}, {f, 1})
// d:
//  ; 4 = MemoryPhi({b, liveOnEntry}, {c, 3})
// e:
//  ; 2 = MemoryPhi({d, 4}, {f, 1})
TEST_F(MemorySSATest, TestAddedEdgeToBlockWithNoPhiAddNewPhis) {
  F = Function::Create(FunctionType::get(B.getVoidTy(), {B.getPtrTy()}, false),
                       GlobalValue::ExternalLinkage, "F", &M);
  Argument *PointerArg = &*F->arg_begin();
  BasicBlock *Entry(BasicBlock::Create(C, "entry", F));
  BasicBlock *ABlock(BasicBlock::Create(C, "a", F));
  BasicBlock *BBlock(BasicBlock::Create(C, "b", F));
  BasicBlock *CBlock(BasicBlock::Create(C, "c", F));
  BasicBlock *DBlock(BasicBlock::Create(C, "d", F));
  BasicBlock *EBlock(BasicBlock::Create(C, "e", F));
  BasicBlock *FBlock(BasicBlock::Create(C, "f", F));

  B.SetInsertPoint(Entry);
  B.CreateCondBr(B.getTrue(), ABlock, FBlock);
  B.SetInsertPoint(ABlock);
  B.CreateCondBr(B.getTrue(), BBlock, CBlock);
  B.SetInsertPoint(BBlock);
  BranchInst::Create(DBlock, BBlock);
  B.SetInsertPoint(CBlock);
  BranchInst::Create(DBlock, CBlock);
  B.SetInsertPoint(DBlock);
  BranchInst::Create(EBlock, DBlock);
  B.SetInsertPoint(FBlock);
  B.CreateStore(B.getInt8(16), PointerArg);
  BranchInst::Create(EBlock, FBlock);

  setupAnalyses();
  MemorySSA &MSSA = *Analyses->MSSA;
  std::unique_ptr<MemorySSAUpdater> MSSAU =
      std::make_unique<MemorySSAUpdater>(&MSSA);

  // Alter CFG, add edge: f -> c
  FBlock->getTerminator()->eraseFromParent();
  B.SetInsertPoint(FBlock);
  B.CreateCondBr(B.getTrue(), CBlock, EBlock);
  SmallVector<CFGUpdate, 1> Updates;
  Updates.push_back({cfg::UpdateKind::Insert, FBlock, CBlock});
  Analyses->DT.applyUpdates(Updates);
  MSSAU->applyInsertUpdates(Updates, Analyses->DT);

  MemoryPhi *MPC = MSSA.getMemoryAccess(CBlock);
  EXPECT_NE(MPC, nullptr);
  MemoryPhi *MPD = MSSA.getMemoryAccess(DBlock);
  EXPECT_NE(MPD, nullptr);
  MemoryPhi *MPE = MSSA.getMemoryAccess(EBlock);
  EXPECT_EQ(MPD, MPE->getIncomingValueForBlock(DBlock));
}

TEST_F(MemorySSATest, TestCallClobber) {
  F = Function::Create(FunctionType::get(B.getVoidTy(), {B.getPtrTy()}, false),
                       GlobalValue::ExternalLinkage, "F", &M);

  Value *Pointer1 = &*F->arg_begin();
  BasicBlock *Entry(BasicBlock::Create(C, "", F));
  B.SetInsertPoint(Entry);
  Value *Pointer2 = B.CreatePtrAdd(Pointer1, B.getInt64(1));
  Instruction *StorePointer1 = B.CreateStore(B.getInt8(0), Pointer1);
  Instruction *StorePointer2 = B.CreateStore(B.getInt8(0), Pointer2);
  Instruction *MemSet = B.CreateMemSet(Pointer2, B.getInt8(0), 1, Align(1));

  setupAnalyses();
  MemorySSA &MSSA = *Analyses->MSSA;
  MemorySSAWalker *Walker = Analyses->Walker;

  MemoryUseOrDef *Store1Access = MSSA.getMemoryAccess(StorePointer1);
  MemoryUseOrDef *Store2Access = MSSA.getMemoryAccess(StorePointer2);
  MemoryUseOrDef *MemSetAccess = MSSA.getMemoryAccess(MemSet);

  MemoryAccess *Pointer1Clobber = Walker->getClobberingMemoryAccess(
      MemSetAccess, MemoryLocation(Pointer1, LocationSize::precise(1)));
  EXPECT_EQ(Pointer1Clobber, Store1Access);

  MemoryAccess *Pointer2Clobber = Walker->getClobberingMemoryAccess(
      MemSetAccess, MemoryLocation(Pointer2, LocationSize::precise(1)));
  EXPECT_EQ(Pointer2Clobber, MemSetAccess);

  MemoryAccess *MemSetClobber = Walker->getClobberingMemoryAccess(MemSetAccess);
  EXPECT_EQ(MemSetClobber, Store2Access);
}

TEST_F(MemorySSATest, TestLoadClobber) {
  F = Function::Create(FunctionType::get(B.getVoidTy(), {B.getPtrTy()}, false),
                       GlobalValue::ExternalLinkage, "F", &M);

  Value *Pointer1 = &*F->arg_begin();
  BasicBlock *Entry(BasicBlock::Create(C, "", F));
  B.SetInsertPoint(Entry);
  Value *Pointer2 = B.CreatePtrAdd(Pointer1, B.getInt64(1));
  Instruction *LoadPointer1 =
      B.CreateLoad(B.getInt8Ty(), Pointer1, /* Volatile */ true);
  Instruction *LoadPointer2 =
      B.CreateLoad(B.getInt8Ty(), Pointer2, /* Volatile */ true);

  setupAnalyses();
  MemorySSA &MSSA = *Analyses->MSSA;
  MemorySSAWalker *Walker = Analyses->Walker;

  MemoryUseOrDef *Load1Access = MSSA.getMemoryAccess(LoadPointer1);
  MemoryUseOrDef *Load2Access = MSSA.getMemoryAccess(LoadPointer2);

  // When providing a memory location, we should never return a load as the
  // clobber.
  MemoryAccess *Pointer1Clobber = Walker->getClobberingMemoryAccess(
      Load2Access, MemoryLocation(Pointer1, LocationSize::precise(1)));
  EXPECT_TRUE(MSSA.isLiveOnEntryDef(Pointer1Clobber));

  MemoryAccess *Pointer2Clobber = Walker->getClobberingMemoryAccess(
      Load2Access, MemoryLocation(Pointer2, LocationSize::precise(1)));
  EXPECT_TRUE(MSSA.isLiveOnEntryDef(Pointer2Clobber));

  MemoryAccess *Load2Clobber = Walker->getClobberingMemoryAccess(Load2Access);
  EXPECT_EQ(Load2Clobber, Load1Access);
}

// We want to test if the location information are retained
// when the IsGuaranteedLoopInvariant function handles a
// memory access referring to a pointer defined in the entry
// block, hence automatically guaranteed to be loop invariant.
TEST_F(MemorySSATest, TestLoopInvariantEntryBlockPointer) {
  SMDiagnostic E;
  auto LocalM =
      parseAssemblyString("define void @test(i64 %a0, i8* %a1, i1* %a2) {\n"
                          "entry:\n"
                          "%v0 = getelementptr i8, i8* %a1, i64 %a0\n"
                          "%v1 = bitcast i8* %v0 to i64*\n"
                          "%v2 = bitcast i8* %v0 to i32*\n"
                          "%v3 = load i1, i1* %a2\n"
                          "br i1 %v3, label %body, label %exit\n"
                          "body:\n"
                          "store i32 1, i32* %v2\n"
                          "br label %exit\n"
                          "exit:\n"
                          "store i64 0, i64* %v1\n"
                          "ret void\n"
                          "}",
                          E, C);
  ASSERT_TRUE(LocalM);
  F = LocalM->getFunction("test");
  ASSERT_TRUE(F);
  // Setup the analysis
  setupAnalyses();
  MemorySSA &MSSA = *Analyses->MSSA;
  // Find the exit block
  for (auto &BB : *F) {
    if (BB.getName() == "exit") {
      // Get the store instruction
      auto *SI = BB.getFirstNonPHI();
      // Get the memory access and location
      MemoryAccess *MA = MSSA.getMemoryAccess(SI);
      MemoryLocation ML = MemoryLocation::get(SI);
      // Use the 'upward_defs_iterator' which internally calls
      // IsGuaranteedLoopInvariant
      auto ItA = upward_defs_begin({MA, ML}, MSSA.getDomTree());
      auto ItB =
          upward_defs_begin({ItA->first, ItA->second}, MSSA.getDomTree());
      // Check if the location information have been retained
      EXPECT_TRUE(ItB->second.Size.isPrecise());
      EXPECT_TRUE(ItB->second.Size.hasValue());
      EXPECT_TRUE(ItB->second.Size.getValue() == 8);
    }
  }
}

TEST_F(MemorySSATest, TestInvariantGroup) {
  SMDiagnostic E;
  auto M = parseAssemblyString("declare void @f(i8*)\n"
                               "define i8 @test(i8* %p) {\n"
                               "entry:\n"
                               "  store i8 42, i8* %p, !invariant.group !0\n"
                               "  call void @f(i8* %p)\n"
                               "  %v = load i8, i8* %p, !invariant.group !0\n"
                               "  ret i8 %v\n"
                               "}\n"
                               "!0 = !{}",
                               E, C);
  ASSERT_TRUE(M);
  F = M->getFunction("test");
  ASSERT_TRUE(F);
  setupAnalyses();
  MemorySSA &MSSA = *Analyses->MSSA;
  MemorySSAWalker *Walker = Analyses->Walker;

  auto &BB = F->getEntryBlock();
  auto &SI = cast<StoreInst>(*BB.begin());
  auto &Call = cast<CallBase>(*std::next(BB.begin()));
  auto &LI = cast<LoadInst>(*std::next(std::next(BB.begin())));

  {
    MemoryAccess *SAccess = MSSA.getMemoryAccess(&SI);
    MemoryAccess *LAccess = MSSA.getMemoryAccess(&LI);
    MemoryAccess *SClobber = Walker->getClobberingMemoryAccess(SAccess);
    EXPECT_TRUE(MSSA.isLiveOnEntryDef(SClobber));
    MemoryAccess *LClobber = Walker->getClobberingMemoryAccess(LAccess);
    EXPECT_EQ(SAccess, LClobber);
  }

  // remove store and verify that the memory accesses still make sense
  MemorySSAUpdater Updater(&MSSA);
  Updater.removeMemoryAccess(&SI);
  SI.eraseFromParent();

  {
    MemoryAccess *CallAccess = MSSA.getMemoryAccess(&Call);
    MemoryAccess *LAccess = MSSA.getMemoryAccess(&LI);
    MemoryAccess *LClobber = Walker->getClobberingMemoryAccess(LAccess);
    EXPECT_EQ(CallAccess, LClobber);
  }
}

static BasicBlock *getBasicBlockByName(Function &F, StringRef Name) {
  for (BasicBlock &BB : F)
    if (BB.getName() == Name)
      return &BB;
  llvm_unreachable("Expected to find basic block!");
}

static Instruction *getInstructionByName(Function &F, StringRef Name) {
  for (BasicBlock &BB : F)
    for (Instruction &I : BB)
      if (I.getName() == Name)
        return &I;
  llvm_unreachable("Expected to find instruction!");
}

TEST_F(MemorySSATest, TestVisitedBlocks) {
  SMDiagnostic E;
  auto M = parseAssemblyString(
      "define void @test(i64* noalias %P, i64 %N) {\n"
      "preheader.n:\n"
      "  br label %header.n\n"
      "header.n:\n"
      "  %n = phi i64 [ 0, %preheader.n ], [ %inc.n, %latch.n ]\n"
      "  %guard.cond.i = icmp slt i64 0, %N\n"
      "  br i1 %guard.cond.i, label %header.i.check, label %other.i\n"
      "header.i.check:\n"
      "  br label %preheader.i\n"
      "preheader.i:\n"
      "  br label %header.i\n"
      "header.i:\n"
      "  %i = phi i64 [ 0, %preheader.i ], [ %inc.i, %header.i ]\n"
      "  %v1 = load i64, i64* %P, align 8\n"
      "  %v2 = load i64, i64* %P, align 8\n"
      "  %inc.i = add nsw i64 %i, 1\n"
      "  %cmp.i = icmp slt i64 %inc.i, %N\n"
      "  br i1 %cmp.i, label %header.i, label %exit.i\n"
      "exit.i:\n"
      "  br label %commonexit\n"
      "other.i:\n"
      "  br label %commonexit\n"
      "commonexit:\n"
      "  br label %latch.n\n"
      "latch.n:\n"
      "  %inc.n = add nsw i64 %n, 1\n"
      "  %cmp.n = icmp slt i64 %inc.n, %N\n"
      "  br i1 %cmp.n, label %header.n, label %exit.n\n"
      "exit.n:\n"
      "  ret void\n"
      "}\n",
      E, C);
  ASSERT_TRUE(M);
  F = M->getFunction("test");
  ASSERT_TRUE(F);
  setupAnalyses();
  MemorySSA &MSSA = *Analyses->MSSA;
  MemorySSAUpdater Updater(&MSSA);

  {
    // Move %v1 before the terminator of %header.i.check
    BasicBlock *BB = getBasicBlockByName(*F, "header.i.check");
    Instruction *LI = getInstructionByName(*F, "v1");
    LI->moveBefore(BB->getTerminator());
    if (MemoryUseOrDef *MUD = MSSA.getMemoryAccess(LI))
      Updater.moveToPlace(MUD, BB, MemorySSA::BeforeTerminator);

    // Change the termiantor of %header.i.check to `br label true, label
    // %preheader.i, label %other.i`
    BB->getTerminator()->eraseFromParent();
    ConstantInt *BoolTrue = ConstantInt::getTrue(F->getContext());
    BranchInst::Create(getBasicBlockByName(*F, "preheader.i"),
                       getBasicBlockByName(*F, "other.i"), BoolTrue, BB);
    SmallVector<DominatorTree::UpdateType, 4> DTUpdates;
    DTUpdates.push_back(DominatorTree::UpdateType(
        DominatorTree::Insert, BB, getBasicBlockByName(*F, "other.i")));
    Updater.applyUpdates(DTUpdates, Analyses->DT, true);
  }

  // After the first moveToPlace(), %other.i is in VisitedBlocks, even after
  // there is a new edge to %other.i, which makes the second moveToPlace()
  // traverse incorrectly.
  {
    // Move %v2 before the terminator of %preheader.i
    BasicBlock *BB = getBasicBlockByName(*F, "preheader.i");
    Instruction *LI = getInstructionByName(*F, "v2");
    LI->moveBefore(BB->getTerminator());
    // Check that there is no assertion of "Incomplete phi during partial
    // rename"
    if (MemoryUseOrDef *MUD = MSSA.getMemoryAccess(LI))
      Updater.moveToPlace(MUD, BB, MemorySSA::BeforeTerminator);
  }
}

TEST_F(MemorySSATest, TestNoDbgInsts) {
  SMDiagnostic E;
  auto M = parseAssemblyString(R"(
      define void @test() presplitcoroutine {
      entry:
        %i = alloca i32
        call void @llvm.dbg.declare(metadata ptr %i, metadata !6, metadata !DIExpression()), !dbg !10
        call void @llvm.dbg.value(metadata ptr %i, metadata !6, metadata !DIExpression()), !dbg !10
        ret void
      }

      declare void @llvm.dbg.declare(metadata, metadata, metadata) nocallback nofree nosync nounwind readnone speculatable willreturn
      declare void @llvm.dbg.value(metadata, metadata, metadata) nocallback nofree nosync nounwind readnone speculatable willreturn

      !llvm.dbg.cu = !{!0}

      !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 15.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !2, splitDebugInlining: false, nameTableKind: None)
      !1 = !DIFile(filename: "repro.cpp", directory: ".")
      !2 = !{}
      !3 = !{i32 7, !"Dwarf Version", i32 4}
      !4 = !{i32 2, !"Debug Info Version", i32 3}
      !5 = !{!"clang version 15.0.0"}
      !6 = !DILocalVariable(name: "i", scope: !7, file: !1, line: 24, type: !10)
      !7 = distinct !DILexicalBlock(scope: !8, file: !1, line: 23, column: 12)
      !8 = distinct !DISubprogram(name: "foo", linkageName: "_Z3foov", scope: !1, file: !1, line: 23, type: !9, scopeLine: 23, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
      !9 = !DISubroutineType(types: !2)
      !10 = !DILocation(line: 24, column: 7, scope: !7)
    )",
    E, C);
  ASSERT_TRUE(M);
  F = M->getFunction("test");
  ASSERT_TRUE(F);
  setupAnalyses();
  MemorySSA &MSSA = *Analyses->MSSA;
  MemorySSAUpdater Updater(&MSSA);

  BasicBlock &Entry = F->front();
  auto I = Entry.begin();
  Instruction *DbgDeclare = cast<Instruction>(I++);
  Instruction *DbgValue = cast<Instruction>(I++);
  ASSERT_EQ(MSSA.getMemoryAccess(DbgDeclare), nullptr);
  ASSERT_EQ(MSSA.getMemoryAccess(DbgValue), nullptr);
}
