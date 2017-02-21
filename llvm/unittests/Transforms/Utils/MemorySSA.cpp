//===- MemorySSA.cpp - Unit tests for MemorySSA ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include "llvm/Transforms/Utils/MemorySSA.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Transforms/Utils/MemorySSAUpdater.h"
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
          BAA(Test.DL, Test.TLI, AC, &DT) {
      AA.addAAResult(BAA);
      MSSA = make_unique<MemorySSA>(*Test.F, &AA, &DT);
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
  F = Function::Create(
      FunctionType::get(B.getVoidTy(), {B.getInt8PtrTy()}, false),
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
  // Add the load
  B.SetInsertPoint(Merge);
  LoadInst *LoadInst = B.CreateLoad(PointerArg);

  // MemoryPHI should already exist.
  MemoryPhi *MP = MSSA.getMemoryAccess(Merge);
  EXPECT_NE(MP, nullptr);

  // Create the load memory acccess
  MemoryUse *LoadAccess = cast<MemoryUse>(
      MSSA.createMemoryAccessInBB(LoadInst, MP, Merge, MemorySSA::Beginning));
  MemoryAccess *DefiningAccess = LoadAccess->getDefiningAccess();
  EXPECT_TRUE(isa<MemoryPhi>(DefiningAccess));
  MSSA.verifyMemorySSA();
}
TEST_F(MemorySSATest, CreateLoadsAndStoreUpdater) {
  // We create a diamond, then build memoryssa with no memory accesses, and
  // incrementally update it by inserting a store in the, entry, a load in the
  // merge point, then a store in the branch, another load in the merge point,
  // and then a store in the entry.
  F = Function::Create(
      FunctionType::get(B.getVoidTy(), {B.getInt8PtrTy()}, false),
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
  MemoryAccess *EntryStoreAccess = MSSA.createMemoryAccessInBB(
      EntryStore, nullptr, Entry, MemorySSA::Beginning);
  Updater.insertDef(cast<MemoryDef>(EntryStoreAccess));

  // Add the load
  B.SetInsertPoint(Merge, Merge->begin());
  LoadInst *FirstLoad = B.CreateLoad(PointerArg);

  // MemoryPHI should not already exist.
  MemoryPhi *MP = MSSA.getMemoryAccess(Merge);
  EXPECT_EQ(MP, nullptr);

  // Create the load memory access
  MemoryUse *FirstLoadAccess = cast<MemoryUse>(MSSA.createMemoryAccessInBB(
      FirstLoad, nullptr, Merge, MemorySSA::Beginning));
  Updater.insertUse(FirstLoadAccess);
  // Should just have a load using the entry access, because it should discover
  // the phi is trivial
  EXPECT_EQ(FirstLoadAccess->getDefiningAccess(), EntryStoreAccess);

  // Create a store on the left
  // Add the store
  B.SetInsertPoint(Left, Left->begin());
  StoreInst *LeftStore = B.CreateStore(B.getInt8(16), PointerArg);
  MemoryAccess *LeftStoreAccess = MSSA.createMemoryAccessInBB(
      LeftStore, nullptr, Left, MemorySSA::Beginning);
  Updater.insertDef(cast<MemoryDef>(LeftStoreAccess), false);
  // We don't touch existing loads, so we need to create a new one to get a phi
  // Add the second load
  B.SetInsertPoint(Merge, Merge->begin());
  LoadInst *SecondLoad = B.CreateLoad(PointerArg);

  // MemoryPHI should not already exist.
  MP = MSSA.getMemoryAccess(Merge);
  EXPECT_EQ(MP, nullptr);

  // Create the load memory access
  MemoryUse *SecondLoadAccess = cast<MemoryUse>(MSSA.createMemoryAccessInBB(
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
  MemoryAccess *SecondEntryStoreAccess = MSSA.createMemoryAccessInBB(
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
  F = Function::Create(
      FunctionType::get(B.getVoidTy(), {B.getInt8PtrTy()}, false),
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
      MSSA.createMemoryAccessInBB(SI, nullptr, Left, MemorySSA::Beginning);
  Updater.insertDef(cast<MemoryDef>(StoreAccess));

  // Add the load
  B.SetInsertPoint(Merge, Merge->begin());
  LoadInst *LoadInst = B.CreateLoad(PointerArg);

  // MemoryPHI should not already exist.
  MemoryPhi *MP = MSSA.getMemoryAccess(Merge);
  EXPECT_EQ(MP, nullptr);

  // Create the load memory acccess
  MemoryUse *LoadAccess = cast<MemoryUse>(MSSA.createMemoryAccessInBB(
      LoadInst, nullptr, Merge, MemorySSA::Beginning));
  Updater.insertUse(LoadAccess);
  MemoryAccess *DefiningAccess = LoadAccess->getDefiningAccess();
  EXPECT_TRUE(isa<MemoryPhi>(DefiningAccess));
  MSSA.verifyMemorySSA();
}

TEST_F(MemorySSATest, MoveAStore) {
  // We create a diamond where there is a in the entry, a store on one side, and
  // a load at the end.  After building MemorySSA, we test updating by moving
  // the store from the side block to the entry block. This destroys the old
  // access.
  F = Function::Create(
      FunctionType::get(B.getVoidTy(), {B.getInt8PtrTy()}, false),
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
  B.CreateLoad(PointerArg);
  setupAnalyses();
  MemorySSA &MSSA = *Analyses->MSSA;

  // Move the store
  SideStore->moveBefore(Entry->getTerminator());
  MemoryAccess *EntryStoreAccess = MSSA.getMemoryAccess(EntryStore);
  MemoryAccess *SideStoreAccess = MSSA.getMemoryAccess(SideStore);
  MemoryAccess *NewStoreAccess = MSSA.createMemoryAccessAfter(
      SideStore, EntryStoreAccess, EntryStoreAccess);
  EntryStoreAccess->replaceAllUsesWith(NewStoreAccess);
  MSSA.removeMemoryAccess(SideStoreAccess);
  MSSA.verifyMemorySSA();
}

TEST_F(MemorySSATest, MoveAStoreUpdater) {
  // We create a diamond where there is a in the entry, a store on one side, and
  // a load at the end.  After building MemorySSA, we test updating by moving
  // the store from the side block to the entry block.  This destroys the old
  // access.
  F = Function::Create(
      FunctionType::get(B.getVoidTy(), {B.getInt8PtrTy()}, false),
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
  auto *MergeLoad = B.CreateLoad(PointerArg);
  setupAnalyses();
  MemorySSA &MSSA = *Analyses->MSSA;
  MemorySSAUpdater Updater(&MSSA);

  // Move the store
  SideStore->moveBefore(Entry->getTerminator());
  auto *EntryStoreAccess = MSSA.getMemoryAccess(EntryStore);
  auto *SideStoreAccess = MSSA.getMemoryAccess(SideStore);
  auto *NewStoreAccess = MSSA.createMemoryAccessAfter(
      SideStore, EntryStoreAccess, EntryStoreAccess);
  // Before, the load will point to a phi of the EntryStore and SideStore.
  auto *LoadAccess = cast<MemoryUse>(MSSA.getMemoryAccess(MergeLoad));
  EXPECT_TRUE(isa<MemoryPhi>(LoadAccess->getDefiningAccess()));
  MemoryPhi *MergePhi = cast<MemoryPhi>(LoadAccess->getDefiningAccess());
  EXPECT_EQ(MergePhi->getIncomingValue(1), EntryStoreAccess);
  EXPECT_EQ(MergePhi->getIncomingValue(0), SideStoreAccess);
  MSSA.removeMemoryAccess(SideStoreAccess);
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
  F = Function::Create(
      FunctionType::get(B.getVoidTy(), {B.getInt8PtrTy()}, false),
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
  auto *MergeLoad = B.CreateLoad(PointerArg);
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
  F = Function::Create(
      FunctionType::get(B.getVoidTy(), {B.getInt8PtrTy()}, false),
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
  auto *MergeLoad = B.CreateLoad(PointerArg);
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
  F = Function::Create(
      FunctionType::get(B.getVoidTy(), {B.getInt8PtrTy()}, false),
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
  LoadInst *LoadInst = B.CreateLoad(PointerArg);

  setupAnalyses();
  MemorySSA &MSSA = *Analyses->MSSA;
  // Before, the load will be a use of a phi<store, liveonentry>.
  MemoryUse *LoadAccess = cast<MemoryUse>(MSSA.getMemoryAccess(LoadInst));
  MemoryDef *StoreAccess = cast<MemoryDef>(MSSA.getMemoryAccess(StoreInst));
  MemoryAccess *DefiningAccess = LoadAccess->getDefiningAccess();
  EXPECT_TRUE(isa<MemoryPhi>(DefiningAccess));
  // Kill the store
  MSSA.removeMemoryAccess(StoreAccess);
  MemoryPhi *MP = cast<MemoryPhi>(DefiningAccess);
  // Verify the phi ended up as liveonentry, liveonentry
  for (auto &Op : MP->incoming_values())
    EXPECT_TRUE(MSSA.isLiveOnEntryDef(cast<MemoryAccess>(Op.get())));
  // Replace the phi uses with the live on entry def
  MP->replaceAllUsesWith(MSSA.getLiveOnEntryDef());
  // Verify the load is now defined by liveOnEntryDef
  EXPECT_TRUE(MSSA.isLiveOnEntryDef(LoadAccess->getDefiningAccess()));
  // Remove the PHI
  MSSA.removeMemoryAccess(MP);
  MSSA.verifyMemorySSA();
}

TEST_F(MemorySSATest, RemoveMemoryAccess) {
  // We create a diamond where there is a store on one side, and then a load
  // after the merge point.  This enables us to test a bunch of different
  // removal cases.
  F = Function::Create(
      FunctionType::get(B.getVoidTy(), {B.getInt8PtrTy()}, false),
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
  LoadInst *LoadInst = B.CreateLoad(PointerArg);

  setupAnalyses();
  MemorySSA &MSSA = *Analyses->MSSA;
  MemorySSAWalker *Walker = Analyses->Walker;

  // Before, the load will be a use of a phi<store, liveonentry>. It should be
  // the same after.
  MemoryUse *LoadAccess = cast<MemoryUse>(MSSA.getMemoryAccess(LoadInst));
  MemoryDef *StoreAccess = cast<MemoryDef>(MSSA.getMemoryAccess(StoreInst));
  MemoryAccess *DefiningAccess = LoadAccess->getDefiningAccess();
  EXPECT_TRUE(isa<MemoryPhi>(DefiningAccess));
  // The load is currently clobbered by one of the phi arguments, so the walker
  // should determine the clobbering access as the phi.
  EXPECT_EQ(DefiningAccess, Walker->getClobberingMemoryAccess(LoadInst));
  MSSA.removeMemoryAccess(StoreAccess);
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
  MSSA.removeMemoryAccess(DefiningAccess);
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
  Instruction *LI = B.CreateLoad(Alloca);

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
  Instruction *ALoad0 = B.CreateLoad(AllocA, "");
  Instruction *BStore = B.CreateStore(Zero, AllocB);
  // Due to use optimization/etc. we make a store to A, which is removed after
  // we build MSSA. This helps keep the test case simple-ish.
  Instruction *KillStore = B.CreateStore(Zero, AllocA);
  Instruction *ALoad = B.CreateLoad(AllocA, "");
  B.CreateBr(IfEnd);

  B.SetInsertPoint(IfEnd);
  Instruction *BelowPhi = B.CreateStore(Zero, AllocA);

  setupAnalyses();
  MemorySSA &MSSA = *Analyses->MSSA;
  MemorySSAWalker *Walker = Analyses->Walker;

  // Kill `KillStore`; it exists solely so that the load after it won't be
  // optimized to FirstStore.
  MSSA.removeMemoryAccess(MSSA.getMemoryAccess(KillStore));
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
  Instruction *Load = B.CreateLoad(AllocA);

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
  Instruction *LIA = B.CreateLoad(AllocaA);

  setupAnalyses();
  MemorySSA &MSSA = *Analyses->MSSA;
  MemorySSAWalker *Walker = Analyses->Walker;

  MemoryAccess *LoadClobber = Walker->getClobberingMemoryAccess(LIA);
  MemoryUse *LoadAccess = cast<MemoryUse>(MSSA.getMemoryAccess(LIA));
  EXPECT_EQ(LoadClobber, MSSA.getMemoryAccess(SIA));
  EXPECT_TRUE(MSSA.isLiveOnEntryDef(Walker->getClobberingMemoryAccess(SIA)));
  MSSA.removeMemoryAccess(LoadAccess);

  // Create the load memory access pointing to an unoptimized place.
  MemoryUse *NewLoadAccess = cast<MemoryUse>(MSSA.createMemoryAccessInBB(
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
  LoadInst *LoadB = B.CreateLoad(B_);
  StoreInst *StoreA1 = B.CreateStore(ConstantInt::get(Int8, 4), A);
  StoreInst *StoreC = B.CreateStore(ConstantInt::get(Int8, 4), C);
  StoreInst *StoreA2 = B.CreateStore(ConstantInt::get(Int8, 4), A);
  LoadInst *LoadC = B.CreateLoad(C);

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
  F = Function::Create(
      FunctionType::get(B.getVoidTy(), {B.getInt8PtrTy()}, false),
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
  LoadInst *LoadInst = B.CreateLoad(FirstArg);
  MemoryUse *LoadAccess = cast<MemoryUse>(MSSA.createMemoryAccessInBB(
      LoadInst, nullptr, AfterLoopBB, MemorySSA::Beginning));
  Updater.insertUse(LoadAccess);
  MSSA.verifyMemorySSA();
}
