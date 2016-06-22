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
    MemorySSA MSSA;
    MemorySSAWalker *Walker;

    TestAnalyses(MemorySSATest &Test)
        : DT(*Test.F), AC(*Test.F), AA(Test.TLI),
          BAA(Test.DL, Test.TLI, AC, &DT), MSSA(*Test.F, &AA, &DT) {
      AA.addAAResult(BAA);
      Walker = MSSA.getWalker();
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

TEST_F(MemorySSATest, CreateALoadAndPhi) {
  // We create a diamond where there is a store on one side, and then after
  // running memory ssa, create a load after the merge point, and use it to test
  // updating by creating an access for the load and a memoryphi.
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

  setupAnalyses();
  MemorySSA &MSSA = Analyses->MSSA;
  // Add the load
  B.SetInsertPoint(Merge);
  LoadInst *LoadInst = B.CreateLoad(PointerArg);
  // Should be no phi to start
  EXPECT_EQ(MSSA.getMemoryAccess(Merge), nullptr);

  // Create the phi
  MemoryPhi *MP = MSSA.createMemoryPhi(Merge);
  MemoryDef *StoreAccess = cast<MemoryDef>(MSSA.getMemoryAccess(StoreInst));
  MP->addIncoming(StoreAccess, Left);
  MP->addIncoming(MSSA.getLiveOnEntryDef(), Right);

  // Create the load memory acccess
  MemoryUse *LoadAccess = cast<MemoryUse>(
      MSSA.createMemoryAccessInBB(LoadInst, MP, Merge, MemorySSA::Beginning));
  MemoryAccess *DefiningAccess = LoadAccess->getDefiningAccess();
  EXPECT_TRUE(isa<MemoryPhi>(DefiningAccess));
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
  MemorySSA &MSSA = Analyses->MSSA;
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
  MemorySSA &MSSA = Analyses->MSSA;
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
  MemorySSA &MSSA = Analyses->MSSA;
  MemorySSAWalker *Walker = Analyses->Walker;

  unsigned I = 0;
  for (StoreInst *V : {S1, S2, S3}) {
    // Everything should be clobbered by its defining access
    MemoryAccess *DefiningAccess =
        cast<MemoryUseOrDef>(MSSA.getMemoryAccess(V))->getDefiningAccess();
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
  MemorySSA &MSSA = Analyses->MSSA;
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
  MemorySSA &MSSA = Analyses->MSSA;
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
