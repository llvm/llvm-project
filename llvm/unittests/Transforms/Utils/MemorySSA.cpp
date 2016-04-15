//===- MemorySSA.cpp - Unit tests for MemorySSA ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include "llvm/IR/DataLayout.h"
#include "llvm/Transforms/Utils/MemorySSA.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "gtest/gtest.h"

using namespace llvm;

TEST(MemorySSA, RemoveMemoryAccess) {
  LLVMContext C;
  std::unique_ptr<Module> M(new Module("Remove memory access", C));
  IRBuilder<> B(C);
  DataLayout DL("e-i64:64-f80:128-n8:16:32:64-S128");
  TargetLibraryInfoImpl TLII;
  TargetLibraryInfo TLI(TLII);

  // We create a diamond where there is a store on one side, and then a load
  // after the merge point.  This enables us to test a bunch of different
  // removal cases.
  Function *F = Function::Create(
      FunctionType::get(B.getVoidTy(), {B.getInt8PtrTy()}, false),
      GlobalValue::ExternalLinkage, "F", M.get());
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

  std::unique_ptr<MemorySSA> MSSA(new MemorySSA(*F));
  std::unique_ptr<DominatorTree> DT(new DominatorTree(*F));
  std::unique_ptr<AssumptionCache> AC(new AssumptionCache(*F));
  AAResults AA(TLI);
  BasicAAResult BAA(DL, TLI, *AC, &*DT);
  AA.addAAResult(BAA);
  std::unique_ptr<MemorySSAWalker> Walker(MSSA->buildMemorySSA(&AA, &*DT));
  // Before, the load will be a use of a phi<store, liveonentry>. It should be
  // the same after.
  MemoryUse *LoadAccess = cast<MemoryUse>(MSSA->getMemoryAccess(LoadInst));
  MemoryDef *StoreAccess = cast<MemoryDef>(MSSA->getMemoryAccess(StoreInst));
  MemoryAccess *DefiningAccess = LoadAccess->getDefiningAccess();
  EXPECT_TRUE(isa<MemoryPhi>(DefiningAccess));
  // The load is currently clobbered by one of the phi arguments, so the walker
  // should determine the clobbering access as the phi.
  EXPECT_EQ(DefiningAccess, Walker->getClobberingMemoryAccess(LoadInst));
  MSSA->removeMemoryAccess(StoreAccess);
  MSSA->verifyMemorySSA();
  // After the removeaccess, let's see if we got the right accesses
  // The load should still point to the phi ...
  EXPECT_EQ(DefiningAccess, LoadAccess->getDefiningAccess());
  // but we should now get live on entry for the clobbering definition of the
  // load, since it will walk past the phi node since every argument is the
  // same.
  EXPECT_TRUE(
      MSSA->isLiveOnEntryDef(Walker->getClobberingMemoryAccess(LoadInst)));

  // The phi should now be a two entry phi with two live on entry defs.
  for (const auto &Op : DefiningAccess->operands()) {
    MemoryAccess *Operand = cast<MemoryAccess>(&*Op);
    EXPECT_TRUE(MSSA->isLiveOnEntryDef(Operand));
  }

  // Now we try to remove the single valued phi
  MSSA->removeMemoryAccess(DefiningAccess);
  MSSA->verifyMemorySSA();
  // Now the load should be a load of live on entry.
  EXPECT_TRUE(MSSA->isLiveOnEntryDef(LoadAccess->getDefiningAccess()));
}
