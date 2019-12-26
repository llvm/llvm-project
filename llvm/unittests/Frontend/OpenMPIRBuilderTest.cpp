//===- llvm/unittest/IR/OpenMPIRBuilderTest.cpp - OpenMPIRBuilder tests ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Frontend/OpenMP/OMPIRBuilder.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace omp;
using namespace types;

namespace {

class OpenMPIRBuilderTest : public testing::Test {
protected:
  void SetUp() override {
    M.reset(new Module("MyModule", Ctx));
    FunctionType *FTy =
        FunctionType::get(Type::getVoidTy(Ctx), {Type::getInt32Ty(Ctx)},
                          /*isVarArg=*/false);
    F = Function::Create(FTy, Function::ExternalLinkage, "", M.get());
    BB = BasicBlock::Create(Ctx, "", F);

    DIBuilder DIB(*M);
    auto File = DIB.createFile("test.dbg", "/");
    auto CU =
        DIB.createCompileUnit(dwarf::DW_LANG_C, File, "llvm-C", true, "", 0);
    auto Type = DIB.createSubroutineType(DIB.getOrCreateTypeArray(None));
    auto SP = DIB.createFunction(
        CU, "foo", "", File, 1, Type, 1, DINode::FlagZero,
        DISubprogram::SPFlagDefinition | DISubprogram::SPFlagOptimized);
    F->setSubprogram(SP);
    auto Scope = DIB.createLexicalBlockFile(SP, File, 0);
    DIB.finalize();
    DL = DebugLoc::get(3, 7, Scope);
  }

  void TearDown() override {
    BB = nullptr;
    M.reset();
    uninitializeTypes();
  }

  LLVMContext Ctx;
  std::unique_ptr<Module> M;
  Function *F;
  BasicBlock *BB;
  DebugLoc DL;
};

TEST_F(OpenMPIRBuilderTest, CreateBarrier) {
  OpenMPIRBuilder OMPBuilder(*M);
  OMPBuilder.initialize();

  IRBuilder<> Builder(BB);

  OMPBuilder.CreateBarrier({IRBuilder<>::InsertPoint()}, OMPD_for);
  EXPECT_TRUE(M->global_empty());
  EXPECT_EQ(M->size(), 1U);
  EXPECT_EQ(F->size(), 1U);
  EXPECT_EQ(BB->size(), 0U);

  OpenMPIRBuilder::LocationDescription Loc({Builder.saveIP()});
  OMPBuilder.CreateBarrier(Loc, OMPD_for);
  EXPECT_FALSE(M->global_empty());
  EXPECT_EQ(M->size(), 3U);
  EXPECT_EQ(F->size(), 1U);
  EXPECT_EQ(BB->size(), 2U);

  CallInst *GTID = dyn_cast<CallInst>(&BB->front());
  EXPECT_NE(GTID, nullptr);
  EXPECT_EQ(GTID->getNumArgOperands(), 1U);
  EXPECT_EQ(GTID->getCalledFunction()->getName(), "__kmpc_global_thread_num");
  EXPECT_FALSE(GTID->getCalledFunction()->doesNotAccessMemory());
  EXPECT_FALSE(GTID->getCalledFunction()->doesNotFreeMemory());

  CallInst *Barrier = dyn_cast<CallInst>(GTID->getNextNode());
  EXPECT_NE(Barrier, nullptr);
  EXPECT_EQ(Barrier->getNumArgOperands(), 2U);
  EXPECT_EQ(Barrier->getCalledFunction()->getName(), "__kmpc_barrier");
  EXPECT_FALSE(Barrier->getCalledFunction()->doesNotAccessMemory());
  EXPECT_FALSE(Barrier->getCalledFunction()->doesNotFreeMemory());

  EXPECT_EQ(cast<CallInst>(Barrier)->getArgOperand(1), GTID);

  Builder.CreateUnreachable();
  EXPECT_FALSE(verifyModule(*M));
}

TEST_F(OpenMPIRBuilderTest, CreateCancelBarrier) {
  using InsertPointTy = OpenMPIRBuilder::InsertPointTy;
  OpenMPIRBuilder OMPBuilder(*M);
  OMPBuilder.initialize();

  BasicBlock *CBB = BasicBlock::Create(Ctx, "", F);
  new UnreachableInst(Ctx, CBB);
  auto FiniCB = [&](InsertPointTy IP) {
    ASSERT_NE(IP.getBlock(), nullptr);
    ASSERT_EQ(IP.getBlock()->end(), IP.getPoint());
    BranchInst::Create(CBB, IP.getBlock());
  };
  OMPBuilder.pushFinalizationCB({FiniCB, OMPD_parallel, true});

  IRBuilder<> Builder(BB);

  OpenMPIRBuilder::LocationDescription Loc({Builder.saveIP()});
  auto NewIP = OMPBuilder.CreateBarrier(Loc, OMPD_for);
  Builder.restoreIP(NewIP);
  EXPECT_FALSE(M->global_empty());
  EXPECT_EQ(M->size(), 3U);
  EXPECT_EQ(F->size(), 4U);
  EXPECT_EQ(BB->size(), 4U);

  CallInst *GTID = dyn_cast<CallInst>(&BB->front());
  EXPECT_NE(GTID, nullptr);
  EXPECT_EQ(GTID->getNumArgOperands(), 1U);
  EXPECT_EQ(GTID->getCalledFunction()->getName(), "__kmpc_global_thread_num");
  EXPECT_FALSE(GTID->getCalledFunction()->doesNotAccessMemory());
  EXPECT_FALSE(GTID->getCalledFunction()->doesNotFreeMemory());

  CallInst *Barrier = dyn_cast<CallInst>(GTID->getNextNode());
  EXPECT_NE(Barrier, nullptr);
  EXPECT_EQ(Barrier->getNumArgOperands(), 2U);
  EXPECT_EQ(Barrier->getCalledFunction()->getName(), "__kmpc_cancel_barrier");
  EXPECT_FALSE(Barrier->getCalledFunction()->doesNotAccessMemory());
  EXPECT_FALSE(Barrier->getCalledFunction()->doesNotFreeMemory());
  EXPECT_EQ(Barrier->getNumUses(), 1U);
  Instruction *BarrierBBTI = Barrier->getParent()->getTerminator();
  EXPECT_EQ(BarrierBBTI->getNumSuccessors(), 2U);
  EXPECT_EQ(BarrierBBTI->getSuccessor(0), NewIP.getBlock());
  EXPECT_EQ(BarrierBBTI->getSuccessor(1)->size(), 1U);
  EXPECT_EQ(BarrierBBTI->getSuccessor(1)->getTerminator()->getNumSuccessors(),
            1U);
  EXPECT_EQ(BarrierBBTI->getSuccessor(1)->getTerminator()->getSuccessor(0),
            CBB);

  EXPECT_EQ(cast<CallInst>(Barrier)->getArgOperand(1), GTID);

  OMPBuilder.popFinalizationCB();

  Builder.CreateUnreachable();
  EXPECT_FALSE(verifyModule(*M));
}

TEST_F(OpenMPIRBuilderTest, DbgLoc) {
  OpenMPIRBuilder OMPBuilder(*M);
  OMPBuilder.initialize();
  F->setName("func");

  IRBuilder<> Builder(BB);

  OpenMPIRBuilder::LocationDescription Loc({Builder.saveIP(), DL});
  OMPBuilder.CreateBarrier(Loc, OMPD_for);
  CallInst *GTID = dyn_cast<CallInst>(&BB->front());
  CallInst *Barrier = dyn_cast<CallInst>(GTID->getNextNode());
  EXPECT_EQ(GTID->getDebugLoc(), DL);
  EXPECT_EQ(Barrier->getDebugLoc(), DL);
  EXPECT_TRUE(isa<GlobalVariable>(Barrier->getOperand(0)));
  if (!isa<GlobalVariable>(Barrier->getOperand(0)))
    return;
  GlobalVariable *Ident = cast<GlobalVariable>(Barrier->getOperand(0));
  EXPECT_TRUE(Ident->hasInitializer());
  if (!Ident->hasInitializer())
    return;
  Constant *Initializer = Ident->getInitializer();
  EXPECT_TRUE(
      isa<GlobalVariable>(Initializer->getOperand(4)->stripPointerCasts()));
  GlobalVariable *SrcStrGlob =
      cast<GlobalVariable>(Initializer->getOperand(4)->stripPointerCasts());
  if (!SrcStrGlob)
    return;
  EXPECT_TRUE(isa<ConstantDataArray>(SrcStrGlob->getInitializer()));
  ConstantDataArray *SrcSrc =
      dyn_cast<ConstantDataArray>(SrcStrGlob->getInitializer());
  if (!SrcSrc)
    return;
  EXPECT_EQ(SrcSrc->getAsCString(), ";test.dbg;foo;3;7;;");
}

TEST_F(OpenMPIRBuilderTest, ParallelSimple) {
  using InsertPointTy = OpenMPIRBuilder::InsertPointTy;
  OpenMPIRBuilder OMPBuilder(*M);
  OMPBuilder.initialize();
  F->setName("func");
  IRBuilder<> Builder(BB);

  OpenMPIRBuilder::LocationDescription Loc({Builder.saveIP(), DL});

  AllocaInst *PrivAI = nullptr;

  unsigned NumBodiesGenerated = 0;
  unsigned NumPrivatizedVars = 0;
  unsigned NumFinalizationPoints = 0;

  auto BodyGenCB = [&](InsertPointTy AllocaIP, InsertPointTy CodeGenIP,
                       BasicBlock &ContinuationIP) {
    ++NumBodiesGenerated;

    Builder.restoreIP(AllocaIP);
    PrivAI = Builder.CreateAlloca(F->arg_begin()->getType());
    Builder.CreateStore(F->arg_begin(), PrivAI);

    Builder.restoreIP(CodeGenIP);
    Value *PrivLoad = Builder.CreateLoad(PrivAI, "local.use");
    Value *Cmp = Builder.CreateICmpNE(F->arg_begin(), PrivLoad);
    Instruction *ThenTerm, *ElseTerm;
    SplitBlockAndInsertIfThenElse(Cmp, CodeGenIP.getBlock()->getTerminator(),
                                  &ThenTerm, &ElseTerm);

    Builder.SetInsertPoint(ThenTerm);
    Builder.CreateBr(&ContinuationIP);
    ThenTerm->eraseFromParent();
  };

  auto PrivCB = [&](InsertPointTy AllocaIP, InsertPointTy CodeGenIP,
                    Value &VPtr, Value *&ReplacementValue) -> InsertPointTy {
    ++NumPrivatizedVars;

    if (!isa<AllocaInst>(VPtr)) {
      EXPECT_EQ(&VPtr, F->arg_begin());
      ReplacementValue = &VPtr;
      return CodeGenIP;
    }

    // Trivial copy (=firstprivate).
    Builder.restoreIP(AllocaIP);
    Type *VTy = VPtr.getType()->getPointerElementType();
    Value *V = Builder.CreateLoad(VTy, &VPtr, VPtr.getName() + ".reload");
    ReplacementValue = Builder.CreateAlloca(VTy, 0, VPtr.getName() + ".copy");
    Builder.restoreIP(CodeGenIP);
    Builder.CreateStore(V, ReplacementValue);
    return CodeGenIP;
  };

  auto FiniCB = [&](InsertPointTy CodeGenIP) { ++NumFinalizationPoints; };

  IRBuilder<>::InsertPoint AfterIP = OMPBuilder.CreateParallel(
      Loc, BodyGenCB, PrivCB, FiniCB, nullptr, nullptr, OMP_PROC_BIND_default, false);

  EXPECT_EQ(NumBodiesGenerated, 1U);
  EXPECT_EQ(NumPrivatizedVars, 1U);
  EXPECT_EQ(NumFinalizationPoints, 1U);

  Builder.restoreIP(AfterIP);
  Builder.CreateRetVoid();

  EXPECT_NE(PrivAI, nullptr);
  Function *OutlinedFn = PrivAI->getFunction();
  EXPECT_NE(F, OutlinedFn);
  EXPECT_FALSE(verifyModule(*M));

  EXPECT_TRUE(OutlinedFn->hasInternalLinkage());
  EXPECT_EQ(OutlinedFn->arg_size(), 3U);

  EXPECT_EQ(&OutlinedFn->getEntryBlock(), PrivAI->getParent());
  EXPECT_EQ(OutlinedFn->getNumUses(), 1U);
  User *Usr = OutlinedFn->user_back();
  ASSERT_TRUE(isa<ConstantExpr>(Usr));
  CallInst *ForkCI = dyn_cast<CallInst>(Usr->user_back());
  ASSERT_NE(ForkCI, nullptr);

  EXPECT_EQ(ForkCI->getCalledFunction()->getName(), "__kmpc_fork_call");
  EXPECT_EQ(ForkCI->getNumArgOperands(), 4U);
  EXPECT_TRUE(isa<GlobalVariable>(ForkCI->getArgOperand(0)));
  EXPECT_EQ(ForkCI->getArgOperand(1),
            ConstantInt::get(Type::getInt32Ty(Ctx), 1U));
  EXPECT_EQ(ForkCI->getArgOperand(2), Usr);
  EXPECT_EQ(ForkCI->getArgOperand(3), F->arg_begin());
}

TEST_F(OpenMPIRBuilderTest, ParallelIfCond) {
  using InsertPointTy = OpenMPIRBuilder::InsertPointTy;
  OpenMPIRBuilder OMPBuilder(*M);
  OMPBuilder.initialize();
  F->setName("func");
  IRBuilder<> Builder(BB);

  OpenMPIRBuilder::LocationDescription Loc({Builder.saveIP(), DL});

  AllocaInst *PrivAI = nullptr;

  unsigned NumBodiesGenerated = 0;
  unsigned NumPrivatizedVars = 0;
  unsigned NumFinalizationPoints = 0;

  auto BodyGenCB = [&](InsertPointTy AllocaIP, InsertPointTy CodeGenIP,
                       BasicBlock &ContinuationIP) {
    ++NumBodiesGenerated;

    Builder.restoreIP(AllocaIP);
    PrivAI = Builder.CreateAlloca(F->arg_begin()->getType());
    Builder.CreateStore(F->arg_begin(), PrivAI);

    Builder.restoreIP(CodeGenIP);
    Value *PrivLoad = Builder.CreateLoad(PrivAI, "local.use");
    Value *Cmp = Builder.CreateICmpNE(F->arg_begin(), PrivLoad);
    Instruction *ThenTerm, *ElseTerm;
    SplitBlockAndInsertIfThenElse(Cmp, CodeGenIP.getBlock()->getTerminator(),
                                  &ThenTerm, &ElseTerm);

    Builder.SetInsertPoint(ThenTerm);
    Builder.CreateBr(&ContinuationIP);
    ThenTerm->eraseFromParent();
  };

  auto PrivCB = [&](InsertPointTy AllocaIP, InsertPointTy CodeGenIP,
                    Value &VPtr, Value *&ReplacementValue) -> InsertPointTy {
    ++NumPrivatizedVars;

    if (!isa<AllocaInst>(VPtr)) {
      EXPECT_EQ(&VPtr, F->arg_begin());
      ReplacementValue = &VPtr;
      return CodeGenIP;
    }

    // Trivial copy (=firstprivate).
    Builder.restoreIP(AllocaIP);
    Type *VTy = VPtr.getType()->getPointerElementType();
    Value *V = Builder.CreateLoad(VTy, &VPtr, VPtr.getName() + ".reload");
    ReplacementValue = Builder.CreateAlloca(VTy, 0, VPtr.getName() + ".copy");
    Builder.restoreIP(CodeGenIP);
    Builder.CreateStore(V, ReplacementValue);
    return CodeGenIP;
  };

  auto FiniCB = [&](InsertPointTy CodeGenIP) {
    ++NumFinalizationPoints;
    // No destructors.
  };

  IRBuilder<>::InsertPoint AfterIP = OMPBuilder.CreateParallel(
      Loc, BodyGenCB, PrivCB, FiniCB, Builder.CreateIsNotNull(F->arg_begin()),
      nullptr, OMP_PROC_BIND_default, false);

  EXPECT_EQ(NumBodiesGenerated, 1U);
  EXPECT_EQ(NumPrivatizedVars, 1U);
  EXPECT_EQ(NumFinalizationPoints, 1U);

  Builder.restoreIP(AfterIP);
  Builder.CreateRetVoid();

  EXPECT_NE(PrivAI, nullptr);
  Function *OutlinedFn = PrivAI->getFunction();
  EXPECT_NE(F, OutlinedFn);
  EXPECT_FALSE(verifyModule(*M, &errs()));

  EXPECT_TRUE(OutlinedFn->hasInternalLinkage());
  EXPECT_EQ(OutlinedFn->arg_size(), 3U);

  EXPECT_EQ(&OutlinedFn->getEntryBlock(), PrivAI->getParent());
  ASSERT_EQ(OutlinedFn->getNumUses(), 2U);

  CallInst *DirectCI = nullptr;
  CallInst *ForkCI = nullptr;
  for (User *Usr : OutlinedFn->users()) {
    if (isa<CallInst>(Usr)) {
      ASSERT_EQ(DirectCI, nullptr);
      DirectCI = cast<CallInst>(Usr);
    } else {
      ASSERT_TRUE(isa<ConstantExpr>(Usr));
      ASSERT_EQ(Usr->getNumUses(), 1U);
      ASSERT_TRUE(isa<CallInst>(Usr->user_back()));
      ForkCI = cast<CallInst>(Usr->user_back());
    }
  }

  EXPECT_EQ(ForkCI->getCalledFunction()->getName(), "__kmpc_fork_call");
  EXPECT_EQ(ForkCI->getNumArgOperands(), 4U);
  EXPECT_TRUE(isa<GlobalVariable>(ForkCI->getArgOperand(0)));
  EXPECT_EQ(ForkCI->getArgOperand(1),
            ConstantInt::get(Type::getInt32Ty(Ctx), 1));
  EXPECT_EQ(ForkCI->getArgOperand(3), F->arg_begin());

  EXPECT_EQ(DirectCI->getCalledFunction(), OutlinedFn);
  EXPECT_EQ(DirectCI->getNumArgOperands(), 3U);
  EXPECT_TRUE(isa<AllocaInst>(DirectCI->getArgOperand(0)));
  EXPECT_TRUE(isa<AllocaInst>(DirectCI->getArgOperand(1)));
  EXPECT_EQ(DirectCI->getArgOperand(2), F->arg_begin());
}

TEST_F(OpenMPIRBuilderTest, ParallelCancelBarrier) {
  using InsertPointTy = OpenMPIRBuilder::InsertPointTy;
  OpenMPIRBuilder OMPBuilder(*M);
  OMPBuilder.initialize();
  F->setName("func");
  IRBuilder<> Builder(BB);

  OpenMPIRBuilder::LocationDescription Loc({Builder.saveIP(), DL});

  unsigned NumBodiesGenerated = 0;
  unsigned NumPrivatizedVars = 0;
  unsigned NumFinalizationPoints = 0;

  CallInst *CheckedBarrier = nullptr;
  auto BodyGenCB = [&](InsertPointTy AllocaIP, InsertPointTy CodeGenIP,
                       BasicBlock &ContinuationIP) {
    ++NumBodiesGenerated;

    Builder.restoreIP(CodeGenIP);

    // Create three barriers, two cancel barriers but only one checked.
    Function *CBFn, *BFn;

    Builder.restoreIP(
        OMPBuilder.CreateBarrier(Builder.saveIP(), OMPD_parallel));

    CBFn = M->getFunction("__kmpc_cancel_barrier");
    BFn = M->getFunction("__kmpc_barrier");
    ASSERT_NE(CBFn, nullptr);
    ASSERT_EQ(BFn, nullptr);
    ASSERT_EQ(CBFn->getNumUses(), 1U);
    ASSERT_TRUE(isa<CallInst>(CBFn->user_back()));
    ASSERT_EQ(CBFn->user_back()->getNumUses(), 1U);
    CheckedBarrier = cast<CallInst>(CBFn->user_back());

    Builder.restoreIP(
        OMPBuilder.CreateBarrier(Builder.saveIP(), OMPD_parallel, true));
    CBFn = M->getFunction("__kmpc_cancel_barrier");
    BFn = M->getFunction("__kmpc_barrier");
    ASSERT_NE(CBFn, nullptr);
    ASSERT_NE(BFn, nullptr);
    ASSERT_EQ(CBFn->getNumUses(), 1U);
    ASSERT_EQ(BFn->getNumUses(), 1U);
    ASSERT_TRUE(isa<CallInst>(BFn->user_back()));
    ASSERT_EQ(BFn->user_back()->getNumUses(), 0U);

    Builder.restoreIP(OMPBuilder.CreateBarrier(Builder.saveIP(), OMPD_parallel,
                                               false, false));
    ASSERT_EQ(CBFn->getNumUses(), 2U);
    ASSERT_EQ(BFn->getNumUses(), 1U);
    ASSERT_TRUE(CBFn->user_back() != CheckedBarrier);
    ASSERT_TRUE(isa<CallInst>(CBFn->user_back()));
    ASSERT_EQ(CBFn->user_back()->getNumUses(), 0U);
  };

  auto PrivCB = [&](InsertPointTy, InsertPointTy, Value &V,
                    Value *&) -> InsertPointTy {
    ++NumPrivatizedVars;
    llvm_unreachable("No privatization callback call expected!");
  };

  FunctionType *FakeDestructorTy =
      FunctionType::get(Type::getVoidTy(Ctx), {Type::getInt32Ty(Ctx)},
                        /*isVarArg=*/false);
  auto *FakeDestructor = Function::Create(
      FakeDestructorTy, Function::ExternalLinkage, "fakeDestructor", M.get());

  auto FiniCB = [&](InsertPointTy IP) {
    ++NumFinalizationPoints;
    Builder.restoreIP(IP);
    Builder.CreateCall(FakeDestructor,
                       {Builder.getInt32(NumFinalizationPoints)});
  };

  IRBuilder<>::InsertPoint AfterIP = OMPBuilder.CreateParallel(
      Loc, BodyGenCB, PrivCB, FiniCB, Builder.CreateIsNotNull(F->arg_begin()),
      nullptr, OMP_PROC_BIND_default, true);

  EXPECT_EQ(NumBodiesGenerated, 1U);
  EXPECT_EQ(NumPrivatizedVars, 0U);
  EXPECT_EQ(NumFinalizationPoints, 2U);
  EXPECT_EQ(FakeDestructor->getNumUses(), 2U);

  Builder.restoreIP(AfterIP);
  Builder.CreateRetVoid();

  EXPECT_FALSE(verifyModule(*M, &errs()));

  BasicBlock *ExitBB = nullptr;
  for (const User *Usr : FakeDestructor->users()) {
    const CallInst *CI = dyn_cast<CallInst>(Usr);
    ASSERT_EQ(CI->getCalledFunction(), FakeDestructor);
    ASSERT_TRUE(isa<BranchInst>(CI->getNextNode()));
    ASSERT_EQ(CI->getNextNode()->getNumSuccessors(), 1U);
    if (ExitBB)
      ASSERT_EQ(CI->getNextNode()->getSuccessor(0), ExitBB);
    else
      ExitBB = CI->getNextNode()->getSuccessor(0);
    ASSERT_EQ(ExitBB->size(), 1U);
    ASSERT_TRUE(isa<ReturnInst>(ExitBB->front()));
  }
}

} // namespace
