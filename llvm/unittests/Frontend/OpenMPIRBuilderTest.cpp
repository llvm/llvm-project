//===- llvm/unittest/IR/OpenMPIRBuilderTest.cpp - OpenMPIRBuilder tests ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Frontend/OpenMP/OMPConstants.h"
#include "llvm/Frontend/OpenMP/OMPIRBuilder.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace omp;

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
    auto File = DIB.createFile("test.dbg", "/src", llvm::None,
                               Optional<StringRef>("/src/test.dbg"));
    auto CU =
        DIB.createCompileUnit(dwarf::DW_LANG_C, File, "llvm-C", true, "", 0);
    auto Type = DIB.createSubroutineType(DIB.getOrCreateTypeArray(None));
    auto SP = DIB.createFunction(
        CU, "foo", "", File, 1, Type, 1, DINode::FlagZero,
        DISubprogram::SPFlagDefinition | DISubprogram::SPFlagOptimized);
    F->setSubprogram(SP);
    auto Scope = DIB.createLexicalBlockFile(SP, File, 0);
    DIB.finalize();
    DL = DILocation::get(Ctx, 3, 7, Scope);
  }

  void TearDown() override {
    BB = nullptr;
    M.reset();
  }

  LLVMContext Ctx;
  std::unique_ptr<Module> M;
  Function *F;
  BasicBlock *BB;
  DebugLoc DL;
};

// Returns the value stored in the given allocation. Returns null if the given
// value is not a result of an allocation, if no value is stored or if there is
// more than one store.
static Value *findStoredValue(Value *AllocaValue) {
  Instruction *Alloca = dyn_cast<AllocaInst>(AllocaValue);
  if (!Alloca)
    return nullptr;
  StoreInst *Store = nullptr;
  for (Use &U : Alloca->uses()) {
    if (auto *CandidateStore = dyn_cast<StoreInst>(U.getUser())) {
      EXPECT_EQ(Store, nullptr);
      Store = CandidateStore;
    }
  }
  if (!Store)
    return nullptr;
  return Store->getValueOperand();
}

TEST_F(OpenMPIRBuilderTest, CreateBarrier) {
  OpenMPIRBuilder OMPBuilder(*M);
  OMPBuilder.initialize();

  IRBuilder<> Builder(BB);

  OMPBuilder.createBarrier({IRBuilder<>::InsertPoint()}, OMPD_for);
  EXPECT_TRUE(M->global_empty());
  EXPECT_EQ(M->size(), 1U);
  EXPECT_EQ(F->size(), 1U);
  EXPECT_EQ(BB->size(), 0U);

  OpenMPIRBuilder::LocationDescription Loc({Builder.saveIP()});
  OMPBuilder.createBarrier(Loc, OMPD_for);
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
  EXPECT_FALSE(verifyModule(*M, &errs()));
}

TEST_F(OpenMPIRBuilderTest, CreateCancel) {
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
  auto NewIP = OMPBuilder.createCancel(Loc, nullptr, OMPD_parallel);
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

  CallInst *Cancel = dyn_cast<CallInst>(GTID->getNextNode());
  EXPECT_NE(Cancel, nullptr);
  EXPECT_EQ(Cancel->getNumArgOperands(), 3U);
  EXPECT_EQ(Cancel->getCalledFunction()->getName(), "__kmpc_cancel");
  EXPECT_FALSE(Cancel->getCalledFunction()->doesNotAccessMemory());
  EXPECT_FALSE(Cancel->getCalledFunction()->doesNotFreeMemory());
  EXPECT_EQ(Cancel->getNumUses(), 1U);
  Instruction *CancelBBTI = Cancel->getParent()->getTerminator();
  EXPECT_EQ(CancelBBTI->getNumSuccessors(), 2U);
  EXPECT_EQ(CancelBBTI->getSuccessor(0), NewIP.getBlock());
  EXPECT_EQ(CancelBBTI->getSuccessor(1)->size(), 1U);
  EXPECT_EQ(CancelBBTI->getSuccessor(1)->getTerminator()->getNumSuccessors(),
            1U);
  EXPECT_EQ(CancelBBTI->getSuccessor(1)->getTerminator()->getSuccessor(0),
            CBB);

  EXPECT_EQ(cast<CallInst>(Cancel)->getArgOperand(1), GTID);

  OMPBuilder.popFinalizationCB();

  Builder.CreateUnreachable();
  EXPECT_FALSE(verifyModule(*M, &errs()));
}

TEST_F(OpenMPIRBuilderTest, CreateCancelIfCond) {
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
  auto NewIP = OMPBuilder.createCancel(Loc, Builder.getTrue(), OMPD_parallel);
  Builder.restoreIP(NewIP);
  EXPECT_FALSE(M->global_empty());
  EXPECT_EQ(M->size(), 3U);
  EXPECT_EQ(F->size(), 7U);
  EXPECT_EQ(BB->size(), 1U);
  ASSERT_TRUE(isa<BranchInst>(BB->getTerminator()));
  ASSERT_EQ(BB->getTerminator()->getNumSuccessors(), 2U);
  BB = BB->getTerminator()->getSuccessor(0);
  EXPECT_EQ(BB->size(), 4U);


  CallInst *GTID = dyn_cast<CallInst>(&BB->front());
  EXPECT_NE(GTID, nullptr);
  EXPECT_EQ(GTID->getNumArgOperands(), 1U);
  EXPECT_EQ(GTID->getCalledFunction()->getName(), "__kmpc_global_thread_num");
  EXPECT_FALSE(GTID->getCalledFunction()->doesNotAccessMemory());
  EXPECT_FALSE(GTID->getCalledFunction()->doesNotFreeMemory());

  CallInst *Cancel = dyn_cast<CallInst>(GTID->getNextNode());
  EXPECT_NE(Cancel, nullptr);
  EXPECT_EQ(Cancel->getNumArgOperands(), 3U);
  EXPECT_EQ(Cancel->getCalledFunction()->getName(), "__kmpc_cancel");
  EXPECT_FALSE(Cancel->getCalledFunction()->doesNotAccessMemory());
  EXPECT_FALSE(Cancel->getCalledFunction()->doesNotFreeMemory());
  EXPECT_EQ(Cancel->getNumUses(), 1U);
  Instruction *CancelBBTI = Cancel->getParent()->getTerminator();
  EXPECT_EQ(CancelBBTI->getNumSuccessors(), 2U);
  EXPECT_EQ(CancelBBTI->getSuccessor(0)->size(), 1U);
  EXPECT_EQ(CancelBBTI->getSuccessor(0)->getUniqueSuccessor(), NewIP.getBlock());
  EXPECT_EQ(CancelBBTI->getSuccessor(1)->size(), 1U);
  EXPECT_EQ(CancelBBTI->getSuccessor(1)->getTerminator()->getNumSuccessors(),
            1U);
  EXPECT_EQ(CancelBBTI->getSuccessor(1)->getTerminator()->getSuccessor(0),
            CBB);

  EXPECT_EQ(cast<CallInst>(Cancel)->getArgOperand(1), GTID);

  OMPBuilder.popFinalizationCB();

  Builder.CreateUnreachable();
  EXPECT_FALSE(verifyModule(*M, &errs()));
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
  auto NewIP = OMPBuilder.createBarrier(Loc, OMPD_for);
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
  EXPECT_FALSE(verifyModule(*M, &errs()));
}

TEST_F(OpenMPIRBuilderTest, DbgLoc) {
  OpenMPIRBuilder OMPBuilder(*M);
  OMPBuilder.initialize();
  F->setName("func");

  IRBuilder<> Builder(BB);

  OpenMPIRBuilder::LocationDescription Loc({Builder.saveIP(), DL});
  OMPBuilder.createBarrier(Loc, OMPD_for);
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
  EXPECT_EQ(SrcSrc->getAsCString(), ";/src/test.dbg;foo;3;7;;");
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
                    Value &Orig, Value &Inner,
                    Value *&ReplacementValue) -> InsertPointTy {
    ++NumPrivatizedVars;

    if (!isa<AllocaInst>(Orig)) {
      EXPECT_EQ(&Orig, F->arg_begin());
      ReplacementValue = &Inner;
      return CodeGenIP;
    }

    // Since the original value is an allocation, it has a pointer type and
    // therefore no additional wrapping should happen.
    EXPECT_EQ(&Orig, &Inner);

    // Trivial copy (=firstprivate).
    Builder.restoreIP(AllocaIP);
    Type *VTy = Inner.getType()->getPointerElementType();
    Value *V = Builder.CreateLoad(VTy, &Inner, Orig.getName() + ".reload");
    ReplacementValue = Builder.CreateAlloca(VTy, 0, Orig.getName() + ".copy");
    Builder.restoreIP(CodeGenIP);
    Builder.CreateStore(V, ReplacementValue);
    return CodeGenIP;
  };

  auto FiniCB = [&](InsertPointTy CodeGenIP) { ++NumFinalizationPoints; };

  IRBuilder<>::InsertPoint AllocaIP(&F->getEntryBlock(),
                                    F->getEntryBlock().getFirstInsertionPt());
  IRBuilder<>::InsertPoint AfterIP =
      OMPBuilder.createParallel(Loc, AllocaIP, BodyGenCB, PrivCB, FiniCB,
                                nullptr, nullptr, OMP_PROC_BIND_default, false);
  EXPECT_EQ(NumBodiesGenerated, 1U);
  EXPECT_EQ(NumPrivatizedVars, 1U);
  EXPECT_EQ(NumFinalizationPoints, 1U);

  Builder.restoreIP(AfterIP);
  Builder.CreateRetVoid();

  OMPBuilder.finalize();

  EXPECT_NE(PrivAI, nullptr);
  Function *OutlinedFn = PrivAI->getFunction();
  EXPECT_NE(F, OutlinedFn);
  EXPECT_FALSE(verifyModule(*M, &errs()));
  EXPECT_TRUE(OutlinedFn->hasFnAttribute(Attribute::NoUnwind));
  EXPECT_TRUE(OutlinedFn->hasFnAttribute(Attribute::NoRecurse));
  EXPECT_TRUE(OutlinedFn->hasParamAttribute(0, Attribute::NoAlias));
  EXPECT_TRUE(OutlinedFn->hasParamAttribute(1, Attribute::NoAlias));

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
  EXPECT_EQ(findStoredValue(ForkCI->getArgOperand(3)), F->arg_begin());
}

TEST_F(OpenMPIRBuilderTest, ParallelNested) {
  using InsertPointTy = OpenMPIRBuilder::InsertPointTy;
  OpenMPIRBuilder OMPBuilder(*M);
  OMPBuilder.initialize();
  F->setName("func");
  IRBuilder<> Builder(BB);

  OpenMPIRBuilder::LocationDescription Loc({Builder.saveIP(), DL});

  unsigned NumInnerBodiesGenerated = 0;
  unsigned NumOuterBodiesGenerated = 0;
  unsigned NumFinalizationPoints = 0;

  auto InnerBodyGenCB = [&](InsertPointTy AllocaIP, InsertPointTy CodeGenIP,
                            BasicBlock &ContinuationIP) {
    ++NumInnerBodiesGenerated;
  };

  auto PrivCB = [&](InsertPointTy AllocaIP, InsertPointTy CodeGenIP,
                    Value &Orig, Value &Inner,
                    Value *&ReplacementValue) -> InsertPointTy {
    // Trivial copy (=firstprivate).
    Builder.restoreIP(AllocaIP);
    Type *VTy = Inner.getType()->getPointerElementType();
    Value *V = Builder.CreateLoad(VTy, &Inner, Orig.getName() + ".reload");
    ReplacementValue = Builder.CreateAlloca(VTy, 0, Orig.getName() + ".copy");
    Builder.restoreIP(CodeGenIP);
    Builder.CreateStore(V, ReplacementValue);
    return CodeGenIP;
  };

  auto FiniCB = [&](InsertPointTy CodeGenIP) { ++NumFinalizationPoints; };

  auto OuterBodyGenCB = [&](InsertPointTy AllocaIP, InsertPointTy CodeGenIP,
                            BasicBlock &ContinuationIP) {
    ++NumOuterBodiesGenerated;
    Builder.restoreIP(CodeGenIP);
    BasicBlock *CGBB = CodeGenIP.getBlock();
    BasicBlock *NewBB = SplitBlock(CGBB, &*CodeGenIP.getPoint());
    CGBB->getTerminator()->eraseFromParent();
    ;

    IRBuilder<>::InsertPoint AfterIP = OMPBuilder.createParallel(
        InsertPointTy(CGBB, CGBB->end()), AllocaIP, InnerBodyGenCB, PrivCB,
        FiniCB, nullptr, nullptr, OMP_PROC_BIND_default, false);

    Builder.restoreIP(AfterIP);
    Builder.CreateBr(NewBB);
  };

  IRBuilder<>::InsertPoint AllocaIP(&F->getEntryBlock(),
                                    F->getEntryBlock().getFirstInsertionPt());
  IRBuilder<>::InsertPoint AfterIP =
      OMPBuilder.createParallel(Loc, AllocaIP, OuterBodyGenCB, PrivCB, FiniCB,
                                nullptr, nullptr, OMP_PROC_BIND_default, false);

  EXPECT_EQ(NumInnerBodiesGenerated, 1U);
  EXPECT_EQ(NumOuterBodiesGenerated, 1U);
  EXPECT_EQ(NumFinalizationPoints, 2U);

  Builder.restoreIP(AfterIP);
  Builder.CreateRetVoid();

  OMPBuilder.finalize();

  EXPECT_EQ(M->size(), 5U);
  for (Function &OutlinedFn : *M) {
    if (F == &OutlinedFn || OutlinedFn.isDeclaration())
      continue;
    EXPECT_FALSE(verifyModule(*M, &errs()));
    EXPECT_TRUE(OutlinedFn.hasFnAttribute(Attribute::NoUnwind));
    EXPECT_TRUE(OutlinedFn.hasFnAttribute(Attribute::NoRecurse));
    EXPECT_TRUE(OutlinedFn.hasParamAttribute(0, Attribute::NoAlias));
    EXPECT_TRUE(OutlinedFn.hasParamAttribute(1, Attribute::NoAlias));

    EXPECT_TRUE(OutlinedFn.hasInternalLinkage());
    EXPECT_EQ(OutlinedFn.arg_size(), 2U);

    EXPECT_EQ(OutlinedFn.getNumUses(), 1U);
    User *Usr = OutlinedFn.user_back();
    ASSERT_TRUE(isa<ConstantExpr>(Usr));
    CallInst *ForkCI = dyn_cast<CallInst>(Usr->user_back());
    ASSERT_NE(ForkCI, nullptr);

    EXPECT_EQ(ForkCI->getCalledFunction()->getName(), "__kmpc_fork_call");
    EXPECT_EQ(ForkCI->getNumArgOperands(), 3U);
    EXPECT_TRUE(isa<GlobalVariable>(ForkCI->getArgOperand(0)));
    EXPECT_EQ(ForkCI->getArgOperand(1),
              ConstantInt::get(Type::getInt32Ty(Ctx), 0U));
    EXPECT_EQ(ForkCI->getArgOperand(2), Usr);
  }
}

TEST_F(OpenMPIRBuilderTest, ParallelNested2Inner) {
  using InsertPointTy = OpenMPIRBuilder::InsertPointTy;
  OpenMPIRBuilder OMPBuilder(*M);
  OMPBuilder.initialize();
  F->setName("func");
  IRBuilder<> Builder(BB);

  OpenMPIRBuilder::LocationDescription Loc({Builder.saveIP(), DL});

  unsigned NumInnerBodiesGenerated = 0;
  unsigned NumOuterBodiesGenerated = 0;
  unsigned NumFinalizationPoints = 0;

  auto InnerBodyGenCB = [&](InsertPointTy AllocaIP, InsertPointTy CodeGenIP,
                            BasicBlock &ContinuationIP) {
    ++NumInnerBodiesGenerated;
  };

  auto PrivCB = [&](InsertPointTy AllocaIP, InsertPointTy CodeGenIP,
                    Value &Orig, Value &Inner,
                    Value *&ReplacementValue) -> InsertPointTy {
    // Trivial copy (=firstprivate).
    Builder.restoreIP(AllocaIP);
    Type *VTy = Inner.getType()->getPointerElementType();
    Value *V = Builder.CreateLoad(VTy, &Inner, Orig.getName() + ".reload");
    ReplacementValue = Builder.CreateAlloca(VTy, 0, Orig.getName() + ".copy");
    Builder.restoreIP(CodeGenIP);
    Builder.CreateStore(V, ReplacementValue);
    return CodeGenIP;
  };

  auto FiniCB = [&](InsertPointTy CodeGenIP) { ++NumFinalizationPoints; };

  auto OuterBodyGenCB = [&](InsertPointTy AllocaIP, InsertPointTy CodeGenIP,
                            BasicBlock &ContinuationIP) {
    ++NumOuterBodiesGenerated;
    Builder.restoreIP(CodeGenIP);
    BasicBlock *CGBB = CodeGenIP.getBlock();
    BasicBlock *NewBB1 = SplitBlock(CGBB, &*CodeGenIP.getPoint());
    BasicBlock *NewBB2 = SplitBlock(NewBB1, &*NewBB1->getFirstInsertionPt());
    CGBB->getTerminator()->eraseFromParent();
    ;
    NewBB1->getTerminator()->eraseFromParent();
    ;

    IRBuilder<>::InsertPoint AfterIP1 = OMPBuilder.createParallel(
        InsertPointTy(CGBB, CGBB->end()), AllocaIP, InnerBodyGenCB, PrivCB,
        FiniCB, nullptr, nullptr, OMP_PROC_BIND_default, false);

    Builder.restoreIP(AfterIP1);
    Builder.CreateBr(NewBB1);

    IRBuilder<>::InsertPoint AfterIP2 = OMPBuilder.createParallel(
        InsertPointTy(NewBB1, NewBB1->end()), AllocaIP, InnerBodyGenCB, PrivCB,
        FiniCB, nullptr, nullptr, OMP_PROC_BIND_default, false);

    Builder.restoreIP(AfterIP2);
    Builder.CreateBr(NewBB2);
  };

  IRBuilder<>::InsertPoint AllocaIP(&F->getEntryBlock(),
                                    F->getEntryBlock().getFirstInsertionPt());
  IRBuilder<>::InsertPoint AfterIP =
      OMPBuilder.createParallel(Loc, AllocaIP, OuterBodyGenCB, PrivCB, FiniCB,
                                nullptr, nullptr, OMP_PROC_BIND_default, false);

  EXPECT_EQ(NumInnerBodiesGenerated, 2U);
  EXPECT_EQ(NumOuterBodiesGenerated, 1U);
  EXPECT_EQ(NumFinalizationPoints, 3U);

  Builder.restoreIP(AfterIP);
  Builder.CreateRetVoid();

  OMPBuilder.finalize();

  EXPECT_EQ(M->size(), 6U);
  for (Function &OutlinedFn : *M) {
    if (F == &OutlinedFn || OutlinedFn.isDeclaration())
      continue;
    EXPECT_FALSE(verifyModule(*M, &errs()));
    EXPECT_TRUE(OutlinedFn.hasFnAttribute(Attribute::NoUnwind));
    EXPECT_TRUE(OutlinedFn.hasFnAttribute(Attribute::NoRecurse));
    EXPECT_TRUE(OutlinedFn.hasParamAttribute(0, Attribute::NoAlias));
    EXPECT_TRUE(OutlinedFn.hasParamAttribute(1, Attribute::NoAlias));

    EXPECT_TRUE(OutlinedFn.hasInternalLinkage());
    EXPECT_EQ(OutlinedFn.arg_size(), 2U);

    unsigned NumAllocas = 0;
    for (Instruction &I : instructions(OutlinedFn))
      NumAllocas += isa<AllocaInst>(I);
    EXPECT_EQ(NumAllocas, 1U);

    EXPECT_EQ(OutlinedFn.getNumUses(), 1U);
    User *Usr = OutlinedFn.user_back();
    ASSERT_TRUE(isa<ConstantExpr>(Usr));
    CallInst *ForkCI = dyn_cast<CallInst>(Usr->user_back());
    ASSERT_NE(ForkCI, nullptr);

    EXPECT_EQ(ForkCI->getCalledFunction()->getName(), "__kmpc_fork_call");
    EXPECT_EQ(ForkCI->getNumArgOperands(), 3U);
    EXPECT_TRUE(isa<GlobalVariable>(ForkCI->getArgOperand(0)));
    EXPECT_EQ(ForkCI->getArgOperand(1),
              ConstantInt::get(Type::getInt32Ty(Ctx), 0U));
    EXPECT_EQ(ForkCI->getArgOperand(2), Usr);
  }
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
                    Value &Orig, Value &Inner,
                    Value *&ReplacementValue) -> InsertPointTy {
    ++NumPrivatizedVars;

    if (!isa<AllocaInst>(Orig)) {
      EXPECT_EQ(&Orig, F->arg_begin());
      ReplacementValue = &Inner;
      return CodeGenIP;
    }

    // Since the original value is an allocation, it has a pointer type and
    // therefore no additional wrapping should happen.
    EXPECT_EQ(&Orig, &Inner);

    // Trivial copy (=firstprivate).
    Builder.restoreIP(AllocaIP);
    Type *VTy = Inner.getType()->getPointerElementType();
    Value *V = Builder.CreateLoad(VTy, &Inner, Orig.getName() + ".reload");
    ReplacementValue = Builder.CreateAlloca(VTy, 0, Orig.getName() + ".copy");
    Builder.restoreIP(CodeGenIP);
    Builder.CreateStore(V, ReplacementValue);
    return CodeGenIP;
  };

  auto FiniCB = [&](InsertPointTy CodeGenIP) {
    ++NumFinalizationPoints;
    // No destructors.
  };

  IRBuilder<>::InsertPoint AllocaIP(&F->getEntryBlock(),
                                    F->getEntryBlock().getFirstInsertionPt());
  IRBuilder<>::InsertPoint AfterIP =
      OMPBuilder.createParallel(Loc, AllocaIP, BodyGenCB, PrivCB, FiniCB,
                                Builder.CreateIsNotNull(F->arg_begin()),
                                nullptr, OMP_PROC_BIND_default, false);

  EXPECT_EQ(NumBodiesGenerated, 1U);
  EXPECT_EQ(NumPrivatizedVars, 1U);
  EXPECT_EQ(NumFinalizationPoints, 1U);

  Builder.restoreIP(AfterIP);
  Builder.CreateRetVoid();
  OMPBuilder.finalize();

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
  Value *StoredForkArg = findStoredValue(ForkCI->getArgOperand(3));
  EXPECT_EQ(StoredForkArg, F->arg_begin());

  EXPECT_EQ(DirectCI->getCalledFunction(), OutlinedFn);
  EXPECT_EQ(DirectCI->getNumArgOperands(), 3U);
  EXPECT_TRUE(isa<AllocaInst>(DirectCI->getArgOperand(0)));
  EXPECT_TRUE(isa<AllocaInst>(DirectCI->getArgOperand(1)));
  Value *StoredDirectArg = findStoredValue(DirectCI->getArgOperand(2));
  EXPECT_EQ(StoredDirectArg, F->arg_begin());
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
        OMPBuilder.createBarrier(Builder.saveIP(), OMPD_parallel));

    CBFn = M->getFunction("__kmpc_cancel_barrier");
    BFn = M->getFunction("__kmpc_barrier");
    ASSERT_NE(CBFn, nullptr);
    ASSERT_EQ(BFn, nullptr);
    ASSERT_EQ(CBFn->getNumUses(), 1U);
    ASSERT_TRUE(isa<CallInst>(CBFn->user_back()));
    ASSERT_EQ(CBFn->user_back()->getNumUses(), 1U);
    CheckedBarrier = cast<CallInst>(CBFn->user_back());

    Builder.restoreIP(
        OMPBuilder.createBarrier(Builder.saveIP(), OMPD_parallel, true));
    CBFn = M->getFunction("__kmpc_cancel_barrier");
    BFn = M->getFunction("__kmpc_barrier");
    ASSERT_NE(CBFn, nullptr);
    ASSERT_NE(BFn, nullptr);
    ASSERT_EQ(CBFn->getNumUses(), 1U);
    ASSERT_EQ(BFn->getNumUses(), 1U);
    ASSERT_TRUE(isa<CallInst>(BFn->user_back()));
    ASSERT_EQ(BFn->user_back()->getNumUses(), 0U);

    Builder.restoreIP(OMPBuilder.createBarrier(Builder.saveIP(), OMPD_parallel,
                                               false, false));
    ASSERT_EQ(CBFn->getNumUses(), 2U);
    ASSERT_EQ(BFn->getNumUses(), 1U);
    ASSERT_TRUE(CBFn->user_back() != CheckedBarrier);
    ASSERT_TRUE(isa<CallInst>(CBFn->user_back()));
    ASSERT_EQ(CBFn->user_back()->getNumUses(), 0U);
  };

  auto PrivCB = [&](InsertPointTy, InsertPointTy, Value &V, Value &,
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

  IRBuilder<>::InsertPoint AllocaIP(&F->getEntryBlock(),
                                    F->getEntryBlock().getFirstInsertionPt());
  IRBuilder<>::InsertPoint AfterIP =
      OMPBuilder.createParallel(Loc, AllocaIP, BodyGenCB, PrivCB, FiniCB,
                                Builder.CreateIsNotNull(F->arg_begin()),
                                nullptr, OMP_PROC_BIND_default, true);

  EXPECT_EQ(NumBodiesGenerated, 1U);
  EXPECT_EQ(NumPrivatizedVars, 0U);
  EXPECT_EQ(NumFinalizationPoints, 2U);
  EXPECT_EQ(FakeDestructor->getNumUses(), 2U);

  Builder.restoreIP(AfterIP);
  Builder.CreateRetVoid();
  OMPBuilder.finalize();

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
    if (!isa<ReturnInst>(ExitBB->front())) {
      ASSERT_TRUE(isa<BranchInst>(ExitBB->front()));
      ASSERT_EQ(cast<BranchInst>(ExitBB->front()).getNumSuccessors(), 1U);
      ASSERT_TRUE(isa<ReturnInst>(
          cast<BranchInst>(ExitBB->front()).getSuccessor(0)->front()));
    }
  }
}

TEST_F(OpenMPIRBuilderTest, ParallelForwardAsPointers) {
  OpenMPIRBuilder OMPBuilder(*M);
  OMPBuilder.initialize();
  F->setName("func");
  IRBuilder<> Builder(BB);
  OpenMPIRBuilder::LocationDescription Loc({Builder.saveIP(), DL});
  using InsertPointTy = OpenMPIRBuilder::InsertPointTy;

  Type *I32Ty = Type::getInt32Ty(M->getContext());
  Type *I32PtrTy = Type::getInt32PtrTy(M->getContext());
  Type *StructTy = StructType::get(I32Ty, I32PtrTy);
  Type *StructPtrTy = StructTy->getPointerTo();
  Type *VoidTy = Type::getVoidTy(M->getContext());
  FunctionCallee RetI32Func = M->getOrInsertFunction("ret_i32", I32Ty);
  FunctionCallee TakeI32Func =
      M->getOrInsertFunction("take_i32", VoidTy, I32Ty);
  FunctionCallee RetI32PtrFunc = M->getOrInsertFunction("ret_i32ptr", I32PtrTy);
  FunctionCallee TakeI32PtrFunc =
      M->getOrInsertFunction("take_i32ptr", VoidTy, I32PtrTy);
  FunctionCallee RetStructFunc = M->getOrInsertFunction("ret_struct", StructTy);
  FunctionCallee TakeStructFunc =
      M->getOrInsertFunction("take_struct", VoidTy, StructTy);
  FunctionCallee RetStructPtrFunc =
      M->getOrInsertFunction("ret_structptr", StructPtrTy);
  FunctionCallee TakeStructPtrFunc =
      M->getOrInsertFunction("take_structPtr", VoidTy, StructPtrTy);
  Value *I32Val = Builder.CreateCall(RetI32Func);
  Value *I32PtrVal = Builder.CreateCall(RetI32PtrFunc);
  Value *StructVal = Builder.CreateCall(RetStructFunc);
  Value *StructPtrVal = Builder.CreateCall(RetStructPtrFunc);

  Instruction *Internal;
  auto BodyGenCB = [&](InsertPointTy AllocaIP, InsertPointTy CodeGenIP,
                       BasicBlock &ContinuationBB) {
    IRBuilder<>::InsertPointGuard Guard(Builder);
    Builder.restoreIP(CodeGenIP);
    Internal = Builder.CreateCall(TakeI32Func, I32Val);
    Builder.CreateCall(TakeI32PtrFunc, I32PtrVal);
    Builder.CreateCall(TakeStructFunc, StructVal);
    Builder.CreateCall(TakeStructPtrFunc, StructPtrVal);
  };
  auto PrivCB = [&](InsertPointTy AllocaIP, InsertPointTy CodeGenIP, Value &,
                    Value &Inner, Value *&ReplacementValue) {
    ReplacementValue = &Inner;
    return CodeGenIP;
  };
  auto FiniCB = [](InsertPointTy) {};

  IRBuilder<>::InsertPoint AllocaIP(&F->getEntryBlock(),
                                    F->getEntryBlock().getFirstInsertionPt());
  IRBuilder<>::InsertPoint AfterIP =
      OMPBuilder.createParallel(Loc, AllocaIP, BodyGenCB, PrivCB, FiniCB,
                                nullptr, nullptr, OMP_PROC_BIND_default, false);
  Builder.restoreIP(AfterIP);
  Builder.CreateRetVoid();

  OMPBuilder.finalize();

  EXPECT_FALSE(verifyModule(*M, &errs()));
  Function *OutlinedFn = Internal->getFunction();

  Type *Arg2Type = OutlinedFn->getArg(2)->getType();
  EXPECT_TRUE(Arg2Type->isPointerTy());
  EXPECT_EQ(Arg2Type->getPointerElementType(), I32Ty);

  // Arguments that need to be passed through pointers and reloaded will get
  // used earlier in the functions and therefore will appear first in the
  // argument list after outlining.
  Type *Arg3Type = OutlinedFn->getArg(3)->getType();
  EXPECT_TRUE(Arg3Type->isPointerTy());
  EXPECT_EQ(Arg3Type->getPointerElementType(), StructTy);

  Type *Arg4Type = OutlinedFn->getArg(4)->getType();
  EXPECT_EQ(Arg4Type, I32PtrTy);

  Type *Arg5Type = OutlinedFn->getArg(5)->getType();
  EXPECT_EQ(Arg5Type, StructPtrTy);
}

TEST_F(OpenMPIRBuilderTest, CanonicalLoopSimple) {
  using InsertPointTy = OpenMPIRBuilder::InsertPointTy;
  OpenMPIRBuilder OMPBuilder(*M);
  OMPBuilder.initialize();
  IRBuilder<> Builder(BB);
  OpenMPIRBuilder::LocationDescription Loc({Builder.saveIP(), DL});
  Value *TripCount = F->getArg(0);

  unsigned NumBodiesGenerated = 0;
  auto LoopBodyGenCB = [&](InsertPointTy CodeGenIP, llvm::Value *LC) {
    NumBodiesGenerated += 1;

    Builder.restoreIP(CodeGenIP);

    Value *Cmp = Builder.CreateICmpEQ(LC, TripCount);
    Instruction *ThenTerm, *ElseTerm;
    SplitBlockAndInsertIfThenElse(Cmp, CodeGenIP.getBlock()->getTerminator(),
                                  &ThenTerm, &ElseTerm);
  };

  CanonicalLoopInfo *Loop =
      OMPBuilder.createCanonicalLoop(Loc, LoopBodyGenCB, TripCount);

  Builder.restoreIP(Loop->getAfterIP());
  ReturnInst *RetInst = Builder.CreateRetVoid();
  OMPBuilder.finalize();

  Loop->assertOK();
  EXPECT_FALSE(verifyModule(*M, &errs()));

  EXPECT_EQ(NumBodiesGenerated, 1U);

  // Verify control flow structure (in addition to Loop->assertOK()).
  EXPECT_EQ(Loop->getPreheader()->getSinglePredecessor(), &F->getEntryBlock());
  EXPECT_EQ(Loop->getAfter(), Builder.GetInsertBlock());

  Instruction *IndVar = Loop->getIndVar();
  EXPECT_TRUE(isa<PHINode>(IndVar));
  EXPECT_EQ(IndVar->getType(), TripCount->getType());
  EXPECT_EQ(IndVar->getParent(), Loop->getHeader());

  EXPECT_EQ(Loop->getTripCount(), TripCount);

  BasicBlock *Body = Loop->getBody();
  Instruction *CmpInst = &Body->getInstList().front();
  EXPECT_TRUE(isa<ICmpInst>(CmpInst));
  EXPECT_EQ(CmpInst->getOperand(0), IndVar);

  BasicBlock *LatchPred = Loop->getLatch()->getSinglePredecessor();
  EXPECT_TRUE(llvm::all_of(successors(Body), [=](BasicBlock *SuccBB) {
    return SuccBB->getSingleSuccessor() == LatchPred;
  }));

  EXPECT_EQ(&Loop->getAfter()->front(), RetInst);
}

TEST_F(OpenMPIRBuilderTest, CanonicalLoopBounds) {
  using InsertPointTy = OpenMPIRBuilder::InsertPointTy;
  OpenMPIRBuilder OMPBuilder(*M);
  OMPBuilder.initialize();
  IRBuilder<> Builder(BB);

  // Check the trip count is computed correctly. We generate the canonical loop
  // but rely on the IRBuilder's constant folder to compute the final result
  // since all inputs are constant. To verify overflow situations, limit the
  // trip count / loop counter widths to 16 bits.
  auto EvalTripCount = [&](int64_t Start, int64_t Stop, int64_t Step,
                           bool IsSigned, bool InclusiveStop) -> int64_t {
    OpenMPIRBuilder::LocationDescription Loc({Builder.saveIP(), DL});
    Type *LCTy = Type::getInt16Ty(Ctx);
    Value *StartVal = ConstantInt::get(LCTy, Start);
    Value *StopVal = ConstantInt::get(LCTy, Stop);
    Value *StepVal = ConstantInt::get(LCTy, Step);
    auto LoopBodyGenCB = [&](InsertPointTy CodeGenIP, llvm::Value *LC) {};
    CanonicalLoopInfo *Loop =
        OMPBuilder.createCanonicalLoop(Loc, LoopBodyGenCB, StartVal, StopVal,
                                       StepVal, IsSigned, InclusiveStop);
    Loop->assertOK();
    Builder.restoreIP(Loop->getAfterIP());
    Value *TripCount = Loop->getTripCount();
    return cast<ConstantInt>(TripCount)->getValue().getZExtValue();
  };

  EXPECT_EQ(EvalTripCount(0, 0, 1, false, false), 0);
  EXPECT_EQ(EvalTripCount(0, 1, 2, false, false), 1);
  EXPECT_EQ(EvalTripCount(0, 42, 1, false, false), 42);
  EXPECT_EQ(EvalTripCount(0, 42, 2, false, false), 21);
  EXPECT_EQ(EvalTripCount(21, 42, 1, false, false), 21);
  EXPECT_EQ(EvalTripCount(0, 5, 5, false, false), 1);
  EXPECT_EQ(EvalTripCount(0, 9, 5, false, false), 2);
  EXPECT_EQ(EvalTripCount(0, 11, 5, false, false), 3);
  EXPECT_EQ(EvalTripCount(0, 0xFFFF, 1, false, false), 0xFFFF);
  EXPECT_EQ(EvalTripCount(0xFFFF, 0, 1, false, false), 0);
  EXPECT_EQ(EvalTripCount(0xFFFE, 0xFFFF, 1, false, false), 1);
  EXPECT_EQ(EvalTripCount(0, 0xFFFF, 0x100, false, false), 0x100);
  EXPECT_EQ(EvalTripCount(0, 0xFFFF, 0xFFFF, false, false), 1);

  EXPECT_EQ(EvalTripCount(0, 6, 5, false, false), 2);
  EXPECT_EQ(EvalTripCount(0, 0xFFFF, 0xFFFE, false, false), 2);
  EXPECT_EQ(EvalTripCount(0, 0, 1, false, true), 1);
  EXPECT_EQ(EvalTripCount(0, 0, 0xFFFF, false, true), 1);
  EXPECT_EQ(EvalTripCount(0, 0xFFFE, 1, false, true), 0xFFFF);
  EXPECT_EQ(EvalTripCount(0, 0xFFFE, 2, false, true), 0x8000);

  EXPECT_EQ(EvalTripCount(0, 0, -1, true, false), 0);
  EXPECT_EQ(EvalTripCount(0, 1, -1, true, true), 0);
  EXPECT_EQ(EvalTripCount(20, 5, -5, true, false), 3);
  EXPECT_EQ(EvalTripCount(20, 5, -5, true, true), 4);
  EXPECT_EQ(EvalTripCount(-4, -2, 2, true, false), 1);
  EXPECT_EQ(EvalTripCount(-4, -3, 2, true, false), 1);
  EXPECT_EQ(EvalTripCount(-4, -2, 2, true, true), 2);

  EXPECT_EQ(EvalTripCount(INT16_MIN, 0, 1, true, false), 0x8000);
  EXPECT_EQ(EvalTripCount(INT16_MIN, 0, 1, true, true), 0x8001);
  EXPECT_EQ(EvalTripCount(INT16_MIN, 0x7FFF, 1, true, false), 0xFFFF);
  EXPECT_EQ(EvalTripCount(INT16_MIN + 1, 0x7FFF, 1, true, true), 0xFFFF);
  EXPECT_EQ(EvalTripCount(INT16_MIN, 0, 0x7FFF, true, false), 2);
  EXPECT_EQ(EvalTripCount(0x7FFF, 0, -1, true, false), 0x7FFF);
  EXPECT_EQ(EvalTripCount(0, INT16_MIN, -1, true, false), 0x8000);
  EXPECT_EQ(EvalTripCount(0, INT16_MIN, -16, true, false), 0x800);
  EXPECT_EQ(EvalTripCount(0x7FFF, INT16_MIN, -1, true, false), 0xFFFF);
  EXPECT_EQ(EvalTripCount(0x7FFF, 1, INT16_MIN, true, false), 1);
  EXPECT_EQ(EvalTripCount(0x7FFF, -1, INT16_MIN, true, true), 2);

  // Finalize the function and verify it.
  Builder.CreateRetVoid();
  OMPBuilder.finalize();
  EXPECT_FALSE(verifyModule(*M, &errs()));
}

TEST_F(OpenMPIRBuilderTest, StaticWorkShareLoop) {
  using InsertPointTy = OpenMPIRBuilder::InsertPointTy;
  OpenMPIRBuilder OMPBuilder(*M);
  OMPBuilder.initialize();
  IRBuilder<> Builder(BB);
  OpenMPIRBuilder::LocationDescription Loc({Builder.saveIP(), DL});

  Type *LCTy = Type::getInt32Ty(Ctx);
  Value *StartVal = ConstantInt::get(LCTy, 10);
  Value *StopVal = ConstantInt::get(LCTy, 52);
  Value *StepVal = ConstantInt::get(LCTy, 2);
  auto LoopBodyGen = [&](InsertPointTy, llvm::Value *) {};

  CanonicalLoopInfo *CLI = OMPBuilder.createCanonicalLoop(
      Loc, LoopBodyGen, StartVal, StopVal, StepVal,
      /*IsSigned=*/false, /*InclusiveStop=*/false);

  Builder.SetInsertPoint(BB, BB->getFirstInsertionPt());
  InsertPointTy AllocaIP = Builder.saveIP();

  CLI = OMPBuilder.createStaticWorkshareLoop(Loc, CLI, AllocaIP,
                                             /*NeedsBarrier=*/true);
  auto AllocaIter = BB->begin();
  ASSERT_GE(std::distance(BB->begin(), BB->end()), 4);
  AllocaInst *PLastIter = dyn_cast<AllocaInst>(&*(AllocaIter++));
  AllocaInst *PLowerBound = dyn_cast<AllocaInst>(&*(AllocaIter++));
  AllocaInst *PUpperBound = dyn_cast<AllocaInst>(&*(AllocaIter++));
  AllocaInst *PStride = dyn_cast<AllocaInst>(&*(AllocaIter++));
  EXPECT_NE(PLastIter, nullptr);
  EXPECT_NE(PLowerBound, nullptr);
  EXPECT_NE(PUpperBound, nullptr);
  EXPECT_NE(PStride, nullptr);

  auto PreheaderIter = CLI->getPreheader()->begin();
  ASSERT_GE(
      std::distance(CLI->getPreheader()->begin(), CLI->getPreheader()->end()),
      7);
  StoreInst *LowerBoundStore = dyn_cast<StoreInst>(&*(PreheaderIter++));
  StoreInst *UpperBoundStore = dyn_cast<StoreInst>(&*(PreheaderIter++));
  StoreInst *StrideStore = dyn_cast<StoreInst>(&*(PreheaderIter++));
  ASSERT_NE(LowerBoundStore, nullptr);
  ASSERT_NE(UpperBoundStore, nullptr);
  ASSERT_NE(StrideStore, nullptr);

  auto *OrigLowerBound =
      dyn_cast<ConstantInt>(LowerBoundStore->getValueOperand());
  auto *OrigUpperBound =
      dyn_cast<ConstantInt>(UpperBoundStore->getValueOperand());
  auto *OrigStride = dyn_cast<ConstantInt>(StrideStore->getValueOperand());
  ASSERT_NE(OrigLowerBound, nullptr);
  ASSERT_NE(OrigUpperBound, nullptr);
  ASSERT_NE(OrigStride, nullptr);
  EXPECT_EQ(OrigLowerBound->getValue(), 0);
  EXPECT_EQ(OrigUpperBound->getValue(), 20);
  EXPECT_EQ(OrigStride->getValue(), 1);

  // Check that the loop IV is updated to account for the lower bound returned
  // by the OpenMP runtime call.
  BinaryOperator *Add = dyn_cast<BinaryOperator>(&CLI->getBody()->front());
  EXPECT_EQ(Add->getOperand(0), CLI->getIndVar());
  auto *LoadedLowerBound = dyn_cast<LoadInst>(Add->getOperand(1));
  ASSERT_NE(LoadedLowerBound, nullptr);
  EXPECT_EQ(LoadedLowerBound->getPointerOperand(), PLowerBound);

  // Check that the trip count is updated to account for the lower and upper
  // bounds return by the OpenMP runtime call.
  auto *AddOne = dyn_cast<Instruction>(CLI->getTripCount());
  ASSERT_NE(AddOne, nullptr);
  ASSERT_TRUE(AddOne->isBinaryOp());
  auto *One = dyn_cast<ConstantInt>(AddOne->getOperand(1));
  ASSERT_NE(One, nullptr);
  EXPECT_EQ(One->getValue(), 1);
  auto *Difference = dyn_cast<Instruction>(AddOne->getOperand(0));
  ASSERT_NE(Difference, nullptr);
  ASSERT_TRUE(Difference->isBinaryOp());
  EXPECT_EQ(Difference->getOperand(1), LoadedLowerBound);
  auto *LoadedUpperBound = dyn_cast<LoadInst>(Difference->getOperand(0));
  ASSERT_NE(LoadedUpperBound, nullptr);
  EXPECT_EQ(LoadedUpperBound->getPointerOperand(), PUpperBound);

  // The original loop iterator should only be used in the condition, in the
  // increment and in the statement that adds the lower bound to it.
  Value *IV = CLI->getIndVar();
  EXPECT_EQ(std::distance(IV->use_begin(), IV->use_end()), 3);

  // The exit block should contain the "fini" call and the barrier call,
  // plus the call to obtain the thread ID.
  BasicBlock *ExitBlock = CLI->getExit();
  size_t NumCallsInExitBlock =
      count_if(*ExitBlock, [](Instruction &I) { return isa<CallInst>(I); });
  EXPECT_EQ(NumCallsInExitBlock, 3u);
}

TEST_F(OpenMPIRBuilderTest, MasterDirective) {
  using InsertPointTy = OpenMPIRBuilder::InsertPointTy;
  OpenMPIRBuilder OMPBuilder(*M);
  OMPBuilder.initialize();
  F->setName("func");
  IRBuilder<> Builder(BB);

  OpenMPIRBuilder::LocationDescription Loc({Builder.saveIP(), DL});

  AllocaInst *PrivAI = nullptr;

  BasicBlock *EntryBB = nullptr;
  BasicBlock *ExitBB = nullptr;
  BasicBlock *ThenBB = nullptr;

  auto BodyGenCB = [&](InsertPointTy AllocaIP, InsertPointTy CodeGenIP,
                       BasicBlock &FiniBB) {
    if (AllocaIP.isSet())
      Builder.restoreIP(AllocaIP);
    else
      Builder.SetInsertPoint(&*(F->getEntryBlock().getFirstInsertionPt()));
    PrivAI = Builder.CreateAlloca(F->arg_begin()->getType());
    Builder.CreateStore(F->arg_begin(), PrivAI);

    llvm::BasicBlock *CodeGenIPBB = CodeGenIP.getBlock();
    llvm::Instruction *CodeGenIPInst = &*CodeGenIP.getPoint();
    EXPECT_EQ(CodeGenIPBB->getTerminator(), CodeGenIPInst);

    Builder.restoreIP(CodeGenIP);

    // collect some info for checks later
    ExitBB = FiniBB.getUniqueSuccessor();
    ThenBB = Builder.GetInsertBlock();
    EntryBB = ThenBB->getUniquePredecessor();

    // simple instructions for body
    Value *PrivLoad = Builder.CreateLoad(PrivAI, "local.use");
    Builder.CreateICmpNE(F->arg_begin(), PrivLoad);
  };

  auto FiniCB = [&](InsertPointTy IP) {
    BasicBlock *IPBB = IP.getBlock();
    EXPECT_NE(IPBB->end(), IP.getPoint());
  };

  Builder.restoreIP(OMPBuilder.createMaster(Builder, BodyGenCB, FiniCB));
  Value *EntryBBTI = EntryBB->getTerminator();
  EXPECT_NE(EntryBBTI, nullptr);
  EXPECT_TRUE(isa<BranchInst>(EntryBBTI));
  BranchInst *EntryBr = cast<BranchInst>(EntryBB->getTerminator());
  EXPECT_TRUE(EntryBr->isConditional());
  EXPECT_EQ(EntryBr->getSuccessor(0), ThenBB);
  EXPECT_EQ(ThenBB->getUniqueSuccessor(), ExitBB);
  EXPECT_EQ(EntryBr->getSuccessor(1), ExitBB);

  CmpInst *CondInst = cast<CmpInst>(EntryBr->getCondition());
  EXPECT_TRUE(isa<CallInst>(CondInst->getOperand(0)));

  CallInst *MasterEntryCI = cast<CallInst>(CondInst->getOperand(0));
  EXPECT_EQ(MasterEntryCI->getNumArgOperands(), 2U);
  EXPECT_EQ(MasterEntryCI->getCalledFunction()->getName(), "__kmpc_master");
  EXPECT_TRUE(isa<GlobalVariable>(MasterEntryCI->getArgOperand(0)));

  CallInst *MasterEndCI = nullptr;
  for (auto &FI : *ThenBB) {
    Instruction *cur = &FI;
    if (isa<CallInst>(cur)) {
      MasterEndCI = cast<CallInst>(cur);
      if (MasterEndCI->getCalledFunction()->getName() == "__kmpc_end_master")
        break;
      MasterEndCI = nullptr;
    }
  }
  EXPECT_NE(MasterEndCI, nullptr);
  EXPECT_EQ(MasterEndCI->getNumArgOperands(), 2U);
  EXPECT_TRUE(isa<GlobalVariable>(MasterEndCI->getArgOperand(0)));
  EXPECT_EQ(MasterEndCI->getArgOperand(1), MasterEntryCI->getArgOperand(1));
}

TEST_F(OpenMPIRBuilderTest, CriticalDirective) {
  using InsertPointTy = OpenMPIRBuilder::InsertPointTy;
  OpenMPIRBuilder OMPBuilder(*M);
  OMPBuilder.initialize();
  F->setName("func");
  IRBuilder<> Builder(BB);

  OpenMPIRBuilder::LocationDescription Loc({Builder.saveIP(), DL});

  AllocaInst *PrivAI = Builder.CreateAlloca(F->arg_begin()->getType());

  BasicBlock *EntryBB = nullptr;

  auto BodyGenCB = [&](InsertPointTy AllocaIP, InsertPointTy CodeGenIP,
                       BasicBlock &FiniBB) {
    // collect some info for checks later
    EntryBB = FiniBB.getUniquePredecessor();

    // actual start for bodyCB
    llvm::BasicBlock *CodeGenIPBB = CodeGenIP.getBlock();
    llvm::Instruction *CodeGenIPInst = &*CodeGenIP.getPoint();
    EXPECT_EQ(CodeGenIPBB->getTerminator(), CodeGenIPInst);
    EXPECT_EQ(EntryBB, CodeGenIPBB);

    // body begin
    Builder.restoreIP(CodeGenIP);
    Builder.CreateStore(F->arg_begin(), PrivAI);
    Value *PrivLoad = Builder.CreateLoad(PrivAI, "local.use");
    Builder.CreateICmpNE(F->arg_begin(), PrivLoad);
  };

  auto FiniCB = [&](InsertPointTy IP) {
    BasicBlock *IPBB = IP.getBlock();
    EXPECT_NE(IPBB->end(), IP.getPoint());
  };

  Builder.restoreIP(OMPBuilder.createCritical(Builder, BodyGenCB, FiniCB,
                                              "testCRT", nullptr));

  Value *EntryBBTI = EntryBB->getTerminator();
  EXPECT_EQ(EntryBBTI, nullptr);

  CallInst *CriticalEntryCI = nullptr;
  for (auto &EI : *EntryBB) {
    Instruction *cur = &EI;
    if (isa<CallInst>(cur)) {
      CriticalEntryCI = cast<CallInst>(cur);
      if (CriticalEntryCI->getCalledFunction()->getName() == "__kmpc_critical")
        break;
      CriticalEntryCI = nullptr;
    }
  }
  EXPECT_NE(CriticalEntryCI, nullptr);
  EXPECT_EQ(CriticalEntryCI->getNumArgOperands(), 3U);
  EXPECT_EQ(CriticalEntryCI->getCalledFunction()->getName(), "__kmpc_critical");
  EXPECT_TRUE(isa<GlobalVariable>(CriticalEntryCI->getArgOperand(0)));

  CallInst *CriticalEndCI = nullptr;
  for (auto &FI : *EntryBB) {
    Instruction *cur = &FI;
    if (isa<CallInst>(cur)) {
      CriticalEndCI = cast<CallInst>(cur);
      if (CriticalEndCI->getCalledFunction()->getName() ==
          "__kmpc_end_critical")
        break;
      CriticalEndCI = nullptr;
    }
  }
  EXPECT_NE(CriticalEndCI, nullptr);
  EXPECT_EQ(CriticalEndCI->getNumArgOperands(), 3U);
  EXPECT_TRUE(isa<GlobalVariable>(CriticalEndCI->getArgOperand(0)));
  EXPECT_EQ(CriticalEndCI->getArgOperand(1), CriticalEntryCI->getArgOperand(1));
  PointerType *CriticalNamePtrTy =
      PointerType::getUnqual(ArrayType::get(Type::getInt32Ty(Ctx), 8));
  EXPECT_EQ(CriticalEndCI->getArgOperand(2), CriticalEntryCI->getArgOperand(2));
  EXPECT_EQ(CriticalEndCI->getArgOperand(2)->getType(), CriticalNamePtrTy);
}

TEST_F(OpenMPIRBuilderTest, CopyinBlocks) {
  OpenMPIRBuilder OMPBuilder(*M);
  OMPBuilder.initialize();
  F->setName("func");
  IRBuilder<> Builder(BB);

  OpenMPIRBuilder::LocationDescription Loc({Builder.saveIP(), DL});

  IntegerType* Int32 = Type::getInt32Ty(M->getContext());
  AllocaInst* MasterAddress = Builder.CreateAlloca(Int32->getPointerTo());
	AllocaInst* PrivAddress = Builder.CreateAlloca(Int32->getPointerTo());

  BasicBlock *EntryBB = BB;

  OMPBuilder.createCopyinClauseBlocks(Builder.saveIP(), MasterAddress,
                                      PrivAddress, Int32, /*BranchtoEnd*/ true);

  BranchInst* EntryBr = dyn_cast_or_null<BranchInst>(EntryBB->getTerminator());

  EXPECT_NE(EntryBr, nullptr);
  EXPECT_TRUE(EntryBr->isConditional());

  BasicBlock* NotMasterBB = EntryBr->getSuccessor(0);
  BasicBlock* CopyinEnd = EntryBr->getSuccessor(1);
  CmpInst* CMP = dyn_cast_or_null<CmpInst>(EntryBr->getCondition());

  EXPECT_NE(CMP, nullptr);
  EXPECT_NE(NotMasterBB, nullptr);
  EXPECT_NE(CopyinEnd, nullptr);

  BranchInst* NotMasterBr = dyn_cast_or_null<BranchInst>(NotMasterBB->getTerminator());
  EXPECT_NE(NotMasterBr, nullptr);
  EXPECT_FALSE(NotMasterBr->isConditional());
  EXPECT_EQ(CopyinEnd,NotMasterBr->getSuccessor(0));
}

TEST_F(OpenMPIRBuilderTest, SingleDirective) {
  using InsertPointTy = OpenMPIRBuilder::InsertPointTy;
  OpenMPIRBuilder OMPBuilder(*M);
  OMPBuilder.initialize();
  F->setName("func");
  IRBuilder<> Builder(BB);

  OpenMPIRBuilder::LocationDescription Loc({Builder.saveIP(), DL});

  AllocaInst *PrivAI = nullptr;

  BasicBlock *EntryBB = nullptr;
  BasicBlock *ExitBB = nullptr;
  BasicBlock *ThenBB = nullptr;

  auto BodyGenCB = [&](InsertPointTy AllocaIP, InsertPointTy CodeGenIP,
                       BasicBlock &FiniBB) {
    if (AllocaIP.isSet())
      Builder.restoreIP(AllocaIP);
    else
      Builder.SetInsertPoint(&*(F->getEntryBlock().getFirstInsertionPt()));
    PrivAI = Builder.CreateAlloca(F->arg_begin()->getType());
    Builder.CreateStore(F->arg_begin(), PrivAI);

    llvm::BasicBlock *CodeGenIPBB = CodeGenIP.getBlock();
    llvm::Instruction *CodeGenIPInst = &*CodeGenIP.getPoint();
    EXPECT_EQ(CodeGenIPBB->getTerminator(), CodeGenIPInst);

    Builder.restoreIP(CodeGenIP);

    // collect some info for checks later
    ExitBB = FiniBB.getUniqueSuccessor();
    ThenBB = Builder.GetInsertBlock();
    EntryBB = ThenBB->getUniquePredecessor();

    // simple instructions for body
    Value *PrivLoad = Builder.CreateLoad(PrivAI, "local.use");
    Builder.CreateICmpNE(F->arg_begin(), PrivLoad);
  };

  auto FiniCB = [&](InsertPointTy IP) {
    BasicBlock *IPBB = IP.getBlock();
    EXPECT_NE(IPBB->end(), IP.getPoint());
  };

  Builder.restoreIP(
      OMPBuilder.createSingle(Builder, BodyGenCB, FiniCB, /*DidIt*/ nullptr));
  Value *EntryBBTI = EntryBB->getTerminator();
  EXPECT_NE(EntryBBTI, nullptr);
  EXPECT_TRUE(isa<BranchInst>(EntryBBTI));
  BranchInst *EntryBr = cast<BranchInst>(EntryBB->getTerminator());
  EXPECT_TRUE(EntryBr->isConditional());
  EXPECT_EQ(EntryBr->getSuccessor(0), ThenBB);
  EXPECT_EQ(ThenBB->getUniqueSuccessor(), ExitBB);
  EXPECT_EQ(EntryBr->getSuccessor(1), ExitBB);

  CmpInst *CondInst = cast<CmpInst>(EntryBr->getCondition());
  EXPECT_TRUE(isa<CallInst>(CondInst->getOperand(0)));

  CallInst *SingleEntryCI = cast<CallInst>(CondInst->getOperand(0));
  EXPECT_EQ(SingleEntryCI->getNumArgOperands(), 2U);
  EXPECT_EQ(SingleEntryCI->getCalledFunction()->getName(), "__kmpc_single");
  EXPECT_TRUE(isa<GlobalVariable>(SingleEntryCI->getArgOperand(0)));

  CallInst *SingleEndCI = nullptr;
  for (auto &FI : *ThenBB) {
    Instruction *cur = &FI;
    if (isa<CallInst>(cur)) {
      SingleEndCI = cast<CallInst>(cur);
      if (SingleEndCI->getCalledFunction()->getName() == "__kmpc_end_single")
        break;
      SingleEndCI = nullptr;
    }
  }
  EXPECT_NE(SingleEndCI, nullptr);
  EXPECT_EQ(SingleEndCI->getNumArgOperands(), 2U);
  EXPECT_TRUE(isa<GlobalVariable>(SingleEndCI->getArgOperand(0)));
  EXPECT_EQ(SingleEndCI->getArgOperand(1), SingleEntryCI->getArgOperand(1));
}

} // namespace
