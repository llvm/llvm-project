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

/// Create an instruction that uses the values in \p Values. We use "printf"
/// just because it is often used for this purpose in test code, but it is never
/// executed here.
static CallInst *createPrintfCall(IRBuilder<> &Builder, StringRef FormatStr,
                                  ArrayRef<Value *> Values) {
  Module *M = Builder.GetInsertBlock()->getParent()->getParent();

  GlobalVariable *GV = Builder.CreateGlobalString(FormatStr, "", 0, M);
  Constant *Zero = ConstantInt::get(Type::getInt32Ty(M->getContext()), 0);
  Constant *Indices[] = {Zero, Zero};
  Constant *FormatStrConst =
      ConstantExpr::getInBoundsGetElementPtr(GV->getValueType(), GV, Indices);

  Function *PrintfDecl = M->getFunction("printf");
  if (!PrintfDecl) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    FunctionType *Ty = FunctionType::get(Builder.getInt32Ty(), true);
    PrintfDecl = Function::Create(Ty, Linkage, "printf", M);
  }

  SmallVector<Value *, 4> Args;
  Args.push_back(FormatStrConst);
  Args.append(Values.begin(), Values.end());
  return Builder.CreateCall(PrintfDecl, Args);
}

/// Verify that blocks in \p RefOrder are corresponds to the depth-first visit
/// order the control flow of \p F.
///
/// This is an easy way to verify the branching structure of the CFG without
/// checking every branch instruction individually. For the CFG of a
/// CanonicalLoopInfo, the Cond BB's terminating branch's first edge is entering
/// the body, i.e. the DFS order corresponds to the execution order with one
/// loop iteration.
static testing::AssertionResult
verifyDFSOrder(Function *F, ArrayRef<BasicBlock *> RefOrder) {
  ArrayRef<BasicBlock *>::iterator It = RefOrder.begin();
  ArrayRef<BasicBlock *>::iterator E = RefOrder.end();

  df_iterator_default_set<BasicBlock *, 16> Visited;
  auto DFS = llvm::depth_first_ext(&F->getEntryBlock(), Visited);

  BasicBlock *Prev = nullptr;
  for (BasicBlock *BB : DFS) {
    if (It != E && BB == *It) {
      Prev = *It;
      ++It;
    }
  }

  if (It == E)
    return testing::AssertionSuccess();
  if (!Prev)
    return testing::AssertionFailure()
           << "Did not find " << (*It)->getName() << " in control flow";
  return testing::AssertionFailure()
         << "Expected " << Prev->getName() << " before " << (*It)->getName()
         << " in control flow";
}

/// Verify that blocks in \p RefOrder are in the same relative order in the
/// linked lists of blocks in \p F. The linked list may contain additional
/// blocks in-between.
///
/// While the order in the linked list is not relevant for semantics, keeping
/// the order roughly in execution order makes its printout easier to read.
static testing::AssertionResult
verifyListOrder(Function *F, ArrayRef<BasicBlock *> RefOrder) {
  ArrayRef<BasicBlock *>::iterator It = RefOrder.begin();
  ArrayRef<BasicBlock *>::iterator E = RefOrder.end();

  BasicBlock *Prev = nullptr;
  for (BasicBlock &BB : *F) {
    if (It != E && &BB == *It) {
      Prev = *It;
      ++It;
    }
  }

  if (It == E)
    return testing::AssertionSuccess();
  if (!Prev)
    return testing::AssertionFailure() << "Did not find " << (*It)->getName()
                                       << " in function " << F->getName();
  return testing::AssertionFailure()
         << "Expected " << Prev->getName() << " before " << (*It)->getName()
         << " in function " << F->getName();
}

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

TEST_F(OpenMPIRBuilderTest, TileSingleLoop) {
  using InsertPointTy = OpenMPIRBuilder::InsertPointTy;
  OpenMPIRBuilder OMPBuilder(*M);
  OMPBuilder.initialize();
  F->setName("func");

  IRBuilder<> Builder(BB);
  OpenMPIRBuilder::LocationDescription Loc({Builder.saveIP(), DL});
  Value *TripCount = F->getArg(0);

  BasicBlock *BodyCode = nullptr;
  Instruction *Call = nullptr;
  auto LoopBodyGenCB = [&](InsertPointTy CodeGenIP, llvm::Value *LC) {
    Builder.restoreIP(CodeGenIP);
    BodyCode = Builder.GetInsertBlock();

    // Add something that consumes the induction variable to the body.
    Call = createPrintfCall(Builder, "%d\\n", {LC});
  };
  CanonicalLoopInfo *Loop =
      OMPBuilder.createCanonicalLoop(Loc, LoopBodyGenCB, TripCount);

  // Finalize the function.
  Builder.restoreIP(Loop->getAfterIP());
  Builder.CreateRetVoid();

  Instruction *OrigIndVar = Loop->getIndVar();
  EXPECT_EQ(Call->getOperand(1), OrigIndVar);

  // Tile the loop.
  Constant *TileSize = ConstantInt::get(Loop->getIndVarType(), APInt(32, 7));
  std::vector<CanonicalLoopInfo *> GenLoops =
      OMPBuilder.tileLoops(DL, {Loop}, {TileSize});

  OMPBuilder.finalize();
  EXPECT_FALSE(verifyModule(*M, &errs()));

  EXPECT_EQ(GenLoops.size(), 2u);
  CanonicalLoopInfo *Floor = GenLoops[0];
  CanonicalLoopInfo *Tile = GenLoops[1];

  BasicBlock *RefOrder[] = {
      Floor->getPreheader(), Floor->getHeader(),   Floor->getCond(),
      Floor->getBody(),      Tile->getPreheader(), Tile->getHeader(),
      Tile->getCond(),       Tile->getBody(),      BodyCode,
      Tile->getLatch(),      Tile->getExit(),      Tile->getAfter(),
      Floor->getLatch(),     Floor->getExit(),     Floor->getAfter(),
  };
  EXPECT_TRUE(verifyDFSOrder(F, RefOrder));
  EXPECT_TRUE(verifyListOrder(F, RefOrder));

  // Check the induction variable.
  EXPECT_EQ(Call->getParent(), BodyCode);
  auto *Shift = cast<AddOperator>(Call->getOperand(1));
  EXPECT_EQ(cast<Instruction>(Shift)->getParent(), Tile->getBody());
  EXPECT_EQ(Shift->getOperand(1), Tile->getIndVar());
  auto *Scale = cast<MulOperator>(Shift->getOperand(0));
  EXPECT_EQ(cast<Instruction>(Scale)->getParent(), Tile->getBody());
  EXPECT_EQ(Scale->getOperand(0), TileSize);
  EXPECT_EQ(Scale->getOperand(1), Floor->getIndVar());
}

TEST_F(OpenMPIRBuilderTest, TileNestedLoops) {
  using InsertPointTy = OpenMPIRBuilder::InsertPointTy;
  OpenMPIRBuilder OMPBuilder(*M);
  OMPBuilder.initialize();
  F->setName("func");

  IRBuilder<> Builder(BB);
  OpenMPIRBuilder::LocationDescription Loc({Builder.saveIP(), DL});
  Value *TripCount = F->getArg(0);
  Type *LCTy = TripCount->getType();

  BasicBlock *BodyCode = nullptr;
  CanonicalLoopInfo *InnerLoop = nullptr;
  auto OuterLoopBodyGenCB = [&](InsertPointTy OuterCodeGenIP,
                                llvm::Value *OuterLC) {
    auto InnerLoopBodyGenCB = [&](InsertPointTy InnerCodeGenIP,
                                  llvm::Value *InnerLC) {
      Builder.restoreIP(InnerCodeGenIP);
      BodyCode = Builder.GetInsertBlock();

      // Add something that consumes the induction variables to the body.
      createPrintfCall(Builder, "i=%d j=%d\\n", {OuterLC, InnerLC});
    };
    InnerLoop = OMPBuilder.createCanonicalLoop(
        OuterCodeGenIP, InnerLoopBodyGenCB, TripCount, "inner");
  };
  CanonicalLoopInfo *OuterLoop = OMPBuilder.createCanonicalLoop(
      Loc, OuterLoopBodyGenCB, TripCount, "outer");

  // Finalize the function.
  Builder.restoreIP(OuterLoop->getAfterIP());
  Builder.CreateRetVoid();

  // Tile to loop nest.
  Constant *OuterTileSize = ConstantInt::get(LCTy, APInt(32, 11));
  Constant *InnerTileSize = ConstantInt::get(LCTy, APInt(32, 7));
  std::vector<CanonicalLoopInfo *> GenLoops = OMPBuilder.tileLoops(
      DL, {OuterLoop, InnerLoop}, {OuterTileSize, InnerTileSize});

  OMPBuilder.finalize();
  EXPECT_FALSE(verifyModule(*M, &errs()));

  EXPECT_EQ(GenLoops.size(), 4u);
  CanonicalLoopInfo *Floor1 = GenLoops[0];
  CanonicalLoopInfo *Floor2 = GenLoops[1];
  CanonicalLoopInfo *Tile1 = GenLoops[2];
  CanonicalLoopInfo *Tile2 = GenLoops[3];

  BasicBlock *RefOrder[] = {
      Floor1->getPreheader(),
      Floor1->getHeader(),
      Floor1->getCond(),
      Floor1->getBody(),
      Floor2->getPreheader(),
      Floor2->getHeader(),
      Floor2->getCond(),
      Floor2->getBody(),
      Tile1->getPreheader(),
      Tile1->getHeader(),
      Tile1->getCond(),
      Tile1->getBody(),
      Tile2->getPreheader(),
      Tile2->getHeader(),
      Tile2->getCond(),
      Tile2->getBody(),
      BodyCode,
      Tile2->getLatch(),
      Tile2->getExit(),
      Tile2->getAfter(),
      Tile1->getLatch(),
      Tile1->getExit(),
      Tile1->getAfter(),
      Floor2->getLatch(),
      Floor2->getExit(),
      Floor2->getAfter(),
      Floor1->getLatch(),
      Floor1->getExit(),
      Floor1->getAfter(),
  };
  EXPECT_TRUE(verifyDFSOrder(F, RefOrder));
  EXPECT_TRUE(verifyListOrder(F, RefOrder));
}

TEST_F(OpenMPIRBuilderTest, TileNestedLoopsWithBounds) {
  using InsertPointTy = OpenMPIRBuilder::InsertPointTy;
  OpenMPIRBuilder OMPBuilder(*M);
  OMPBuilder.initialize();
  F->setName("func");

  IRBuilder<> Builder(BB);
  Value *TripCount = F->getArg(0);
  Type *LCTy = TripCount->getType();

  Value *OuterStartVal = ConstantInt::get(LCTy, 2);
  Value *OuterStopVal = TripCount;
  Value *OuterStep = ConstantInt::get(LCTy, 5);
  Value *InnerStartVal = ConstantInt::get(LCTy, 13);
  Value *InnerStopVal = TripCount;
  Value *InnerStep = ConstantInt::get(LCTy, 3);

  // Fix an insertion point for ComputeIP.
  BasicBlock *LoopNextEnter =
      BasicBlock::Create(M->getContext(), "loopnest.enter", F,
                         Builder.GetInsertBlock()->getNextNode());
  BranchInst *EnterBr = Builder.CreateBr(LoopNextEnter);
  InsertPointTy ComputeIP{EnterBr->getParent(), EnterBr->getIterator()};

  InsertPointTy LoopIP{LoopNextEnter, LoopNextEnter->begin()};
  OpenMPIRBuilder::LocationDescription Loc({LoopIP, DL});

  BasicBlock *BodyCode = nullptr;
  CanonicalLoopInfo *InnerLoop = nullptr;
  CallInst *Call = nullptr;
  auto OuterLoopBodyGenCB = [&](InsertPointTy OuterCodeGenIP,
                                llvm::Value *OuterLC) {
    auto InnerLoopBodyGenCB = [&](InsertPointTy InnerCodeGenIP,
                                  llvm::Value *InnerLC) {
      Builder.restoreIP(InnerCodeGenIP);
      BodyCode = Builder.GetInsertBlock();

      // Add something that consumes the induction variable to the body.
      Call = createPrintfCall(Builder, "i=%d j=%d\\n", {OuterLC, InnerLC});
    };
    InnerLoop = OMPBuilder.createCanonicalLoop(
        OuterCodeGenIP, InnerLoopBodyGenCB, InnerStartVal, InnerStopVal,
        InnerStep, false, false, ComputeIP, "inner");
  };
  CanonicalLoopInfo *OuterLoop = OMPBuilder.createCanonicalLoop(
      Loc, OuterLoopBodyGenCB, OuterStartVal, OuterStopVal, OuterStep, false,
      false, ComputeIP, "outer");

  // Finalize the function
  Builder.restoreIP(OuterLoop->getAfterIP());
  Builder.CreateRetVoid();

  // Tile the loop nest.
  Constant *TileSize0 = ConstantInt::get(LCTy, APInt(32, 11));
  Constant *TileSize1 = ConstantInt::get(LCTy, APInt(32, 7));
  std::vector<CanonicalLoopInfo *> GenLoops =
      OMPBuilder.tileLoops(DL, {OuterLoop, InnerLoop}, {TileSize0, TileSize1});

  OMPBuilder.finalize();
  EXPECT_FALSE(verifyModule(*M, &errs()));

  EXPECT_EQ(GenLoops.size(), 4u);
  CanonicalLoopInfo *Floor0 = GenLoops[0];
  CanonicalLoopInfo *Floor1 = GenLoops[1];
  CanonicalLoopInfo *Tile0 = GenLoops[2];
  CanonicalLoopInfo *Tile1 = GenLoops[3];

  BasicBlock *RefOrder[] = {
      Floor0->getPreheader(),
      Floor0->getHeader(),
      Floor0->getCond(),
      Floor0->getBody(),
      Floor1->getPreheader(),
      Floor1->getHeader(),
      Floor1->getCond(),
      Floor1->getBody(),
      Tile0->getPreheader(),
      Tile0->getHeader(),
      Tile0->getCond(),
      Tile0->getBody(),
      Tile1->getPreheader(),
      Tile1->getHeader(),
      Tile1->getCond(),
      Tile1->getBody(),
      BodyCode,
      Tile1->getLatch(),
      Tile1->getExit(),
      Tile1->getAfter(),
      Tile0->getLatch(),
      Tile0->getExit(),
      Tile0->getAfter(),
      Floor1->getLatch(),
      Floor1->getExit(),
      Floor1->getAfter(),
      Floor0->getLatch(),
      Floor0->getExit(),
      Floor0->getAfter(),
  };
  EXPECT_TRUE(verifyDFSOrder(F, RefOrder));
  EXPECT_TRUE(verifyListOrder(F, RefOrder));

  EXPECT_EQ(Call->getParent(), BodyCode);

  auto *RangeShift0 = cast<AddOperator>(Call->getOperand(1));
  EXPECT_EQ(RangeShift0->getOperand(1), OuterStartVal);
  auto *RangeScale0 = cast<MulOperator>(RangeShift0->getOperand(0));
  EXPECT_EQ(RangeScale0->getOperand(1), OuterStep);
  auto *TileShift0 = cast<AddOperator>(RangeScale0->getOperand(0));
  EXPECT_EQ(cast<Instruction>(TileShift0)->getParent(), Tile1->getBody());
  EXPECT_EQ(TileShift0->getOperand(1), Tile0->getIndVar());
  auto *TileScale0 = cast<MulOperator>(TileShift0->getOperand(0));
  EXPECT_EQ(cast<Instruction>(TileScale0)->getParent(), Tile1->getBody());
  EXPECT_EQ(TileScale0->getOperand(0), TileSize0);
  EXPECT_EQ(TileScale0->getOperand(1), Floor0->getIndVar());

  auto *RangeShift1 = cast<AddOperator>(Call->getOperand(2));
  EXPECT_EQ(cast<Instruction>(RangeShift1)->getParent(), BodyCode);
  EXPECT_EQ(RangeShift1->getOperand(1), InnerStartVal);
  auto *RangeScale1 = cast<MulOperator>(RangeShift1->getOperand(0));
  EXPECT_EQ(cast<Instruction>(RangeScale1)->getParent(), BodyCode);
  EXPECT_EQ(RangeScale1->getOperand(1), InnerStep);
  auto *TileShift1 = cast<AddOperator>(RangeScale1->getOperand(0));
  EXPECT_EQ(cast<Instruction>(TileShift1)->getParent(), Tile1->getBody());
  EXPECT_EQ(TileShift1->getOperand(1), Tile1->getIndVar());
  auto *TileScale1 = cast<MulOperator>(TileShift1->getOperand(0));
  EXPECT_EQ(cast<Instruction>(TileScale1)->getParent(), Tile1->getBody());
  EXPECT_EQ(TileScale1->getOperand(0), TileSize1);
  EXPECT_EQ(TileScale1->getOperand(1), Floor1->getIndVar());
}

TEST_F(OpenMPIRBuilderTest, TileSingleLoopCounts) {
  using InsertPointTy = OpenMPIRBuilder::InsertPointTy;
  OpenMPIRBuilder OMPBuilder(*M);
  OMPBuilder.initialize();
  IRBuilder<> Builder(BB);

  // Create a loop, tile it, and extract its trip count. All input values are
  // constant and IRBuilder evaluates all-constant arithmetic inplace, such that
  // the floor trip count itself will be a ConstantInt. Unfortunately we cannot
  // do the same for the tile loop.
  auto GetFloorCount = [&](int64_t Start, int64_t Stop, int64_t Step,
                           bool IsSigned, bool InclusiveStop,
                           int64_t TileSize) -> uint64_t {
    OpenMPIRBuilder::LocationDescription Loc(Builder.saveIP(), DL);
    Type *LCTy = Type::getInt16Ty(Ctx);
    Value *StartVal = ConstantInt::get(LCTy, Start);
    Value *StopVal = ConstantInt::get(LCTy, Stop);
    Value *StepVal = ConstantInt::get(LCTy, Step);

    // Generate a loop.
    auto LoopBodyGenCB = [&](InsertPointTy CodeGenIP, llvm::Value *LC) {};
    CanonicalLoopInfo *Loop =
        OMPBuilder.createCanonicalLoop(Loc, LoopBodyGenCB, StartVal, StopVal,
                                       StepVal, IsSigned, InclusiveStop);

    // Tile the loop.
    Value *TileSizeVal = ConstantInt::get(LCTy, TileSize);
    std::vector<CanonicalLoopInfo *> GenLoops =
        OMPBuilder.tileLoops(Loc.DL, {Loop}, {TileSizeVal});

    // Set the insertion pointer to after loop, where the next loop will be
    // emitted.
    Builder.restoreIP(Loop->getAfterIP());

    // Extract the trip count.
    CanonicalLoopInfo *FloorLoop = GenLoops[0];
    Value *FloorTripCount = FloorLoop->getTripCount();
    return cast<ConstantInt>(FloorTripCount)->getValue().getZExtValue();
  };

  // Empty iteration domain.
  EXPECT_EQ(GetFloorCount(0, 0, 1, false, false, 7), 0u);
  EXPECT_EQ(GetFloorCount(0, -1, 1, false, true, 7), 0u);
  EXPECT_EQ(GetFloorCount(-1, -1, -1, true, false, 7), 0u);
  EXPECT_EQ(GetFloorCount(-1, 0, -1, true, true, 7), 0u);
  EXPECT_EQ(GetFloorCount(-1, -1, 3, true, false, 7), 0u);

  // Only complete tiles.
  EXPECT_EQ(GetFloorCount(0, 14, 1, false, false, 7), 2u);
  EXPECT_EQ(GetFloorCount(0, 14, 1, false, false, 7), 2u);
  EXPECT_EQ(GetFloorCount(1, 15, 1, false, false, 7), 2u);
  EXPECT_EQ(GetFloorCount(0, -14, -1, true, false, 7), 2u);
  EXPECT_EQ(GetFloorCount(-1, -14, -1, true, true, 7), 2u);
  EXPECT_EQ(GetFloorCount(0, 3 * 7 * 2, 3, false, false, 7), 2u);

  // Only a partial tile.
  EXPECT_EQ(GetFloorCount(0, 1, 1, false, false, 7), 1u);
  EXPECT_EQ(GetFloorCount(0, 6, 1, false, false, 7), 1u);
  EXPECT_EQ(GetFloorCount(-1, 1, 3, true, false, 7), 1u);
  EXPECT_EQ(GetFloorCount(-1, -2, -1, true, false, 7), 1u);
  EXPECT_EQ(GetFloorCount(0, 2, 3, false, false, 7), 1u);

  // Complete and partial tiles.
  EXPECT_EQ(GetFloorCount(0, 13, 1, false, false, 7), 2u);
  EXPECT_EQ(GetFloorCount(0, 15, 1, false, false, 7), 3u);
  EXPECT_EQ(GetFloorCount(-1, -14, -1, true, false, 7), 2u);
  EXPECT_EQ(GetFloorCount(0, 3 * 7 * 5 - 1, 3, false, false, 7), 5u);
  EXPECT_EQ(GetFloorCount(-1, -3 * 7 * 5, -3, true, false, 7), 5u);

  // Close to 16-bit integer range.
  EXPECT_EQ(GetFloorCount(0, 0xFFFF, 1, false, false, 1), 0xFFFFu);
  EXPECT_EQ(GetFloorCount(0, 0xFFFF, 1, false, false, 7), 0xFFFFu / 7 + 1);
  EXPECT_EQ(GetFloorCount(0, 0xFFFE, 1, false, true, 7), 0xFFFFu / 7 + 1);
  EXPECT_EQ(GetFloorCount(-0x8000, 0x7FFF, 1, true, false, 7), 0xFFFFu / 7 + 1);
  EXPECT_EQ(GetFloorCount(-0x7FFF, 0x7FFF, 1, true, true, 7), 0xFFFFu / 7 + 1);
  EXPECT_EQ(GetFloorCount(0, 0xFFFE, 1, false, false, 0xFFFF), 1u);
  EXPECT_EQ(GetFloorCount(-0x8000, 0x7FFF, 1, true, false, 0xFFFF), 1u);

  // Finalize the function.
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
