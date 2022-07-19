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
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/Casting.h"
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

/// Populate Calls with call instructions calling the function with the given
/// FnID from the given function F.
static void findCalls(Function *F, omp::RuntimeFunction FnID,
                      OpenMPIRBuilder &OMPBuilder,
                      SmallVectorImpl<CallInst *> &Calls) {
  Function *Fn = OMPBuilder.getOrCreateRuntimeFunctionPtr(FnID);
  for (BasicBlock &BB : *F) {
    for (Instruction &I : BB) {
      auto *Call = dyn_cast<CallInst>(&I);
      if (Call && Call->getCalledFunction() == Fn)
        Calls.push_back(Call);
    }
  }
}

/// Assuming \p F contains only one call to the function with the given \p FnID,
/// return that call.
static CallInst *findSingleCall(Function *F, omp::RuntimeFunction FnID,
                                OpenMPIRBuilder &OMPBuilder) {
  SmallVector<CallInst *, 1> Calls;
  findCalls(F, FnID, OMPBuilder, Calls);
  EXPECT_EQ(1u, Calls.size());
  if (Calls.size() != 1)
    return nullptr;
  return Calls.front();
}

static omp::ScheduleKind getSchedKind(omp::OMPScheduleType SchedType) {
  switch (SchedType & ~omp::OMPScheduleType::ModifierMask) {
  case omp::OMPScheduleType::BaseDynamicChunked:
    return omp::OMP_SCHEDULE_Dynamic;
  case omp::OMPScheduleType::BaseGuidedChunked:
    return omp::OMP_SCHEDULE_Guided;
  case omp::OMPScheduleType::BaseAuto:
    return omp::OMP_SCHEDULE_Auto;
  case omp::OMPScheduleType::BaseRuntime:
    return omp::OMP_SCHEDULE_Runtime;
  default:
    llvm_unreachable("unknown type for this test");
  }
}

class OpenMPIRBuilderTest : public testing::Test {
protected:
  void SetUp() override {
    Ctx.setOpaquePointers(false); // TODO: Update tests for opaque pointers.
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

  /// Create a function with a simple loop that calls printf using the logical
  /// loop counter for use with tests that need a CanonicalLoopInfo object.
  CanonicalLoopInfo *buildSingleLoopFunction(DebugLoc DL,
                                             OpenMPIRBuilder &OMPBuilder,
                                             int UseIVBits,
                                             CallInst **Call = nullptr,
                                             BasicBlock **BodyCode = nullptr) {
    OMPBuilder.initialize();
    F->setName("func");

    IRBuilder<> Builder(BB);
    OpenMPIRBuilder::LocationDescription Loc({Builder.saveIP(), DL});
    Value *TripCount = F->getArg(0);

    Type *IVType = Type::getIntNTy(Builder.getContext(), UseIVBits);
    Value *CastedTripCount =
        Builder.CreateZExtOrTrunc(TripCount, IVType, "tripcount");

    auto LoopBodyGenCB = [&](OpenMPIRBuilder::InsertPointTy CodeGenIP,
                             llvm::Value *LC) {
      Builder.restoreIP(CodeGenIP);
      if (BodyCode)
        *BodyCode = Builder.GetInsertBlock();

      // Add something that consumes the induction variable to the body.
      CallInst *CallInst = createPrintfCall(Builder, "%d\\n", {LC});
      if (Call)
        *Call = CallInst;
    };
    CanonicalLoopInfo *Loop =
        OMPBuilder.createCanonicalLoop(Loc, LoopBodyGenCB, CastedTripCount);

    // Finalize the function.
    Builder.restoreIP(Loop->getAfterIP());
    Builder.CreateRetVoid();

    return Loop;
  }

  LLVMContext Ctx;
  std::unique_ptr<Module> M;
  Function *F;
  BasicBlock *BB;
  DebugLoc DL;
};

class OpenMPIRBuilderTestWithParams
    : public OpenMPIRBuilderTest,
      public ::testing::WithParamInterface<omp::OMPScheduleType> {};

class OpenMPIRBuilderTestWithIVBits
    : public OpenMPIRBuilderTest,
      public ::testing::WithParamInterface<int> {};

// Returns the value stored in the given allocation. Returns null if the given
// value is not a result of an InstTy instruction, if no value is stored or if
// there is more than one store.
template <typename InstTy> static Value *findStoredValue(Value *AllocaValue) {
  Instruction *Inst = dyn_cast<InstTy>(AllocaValue);
  if (!Inst)
    return nullptr;
  StoreInst *Store = nullptr;
  for (Use &U : Inst->uses()) {
    if (auto *CandidateStore = dyn_cast<StoreInst>(U.getUser())) {
      EXPECT_EQ(Store, nullptr);
      Store = CandidateStore;
    }
  }
  if (!Store)
    return nullptr;
  return Store->getValueOperand();
}

// Returns the value stored in the aggregate argument of an outlined function,
// or nullptr if it is not found.
static Value *findStoredValueInAggregateAt(LLVMContext &Ctx, Value *Aggregate,
                                           unsigned Idx) {
  GetElementPtrInst *GEPAtIdx = nullptr;
  // Find GEP instruction at that index.
  for (User *Usr : Aggregate->users()) {
    GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(Usr);
    if (!GEP)
      continue;

    if (GEP->getOperand(2) != ConstantInt::get(Type::getInt32Ty(Ctx), Idx))
      continue;

    EXPECT_EQ(GEPAtIdx, nullptr);
    GEPAtIdx = GEP;
  }

  EXPECT_NE(GEPAtIdx, nullptr);
  EXPECT_EQ(GEPAtIdx->getNumUses(), 1U);

  // Find the value stored to the aggregate.
  StoreInst *StoreToAgg = dyn_cast<StoreInst>(*GEPAtIdx->user_begin());
  Value *StoredAggValue = StoreToAgg->getValueOperand();

  Value *StoredValue = nullptr;

  // Find the value stored to the value stored in the aggregate.
  for (User *Usr : StoredAggValue->users()) {
    StoreInst *Store = dyn_cast<StoreInst>(Usr);
    if (!Store)
      continue;

    if (Store->getPointerOperand() != StoredAggValue)
      continue;

    EXPECT_EQ(StoredValue, nullptr);
    StoredValue = Store->getValueOperand();
  }

  return StoredValue;
}

// Returns the aggregate that the value is originating from.
static Value *findAggregateFromValue(Value *V) {
  // Expects a load instruction that loads from the aggregate.
  LoadInst *Load = dyn_cast<LoadInst>(V);
  EXPECT_NE(Load, nullptr);
  // Find the GEP instruction used in the load instruction.
  GetElementPtrInst *GEP =
      dyn_cast<GetElementPtrInst>(Load->getPointerOperand());
  EXPECT_NE(GEP, nullptr);
  // Find the aggregate used in the GEP instruction.
  Value *Aggregate = GEP->getPointerOperand();

  return Aggregate;
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
  EXPECT_EQ(GTID->arg_size(), 1U);
  EXPECT_EQ(GTID->getCalledFunction()->getName(), "__kmpc_global_thread_num");
  EXPECT_FALSE(GTID->getCalledFunction()->doesNotAccessMemory());
  EXPECT_FALSE(GTID->getCalledFunction()->doesNotFreeMemory());

  CallInst *Barrier = dyn_cast<CallInst>(GTID->getNextNode());
  EXPECT_NE(Barrier, nullptr);
  EXPECT_EQ(Barrier->arg_size(), 2U);
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
  EXPECT_EQ(M->size(), 4U);
  EXPECT_EQ(F->size(), 4U);
  EXPECT_EQ(BB->size(), 4U);

  CallInst *GTID = dyn_cast<CallInst>(&BB->front());
  EXPECT_NE(GTID, nullptr);
  EXPECT_EQ(GTID->arg_size(), 1U);
  EXPECT_EQ(GTID->getCalledFunction()->getName(), "__kmpc_global_thread_num");
  EXPECT_FALSE(GTID->getCalledFunction()->doesNotAccessMemory());
  EXPECT_FALSE(GTID->getCalledFunction()->doesNotFreeMemory());

  CallInst *Cancel = dyn_cast<CallInst>(GTID->getNextNode());
  EXPECT_NE(Cancel, nullptr);
  EXPECT_EQ(Cancel->arg_size(), 3U);
  EXPECT_EQ(Cancel->getCalledFunction()->getName(), "__kmpc_cancel");
  EXPECT_FALSE(Cancel->getCalledFunction()->doesNotAccessMemory());
  EXPECT_FALSE(Cancel->getCalledFunction()->doesNotFreeMemory());
  EXPECT_EQ(Cancel->getNumUses(), 1U);
  Instruction *CancelBBTI = Cancel->getParent()->getTerminator();
  EXPECT_EQ(CancelBBTI->getNumSuccessors(), 2U);
  EXPECT_EQ(CancelBBTI->getSuccessor(0), NewIP.getBlock());
  EXPECT_EQ(CancelBBTI->getSuccessor(1)->size(), 3U);
  CallInst *GTID1 = dyn_cast<CallInst>(&CancelBBTI->getSuccessor(1)->front());
  EXPECT_NE(GTID1, nullptr);
  EXPECT_EQ(GTID1->arg_size(), 1U);
  EXPECT_EQ(GTID1->getCalledFunction()->getName(), "__kmpc_global_thread_num");
  EXPECT_FALSE(GTID1->getCalledFunction()->doesNotAccessMemory());
  EXPECT_FALSE(GTID1->getCalledFunction()->doesNotFreeMemory());
  CallInst *Barrier = dyn_cast<CallInst>(GTID1->getNextNode());
  EXPECT_NE(Barrier, nullptr);
  EXPECT_EQ(Barrier->arg_size(), 2U);
  EXPECT_EQ(Barrier->getCalledFunction()->getName(), "__kmpc_cancel_barrier");
  EXPECT_FALSE(Barrier->getCalledFunction()->doesNotAccessMemory());
  EXPECT_FALSE(Barrier->getCalledFunction()->doesNotFreeMemory());
  EXPECT_EQ(Barrier->getNumUses(), 0U);
  EXPECT_EQ(CancelBBTI->getSuccessor(1)->getTerminator()->getNumSuccessors(),
            1U);
  EXPECT_EQ(CancelBBTI->getSuccessor(1)->getTerminator()->getSuccessor(0), CBB);

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
  EXPECT_EQ(M->size(), 4U);
  EXPECT_EQ(F->size(), 7U);
  EXPECT_EQ(BB->size(), 1U);
  ASSERT_TRUE(isa<BranchInst>(BB->getTerminator()));
  ASSERT_EQ(BB->getTerminator()->getNumSuccessors(), 2U);
  BB = BB->getTerminator()->getSuccessor(0);
  EXPECT_EQ(BB->size(), 4U);

  CallInst *GTID = dyn_cast<CallInst>(&BB->front());
  EXPECT_NE(GTID, nullptr);
  EXPECT_EQ(GTID->arg_size(), 1U);
  EXPECT_EQ(GTID->getCalledFunction()->getName(), "__kmpc_global_thread_num");
  EXPECT_FALSE(GTID->getCalledFunction()->doesNotAccessMemory());
  EXPECT_FALSE(GTID->getCalledFunction()->doesNotFreeMemory());

  CallInst *Cancel = dyn_cast<CallInst>(GTID->getNextNode());
  EXPECT_NE(Cancel, nullptr);
  EXPECT_EQ(Cancel->arg_size(), 3U);
  EXPECT_EQ(Cancel->getCalledFunction()->getName(), "__kmpc_cancel");
  EXPECT_FALSE(Cancel->getCalledFunction()->doesNotAccessMemory());
  EXPECT_FALSE(Cancel->getCalledFunction()->doesNotFreeMemory());
  EXPECT_EQ(Cancel->getNumUses(), 1U);
  Instruction *CancelBBTI = Cancel->getParent()->getTerminator();
  EXPECT_EQ(CancelBBTI->getNumSuccessors(), 2U);
  EXPECT_EQ(CancelBBTI->getSuccessor(0)->size(), 1U);
  EXPECT_EQ(CancelBBTI->getSuccessor(0)->getUniqueSuccessor(),
            NewIP.getBlock());
  EXPECT_EQ(CancelBBTI->getSuccessor(1)->size(), 3U);
  CallInst *GTID1 = dyn_cast<CallInst>(&CancelBBTI->getSuccessor(1)->front());
  EXPECT_NE(GTID1, nullptr);
  EXPECT_EQ(GTID1->arg_size(), 1U);
  EXPECT_EQ(GTID1->getCalledFunction()->getName(), "__kmpc_global_thread_num");
  EXPECT_FALSE(GTID1->getCalledFunction()->doesNotAccessMemory());
  EXPECT_FALSE(GTID1->getCalledFunction()->doesNotFreeMemory());
  CallInst *Barrier = dyn_cast<CallInst>(GTID1->getNextNode());
  EXPECT_NE(Barrier, nullptr);
  EXPECT_EQ(Barrier->arg_size(), 2U);
  EXPECT_EQ(Barrier->getCalledFunction()->getName(), "__kmpc_cancel_barrier");
  EXPECT_FALSE(Barrier->getCalledFunction()->doesNotAccessMemory());
  EXPECT_FALSE(Barrier->getCalledFunction()->doesNotFreeMemory());
  EXPECT_EQ(Barrier->getNumUses(), 0U);
  EXPECT_EQ(CancelBBTI->getSuccessor(1)->getTerminator()->getNumSuccessors(),
            1U);
  EXPECT_EQ(CancelBBTI->getSuccessor(1)->getTerminator()->getSuccessor(0), CBB);

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
  EXPECT_EQ(GTID->arg_size(), 1U);
  EXPECT_EQ(GTID->getCalledFunction()->getName(), "__kmpc_global_thread_num");
  EXPECT_FALSE(GTID->getCalledFunction()->doesNotAccessMemory());
  EXPECT_FALSE(GTID->getCalledFunction()->doesNotFreeMemory());

  CallInst *Barrier = dyn_cast<CallInst>(GTID->getNextNode());
  EXPECT_NE(Barrier, nullptr);
  EXPECT_EQ(Barrier->arg_size(), 2U);
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

  BasicBlock *EnterBB = BasicBlock::Create(Ctx, "parallel.enter", F);
  Builder.CreateBr(EnterBB);
  Builder.SetInsertPoint(EnterBB);
  OpenMPIRBuilder::LocationDescription Loc({Builder.saveIP(), DL});

  AllocaInst *PrivAI = nullptr;

  unsigned NumBodiesGenerated = 0;
  unsigned NumPrivatizedVars = 0;
  unsigned NumFinalizationPoints = 0;

  auto BodyGenCB = [&](InsertPointTy AllocaIP, InsertPointTy CodeGenIP) {
    ++NumBodiesGenerated;

    Builder.restoreIP(AllocaIP);
    PrivAI = Builder.CreateAlloca(F->arg_begin()->getType());
    Builder.CreateStore(F->arg_begin(), PrivAI);

    Builder.restoreIP(CodeGenIP);
    Value *PrivLoad =
        Builder.CreateLoad(PrivAI->getAllocatedType(), PrivAI, "local.use");
    Value *Cmp = Builder.CreateICmpNE(F->arg_begin(), PrivLoad);
    Instruction *ThenTerm, *ElseTerm;
    SplitBlockAndInsertIfThenElse(Cmp, CodeGenIP.getBlock()->getTerminator(),
                                  &ThenTerm, &ElseTerm);
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
    Type *VTy = ReplacementValue->getType();
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
  EXPECT_EQ(ForkCI->arg_size(), 4U);
  EXPECT_TRUE(isa<GlobalVariable>(ForkCI->getArgOperand(0)));
  EXPECT_EQ(ForkCI->getArgOperand(1),
            ConstantInt::get(Type::getInt32Ty(Ctx), 1U));
  EXPECT_EQ(ForkCI->getArgOperand(2), Usr);
  Value *StoredValue =
      findStoredValueInAggregateAt(Ctx, ForkCI->getArgOperand(3), 0);
  EXPECT_EQ(StoredValue, F->arg_begin());
}

TEST_F(OpenMPIRBuilderTest, ParallelNested) {
  using InsertPointTy = OpenMPIRBuilder::InsertPointTy;
  OpenMPIRBuilder OMPBuilder(*M);
  OMPBuilder.initialize();
  F->setName("func");
  IRBuilder<> Builder(BB);

  BasicBlock *EnterBB = BasicBlock::Create(Ctx, "parallel.enter", F);
  Builder.CreateBr(EnterBB);
  Builder.SetInsertPoint(EnterBB);
  OpenMPIRBuilder::LocationDescription Loc({Builder.saveIP(), DL});

  unsigned NumInnerBodiesGenerated = 0;
  unsigned NumOuterBodiesGenerated = 0;
  unsigned NumFinalizationPoints = 0;

  auto InnerBodyGenCB = [&](InsertPointTy AllocaIP, InsertPointTy CodeGenIP) {
    ++NumInnerBodiesGenerated;
  };

  auto PrivCB = [&](InsertPointTy AllocaIP, InsertPointTy CodeGenIP,
                    Value &Orig, Value &Inner,
                    Value *&ReplacementValue) -> InsertPointTy {
    // Trivial copy (=firstprivate).
    Builder.restoreIP(AllocaIP);
    Type *VTy = ReplacementValue->getType();
    Value *V = Builder.CreateLoad(VTy, &Inner, Orig.getName() + ".reload");
    ReplacementValue = Builder.CreateAlloca(VTy, 0, Orig.getName() + ".copy");
    Builder.restoreIP(CodeGenIP);
    Builder.CreateStore(V, ReplacementValue);
    return CodeGenIP;
  };

  auto FiniCB = [&](InsertPointTy CodeGenIP) { ++NumFinalizationPoints; };

  auto OuterBodyGenCB = [&](InsertPointTy AllocaIP, InsertPointTy CodeGenIP) {
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
    EXPECT_EQ(ForkCI->arg_size(), 3U);
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

  BasicBlock *EnterBB = BasicBlock::Create(Ctx, "parallel.enter", F);
  Builder.CreateBr(EnterBB);
  Builder.SetInsertPoint(EnterBB);
  OpenMPIRBuilder::LocationDescription Loc({Builder.saveIP(), DL});

  unsigned NumInnerBodiesGenerated = 0;
  unsigned NumOuterBodiesGenerated = 0;
  unsigned NumFinalizationPoints = 0;

  auto InnerBodyGenCB = [&](InsertPointTy AllocaIP, InsertPointTy CodeGenIP) {
    ++NumInnerBodiesGenerated;
  };

  auto PrivCB = [&](InsertPointTy AllocaIP, InsertPointTy CodeGenIP,
                    Value &Orig, Value &Inner,
                    Value *&ReplacementValue) -> InsertPointTy {
    // Trivial copy (=firstprivate).
    Builder.restoreIP(AllocaIP);
    Type *VTy = ReplacementValue->getType();
    Value *V = Builder.CreateLoad(VTy, &Inner, Orig.getName() + ".reload");
    ReplacementValue = Builder.CreateAlloca(VTy, 0, Orig.getName() + ".copy");
    Builder.restoreIP(CodeGenIP);
    Builder.CreateStore(V, ReplacementValue);
    return CodeGenIP;
  };

  auto FiniCB = [&](InsertPointTy CodeGenIP) { ++NumFinalizationPoints; };

  auto OuterBodyGenCB = [&](InsertPointTy AllocaIP, InsertPointTy CodeGenIP) {
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
    EXPECT_EQ(ForkCI->arg_size(), 3U);
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

  BasicBlock *EnterBB = BasicBlock::Create(Ctx, "parallel.enter", F);
  Builder.CreateBr(EnterBB);
  Builder.SetInsertPoint(EnterBB);
  OpenMPIRBuilder::LocationDescription Loc({Builder.saveIP(), DL});

  AllocaInst *PrivAI = nullptr;

  unsigned NumBodiesGenerated = 0;
  unsigned NumPrivatizedVars = 0;
  unsigned NumFinalizationPoints = 0;

  auto BodyGenCB = [&](InsertPointTy AllocaIP, InsertPointTy CodeGenIP) {
    ++NumBodiesGenerated;

    Builder.restoreIP(AllocaIP);
    PrivAI = Builder.CreateAlloca(F->arg_begin()->getType());
    Builder.CreateStore(F->arg_begin(), PrivAI);

    Builder.restoreIP(CodeGenIP);
    Value *PrivLoad =
        Builder.CreateLoad(PrivAI->getAllocatedType(), PrivAI, "local.use");
    Value *Cmp = Builder.CreateICmpNE(F->arg_begin(), PrivLoad);
    Instruction *ThenTerm, *ElseTerm;
    SplitBlockAndInsertIfThenElse(Cmp, &*Builder.GetInsertPoint(), &ThenTerm,
                                  &ElseTerm);
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
    Type *VTy = ReplacementValue->getType();
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
  EXPECT_EQ(ForkCI->arg_size(), 4U);
  EXPECT_TRUE(isa<GlobalVariable>(ForkCI->getArgOperand(0)));
  EXPECT_EQ(ForkCI->getArgOperand(1),
            ConstantInt::get(Type::getInt32Ty(Ctx), 1));
  Value *StoredForkArg =
      findStoredValueInAggregateAt(Ctx, ForkCI->getArgOperand(3), 0);
  EXPECT_EQ(StoredForkArg, F->arg_begin());

  EXPECT_EQ(DirectCI->getCalledFunction(), OutlinedFn);
  EXPECT_EQ(DirectCI->arg_size(), 3U);
  EXPECT_TRUE(isa<AllocaInst>(DirectCI->getArgOperand(0)));
  EXPECT_TRUE(isa<AllocaInst>(DirectCI->getArgOperand(1)));
  Value *StoredDirectArg =
      findStoredValueInAggregateAt(Ctx, DirectCI->getArgOperand(2), 0);
  EXPECT_EQ(StoredDirectArg, F->arg_begin());
}

TEST_F(OpenMPIRBuilderTest, ParallelCancelBarrier) {
  using InsertPointTy = OpenMPIRBuilder::InsertPointTy;
  OpenMPIRBuilder OMPBuilder(*M);
  OMPBuilder.initialize();
  F->setName("func");
  IRBuilder<> Builder(BB);

  BasicBlock *EnterBB = BasicBlock::Create(Ctx, "parallel.enter", F);
  Builder.CreateBr(EnterBB);
  Builder.SetInsertPoint(EnterBB);
  OpenMPIRBuilder::LocationDescription Loc({Builder.saveIP(), DL});

  unsigned NumBodiesGenerated = 0;
  unsigned NumPrivatizedVars = 0;
  unsigned NumFinalizationPoints = 0;

  CallInst *CheckedBarrier = nullptr;
  auto BodyGenCB = [&](InsertPointTy AllocaIP, InsertPointTy CodeGenIP) {
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
  StructType *ArgStructTy =
      StructType::get(I32PtrTy, StructPtrTy, I32PtrTy, StructPtrTy);
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
  auto BodyGenCB = [&](InsertPointTy AllocaIP, InsertPointTy CodeGenIP) {
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
  EXPECT_TRUE(
      cast<PointerType>(Arg2Type)->isOpaqueOrPointeeTypeMatches(ArgStructTy));
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

TEST_F(OpenMPIRBuilderTest, CollapseNestedLoops) {
  using InsertPointTy = OpenMPIRBuilder::InsertPointTy;
  OpenMPIRBuilder OMPBuilder(*M);
  OMPBuilder.initialize();
  F->setName("func");

  IRBuilder<> Builder(BB);

  Type *LCTy = F->getArg(0)->getType();
  Constant *One = ConstantInt::get(LCTy, 1);
  Constant *Two = ConstantInt::get(LCTy, 2);
  Value *OuterTripCount =
      Builder.CreateAdd(F->getArg(0), Two, "tripcount.outer");
  Value *InnerTripCount =
      Builder.CreateAdd(F->getArg(0), One, "tripcount.inner");

  // Fix an insertion point for ComputeIP.
  BasicBlock *LoopNextEnter =
      BasicBlock::Create(M->getContext(), "loopnest.enter", F,
                         Builder.GetInsertBlock()->getNextNode());
  BranchInst *EnterBr = Builder.CreateBr(LoopNextEnter);
  InsertPointTy ComputeIP{EnterBr->getParent(), EnterBr->getIterator()};

  Builder.SetInsertPoint(LoopNextEnter);
  OpenMPIRBuilder::LocationDescription OuterLoc(Builder.saveIP(), DL);

  CanonicalLoopInfo *InnerLoop = nullptr;
  CallInst *InbetweenLead = nullptr;
  CallInst *InbetweenTrail = nullptr;
  CallInst *Call = nullptr;
  auto OuterLoopBodyGenCB = [&](InsertPointTy OuterCodeGenIP, Value *OuterLC) {
    Builder.restoreIP(OuterCodeGenIP);
    InbetweenLead =
        createPrintfCall(Builder, "In-between lead i=%d\\n", {OuterLC});

    auto InnerLoopBodyGenCB = [&](InsertPointTy InnerCodeGenIP,
                                  Value *InnerLC) {
      Builder.restoreIP(InnerCodeGenIP);
      Call = createPrintfCall(Builder, "body i=%d j=%d\\n", {OuterLC, InnerLC});
    };
    InnerLoop = OMPBuilder.createCanonicalLoop(
        Builder.saveIP(), InnerLoopBodyGenCB, InnerTripCount, "inner");

    Builder.restoreIP(InnerLoop->getAfterIP());
    InbetweenTrail =
        createPrintfCall(Builder, "In-between trail i=%d\\n", {OuterLC});
  };
  CanonicalLoopInfo *OuterLoop = OMPBuilder.createCanonicalLoop(
      OuterLoc, OuterLoopBodyGenCB, OuterTripCount, "outer");

  // Finish the function.
  Builder.restoreIP(OuterLoop->getAfterIP());
  Builder.CreateRetVoid();

  CanonicalLoopInfo *Collapsed =
      OMPBuilder.collapseLoops(DL, {OuterLoop, InnerLoop}, ComputeIP);

  OMPBuilder.finalize();
  EXPECT_FALSE(verifyModule(*M, &errs()));

  // Verify control flow and BB order.
  BasicBlock *RefOrder[] = {
      Collapsed->getPreheader(),   Collapsed->getHeader(),
      Collapsed->getCond(),        Collapsed->getBody(),
      InbetweenLead->getParent(),  Call->getParent(),
      InbetweenTrail->getParent(), Collapsed->getLatch(),
      Collapsed->getExit(),        Collapsed->getAfter(),
  };
  EXPECT_TRUE(verifyDFSOrder(F, RefOrder));
  EXPECT_TRUE(verifyListOrder(F, RefOrder));

  // Verify the total trip count.
  auto *TripCount = cast<MulOperator>(Collapsed->getTripCount());
  EXPECT_EQ(TripCount->getOperand(0), OuterTripCount);
  EXPECT_EQ(TripCount->getOperand(1), InnerTripCount);

  // Verify the changed indvar.
  auto *OuterIV = cast<BinaryOperator>(Call->getOperand(1));
  EXPECT_EQ(OuterIV->getOpcode(), Instruction::UDiv);
  EXPECT_EQ(OuterIV->getParent(), Collapsed->getBody());
  EXPECT_EQ(OuterIV->getOperand(1), InnerTripCount);
  EXPECT_EQ(OuterIV->getOperand(0), Collapsed->getIndVar());

  auto *InnerIV = cast<BinaryOperator>(Call->getOperand(2));
  EXPECT_EQ(InnerIV->getOpcode(), Instruction::URem);
  EXPECT_EQ(InnerIV->getParent(), Collapsed->getBody());
  EXPECT_EQ(InnerIV->getOperand(0), Collapsed->getIndVar());
  EXPECT_EQ(InnerIV->getOperand(1), InnerTripCount);

  EXPECT_EQ(InbetweenLead->getOperand(1), OuterIV);
  EXPECT_EQ(InbetweenTrail->getOperand(1), OuterIV);
}

TEST_F(OpenMPIRBuilderTest, TileSingleLoop) {
  OpenMPIRBuilder OMPBuilder(*M);
  CallInst *Call;
  BasicBlock *BodyCode;
  CanonicalLoopInfo *Loop =
      buildSingleLoopFunction(DL, OMPBuilder, 32, &Call, &BodyCode);

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
    InsertPointTy AfterIP = Loop->getAfterIP();

    // Tile the loop.
    Value *TileSizeVal = ConstantInt::get(LCTy, TileSize);
    std::vector<CanonicalLoopInfo *> GenLoops =
        OMPBuilder.tileLoops(Loc.DL, {Loop}, {TileSizeVal});

    // Set the insertion pointer to after loop, where the next loop will be
    // emitted.
    Builder.restoreIP(AfterIP);

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

TEST_F(OpenMPIRBuilderTest, ApplySimd) {
  OpenMPIRBuilder OMPBuilder(*M);

  CanonicalLoopInfo *CLI = buildSingleLoopFunction(DL, OMPBuilder, 32);

  // Simd-ize the loop.
  OMPBuilder.applySimd(CLI, nullptr);

  OMPBuilder.finalize();
  EXPECT_FALSE(verifyModule(*M, &errs()));

  PassBuilder PB;
  FunctionAnalysisManager FAM;
  PB.registerFunctionAnalyses(FAM);
  LoopInfo &LI = FAM.getResult<LoopAnalysis>(*F);

  const std::vector<Loop *> &TopLvl = LI.getTopLevelLoops();
  EXPECT_EQ(TopLvl.size(), 1u);

  Loop *L = TopLvl.front();
  EXPECT_TRUE(findStringMetadataForLoop(L, "llvm.loop.parallel_accesses"));
  EXPECT_TRUE(getBooleanLoopAttribute(L, "llvm.loop.vectorize.enable"));

  // Check for llvm.access.group metadata attached to the printf
  // function in the loop body.
  BasicBlock *LoopBody = CLI->getBody();
  EXPECT_TRUE(any_of(*LoopBody, [](Instruction &I) {
    return I.getMetadata("llvm.access.group") != nullptr;
  }));
}

TEST_F(OpenMPIRBuilderTest, ApplySimdlen) {
  OpenMPIRBuilder OMPBuilder(*M);

  CanonicalLoopInfo *CLI = buildSingleLoopFunction(DL, OMPBuilder, 32);

  // Simd-ize the loop.
  OMPBuilder.applySimd(CLI, ConstantInt::get(Type::getInt32Ty(Ctx), 3));

  OMPBuilder.finalize();
  EXPECT_FALSE(verifyModule(*M, &errs()));

  PassBuilder PB;
  FunctionAnalysisManager FAM;
  PB.registerFunctionAnalyses(FAM);
  LoopInfo &LI = FAM.getResult<LoopAnalysis>(*F);

  const std::vector<Loop *> &TopLvl = LI.getTopLevelLoops();
  EXPECT_EQ(TopLvl.size(), 1u);

  Loop *L = TopLvl.front();
  EXPECT_TRUE(findStringMetadataForLoop(L, "llvm.loop.parallel_accesses"));
  EXPECT_TRUE(getBooleanLoopAttribute(L, "llvm.loop.vectorize.enable"));
  EXPECT_EQ(getIntLoopAttribute(L, "llvm.loop.vectorize.width"), 3);

  // Check for llvm.access.group metadata attached to the printf
  // function in the loop body.
  BasicBlock *LoopBody = CLI->getBody();
  EXPECT_TRUE(any_of(*LoopBody, [](Instruction &I) {
    return I.getMetadata("llvm.access.group") != nullptr;
  }));
}

TEST_F(OpenMPIRBuilderTest, UnrollLoopFull) {
  OpenMPIRBuilder OMPBuilder(*M);

  CanonicalLoopInfo *CLI = buildSingleLoopFunction(DL, OMPBuilder, 32);

  // Unroll the loop.
  OMPBuilder.unrollLoopFull(DL, CLI);

  OMPBuilder.finalize();
  EXPECT_FALSE(verifyModule(*M, &errs()));

  PassBuilder PB;
  FunctionAnalysisManager FAM;
  PB.registerFunctionAnalyses(FAM);
  LoopInfo &LI = FAM.getResult<LoopAnalysis>(*F);

  const std::vector<Loop *> &TopLvl = LI.getTopLevelLoops();
  EXPECT_EQ(TopLvl.size(), 1u);

  Loop *L = TopLvl.front();
  EXPECT_TRUE(getBooleanLoopAttribute(L, "llvm.loop.unroll.enable"));
  EXPECT_TRUE(getBooleanLoopAttribute(L, "llvm.loop.unroll.full"));
}

TEST_F(OpenMPIRBuilderTest, UnrollLoopPartial) {
  OpenMPIRBuilder OMPBuilder(*M);
  CanonicalLoopInfo *CLI = buildSingleLoopFunction(DL, OMPBuilder, 32);

  // Unroll the loop.
  CanonicalLoopInfo *UnrolledLoop = nullptr;
  OMPBuilder.unrollLoopPartial(DL, CLI, 5, &UnrolledLoop);
  ASSERT_NE(UnrolledLoop, nullptr);

  OMPBuilder.finalize();
  EXPECT_FALSE(verifyModule(*M, &errs()));
  UnrolledLoop->assertOK();

  PassBuilder PB;
  FunctionAnalysisManager FAM;
  PB.registerFunctionAnalyses(FAM);
  LoopInfo &LI = FAM.getResult<LoopAnalysis>(*F);

  const std::vector<Loop *> &TopLvl = LI.getTopLevelLoops();
  EXPECT_EQ(TopLvl.size(), 1u);
  Loop *Outer = TopLvl.front();
  EXPECT_EQ(Outer->getHeader(), UnrolledLoop->getHeader());
  EXPECT_EQ(Outer->getLoopLatch(), UnrolledLoop->getLatch());
  EXPECT_EQ(Outer->getExitingBlock(), UnrolledLoop->getCond());
  EXPECT_EQ(Outer->getExitBlock(), UnrolledLoop->getExit());

  EXPECT_EQ(Outer->getSubLoops().size(), 1u);
  Loop *Inner = Outer->getSubLoops().front();

  EXPECT_TRUE(getBooleanLoopAttribute(Inner, "llvm.loop.unroll.enable"));
  EXPECT_EQ(getIntLoopAttribute(Inner, "llvm.loop.unroll.count"), 5);
}

TEST_F(OpenMPIRBuilderTest, UnrollLoopHeuristic) {
  OpenMPIRBuilder OMPBuilder(*M);

  CanonicalLoopInfo *CLI = buildSingleLoopFunction(DL, OMPBuilder, 32);

  // Unroll the loop.
  OMPBuilder.unrollLoopHeuristic(DL, CLI);

  OMPBuilder.finalize();
  EXPECT_FALSE(verifyModule(*M, &errs()));

  PassBuilder PB;
  FunctionAnalysisManager FAM;
  PB.registerFunctionAnalyses(FAM);
  LoopInfo &LI = FAM.getResult<LoopAnalysis>(*F);

  const std::vector<Loop *> &TopLvl = LI.getTopLevelLoops();
  EXPECT_EQ(TopLvl.size(), 1u);

  Loop *L = TopLvl.front();
  EXPECT_TRUE(getBooleanLoopAttribute(L, "llvm.loop.unroll.enable"));
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
  BasicBlock *Preheader = CLI->getPreheader();
  BasicBlock *Body = CLI->getBody();
  Value *IV = CLI->getIndVar();
  BasicBlock *ExitBlock = CLI->getExit();

  Builder.SetInsertPoint(BB, BB->getFirstInsertionPt());
  InsertPointTy AllocaIP = Builder.saveIP();

  OMPBuilder.applyWorkshareLoop(DL, CLI, AllocaIP, /*NeedsBarrier=*/true,
                                OMP_SCHEDULE_Static);

  BasicBlock *Cond = Body->getSinglePredecessor();
  Instruction *Cmp = &*Cond->begin();
  Value *TripCount = Cmp->getOperand(1);

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

  auto PreheaderIter = Preheader->begin();
  ASSERT_GE(std::distance(Preheader->begin(), Preheader->end()), 7);
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
  BinaryOperator *Add = dyn_cast<BinaryOperator>(&Body->front());
  EXPECT_EQ(Add->getOperand(0), IV);
  auto *LoadedLowerBound = dyn_cast<LoadInst>(Add->getOperand(1));
  ASSERT_NE(LoadedLowerBound, nullptr);
  EXPECT_EQ(LoadedLowerBound->getPointerOperand(), PLowerBound);

  // Check that the trip count is updated to account for the lower and upper
  // bounds return by the OpenMP runtime call.
  auto *AddOne = dyn_cast<Instruction>(TripCount);
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
  EXPECT_EQ(std::distance(IV->use_begin(), IV->use_end()), 3);

  // The exit block should contain the "fini" call and the barrier call,
  // plus the call to obtain the thread ID.
  size_t NumCallsInExitBlock =
      count_if(*ExitBlock, [](Instruction &I) { return isa<CallInst>(I); });
  EXPECT_EQ(NumCallsInExitBlock, 3u);
}

TEST_P(OpenMPIRBuilderTestWithIVBits, StaticChunkedWorkshareLoop) {
  unsigned IVBits = GetParam();

  using InsertPointTy = OpenMPIRBuilder::InsertPointTy;
  OpenMPIRBuilder OMPBuilder(*M);

  BasicBlock *Body;
  CallInst *Call;
  CanonicalLoopInfo *CLI =
      buildSingleLoopFunction(DL, OMPBuilder, IVBits, &Call, &Body);

  Instruction *OrigIndVar = CLI->getIndVar();
  EXPECT_EQ(Call->getOperand(1), OrigIndVar);

  Type *LCTy = Type::getInt32Ty(Ctx);
  Value *ChunkSize = ConstantInt::get(LCTy, 5);
  InsertPointTy AllocaIP{&F->getEntryBlock(),
                         F->getEntryBlock().getFirstInsertionPt()};
  OMPBuilder.applyWorkshareLoop(DL, CLI, AllocaIP, /*NeedsBarrier=*/true,
                                OMP_SCHEDULE_Static, ChunkSize);

  OMPBuilder.finalize();
  EXPECT_FALSE(verifyModule(*M, &errs()));

  BasicBlock *Entry = &F->getEntryBlock();
  BasicBlock *Preheader = Entry->getSingleSuccessor();

  BasicBlock *DispatchPreheader = Preheader->getSingleSuccessor();
  BasicBlock *DispatchHeader = DispatchPreheader->getSingleSuccessor();
  BasicBlock *DispatchCond = DispatchHeader->getSingleSuccessor();
  BasicBlock *DispatchBody = succ_begin(DispatchCond)[0];
  BasicBlock *DispatchExit = succ_begin(DispatchCond)[1];
  BasicBlock *DispatchAfter = DispatchExit->getSingleSuccessor();
  BasicBlock *Return = DispatchAfter->getSingleSuccessor();

  BasicBlock *ChunkPreheader = DispatchBody->getSingleSuccessor();
  BasicBlock *ChunkHeader = ChunkPreheader->getSingleSuccessor();
  BasicBlock *ChunkCond = ChunkHeader->getSingleSuccessor();
  BasicBlock *ChunkBody = succ_begin(ChunkCond)[0];
  BasicBlock *ChunkExit = succ_begin(ChunkCond)[1];
  BasicBlock *ChunkInc = ChunkBody->getSingleSuccessor();
  BasicBlock *ChunkAfter = ChunkExit->getSingleSuccessor();

  BasicBlock *DispatchInc = ChunkAfter;

  EXPECT_EQ(ChunkBody, Body);
  EXPECT_EQ(ChunkInc->getSingleSuccessor(), ChunkHeader);
  EXPECT_EQ(DispatchInc->getSingleSuccessor(), DispatchHeader);

  EXPECT_TRUE(isa<ReturnInst>(Return->front()));

  Value *NewIV = Call->getOperand(1);
  EXPECT_EQ(NewIV->getType()->getScalarSizeInBits(), IVBits);

  CallInst *InitCall = findSingleCall(
      F,
      (IVBits > 32) ? omp::RuntimeFunction::OMPRTL___kmpc_for_static_init_8u
                    : omp::RuntimeFunction::OMPRTL___kmpc_for_static_init_4u,
      OMPBuilder);
  EXPECT_EQ(InitCall->getParent(), Preheader);
  EXPECT_EQ(cast<ConstantInt>(InitCall->getArgOperand(2))->getSExtValue(), 33);
  EXPECT_EQ(cast<ConstantInt>(InitCall->getArgOperand(7))->getSExtValue(), 1);
  EXPECT_EQ(cast<ConstantInt>(InitCall->getArgOperand(8))->getSExtValue(), 5);

  CallInst *FiniCall = findSingleCall(
      F, omp::RuntimeFunction::OMPRTL___kmpc_for_static_fini, OMPBuilder);
  EXPECT_EQ(FiniCall->getParent(), DispatchExit);

  CallInst *BarrierCall = findSingleCall(
      F, omp::RuntimeFunction::OMPRTL___kmpc_barrier, OMPBuilder);
  EXPECT_EQ(BarrierCall->getParent(), DispatchExit);
}

INSTANTIATE_TEST_SUITE_P(IVBits, OpenMPIRBuilderTestWithIVBits,
                         ::testing::Values(8, 16, 32, 64));

TEST_P(OpenMPIRBuilderTestWithParams, DynamicWorkShareLoop) {
  using InsertPointTy = OpenMPIRBuilder::InsertPointTy;
  OpenMPIRBuilder OMPBuilder(*M);
  OMPBuilder.initialize();
  IRBuilder<> Builder(BB);
  OpenMPIRBuilder::LocationDescription Loc({Builder.saveIP(), DL});

  omp::OMPScheduleType SchedType = GetParam();
  uint32_t ChunkSize = 1;
  switch (SchedType & ~OMPScheduleType::ModifierMask) {
  case omp::OMPScheduleType::BaseDynamicChunked:
  case omp::OMPScheduleType::BaseGuidedChunked:
    ChunkSize = 7;
    break;
  case omp::OMPScheduleType::BaseAuto:
  case omp::OMPScheduleType::BaseRuntime:
    ChunkSize = 1;
    break;
  default:
    assert(0 && "unknown type for this test");
    break;
  }

  Type *LCTy = Type::getInt32Ty(Ctx);
  Value *StartVal = ConstantInt::get(LCTy, 10);
  Value *StopVal = ConstantInt::get(LCTy, 52);
  Value *StepVal = ConstantInt::get(LCTy, 2);
  Value *ChunkVal =
      (ChunkSize == 1) ? nullptr : ConstantInt::get(LCTy, ChunkSize);
  auto LoopBodyGen = [&](InsertPointTy, llvm::Value *) {};

  CanonicalLoopInfo *CLI = OMPBuilder.createCanonicalLoop(
      Loc, LoopBodyGen, StartVal, StopVal, StepVal,
      /*IsSigned=*/false, /*InclusiveStop=*/false);

  Builder.SetInsertPoint(BB, BB->getFirstInsertionPt());
  InsertPointTy AllocaIP = Builder.saveIP();

  // Collect all the info from CLI, as it isn't usable after the call to
  // createDynamicWorkshareLoop.
  InsertPointTy AfterIP = CLI->getAfterIP();
  BasicBlock *Preheader = CLI->getPreheader();
  BasicBlock *ExitBlock = CLI->getExit();
  BasicBlock *LatchBlock = CLI->getLatch();
  Value *IV = CLI->getIndVar();

  InsertPointTy EndIP = OMPBuilder.applyWorkshareLoop(
      DL, CLI, AllocaIP, /*NeedsBarrier=*/true, getSchedKind(SchedType),
      ChunkVal, /*Simd=*/false,
      (SchedType & omp::OMPScheduleType::ModifierMonotonic) ==
          omp::OMPScheduleType::ModifierMonotonic,
      (SchedType & omp::OMPScheduleType::ModifierNonmonotonic) ==
          omp::OMPScheduleType::ModifierNonmonotonic,
      /*Ordered=*/false);

  // The returned value should be the "after" point.
  ASSERT_EQ(EndIP.getBlock(), AfterIP.getBlock());
  ASSERT_EQ(EndIP.getPoint(), AfterIP.getPoint());

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

  auto PreheaderIter = Preheader->begin();
  ASSERT_GE(std::distance(Preheader->begin(), Preheader->end()), 6);
  StoreInst *LowerBoundStore = dyn_cast<StoreInst>(&*(PreheaderIter++));
  StoreInst *UpperBoundStore = dyn_cast<StoreInst>(&*(PreheaderIter++));
  StoreInst *StrideStore = dyn_cast<StoreInst>(&*(PreheaderIter++));
  ASSERT_NE(LowerBoundStore, nullptr);
  ASSERT_NE(UpperBoundStore, nullptr);
  ASSERT_NE(StrideStore, nullptr);

  CallInst *ThreadIdCall = dyn_cast<CallInst>(&*(PreheaderIter++));
  ASSERT_NE(ThreadIdCall, nullptr);
  EXPECT_EQ(ThreadIdCall->getCalledFunction()->getName(),
            "__kmpc_global_thread_num");

  CallInst *InitCall = dyn_cast<CallInst>(&*PreheaderIter);

  ASSERT_NE(InitCall, nullptr);
  EXPECT_EQ(InitCall->getCalledFunction()->getName(),
            "__kmpc_dispatch_init_4u");
  EXPECT_EQ(InitCall->arg_size(), 7U);
  EXPECT_EQ(InitCall->getArgOperand(6), ConstantInt::get(LCTy, ChunkSize));
  ConstantInt *SchedVal = cast<ConstantInt>(InitCall->getArgOperand(2));
  if ((SchedType & OMPScheduleType::MonotonicityMask) ==
      OMPScheduleType::None) {
    // Implementation is allowed to add default nonmonotonicity flag
    EXPECT_EQ(
        static_cast<OMPScheduleType>(SchedVal->getValue().getZExtValue()) |
            OMPScheduleType::ModifierNonmonotonic,
        SchedType | OMPScheduleType::ModifierNonmonotonic);
  } else {
    EXPECT_EQ(static_cast<OMPScheduleType>(SchedVal->getValue().getZExtValue()),
              SchedType);
  }

  ConstantInt *OrigLowerBound =
      dyn_cast<ConstantInt>(LowerBoundStore->getValueOperand());
  ConstantInt *OrigUpperBound =
      dyn_cast<ConstantInt>(UpperBoundStore->getValueOperand());
  ConstantInt *OrigStride =
      dyn_cast<ConstantInt>(StrideStore->getValueOperand());
  ASSERT_NE(OrigLowerBound, nullptr);
  ASSERT_NE(OrigUpperBound, nullptr);
  ASSERT_NE(OrigStride, nullptr);
  EXPECT_EQ(OrigLowerBound->getValue(), 1);
  EXPECT_EQ(OrigUpperBound->getValue(), 21);
  EXPECT_EQ(OrigStride->getValue(), 1);

  CallInst *FiniCall = dyn_cast<CallInst>(
      &*(LatchBlock->getTerminator()->getPrevNonDebugInstruction(true)));
  EXPECT_EQ(FiniCall, nullptr);

  // The original loop iterator should only be used in the condition, in the
  // increment and in the statement that adds the lower bound to it.
  EXPECT_EQ(std::distance(IV->use_begin(), IV->use_end()), 3);

  // The exit block should contain the barrier call, plus the call to obtain
  // the thread ID.
  size_t NumCallsInExitBlock =
      count_if(*ExitBlock, [](Instruction &I) { return isa<CallInst>(I); });
  EXPECT_EQ(NumCallsInExitBlock, 2u);

  // Add a termination to our block and check that it is internally consistent.
  Builder.restoreIP(EndIP);
  Builder.CreateRetVoid();
  OMPBuilder.finalize();
  EXPECT_FALSE(verifyModule(*M, &errs()));
}

INSTANTIATE_TEST_SUITE_P(
    OpenMPWSLoopSchedulingTypes, OpenMPIRBuilderTestWithParams,
    ::testing::Values(omp::OMPScheduleType::UnorderedDynamicChunked,
                      omp::OMPScheduleType::UnorderedGuidedChunked,
                      omp::OMPScheduleType::UnorderedAuto,
                      omp::OMPScheduleType::UnorderedRuntime,
                      omp::OMPScheduleType::UnorderedDynamicChunked |
                          omp::OMPScheduleType::ModifierMonotonic,
                      omp::OMPScheduleType::UnorderedDynamicChunked |
                          omp::OMPScheduleType::ModifierNonmonotonic,
                      omp::OMPScheduleType::UnorderedGuidedChunked |
                          omp::OMPScheduleType::ModifierMonotonic,
                      omp::OMPScheduleType::UnorderedGuidedChunked |
                          omp::OMPScheduleType::ModifierNonmonotonic,
                      omp::OMPScheduleType::UnorderedAuto |
                          omp::OMPScheduleType::ModifierMonotonic,
                      omp::OMPScheduleType::UnorderedRuntime |
                          omp::OMPScheduleType::ModifierMonotonic));

TEST_F(OpenMPIRBuilderTest, DynamicWorkShareLoopOrdered) {
  using InsertPointTy = OpenMPIRBuilder::InsertPointTy;
  OpenMPIRBuilder OMPBuilder(*M);
  OMPBuilder.initialize();
  IRBuilder<> Builder(BB);
  OpenMPIRBuilder::LocationDescription Loc({Builder.saveIP(), DL});

  uint32_t ChunkSize = 1;
  Type *LCTy = Type::getInt32Ty(Ctx);
  Value *StartVal = ConstantInt::get(LCTy, 10);
  Value *StopVal = ConstantInt::get(LCTy, 52);
  Value *StepVal = ConstantInt::get(LCTy, 2);
  Value *ChunkVal = ConstantInt::get(LCTy, ChunkSize);
  auto LoopBodyGen = [&](InsertPointTy, llvm::Value *) {};

  CanonicalLoopInfo *CLI = OMPBuilder.createCanonicalLoop(
      Loc, LoopBodyGen, StartVal, StopVal, StepVal,
      /*IsSigned=*/false, /*InclusiveStop=*/false);

  Builder.SetInsertPoint(BB, BB->getFirstInsertionPt());
  InsertPointTy AllocaIP = Builder.saveIP();

  // Collect all the info from CLI, as it isn't usable after the call to
  // createDynamicWorkshareLoop.
  BasicBlock *Preheader = CLI->getPreheader();
  BasicBlock *ExitBlock = CLI->getExit();
  BasicBlock *LatchBlock = CLI->getLatch();
  Value *IV = CLI->getIndVar();

  InsertPointTy EndIP = OMPBuilder.applyWorkshareLoop(
      DL, CLI, AllocaIP, /*NeedsBarrier=*/true, OMP_SCHEDULE_Static, ChunkVal,
      /*HasSimdModifier=*/false, /*HasMonotonicModifier=*/false,
      /*HasNonmonotonicModifier=*/false,
      /*HasOrderedClause=*/true);

  // Add a termination to our block and check that it is internally consistent.
  Builder.restoreIP(EndIP);
  Builder.CreateRetVoid();
  OMPBuilder.finalize();
  EXPECT_FALSE(verifyModule(*M, &errs()));

  CallInst *InitCall = nullptr;
  for (Instruction &EI : *Preheader) {
    Instruction *Cur = &EI;
    if (isa<CallInst>(Cur)) {
      InitCall = cast<CallInst>(Cur);
      if (InitCall->getCalledFunction()->getName() == "__kmpc_dispatch_init_4u")
        break;
      InitCall = nullptr;
    }
  }
  EXPECT_NE(InitCall, nullptr);
  EXPECT_EQ(InitCall->arg_size(), 7U);
  ConstantInt *SchedVal = cast<ConstantInt>(InitCall->getArgOperand(2));
  EXPECT_EQ(SchedVal->getValue(),
            static_cast<uint64_t>(OMPScheduleType::OrderedStaticChunked));

  CallInst *FiniCall = dyn_cast<CallInst>(
      &*(LatchBlock->getTerminator()->getPrevNonDebugInstruction(true)));
  ASSERT_NE(FiniCall, nullptr);
  EXPECT_EQ(FiniCall->getCalledFunction()->getName(),
            "__kmpc_dispatch_fini_4u");
  EXPECT_EQ(FiniCall->arg_size(), 2U);
  EXPECT_EQ(InitCall->getArgOperand(0), FiniCall->getArgOperand(0));
  EXPECT_EQ(InitCall->getArgOperand(1), FiniCall->getArgOperand(1));

  // The original loop iterator should only be used in the condition, in the
  // increment and in the statement that adds the lower bound to it.
  EXPECT_EQ(std::distance(IV->use_begin(), IV->use_end()), 3);

  // The exit block should contain the barrier call, plus the call to obtain
  // the thread ID.
  size_t NumCallsInExitBlock =
      count_if(*ExitBlock, [](Instruction &I) { return isa<CallInst>(I); });
  EXPECT_EQ(NumCallsInExitBlock, 2u);
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
  BasicBlock *ThenBB = nullptr;

  auto BodyGenCB = [&](InsertPointTy AllocaIP, InsertPointTy CodeGenIP) {
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
    ThenBB = Builder.GetInsertBlock();
    EntryBB = ThenBB->getUniquePredecessor();

    // simple instructions for body
    Value *PrivLoad =
        Builder.CreateLoad(PrivAI->getAllocatedType(), PrivAI, "local.use");
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
  BasicBlock *ExitBB = ThenBB->getUniqueSuccessor();
  EXPECT_EQ(EntryBr->getSuccessor(1), ExitBB);

  CmpInst *CondInst = cast<CmpInst>(EntryBr->getCondition());
  EXPECT_TRUE(isa<CallInst>(CondInst->getOperand(0)));

  CallInst *MasterEntryCI = cast<CallInst>(CondInst->getOperand(0));
  EXPECT_EQ(MasterEntryCI->arg_size(), 2U);
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
  EXPECT_EQ(MasterEndCI->arg_size(), 2U);
  EXPECT_TRUE(isa<GlobalVariable>(MasterEndCI->getArgOperand(0)));
  EXPECT_EQ(MasterEndCI->getArgOperand(1), MasterEntryCI->getArgOperand(1));
}

TEST_F(OpenMPIRBuilderTest, MaskedDirective) {
  using InsertPointTy = OpenMPIRBuilder::InsertPointTy;
  OpenMPIRBuilder OMPBuilder(*M);
  OMPBuilder.initialize();
  F->setName("func");
  IRBuilder<> Builder(BB);

  OpenMPIRBuilder::LocationDescription Loc({Builder.saveIP(), DL});

  AllocaInst *PrivAI = nullptr;

  BasicBlock *EntryBB = nullptr;
  BasicBlock *ThenBB = nullptr;

  auto BodyGenCB = [&](InsertPointTy AllocaIP, InsertPointTy CodeGenIP) {
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
    ThenBB = Builder.GetInsertBlock();
    EntryBB = ThenBB->getUniquePredecessor();

    // simple instructions for body
    Value *PrivLoad =
        Builder.CreateLoad(PrivAI->getAllocatedType(), PrivAI, "local.use");
    Builder.CreateICmpNE(F->arg_begin(), PrivLoad);
  };

  auto FiniCB = [&](InsertPointTy IP) {
    BasicBlock *IPBB = IP.getBlock();
    EXPECT_NE(IPBB->end(), IP.getPoint());
  };

  Constant *Filter = ConstantInt::get(Type::getInt32Ty(M->getContext()), 0);
  Builder.restoreIP(
      OMPBuilder.createMasked(Builder, BodyGenCB, FiniCB, Filter));
  Value *EntryBBTI = EntryBB->getTerminator();
  EXPECT_NE(EntryBBTI, nullptr);
  EXPECT_TRUE(isa<BranchInst>(EntryBBTI));
  BranchInst *EntryBr = cast<BranchInst>(EntryBB->getTerminator());
  EXPECT_TRUE(EntryBr->isConditional());
  EXPECT_EQ(EntryBr->getSuccessor(0), ThenBB);
  BasicBlock *ExitBB = ThenBB->getUniqueSuccessor();
  EXPECT_EQ(EntryBr->getSuccessor(1), ExitBB);

  CmpInst *CondInst = cast<CmpInst>(EntryBr->getCondition());
  EXPECT_TRUE(isa<CallInst>(CondInst->getOperand(0)));

  CallInst *MaskedEntryCI = cast<CallInst>(CondInst->getOperand(0));
  EXPECT_EQ(MaskedEntryCI->arg_size(), 3U);
  EXPECT_EQ(MaskedEntryCI->getCalledFunction()->getName(), "__kmpc_masked");
  EXPECT_TRUE(isa<GlobalVariable>(MaskedEntryCI->getArgOperand(0)));

  CallInst *MaskedEndCI = nullptr;
  for (auto &FI : *ThenBB) {
    Instruction *cur = &FI;
    if (isa<CallInst>(cur)) {
      MaskedEndCI = cast<CallInst>(cur);
      if (MaskedEndCI->getCalledFunction()->getName() == "__kmpc_end_masked")
        break;
      MaskedEndCI = nullptr;
    }
  }
  EXPECT_NE(MaskedEndCI, nullptr);
  EXPECT_EQ(MaskedEndCI->arg_size(), 2U);
  EXPECT_TRUE(isa<GlobalVariable>(MaskedEndCI->getArgOperand(0)));
  EXPECT_EQ(MaskedEndCI->getArgOperand(1), MaskedEntryCI->getArgOperand(1));
}

TEST_F(OpenMPIRBuilderTest, CriticalDirective) {
  using InsertPointTy = OpenMPIRBuilder::InsertPointTy;
  OpenMPIRBuilder OMPBuilder(*M);
  OMPBuilder.initialize();
  F->setName("func");
  IRBuilder<> Builder(BB);

  OpenMPIRBuilder::LocationDescription Loc({Builder.saveIP(), DL});

  AllocaInst *PrivAI = Builder.CreateAlloca(F->arg_begin()->getType());

  auto BodyGenCB = [&](InsertPointTy AllocaIP, InsertPointTy CodeGenIP) {
    // actual start for bodyCB
    llvm::BasicBlock *CodeGenIPBB = CodeGenIP.getBlock();
    llvm::Instruction *CodeGenIPInst = &*CodeGenIP.getPoint();
    EXPECT_EQ(CodeGenIPBB->getTerminator(), CodeGenIPInst);

    // body begin
    Builder.restoreIP(CodeGenIP);
    Builder.CreateStore(F->arg_begin(), PrivAI);
    Value *PrivLoad =
        Builder.CreateLoad(PrivAI->getAllocatedType(), PrivAI, "local.use");
    Builder.CreateICmpNE(F->arg_begin(), PrivLoad);
  };

  auto FiniCB = [&](InsertPointTy IP) {
    BasicBlock *IPBB = IP.getBlock();
    EXPECT_NE(IPBB->end(), IP.getPoint());
  };
  BasicBlock *EntryBB = Builder.GetInsertBlock();

  Builder.restoreIP(OMPBuilder.createCritical(Builder, BodyGenCB, FiniCB,
                                              "testCRT", nullptr));

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
  EXPECT_EQ(CriticalEntryCI->arg_size(), 3U);
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
  EXPECT_EQ(CriticalEndCI->arg_size(), 3U);
  EXPECT_TRUE(isa<GlobalVariable>(CriticalEndCI->getArgOperand(0)));
  EXPECT_EQ(CriticalEndCI->getArgOperand(1), CriticalEntryCI->getArgOperand(1));
  PointerType *CriticalNamePtrTy =
      PointerType::getUnqual(ArrayType::get(Type::getInt32Ty(Ctx), 8));
  EXPECT_EQ(CriticalEndCI->getArgOperand(2), CriticalEntryCI->getArgOperand(2));
  EXPECT_EQ(CriticalEndCI->getArgOperand(2)->getType(), CriticalNamePtrTy);
}

TEST_F(OpenMPIRBuilderTest, OrderedDirectiveDependSource) {
  using InsertPointTy = OpenMPIRBuilder::InsertPointTy;
  OpenMPIRBuilder OMPBuilder(*M);
  OMPBuilder.initialize();
  F->setName("func");
  IRBuilder<> Builder(BB);
  LLVMContext &Ctx = M->getContext();

  OpenMPIRBuilder::LocationDescription Loc({Builder.saveIP(), DL});

  InsertPointTy AllocaIP(&F->getEntryBlock(),
                         F->getEntryBlock().getFirstInsertionPt());

  unsigned NumLoops = 2;
  SmallVector<Value *, 2> StoreValues;
  Type *LCTy = Type::getInt64Ty(Ctx);
  StoreValues.emplace_back(ConstantInt::get(LCTy, 1));
  StoreValues.emplace_back(ConstantInt::get(LCTy, 2));

  // Test for "#omp ordered depend(source)"
  Builder.restoreIP(OMPBuilder.createOrderedDepend(Builder, AllocaIP, NumLoops,
                                                   StoreValues, ".cnt.addr",
                                                   /*IsDependSource=*/true));

  Builder.CreateRetVoid();
  OMPBuilder.finalize();
  EXPECT_FALSE(verifyModule(*M, &errs()));

  AllocaInst *AllocInst = dyn_cast<AllocaInst>(&BB->front());
  ASSERT_NE(AllocInst, nullptr);
  ArrayType *ArrType = dyn_cast<ArrayType>(AllocInst->getAllocatedType());
  EXPECT_EQ(ArrType->getNumElements(), NumLoops);
  EXPECT_TRUE(
      AllocInst->getAllocatedType()->getArrayElementType()->isIntegerTy(64));

  Instruction *IterInst = dyn_cast<Instruction>(AllocInst);
  for (unsigned Iter = 0; Iter < NumLoops; Iter++) {
    GetElementPtrInst *DependAddrGEPIter =
        dyn_cast<GetElementPtrInst>(IterInst->getNextNode());
    ASSERT_NE(DependAddrGEPIter, nullptr);
    EXPECT_EQ(DependAddrGEPIter->getPointerOperand(), AllocInst);
    EXPECT_EQ(DependAddrGEPIter->getNumIndices(), (unsigned)2);
    auto *FirstIdx = dyn_cast<ConstantInt>(DependAddrGEPIter->getOperand(1));
    auto *SecondIdx = dyn_cast<ConstantInt>(DependAddrGEPIter->getOperand(2));
    ASSERT_NE(FirstIdx, nullptr);
    ASSERT_NE(SecondIdx, nullptr);
    EXPECT_EQ(FirstIdx->getValue(), 0);
    EXPECT_EQ(SecondIdx->getValue(), Iter);
    StoreInst *StoreValue =
        dyn_cast<StoreInst>(DependAddrGEPIter->getNextNode());
    ASSERT_NE(StoreValue, nullptr);
    EXPECT_EQ(StoreValue->getValueOperand(), StoreValues[Iter]);
    EXPECT_EQ(StoreValue->getPointerOperand(), DependAddrGEPIter);
    EXPECT_EQ(StoreValue->getAlign(), Align(8));
    IterInst = dyn_cast<Instruction>(StoreValue);
  }

  GetElementPtrInst *DependBaseAddrGEP =
      dyn_cast<GetElementPtrInst>(IterInst->getNextNode());
  ASSERT_NE(DependBaseAddrGEP, nullptr);
  EXPECT_EQ(DependBaseAddrGEP->getPointerOperand(), AllocInst);
  EXPECT_EQ(DependBaseAddrGEP->getNumIndices(), (unsigned)2);
  auto *FirstIdx = dyn_cast<ConstantInt>(DependBaseAddrGEP->getOperand(1));
  auto *SecondIdx = dyn_cast<ConstantInt>(DependBaseAddrGEP->getOperand(2));
  ASSERT_NE(FirstIdx, nullptr);
  ASSERT_NE(SecondIdx, nullptr);
  EXPECT_EQ(FirstIdx->getValue(), 0);
  EXPECT_EQ(SecondIdx->getValue(), 0);

  CallInst *GTID = dyn_cast<CallInst>(DependBaseAddrGEP->getNextNode());
  ASSERT_NE(GTID, nullptr);
  EXPECT_EQ(GTID->arg_size(), 1U);
  EXPECT_EQ(GTID->getCalledFunction()->getName(), "__kmpc_global_thread_num");
  EXPECT_FALSE(GTID->getCalledFunction()->doesNotAccessMemory());
  EXPECT_FALSE(GTID->getCalledFunction()->doesNotFreeMemory());

  CallInst *Depend = dyn_cast<CallInst>(GTID->getNextNode());
  ASSERT_NE(Depend, nullptr);
  EXPECT_EQ(Depend->arg_size(), 3U);
  EXPECT_EQ(Depend->getCalledFunction()->getName(), "__kmpc_doacross_post");
  EXPECT_TRUE(isa<GlobalVariable>(Depend->getArgOperand(0)));
  EXPECT_EQ(Depend->getArgOperand(1), GTID);
  EXPECT_EQ(Depend->getArgOperand(2), DependBaseAddrGEP);
}

TEST_F(OpenMPIRBuilderTest, OrderedDirectiveDependSink) {
  using InsertPointTy = OpenMPIRBuilder::InsertPointTy;
  OpenMPIRBuilder OMPBuilder(*M);
  OMPBuilder.initialize();
  F->setName("func");
  IRBuilder<> Builder(BB);
  LLVMContext &Ctx = M->getContext();

  OpenMPIRBuilder::LocationDescription Loc({Builder.saveIP(), DL});

  InsertPointTy AllocaIP(&F->getEntryBlock(),
                         F->getEntryBlock().getFirstInsertionPt());

  unsigned NumLoops = 2;
  SmallVector<Value *, 2> StoreValues;
  Type *LCTy = Type::getInt64Ty(Ctx);
  StoreValues.emplace_back(ConstantInt::get(LCTy, 1));
  StoreValues.emplace_back(ConstantInt::get(LCTy, 2));

  // Test for "#omp ordered depend(sink: vec)"
  Builder.restoreIP(OMPBuilder.createOrderedDepend(Builder, AllocaIP, NumLoops,
                                                   StoreValues, ".cnt.addr",
                                                   /*IsDependSource=*/false));

  Builder.CreateRetVoid();
  OMPBuilder.finalize();
  EXPECT_FALSE(verifyModule(*M, &errs()));

  AllocaInst *AllocInst = dyn_cast<AllocaInst>(&BB->front());
  ASSERT_NE(AllocInst, nullptr);
  ArrayType *ArrType = dyn_cast<ArrayType>(AllocInst->getAllocatedType());
  EXPECT_EQ(ArrType->getNumElements(), NumLoops);
  EXPECT_TRUE(
      AllocInst->getAllocatedType()->getArrayElementType()->isIntegerTy(64));

  Instruction *IterInst = dyn_cast<Instruction>(AllocInst);
  for (unsigned Iter = 0; Iter < NumLoops; Iter++) {
    GetElementPtrInst *DependAddrGEPIter =
        dyn_cast<GetElementPtrInst>(IterInst->getNextNode());
    ASSERT_NE(DependAddrGEPIter, nullptr);
    EXPECT_EQ(DependAddrGEPIter->getPointerOperand(), AllocInst);
    EXPECT_EQ(DependAddrGEPIter->getNumIndices(), (unsigned)2);
    auto *FirstIdx = dyn_cast<ConstantInt>(DependAddrGEPIter->getOperand(1));
    auto *SecondIdx = dyn_cast<ConstantInt>(DependAddrGEPIter->getOperand(2));
    ASSERT_NE(FirstIdx, nullptr);
    ASSERT_NE(SecondIdx, nullptr);
    EXPECT_EQ(FirstIdx->getValue(), 0);
    EXPECT_EQ(SecondIdx->getValue(), Iter);
    StoreInst *StoreValue =
        dyn_cast<StoreInst>(DependAddrGEPIter->getNextNode());
    ASSERT_NE(StoreValue, nullptr);
    EXPECT_EQ(StoreValue->getValueOperand(), StoreValues[Iter]);
    EXPECT_EQ(StoreValue->getPointerOperand(), DependAddrGEPIter);
    EXPECT_EQ(StoreValue->getAlign(), Align(8));
    IterInst = dyn_cast<Instruction>(StoreValue);
  }

  GetElementPtrInst *DependBaseAddrGEP =
      dyn_cast<GetElementPtrInst>(IterInst->getNextNode());
  ASSERT_NE(DependBaseAddrGEP, nullptr);
  EXPECT_EQ(DependBaseAddrGEP->getPointerOperand(), AllocInst);
  EXPECT_EQ(DependBaseAddrGEP->getNumIndices(), (unsigned)2);
  auto *FirstIdx = dyn_cast<ConstantInt>(DependBaseAddrGEP->getOperand(1));
  auto *SecondIdx = dyn_cast<ConstantInt>(DependBaseAddrGEP->getOperand(2));
  ASSERT_NE(FirstIdx, nullptr);
  ASSERT_NE(SecondIdx, nullptr);
  EXPECT_EQ(FirstIdx->getValue(), 0);
  EXPECT_EQ(SecondIdx->getValue(), 0);

  CallInst *GTID = dyn_cast<CallInst>(DependBaseAddrGEP->getNextNode());
  ASSERT_NE(GTID, nullptr);
  EXPECT_EQ(GTID->arg_size(), 1U);
  EXPECT_EQ(GTID->getCalledFunction()->getName(), "__kmpc_global_thread_num");
  EXPECT_FALSE(GTID->getCalledFunction()->doesNotAccessMemory());
  EXPECT_FALSE(GTID->getCalledFunction()->doesNotFreeMemory());

  CallInst *Depend = dyn_cast<CallInst>(GTID->getNextNode());
  ASSERT_NE(Depend, nullptr);
  EXPECT_EQ(Depend->arg_size(), 3U);
  EXPECT_EQ(Depend->getCalledFunction()->getName(), "__kmpc_doacross_wait");
  EXPECT_TRUE(isa<GlobalVariable>(Depend->getArgOperand(0)));
  EXPECT_EQ(Depend->getArgOperand(1), GTID);
  EXPECT_EQ(Depend->getArgOperand(2), DependBaseAddrGEP);
}

TEST_F(OpenMPIRBuilderTest, OrderedDirectiveThreads) {
  using InsertPointTy = OpenMPIRBuilder::InsertPointTy;
  OpenMPIRBuilder OMPBuilder(*M);
  OMPBuilder.initialize();
  F->setName("func");
  IRBuilder<> Builder(BB);

  OpenMPIRBuilder::LocationDescription Loc({Builder.saveIP(), DL});

  AllocaInst *PrivAI =
      Builder.CreateAlloca(F->arg_begin()->getType(), nullptr, "priv.inst");

  auto BodyGenCB = [&](InsertPointTy AllocaIP, InsertPointTy CodeGenIP) {
    llvm::BasicBlock *CodeGenIPBB = CodeGenIP.getBlock();
    llvm::Instruction *CodeGenIPInst = &*CodeGenIP.getPoint();
    EXPECT_EQ(CodeGenIPBB->getTerminator(), CodeGenIPInst);

    Builder.restoreIP(CodeGenIP);
    Builder.CreateStore(F->arg_begin(), PrivAI);
    Value *PrivLoad =
        Builder.CreateLoad(PrivAI->getAllocatedType(), PrivAI, "local.use");
    Builder.CreateICmpNE(F->arg_begin(), PrivLoad);
  };

  auto FiniCB = [&](InsertPointTy IP) {
    BasicBlock *IPBB = IP.getBlock();
    EXPECT_NE(IPBB->end(), IP.getPoint());
  };

  // Test for "#omp ordered [threads]"
  BasicBlock *EntryBB = Builder.GetInsertBlock();
  Builder.restoreIP(
      OMPBuilder.createOrderedThreadsSimd(Builder, BodyGenCB, FiniCB, true));

  Builder.CreateRetVoid();
  OMPBuilder.finalize();
  EXPECT_FALSE(verifyModule(*M, &errs()));

  EXPECT_NE(EntryBB->getTerminator(), nullptr);

  CallInst *OrderedEntryCI = nullptr;
  for (auto &EI : *EntryBB) {
    Instruction *Cur = &EI;
    if (isa<CallInst>(Cur)) {
      OrderedEntryCI = cast<CallInst>(Cur);
      if (OrderedEntryCI->getCalledFunction()->getName() == "__kmpc_ordered")
        break;
      OrderedEntryCI = nullptr;
    }
  }
  EXPECT_NE(OrderedEntryCI, nullptr);
  EXPECT_EQ(OrderedEntryCI->arg_size(), 2U);
  EXPECT_EQ(OrderedEntryCI->getCalledFunction()->getName(), "__kmpc_ordered");
  EXPECT_TRUE(isa<GlobalVariable>(OrderedEntryCI->getArgOperand(0)));

  CallInst *OrderedEndCI = nullptr;
  for (auto &FI : *EntryBB) {
    Instruction *Cur = &FI;
    if (isa<CallInst>(Cur)) {
      OrderedEndCI = cast<CallInst>(Cur);
      if (OrderedEndCI->getCalledFunction()->getName() == "__kmpc_end_ordered")
        break;
      OrderedEndCI = nullptr;
    }
  }
  EXPECT_NE(OrderedEndCI, nullptr);
  EXPECT_EQ(OrderedEndCI->arg_size(), 2U);
  EXPECT_TRUE(isa<GlobalVariable>(OrderedEndCI->getArgOperand(0)));
  EXPECT_EQ(OrderedEndCI->getArgOperand(1), OrderedEntryCI->getArgOperand(1));
}

TEST_F(OpenMPIRBuilderTest, OrderedDirectiveSimd) {
  using InsertPointTy = OpenMPIRBuilder::InsertPointTy;
  OpenMPIRBuilder OMPBuilder(*M);
  OMPBuilder.initialize();
  F->setName("func");
  IRBuilder<> Builder(BB);

  OpenMPIRBuilder::LocationDescription Loc({Builder.saveIP(), DL});

  AllocaInst *PrivAI =
      Builder.CreateAlloca(F->arg_begin()->getType(), nullptr, "priv.inst");

  auto BodyGenCB = [&](InsertPointTy AllocaIP, InsertPointTy CodeGenIP) {
    llvm::BasicBlock *CodeGenIPBB = CodeGenIP.getBlock();
    llvm::Instruction *CodeGenIPInst = &*CodeGenIP.getPoint();
    EXPECT_EQ(CodeGenIPBB->getTerminator(), CodeGenIPInst);

    Builder.restoreIP(CodeGenIP);
    Builder.CreateStore(F->arg_begin(), PrivAI);
    Value *PrivLoad =
        Builder.CreateLoad(PrivAI->getAllocatedType(), PrivAI, "local.use");
    Builder.CreateICmpNE(F->arg_begin(), PrivLoad);
  };

  auto FiniCB = [&](InsertPointTy IP) {
    BasicBlock *IPBB = IP.getBlock();
    EXPECT_NE(IPBB->end(), IP.getPoint());
  };

  // Test for "#omp ordered simd"
  BasicBlock *EntryBB = Builder.GetInsertBlock();
  Builder.restoreIP(
      OMPBuilder.createOrderedThreadsSimd(Builder, BodyGenCB, FiniCB, false));

  Builder.CreateRetVoid();
  OMPBuilder.finalize();
  EXPECT_FALSE(verifyModule(*M, &errs()));

  EXPECT_NE(EntryBB->getTerminator(), nullptr);

  CallInst *OrderedEntryCI = nullptr;
  for (auto &EI : *EntryBB) {
    Instruction *Cur = &EI;
    if (isa<CallInst>(Cur)) {
      OrderedEntryCI = cast<CallInst>(Cur);
      if (OrderedEntryCI->getCalledFunction()->getName() == "__kmpc_ordered")
        break;
      OrderedEntryCI = nullptr;
    }
  }
  EXPECT_EQ(OrderedEntryCI, nullptr);

  CallInst *OrderedEndCI = nullptr;
  for (auto &FI : *EntryBB) {
    Instruction *Cur = &FI;
    if (isa<CallInst>(Cur)) {
      OrderedEndCI = cast<CallInst>(Cur);
      if (OrderedEndCI->getCalledFunction()->getName() == "__kmpc_end_ordered")
        break;
      OrderedEndCI = nullptr;
    }
  }
  EXPECT_EQ(OrderedEndCI, nullptr);
}

TEST_F(OpenMPIRBuilderTest, CopyinBlocks) {
  OpenMPIRBuilder OMPBuilder(*M);
  OMPBuilder.initialize();
  F->setName("func");
  IRBuilder<> Builder(BB);

  OpenMPIRBuilder::LocationDescription Loc({Builder.saveIP(), DL});

  IntegerType *Int32 = Type::getInt32Ty(M->getContext());
  AllocaInst *MasterAddress = Builder.CreateAlloca(Int32->getPointerTo());
  AllocaInst *PrivAddress = Builder.CreateAlloca(Int32->getPointerTo());

  BasicBlock *EntryBB = BB;

  OMPBuilder.createCopyinClauseBlocks(Builder.saveIP(), MasterAddress,
                                      PrivAddress, Int32, /*BranchtoEnd*/ true);

  BranchInst *EntryBr = dyn_cast_or_null<BranchInst>(EntryBB->getTerminator());

  EXPECT_NE(EntryBr, nullptr);
  EXPECT_TRUE(EntryBr->isConditional());

  BasicBlock *NotMasterBB = EntryBr->getSuccessor(0);
  BasicBlock *CopyinEnd = EntryBr->getSuccessor(1);
  CmpInst *CMP = dyn_cast_or_null<CmpInst>(EntryBr->getCondition());

  EXPECT_NE(CMP, nullptr);
  EXPECT_NE(NotMasterBB, nullptr);
  EXPECT_NE(CopyinEnd, nullptr);

  BranchInst *NotMasterBr =
      dyn_cast_or_null<BranchInst>(NotMasterBB->getTerminator());
  EXPECT_NE(NotMasterBr, nullptr);
  EXPECT_FALSE(NotMasterBr->isConditional());
  EXPECT_EQ(CopyinEnd, NotMasterBr->getSuccessor(0));
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
  BasicBlock *ThenBB = nullptr;

  auto BodyGenCB = [&](InsertPointTy AllocaIP, InsertPointTy CodeGenIP) {
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
    ThenBB = Builder.GetInsertBlock();
    EntryBB = ThenBB->getUniquePredecessor();

    // simple instructions for body
    Value *PrivLoad =
        Builder.CreateLoad(PrivAI->getAllocatedType(), PrivAI, "local.use");
    Builder.CreateICmpNE(F->arg_begin(), PrivLoad);
  };

  auto FiniCB = [&](InsertPointTy IP) {
    BasicBlock *IPBB = IP.getBlock();
    EXPECT_NE(IPBB->end(), IP.getPoint());
  };

  Builder.restoreIP(OMPBuilder.createSingle(
      Builder, BodyGenCB, FiniCB, /*IsNowait*/ false, /*DidIt*/ nullptr));
  Value *EntryBBTI = EntryBB->getTerminator();
  EXPECT_NE(EntryBBTI, nullptr);
  EXPECT_TRUE(isa<BranchInst>(EntryBBTI));
  BranchInst *EntryBr = cast<BranchInst>(EntryBB->getTerminator());
  EXPECT_TRUE(EntryBr->isConditional());
  EXPECT_EQ(EntryBr->getSuccessor(0), ThenBB);
  BasicBlock *ExitBB = ThenBB->getUniqueSuccessor();
  EXPECT_EQ(EntryBr->getSuccessor(1), ExitBB);

  CmpInst *CondInst = cast<CmpInst>(EntryBr->getCondition());
  EXPECT_TRUE(isa<CallInst>(CondInst->getOperand(0)));

  CallInst *SingleEntryCI = cast<CallInst>(CondInst->getOperand(0));
  EXPECT_EQ(SingleEntryCI->arg_size(), 2U);
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
  EXPECT_EQ(SingleEndCI->arg_size(), 2U);
  EXPECT_TRUE(isa<GlobalVariable>(SingleEndCI->getArgOperand(0)));
  EXPECT_EQ(SingleEndCI->getArgOperand(1), SingleEntryCI->getArgOperand(1));

  bool FoundBarrier = false;
  for (auto &FI : *ExitBB) {
    Instruction *cur = &FI;
    if (auto CI = dyn_cast<CallInst>(cur)) {
      if (CI->getCalledFunction()->getName() == "__kmpc_barrier") {
        FoundBarrier = true;
        break;
      }
    }
  }
  EXPECT_TRUE(FoundBarrier);
}

TEST_F(OpenMPIRBuilderTest, SingleDirectiveNowait) {
  using InsertPointTy = OpenMPIRBuilder::InsertPointTy;
  OpenMPIRBuilder OMPBuilder(*M);
  OMPBuilder.initialize();
  F->setName("func");
  IRBuilder<> Builder(BB);

  OpenMPIRBuilder::LocationDescription Loc({Builder.saveIP(), DL});

  AllocaInst *PrivAI = nullptr;

  BasicBlock *EntryBB = nullptr;
  BasicBlock *ThenBB = nullptr;

  auto BodyGenCB = [&](InsertPointTy AllocaIP, InsertPointTy CodeGenIP) {
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
    ThenBB = Builder.GetInsertBlock();
    EntryBB = ThenBB->getUniquePredecessor();

    // simple instructions for body
    Value *PrivLoad =
        Builder.CreateLoad(PrivAI->getAllocatedType(), PrivAI, "local.use");
    Builder.CreateICmpNE(F->arg_begin(), PrivLoad);
  };

  auto FiniCB = [&](InsertPointTy IP) {
    BasicBlock *IPBB = IP.getBlock();
    EXPECT_NE(IPBB->end(), IP.getPoint());
  };

  Builder.restoreIP(OMPBuilder.createSingle(
      Builder, BodyGenCB, FiniCB, /*IsNowait*/ true, /*DidIt*/ nullptr));
  Value *EntryBBTI = EntryBB->getTerminator();
  EXPECT_NE(EntryBBTI, nullptr);
  EXPECT_TRUE(isa<BranchInst>(EntryBBTI));
  BranchInst *EntryBr = cast<BranchInst>(EntryBB->getTerminator());
  EXPECT_TRUE(EntryBr->isConditional());
  EXPECT_EQ(EntryBr->getSuccessor(0), ThenBB);
  BasicBlock *ExitBB = ThenBB->getUniqueSuccessor();
  EXPECT_EQ(EntryBr->getSuccessor(1), ExitBB);

  CmpInst *CondInst = cast<CmpInst>(EntryBr->getCondition());
  EXPECT_TRUE(isa<CallInst>(CondInst->getOperand(0)));

  CallInst *SingleEntryCI = cast<CallInst>(CondInst->getOperand(0));
  EXPECT_EQ(SingleEntryCI->arg_size(), 2U);
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
  EXPECT_EQ(SingleEndCI->arg_size(), 2U);
  EXPECT_TRUE(isa<GlobalVariable>(SingleEndCI->getArgOperand(0)));
  EXPECT_EQ(SingleEndCI->getArgOperand(1), SingleEntryCI->getArgOperand(1));

  CallInst *ExitBarrier = nullptr;
  for (auto &FI : *ExitBB) {
    Instruction *cur = &FI;
    if (auto CI = dyn_cast<CallInst>(cur)) {
      if (CI->getCalledFunction()->getName() == "__kmpc_barrier") {
        ExitBarrier = CI;
        break;
      }
    }
  }
  EXPECT_EQ(ExitBarrier, nullptr);
}

TEST_F(OpenMPIRBuilderTest, OMPAtomicReadFlt) {
  OpenMPIRBuilder OMPBuilder(*M);
  OMPBuilder.initialize();
  F->setName("func");
  IRBuilder<> Builder(BB);

  OpenMPIRBuilder::LocationDescription Loc({Builder.saveIP(), DL});

  Type *Float32 = Type::getFloatTy(M->getContext());
  AllocaInst *XVal = Builder.CreateAlloca(Float32);
  XVal->setName("AtomicVar");
  AllocaInst *VVal = Builder.CreateAlloca(Float32);
  VVal->setName("AtomicRead");
  AtomicOrdering AO = AtomicOrdering::Monotonic;
  OpenMPIRBuilder::AtomicOpValue X = {XVal, Float32, false, false};
  OpenMPIRBuilder::AtomicOpValue V = {VVal, Float32, false, false};

  Builder.restoreIP(OMPBuilder.createAtomicRead(Loc, X, V, AO));

  IntegerType *IntCastTy =
      IntegerType::get(M->getContext(), Float32->getScalarSizeInBits());

  BitCastInst *CastFrmFlt = cast<BitCastInst>(VVal->getNextNode());
  EXPECT_EQ(CastFrmFlt->getSrcTy(), Float32->getPointerTo());
  EXPECT_EQ(CastFrmFlt->getDestTy(), IntCastTy->getPointerTo());
  EXPECT_EQ(CastFrmFlt->getOperand(0), XVal);

  LoadInst *AtomicLoad = cast<LoadInst>(CastFrmFlt->getNextNode());
  EXPECT_TRUE(AtomicLoad->isAtomic());
  EXPECT_EQ(AtomicLoad->getPointerOperand(), CastFrmFlt);

  BitCastInst *CastToFlt = cast<BitCastInst>(AtomicLoad->getNextNode());
  EXPECT_EQ(CastToFlt->getSrcTy(), IntCastTy);
  EXPECT_EQ(CastToFlt->getDestTy(), Float32);
  EXPECT_EQ(CastToFlt->getOperand(0), AtomicLoad);

  StoreInst *StoreofAtomic = cast<StoreInst>(CastToFlt->getNextNode());
  EXPECT_EQ(StoreofAtomic->getValueOperand(), CastToFlt);
  EXPECT_EQ(StoreofAtomic->getPointerOperand(), VVal);

  Builder.CreateRetVoid();
  OMPBuilder.finalize();
  EXPECT_FALSE(verifyModule(*M, &errs()));
}

TEST_F(OpenMPIRBuilderTest, OMPAtomicReadInt) {
  OpenMPIRBuilder OMPBuilder(*M);
  OMPBuilder.initialize();
  F->setName("func");
  IRBuilder<> Builder(BB);

  OpenMPIRBuilder::LocationDescription Loc({Builder.saveIP(), DL});

  IntegerType *Int32 = Type::getInt32Ty(M->getContext());
  AllocaInst *XVal = Builder.CreateAlloca(Int32);
  XVal->setName("AtomicVar");
  AllocaInst *VVal = Builder.CreateAlloca(Int32);
  VVal->setName("AtomicRead");
  AtomicOrdering AO = AtomicOrdering::Monotonic;
  OpenMPIRBuilder::AtomicOpValue X = {XVal, Int32, false, false};
  OpenMPIRBuilder::AtomicOpValue V = {VVal, Int32, false, false};

  BasicBlock *EntryBB = BB;

  Builder.restoreIP(OMPBuilder.createAtomicRead(Loc, X, V, AO));
  LoadInst *AtomicLoad = nullptr;
  StoreInst *StoreofAtomic = nullptr;

  for (Instruction &Cur : *EntryBB) {
    if (isa<LoadInst>(Cur)) {
      AtomicLoad = cast<LoadInst>(&Cur);
      if (AtomicLoad->getPointerOperand() == XVal)
        continue;
      AtomicLoad = nullptr;
    } else if (isa<StoreInst>(Cur)) {
      StoreofAtomic = cast<StoreInst>(&Cur);
      if (StoreofAtomic->getPointerOperand() == VVal)
        continue;
      StoreofAtomic = nullptr;
    }
  }

  EXPECT_NE(AtomicLoad, nullptr);
  EXPECT_TRUE(AtomicLoad->isAtomic());

  EXPECT_NE(StoreofAtomic, nullptr);
  EXPECT_EQ(StoreofAtomic->getValueOperand(), AtomicLoad);

  Builder.CreateRetVoid();
  OMPBuilder.finalize();

  EXPECT_FALSE(verifyModule(*M, &errs()));
}

TEST_F(OpenMPIRBuilderTest, OMPAtomicWriteFlt) {
  OpenMPIRBuilder OMPBuilder(*M);
  OMPBuilder.initialize();
  F->setName("func");
  IRBuilder<> Builder(BB);

  OpenMPIRBuilder::LocationDescription Loc({Builder.saveIP(), DL});

  LLVMContext &Ctx = M->getContext();
  Type *Float32 = Type::getFloatTy(Ctx);
  AllocaInst *XVal = Builder.CreateAlloca(Float32);
  XVal->setName("AtomicVar");
  OpenMPIRBuilder::AtomicOpValue X = {XVal, Float32, false, false};
  AtomicOrdering AO = AtomicOrdering::Monotonic;
  Constant *ValToWrite = ConstantFP::get(Float32, 1.0);

  Builder.restoreIP(OMPBuilder.createAtomicWrite(Loc, X, ValToWrite, AO));

  IntegerType *IntCastTy =
      IntegerType::get(M->getContext(), Float32->getScalarSizeInBits());

  BitCastInst *CastFrmFlt = cast<BitCastInst>(XVal->getNextNode());
  EXPECT_EQ(CastFrmFlt->getSrcTy(), Float32->getPointerTo());
  EXPECT_EQ(CastFrmFlt->getDestTy(), IntCastTy->getPointerTo());
  EXPECT_EQ(CastFrmFlt->getOperand(0), XVal);

  Value *ExprCast = Builder.CreateBitCast(ValToWrite, IntCastTy);

  StoreInst *StoreofAtomic = cast<StoreInst>(CastFrmFlt->getNextNode());
  EXPECT_EQ(StoreofAtomic->getValueOperand(), ExprCast);
  EXPECT_EQ(StoreofAtomic->getPointerOperand(), CastFrmFlt);
  EXPECT_TRUE(StoreofAtomic->isAtomic());

  Builder.CreateRetVoid();
  OMPBuilder.finalize();
  EXPECT_FALSE(verifyModule(*M, &errs()));
}

TEST_F(OpenMPIRBuilderTest, OMPAtomicWriteInt) {
  OpenMPIRBuilder OMPBuilder(*M);
  OMPBuilder.initialize();
  F->setName("func");
  IRBuilder<> Builder(BB);

  OpenMPIRBuilder::LocationDescription Loc({Builder.saveIP(), DL});

  LLVMContext &Ctx = M->getContext();
  IntegerType *Int32 = Type::getInt32Ty(Ctx);
  AllocaInst *XVal = Builder.CreateAlloca(Int32);
  XVal->setName("AtomicVar");
  OpenMPIRBuilder::AtomicOpValue X = {XVal, Int32, false, false};
  AtomicOrdering AO = AtomicOrdering::Monotonic;
  ConstantInt *ValToWrite = ConstantInt::get(Type::getInt32Ty(Ctx), 1U);

  BasicBlock *EntryBB = BB;

  Builder.restoreIP(OMPBuilder.createAtomicWrite(Loc, X, ValToWrite, AO));

  StoreInst *StoreofAtomic = nullptr;

  for (Instruction &Cur : *EntryBB) {
    if (isa<StoreInst>(Cur)) {
      StoreofAtomic = cast<StoreInst>(&Cur);
      if (StoreofAtomic->getPointerOperand() == XVal)
        continue;
      StoreofAtomic = nullptr;
    }
  }

  EXPECT_NE(StoreofAtomic, nullptr);
  EXPECT_TRUE(StoreofAtomic->isAtomic());
  EXPECT_EQ(StoreofAtomic->getValueOperand(), ValToWrite);

  Builder.CreateRetVoid();
  OMPBuilder.finalize();
  EXPECT_FALSE(verifyModule(*M, &errs()));
}

TEST_F(OpenMPIRBuilderTest, OMPAtomicUpdate) {
  OpenMPIRBuilder OMPBuilder(*M);
  OMPBuilder.initialize();
  F->setName("func");
  IRBuilder<> Builder(BB);

  OpenMPIRBuilder::LocationDescription Loc({Builder.saveIP(), DL});

  IntegerType *Int32 = Type::getInt32Ty(M->getContext());
  AllocaInst *XVal = Builder.CreateAlloca(Int32);
  XVal->setName("AtomicVar");
  Builder.CreateStore(ConstantInt::get(Type::getInt32Ty(Ctx), 0U), XVal);
  OpenMPIRBuilder::AtomicOpValue X = {XVal, Int32, false, false};
  AtomicOrdering AO = AtomicOrdering::Monotonic;
  ConstantInt *ConstVal = ConstantInt::get(Type::getInt32Ty(Ctx), 1U);
  Value *Expr = nullptr;
  AtomicRMWInst::BinOp RMWOp = AtomicRMWInst::Sub;
  bool IsXLHSInRHSPart = false;

  BasicBlock *EntryBB = BB;
  OpenMPIRBuilder::InsertPointTy AllocaIP(EntryBB,
                                          EntryBB->getFirstInsertionPt());
  Value *Sub = nullptr;

  auto UpdateOp = [&](Value *Atomic, IRBuilder<> &IRB) {
    Sub = IRB.CreateSub(ConstVal, Atomic);
    return Sub;
  };
  Builder.restoreIP(OMPBuilder.createAtomicUpdate(
      Builder, AllocaIP, X, Expr, AO, RMWOp, UpdateOp, IsXLHSInRHSPart));
  BasicBlock *ContBB = EntryBB->getSingleSuccessor();
  BranchInst *ContTI = dyn_cast<BranchInst>(ContBB->getTerminator());
  EXPECT_NE(ContTI, nullptr);
  BasicBlock *EndBB = ContTI->getSuccessor(0);
  EXPECT_TRUE(ContTI->isConditional());
  EXPECT_EQ(ContTI->getSuccessor(1), ContBB);
  EXPECT_NE(EndBB, nullptr);

  PHINode *Phi = dyn_cast<PHINode>(&ContBB->front());
  EXPECT_NE(Phi, nullptr);
  EXPECT_EQ(Phi->getNumIncomingValues(), 2U);
  EXPECT_EQ(Phi->getIncomingBlock(0), EntryBB);
  EXPECT_EQ(Phi->getIncomingBlock(1), ContBB);

  EXPECT_EQ(Sub->getNumUses(), 1U);
  StoreInst *St = dyn_cast<StoreInst>(Sub->user_back());
  AllocaInst *UpdateTemp = dyn_cast<AllocaInst>(St->getPointerOperand());

  ExtractValueInst *ExVI1 =
      dyn_cast<ExtractValueInst>(Phi->getIncomingValueForBlock(ContBB));
  EXPECT_NE(ExVI1, nullptr);
  AtomicCmpXchgInst *CmpExchg =
      dyn_cast<AtomicCmpXchgInst>(ExVI1->getAggregateOperand());
  EXPECT_NE(CmpExchg, nullptr);
  EXPECT_EQ(CmpExchg->getPointerOperand(), XVal);
  EXPECT_EQ(CmpExchg->getCompareOperand(), Phi);
  EXPECT_EQ(CmpExchg->getSuccessOrdering(), AtomicOrdering::Monotonic);

  LoadInst *Ld = dyn_cast<LoadInst>(CmpExchg->getNewValOperand());
  EXPECT_NE(Ld, nullptr);
  EXPECT_EQ(UpdateTemp, Ld->getPointerOperand());

  Builder.CreateRetVoid();
  OMPBuilder.finalize();
  EXPECT_FALSE(verifyModule(*M, &errs()));
}

TEST_F(OpenMPIRBuilderTest, OMPAtomicUpdateFloat) {
  OpenMPIRBuilder OMPBuilder(*M);
  OMPBuilder.initialize();
  F->setName("func");
  IRBuilder<> Builder(BB);

  OpenMPIRBuilder::LocationDescription Loc({Builder.saveIP(), DL});

  Type *FloatTy = Type::getFloatTy(M->getContext());
  AllocaInst *XVal = Builder.CreateAlloca(FloatTy);
  XVal->setName("AtomicVar");
  Builder.CreateStore(ConstantFP::get(Type::getFloatTy(Ctx), 0.0), XVal);
  OpenMPIRBuilder::AtomicOpValue X = {XVal, FloatTy, false, false};
  AtomicOrdering AO = AtomicOrdering::Monotonic;
  Constant *ConstVal = ConstantFP::get(Type::getFloatTy(Ctx), 1.0);
  Value *Expr = nullptr;
  AtomicRMWInst::BinOp RMWOp = AtomicRMWInst::FSub;
  bool IsXLHSInRHSPart = false;

  BasicBlock *EntryBB = BB;
  OpenMPIRBuilder::InsertPointTy AllocaIP(EntryBB,
                                          EntryBB->getFirstInsertionPt());
  Value *Sub = nullptr;

  auto UpdateOp = [&](Value *Atomic, IRBuilder<> &IRB) {
    Sub = IRB.CreateFSub(ConstVal, Atomic);
    return Sub;
  };
  Builder.restoreIP(OMPBuilder.createAtomicUpdate(
      Builder, AllocaIP, X, Expr, AO, RMWOp, UpdateOp, IsXLHSInRHSPart));
  BasicBlock *ContBB = EntryBB->getSingleSuccessor();
  BranchInst *ContTI = dyn_cast<BranchInst>(ContBB->getTerminator());
  EXPECT_NE(ContTI, nullptr);
  BasicBlock *EndBB = ContTI->getSuccessor(0);
  EXPECT_TRUE(ContTI->isConditional());
  EXPECT_EQ(ContTI->getSuccessor(1), ContBB);
  EXPECT_NE(EndBB, nullptr);

  PHINode *Phi = dyn_cast<PHINode>(&ContBB->front());
  EXPECT_NE(Phi, nullptr);
  EXPECT_EQ(Phi->getNumIncomingValues(), 2U);
  EXPECT_EQ(Phi->getIncomingBlock(0), EntryBB);
  EXPECT_EQ(Phi->getIncomingBlock(1), ContBB);

  EXPECT_EQ(Sub->getNumUses(), 1U);
  StoreInst *St = dyn_cast<StoreInst>(Sub->user_back());
  AllocaInst *UpdateTemp = dyn_cast<AllocaInst>(St->getPointerOperand());

  ExtractValueInst *ExVI1 =
      dyn_cast<ExtractValueInst>(Phi->getIncomingValueForBlock(ContBB));
  EXPECT_NE(ExVI1, nullptr);
  AtomicCmpXchgInst *CmpExchg =
      dyn_cast<AtomicCmpXchgInst>(ExVI1->getAggregateOperand());
  EXPECT_NE(CmpExchg, nullptr);
  BitCastInst *BitCastNew =
      dyn_cast<BitCastInst>(CmpExchg->getPointerOperand());
  EXPECT_NE(BitCastNew, nullptr);
  EXPECT_EQ(BitCastNew->getOperand(0), XVal);
  EXPECT_EQ(CmpExchg->getCompareOperand(), Phi);
  EXPECT_EQ(CmpExchg->getSuccessOrdering(), AtomicOrdering::Monotonic);

  LoadInst *Ld = dyn_cast<LoadInst>(CmpExchg->getNewValOperand());
  EXPECT_NE(Ld, nullptr);
  BitCastInst *BitCastOld = dyn_cast<BitCastInst>(Ld->getPointerOperand());
  EXPECT_NE(BitCastOld, nullptr);
  EXPECT_EQ(UpdateTemp, BitCastOld->getOperand(0));

  Builder.CreateRetVoid();
  OMPBuilder.finalize();
  EXPECT_FALSE(verifyModule(*M, &errs()));
}

TEST_F(OpenMPIRBuilderTest, OMPAtomicUpdateIntr) {
  OpenMPIRBuilder OMPBuilder(*M);
  OMPBuilder.initialize();
  F->setName("func");
  IRBuilder<> Builder(BB);

  OpenMPIRBuilder::LocationDescription Loc({Builder.saveIP(), DL});

  Type *IntTy = Type::getInt32Ty(M->getContext());
  AllocaInst *XVal = Builder.CreateAlloca(IntTy);
  XVal->setName("AtomicVar");
  Builder.CreateStore(ConstantInt::get(Type::getInt32Ty(Ctx), 0), XVal);
  OpenMPIRBuilder::AtomicOpValue X = {XVal, IntTy, false, false};
  AtomicOrdering AO = AtomicOrdering::Monotonic;
  Constant *ConstVal = ConstantInt::get(Type::getInt32Ty(Ctx), 1);
  Value *Expr = ConstantInt::get(Type::getInt32Ty(Ctx), 1);
  AtomicRMWInst::BinOp RMWOp = AtomicRMWInst::UMax;
  bool IsXLHSInRHSPart = false;

  BasicBlock *EntryBB = BB;
  OpenMPIRBuilder::InsertPointTy AllocaIP(EntryBB,
                                          EntryBB->getFirstInsertionPt());
  Value *Sub = nullptr;

  auto UpdateOp = [&](Value *Atomic, IRBuilder<> &IRB) {
    Sub = IRB.CreateSub(ConstVal, Atomic);
    return Sub;
  };
  Builder.restoreIP(OMPBuilder.createAtomicUpdate(
      Builder, AllocaIP, X, Expr, AO, RMWOp, UpdateOp, IsXLHSInRHSPart));
  BasicBlock *ContBB = EntryBB->getSingleSuccessor();
  BranchInst *ContTI = dyn_cast<BranchInst>(ContBB->getTerminator());
  EXPECT_NE(ContTI, nullptr);
  BasicBlock *EndBB = ContTI->getSuccessor(0);
  EXPECT_TRUE(ContTI->isConditional());
  EXPECT_EQ(ContTI->getSuccessor(1), ContBB);
  EXPECT_NE(EndBB, nullptr);

  PHINode *Phi = dyn_cast<PHINode>(&ContBB->front());
  EXPECT_NE(Phi, nullptr);
  EXPECT_EQ(Phi->getNumIncomingValues(), 2U);
  EXPECT_EQ(Phi->getIncomingBlock(0), EntryBB);
  EXPECT_EQ(Phi->getIncomingBlock(1), ContBB);

  EXPECT_EQ(Sub->getNumUses(), 1U);
  StoreInst *St = dyn_cast<StoreInst>(Sub->user_back());
  AllocaInst *UpdateTemp = dyn_cast<AllocaInst>(St->getPointerOperand());

  ExtractValueInst *ExVI1 =
      dyn_cast<ExtractValueInst>(Phi->getIncomingValueForBlock(ContBB));
  EXPECT_NE(ExVI1, nullptr);
  AtomicCmpXchgInst *CmpExchg =
      dyn_cast<AtomicCmpXchgInst>(ExVI1->getAggregateOperand());
  EXPECT_NE(CmpExchg, nullptr);
  EXPECT_EQ(CmpExchg->getPointerOperand(), XVal);
  EXPECT_EQ(CmpExchg->getCompareOperand(), Phi);
  EXPECT_EQ(CmpExchg->getSuccessOrdering(), AtomicOrdering::Monotonic);

  LoadInst *Ld = dyn_cast<LoadInst>(CmpExchg->getNewValOperand());
  EXPECT_NE(Ld, nullptr);
  EXPECT_EQ(UpdateTemp, Ld->getPointerOperand());

  Builder.CreateRetVoid();
  OMPBuilder.finalize();
  EXPECT_FALSE(verifyModule(*M, &errs()));
}

TEST_F(OpenMPIRBuilderTest, OMPAtomicCapture) {
  OpenMPIRBuilder OMPBuilder(*M);
  OMPBuilder.initialize();
  F->setName("func");
  IRBuilder<> Builder(BB);

  OpenMPIRBuilder::LocationDescription Loc({Builder.saveIP(), DL});

  LLVMContext &Ctx = M->getContext();
  IntegerType *Int32 = Type::getInt32Ty(Ctx);
  AllocaInst *XVal = Builder.CreateAlloca(Int32);
  XVal->setName("AtomicVar");
  AllocaInst *VVal = Builder.CreateAlloca(Int32);
  VVal->setName("AtomicCapTar");
  StoreInst *Init =
      Builder.CreateStore(ConstantInt::get(Type::getInt32Ty(Ctx), 0U), XVal);

  OpenMPIRBuilder::AtomicOpValue X = {XVal, Int32, false, false};
  OpenMPIRBuilder::AtomicOpValue V = {VVal, Int32, false, false};
  AtomicOrdering AO = AtomicOrdering::Monotonic;
  ConstantInt *Expr = ConstantInt::get(Type::getInt32Ty(Ctx), 1U);
  AtomicRMWInst::BinOp RMWOp = AtomicRMWInst::Add;
  bool IsXLHSInRHSPart = true;
  bool IsPostfixUpdate = true;
  bool UpdateExpr = true;

  BasicBlock *EntryBB = BB;
  OpenMPIRBuilder::InsertPointTy AllocaIP(EntryBB,
                                          EntryBB->getFirstInsertionPt());

  // integer update - not used
  auto UpdateOp = [&](Value *Atomic, IRBuilder<> &IRB) { return nullptr; };

  Builder.restoreIP(OMPBuilder.createAtomicCapture(
      Builder, AllocaIP, X, V, Expr, AO, RMWOp, UpdateOp, UpdateExpr,
      IsPostfixUpdate, IsXLHSInRHSPart));
  EXPECT_EQ(EntryBB->getParent()->size(), 1U);
  AtomicRMWInst *ARWM = dyn_cast<AtomicRMWInst>(Init->getNextNode());
  EXPECT_NE(ARWM, nullptr);
  EXPECT_EQ(ARWM->getPointerOperand(), XVal);
  EXPECT_EQ(ARWM->getOperation(), RMWOp);
  StoreInst *St = dyn_cast<StoreInst>(ARWM->user_back());
  EXPECT_NE(St, nullptr);
  EXPECT_EQ(St->getPointerOperand(), VVal);

  Builder.CreateRetVoid();
  OMPBuilder.finalize();
  EXPECT_FALSE(verifyModule(*M, &errs()));
}

TEST_F(OpenMPIRBuilderTest, OMPAtomicCompare) {
  OpenMPIRBuilder OMPBuilder(*M);
  OMPBuilder.initialize();
  F->setName("func");
  IRBuilder<> Builder(BB);

  OpenMPIRBuilder::LocationDescription Loc({Builder.saveIP(), DL});

  LLVMContext &Ctx = M->getContext();
  IntegerType *Int32 = Type::getInt32Ty(Ctx);
  AllocaInst *XVal = Builder.CreateAlloca(Int32);
  XVal->setName("x");
  StoreInst *Init =
      Builder.CreateStore(ConstantInt::get(Type::getInt32Ty(Ctx), 0U), XVal);

  OpenMPIRBuilder::AtomicOpValue XSigned = {XVal, Int32, true, false};
  OpenMPIRBuilder::AtomicOpValue XUnsigned = {XVal, Int32, false, false};
  // V and R are not used in atomic compare
  OpenMPIRBuilder::AtomicOpValue V = {nullptr, nullptr, false, false};
  OpenMPIRBuilder::AtomicOpValue R = {nullptr, nullptr, false, false};
  AtomicOrdering AO = AtomicOrdering::Monotonic;
  ConstantInt *Expr = ConstantInt::get(Type::getInt32Ty(Ctx), 1U);
  ConstantInt *D = ConstantInt::get(Type::getInt32Ty(Ctx), 1U);
  OMPAtomicCompareOp OpMax = OMPAtomicCompareOp::MAX;
  OMPAtomicCompareOp OpEQ = OMPAtomicCompareOp::EQ;

  Builder.restoreIP(OMPBuilder.createAtomicCompare(
      Builder, XSigned, V, R, Expr, nullptr, AO, OpMax, true, false, false));
  Builder.restoreIP(OMPBuilder.createAtomicCompare(
      Builder, XUnsigned, V, R, Expr, nullptr, AO, OpMax, false, false, false));
  Builder.restoreIP(OMPBuilder.createAtomicCompare(
      Builder, XSigned, V, R, Expr, D, AO, OpEQ, true, false, false));

  BasicBlock *EntryBB = BB;
  EXPECT_EQ(EntryBB->getParent()->size(), 1U);
  EXPECT_EQ(EntryBB->size(), 5U);

  AtomicRMWInst *ARWM1 = dyn_cast<AtomicRMWInst>(Init->getNextNode());
  EXPECT_NE(ARWM1, nullptr);
  EXPECT_EQ(ARWM1->getPointerOperand(), XVal);
  EXPECT_EQ(ARWM1->getValOperand(), Expr);
  EXPECT_EQ(ARWM1->getOperation(), AtomicRMWInst::Min);

  AtomicRMWInst *ARWM2 = dyn_cast<AtomicRMWInst>(ARWM1->getNextNode());
  EXPECT_NE(ARWM2, nullptr);
  EXPECT_EQ(ARWM2->getPointerOperand(), XVal);
  EXPECT_EQ(ARWM2->getValOperand(), Expr);
  EXPECT_EQ(ARWM2->getOperation(), AtomicRMWInst::UMax);

  AtomicCmpXchgInst *AXCHG = dyn_cast<AtomicCmpXchgInst>(ARWM2->getNextNode());
  EXPECT_NE(AXCHG, nullptr);
  EXPECT_EQ(AXCHG->getPointerOperand(), XVal);
  EXPECT_EQ(AXCHG->getCompareOperand(), Expr);
  EXPECT_EQ(AXCHG->getNewValOperand(), D);

  Builder.CreateRetVoid();
  OMPBuilder.finalize();
  EXPECT_FALSE(verifyModule(*M, &errs()));
}

TEST_F(OpenMPIRBuilderTest, OMPAtomicCompareCapture) {
  OpenMPIRBuilder OMPBuilder(*M);
  OMPBuilder.initialize();
  F->setName("func");
  IRBuilder<> Builder(BB);

  OpenMPIRBuilder::LocationDescription Loc({Builder.saveIP(), DL});

  LLVMContext &Ctx = M->getContext();
  IntegerType *Int32 = Type::getInt32Ty(Ctx);
  AllocaInst *XVal = Builder.CreateAlloca(Int32);
  XVal->setName("x");
  AllocaInst *VVal = Builder.CreateAlloca(Int32);
  VVal->setName("v");
  AllocaInst *RVal = Builder.CreateAlloca(Int32);
  RVal->setName("r");

  StoreInst *Init =
      Builder.CreateStore(ConstantInt::get(Type::getInt32Ty(Ctx), 0U), XVal);

  OpenMPIRBuilder::AtomicOpValue X = {XVal, Int32, true, false};
  OpenMPIRBuilder::AtomicOpValue V = {VVal, Int32, false, false};
  OpenMPIRBuilder::AtomicOpValue NoV = {nullptr, nullptr, false, false};
  OpenMPIRBuilder::AtomicOpValue R = {RVal, Int32, false, false};
  OpenMPIRBuilder::AtomicOpValue NoR = {nullptr, nullptr, false, false};

  AtomicOrdering AO = AtomicOrdering::Monotonic;
  ConstantInt *Expr = ConstantInt::get(Type::getInt32Ty(Ctx), 1U);
  ConstantInt *D = ConstantInt::get(Type::getInt32Ty(Ctx), 1U);
  OMPAtomicCompareOp OpMax = OMPAtomicCompareOp::MAX;
  OMPAtomicCompareOp OpEQ = OMPAtomicCompareOp::EQ;

  // { cond-update-stmt v = x; }
  Builder.restoreIP(OMPBuilder.createAtomicCompare(
      Builder, X, V, NoR, Expr, D, AO, OpEQ, /* IsXBinopExpr */ true,
      /* IsPostfixUpdate */ false,
      /* IsFailOnly */ false));
  // { v = x; cond-update-stmt }
  Builder.restoreIP(OMPBuilder.createAtomicCompare(
      Builder, X, V, NoR, Expr, D, AO, OpEQ, /* IsXBinopExpr */ true,
      /* IsPostfixUpdate */ true,
      /* IsFailOnly */ false));
  // if(x == e) { x = d; } else { v = x; }
  Builder.restoreIP(OMPBuilder.createAtomicCompare(
      Builder, X, V, NoR, Expr, D, AO, OpEQ, /* IsXBinopExpr */ true,
      /* IsPostfixUpdate */ false,
      /* IsFailOnly */ true));
  // { r = x == e; if(r) { x = d; } }
  Builder.restoreIP(OMPBuilder.createAtomicCompare(
      Builder, X, NoV, R, Expr, D, AO, OpEQ, /* IsXBinopExpr */ true,
      /* IsPostfixUpdate */ false,
      /* IsFailOnly */ false));
  // { r = x == e; if(r) { x = d; } else { v = x; } }
  Builder.restoreIP(OMPBuilder.createAtomicCompare(
      Builder, X, V, R, Expr, D, AO, OpEQ, /* IsXBinopExpr */ true,
      /* IsPostfixUpdate */ false,
      /* IsFailOnly */ true));

  // { v = x; cond-update-stmt }
  Builder.restoreIP(OMPBuilder.createAtomicCompare(
      Builder, X, V, NoR, Expr, nullptr, AO, OpMax, /* IsXBinopExpr */ true,
      /* IsPostfixUpdate */ true,
      /* IsFailOnly */ false));
  // { cond-update-stmt v = x; }
  Builder.restoreIP(OMPBuilder.createAtomicCompare(
      Builder, X, V, NoR, Expr, nullptr, AO, OpMax, /* IsXBinopExpr */ false,
      /* IsPostfixUpdate */ false,
      /* IsFailOnly */ false));

  BasicBlock *EntryBB = BB;
  EXPECT_EQ(EntryBB->getParent()->size(), 5U);
  BasicBlock *Cont1 = dyn_cast<BasicBlock>(EntryBB->getNextNode());
  EXPECT_NE(Cont1, nullptr);
  BasicBlock *Exit1 = dyn_cast<BasicBlock>(Cont1->getNextNode());
  EXPECT_NE(Exit1, nullptr);
  BasicBlock *Cont2 = dyn_cast<BasicBlock>(Exit1->getNextNode());
  EXPECT_NE(Cont2, nullptr);
  BasicBlock *Exit2 = dyn_cast<BasicBlock>(Cont2->getNextNode());
  EXPECT_NE(Exit2, nullptr);

  AtomicCmpXchgInst *CmpXchg1 =
      dyn_cast<AtomicCmpXchgInst>(Init->getNextNode());
  EXPECT_NE(CmpXchg1, nullptr);
  EXPECT_EQ(CmpXchg1->getPointerOperand(), XVal);
  EXPECT_EQ(CmpXchg1->getCompareOperand(), Expr);
  EXPECT_EQ(CmpXchg1->getNewValOperand(), D);
  ExtractValueInst *ExtVal1 =
      dyn_cast<ExtractValueInst>(CmpXchg1->getNextNode());
  EXPECT_NE(ExtVal1, nullptr);
  EXPECT_EQ(ExtVal1->getAggregateOperand(), CmpXchg1);
  EXPECT_EQ(ExtVal1->getIndices(), ArrayRef<unsigned int>(0U));
  ExtractValueInst *ExtVal2 =
      dyn_cast<ExtractValueInst>(ExtVal1->getNextNode());
  EXPECT_NE(ExtVal2, nullptr);
  EXPECT_EQ(ExtVal2->getAggregateOperand(), CmpXchg1);
  EXPECT_EQ(ExtVal2->getIndices(), ArrayRef<unsigned int>(1U));
  SelectInst *Sel1 = dyn_cast<SelectInst>(ExtVal2->getNextNode());
  EXPECT_NE(Sel1, nullptr);
  EXPECT_EQ(Sel1->getCondition(), ExtVal2);
  EXPECT_EQ(Sel1->getTrueValue(), Expr);
  EXPECT_EQ(Sel1->getFalseValue(), ExtVal1);
  StoreInst *Store1 = dyn_cast<StoreInst>(Sel1->getNextNode());
  EXPECT_NE(Store1, nullptr);
  EXPECT_EQ(Store1->getPointerOperand(), VVal);
  EXPECT_EQ(Store1->getValueOperand(), Sel1);

  AtomicCmpXchgInst *CmpXchg2 =
      dyn_cast<AtomicCmpXchgInst>(Store1->getNextNode());
  EXPECT_NE(CmpXchg2, nullptr);
  EXPECT_EQ(CmpXchg2->getPointerOperand(), XVal);
  EXPECT_EQ(CmpXchg2->getCompareOperand(), Expr);
  EXPECT_EQ(CmpXchg2->getNewValOperand(), D);
  ExtractValueInst *ExtVal3 =
      dyn_cast<ExtractValueInst>(CmpXchg2->getNextNode());
  EXPECT_NE(ExtVal3, nullptr);
  EXPECT_EQ(ExtVal3->getAggregateOperand(), CmpXchg2);
  EXPECT_EQ(ExtVal3->getIndices(), ArrayRef<unsigned int>(0U));
  StoreInst *Store2 = dyn_cast<StoreInst>(ExtVal3->getNextNode());
  EXPECT_NE(Store2, nullptr);
  EXPECT_EQ(Store2->getPointerOperand(), VVal);
  EXPECT_EQ(Store2->getValueOperand(), ExtVal3);

  AtomicCmpXchgInst *CmpXchg3 =
      dyn_cast<AtomicCmpXchgInst>(Store2->getNextNode());
  EXPECT_NE(CmpXchg3, nullptr);
  EXPECT_EQ(CmpXchg3->getPointerOperand(), XVal);
  EXPECT_EQ(CmpXchg3->getCompareOperand(), Expr);
  EXPECT_EQ(CmpXchg3->getNewValOperand(), D);
  ExtractValueInst *ExtVal4 =
      dyn_cast<ExtractValueInst>(CmpXchg3->getNextNode());
  EXPECT_NE(ExtVal4, nullptr);
  EXPECT_EQ(ExtVal4->getAggregateOperand(), CmpXchg3);
  EXPECT_EQ(ExtVal4->getIndices(), ArrayRef<unsigned int>(0U));
  ExtractValueInst *ExtVal5 =
      dyn_cast<ExtractValueInst>(ExtVal4->getNextNode());
  EXPECT_NE(ExtVal5, nullptr);
  EXPECT_EQ(ExtVal5->getAggregateOperand(), CmpXchg3);
  EXPECT_EQ(ExtVal5->getIndices(), ArrayRef<unsigned int>(1U));
  BranchInst *Br1 = dyn_cast<BranchInst>(ExtVal5->getNextNode());
  EXPECT_NE(Br1, nullptr);
  EXPECT_EQ(Br1->isConditional(), true);
  EXPECT_EQ(Br1->getCondition(), ExtVal5);
  EXPECT_EQ(Br1->getSuccessor(0), Exit1);
  EXPECT_EQ(Br1->getSuccessor(1), Cont1);

  StoreInst *Store3 = dyn_cast<StoreInst>(&Cont1->front());
  EXPECT_NE(Store3, nullptr);
  EXPECT_EQ(Store3->getPointerOperand(), VVal);
  EXPECT_EQ(Store3->getValueOperand(), ExtVal4);
  BranchInst *Br2 = dyn_cast<BranchInst>(Store3->getNextNode());
  EXPECT_NE(Br2, nullptr);
  EXPECT_EQ(Br2->isUnconditional(), true);
  EXPECT_EQ(Br2->getSuccessor(0), Exit1);

  AtomicCmpXchgInst *CmpXchg4 = dyn_cast<AtomicCmpXchgInst>(&Exit1->front());
  EXPECT_NE(CmpXchg4, nullptr);
  EXPECT_EQ(CmpXchg4->getPointerOperand(), XVal);
  EXPECT_EQ(CmpXchg4->getCompareOperand(), Expr);
  EXPECT_EQ(CmpXchg4->getNewValOperand(), D);
  ExtractValueInst *ExtVal6 =
      dyn_cast<ExtractValueInst>(CmpXchg4->getNextNode());
  EXPECT_NE(ExtVal6, nullptr);
  EXPECT_EQ(ExtVal6->getAggregateOperand(), CmpXchg4);
  EXPECT_EQ(ExtVal6->getIndices(), ArrayRef<unsigned int>(1U));
  ZExtInst *ZExt1 = dyn_cast<ZExtInst>(ExtVal6->getNextNode());
  EXPECT_NE(ZExt1, nullptr);
  EXPECT_EQ(ZExt1->getDestTy(), Int32);
  StoreInst *Store4 = dyn_cast<StoreInst>(ZExt1->getNextNode());
  EXPECT_NE(Store4, nullptr);
  EXPECT_EQ(Store4->getPointerOperand(), RVal);
  EXPECT_EQ(Store4->getValueOperand(), ZExt1);

  AtomicCmpXchgInst *CmpXchg5 =
      dyn_cast<AtomicCmpXchgInst>(Store4->getNextNode());
  EXPECT_NE(CmpXchg5, nullptr);
  EXPECT_EQ(CmpXchg5->getPointerOperand(), XVal);
  EXPECT_EQ(CmpXchg5->getCompareOperand(), Expr);
  EXPECT_EQ(CmpXchg5->getNewValOperand(), D);
  ExtractValueInst *ExtVal7 =
      dyn_cast<ExtractValueInst>(CmpXchg5->getNextNode());
  EXPECT_NE(ExtVal7, nullptr);
  EXPECT_EQ(ExtVal7->getAggregateOperand(), CmpXchg5);
  EXPECT_EQ(ExtVal7->getIndices(), ArrayRef<unsigned int>(0U));
  ExtractValueInst *ExtVal8 =
      dyn_cast<ExtractValueInst>(ExtVal7->getNextNode());
  EXPECT_NE(ExtVal8, nullptr);
  EXPECT_EQ(ExtVal8->getAggregateOperand(), CmpXchg5);
  EXPECT_EQ(ExtVal8->getIndices(), ArrayRef<unsigned int>(1U));
  BranchInst *Br3 = dyn_cast<BranchInst>(ExtVal8->getNextNode());
  EXPECT_NE(Br3, nullptr);
  EXPECT_EQ(Br3->isConditional(), true);
  EXPECT_EQ(Br3->getCondition(), ExtVal8);
  EXPECT_EQ(Br3->getSuccessor(0), Exit2);
  EXPECT_EQ(Br3->getSuccessor(1), Cont2);

  StoreInst *Store5 = dyn_cast<StoreInst>(&Cont2->front());
  EXPECT_NE(Store5, nullptr);
  EXPECT_EQ(Store5->getPointerOperand(), VVal);
  EXPECT_EQ(Store5->getValueOperand(), ExtVal7);
  BranchInst *Br4 = dyn_cast<BranchInst>(Store5->getNextNode());
  EXPECT_NE(Br4, nullptr);
  EXPECT_EQ(Br4->isUnconditional(), true);
  EXPECT_EQ(Br4->getSuccessor(0), Exit2);

  ExtractValueInst *ExtVal9 = dyn_cast<ExtractValueInst>(&Exit2->front());
  EXPECT_NE(ExtVal9, nullptr);
  EXPECT_EQ(ExtVal9->getAggregateOperand(), CmpXchg5);
  EXPECT_EQ(ExtVal9->getIndices(), ArrayRef<unsigned int>(1U));
  ZExtInst *ZExt2 = dyn_cast<ZExtInst>(ExtVal9->getNextNode());
  EXPECT_NE(ZExt2, nullptr);
  EXPECT_EQ(ZExt2->getDestTy(), Int32);
  StoreInst *Store6 = dyn_cast<StoreInst>(ZExt2->getNextNode());
  EXPECT_NE(Store6, nullptr);
  EXPECT_EQ(Store6->getPointerOperand(), RVal);
  EXPECT_EQ(Store6->getValueOperand(), ZExt2);

  AtomicRMWInst *ARWM1 = dyn_cast<AtomicRMWInst>(Store6->getNextNode());
  EXPECT_NE(ARWM1, nullptr);
  EXPECT_EQ(ARWM1->getPointerOperand(), XVal);
  EXPECT_EQ(ARWM1->getValOperand(), Expr);
  EXPECT_EQ(ARWM1->getOperation(), AtomicRMWInst::Min);
  StoreInst *Store7 = dyn_cast<StoreInst>(ARWM1->getNextNode());
  EXPECT_NE(Store7, nullptr);
  EXPECT_EQ(Store7->getPointerOperand(), VVal);
  EXPECT_EQ(Store7->getValueOperand(), ARWM1);

  AtomicRMWInst *ARWM2 = dyn_cast<AtomicRMWInst>(Store7->getNextNode());
  EXPECT_NE(ARWM2, nullptr);
  EXPECT_EQ(ARWM2->getPointerOperand(), XVal);
  EXPECT_EQ(ARWM2->getValOperand(), Expr);
  EXPECT_EQ(ARWM2->getOperation(), AtomicRMWInst::Max);
  CmpInst *Cmp1 = dyn_cast<CmpInst>(ARWM2->getNextNode());
  EXPECT_NE(Cmp1, nullptr);
  EXPECT_EQ(Cmp1->getPredicate(), CmpInst::ICMP_SGT);
  EXPECT_EQ(Cmp1->getOperand(0), ARWM2);
  EXPECT_EQ(Cmp1->getOperand(1), Expr);
  SelectInst *Sel2 = dyn_cast<SelectInst>(Cmp1->getNextNode());
  EXPECT_NE(Sel2, nullptr);
  EXPECT_EQ(Sel2->getCondition(), Cmp1);
  EXPECT_EQ(Sel2->getTrueValue(), Expr);
  EXPECT_EQ(Sel2->getFalseValue(), ARWM2);
  StoreInst *Store8 = dyn_cast<StoreInst>(Sel2->getNextNode());
  EXPECT_NE(Store8, nullptr);
  EXPECT_EQ(Store8->getPointerOperand(), VVal);
  EXPECT_EQ(Store8->getValueOperand(), Sel2);

  Builder.CreateRetVoid();
  OMPBuilder.finalize();
  EXPECT_FALSE(verifyModule(*M, &errs()));
}

/// Returns the single instruction of InstTy type in BB that uses the value V.
/// If there is more than one such instruction, returns null.
template <typename InstTy>
static InstTy *findSingleUserInBlock(Value *V, BasicBlock *BB) {
  InstTy *Result = nullptr;
  for (User *U : V->users()) {
    auto *Inst = dyn_cast<InstTy>(U);
    if (!Inst || Inst->getParent() != BB)
      continue;
    if (Result)
      return nullptr;
    Result = Inst;
  }
  return Result;
}

/// Returns true if BB contains a simple binary reduction that loads a value
/// from Accum, performs some binary operation with it, and stores it back to
/// Accum.
static bool isSimpleBinaryReduction(Value *Accum, BasicBlock *BB,
                                    Instruction::BinaryOps *OpCode = nullptr) {
  StoreInst *Store = findSingleUserInBlock<StoreInst>(Accum, BB);
  if (!Store)
    return false;
  auto *Stored = dyn_cast<BinaryOperator>(Store->getOperand(0));
  if (!Stored)
    return false;
  if (OpCode && *OpCode != Stored->getOpcode())
    return false;
  auto *Load = dyn_cast<LoadInst>(Stored->getOperand(0));
  return Load && Load->getOperand(0) == Accum;
}

/// Returns true if BB contains a binary reduction that reduces V using a binary
/// operator into an accumulator that is a function argument.
static bool isValueReducedToFuncArg(Value *V, BasicBlock *BB) {
  auto *ReductionOp = findSingleUserInBlock<BinaryOperator>(V, BB);
  if (!ReductionOp)
    return false;

  auto *GlobalLoad = dyn_cast<LoadInst>(ReductionOp->getOperand(0));
  if (!GlobalLoad)
    return false;

  auto *Store = findSingleUserInBlock<StoreInst>(ReductionOp, BB);
  if (!Store)
    return false;

  return Store->getPointerOperand() == GlobalLoad->getPointerOperand() &&
         isa<Argument>(findAggregateFromValue(GlobalLoad->getPointerOperand()));
}

/// Finds among users of Ptr a pair of GEP instructions with indices [0, 0] and
/// [0, 1], respectively, and assigns results of these instructions to Zero and
/// One. Returns true on success, false on failure or if such instructions are
/// not unique among the users of Ptr.
static bool findGEPZeroOne(Value *Ptr, Value *&Zero, Value *&One) {
  Zero = nullptr;
  One = nullptr;
  for (User *U : Ptr->users()) {
    if (auto *GEP = dyn_cast<GetElementPtrInst>(U)) {
      if (GEP->getNumIndices() != 2)
        continue;
      auto *FirstIdx = dyn_cast<ConstantInt>(GEP->getOperand(1));
      auto *SecondIdx = dyn_cast<ConstantInt>(GEP->getOperand(2));
      EXPECT_NE(FirstIdx, nullptr);
      EXPECT_NE(SecondIdx, nullptr);

      EXPECT_TRUE(FirstIdx->isZero());
      if (SecondIdx->isZero()) {
        if (Zero)
          return false;
        Zero = GEP;
      } else if (SecondIdx->isOne()) {
        if (One)
          return false;
        One = GEP;
      } else {
        return false;
      }
    }
  }
  return Zero != nullptr && One != nullptr;
}

static OpenMPIRBuilder::InsertPointTy
sumReduction(OpenMPIRBuilder::InsertPointTy IP, Value *LHS, Value *RHS,
             Value *&Result) {
  IRBuilder<> Builder(IP.getBlock(), IP.getPoint());
  Result = Builder.CreateFAdd(LHS, RHS, "red.add");
  return Builder.saveIP();
}

static OpenMPIRBuilder::InsertPointTy
sumAtomicReduction(OpenMPIRBuilder::InsertPointTy IP, Type *Ty, Value *LHS,
                   Value *RHS) {
  IRBuilder<> Builder(IP.getBlock(), IP.getPoint());
  Value *Partial = Builder.CreateLoad(Ty, RHS, "red.partial");
  Builder.CreateAtomicRMW(AtomicRMWInst::FAdd, LHS, Partial, None,
                          AtomicOrdering::Monotonic);
  return Builder.saveIP();
}

static OpenMPIRBuilder::InsertPointTy
xorReduction(OpenMPIRBuilder::InsertPointTy IP, Value *LHS, Value *RHS,
             Value *&Result) {
  IRBuilder<> Builder(IP.getBlock(), IP.getPoint());
  Result = Builder.CreateXor(LHS, RHS, "red.xor");
  return Builder.saveIP();
}

static OpenMPIRBuilder::InsertPointTy
xorAtomicReduction(OpenMPIRBuilder::InsertPointTy IP, Type *Ty, Value *LHS,
                   Value *RHS) {
  IRBuilder<> Builder(IP.getBlock(), IP.getPoint());
  Value *Partial = Builder.CreateLoad(Ty, RHS, "red.partial");
  Builder.CreateAtomicRMW(AtomicRMWInst::Xor, LHS, Partial, None,
                          AtomicOrdering::Monotonic);
  return Builder.saveIP();
}

TEST_F(OpenMPIRBuilderTest, CreateReductions) {
  using InsertPointTy = OpenMPIRBuilder::InsertPointTy;
  OpenMPIRBuilder OMPBuilder(*M);
  OMPBuilder.initialize();
  F->setName("func");
  IRBuilder<> Builder(BB);

  BasicBlock *EnterBB = BasicBlock::Create(Ctx, "parallel.enter", F);
  Builder.CreateBr(EnterBB);
  Builder.SetInsertPoint(EnterBB);
  OpenMPIRBuilder::LocationDescription Loc({Builder.saveIP(), DL});

  // Create variables to be reduced.
  InsertPointTy OuterAllocaIP(&F->getEntryBlock(),
                              F->getEntryBlock().getFirstInsertionPt());
  Type *SumType = Builder.getFloatTy();
  Type *XorType = Builder.getInt32Ty();
  Value *SumReduced;
  Value *XorReduced;
  {
    IRBuilderBase::InsertPointGuard Guard(Builder);
    Builder.restoreIP(OuterAllocaIP);
    SumReduced = Builder.CreateAlloca(SumType);
    XorReduced = Builder.CreateAlloca(XorType);
  }

  // Store initial values of reductions into global variables.
  Builder.CreateStore(ConstantFP::get(Builder.getFloatTy(), 0.0), SumReduced);
  Builder.CreateStore(Builder.getInt32(1), XorReduced);

  // The loop body computes two reductions:
  //   sum of (float) thread-id;
  //   xor of thread-id;
  // and store the result in global variables.
  InsertPointTy BodyIP, BodyAllocaIP;
  auto BodyGenCB = [&](InsertPointTy InnerAllocaIP, InsertPointTy CodeGenIP) {
    IRBuilderBase::InsertPointGuard Guard(Builder);
    Builder.restoreIP(CodeGenIP);

    uint32_t StrSize;
    Constant *SrcLocStr = OMPBuilder.getOrCreateSrcLocStr(Loc, StrSize);
    Value *Ident = OMPBuilder.getOrCreateIdent(SrcLocStr, StrSize);
    Value *TID = OMPBuilder.getOrCreateThreadID(Ident);
    Value *SumLocal =
        Builder.CreateUIToFP(TID, Builder.getFloatTy(), "sum.local");
    Value *SumPartial = Builder.CreateLoad(SumType, SumReduced, "sum.partial");
    Value *XorPartial = Builder.CreateLoad(XorType, XorReduced, "xor.partial");
    Value *Sum = Builder.CreateFAdd(SumPartial, SumLocal, "sum");
    Value *Xor = Builder.CreateXor(XorPartial, TID, "xor");
    Builder.CreateStore(Sum, SumReduced);
    Builder.CreateStore(Xor, XorReduced);

    BodyIP = Builder.saveIP();
    BodyAllocaIP = InnerAllocaIP;
  };

  // Privatization for reduction creates local copies of reduction variables and
  // initializes them to reduction-neutral values.
  Value *SumPrivatized;
  Value *XorPrivatized;
  auto PrivCB = [&](InsertPointTy InnerAllocaIP, InsertPointTy CodeGenIP,
                    Value &Original, Value &Inner, Value *&ReplVal) {
    IRBuilderBase::InsertPointGuard Guard(Builder);
    Builder.restoreIP(InnerAllocaIP);
    if (&Original == SumReduced) {
      SumPrivatized = Builder.CreateAlloca(Builder.getFloatTy());
      ReplVal = SumPrivatized;
    } else if (&Original == XorReduced) {
      XorPrivatized = Builder.CreateAlloca(Builder.getInt32Ty());
      ReplVal = XorPrivatized;
    } else {
      ReplVal = &Inner;
      return CodeGenIP;
    }

    Builder.restoreIP(CodeGenIP);
    if (&Original == SumReduced)
      Builder.CreateStore(ConstantFP::get(Builder.getFloatTy(), 0.0),
                          SumPrivatized);
    else if (&Original == XorReduced)
      Builder.CreateStore(Builder.getInt32(0), XorPrivatized);

    return Builder.saveIP();
  };

  // Do nothing in finalization.
  auto FiniCB = [&](InsertPointTy CodeGenIP) { return CodeGenIP; };

  InsertPointTy AfterIP =
      OMPBuilder.createParallel(Loc, OuterAllocaIP, BodyGenCB, PrivCB, FiniCB,
                                /* IfCondition */ nullptr,
                                /* NumThreads */ nullptr, OMP_PROC_BIND_default,
                                /* IsCancellable */ false);
  Builder.restoreIP(AfterIP);

  OpenMPIRBuilder::ReductionInfo ReductionInfos[] = {
      {SumType, SumReduced, SumPrivatized, sumReduction, sumAtomicReduction},
      {XorType, XorReduced, XorPrivatized, xorReduction, xorAtomicReduction}};

  OMPBuilder.createReductions(BodyIP, BodyAllocaIP, ReductionInfos);

  Builder.restoreIP(AfterIP);
  Builder.CreateRetVoid();

  OMPBuilder.finalize(F);

  // The IR must be valid.
  EXPECT_FALSE(verifyModule(*M));

  // Outlining must have happened.
  SmallVector<CallInst *> ForkCalls;
  findCalls(F, omp::RuntimeFunction::OMPRTL___kmpc_fork_call, OMPBuilder,
            ForkCalls);
  ASSERT_EQ(ForkCalls.size(), 1u);
  Value *CalleeVal = cast<Constant>(ForkCalls[0]->getOperand(2))->getOperand(0);
  Function *Outlined = dyn_cast<Function>(CalleeVal);
  EXPECT_NE(Outlined, nullptr);

  // Check that the lock variable was created with the expected name.
  GlobalVariable *LockVar =
      M->getGlobalVariable(".gomp_critical_user_.reduction.var");
  EXPECT_NE(LockVar, nullptr);

  // Find the allocation of a local array that will be used to call the runtime
  // reduciton function.
  BasicBlock &AllocBlock = Outlined->getEntryBlock();
  Value *LocalArray = nullptr;
  for (Instruction &I : AllocBlock) {
    if (AllocaInst *Alloc = dyn_cast<AllocaInst>(&I)) {
      if (!Alloc->getAllocatedType()->isArrayTy() ||
          !Alloc->getAllocatedType()->getArrayElementType()->isPointerTy())
        continue;
      LocalArray = Alloc;
      break;
    }
  }
  ASSERT_NE(LocalArray, nullptr);

  // Find the call to the runtime reduction function.
  BasicBlock *BB = AllocBlock.getUniqueSuccessor();
  Value *LocalArrayPtr = nullptr;
  Value *ReductionFnVal = nullptr;
  Value *SwitchArg = nullptr;
  for (Instruction &I : *BB) {
    if (CallInst *Call = dyn_cast<CallInst>(&I)) {
      if (Call->getCalledFunction() !=
          OMPBuilder.getOrCreateRuntimeFunctionPtr(
              RuntimeFunction::OMPRTL___kmpc_reduce))
        continue;
      LocalArrayPtr = Call->getOperand(4);
      ReductionFnVal = Call->getOperand(5);
      SwitchArg = Call;
      break;
    }
  }

  // Check that the local array is passed to the function.
  ASSERT_NE(LocalArrayPtr, nullptr);
  BitCastInst *BitCast = dyn_cast<BitCastInst>(LocalArrayPtr);
  ASSERT_NE(BitCast, nullptr);
  EXPECT_EQ(BitCast->getOperand(0), LocalArray);

  // Find the GEP instructions preceding stores to the local array.
  Value *FirstArrayElemPtr = nullptr;
  Value *SecondArrayElemPtr = nullptr;
  EXPECT_EQ(LocalArray->getNumUses(), 3u);
  ASSERT_TRUE(
      findGEPZeroOne(LocalArray, FirstArrayElemPtr, SecondArrayElemPtr));

  // Check that the values stored into the local array are privatized reduction
  // variables.
  auto *FirstStored = dyn_cast_or_null<BitCastInst>(
      findStoredValue<GetElementPtrInst>(FirstArrayElemPtr));
  auto *SecondStored = dyn_cast_or_null<BitCastInst>(
      findStoredValue<GetElementPtrInst>(SecondArrayElemPtr));
  ASSERT_NE(FirstStored, nullptr);
  ASSERT_NE(SecondStored, nullptr);
  Value *FirstPrivatized = FirstStored->getOperand(0);
  Value *SecondPrivatized = SecondStored->getOperand(0);
  EXPECT_TRUE(
      isSimpleBinaryReduction(FirstPrivatized, FirstStored->getParent()));
  EXPECT_TRUE(
      isSimpleBinaryReduction(SecondPrivatized, SecondStored->getParent()));

  // Check that the result of the runtime reduction call is used for further
  // dispatch.
  ASSERT_EQ(SwitchArg->getNumUses(), 1u);
  SwitchInst *Switch = dyn_cast<SwitchInst>(*SwitchArg->user_begin());
  ASSERT_NE(Switch, nullptr);
  EXPECT_EQ(Switch->getNumSuccessors(), 3u);
  BasicBlock *NonAtomicBB = Switch->case_begin()->getCaseSuccessor();
  BasicBlock *AtomicBB = std::next(Switch->case_begin())->getCaseSuccessor();

  // Non-atomic block contains reductions to the global reduction variable,
  // which is passed into the outlined function as an argument.
  Value *FirstLoad =
      findSingleUserInBlock<LoadInst>(FirstPrivatized, NonAtomicBB);
  Value *SecondLoad =
      findSingleUserInBlock<LoadInst>(SecondPrivatized, NonAtomicBB);
  EXPECT_TRUE(isValueReducedToFuncArg(FirstLoad, NonAtomicBB));
  EXPECT_TRUE(isValueReducedToFuncArg(SecondLoad, NonAtomicBB));

  // Atomic block also constains reductions to the global reduction variable.
  FirstLoad = findSingleUserInBlock<LoadInst>(FirstPrivatized, AtomicBB);
  SecondLoad = findSingleUserInBlock<LoadInst>(SecondPrivatized, AtomicBB);
  auto *FirstAtomic = findSingleUserInBlock<AtomicRMWInst>(FirstLoad, AtomicBB);
  auto *SecondAtomic =
      findSingleUserInBlock<AtomicRMWInst>(SecondLoad, AtomicBB);
  ASSERT_NE(FirstAtomic, nullptr);
  Value *AtomicStorePointer = FirstAtomic->getPointerOperand();
  EXPECT_TRUE(isa<Argument>(findAggregateFromValue(AtomicStorePointer)));
  ASSERT_NE(SecondAtomic, nullptr);
  AtomicStorePointer = SecondAtomic->getPointerOperand();
  EXPECT_TRUE(isa<Argument>(findAggregateFromValue(AtomicStorePointer)));

  // Check that the separate reduction function also performs (non-atomic)
  // reductions after extracting reduction variables from its arguments.
  Function *ReductionFn = cast<Function>(ReductionFnVal);
  BasicBlock *FnReductionBB = &ReductionFn->getEntryBlock();
  auto *Bitcast =
      findSingleUserInBlock<BitCastInst>(ReductionFn->getArg(0), FnReductionBB);
  Value *FirstLHSPtr;
  Value *SecondLHSPtr;
  ASSERT_TRUE(findGEPZeroOne(Bitcast, FirstLHSPtr, SecondLHSPtr));
  Value *Opaque = findSingleUserInBlock<LoadInst>(FirstLHSPtr, FnReductionBB);
  ASSERT_NE(Opaque, nullptr);
  Bitcast = findSingleUserInBlock<BitCastInst>(Opaque, FnReductionBB);
  ASSERT_NE(Bitcast, nullptr);
  EXPECT_TRUE(isSimpleBinaryReduction(Bitcast, FnReductionBB));
  Opaque = findSingleUserInBlock<LoadInst>(SecondLHSPtr, FnReductionBB);
  ASSERT_NE(Opaque, nullptr);
  Bitcast = findSingleUserInBlock<BitCastInst>(Opaque, FnReductionBB);
  ASSERT_NE(Bitcast, nullptr);
  EXPECT_TRUE(isSimpleBinaryReduction(Bitcast, FnReductionBB));

  Bitcast =
      findSingleUserInBlock<BitCastInst>(ReductionFn->getArg(1), FnReductionBB);
  Value *FirstRHS;
  Value *SecondRHS;
  EXPECT_TRUE(findGEPZeroOne(Bitcast, FirstRHS, SecondRHS));
}

TEST_F(OpenMPIRBuilderTest, CreateTwoReductions) {
  using InsertPointTy = OpenMPIRBuilder::InsertPointTy;
  OpenMPIRBuilder OMPBuilder(*M);
  OMPBuilder.initialize();
  F->setName("func");
  IRBuilder<> Builder(BB);

  BasicBlock *EnterBB = BasicBlock::Create(Ctx, "parallel.enter", F);
  Builder.CreateBr(EnterBB);
  Builder.SetInsertPoint(EnterBB);
  OpenMPIRBuilder::LocationDescription Loc({Builder.saveIP(), DL});

  // Create variables to be reduced.
  InsertPointTy OuterAllocaIP(&F->getEntryBlock(),
                              F->getEntryBlock().getFirstInsertionPt());
  Type *SumType = Builder.getFloatTy();
  Type *XorType = Builder.getInt32Ty();
  Value *SumReduced;
  Value *XorReduced;
  {
    IRBuilderBase::InsertPointGuard Guard(Builder);
    Builder.restoreIP(OuterAllocaIP);
    SumReduced = Builder.CreateAlloca(SumType);
    XorReduced = Builder.CreateAlloca(XorType);
  }

  // Store initial values of reductions into global variables.
  Builder.CreateStore(ConstantFP::get(Builder.getFloatTy(), 0.0), SumReduced);
  Builder.CreateStore(Builder.getInt32(1), XorReduced);

  InsertPointTy FirstBodyIP, FirstBodyAllocaIP;
  auto FirstBodyGenCB = [&](InsertPointTy InnerAllocaIP,
                            InsertPointTy CodeGenIP) {
    IRBuilderBase::InsertPointGuard Guard(Builder);
    Builder.restoreIP(CodeGenIP);

    uint32_t StrSize;
    Constant *SrcLocStr = OMPBuilder.getOrCreateSrcLocStr(Loc, StrSize);
    Value *Ident = OMPBuilder.getOrCreateIdent(SrcLocStr, StrSize);
    Value *TID = OMPBuilder.getOrCreateThreadID(Ident);
    Value *SumLocal =
        Builder.CreateUIToFP(TID, Builder.getFloatTy(), "sum.local");
    Value *SumPartial = Builder.CreateLoad(SumType, SumReduced, "sum.partial");
    Value *Sum = Builder.CreateFAdd(SumPartial, SumLocal, "sum");
    Builder.CreateStore(Sum, SumReduced);

    FirstBodyIP = Builder.saveIP();
    FirstBodyAllocaIP = InnerAllocaIP;
  };

  InsertPointTy SecondBodyIP, SecondBodyAllocaIP;
  auto SecondBodyGenCB = [&](InsertPointTy InnerAllocaIP,
                             InsertPointTy CodeGenIP) {
    IRBuilderBase::InsertPointGuard Guard(Builder);
    Builder.restoreIP(CodeGenIP);

    uint32_t StrSize;
    Constant *SrcLocStr = OMPBuilder.getOrCreateSrcLocStr(Loc, StrSize);
    Value *Ident = OMPBuilder.getOrCreateIdent(SrcLocStr, StrSize);
    Value *TID = OMPBuilder.getOrCreateThreadID(Ident);
    Value *XorPartial = Builder.CreateLoad(XorType, XorReduced, "xor.partial");
    Value *Xor = Builder.CreateXor(XorPartial, TID, "xor");
    Builder.CreateStore(Xor, XorReduced);

    SecondBodyIP = Builder.saveIP();
    SecondBodyAllocaIP = InnerAllocaIP;
  };

  // Privatization for reduction creates local copies of reduction variables and
  // initializes them to reduction-neutral values. The same privatization
  // callback is used for both loops, with dispatch based on the value being
  // privatized.
  Value *SumPrivatized;
  Value *XorPrivatized;
  auto PrivCB = [&](InsertPointTy InnerAllocaIP, InsertPointTy CodeGenIP,
                    Value &Original, Value &Inner, Value *&ReplVal) {
    IRBuilderBase::InsertPointGuard Guard(Builder);
    Builder.restoreIP(InnerAllocaIP);
    if (&Original == SumReduced) {
      SumPrivatized = Builder.CreateAlloca(Builder.getFloatTy());
      ReplVal = SumPrivatized;
    } else if (&Original == XorReduced) {
      XorPrivatized = Builder.CreateAlloca(Builder.getInt32Ty());
      ReplVal = XorPrivatized;
    } else {
      ReplVal = &Inner;
      return CodeGenIP;
    }

    Builder.restoreIP(CodeGenIP);
    if (&Original == SumReduced)
      Builder.CreateStore(ConstantFP::get(Builder.getFloatTy(), 0.0),
                          SumPrivatized);
    else if (&Original == XorReduced)
      Builder.CreateStore(Builder.getInt32(0), XorPrivatized);

    return Builder.saveIP();
  };

  // Do nothing in finalization.
  auto FiniCB = [&](InsertPointTy CodeGenIP) { return CodeGenIP; };

  Builder.restoreIP(
      OMPBuilder.createParallel(Loc, OuterAllocaIP, FirstBodyGenCB, PrivCB,
                                FiniCB, /* IfCondition */ nullptr,
                                /* NumThreads */ nullptr, OMP_PROC_BIND_default,
                                /* IsCancellable */ false));
  InsertPointTy AfterIP = OMPBuilder.createParallel(
      {Builder.saveIP(), DL}, OuterAllocaIP, SecondBodyGenCB, PrivCB, FiniCB,
      /* IfCondition */ nullptr,
      /* NumThreads */ nullptr, OMP_PROC_BIND_default,
      /* IsCancellable */ false);

  OMPBuilder.createReductions(
      FirstBodyIP, FirstBodyAllocaIP,
      {{SumType, SumReduced, SumPrivatized, sumReduction, sumAtomicReduction}});
  OMPBuilder.createReductions(
      SecondBodyIP, SecondBodyAllocaIP,
      {{XorType, XorReduced, XorPrivatized, xorReduction, xorAtomicReduction}});

  Builder.restoreIP(AfterIP);
  Builder.CreateRetVoid();

  OMPBuilder.finalize(F);

  // The IR must be valid.
  EXPECT_FALSE(verifyModule(*M));

  // Two different outlined functions must have been created.
  SmallVector<CallInst *> ForkCalls;
  findCalls(F, omp::RuntimeFunction::OMPRTL___kmpc_fork_call, OMPBuilder,
            ForkCalls);
  ASSERT_EQ(ForkCalls.size(), 2u);
  Value *CalleeVal = cast<Constant>(ForkCalls[0]->getOperand(2))->getOperand(0);
  Function *FirstCallee = cast<Function>(CalleeVal);
  CalleeVal = cast<Constant>(ForkCalls[1]->getOperand(2))->getOperand(0);
  Function *SecondCallee = cast<Function>(CalleeVal);
  EXPECT_NE(FirstCallee, SecondCallee);

  // Two different reduction functions must have been created.
  SmallVector<CallInst *> ReduceCalls;
  findCalls(FirstCallee, omp::RuntimeFunction::OMPRTL___kmpc_reduce, OMPBuilder,
            ReduceCalls);
  ASSERT_EQ(ReduceCalls.size(), 1u);
  auto *AddReduction = cast<Function>(ReduceCalls[0]->getOperand(5));
  ReduceCalls.clear();
  findCalls(SecondCallee, omp::RuntimeFunction::OMPRTL___kmpc_reduce,
            OMPBuilder, ReduceCalls);
  auto *XorReduction = cast<Function>(ReduceCalls[0]->getOperand(5));
  EXPECT_NE(AddReduction, XorReduction);

  // Each reduction function does its own kind of reduction.
  BasicBlock *FnReductionBB = &AddReduction->getEntryBlock();
  auto *Bitcast = findSingleUserInBlock<BitCastInst>(AddReduction->getArg(0),
                                                     FnReductionBB);
  ASSERT_NE(Bitcast, nullptr);
  Value *FirstLHSPtr =
      findSingleUserInBlock<GetElementPtrInst>(Bitcast, FnReductionBB);
  ASSERT_NE(FirstLHSPtr, nullptr);
  Value *Opaque = findSingleUserInBlock<LoadInst>(FirstLHSPtr, FnReductionBB);
  ASSERT_NE(Opaque, nullptr);
  Bitcast = findSingleUserInBlock<BitCastInst>(Opaque, FnReductionBB);
  ASSERT_NE(Bitcast, nullptr);
  Instruction::BinaryOps Opcode = Instruction::FAdd;
  EXPECT_TRUE(isSimpleBinaryReduction(Bitcast, FnReductionBB, &Opcode));

  FnReductionBB = &XorReduction->getEntryBlock();
  Bitcast = findSingleUserInBlock<BitCastInst>(XorReduction->getArg(0),
                                               FnReductionBB);
  ASSERT_NE(Bitcast, nullptr);
  Value *SecondLHSPtr =
      findSingleUserInBlock<GetElementPtrInst>(Bitcast, FnReductionBB);
  ASSERT_NE(FirstLHSPtr, nullptr);
  Opaque = findSingleUserInBlock<LoadInst>(SecondLHSPtr, FnReductionBB);
  ASSERT_NE(Opaque, nullptr);
  Bitcast = findSingleUserInBlock<BitCastInst>(Opaque, FnReductionBB);
  ASSERT_NE(Bitcast, nullptr);
  Opcode = Instruction::Xor;
  EXPECT_TRUE(isSimpleBinaryReduction(Bitcast, FnReductionBB, &Opcode));
}

TEST_F(OpenMPIRBuilderTest, CreateSectionsSimple) {
  using InsertPointTy = OpenMPIRBuilder::InsertPointTy;
  using BodyGenCallbackTy = llvm::OpenMPIRBuilder::StorableBodyGenCallbackTy;
  OpenMPIRBuilder OMPBuilder(*M);
  OMPBuilder.initialize();
  F->setName("func");
  IRBuilder<> Builder(BB);

  BasicBlock *EnterBB = BasicBlock::Create(Ctx, "sections.enter", F);
  Builder.CreateBr(EnterBB);
  Builder.SetInsertPoint(EnterBB);
  OpenMPIRBuilder::LocationDescription Loc({Builder.saveIP(), DL});

  llvm::SmallVector<BodyGenCallbackTy, 4> SectionCBVector;
  llvm::SmallVector<BasicBlock *, 4> CaseBBs;

  auto FiniCB = [&](InsertPointTy IP) {};
  auto SectionCB = [&](InsertPointTy AllocaIP, InsertPointTy CodeGenIP) {};
  SectionCBVector.push_back(SectionCB);

  auto PrivCB = [](InsertPointTy AllocaIP, InsertPointTy CodeGenIP,
                   llvm::Value &, llvm::Value &Val,
                   llvm::Value *&ReplVal) { return CodeGenIP; };
  IRBuilder<>::InsertPoint AllocaIP(&F->getEntryBlock(),
                                    F->getEntryBlock().getFirstInsertionPt());
  Builder.restoreIP(OMPBuilder.createSections(Loc, AllocaIP, SectionCBVector,
                                              PrivCB, FiniCB, false, false));
  Builder.CreateRetVoid(); // Required at the end of the function
  EXPECT_NE(F->getEntryBlock().getTerminator(), nullptr);
  EXPECT_FALSE(verifyModule(*M, &errs()));
}

TEST_F(OpenMPIRBuilderTest, CreateSections) {
  using InsertPointTy = OpenMPIRBuilder::InsertPointTy;
  using BodyGenCallbackTy = llvm::OpenMPIRBuilder::StorableBodyGenCallbackTy;
  OpenMPIRBuilder OMPBuilder(*M);
  OMPBuilder.initialize();
  F->setName("func");
  IRBuilder<> Builder(BB);

  OpenMPIRBuilder::LocationDescription Loc({Builder.saveIP(), DL});
  llvm::SmallVector<BodyGenCallbackTy, 4> SectionCBVector;
  llvm::SmallVector<BasicBlock *, 4> CaseBBs;

  BasicBlock *SwitchBB = nullptr;
  AllocaInst *PrivAI = nullptr;
  SwitchInst *Switch = nullptr;

  unsigned NumBodiesGenerated = 0;
  unsigned NumFiniCBCalls = 0;
  PrivAI = Builder.CreateAlloca(F->arg_begin()->getType());

  auto FiniCB = [&](InsertPointTy IP) {
    ++NumFiniCBCalls;
    BasicBlock *IPBB = IP.getBlock();
    EXPECT_NE(IPBB->end(), IP.getPoint());
  };

  auto SectionCB = [&](InsertPointTy AllocaIP, InsertPointTy CodeGenIP) {
    ++NumBodiesGenerated;
    CaseBBs.push_back(CodeGenIP.getBlock());
    SwitchBB = CodeGenIP.getBlock()->getSinglePredecessor();
    Builder.restoreIP(CodeGenIP);
    Builder.CreateStore(F->arg_begin(), PrivAI);
    Value *PrivLoad =
        Builder.CreateLoad(F->arg_begin()->getType(), PrivAI, "local.alloca");
    Builder.CreateICmpNE(F->arg_begin(), PrivLoad);
  };
  auto PrivCB = [](InsertPointTy AllocaIP, InsertPointTy CodeGenIP,
                   llvm::Value &, llvm::Value &Val, llvm::Value *&ReplVal) {
    // TODO: Privatization not implemented yet
    return CodeGenIP;
  };

  SectionCBVector.push_back(SectionCB);
  SectionCBVector.push_back(SectionCB);

  IRBuilder<>::InsertPoint AllocaIP(&F->getEntryBlock(),
                                    F->getEntryBlock().getFirstInsertionPt());
  Builder.restoreIP(OMPBuilder.createSections(Loc, AllocaIP, SectionCBVector,
                                              PrivCB, FiniCB, false, false));
  Builder.CreateRetVoid(); // Required at the end of the function

  // Switch BB's predecessor is loop condition BB, whose successor at index 1 is
  // loop's exit BB
  BasicBlock *ForExitBB =
      SwitchBB->getSinglePredecessor()->getTerminator()->getSuccessor(1);
  EXPECT_NE(ForExitBB, nullptr);

  EXPECT_NE(PrivAI, nullptr);
  Function *OutlinedFn = PrivAI->getFunction();
  EXPECT_EQ(F, OutlinedFn);
  EXPECT_FALSE(verifyModule(*M, &errs()));
  EXPECT_EQ(OutlinedFn->arg_size(), 1U);

  BasicBlock *LoopPreheaderBB =
      OutlinedFn->getEntryBlock().getSingleSuccessor();
  // loop variables are 5 - lower bound, upper bound, stride, islastiter, and
  // iterator/counter
  bool FoundForInit = false;
  for (Instruction &Inst : *LoopPreheaderBB) {
    if (isa<CallInst>(Inst)) {
      if (cast<CallInst>(&Inst)->getCalledFunction()->getName() ==
          "__kmpc_for_static_init_4u") {
        FoundForInit = true;
      }
    }
  }
  EXPECT_EQ(FoundForInit, true);

  bool FoundForExit = false;
  bool FoundBarrier = false;
  for (Instruction &Inst : *ForExitBB) {
    if (isa<CallInst>(Inst)) {
      if (cast<CallInst>(&Inst)->getCalledFunction()->getName() ==
          "__kmpc_for_static_fini") {
        FoundForExit = true;
      }
      if (cast<CallInst>(&Inst)->getCalledFunction()->getName() ==
          "__kmpc_barrier") {
        FoundBarrier = true;
      }
      if (FoundForExit && FoundBarrier)
        break;
    }
  }
  EXPECT_EQ(FoundForExit, true);
  EXPECT_EQ(FoundBarrier, true);

  EXPECT_NE(SwitchBB, nullptr);
  EXPECT_NE(SwitchBB->getTerminator(), nullptr);
  EXPECT_EQ(isa<SwitchInst>(SwitchBB->getTerminator()), true);
  Switch = cast<SwitchInst>(SwitchBB->getTerminator());
  EXPECT_EQ(Switch->getNumCases(), 2U);

  EXPECT_EQ(CaseBBs.size(), 2U);
  for (auto *&CaseBB : CaseBBs) {
    EXPECT_EQ(CaseBB->getParent(), OutlinedFn);
  }

  ASSERT_EQ(NumBodiesGenerated, 2U);
  ASSERT_EQ(NumFiniCBCalls, 1U);
  EXPECT_FALSE(verifyModule(*M, &errs()));
}

TEST_F(OpenMPIRBuilderTest, CreateSectionsNoWait) {
  using InsertPointTy = OpenMPIRBuilder::InsertPointTy;
  using BodyGenCallbackTy = llvm::OpenMPIRBuilder::StorableBodyGenCallbackTy;
  OpenMPIRBuilder OMPBuilder(*M);
  OMPBuilder.initialize();
  F->setName("func");
  IRBuilder<> Builder(BB);

  BasicBlock *EnterBB = BasicBlock::Create(Ctx, "sections.enter", F);
  Builder.CreateBr(EnterBB);
  Builder.SetInsertPoint(EnterBB);
  OpenMPIRBuilder::LocationDescription Loc({Builder.saveIP(), DL});

  IRBuilder<>::InsertPoint AllocaIP(&F->getEntryBlock(),
                                    F->getEntryBlock().getFirstInsertionPt());
  llvm::SmallVector<BodyGenCallbackTy, 4> SectionCBVector;
  auto PrivCB = [](InsertPointTy AllocaIP, InsertPointTy CodeGenIP,
                   llvm::Value &, llvm::Value &Val,
                   llvm::Value *&ReplVal) { return CodeGenIP; };
  auto FiniCB = [&](InsertPointTy IP) {};

  Builder.restoreIP(OMPBuilder.createSections(Loc, AllocaIP, SectionCBVector,
                                              PrivCB, FiniCB, false, true));
  Builder.CreateRetVoid(); // Required at the end of the function
  for (auto &Inst : instructions(*F)) {
    EXPECT_FALSE(isa<CallInst>(Inst) &&
                 cast<CallInst>(&Inst)->getCalledFunction()->getName() ==
                     "__kmpc_barrier" &&
                 "call to function __kmpc_barrier found with nowait");
  }
}

TEST_F(OpenMPIRBuilderTest, CreateOffloadMaptypes) {
  OpenMPIRBuilder OMPBuilder(*M);
  OMPBuilder.initialize();

  IRBuilder<> Builder(BB);

  SmallVector<uint64_t> Mappings = {0, 1};
  GlobalVariable *OffloadMaptypesGlobal =
      OMPBuilder.createOffloadMaptypes(Mappings, "offload_maptypes");
  EXPECT_FALSE(M->global_empty());
  EXPECT_EQ(OffloadMaptypesGlobal->getName(), "offload_maptypes");
  EXPECT_TRUE(OffloadMaptypesGlobal->isConstant());
  EXPECT_TRUE(OffloadMaptypesGlobal->hasGlobalUnnamedAddr());
  EXPECT_TRUE(OffloadMaptypesGlobal->hasPrivateLinkage());
  EXPECT_TRUE(OffloadMaptypesGlobal->hasInitializer());
  Constant *Initializer = OffloadMaptypesGlobal->getInitializer();
  EXPECT_TRUE(isa<ConstantDataArray>(Initializer));
  ConstantDataArray *MappingInit = dyn_cast<ConstantDataArray>(Initializer);
  EXPECT_EQ(MappingInit->getNumElements(), Mappings.size());
  EXPECT_TRUE(MappingInit->getType()->getElementType()->isIntegerTy(64));
  Constant *CA = ConstantDataArray::get(Builder.getContext(), Mappings);
  EXPECT_EQ(MappingInit, CA);
}

TEST_F(OpenMPIRBuilderTest, CreateOffloadMapnames) {
  OpenMPIRBuilder OMPBuilder(*M);
  OMPBuilder.initialize();

  IRBuilder<> Builder(BB);

  uint32_t StrSize;
  Constant *Cst1 =
      OMPBuilder.getOrCreateSrcLocStr("array1", "file1", 2, 5, StrSize);
  Constant *Cst2 =
      OMPBuilder.getOrCreateSrcLocStr("array2", "file1", 3, 5, StrSize);
  SmallVector<llvm::Constant *> Names = {Cst1, Cst2};

  GlobalVariable *OffloadMaptypesGlobal =
      OMPBuilder.createOffloadMapnames(Names, "offload_mapnames");
  EXPECT_FALSE(M->global_empty());
  EXPECT_EQ(OffloadMaptypesGlobal->getName(), "offload_mapnames");
  EXPECT_TRUE(OffloadMaptypesGlobal->isConstant());
  EXPECT_FALSE(OffloadMaptypesGlobal->hasGlobalUnnamedAddr());
  EXPECT_TRUE(OffloadMaptypesGlobal->hasPrivateLinkage());
  EXPECT_TRUE(OffloadMaptypesGlobal->hasInitializer());
  Constant *Initializer = OffloadMaptypesGlobal->getInitializer();
  EXPECT_TRUE(isa<Constant>(Initializer->getOperand(0)->stripPointerCasts()));
  EXPECT_TRUE(isa<Constant>(Initializer->getOperand(1)->stripPointerCasts()));

  GlobalVariable *Name1Gbl =
      cast<GlobalVariable>(Initializer->getOperand(0)->stripPointerCasts());
  EXPECT_TRUE(isa<ConstantDataArray>(Name1Gbl->getInitializer()));
  ConstantDataArray *Name1GblCA =
      dyn_cast<ConstantDataArray>(Name1Gbl->getInitializer());
  EXPECT_EQ(Name1GblCA->getAsCString(), ";file1;array1;2;5;;");

  GlobalVariable *Name2Gbl =
      cast<GlobalVariable>(Initializer->getOperand(1)->stripPointerCasts());
  EXPECT_TRUE(isa<ConstantDataArray>(Name2Gbl->getInitializer()));
  ConstantDataArray *Name2GblCA =
      dyn_cast<ConstantDataArray>(Name2Gbl->getInitializer());
  EXPECT_EQ(Name2GblCA->getAsCString(), ";file1;array2;3;5;;");

  EXPECT_TRUE(Initializer->getType()->getArrayElementType()->isPointerTy());
  EXPECT_EQ(Initializer->getType()->getArrayNumElements(), Names.size());
}

TEST_F(OpenMPIRBuilderTest, CreateMapperAllocas) {
  OpenMPIRBuilder OMPBuilder(*M);
  OMPBuilder.initialize();
  F->setName("func");
  IRBuilder<> Builder(BB);

  OpenMPIRBuilder::LocationDescription Loc({Builder.saveIP(), DL});

  unsigned TotalNbOperand = 2;

  OpenMPIRBuilder::MapperAllocas MapperAllocas;
  IRBuilder<>::InsertPoint AllocaIP(&F->getEntryBlock(),
                                    F->getEntryBlock().getFirstInsertionPt());
  OMPBuilder.createMapperAllocas(Loc, AllocaIP, TotalNbOperand, MapperAllocas);
  EXPECT_NE(MapperAllocas.ArgsBase, nullptr);
  EXPECT_NE(MapperAllocas.Args, nullptr);
  EXPECT_NE(MapperAllocas.ArgSizes, nullptr);
  EXPECT_TRUE(MapperAllocas.ArgsBase->getAllocatedType()->isArrayTy());
  ArrayType *ArrType =
      dyn_cast<ArrayType>(MapperAllocas.ArgsBase->getAllocatedType());
  EXPECT_EQ(ArrType->getNumElements(), TotalNbOperand);
  EXPECT_TRUE(MapperAllocas.ArgsBase->getAllocatedType()
                  ->getArrayElementType()
                  ->isPointerTy());
  EXPECT_TRUE(
      cast<PointerType>(
          MapperAllocas.ArgsBase->getAllocatedType()->getArrayElementType())
          ->isOpaqueOrPointeeTypeMatches(Builder.getInt8Ty()));

  EXPECT_TRUE(MapperAllocas.Args->getAllocatedType()->isArrayTy());
  ArrType = dyn_cast<ArrayType>(MapperAllocas.Args->getAllocatedType());
  EXPECT_EQ(ArrType->getNumElements(), TotalNbOperand);
  EXPECT_TRUE(MapperAllocas.Args->getAllocatedType()
                  ->getArrayElementType()
                  ->isPointerTy());
  EXPECT_TRUE(cast<PointerType>(
                  MapperAllocas.Args->getAllocatedType()->getArrayElementType())
                  ->isOpaqueOrPointeeTypeMatches(Builder.getInt8Ty()));

  EXPECT_TRUE(MapperAllocas.ArgSizes->getAllocatedType()->isArrayTy());
  ArrType = dyn_cast<ArrayType>(MapperAllocas.ArgSizes->getAllocatedType());
  EXPECT_EQ(ArrType->getNumElements(), TotalNbOperand);
  EXPECT_TRUE(MapperAllocas.ArgSizes->getAllocatedType()
                  ->getArrayElementType()
                  ->isIntegerTy(64));
}

TEST_F(OpenMPIRBuilderTest, EmitMapperCall) {
  OpenMPIRBuilder OMPBuilder(*M);
  OMPBuilder.initialize();
  F->setName("func");
  IRBuilder<> Builder(BB);
  LLVMContext &Ctx = M->getContext();

  OpenMPIRBuilder::LocationDescription Loc({Builder.saveIP(), DL});

  unsigned TotalNbOperand = 2;

  OpenMPIRBuilder::MapperAllocas MapperAllocas;
  IRBuilder<>::InsertPoint AllocaIP(&F->getEntryBlock(),
                                    F->getEntryBlock().getFirstInsertionPt());
  OMPBuilder.createMapperAllocas(Loc, AllocaIP, TotalNbOperand, MapperAllocas);

  auto *BeginMapperFunc = OMPBuilder.getOrCreateRuntimeFunctionPtr(
      omp::OMPRTL___tgt_target_data_begin_mapper);

  SmallVector<uint64_t> Flags = {0, 2};

  uint32_t StrSize;
  Constant *SrcLocCst =
      OMPBuilder.getOrCreateSrcLocStr("", "file1", 2, 5, StrSize);
  Value *SrcLocInfo = OMPBuilder.getOrCreateIdent(SrcLocCst, StrSize);

  Constant *Cst1 =
      OMPBuilder.getOrCreateSrcLocStr("array1", "file1", 2, 5, StrSize);
  Constant *Cst2 =
      OMPBuilder.getOrCreateSrcLocStr("array2", "file1", 3, 5, StrSize);
  SmallVector<llvm::Constant *> Names = {Cst1, Cst2};

  GlobalVariable *Maptypes =
      OMPBuilder.createOffloadMaptypes(Flags, ".offload_maptypes");
  Value *MaptypesArg = Builder.CreateConstInBoundsGEP2_32(
      ArrayType::get(Type::getInt64Ty(Ctx), TotalNbOperand), Maptypes,
      /*Idx0=*/0, /*Idx1=*/0);

  GlobalVariable *Mapnames =
      OMPBuilder.createOffloadMapnames(Names, ".offload_mapnames");
  Value *MapnamesArg = Builder.CreateConstInBoundsGEP2_32(
      ArrayType::get(Type::getInt8PtrTy(Ctx), TotalNbOperand), Mapnames,
      /*Idx0=*/0, /*Idx1=*/0);

  OMPBuilder.emitMapperCall(Builder.saveIP(), BeginMapperFunc, SrcLocInfo,
                            MaptypesArg, MapnamesArg, MapperAllocas, -1,
                            TotalNbOperand);

  CallInst *MapperCall = dyn_cast<CallInst>(&BB->back());
  EXPECT_NE(MapperCall, nullptr);
  EXPECT_EQ(MapperCall->arg_size(), 9U);
  EXPECT_EQ(MapperCall->getCalledFunction()->getName(),
            "__tgt_target_data_begin_mapper");
  EXPECT_EQ(MapperCall->getOperand(0), SrcLocInfo);
  EXPECT_TRUE(MapperCall->getOperand(1)->getType()->isIntegerTy(64));
  EXPECT_TRUE(MapperCall->getOperand(2)->getType()->isIntegerTy(32));

  EXPECT_EQ(MapperCall->getOperand(6), MaptypesArg);
  EXPECT_EQ(MapperCall->getOperand(7), MapnamesArg);
  EXPECT_TRUE(MapperCall->getOperand(8)->getType()->isPointerTy());
}

TEST_F(OpenMPIRBuilderTest, CreateTask) {
  using InsertPointTy = OpenMPIRBuilder::InsertPointTy;
  OpenMPIRBuilder OMPBuilder(*M);
  OMPBuilder.initialize();
  F->setName("func");
  IRBuilder<> Builder(BB);

  AllocaInst *ValPtr32 = Builder.CreateAlloca(Builder.getInt32Ty());
  AllocaInst *ValPtr128 = Builder.CreateAlloca(Builder.getInt128Ty());
  Value *Val128 =
      Builder.CreateLoad(Builder.getInt128Ty(), ValPtr128, "bodygen.load");

  auto BodyGenCB = [&](InsertPointTy AllocaIP, InsertPointTy CodeGenIP) {
    Builder.restoreIP(AllocaIP);
    AllocaInst *Local128 = Builder.CreateAlloca(Builder.getInt128Ty(), nullptr,
                                                "bodygen.alloca128");

    Builder.restoreIP(CodeGenIP);
    // Loading and storing captured pointer and values
    Builder.CreateStore(Val128, Local128);
    Value *Val32 = Builder.CreateLoad(ValPtr32->getAllocatedType(), ValPtr32,
                                      "bodygen.load32");

    LoadInst *PrivLoad128 = Builder.CreateLoad(
        Local128->getAllocatedType(), Local128, "bodygen.local.load128");
    Value *Cmp = Builder.CreateICmpNE(
        Val32, Builder.CreateTrunc(PrivLoad128, Val32->getType()));
    Instruction *ThenTerm, *ElseTerm;
    SplitBlockAndInsertIfThenElse(Cmp, CodeGenIP.getBlock()->getTerminator(),
                                  &ThenTerm, &ElseTerm);
  };

  BasicBlock *AllocaBB = Builder.GetInsertBlock();
  BasicBlock *BodyBB = splitBB(Builder, /*CreateBranch=*/true, "alloca.split");
  OpenMPIRBuilder::LocationDescription Loc(
      InsertPointTy(BodyBB, BodyBB->getFirstInsertionPt()), DL);
  Builder.restoreIP(OMPBuilder.createTask(
      Loc, InsertPointTy(AllocaBB, AllocaBB->getFirstInsertionPt()),
      BodyGenCB));
  OMPBuilder.finalize();
  Builder.CreateRetVoid();

  EXPECT_FALSE(verifyModule(*M, &errs()));

  CallInst *TaskAllocCall = dyn_cast<CallInst>(
      OMPBuilder.getOrCreateRuntimeFunctionPtr(OMPRTL___kmpc_omp_task_alloc)
          ->user_back());

  // Verify the Ident argument
  GlobalVariable *Ident = cast<GlobalVariable>(TaskAllocCall->getArgOperand(0));
  ASSERT_NE(Ident, nullptr);
  EXPECT_TRUE(Ident->hasInitializer());
  Constant *Initializer = Ident->getInitializer();
  GlobalVariable *SrcStrGlob =
      cast<GlobalVariable>(Initializer->getOperand(4)->stripPointerCasts());
  ASSERT_NE(SrcStrGlob, nullptr);
  ConstantDataArray *SrcSrc =
      dyn_cast<ConstantDataArray>(SrcStrGlob->getInitializer());
  ASSERT_NE(SrcSrc, nullptr);

  // Verify the num_threads argument.
  CallInst *GTID = dyn_cast<CallInst>(TaskAllocCall->getArgOperand(1));
  ASSERT_NE(GTID, nullptr);
  EXPECT_EQ(GTID->arg_size(), 1U);
  EXPECT_EQ(GTID->getCalledFunction()->getName(), "__kmpc_global_thread_num");

  // Verify the flags
  // TODO: Check for others flags. Currently testing only for tiedness.
  ConstantInt *Flags = dyn_cast<ConstantInt>(TaskAllocCall->getArgOperand(2));
  ASSERT_NE(Flags, nullptr);
  EXPECT_EQ(Flags->getSExtValue(), 1);

  // Verify the data size
  ConstantInt *DataSize =
      dyn_cast<ConstantInt>(TaskAllocCall->getArgOperand(3));
  ASSERT_NE(DataSize, nullptr);
  EXPECT_EQ(DataSize->getSExtValue(), 24); // 64-bit pointer + 128-bit integer

  // TODO: Verify size of shared clause variables

  // Verify Wrapper function
  Function *WrapperFunc =
      dyn_cast<Function>(TaskAllocCall->getArgOperand(5)->stripPointerCasts());
  ASSERT_NE(WrapperFunc, nullptr);
  EXPECT_FALSE(WrapperFunc->isDeclaration());
  CallInst *OutlinedFnCall = dyn_cast<CallInst>(WrapperFunc->begin()->begin());
  ASSERT_NE(OutlinedFnCall, nullptr);
  EXPECT_EQ(WrapperFunc->getArg(0)->getType(), Builder.getInt32Ty());
  EXPECT_EQ(OutlinedFnCall->getArgOperand(0), WrapperFunc->getArg(1));

  // Verify the presence of `trunc` and `icmp` instructions in Outlined function
  Function *OutlinedFn = OutlinedFnCall->getCalledFunction();
  ASSERT_NE(OutlinedFn, nullptr);
  EXPECT_TRUE(any_of(instructions(OutlinedFn),
                     [](Instruction &inst) { return isa<TruncInst>(&inst); }));
  EXPECT_TRUE(any_of(instructions(OutlinedFn),
                     [](Instruction &inst) { return isa<ICmpInst>(&inst); }));

  // Verify the execution of the task
  CallInst *TaskCall = dyn_cast<CallInst>(
      OMPBuilder.getOrCreateRuntimeFunctionPtr(OMPRTL___kmpc_omp_task)
          ->user_back());
  ASSERT_NE(TaskCall, nullptr);
  EXPECT_EQ(TaskCall->getArgOperand(0), Ident);
  EXPECT_EQ(TaskCall->getArgOperand(1), GTID);
  EXPECT_EQ(TaskCall->getArgOperand(2), TaskAllocCall);

  // Verify that the argument data has been copied
  for (User *in : TaskAllocCall->users()) {
    if (MemCpyInst *memCpyInst = dyn_cast<MemCpyInst>(in)) {
      EXPECT_EQ(memCpyInst->getDest(), TaskAllocCall);
    }
  }
}

TEST_F(OpenMPIRBuilderTest, CreateTaskNoArgs) {
  using InsertPointTy = OpenMPIRBuilder::InsertPointTy;
  OpenMPIRBuilder OMPBuilder(*M);
  OMPBuilder.initialize();
  F->setName("func");
  IRBuilder<> Builder(BB);

  auto BodyGenCB = [&](InsertPointTy AllocaIP, InsertPointTy CodeGenIP) {};

  BasicBlock *AllocaBB = Builder.GetInsertBlock();
  BasicBlock *BodyBB = splitBB(Builder, /*CreateBranch=*/true, "alloca.split");
  OpenMPIRBuilder::LocationDescription Loc(
      InsertPointTy(BodyBB, BodyBB->getFirstInsertionPt()), DL);
  Builder.restoreIP(OMPBuilder.createTask(
      Loc, InsertPointTy(AllocaBB, AllocaBB->getFirstInsertionPt()),
      BodyGenCB));
  OMPBuilder.finalize();
  Builder.CreateRetVoid();

  EXPECT_FALSE(verifyModule(*M, &errs()));
}

TEST_F(OpenMPIRBuilderTest, CreateTaskUntied) {
  using InsertPointTy = OpenMPIRBuilder::InsertPointTy;
  OpenMPIRBuilder OMPBuilder(*M);
  OMPBuilder.initialize();
  F->setName("func");
  IRBuilder<> Builder(BB);
  auto BodyGenCB = [&](InsertPointTy AllocaIP, InsertPointTy CodeGenIP) {};
  BasicBlock *AllocaBB = Builder.GetInsertBlock();
  BasicBlock *BodyBB = splitBB(Builder, /*CreateBranch=*/true, "alloca.split");
  OpenMPIRBuilder::LocationDescription Loc(
      InsertPointTy(BodyBB, BodyBB->getFirstInsertionPt()), DL);
  Builder.restoreIP(OMPBuilder.createTask(
      Loc, InsertPointTy(AllocaBB, AllocaBB->getFirstInsertionPt()), BodyGenCB,
      /*Tied=*/false));
  OMPBuilder.finalize();
  Builder.CreateRetVoid();

  // Check for the `Tied` argument
  CallInst *TaskAllocCall = dyn_cast<CallInst>(
      OMPBuilder.getOrCreateRuntimeFunctionPtr(OMPRTL___kmpc_omp_task_alloc)
          ->user_back());
  ASSERT_NE(TaskAllocCall, nullptr);
  ConstantInt *Flags = dyn_cast<ConstantInt>(TaskAllocCall->getArgOperand(2));
  ASSERT_NE(Flags, nullptr);
  EXPECT_EQ(Flags->getZExtValue() & 1U, 0U);

  EXPECT_FALSE(verifyModule(*M, &errs()));
}

TEST_F(OpenMPIRBuilderTest, CreateTaskFinal) {
  using InsertPointTy = OpenMPIRBuilder::InsertPointTy;
  OpenMPIRBuilder OMPBuilder(*M);
  OMPBuilder.initialize();
  F->setName("func");
  IRBuilder<> Builder(BB);
  auto BodyGenCB = [&](InsertPointTy AllocaIP, InsertPointTy CodeGenIP) {};
  IRBuilderBase::InsertPoint AllocaIP = Builder.saveIP();
  BasicBlock *BodyBB = splitBB(Builder, /*CreateBranch=*/true, "alloca.split");
  Builder.SetInsertPoint(BodyBB);
  Value *Final = Builder.CreateICmp(
      CmpInst::Predicate::ICMP_EQ, F->getArg(0),
      ConstantInt::get(Type::getInt32Ty(M->getContext()), 0U));
  OpenMPIRBuilder::LocationDescription Loc(Builder.saveIP(), DL);
  Builder.restoreIP(OMPBuilder.createTask(Loc, AllocaIP, BodyGenCB,
                                          /*Tied=*/false, Final));
  OMPBuilder.finalize();
  Builder.CreateRetVoid();

  // Check for the `Tied` argument
  CallInst *TaskAllocCall = dyn_cast<CallInst>(
      OMPBuilder.getOrCreateRuntimeFunctionPtr(OMPRTL___kmpc_omp_task_alloc)
          ->user_back());
  ASSERT_NE(TaskAllocCall, nullptr);
  BinaryOperator *OrInst =
      dyn_cast<BinaryOperator>(TaskAllocCall->getArgOperand(2));
  ASSERT_NE(OrInst, nullptr);
  EXPECT_EQ(OrInst->getOpcode(), BinaryOperator::BinaryOps::Or);

  // One of the arguments to `or` instruction is the tied flag, which is equal
  // to zero.
  EXPECT_TRUE(any_of(OrInst->operands(), [](Value *op) {
    if (ConstantInt *TiedValue = dyn_cast<ConstantInt>(op))
      return TiedValue->getSExtValue() == 0;
    return false;
  }));

  // One of the arguments to `or` instruction is the final condition.
  EXPECT_TRUE(any_of(OrInst->operands(), [Final](Value *op) {
    if (SelectInst *Select = dyn_cast<SelectInst>(op)) {
      ConstantInt *TrueValue = dyn_cast<ConstantInt>(Select->getTrueValue());
      ConstantInt *FalseValue = dyn_cast<ConstantInt>(Select->getFalseValue());
      if (!TrueValue || !FalseValue)
        return false;
      return Select->getCondition() == Final &&
             TrueValue->getSExtValue() == 2 && FalseValue->getSExtValue() == 0;
    }
    return false;
  }));

  EXPECT_FALSE(verifyModule(*M, &errs()));
}

TEST_F(OpenMPIRBuilderTest, CreateTaskgroup) {
  using InsertPointTy = OpenMPIRBuilder::InsertPointTy;
  OpenMPIRBuilder OMPBuilder(*M);
  OMPBuilder.initialize();
  F->setName("func");
  IRBuilder<> Builder(BB);

  AllocaInst *ValPtr32 = Builder.CreateAlloca(Builder.getInt32Ty());
  AllocaInst *ValPtr128 = Builder.CreateAlloca(Builder.getInt128Ty());
  Value *Val128 =
      Builder.CreateLoad(Builder.getInt128Ty(), ValPtr128, "bodygen.load");
  Instruction *ThenTerm, *ElseTerm;

  Value *InternalStoreInst, *InternalLoad32, *InternalLoad128, *InternalIfCmp;

  auto BodyGenCB = [&](InsertPointTy AllocaIP, InsertPointTy CodeGenIP) {
    Builder.restoreIP(AllocaIP);
    AllocaInst *Local128 = Builder.CreateAlloca(Builder.getInt128Ty(), nullptr,
                                                "bodygen.alloca128");

    Builder.restoreIP(CodeGenIP);
    // Loading and storing captured pointer and values
    InternalStoreInst = Builder.CreateStore(Val128, Local128);
    InternalLoad32 = Builder.CreateLoad(ValPtr32->getAllocatedType(), ValPtr32,
                                        "bodygen.load32");

    InternalLoad128 = Builder.CreateLoad(Local128->getAllocatedType(), Local128,
                                         "bodygen.local.load128");
    InternalIfCmp = Builder.CreateICmpNE(
        InternalLoad32,
        Builder.CreateTrunc(InternalLoad128, InternalLoad32->getType()));
    SplitBlockAndInsertIfThenElse(InternalIfCmp,
                                  CodeGenIP.getBlock()->getTerminator(),
                                  &ThenTerm, &ElseTerm);
  };

  BasicBlock *AllocaBB = Builder.GetInsertBlock();
  BasicBlock *BodyBB = splitBB(Builder, /*CreateBranch=*/true, "alloca.split");
  OpenMPIRBuilder::LocationDescription Loc(
      InsertPointTy(BodyBB, BodyBB->getFirstInsertionPt()), DL);
  Builder.restoreIP(OMPBuilder.createTaskgroup(
      Loc, InsertPointTy(AllocaBB, AllocaBB->getFirstInsertionPt()),
      BodyGenCB));
  OMPBuilder.finalize();
  Builder.CreateRetVoid();

  EXPECT_FALSE(verifyModule(*M, &errs()));

  CallInst *TaskgroupCall = dyn_cast<CallInst>(
      OMPBuilder.getOrCreateRuntimeFunctionPtr(OMPRTL___kmpc_taskgroup)
          ->user_back());
  ASSERT_NE(TaskgroupCall, nullptr);
  CallInst *EndTaskgroupCall = dyn_cast<CallInst>(
      OMPBuilder.getOrCreateRuntimeFunctionPtr(OMPRTL___kmpc_end_taskgroup)
          ->user_back());
  ASSERT_NE(EndTaskgroupCall, nullptr);

  // Verify the Ident argument
  GlobalVariable *Ident = cast<GlobalVariable>(TaskgroupCall->getArgOperand(0));
  ASSERT_NE(Ident, nullptr);
  EXPECT_TRUE(Ident->hasInitializer());
  Constant *Initializer = Ident->getInitializer();
  GlobalVariable *SrcStrGlob =
      cast<GlobalVariable>(Initializer->getOperand(4)->stripPointerCasts());
  ASSERT_NE(SrcStrGlob, nullptr);
  ConstantDataArray *SrcSrc =
      dyn_cast<ConstantDataArray>(SrcStrGlob->getInitializer());
  ASSERT_NE(SrcSrc, nullptr);

  // Verify the num_threads argument.
  CallInst *GTID = dyn_cast<CallInst>(TaskgroupCall->getArgOperand(1));
  ASSERT_NE(GTID, nullptr);
  EXPECT_EQ(GTID->arg_size(), 1U);
  EXPECT_EQ(GTID->getCalledFunction(), OMPBuilder.getOrCreateRuntimeFunctionPtr(
                                           OMPRTL___kmpc_global_thread_num));

  // Checking the general structure of the IR generated is same as expected.
  Instruction *GeneratedStoreInst = TaskgroupCall->getNextNonDebugInstruction();
  EXPECT_EQ(GeneratedStoreInst, InternalStoreInst);
  Instruction *GeneratedLoad32 =
      GeneratedStoreInst->getNextNonDebugInstruction();
  EXPECT_EQ(GeneratedLoad32, InternalLoad32);
  Instruction *GeneratedLoad128 = GeneratedLoad32->getNextNonDebugInstruction();
  EXPECT_EQ(GeneratedLoad128, InternalLoad128);

  // Checking the ordering because of the if statements and that
  // `__kmp_end_taskgroup` call is after the if branching.
  BasicBlock *RefOrder[] = {TaskgroupCall->getParent(), ThenTerm->getParent(),
                            ThenTerm->getSuccessor(0),
                            EndTaskgroupCall->getParent(),
                            ElseTerm->getParent()};
  verifyDFSOrder(F, RefOrder);
}

TEST_F(OpenMPIRBuilderTest, CreateTaskgroupWithTasks) {
  using InsertPointTy = OpenMPIRBuilder::InsertPointTy;
  OpenMPIRBuilder OMPBuilder(*M);
  OMPBuilder.initialize();
  F->setName("func");
  IRBuilder<> Builder(BB);

  auto BodyGenCB = [&](InsertPointTy AllocaIP, InsertPointTy CodeGenIP) {
    Builder.restoreIP(AllocaIP);
    AllocaInst *Alloca32 =
        Builder.CreateAlloca(Builder.getInt32Ty(), nullptr, "bodygen.alloca32");
    AllocaInst *Alloca64 =
        Builder.CreateAlloca(Builder.getInt64Ty(), nullptr, "bodygen.alloca64");
    Builder.restoreIP(CodeGenIP);
    auto TaskBodyGenCB1 = [&](InsertPointTy AllocaIP, InsertPointTy CodeGenIP) {
      Builder.restoreIP(CodeGenIP);
      LoadInst *LoadValue =
          Builder.CreateLoad(Alloca64->getAllocatedType(), Alloca64);
      Value *AddInst = Builder.CreateAdd(LoadValue, Builder.getInt64(64));
      Builder.CreateStore(AddInst, Alloca64);
    };
    OpenMPIRBuilder::LocationDescription Loc(Builder.saveIP(), DL);
    Builder.restoreIP(OMPBuilder.createTask(Loc, AllocaIP, TaskBodyGenCB1));

    auto TaskBodyGenCB2 = [&](InsertPointTy AllocaIP, InsertPointTy CodeGenIP) {
      Builder.restoreIP(CodeGenIP);
      LoadInst *LoadValue =
          Builder.CreateLoad(Alloca32->getAllocatedType(), Alloca32);
      Value *AddInst = Builder.CreateAdd(LoadValue, Builder.getInt32(32));
      Builder.CreateStore(AddInst, Alloca32);
    };
    OpenMPIRBuilder::LocationDescription Loc2(Builder.saveIP(), DL);
    Builder.restoreIP(OMPBuilder.createTask(Loc2, AllocaIP, TaskBodyGenCB2));
  };

  BasicBlock *AllocaBB = Builder.GetInsertBlock();
  BasicBlock *BodyBB = splitBB(Builder, /*CreateBranch=*/true, "alloca.split");
  OpenMPIRBuilder::LocationDescription Loc(
      InsertPointTy(BodyBB, BodyBB->getFirstInsertionPt()), DL);
  Builder.restoreIP(OMPBuilder.createTaskgroup(
      Loc, InsertPointTy(AllocaBB, AllocaBB->getFirstInsertionPt()),
      BodyGenCB));
  OMPBuilder.finalize();
  Builder.CreateRetVoid();

  EXPECT_FALSE(verifyModule(*M, &errs()));

  CallInst *TaskgroupCall = dyn_cast<CallInst>(
      OMPBuilder.getOrCreateRuntimeFunctionPtr(OMPRTL___kmpc_taskgroup)
          ->user_back());
  ASSERT_NE(TaskgroupCall, nullptr);
  CallInst *EndTaskgroupCall = dyn_cast<CallInst>(
      OMPBuilder.getOrCreateRuntimeFunctionPtr(OMPRTL___kmpc_end_taskgroup)
          ->user_back());
  ASSERT_NE(EndTaskgroupCall, nullptr);

  Function *TaskAllocFn =
      OMPBuilder.getOrCreateRuntimeFunctionPtr(OMPRTL___kmpc_omp_task_alloc);
  ASSERT_EQ(TaskAllocFn->getNumUses(), 2u);

  CallInst *FirstTaskAllocCall =
      dyn_cast_or_null<CallInst>(*TaskAllocFn->users().begin());
  CallInst *SecondTaskAllocCall =
      dyn_cast_or_null<CallInst>(*TaskAllocFn->users().begin()++);
  ASSERT_NE(FirstTaskAllocCall, nullptr);
  ASSERT_NE(SecondTaskAllocCall, nullptr);

  // Verify that the tasks have been generated in order and inside taskgroup
  // construct.
  BasicBlock *RefOrder[] = {
      TaskgroupCall->getParent(), FirstTaskAllocCall->getParent(),
      SecondTaskAllocCall->getParent(), EndTaskgroupCall->getParent()};
  verifyDFSOrder(F, RefOrder);
}

} // namespace
