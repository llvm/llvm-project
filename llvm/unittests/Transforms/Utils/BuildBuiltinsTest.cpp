//===- BuildBuiltinsTest.cpp - Unit tests for BasicBlockUtils -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/BuildBuiltins.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

static void
followBackwardsLookForWrites(Value *Ptr, BasicBlock *StartFromBB,
                             BasicBlock::reverse_iterator StartFromIt,
                             DenseSet<BasicBlock *> &Visited,
                             SmallVectorImpl<Instruction *> &WriteAccs) {
  for (auto &&I : make_range(StartFromIt, StartFromBB->rend())) {
    if (!I.mayHaveSideEffects())
      continue;
    if (isa<LoadInst>(I))
      continue;

    if (auto *SI = dyn_cast<StoreInst>(&I)) {
      if (SI->getPointerOperand() == Ptr) {
        WriteAccs.push_back(SI);
        return;
      }
      continue;
    }
    if (auto *CmpXchg = dyn_cast<AtomicCmpXchgInst>(&I)) {
      if (CmpXchg->getPointerOperand() == Ptr) {
        WriteAccs.push_back(CmpXchg);
        return;
      }
      continue;
    }

    if (auto *ARMW = dyn_cast<AtomicRMWInst>(&I)) {
      if (ARMW->getPointerOperand() == Ptr) {
        WriteAccs.push_back(ARMW);
        return;
      }
      continue;
    }

    if (auto *CI = dyn_cast<CallInst>(&I)) {
      MemoryEffects ME = CI->getMemoryEffects();

      if (isModSet(ME.getModRef(IRMemLocation::Other))) {
        WriteAccs.push_back(CI);
        return;
      }

      if (isModSet(ME.getModRef(IRMemLocation::ArgMem))) {
        for (auto &&Ops : CI->args()) {
          if (Ops.get() == Ptr) {
            WriteAccs.push_back(CI);
            return;
          }
        }
      }
      continue;
    }

    llvm_unreachable("TODO: Can accs this ptr?");
  }

  Visited.insert(StartFromBB);
  for (BasicBlock *Pred : predecessors(StartFromBB)) {
    if (Visited.contains(Pred))
      continue;

    followBackwardsLookForWrites(Ptr, Pred, Pred->rbegin(), Visited, WriteAccs);
  }
};

static Instruction *getUniquePreviousStore(Value *Ptr, BasicBlock *FromBB) {
  SmallVector<Instruction *, 1> WriteAccs;
  DenseSet<BasicBlock *> Visited;
  followBackwardsLookForWrites(Ptr, FromBB, FromBB->rbegin(), Visited,
                               WriteAccs);
  if (WriteAccs.size() == 1)
    return WriteAccs.front();
  return nullptr;
}

class BuildBuiltinsTests : public testing::Test {
protected:
  LLVMContext Ctx;
  std::unique_ptr<Module> M;
  DataLayout DL;
  std::unique_ptr<TargetLibraryInfoImpl> TLII;
  std::unique_ptr<TargetLibraryInfo> TLI;
  Function *F = nullptr;
  Argument *PtrArg = nullptr;
  Argument *RetArg = nullptr;
  Argument *ExpectedArg = nullptr;
  Argument *DesiredArg = nullptr;

  Argument *ValArg = nullptr;
  Argument *PredArg = nullptr;
  Argument *MemorderArg = nullptr;
  Argument *FailMemorderArg = nullptr;

  BasicBlock *EntryBB = nullptr;
  IRBuilder<> Builder;

  BuildBuiltinsTests() : Builder(Ctx) {}

  void SetUp() override {
    M.reset(new Module("TestModule", Ctx));
    DL = M->getDataLayout();

    Triple T(M->getTargetTriple());
    TLII.reset(new TargetLibraryInfoImpl(T));
    TLI.reset(new TargetLibraryInfo(*TLII));

    FunctionType *FTy =
        FunctionType::get(Type::getVoidTy(Ctx),
                          {PointerType::get(Ctx, 0), PointerType::get(Ctx, 0),
                           PointerType::get(Ctx, 0), PointerType::get(Ctx, 0),
                           Type::getInt32Ty(Ctx), Type::getInt1Ty(Ctx),
                           Type::getInt32Ty(Ctx), Type::getInt32Ty(Ctx)},
                          /*isVarArg=*/false);
    F = Function::Create(FTy, Function::ExternalLinkage, "TestFunction",
                         M.get());
    PtrArg = F->getArg(0);
    PtrArg->setName("atomic_ptr");
    RetArg = F->getArg(1);
    RetArg->setName("ret_ptr");

    ExpectedArg = F->getArg(2);
    ExpectedArg->setName("expected_ptr");
    DesiredArg = F->getArg(3);
    DesiredArg->setName("desired_ptr");

    ValArg = F->getArg(4);
    ValArg->setName("valarg");
    PredArg = F->getArg(5);
    PredArg->setName("predarg");

    MemorderArg = F->getArg(6);
    MemorderArg->setName("memorderarg_success");
    FailMemorderArg = F->getArg(7);
    FailMemorderArg->setName("memorderarg_failure");

    EntryBB = BasicBlock::Create(Ctx, "entry", F);
    Builder.SetInsertPoint(EntryBB);
    ReturnInst *RetInst = Builder.CreateRetVoid();
    Builder.SetInsertPoint(RetInst);
  }

  void TearDown() override {
    EntryBB = nullptr;
    F = nullptr;
    M.reset();
  }
};

TEST_F(BuildBuiltinsTests, AtomicLoad) {
  ASSERT_THAT_ERROR(emitAtomicLoadBuiltin(
                        /*AtomicPtr=*/PtrArg, /*RetPtr=*/RetArg,
                        /*TypeOrSize=*/Builder.getInt32Ty(),
                        /*IsVolatile=*/false,
                        /*Memorder=*/AtomicOrdering::SequentiallyConsistent,
                        /*Scope=*/SyncScope::System,
                        /*Align=*/{},
                        /*Builder=*/Builder,
                        /*EmitOptions=*/AtomicEmitOptions(DL, TLI.get()),
                        /*Name=*/"atomic_load"),
                    Succeeded());
  EXPECT_FALSE(verifyModule(*M, &errs()));

  // clang-format off
  // %atomic_load.atomic.load = load atomic i32, ptr %atomic_ptr seq_cst, align 1
  // store i32 %atomic_load.atomic.load, ptr %ret_ptr, align 4
  // clang-format on

  // Follow use-def and load-store chains to discover instructions
  StoreInst *Store = cast<StoreInst>(getUniquePreviousStore(RetArg, EntryBB));
  LoadInst *AtomicLoadAtomicLoad = cast<LoadInst>(Store->getValueOperand());

  // %atomic_load.atomic.load = load atomic i32, ptr %atomic_ptr seq_cst, a...
  EXPECT_EQ(AtomicLoadAtomicLoad->getName(), "atomic_load.atomic.load");
  EXPECT_EQ(AtomicLoadAtomicLoad->getParent(), EntryBB);
  EXPECT_EQ(AtomicLoadAtomicLoad->getType(), Type::getInt32Ty(Ctx));
  EXPECT_FALSE(AtomicLoadAtomicLoad->isVolatile());
  EXPECT_EQ(AtomicLoadAtomicLoad->getAlign(), 1);
  EXPECT_EQ(AtomicLoadAtomicLoad->getOrdering(),
            AtomicOrdering::SequentiallyConsistent);
  EXPECT_EQ(AtomicLoadAtomicLoad->getSyncScopeID(), SyncScope::System);
  EXPECT_EQ(AtomicLoadAtomicLoad->getPointerOperand(), PtrArg);

  // store i32 %atomic_load.atomic.load, ptr %ret_ptr, align 4
  EXPECT_TRUE(Store->getName().empty());
  EXPECT_EQ(Store->getParent(), EntryBB);
  EXPECT_TRUE(Store->isSimple());
  EXPECT_EQ(Store->getValueOperand(), AtomicLoadAtomicLoad);
  EXPECT_EQ(Store->getPointerOperand(), RetArg);
}

TEST_F(BuildBuiltinsTests, AtomicLoad_SizedLibcall) {
  AtomicEmitOptions EO(DL, TLI.get());
  EO.AllowInstruction = false;

  ASSERT_THAT_ERROR(emitAtomicLoadBuiltin(
                        /*AtomicPtr=*/PtrArg, /*RetPtr=*/RetArg,
                        /*TypeOrSize=*/Builder.getInt32Ty(),
                        /*IsVolatile=*/false,
                        /*Memorder=*/AtomicOrdering::SequentiallyConsistent,
                        /*Scope=*/SyncScope::System,
                        /*Align=*/{},
                        /*Builder=*/Builder,
                        /*EmitOptions=*/EO,
                        /*Name=*/"atomic_load"),
                    Succeeded());
  EXPECT_FALSE(verifyModule(*M, &errs()));

  // Follow use-def and load-store chains to discover instructions
  CallInst *Call = cast<CallInst>(getUniquePreviousStore(PtrArg, EntryBB));

  // call void @__atomic_load(i64 4, ptr %atomic_ptr, ptr %ret_ptr, i32 5)
  EXPECT_TRUE(Call->getName().empty());
  EXPECT_EQ(Call->getParent(), EntryBB);
  EXPECT_EQ(Call->getType(), Type::getVoidTy(Ctx));
  EXPECT_EQ(Call->getName(), "");
  EXPECT_FALSE(Call->isMustTailCall());
  EXPECT_FALSE(Call->isTailCall());
  EXPECT_EQ(Call->getCallingConv(), CallingConv::C);
  EXPECT_EQ(cast<ConstantInt>(Call->getArgOperand(0))->getZExtValue(), 4);
  EXPECT_EQ(Call->getArgOperand(1), PtrArg);
  EXPECT_EQ(Call->getArgOperand(2), RetArg);
  EXPECT_EQ(cast<ConstantInt>(Call->getArgOperand(3))->getZExtValue(), 5);
  EXPECT_EQ(Call->getCalledFunction(), M->getFunction("__atomic_load"));
}

TEST_F(BuildBuiltinsTests, AtomicLoad_Libcall) {
  AtomicEmitOptions EO(DL, TLI.get());
  EO.AllowInstruction = false;
  EO.AllowSizedLibcall = false;
  ASSERT_THAT_ERROR(emitAtomicLoadBuiltin(
                        /*AtomicPtr=*/PtrArg, /*RetPtr=*/RetArg,
                        /*TypeOrSize=*/Builder.getInt32Ty(),
                        /*IsVolatile=*/false,
                        /*Memorder=*/AtomicOrdering::SequentiallyConsistent,
                        /*Scope=*/SyncScope::System,
                        /*Align=*/{},
                        /*Builder=*/Builder,
                        /*EmitOptions=*/EO,
                        /*Name=*/"atomic_load"),
                    Succeeded());
  EXPECT_FALSE(verifyModule(*M, &errs()));

  // Follow use-def and load-store chains to discover instructions
  CallInst *Call = cast<CallInst>(getUniquePreviousStore(PtrArg, EntryBB));

  // call void @__atomic_load(i64 4, ptr %atomic_ptr, ptr %ret_ptr, i32 5)
  EXPECT_TRUE(Call->getName().empty());
  EXPECT_EQ(Call->getParent(), EntryBB);
  EXPECT_EQ(Call->getType(), Type::getVoidTy(Ctx));
  EXPECT_EQ(Call->getName(), "");
  EXPECT_FALSE(Call->isMustTailCall());
  EXPECT_FALSE(Call->isTailCall());
  EXPECT_EQ(Call->getCallingConv(), CallingConv::C);
  EXPECT_EQ(cast<ConstantInt>(Call->getArgOperand(0))->getZExtValue(), 4);
  EXPECT_EQ(Call->getArgOperand(1), PtrArg);
  EXPECT_EQ(Call->getArgOperand(2), RetArg);
  EXPECT_EQ(cast<ConstantInt>(Call->getArgOperand(3))->getZExtValue(), 5);
  EXPECT_EQ(Call->getCalledFunction(), M->getFunction("__atomic_load"));
}

TEST_F(BuildBuiltinsTests, AtomicLoad_Volatile) {
  ASSERT_THAT_ERROR(emitAtomicLoadBuiltin(
                        /*AtomicPtr=*/PtrArg, /*RetPtr=*/RetArg,
                        /*TypeOrSize=*/Builder.getInt32Ty(),
                        /*IsVolatile=*/true,
                        /*Memorder=*/AtomicOrdering::SequentiallyConsistent,
                        /*Scope=*/SyncScope::System,
                        /*Align=*/{},
                        /*Builder=*/Builder,
                        /*EmitOptions=*/AtomicEmitOptions(DL, TLI.get()),
                        /*Name=*/"atomic_load"),
                    Succeeded());
  EXPECT_FALSE(verifyModule(*M, &errs()));

  // clang-format off
  // %atomic_load.atomic.load = load atomic volatile i32, ptr %atomic_ptr seq_cst, align 1
  // store volatile i32 %atomic_load.atomic.load, ptr %ret_ptr, align 4
  // clang-format on

  // Follow use-def and load-store chains to discover instructions
  StoreInst *Store = cast<StoreInst>(getUniquePreviousStore(RetArg, EntryBB));
  LoadInst *AtomicLoadAtomicLoad = cast<LoadInst>(Store->getValueOperand());

  // %atomic_load.atomic.load = load atomic volatile i32, ptr %atomic_ptr s...
  EXPECT_EQ(AtomicLoadAtomicLoad->getName(), "atomic_load.atomic.load");
  EXPECT_EQ(AtomicLoadAtomicLoad->getParent(), EntryBB);
  EXPECT_EQ(AtomicLoadAtomicLoad->getType(), Type::getInt32Ty(Ctx));
  EXPECT_TRUE(AtomicLoadAtomicLoad->isVolatile());
  EXPECT_EQ(AtomicLoadAtomicLoad->getAlign(), 1);
  EXPECT_EQ(AtomicLoadAtomicLoad->getOrdering(),
            AtomicOrdering::SequentiallyConsistent);
  EXPECT_EQ(AtomicLoadAtomicLoad->getSyncScopeID(), SyncScope::System);
  EXPECT_EQ(AtomicLoadAtomicLoad->getPointerOperand(), PtrArg);

  // store volatile i32 %atomic_load.atomic.load, ptr %ret_ptr, align 4
  EXPECT_TRUE(Store->getName().empty());
  EXPECT_EQ(Store->getParent(), EntryBB);
  EXPECT_TRUE(Store->isVolatile());
  EXPECT_EQ(Store->getAlign(), 4);
  EXPECT_EQ(Store->getOrdering(), AtomicOrdering::NotAtomic);
  EXPECT_EQ(Store->getSyncScopeID(), SyncScope::System);
  EXPECT_EQ(Store->getValueOperand(), AtomicLoadAtomicLoad);
  EXPECT_EQ(Store->getPointerOperand(), RetArg);
}

TEST_F(BuildBuiltinsTests, AtomicLoad_Memorder) {
  ASSERT_THAT_ERROR(emitAtomicLoadBuiltin(
                        /*AtomicPtr=*/PtrArg, /*RetPtr=*/RetArg,
                        /*TypeOrSize=*/Builder.getInt32Ty(),
                        /*IsVolatile=*/false,
                        /*Memorder=*/AtomicOrdering::Monotonic,
                        /*Scope=*/SyncScope::System,
                        /*Align=*/{},
                        /*Builder=*/Builder,
                        /*EmitOptions=*/AtomicEmitOptions(DL, TLI.get()),
                        /*Name=*/"atomic_load"),
                    Succeeded());
  EXPECT_FALSE(verifyModule(*M, &errs()));

  // clang-format off
  // %atomic_load.atomic.load = load atomic i32, ptr %atomic_ptr monotonic, align 1
  // store i32 %atomic_load.atomic.load, ptr %ret_ptr, align 4
  // clang-format on

  // Follow use-def and load-store chains to discover instructions
  StoreInst *Store = cast<StoreInst>(getUniquePreviousStore(RetArg, EntryBB));
  LoadInst *AtomicLoadAtomicLoad = cast<LoadInst>(Store->getValueOperand());

  // %atomic_load.atomic.load = load atomic i32, ptr %atomic_ptr monotonic,...
  EXPECT_EQ(AtomicLoadAtomicLoad->getName(), "atomic_load.atomic.load");
  EXPECT_EQ(AtomicLoadAtomicLoad->getParent(), EntryBB);
  EXPECT_EQ(AtomicLoadAtomicLoad->getType(), Type::getInt32Ty(Ctx));
  EXPECT_FALSE(AtomicLoadAtomicLoad->isVolatile());
  EXPECT_EQ(AtomicLoadAtomicLoad->getAlign(), 1);
  EXPECT_EQ(AtomicLoadAtomicLoad->getOrdering(), AtomicOrdering::Monotonic);
  EXPECT_EQ(AtomicLoadAtomicLoad->getSyncScopeID(), SyncScope::System);
  EXPECT_EQ(AtomicLoadAtomicLoad->getPointerOperand(), PtrArg);

  // store i32 %atomic_load.atomic.load, ptr %ret_ptr, align 4
  EXPECT_TRUE(Store->getName().empty());
  EXPECT_EQ(Store->getParent(), EntryBB);
  EXPECT_TRUE(Store->isSimple());
  EXPECT_EQ(Store->getValueOperand(), AtomicLoadAtomicLoad);
  EXPECT_EQ(Store->getPointerOperand(), RetArg);
}

TEST_F(BuildBuiltinsTests, AtomicLoad_Memorder_CABI) {
  ASSERT_THAT_ERROR(emitAtomicLoadBuiltin(
                        /*AtomicPtr=*/PtrArg, /*RetPtr=*/RetArg,
                        /*TypeOrSize=*/Builder.getInt32Ty(),
                        /*IsVolatile=*/false,
                        /*Memorder=*/AtomicOrderingCABI::relaxed,
                        /*Scope=*/SyncScope::System,
                        /*Align=*/{},
                        /*Builder=*/Builder,
                        /*EmitOptions=*/AtomicEmitOptions(DL, TLI.get()),
                        /*Name=*/"atomic_load"),
                    Succeeded());
  EXPECT_FALSE(verifyModule(*M, &errs()));

  // clang-format off
  // %atomic_load.atomic.load = load atomic i32, ptr %atomic_ptr monotonic, align 1
  // store i32 %atomic_load.atomic.load, ptr %ret_ptr, align 4
  // clang-format on

  // Follow use-def and load-store chains to discover instructions
  StoreInst *Store = cast<StoreInst>(getUniquePreviousStore(RetArg, EntryBB));
  LoadInst *AtomicLoadAtomicLoad = cast<LoadInst>(Store->getValueOperand());

  // %atomic_load.atomic.load = load atomic i32, ptr %atomic_ptr monotonic,...
  EXPECT_EQ(AtomicLoadAtomicLoad->getName(), "atomic_load.atomic.load");
  EXPECT_EQ(AtomicLoadAtomicLoad->getParent(), EntryBB);
  EXPECT_EQ(AtomicLoadAtomicLoad->getType(), Type::getInt32Ty(Ctx));
  EXPECT_FALSE(AtomicLoadAtomicLoad->isVolatile());
  EXPECT_EQ(AtomicLoadAtomicLoad->getAlign(), 1);
  EXPECT_EQ(AtomicLoadAtomicLoad->getOrdering(), AtomicOrdering::Monotonic);
  EXPECT_EQ(AtomicLoadAtomicLoad->getSyncScopeID(), SyncScope::System);
  EXPECT_EQ(AtomicLoadAtomicLoad->getPointerOperand(), PtrArg);

  // store i32 %atomic_load.atomic.load, ptr %ret_ptr, align 4
  EXPECT_TRUE(Store->getName().empty());
  EXPECT_EQ(Store->getParent(), EntryBB);
  EXPECT_TRUE(Store->isSimple());
  EXPECT_EQ(Store->getValueOperand(), AtomicLoadAtomicLoad);
  EXPECT_EQ(Store->getPointerOperand(), RetArg);
}

TEST_F(BuildBuiltinsTests, AtomicLoad_Memorder_Switch) {
  ASSERT_THAT_ERROR(emitAtomicLoadBuiltin(
                        /*AtomicPtr=*/PtrArg, /*RetPtr=*/RetArg,
                        /*TypeOrSize=*/Builder.getInt32Ty(),
                        /*IsVolatile=*/false,
                        /*Memorder=*/MemorderArg,
                        /*Scope=*/SyncScope::System,
                        /*Align=*/{},
                        /*Builder=*/Builder,
                        /*EmitOptions=*/AtomicEmitOptions(DL, TLI.get()),
                        /*Name=*/"atomic_load"),
                    Succeeded());
  EXPECT_FALSE(verifyModule(*M, &errs()));
  BasicBlock *ExitBB = Builder.GetInsertBlock();

  // clang-format off
  // entry:
  //   switch i32 %memorderarg_success, label %atomic_load.atomic.monotonic [
  //     i32 1, label %atomic_load.atomic.acquire
  //     i32 2, label %atomic_load.atomic.acquire
  //     i32 5, label %atomic_load.atomic.seqcst
  //   ]
  //
  // atomic_load.atomic.monotonic:                     ; preds = %entry
  //   %atomic_load.atomic.load = load atomic i32, ptr %atomic_ptr monotonic, align 1
  //   store i32 %atomic_load.atomic.load, ptr %ret_ptr, align 4
  //   br label %atomic_load.atomic.memorder.continue
  //
  // atomic_load.atomic.acquire:                       ; preds = %entry, %entry
  //   %atomic_load.atomic.load1 = load atomic i32, ptr %atomic_ptr acquire, align 1
  //   store i32 %atomic_load.atomic.load1, ptr %ret_ptr, align 4
  //   br label %atomic_load.atomic.memorder.continue
  //
  // atomic_load.atomic.seqcst:                        ; preds = %entry
  //   %atomic_load.atomic.load2 = load atomic i32, ptr %atomic_ptr seq_cst, align 1
  //   store i32 %atomic_load.atomic.load2, ptr %ret_ptr, align 4
  //   br label %atomic_load.atomic.memorder.continue
  //
  // atomic_load.atomic.memorder.continue:             ; preds = %atomic_load.atomic.seqcst, %atomic_load.atomic.acquire, %atomic_load.atomic.monotonic
  //   %atomic_load.atomic.memorder.success = phi i1 [ true, %atomic_load.atomic.monotonic ], [ true, %atomic_load.atomic.acquire ], [ true, %atomic_load.atomic.seqcst ]
  //   ret void
  // clang-format on

  // Discover control flow graph
  SwitchInst *Switch = cast<SwitchInst>(EntryBB->getTerminator());
  BasicBlock *AtomicLoadAtomicAcquire =
      cast<BasicBlock>(Switch->getSuccessor(1));
  BasicBlock *AtomicLoadAtomicSeqcst =
      cast<BasicBlock>(Switch->getSuccessor(3));
  BasicBlock *AtomicLoadAtomicMonotonic =
      cast<BasicBlock>(Switch->getDefaultDest());
  BranchInst *Branch1 =
      cast<BranchInst>(AtomicLoadAtomicMonotonic->getTerminator());
  BranchInst *Branch2 =
      cast<BranchInst>(AtomicLoadAtomicAcquire->getTerminator());
  BranchInst *Branch3 =
      cast<BranchInst>(AtomicLoadAtomicSeqcst->getTerminator());
  ReturnInst *Return = cast<ReturnInst>(ExitBB->getTerminator());

  // Follow use-def and load-store chains to discover instructions
  StoreInst *Store1 =
      cast<StoreInst>(getUniquePreviousStore(RetArg, AtomicLoadAtomicSeqcst));
  LoadInst *AtomicLoadAtomicLoad2 = cast<LoadInst>(Store1->getValueOperand());
  StoreInst *Store2 = cast<StoreInst>(
      getUniquePreviousStore(RetArg, AtomicLoadAtomicMonotonic));
  LoadInst *AtomicLoadAtomicLoad = cast<LoadInst>(Store2->getValueOperand());
  StoreInst *Store3 =
      cast<StoreInst>(getUniquePreviousStore(RetArg, AtomicLoadAtomicAcquire));
  LoadInst *AtomicLoadAtomicLoad1 = cast<LoadInst>(Store3->getValueOperand());

  // switch i32 %memorderarg_success, label %atomic_load.atomic.monotonic [
  //   i32 1, label %atomic_load.atomic.acquire
  //   i32 2, label %atomic_load.atomic.acquire
  //   i32 5, label %atomic_load.atomic.seqcst
  // ]
  EXPECT_TRUE(Switch->getName().empty());
  EXPECT_EQ(Switch->getParent(), EntryBB);
  EXPECT_EQ(Switch->getType(), Type::getVoidTy(Ctx));
  EXPECT_EQ(Switch->getCondition(), MemorderArg);
  EXPECT_EQ(Switch->getDefaultDest(), AtomicLoadAtomicMonotonic);
  EXPECT_EQ(cast<ConstantInt>(Switch->getOperand(2))->getZExtValue(), 1);
  EXPECT_EQ(Switch->getOperand(3), AtomicLoadAtomicAcquire);
  EXPECT_EQ(cast<ConstantInt>(Switch->getOperand(4))->getZExtValue(), 2);
  EXPECT_EQ(Switch->getOperand(5), AtomicLoadAtomicAcquire);
  EXPECT_EQ(cast<ConstantInt>(Switch->getOperand(6))->getZExtValue(), 5);
  EXPECT_EQ(Switch->getOperand(7), AtomicLoadAtomicSeqcst);

  // %atomic_load.atomic.load = load atomic i32, ptr %atomic_ptr monotonic,...
  EXPECT_EQ(AtomicLoadAtomicLoad->getName(), "atomic_load.atomic.load");
  EXPECT_EQ(AtomicLoadAtomicLoad->getParent(), AtomicLoadAtomicMonotonic);
  EXPECT_EQ(AtomicLoadAtomicLoad->getType(), Type::getInt32Ty(Ctx));
  EXPECT_FALSE(AtomicLoadAtomicLoad->isVolatile());
  EXPECT_EQ(AtomicLoadAtomicLoad->getAlign(), 1);
  EXPECT_EQ(AtomicLoadAtomicLoad->getOrdering(), AtomicOrdering::Monotonic);
  EXPECT_EQ(AtomicLoadAtomicLoad->getSyncScopeID(), SyncScope::System);
  EXPECT_EQ(AtomicLoadAtomicLoad->getPointerOperand(), PtrArg);

  // store i32 %atomic_load.atomic.load, ptr %ret_ptr, align 4
  EXPECT_TRUE(Store2->getName().empty());
  EXPECT_EQ(Store2->getParent(), AtomicLoadAtomicMonotonic);
  EXPECT_TRUE(Store2->isSimple());
  EXPECT_EQ(Store2->getValueOperand(), AtomicLoadAtomicLoad);
  EXPECT_EQ(Store2->getPointerOperand(), RetArg);

  // br label %atomic_load.atomic.memorder.continue
  EXPECT_TRUE(Branch1->getName().empty());
  EXPECT_EQ(Branch1->getParent(), AtomicLoadAtomicMonotonic);
  EXPECT_EQ(Branch1->getType(), Type::getVoidTy(Ctx));
  EXPECT_TRUE(Branch1->isUnconditional());
  EXPECT_EQ(Branch1->getOperand(0), ExitBB);

  // %atomic_load.atomic.load1 = load atomic i32, ptr %atomic_ptr acquire, ...
  EXPECT_EQ(AtomicLoadAtomicLoad1->getName(), "atomic_load.atomic.load1");
  EXPECT_EQ(AtomicLoadAtomicLoad1->getParent(), AtomicLoadAtomicAcquire);
  EXPECT_EQ(AtomicLoadAtomicLoad1->getType(), Type::getInt32Ty(Ctx));
  EXPECT_FALSE(AtomicLoadAtomicLoad1->isVolatile());
  EXPECT_EQ(AtomicLoadAtomicLoad1->getAlign(), 1);
  EXPECT_EQ(AtomicLoadAtomicLoad1->getOrdering(), AtomicOrdering::Acquire);
  EXPECT_EQ(AtomicLoadAtomicLoad1->getSyncScopeID(), SyncScope::System);
  EXPECT_EQ(AtomicLoadAtomicLoad1->getPointerOperand(), PtrArg);

  // store i32 %atomic_load.atomic.load1, ptr %ret_ptr, align 4
  EXPECT_TRUE(Store3->getName().empty());
  EXPECT_EQ(Store3->getParent(), AtomicLoadAtomicAcquire);
  EXPECT_TRUE(Store3->isSimple());
  EXPECT_EQ(Store3->getValueOperand(), AtomicLoadAtomicLoad1);
  EXPECT_EQ(Store3->getPointerOperand(), RetArg);

  // br label %atomic_load.atomic.memorder.continue
  EXPECT_TRUE(Branch2->getName().empty());
  EXPECT_EQ(Branch2->getParent(), AtomicLoadAtomicAcquire);
  EXPECT_EQ(Branch2->getType(), Type::getVoidTy(Ctx));
  EXPECT_TRUE(Branch2->isUnconditional());
  EXPECT_EQ(Branch2->getOperand(0), ExitBB);

  // %atomic_load.atomic.load2 = load atomic i32, ptr %atomic_ptr seq_cst, ...
  EXPECT_EQ(AtomicLoadAtomicLoad2->getName(), "atomic_load.atomic.load2");
  EXPECT_EQ(AtomicLoadAtomicLoad2->getParent(), AtomicLoadAtomicSeqcst);
  EXPECT_EQ(AtomicLoadAtomicLoad2->getType(), Type::getInt32Ty(Ctx));
  EXPECT_FALSE(AtomicLoadAtomicLoad2->isVolatile());
  EXPECT_EQ(AtomicLoadAtomicLoad2->getAlign(), 1);
  EXPECT_EQ(AtomicLoadAtomicLoad2->getOrdering(),
            AtomicOrdering::SequentiallyConsistent);
  EXPECT_EQ(AtomicLoadAtomicLoad2->getSyncScopeID(), SyncScope::System);
  EXPECT_EQ(AtomicLoadAtomicLoad2->getPointerOperand(), PtrArg);

  // store i32 %atomic_load.atomic.load2, ptr %ret_ptr, align 4
  EXPECT_TRUE(Store1->getName().empty());
  EXPECT_EQ(Store1->getParent(), AtomicLoadAtomicSeqcst);
  EXPECT_TRUE(Store1->isSimple());
  EXPECT_EQ(Store1->getValueOperand(), AtomicLoadAtomicLoad2);
  EXPECT_EQ(Store1->getPointerOperand(), RetArg);

  // br label %atomic_load.atomic.memorder.continue
  EXPECT_TRUE(Branch3->getName().empty());
  EXPECT_EQ(Branch3->getParent(), AtomicLoadAtomicSeqcst);
  EXPECT_EQ(Branch3->getType(), Type::getVoidTy(Ctx));
  EXPECT_TRUE(Branch3->isUnconditional());
  EXPECT_EQ(Branch3->getOperand(0), ExitBB);

  // ret void
  EXPECT_TRUE(Return->getName().empty());
  EXPECT_EQ(Return->getParent(), ExitBB);
  EXPECT_EQ(Return->getType(), Type::getVoidTy(Ctx));
}

TEST_F(BuildBuiltinsTests, AtomicLoad_SyncScope) {
  ASSERT_THAT_ERROR(emitAtomicLoadBuiltin(
                        /*AtomicPtr=*/PtrArg, /*RetPtr=*/RetArg,
                        /*TypeOrSize=*/Builder.getInt32Ty(),
                        /*IsVolatile=*/false,
                        /*Memorder=*/AtomicOrdering::SequentiallyConsistent,
                        /*Scope=*/SyncScope::SingleThread,
                        /*Align=*/{},
                        /*Builder=*/Builder,
                        /*EmitOptions=*/AtomicEmitOptions(DL, TLI.get()),
                        /*Name=*/"atomic_load"),
                    Succeeded());
  EXPECT_FALSE(verifyModule(*M, &errs()));

  // clang-format off
  // %atomic_load.atomic.load = load atomic i32, ptr %atomic_ptr syncscope("singlethread") seq_cst, align 1
  // store i32 %atomic_load.atomic.load, ptr %ret_ptr, align 4
  // clang-format on

  // Follow use-def and load-store chains to discover instructions
  StoreInst *Store = cast<StoreInst>(getUniquePreviousStore(RetArg, EntryBB));
  LoadInst *AtomicLoadAtomicLoad = cast<LoadInst>(Store->getValueOperand());

  // %atomic_load.atomic.load = load atomic i32, ptr %atomic_ptr syncscope(...
  EXPECT_EQ(AtomicLoadAtomicLoad->getName(), "atomic_load.atomic.load");
  EXPECT_EQ(AtomicLoadAtomicLoad->getParent(), EntryBB);
  EXPECT_EQ(AtomicLoadAtomicLoad->getType(), Type::getInt32Ty(Ctx));
  EXPECT_FALSE(AtomicLoadAtomicLoad->isVolatile());
  EXPECT_EQ(AtomicLoadAtomicLoad->getAlign(), 1);
  EXPECT_EQ(AtomicLoadAtomicLoad->getOrdering(),
            AtomicOrdering::SequentiallyConsistent);
  EXPECT_EQ(AtomicLoadAtomicLoad->getSyncScopeID(), SyncScope::SingleThread);
  EXPECT_EQ(AtomicLoadAtomicLoad->getPointerOperand(), PtrArg);

  // store i32 %atomic_load.atomic.load, ptr %ret_ptr, align 4
  EXPECT_TRUE(Store->getName().empty());
  EXPECT_EQ(Store->getParent(), EntryBB);
  EXPECT_TRUE(Store->isSimple());
  EXPECT_EQ(Store->getValueOperand(), AtomicLoadAtomicLoad);
  EXPECT_EQ(Store->getPointerOperand(), RetArg);
}

TEST_F(BuildBuiltinsTests, AtomicLoad_Float) {
  ASSERT_THAT_ERROR(emitAtomicLoadBuiltin(
                        /*AtomicPtr=*/PtrArg, /*RetPtr=*/RetArg,
                        /*TypeOrSize=*/Builder.getFloatTy(),
                        /*IsVolatile=*/false,
                        /*Memorder=*/AtomicOrdering::SequentiallyConsistent,
                        /*Scope=*/SyncScope::System,
                        /*Align=*/{},
                        /*Builder=*/Builder,
                        /*EmitOptions=*/AtomicEmitOptions(DL, TLI.get()),
                        /*Name=*/"atomic_load"),
                    Succeeded());
  EXPECT_FALSE(verifyModule(*M, &errs()));

  // clang-format off
  // %atomic_load.atomic.load = load atomic float, ptr %atomic_ptr seq_cst, align 1
  // store float %atomic_load.atomic.load, ptr %ret_ptr, align 4
  // clang-format on

  // Follow use-def and load-store chains to discover instructions
  StoreInst *Store = cast<StoreInst>(getUniquePreviousStore(RetArg, EntryBB));
  LoadInst *AtomicLoadAtomicLoad = cast<LoadInst>(Store->getValueOperand());

  // %atomic_load.atomic.load = load atomic float, ptr %atomic_ptr seq_cst,...
  EXPECT_EQ(AtomicLoadAtomicLoad->getName(), "atomic_load.atomic.load");
  EXPECT_EQ(AtomicLoadAtomicLoad->getParent(), EntryBB);
  EXPECT_EQ(AtomicLoadAtomicLoad->getType(), Type::getFloatTy(Ctx));
  EXPECT_FALSE(AtomicLoadAtomicLoad->isVolatile());
  EXPECT_EQ(AtomicLoadAtomicLoad->getAlign(), 1);
  EXPECT_EQ(AtomicLoadAtomicLoad->getOrdering(),
            AtomicOrdering::SequentiallyConsistent);
  EXPECT_EQ(AtomicLoadAtomicLoad->getSyncScopeID(), SyncScope::System);
  EXPECT_EQ(AtomicLoadAtomicLoad->getPointerOperand(), PtrArg);

  // store float %atomic_load.atomic.load, ptr %ret_ptr, align 4
  EXPECT_TRUE(Store->getName().empty());
  EXPECT_EQ(Store->getParent(), EntryBB);
  EXPECT_TRUE(Store->isSimple());
  EXPECT_EQ(Store->getValueOperand(), AtomicLoadAtomicLoad);
  EXPECT_EQ(Store->getPointerOperand(), RetArg);
}

TEST_F(BuildBuiltinsTests, AtomicLoad_FP80) {
  ASSERT_THAT_ERROR(emitAtomicLoadBuiltin(
                        /*AtomicPtr=*/PtrArg, /*RetPtr=*/RetArg,
                        /*TypeOrSize=*/Type::getX86_FP80Ty(Ctx),
                        /*IsVolatile=*/false,
                        /*Memorder=*/AtomicOrdering::SequentiallyConsistent,
                        /*Scope=*/SyncScope::System,
                        /*Align=*/{},
                        /*Builder=*/Builder,
                        /*EmitOptions=*/AtomicEmitOptions(DL, TLI.get()),
                        /*Name=*/"atomic_load"),
                    Succeeded());
  EXPECT_FALSE(verifyModule(*M, &errs()));

  // Follow use-def and load-store chains to discover instructions
  CallInst *Call = cast<CallInst>(getUniquePreviousStore(PtrArg, EntryBB));

  // call void @__atomic_load(i64 10, ptr %atomic_ptr, ptr %ret_ptr, i32 5)
  EXPECT_TRUE(Call->getName().empty());
  EXPECT_EQ(Call->getParent(), EntryBB);
  EXPECT_EQ(Call->getType(), Type::getVoidTy(Ctx));
  EXPECT_EQ(Call->getName(), "");
  EXPECT_FALSE(Call->isMustTailCall());
  EXPECT_FALSE(Call->isTailCall());
  EXPECT_EQ(Call->getCallingConv(), CallingConv::C);
  EXPECT_EQ(cast<ConstantInt>(Call->getArgOperand(0))->getZExtValue(), 10);
  EXPECT_EQ(Call->getArgOperand(1), PtrArg);
  EXPECT_EQ(Call->getArgOperand(2), RetArg);
  EXPECT_EQ(cast<ConstantInt>(Call->getArgOperand(3))->getZExtValue(), 5);
  EXPECT_EQ(Call->getCalledFunction(), M->getFunction("__atomic_load"));
}

TEST_F(BuildBuiltinsTests, AtomicLoad_Ptr) {
  ASSERT_THAT_ERROR(emitAtomicLoadBuiltin(
                        /*AtomicPtr=*/PtrArg, /*RetPtr=*/RetArg,
                        /*TypeOrSize=*/Builder.getPtrTy(), /*IsVolatile=*/false,
                        /*Memorder=*/AtomicOrdering::SequentiallyConsistent,
                        /*Scope=*/SyncScope::System,
                        /*Align=*/{},
                        /*Builder=*/Builder,
                        /*EmitOptions=*/AtomicEmitOptions(DL, TLI.get()),
                        /*Name=*/"atomic_load"),
                    Succeeded());
  EXPECT_FALSE(verifyModule(*M, &errs()));

  // clang-format off
  // %atomic_load.atomic.load = load atomic ptr, ptr %atomic_ptr seq_cst, align 1
  // store ptr %atomic_load.atomic.load, ptr %ret_ptr, align 8
  // clang-format on

  // Follow use-def and load-store chains to discover instructions
  StoreInst *Store = cast<StoreInst>(getUniquePreviousStore(RetArg, EntryBB));
  LoadInst *AtomicLoadAtomicLoad = cast<LoadInst>(Store->getValueOperand());

  // %atomic_load.atomic.load = load atomic ptr, ptr %atomic_ptr seq_cst, a...
  EXPECT_EQ(AtomicLoadAtomicLoad->getName(), "atomic_load.atomic.load");
  EXPECT_EQ(AtomicLoadAtomicLoad->getParent(), EntryBB);
  EXPECT_EQ(AtomicLoadAtomicLoad->getType(), PointerType::get(Ctx, 0));
  EXPECT_FALSE(AtomicLoadAtomicLoad->isVolatile());
  EXPECT_EQ(AtomicLoadAtomicLoad->getAlign(), 1);
  EXPECT_EQ(AtomicLoadAtomicLoad->getOrdering(),
            AtomicOrdering::SequentiallyConsistent);
  EXPECT_EQ(AtomicLoadAtomicLoad->getSyncScopeID(), SyncScope::System);
  EXPECT_EQ(AtomicLoadAtomicLoad->getPointerOperand(), PtrArg);

  // store ptr %atomic_load.atomic.load, ptr %ret_ptr, align 8
  EXPECT_TRUE(Store->getName().empty());
  EXPECT_EQ(Store->getParent(), EntryBB);
  EXPECT_TRUE(Store->isSimple());
  EXPECT_EQ(Store->getValueOperand(), AtomicLoadAtomicLoad);
  EXPECT_EQ(Store->getPointerOperand(), RetArg);
}

TEST_F(BuildBuiltinsTests, AtomicLoad_Struct) {
  // A struct that is small enough to be covered with a single instruction
  StructType *STy =
      StructType::get(Ctx, {Builder.getFloatTy(), Builder.getFloatTy()});

  ASSERT_THAT_ERROR(emitAtomicLoadBuiltin(
                        /*AtomicPtr=*/PtrArg, /*RetPtr=*/RetArg,
                        /*TypeOrSize=*/STy,
                        /*IsVolatile=*/false,
                        /*Memorder=*/AtomicOrdering::SequentiallyConsistent,
                        /*Scope=*/SyncScope::System,
                        /*Align=*/{},
                        /*Builder=*/Builder,
                        /*EmitOptions=*/AtomicEmitOptions(DL, TLI.get()),
                        /*Name=*/"atomic_load"),
                    Succeeded());
  EXPECT_FALSE(verifyModule(*M, &errs()));

  // clang-format off
  // %atomic_load.atomic.load = load atomic i64, ptr %atomic_ptr seq_cst, align 1
  // store i64 %atomic_load.atomic.load, ptr %ret_ptr, align 4
  // clang-format on

  // Follow use-def and load-store chains to discover instructions
  StoreInst *Store = cast<StoreInst>(getUniquePreviousStore(RetArg, EntryBB));
  LoadInst *AtomicLoadAtomicLoad = cast<LoadInst>(Store->getValueOperand());

  // %atomic_load.atomic.load = load atomic i64, ptr %atomic_ptr seq_cst, a...
  EXPECT_EQ(AtomicLoadAtomicLoad->getName(), "atomic_load.atomic.load");
  EXPECT_EQ(AtomicLoadAtomicLoad->getParent(), EntryBB);
  EXPECT_EQ(AtomicLoadAtomicLoad->getType(), Type::getInt64Ty(Ctx));
  EXPECT_FALSE(AtomicLoadAtomicLoad->isVolatile());
  EXPECT_EQ(AtomicLoadAtomicLoad->getAlign(), 1);
  EXPECT_EQ(AtomicLoadAtomicLoad->getOrdering(),
            AtomicOrdering::SequentiallyConsistent);
  EXPECT_EQ(AtomicLoadAtomicLoad->getSyncScopeID(), SyncScope::System);
  EXPECT_EQ(AtomicLoadAtomicLoad->getPointerOperand(), PtrArg);

  // store i64 %atomic_load.atomic.load, ptr %ret_ptr, align 4
  EXPECT_TRUE(Store->getName().empty());
  EXPECT_EQ(Store->getParent(), EntryBB);
  EXPECT_TRUE(Store->isSimple());
  EXPECT_EQ(Store->getValueOperand(), AtomicLoadAtomicLoad);
  EXPECT_EQ(Store->getPointerOperand(), RetArg);
}

TEST_F(BuildBuiltinsTests, AtomicLoad_Array) {
  // A type that is too large for atomic instructions
  ArrayType *ATy = ArrayType::get(Builder.getFloatTy(), 19);

  ASSERT_THAT_ERROR(emitAtomicLoadBuiltin(
                        /*AtomicPtr=*/PtrArg, /*RetPtr=*/RetArg,
                        /*TypeOrSize=*/ATy,
                        /*IsVolatile=*/false,
                        /*Memorder=*/AtomicOrdering::SequentiallyConsistent,
                        /*Scope=*/SyncScope::System,
                        /*Align=*/{},
                        /*Builder=*/Builder,
                        /*EmitOptions=*/AtomicEmitOptions(DL, TLI.get()),
                        /*Name=*/"atomic_load"),
                    Succeeded());
  EXPECT_FALSE(verifyModule(*M, &errs()));

  // Follow use-def and load-store chains to discover instructions
  CallInst *Call = cast<CallInst>(getUniquePreviousStore(PtrArg, EntryBB));

  // call void @__atomic_load(i64 76, ptr %atomic_ptr, ptr %ret_ptr, i32 5)
  EXPECT_TRUE(Call->getName().empty());
  EXPECT_EQ(Call->getParent(), EntryBB);
  EXPECT_EQ(Call->getType(), Type::getVoidTy(Ctx));
  EXPECT_EQ(Call->getName(), "");
  EXPECT_FALSE(Call->isMustTailCall());
  EXPECT_FALSE(Call->isTailCall());
  EXPECT_EQ(Call->getCallingConv(), CallingConv::C);
  EXPECT_EQ(cast<ConstantInt>(Call->getArgOperand(0))->getZExtValue(), 76);
  EXPECT_EQ(Call->getArgOperand(1), PtrArg);
  EXPECT_EQ(Call->getArgOperand(2), RetArg);
  EXPECT_EQ(cast<ConstantInt>(Call->getArgOperand(3))->getZExtValue(), 5);
  EXPECT_EQ(Call->getCalledFunction(), M->getFunction("__atomic_load"));
}

TEST_F(BuildBuiltinsTests, AtomicLoad_Array_NoLibatomic) {
  // Use a triple that does not support libatomic (according to
  // initializeLibCalls in TargetLibraryInfo.cpp)
  Triple T("x86_64-scei-ps4");
  TLII.reset(new TargetLibraryInfoImpl(T));
  TLI.reset(new TargetLibraryInfo(*TLII));

  // A type that is too large for atomic instructions
  ArrayType *ATy = ArrayType::get(Builder.getFloatTy(), 19);

  ASSERT_THAT_ERROR(
      emitAtomicLoadBuiltin(
          /*AtomicPtr=*/PtrArg, /*RetPtr=*/RetArg, /*TypeOrSize=*/ATy,
          /*IsVolatile=*/false,
          /*Memorder=*/AtomicOrdering::SequentiallyConsistent,
          /*Scope=*/SyncScope::System,
          /*Align=*/{},
          /*Builder=*/Builder,
          /*EmitOptions=*/AtomicEmitOptions(DL, TLI.get()),
          /*Name=*/"atomic_load"),
      FailedWithMessage(
          "__atomic_load builtin not supported by any available means"));
}

TEST_F(BuildBuiltinsTests, AtomicLoad_DataSize) {
  ASSERT_THAT_ERROR(emitAtomicLoadBuiltin(
                        /*AtomicPtr=*/PtrArg, /*RetPtr=*/RetArg,
                        /*TypeOrSize=*/static_cast<uint64_t>(6),
                        /*IsVolatile=*/false,
                        /*Memorder=*/AtomicOrdering::SequentiallyConsistent,
                        /*Scope=*/SyncScope::System, /*Align=*/{},
                        /*Builder=*/Builder,
                        /*EmitOptions=*/AtomicEmitOptions(DL, TLI.get()),
                        /*Name=*/"atomic_load"),
                    Succeeded());
  EXPECT_FALSE(verifyModule(*M, &errs()));

  // Follow use-def and load-store chains to discover instructions
  CallInst *Call = cast<CallInst>(getUniquePreviousStore(PtrArg, EntryBB));

  // call void @__atomic_load(i64 6, ptr %atomic_ptr, ptr %ret_ptr, i32 5)
  EXPECT_TRUE(Call->getName().empty());
  EXPECT_EQ(Call->getParent(), EntryBB);
  EXPECT_EQ(Call->getType(), Type::getVoidTy(Ctx));
  EXPECT_EQ(Call->getName(), "");
  EXPECT_FALSE(Call->isMustTailCall());
  EXPECT_FALSE(Call->isTailCall());
  EXPECT_EQ(Call->getCallingConv(), CallingConv::C);
  EXPECT_EQ(cast<ConstantInt>(Call->getArgOperand(0))->getZExtValue(), 6);
  EXPECT_EQ(Call->getArgOperand(1), PtrArg);
  EXPECT_EQ(Call->getArgOperand(2), RetArg);
  EXPECT_EQ(cast<ConstantInt>(Call->getArgOperand(3))->getZExtValue(), 5);
  EXPECT_EQ(Call->getCalledFunction(), M->getFunction("__atomic_load"));
}

TEST_F(BuildBuiltinsTests, AtomicLoad_Align) {
  ASSERT_THAT_ERROR(emitAtomicLoadBuiltin(
                        /*AtomicPtr=*/PtrArg, /*RetPtr=*/RetArg,
                        /*TypeOrSize=*/Builder.getInt32Ty(),
                        /*IsVolatile=*/false,
                        /*Memorder=*/AtomicOrdering::SequentiallyConsistent,
                        /*Scope=*/SyncScope::System,
                        /*Align=*/Align(8),
                        /*Builder=*/Builder,
                        /*EmitOptions=*/AtomicEmitOptions(DL, TLI.get()),
                        /*Name=*/"atomic_load"),
                    Succeeded());
  EXPECT_FALSE(verifyModule(*M, &errs()));

  // clang-format off
  // %atomic_load.atomic.load = load atomic i32, ptr %atomic_ptr seq_cst, align 8
  // store i32 %atomic_load.atomic.load, ptr %ret_ptr, align 4
  // clang-format on

  // Follow use-def and load-store chains to discover instructions
  StoreInst *Store = cast<StoreInst>(getUniquePreviousStore(RetArg, EntryBB));
  LoadInst *AtomicLoadAtomicLoad = cast<LoadInst>(Store->getValueOperand());

  // %atomic_load.atomic.load = load atomic i32, ptr %atomic_ptr seq_cst, a...
  EXPECT_EQ(AtomicLoadAtomicLoad->getName(), "atomic_load.atomic.load");
  EXPECT_EQ(AtomicLoadAtomicLoad->getParent(), EntryBB);
  EXPECT_EQ(AtomicLoadAtomicLoad->getType(), Type::getInt32Ty(Ctx));
  EXPECT_FALSE(AtomicLoadAtomicLoad->isVolatile());
  EXPECT_EQ(AtomicLoadAtomicLoad->getAlign(), 8);
  EXPECT_EQ(AtomicLoadAtomicLoad->getOrdering(),
            AtomicOrdering::SequentiallyConsistent);
  EXPECT_EQ(AtomicLoadAtomicLoad->getSyncScopeID(), SyncScope::System);
  EXPECT_EQ(AtomicLoadAtomicLoad->getPointerOperand(), PtrArg);

  // store i32 %atomic_load.atomic.load, ptr %ret_ptr, align 4
  EXPECT_TRUE(Store->getName().empty());
  EXPECT_EQ(Store->getParent(), EntryBB);
  EXPECT_TRUE(Store->isSimple());
  EXPECT_EQ(Store->getValueOperand(), AtomicLoadAtomicLoad);
  EXPECT_EQ(Store->getPointerOperand(), RetArg);
}

TEST_F(BuildBuiltinsTests, AtomicStore) {
  ASSERT_THAT_ERROR(emitAtomicStoreBuiltin(
                        /*AtomicPtr=*/PtrArg, /*RetPtr=*/RetArg,
                        /*TypeOrSize=*/Builder.getInt32Ty(),
                        /*IsVolatile=*/false,
                        /*Memorder=*/AtomicOrdering::SequentiallyConsistent,
                        /*Scope=*/SyncScope::System,
                        /*Align=*/{},
                        /*Builder=*/Builder,
                        /*EmitOptions=*/AtomicEmitOptions(DL, TLI.get()),
                        /*Name=*/"atomic_store"),
                    Succeeded());
  EXPECT_FALSE(verifyModule(*M, &errs()));

  // clang-format off
  // %atomic_store.atomic.val = load i32, ptr %ret_ptr, align 4
  // store atomic i32 %atomic_store.atomic.val, ptr %atomic_ptr seq_cst, align 1
  // clang-format on

  // Follow use-def and load-store chains to discover instructions
  StoreInst *Store = cast<StoreInst>(getUniquePreviousStore(PtrArg, EntryBB));
  LoadInst *AtomicStoreAtomicVal = cast<LoadInst>(Store->getValueOperand());

  // %atomic_store.atomic.val = load i32, ptr %ret_ptr, align 4
  EXPECT_EQ(AtomicStoreAtomicVal->getName(), "atomic_store.atomic.val");
  EXPECT_EQ(AtomicStoreAtomicVal->getParent(), EntryBB);
  EXPECT_EQ(AtomicStoreAtomicVal->getType(), Type::getInt32Ty(Ctx));
  EXPECT_TRUE(AtomicStoreAtomicVal->isSimple());
  EXPECT_EQ(AtomicStoreAtomicVal->getPointerOperand(), RetArg);

  // store atomic i32 %atomic_store.atomic.val, ptr %atomic_ptr seq_cst, al...
  EXPECT_TRUE(Store->getName().empty());
  EXPECT_EQ(Store->getParent(), EntryBB);
  EXPECT_FALSE(Store->isVolatile());
  EXPECT_EQ(Store->getAlign(), 1);
  EXPECT_EQ(Store->getOrdering(), AtomicOrdering::SequentiallyConsistent);
  EXPECT_EQ(Store->getSyncScopeID(), SyncScope::System);
  EXPECT_EQ(Store->getValueOperand(), AtomicStoreAtomicVal);
  EXPECT_EQ(Store->getPointerOperand(), PtrArg);
}

TEST_F(BuildBuiltinsTests, AtomicStore_SizedLibcall) {
  AtomicEmitOptions EO(DL, TLI.get());
  EO.AllowInstruction = false;
  ASSERT_THAT_ERROR(emitAtomicStoreBuiltin(
                        /*AtomicPtr=*/PtrArg, /*RetPtr=*/RetArg,
                        /*TypeOrSize=*/Builder.getInt32Ty(),
                        /*IsVolatile=*/false,
                        /*Memorder=*/AtomicOrdering::SequentiallyConsistent,
                        /*Scope=*/SyncScope::System,
                        /*Align=*/{},
                        /*Builder=*/Builder,
                        /*EmitOptions=*/EO,
                        /*Name=*/"atomic_store"),
                    Succeeded());
  EXPECT_FALSE(verifyModule(*M, &errs()));

  // Follow use-def and load-store chains to discover instructions
  CallInst *Call = cast<CallInst>(getUniquePreviousStore(PtrArg, EntryBB));

  // call void @__atomic_store(i64 4, ptr %atomic_ptr, ptr %ret_ptr, i32 5)
  EXPECT_TRUE(Call->getName().empty());
  EXPECT_EQ(Call->getParent(), EntryBB);
  EXPECT_EQ(Call->getType(), Type::getVoidTy(Ctx));
  EXPECT_EQ(Call->getName(), "");
  EXPECT_FALSE(Call->isMustTailCall());
  EXPECT_FALSE(Call->isTailCall());
  EXPECT_EQ(Call->getCallingConv(), CallingConv::C);
  EXPECT_EQ(cast<ConstantInt>(Call->getArgOperand(0))->getZExtValue(), 4);
  EXPECT_EQ(Call->getArgOperand(1), PtrArg);
  EXPECT_EQ(Call->getArgOperand(2), RetArg);
  EXPECT_EQ(cast<ConstantInt>(Call->getArgOperand(3))->getZExtValue(), 5);
  EXPECT_EQ(Call->getCalledFunction(), M->getFunction("__atomic_store"));
}

TEST_F(BuildBuiltinsTests, AtomicStore_Libcall) {
  AtomicEmitOptions EO(DL, TLI.get());
  EO.AllowInstruction = false;
  EO.AllowSizedLibcall = false;
  ASSERT_THAT_ERROR(emitAtomicStoreBuiltin(
                        /*AtomicPtr=*/PtrArg, /*RetPtr=*/RetArg,
                        /*TypeOrSize=*/Builder.getInt32Ty(),
                        /*IsVolatile=*/false,
                        /*Memorder=*/AtomicOrdering::SequentiallyConsistent,
                        /*Scope=*/SyncScope::System,
                        /*Align=*/{},
                        /*Builder=*/Builder,
                        /*EmitOptions=*/EO,
                        /*Name=*/"atomic_store"),
                    Succeeded());
  EXPECT_FALSE(verifyModule(*M, &errs()));

  // Follow use-def and load-store chains to discover instructions
  CallInst *Call = cast<CallInst>(getUniquePreviousStore(PtrArg, EntryBB));

  // call void @__atomic_store(i64 4, ptr %atomic_ptr, ptr %ret_ptr, i32 5)
  EXPECT_TRUE(Call->getName().empty());
  EXPECT_EQ(Call->getParent(), EntryBB);
  EXPECT_EQ(Call->getType(), Type::getVoidTy(Ctx));
  EXPECT_EQ(Call->getName(), "");
  EXPECT_FALSE(Call->isMustTailCall());
  EXPECT_FALSE(Call->isTailCall());
  EXPECT_EQ(Call->getCallingConv(), CallingConv::C);
  EXPECT_EQ(cast<ConstantInt>(Call->getArgOperand(0))->getZExtValue(), 4);
  EXPECT_EQ(Call->getArgOperand(1), PtrArg);
  EXPECT_EQ(Call->getArgOperand(2), RetArg);
  EXPECT_EQ(cast<ConstantInt>(Call->getArgOperand(3))->getZExtValue(), 5);
  EXPECT_EQ(Call->getCalledFunction(), M->getFunction("__atomic_store"));
}

TEST_F(BuildBuiltinsTests, AtomicStore_Volatile) {
  ASSERT_THAT_ERROR(emitAtomicStoreBuiltin(
                        /*AtomicPtr=*/PtrArg, /*RetPtr=*/RetArg,
                        /*TypeOrSize=*/Builder.getInt32Ty(),
                        /*IsVolatile=*/true,
                        /*Memorder=*/AtomicOrdering::SequentiallyConsistent,
                        /*Scope=*/SyncScope::System,
                        /*Align=*/{},
                        /*Builder=*/Builder,
                        /*EmitOptions=*/AtomicEmitOptions(DL, TLI.get()),
                        /*Name=*/"atomic_store"),
                    Succeeded());
  EXPECT_FALSE(verifyModule(*M, &errs()));

  // clang-format off
  // %atomic_store.atomic.val = load i32, ptr %ret_ptr, align 4
  // store atomic volatile i32 %atomic_store.atomic.val, ptr %atomic_ptr seq_cst, align 1
  // clang-format on

  // Follow use-def and load-store chains to discover instructions
  StoreInst *Store = cast<StoreInst>(getUniquePreviousStore(PtrArg, EntryBB));
  LoadInst *AtomicStoreAtomicVal = cast<LoadInst>(Store->getValueOperand());

  // %atomic_store.atomic.val = load i32, ptr %ret_ptr, align 4
  EXPECT_EQ(AtomicStoreAtomicVal->getName(), "atomic_store.atomic.val");
  EXPECT_EQ(AtomicStoreAtomicVal->getParent(), EntryBB);
  EXPECT_EQ(AtomicStoreAtomicVal->getType(), Type::getInt32Ty(Ctx));
  EXPECT_TRUE(AtomicStoreAtomicVal->isSimple());
  EXPECT_EQ(AtomicStoreAtomicVal->getPointerOperand(), RetArg);

  // store atomic volatile i32 %atomic_store.atomic.val, ptr %atomic_ptr se...
  EXPECT_TRUE(Store->getName().empty());
  EXPECT_EQ(Store->getParent(), EntryBB);
  EXPECT_TRUE(Store->isVolatile());
  EXPECT_EQ(Store->getAlign(), 1);
  EXPECT_EQ(Store->getOrdering(), AtomicOrdering::SequentiallyConsistent);
  EXPECT_EQ(Store->getSyncScopeID(), SyncScope::System);
  EXPECT_EQ(Store->getValueOperand(), AtomicStoreAtomicVal);
  EXPECT_EQ(Store->getPointerOperand(), PtrArg);
}

TEST_F(BuildBuiltinsTests, AtomicStore_Memorder) {
  ASSERT_THAT_ERROR(emitAtomicStoreBuiltin(
                        /*AtomicPtr=*/PtrArg, /*RetPtr=*/RetArg,
                        /*TypeOrSize=*/Builder.getInt32Ty(),
                        /*IsVolatile=*/false,
                        /*Memorder=*/AtomicOrdering::Monotonic,
                        /*Scope=*/SyncScope::System,
                        /*Align=*/{},
                        /*Builder=*/Builder,
                        /*EmitOptions=*/AtomicEmitOptions(DL, TLI.get()),
                        /*Name=*/"atomic_store"),
                    Succeeded());
  EXPECT_FALSE(verifyModule(*M, &errs()));

  // clang-format off
  // %atomic_store.atomic.val = load i32, ptr %ret_ptr, align 4
  // store atomic i32 %atomic_store.atomic.val, ptr %atomic_ptr monotonic, align 1
  // clang-format on

  // Follow use-def and load-store chains to discover instructions
  StoreInst *Store = cast<StoreInst>(getUniquePreviousStore(PtrArg, EntryBB));
  LoadInst *AtomicStoreAtomicVal = cast<LoadInst>(Store->getValueOperand());

  // %atomic_store.atomic.val = load i32, ptr %ret_ptr, align 4
  EXPECT_EQ(AtomicStoreAtomicVal->getName(), "atomic_store.atomic.val");
  EXPECT_EQ(AtomicStoreAtomicVal->getParent(), EntryBB);
  EXPECT_EQ(AtomicStoreAtomicVal->getType(), Type::getInt32Ty(Ctx));
  EXPECT_TRUE(AtomicStoreAtomicVal->isSimple());
  EXPECT_EQ(AtomicStoreAtomicVal->getPointerOperand(), RetArg);

  // store atomic i32 %atomic_store.atomic.val, ptr %atomic_ptr monotonic, ...
  EXPECT_TRUE(Store->getName().empty());
  EXPECT_EQ(Store->getParent(), EntryBB);
  EXPECT_FALSE(Store->isVolatile());
  EXPECT_EQ(Store->getAlign(), 1);
  EXPECT_EQ(Store->getOrdering(), AtomicOrdering::Monotonic);
  EXPECT_EQ(Store->getSyncScopeID(), SyncScope::System);
  EXPECT_EQ(Store->getValueOperand(), AtomicStoreAtomicVal);
  EXPECT_EQ(Store->getPointerOperand(), PtrArg);
}

TEST_F(BuildBuiltinsTests, AtomicStore_Memorder_CABI) {
  ASSERT_THAT_ERROR(emitAtomicStoreBuiltin(
                        /*AtomicPtr=*/PtrArg, /*RetPtr=*/RetArg,
                        /*TypeOrSize=*/Builder.getInt32Ty(),
                        /*IsVolatile=*/false,
                        /*Memorder=*/AtomicOrderingCABI::relaxed,
                        /*Scope=*/SyncScope::System,
                        /*Align=*/{},
                        /*Builder=*/Builder,
                        /*EmitOptions=*/AtomicEmitOptions(DL, TLI.get()),
                        /*Name=*/"atomic_store"),
                    Succeeded());
  EXPECT_FALSE(verifyModule(*M, &errs()));

  // clang-format off
  // %atomic_store.atomic.val = load i32, ptr %ret_ptr, align 4
  // store atomic i32 %atomic_store.atomic.val, ptr %atomic_ptr monotonic, align 1
  // clang-format on

  // Follow use-def and load-store chains to discover instructions
  StoreInst *Store = cast<StoreInst>(getUniquePreviousStore(PtrArg, EntryBB));
  LoadInst *AtomicStoreAtomicVal = cast<LoadInst>(Store->getValueOperand());

  // %atomic_store.atomic.val = load i32, ptr %ret_ptr, align 4
  EXPECT_EQ(AtomicStoreAtomicVal->getName(), "atomic_store.atomic.val");
  EXPECT_EQ(AtomicStoreAtomicVal->getParent(), EntryBB);
  EXPECT_EQ(AtomicStoreAtomicVal->getType(), Type::getInt32Ty(Ctx));
  EXPECT_TRUE(AtomicStoreAtomicVal->isSimple());
  EXPECT_EQ(AtomicStoreAtomicVal->getPointerOperand(), RetArg);

  // store atomic i32 %atomic_store.atomic.val, ptr %atomic_ptr monotonic, ...
  EXPECT_TRUE(Store->getName().empty());
  EXPECT_EQ(Store->getParent(), EntryBB);
  EXPECT_FALSE(Store->isVolatile());
  EXPECT_EQ(Store->getAlign(), 1);
  EXPECT_EQ(Store->getOrdering(), AtomicOrdering::Monotonic);
  EXPECT_EQ(Store->getSyncScopeID(), SyncScope::System);
  EXPECT_EQ(Store->getValueOperand(), AtomicStoreAtomicVal);
  EXPECT_EQ(Store->getPointerOperand(), PtrArg);
}

TEST_F(BuildBuiltinsTests, AtomicStore_Memorder_Switch) {
  ASSERT_THAT_ERROR(emitAtomicStoreBuiltin(
                        /*AtomicPtr=*/PtrArg, /*RetPtr=*/RetArg,
                        /*TypeOrSize=*/Builder.getInt32Ty(),
                        /*IsVolatile=*/false,
                        /*Memorder=*/MemorderArg,
                        /*Scope=*/SyncScope::System,
                        /*Align=*/{},
                        /*Builder=*/Builder,
                        /*EmitOptions=*/AtomicEmitOptions(DL, TLI.get()),
                        /*Name=*/"atomic_store"),
                    Succeeded());
  EXPECT_FALSE(verifyModule(*M, &errs()));
  BasicBlock *ExitBB = Builder.GetInsertBlock();

  // clang-format off
  // entry:
  //   %atomic_store.atomic.val = load i32, ptr %ret_ptr, align 4
  //   switch i32 %memorderarg_success, label %atomic_store.atomic.monotonic [
  //     i32 3, label %atomic_store.atomic.release
  //     i32 5, label %atomic_store.atomic.seqcst
  //   ]
  //
  // atomic_store.atomic.monotonic:                    ; preds = %entry
  //   store atomic i32 %atomic_store.atomic.val, ptr %atomic_ptr monotonic, align 1
  //   br label %atomic_store.atomic.memorder.continue
  //
  // atomic_store.atomic.release:                      ; preds = %entry
  //   store atomic i32 %atomic_store.atomic.val, ptr %atomic_ptr release, align 1
  //   br label %atomic_store.atomic.memorder.continue
  //
  // atomic_store.atomic.seqcst:                       ; preds = %entry
  //   store atomic i32 %atomic_store.atomic.val, ptr %atomic_ptr seq_cst, align 1
  //   br label %atomic_store.atomic.memorder.continue
  //
  // atomic_store.atomic.memorder.continue:            ; preds = %atomic_store.atomic.seqcst, %atomic_store.atomic.release, %atomic_store.atomic.monotonic
  //   %atomic_store.atomic.memorder.success = phi i1 [ true, %atomic_store.atomic.monotonic ], [ true, %atomic_store.atomic.release ], [ true, %atomic_store.atomic.seqcst ]
  //   ret void
  // clang-format on

  // Discover control flow graph
  SwitchInst *Switch = cast<SwitchInst>(EntryBB->getTerminator());
  BasicBlock *AtomicStoreAtomicRelease =
      cast<BasicBlock>(Switch->getSuccessor(1));
  BasicBlock *AtomicStoreAtomicSeqcst =
      cast<BasicBlock>(Switch->getSuccessor(2));
  BasicBlock *AtomicStoreAtomicMonotonic =
      cast<BasicBlock>(Switch->getDefaultDest());
  BranchInst *Branch1 =
      cast<BranchInst>(AtomicStoreAtomicMonotonic->getTerminator());
  BranchInst *Branch2 =
      cast<BranchInst>(AtomicStoreAtomicRelease->getTerminator());
  BranchInst *Branch3 =
      cast<BranchInst>(AtomicStoreAtomicSeqcst->getTerminator());
  ReturnInst *Return = cast<ReturnInst>(ExitBB->getTerminator());

  // Follow use-def and load-store chains to discover instructions
  StoreInst *Store1 =
      cast<StoreInst>(getUniquePreviousStore(PtrArg, AtomicStoreAtomicSeqcst));
  LoadInst *AtomicStoreAtomicVal = cast<LoadInst>(Store1->getValueOperand());
  StoreInst *Store2 = cast<StoreInst>(
      getUniquePreviousStore(PtrArg, AtomicStoreAtomicMonotonic));
  StoreInst *Store3 =
      cast<StoreInst>(getUniquePreviousStore(PtrArg, AtomicStoreAtomicRelease));

  // %atomic_store.atomic.val = load i32, ptr %ret_ptr, align 4
  EXPECT_EQ(AtomicStoreAtomicVal->getName(), "atomic_store.atomic.val");
  EXPECT_EQ(AtomicStoreAtomicVal->getParent(), EntryBB);
  EXPECT_EQ(AtomicStoreAtomicVal->getType(), Type::getInt32Ty(Ctx));
  EXPECT_TRUE(AtomicStoreAtomicVal->isSimple());
  EXPECT_EQ(AtomicStoreAtomicVal->getPointerOperand(), RetArg);

  // switch i32 %memorderarg_success, label %atomic_store.atomic.monotonic [
  //   i32 3, label %atomic_store.atomic.release
  //   i32 5, label %atomic_store.atomic.seqcst
  // ]
  EXPECT_TRUE(Switch->getName().empty());
  EXPECT_EQ(Switch->getParent(), EntryBB);
  EXPECT_EQ(Switch->getType(), Type::getVoidTy(Ctx));
  EXPECT_EQ(Switch->getCondition(), MemorderArg);
  EXPECT_EQ(Switch->getDefaultDest(), AtomicStoreAtomicMonotonic);
  EXPECT_EQ(cast<ConstantInt>(Switch->getOperand(2))->getZExtValue(), 3);
  EXPECT_EQ(Switch->getOperand(3), AtomicStoreAtomicRelease);
  EXPECT_EQ(cast<ConstantInt>(Switch->getOperand(4))->getZExtValue(), 5);
  EXPECT_EQ(Switch->getOperand(5), AtomicStoreAtomicSeqcst);

  // store atomic i32 %atomic_store.atomic.val, ptr %atomic_ptr monotonic, ...
  EXPECT_TRUE(Store2->getName().empty());
  EXPECT_EQ(Store2->getParent(), AtomicStoreAtomicMonotonic);
  EXPECT_FALSE(Store2->isVolatile());
  EXPECT_EQ(Store2->getAlign(), 1);
  EXPECT_EQ(Store2->getOrdering(), AtomicOrdering::Monotonic);
  EXPECT_EQ(Store2->getSyncScopeID(), SyncScope::System);
  EXPECT_EQ(Store2->getValueOperand(), AtomicStoreAtomicVal);
  EXPECT_EQ(Store2->getPointerOperand(), PtrArg);

  // br label %atomic_store.atomic.memorder.continue
  EXPECT_TRUE(Branch1->getName().empty());
  EXPECT_EQ(Branch1->getParent(), AtomicStoreAtomicMonotonic);
  EXPECT_EQ(Branch1->getType(), Type::getVoidTy(Ctx));
  EXPECT_TRUE(Branch1->isUnconditional());
  EXPECT_EQ(Branch1->getOperand(0), ExitBB);

  // store atomic i32 %atomic_store.atomic.val, ptr %atomic_ptr release, al...
  EXPECT_TRUE(Store3->getName().empty());
  EXPECT_EQ(Store3->getParent(), AtomicStoreAtomicRelease);
  EXPECT_FALSE(Store3->isVolatile());
  EXPECT_EQ(Store3->getAlign(), 1);
  EXPECT_EQ(Store3->getOrdering(), AtomicOrdering::Release);
  EXPECT_EQ(Store3->getSyncScopeID(), SyncScope::System);
  EXPECT_EQ(Store3->getValueOperand(), AtomicStoreAtomicVal);
  EXPECT_EQ(Store3->getPointerOperand(), PtrArg);

  // br label %atomic_store.atomic.memorder.continue
  EXPECT_TRUE(Branch2->getName().empty());
  EXPECT_EQ(Branch2->getParent(), AtomicStoreAtomicRelease);
  EXPECT_EQ(Branch2->getType(), Type::getVoidTy(Ctx));
  EXPECT_TRUE(Branch2->isUnconditional());
  EXPECT_EQ(Branch2->getOperand(0), ExitBB);

  // store atomic i32 %atomic_store.atomic.val, ptr %atomic_ptr seq_cst, al...
  EXPECT_TRUE(Store1->getName().empty());
  EXPECT_EQ(Store1->getParent(), AtomicStoreAtomicSeqcst);
  EXPECT_FALSE(Store1->isVolatile());
  EXPECT_EQ(Store1->getAlign(), 1);
  EXPECT_EQ(Store1->getOrdering(), AtomicOrdering::SequentiallyConsistent);
  EXPECT_EQ(Store1->getSyncScopeID(), SyncScope::System);
  EXPECT_EQ(Store1->getValueOperand(), AtomicStoreAtomicVal);
  EXPECT_EQ(Store1->getPointerOperand(), PtrArg);

  // br label %atomic_store.atomic.memorder.continue
  EXPECT_TRUE(Branch3->getName().empty());
  EXPECT_EQ(Branch3->getParent(), AtomicStoreAtomicSeqcst);
  EXPECT_EQ(Branch3->getType(), Type::getVoidTy(Ctx));
  EXPECT_TRUE(Branch3->isUnconditional());
  EXPECT_EQ(Branch3->getOperand(0), ExitBB);

  // ret void
  EXPECT_TRUE(Return->getName().empty());
  EXPECT_EQ(Return->getParent(), ExitBB);
  EXPECT_EQ(Return->getType(), Type::getVoidTy(Ctx));
}

TEST_F(BuildBuiltinsTests, AtomicStore_SyncScope) {
  ASSERT_THAT_ERROR(emitAtomicStoreBuiltin(
                        /*AtomicPtr=*/PtrArg, /*RetPtr=*/RetArg,
                        /*TypeOrSize=*/Builder.getInt32Ty(),
                        /*IsVolatile=*/false,
                        /*Memorder=*/AtomicOrdering::SequentiallyConsistent,
                        /*Scope=*/SyncScope::SingleThread,
                        /*Align=*/{},
                        /*Builder=*/Builder,
                        /*EmitOptions=*/AtomicEmitOptions(DL, TLI.get()),
                        /*Name=*/"atomic_store"),
                    Succeeded());
  EXPECT_FALSE(verifyModule(*M, &errs()));

  // clang-format off
  // %atomic_store.atomic.val = load i32, ptr %ret_ptr, align 4
  // store atomic i32 %atomic_store.atomic.val, ptr %atomic_ptr syncscope("singlethread") seq_cst, align 1
  // clang-format on

  // Follow use-def and load-store chains to discover instructions
  StoreInst *Store = cast<StoreInst>(getUniquePreviousStore(PtrArg, EntryBB));
  LoadInst *AtomicStoreAtomicVal = cast<LoadInst>(Store->getValueOperand());

  // %atomic_store.atomic.val = load i32, ptr %ret_ptr, align 4
  EXPECT_EQ(AtomicStoreAtomicVal->getName(), "atomic_store.atomic.val");
  EXPECT_EQ(AtomicStoreAtomicVal->getParent(), EntryBB);
  EXPECT_EQ(AtomicStoreAtomicVal->getType(), Type::getInt32Ty(Ctx));
  EXPECT_TRUE(AtomicStoreAtomicVal->isSimple());
  EXPECT_EQ(AtomicStoreAtomicVal->getPointerOperand(), RetArg);

  // store atomic i32 %atomic_store.atomic.val, ptr %atomic_ptr syncscope("...
  EXPECT_TRUE(Store->getName().empty());
  EXPECT_EQ(Store->getParent(), EntryBB);
  EXPECT_FALSE(Store->isVolatile());
  EXPECT_EQ(Store->getAlign(), 1);
  EXPECT_EQ(Store->getOrdering(), AtomicOrdering::SequentiallyConsistent);
  EXPECT_EQ(Store->getSyncScopeID(), SyncScope::SingleThread);
  EXPECT_EQ(Store->getValueOperand(), AtomicStoreAtomicVal);
  EXPECT_EQ(Store->getPointerOperand(), PtrArg);
}

TEST_F(BuildBuiltinsTests, AtomicStore_Float) {
  ASSERT_THAT_ERROR(emitAtomicStoreBuiltin(
                        /*AtomicPtr=*/PtrArg, /*RetPtr=*/RetArg,
                        /*TypeOrSize=*/Builder.getFloatTy(),
                        /*IsVolatile=*/false,
                        /*Memorder=*/AtomicOrdering::SequentiallyConsistent,
                        /*Scope=*/SyncScope::System,
                        /*Align=*/{},
                        /*Builder=*/Builder,
                        /*EmitOptions=*/AtomicEmitOptions(DL, TLI.get()),
                        /*Name=*/"atomic_store"),
                    Succeeded());
  EXPECT_FALSE(verifyModule(*M, &errs()));

  // clang-format off
  // %atomic_store.atomic.val = load float, ptr %ret_ptr, align 4
  // store atomic float %atomic_store.atomic.val, ptr %atomic_ptr seq_cst, align 1
  // clang-format on

  // Follow use-def and load-store chains to discover instructions
  StoreInst *Store = cast<StoreInst>(getUniquePreviousStore(PtrArg, EntryBB));
  LoadInst *AtomicStoreAtomicVal = cast<LoadInst>(Store->getValueOperand());

  // %atomic_store.atomic.val = load float, ptr %ret_ptr, align 4
  EXPECT_EQ(AtomicStoreAtomicVal->getName(), "atomic_store.atomic.val");
  EXPECT_EQ(AtomicStoreAtomicVal->getParent(), EntryBB);
  EXPECT_EQ(AtomicStoreAtomicVal->getType(), Type::getFloatTy(Ctx));
  EXPECT_TRUE(AtomicStoreAtomicVal->isSimple());
  EXPECT_EQ(AtomicStoreAtomicVal->getPointerOperand(), RetArg);

  // store atomic float %atomic_store.atomic.val, ptr %atomic_ptr seq_cst, ...
  EXPECT_TRUE(Store->getName().empty());
  EXPECT_EQ(Store->getParent(), EntryBB);
  EXPECT_FALSE(Store->isVolatile());
  EXPECT_EQ(Store->getAlign(), 1);
  EXPECT_EQ(Store->getOrdering(), AtomicOrdering::SequentiallyConsistent);
  EXPECT_EQ(Store->getSyncScopeID(), SyncScope::System);
  EXPECT_EQ(Store->getValueOperand(), AtomicStoreAtomicVal);
  EXPECT_EQ(Store->getPointerOperand(), PtrArg);
}

TEST_F(BuildBuiltinsTests, AtomicStore_FP80) {
  ASSERT_THAT_ERROR(emitAtomicStoreBuiltin(
                        /*AtomicPtr=*/PtrArg, /*RetPtr=*/RetArg,
                        /*TypeOrSize=*/Type::getX86_FP80Ty(Ctx),
                        /*IsVolatile=*/false,
                        /*Memorder=*/AtomicOrdering::SequentiallyConsistent,
                        /*Scope=*/SyncScope::System,
                        /*Align=*/{},
                        /*Builder=*/Builder,
                        /*EmitOptions=*/AtomicEmitOptions(DL, TLI.get()),
                        /*Name=*/"atomic_store"),
                    Succeeded());
  EXPECT_FALSE(verifyModule(*M, &errs()));

  // Follow use-def and load-store chains to discover instructions
  CallInst *Call = cast<CallInst>(getUniquePreviousStore(PtrArg, EntryBB));

  // call void @__atomic_store(i64 10, ptr %atomic_ptr, ptr %ret_ptr, i32 5)
  EXPECT_TRUE(Call->getName().empty());
  EXPECT_EQ(Call->getParent(), EntryBB);
  EXPECT_EQ(Call->getType(), Type::getVoidTy(Ctx));
  EXPECT_EQ(Call->getName(), "");
  EXPECT_FALSE(Call->isMustTailCall());
  EXPECT_FALSE(Call->isTailCall());
  EXPECT_EQ(Call->getCallingConv(), CallingConv::C);
  EXPECT_EQ(cast<ConstantInt>(Call->getArgOperand(0))->getZExtValue(), 10);
  EXPECT_EQ(Call->getArgOperand(1), PtrArg);
  EXPECT_EQ(Call->getArgOperand(2), RetArg);
  EXPECT_EQ(cast<ConstantInt>(Call->getArgOperand(3))->getZExtValue(), 5);
  EXPECT_EQ(Call->getCalledFunction(), M->getFunction("__atomic_store"));
}

TEST_F(BuildBuiltinsTests, AtomicStore_Ptr) {
  ASSERT_THAT_ERROR(emitAtomicStoreBuiltin(
                        /*AtomicPtr=*/PtrArg, /*RetPtr=*/RetArg,
                        /*TypeOrSize=*/Builder.getPtrTy(), /*IsVolatile=*/false,
                        /*Memorder=*/AtomicOrdering::SequentiallyConsistent,
                        /*Scope=*/SyncScope::System,
                        /*Align=*/{},
                        /*Builder=*/Builder,
                        /*EmitOptions=*/AtomicEmitOptions(DL, TLI.get()),
                        /*Name=*/"atomic_store"),
                    Succeeded());
  EXPECT_FALSE(verifyModule(*M, &errs()));

  // clang-format off
  // %atomic_store.atomic.val = load ptr, ptr %ret_ptr, align 8
  // store atomic ptr %atomic_store.atomic.val, ptr %atomic_ptr seq_cst, align 1
  // clang-format on

  // Follow use-def and load-store chains to discover instructions
  StoreInst *Store = cast<StoreInst>(getUniquePreviousStore(PtrArg, EntryBB));
  LoadInst *AtomicStoreAtomicVal = cast<LoadInst>(Store->getValueOperand());

  // %atomic_store.atomic.val = load ptr, ptr %ret_ptr, align 8
  EXPECT_EQ(AtomicStoreAtomicVal->getName(), "atomic_store.atomic.val");
  EXPECT_EQ(AtomicStoreAtomicVal->getParent(), EntryBB);
  EXPECT_EQ(AtomicStoreAtomicVal->getType(), PointerType::get(Ctx, 0));
  EXPECT_TRUE(AtomicStoreAtomicVal->isSimple());
  EXPECT_EQ(AtomicStoreAtomicVal->getPointerOperand(), RetArg);

  // store atomic ptr %atomic_store.atomic.val, ptr %atomic_ptr seq_cst, al...
  EXPECT_TRUE(Store->getName().empty());
  EXPECT_EQ(Store->getParent(), EntryBB);
  EXPECT_FALSE(Store->isVolatile());
  EXPECT_EQ(Store->getAlign(), 1);
  EXPECT_EQ(Store->getOrdering(), AtomicOrdering::SequentiallyConsistent);
  EXPECT_EQ(Store->getSyncScopeID(), SyncScope::System);
  EXPECT_EQ(Store->getValueOperand(), AtomicStoreAtomicVal);
  EXPECT_EQ(Store->getPointerOperand(), PtrArg);
}

TEST_F(BuildBuiltinsTests, AtomicStore_Struct) {
  // A struct that is small enough to be covered with a single instruction
  StructType *STy =
      StructType::get(Ctx, {Builder.getFloatTy(), Builder.getFloatTy()});

  ASSERT_THAT_ERROR(emitAtomicStoreBuiltin(
                        /*AtomicPtr=*/PtrArg, /*RetPtr=*/RetArg,
                        /*TypeOrSize=*/STy,
                        /*IsVolatile=*/false,
                        /*Memorder=*/AtomicOrdering::SequentiallyConsistent,
                        /*Scope=*/SyncScope::System,
                        /*Align=*/{},
                        /*Builder=*/Builder,
                        /*EmitOptions=*/AtomicEmitOptions(DL, TLI.get()),
                        /*Name=*/"atomic_store"),
                    Succeeded());
  EXPECT_FALSE(verifyModule(*M, &errs()));

  // clang-format off
  // %atomic_store.atomic.val = load i64, ptr %ret_ptr, align 4
  // store atomic i64 %atomic_store.atomic.val, ptr %atomic_ptr seq_cst, align 1
  // clang-format on

  // Follow use-def and load-store chains to discover instructions
  StoreInst *Store = cast<StoreInst>(getUniquePreviousStore(PtrArg, EntryBB));
  LoadInst *AtomicStoreAtomicVal = cast<LoadInst>(Store->getValueOperand());

  // %atomic_store.atomic.val = load i64, ptr %ret_ptr, align 4
  EXPECT_EQ(AtomicStoreAtomicVal->getName(), "atomic_store.atomic.val");
  EXPECT_EQ(AtomicStoreAtomicVal->getParent(), EntryBB);
  EXPECT_EQ(AtomicStoreAtomicVal->getType(), Type::getInt64Ty(Ctx));
  EXPECT_TRUE(AtomicStoreAtomicVal->isSimple());
  EXPECT_EQ(AtomicStoreAtomicVal->getPointerOperand(), RetArg);

  // store atomic i64 %atomic_store.atomic.val, ptr %atomic_ptr seq_cst, al...
  EXPECT_TRUE(Store->getName().empty());
  EXPECT_EQ(Store->getParent(), EntryBB);
  EXPECT_FALSE(Store->isVolatile());
  EXPECT_EQ(Store->getAlign(), 1);
  EXPECT_EQ(Store->getOrdering(), AtomicOrdering::SequentiallyConsistent);
  EXPECT_EQ(Store->getSyncScopeID(), SyncScope::System);
  EXPECT_EQ(Store->getValueOperand(), AtomicStoreAtomicVal);
  EXPECT_EQ(Store->getPointerOperand(), PtrArg);
}

TEST_F(BuildBuiltinsTests, AtomicStore_Array) {
  // A type that is too large for atomic instructions
  ArrayType *ATy = ArrayType::get(Builder.getFloatTy(), 19);

  ASSERT_THAT_ERROR(emitAtomicStoreBuiltin(
                        /*AtomicPtr=*/PtrArg, /*RetPtr=*/RetArg,
                        /*TypeOrSize=*/ATy,
                        /*IsVolatile=*/false,
                        /*Memorder=*/AtomicOrdering::SequentiallyConsistent,
                        /*Scope=*/SyncScope::System,
                        /*Align=*/{},
                        /*Builder=*/Builder,
                        /*EmitOptions=*/AtomicEmitOptions(DL, TLI.get()),
                        /*Name=*/"atomic_store"),
                    Succeeded());
  EXPECT_FALSE(verifyModule(*M, &errs()));

  // Follow use-def and load-store chains to discover instructions
  CallInst *Call = cast<CallInst>(getUniquePreviousStore(PtrArg, EntryBB));

  // call void @__atomic_store(i64 76, ptr %atomic_ptr, ptr %ret_ptr, i32 5)
  EXPECT_TRUE(Call->getName().empty());
  EXPECT_EQ(Call->getParent(), EntryBB);
  EXPECT_EQ(Call->getType(), Type::getVoidTy(Ctx));
  EXPECT_EQ(Call->getName(), "");
  EXPECT_FALSE(Call->isMustTailCall());
  EXPECT_FALSE(Call->isTailCall());
  EXPECT_EQ(Call->getCallingConv(), CallingConv::C);
  EXPECT_EQ(cast<ConstantInt>(Call->getArgOperand(0))->getZExtValue(), 76);
  EXPECT_EQ(Call->getArgOperand(1), PtrArg);
  EXPECT_EQ(Call->getArgOperand(2), RetArg);
  EXPECT_EQ(cast<ConstantInt>(Call->getArgOperand(3))->getZExtValue(), 5);
  EXPECT_EQ(Call->getCalledFunction(), M->getFunction("__atomic_store"));
}

TEST_F(BuildBuiltinsTests, AtomicStore_Array_NoLibatomic) {
  // Use a triple that does not support libatomic (according to
  // initializeLibCalls in TargetLibraryInfo.cpp)
  Triple T("x86_64-scei-ps4");
  TLII.reset(new TargetLibraryInfoImpl(T));
  TLI.reset(new TargetLibraryInfo(*TLII));

  // A type that is too large for atomic instructions
  ArrayType *ATy = ArrayType::get(Builder.getFloatTy(), 19);

  ASSERT_THAT_ERROR(
      emitAtomicStoreBuiltin(
          /*AtomicPtr=*/PtrArg, /*RetPtr=*/RetArg, /*TypeOrSize=*/ATy,
          /*IsVolatile=*/false,
          /*Memorder=*/AtomicOrdering::SequentiallyConsistent,
          /*Scope=*/SyncScope::System,
          /*Align=*/{},
          /*Builder=*/Builder,
          /*EmitOptions=*/AtomicEmitOptions(DL, TLI.get()),
          /*Name=*/"atomic_store"),
      FailedWithMessage(
          "__atomic_store builtin not supported by any available means"));
}

TEST_F(BuildBuiltinsTests, AtomicStore_DataSize) {
  ASSERT_THAT_ERROR(emitAtomicStoreBuiltin(
                        /*AtomicPtr=*/PtrArg, /*RetPtr=*/RetArg,
                        /*TypeOrSize=*/static_cast<uint64_t>(6),
                        /*IsVolatile=*/false,
                        /*Memorder=*/AtomicOrdering::SequentiallyConsistent,
                        /*Scope=*/SyncScope::System, /*Align=*/{}, Builder,
                        /*EmitOptions=*/AtomicEmitOptions(DL, TLI.get()),
                        /*Name=*/"atomic_store"),
                    Succeeded());
  EXPECT_FALSE(verifyModule(*M, &errs()));

  // Follow use-def and load-store chains to discover instructions
  CallInst *Call = cast<CallInst>(getUniquePreviousStore(PtrArg, EntryBB));

  // call void @__atomic_store(i64 6, ptr %atomic_ptr, ptr %ret_ptr, i32 5)
  EXPECT_TRUE(Call->getName().empty());
  EXPECT_EQ(Call->getParent(), EntryBB);
  EXPECT_EQ(Call->getType(), Type::getVoidTy(Ctx));
  EXPECT_EQ(Call->getName(), "");
  EXPECT_FALSE(Call->isMustTailCall());
  EXPECT_FALSE(Call->isTailCall());
  EXPECT_EQ(Call->getCallingConv(), CallingConv::C);
  EXPECT_EQ(cast<ConstantInt>(Call->getArgOperand(0))->getZExtValue(), 6);
  EXPECT_EQ(Call->getArgOperand(1), PtrArg);
  EXPECT_EQ(Call->getArgOperand(2), RetArg);
  EXPECT_EQ(cast<ConstantInt>(Call->getArgOperand(3))->getZExtValue(), 5);
  EXPECT_EQ(Call->getCalledFunction(), M->getFunction("__atomic_store"));
}

TEST_F(BuildBuiltinsTests, AtomicStore_Align) {
  ASSERT_THAT_ERROR(emitAtomicStoreBuiltin(
                        /*AtomicPtr=*/PtrArg, /*RetPtr=*/RetArg,
                        /*TypeOrSize=*/Builder.getInt32Ty(),
                        /*IsVolatile=*/false,
                        /*Memorder=*/AtomicOrdering::SequentiallyConsistent,
                        /*Scope=*/SyncScope::System,
                        /*Align=*/Align(8),
                        /*Builder=*/Builder,
                        /*EmitOptions=*/AtomicEmitOptions(DL, TLI.get()),
                        /*Name=*/"atomic_store"),
                    Succeeded());
  EXPECT_FALSE(verifyModule(*M, &errs()));

  // clang-format off
  // %atomic_store.atomic.val = load i32, ptr %ret_ptr, align 4
  // store atomic i32 %atomic_store.atomic.val, ptr %atomic_ptr seq_cst, align 8
  // clang-format on

  // Follow use-def and load-store chains to discover instructions
  StoreInst *Store = cast<StoreInst>(getUniquePreviousStore(PtrArg, EntryBB));
  LoadInst *AtomicStoreAtomicVal = cast<LoadInst>(Store->getValueOperand());

  // %atomic_store.atomic.val = load i32, ptr %ret_ptr, align 4
  EXPECT_EQ(AtomicStoreAtomicVal->getName(), "atomic_store.atomic.val");
  EXPECT_EQ(AtomicStoreAtomicVal->getParent(), EntryBB);
  EXPECT_EQ(AtomicStoreAtomicVal->getType(), Type::getInt32Ty(Ctx));
  EXPECT_TRUE(AtomicStoreAtomicVal->isSimple());
  EXPECT_EQ(AtomicStoreAtomicVal->getPointerOperand(), RetArg);

  // store atomic i32 %atomic_store.atomic.val, ptr %atomic_ptr seq_cst, al...
  EXPECT_TRUE(Store->getName().empty());
  EXPECT_EQ(Store->getParent(), EntryBB);
  EXPECT_FALSE(Store->isVolatile());
  EXPECT_EQ(Store->getAlign(), 8);
  EXPECT_EQ(Store->getOrdering(), AtomicOrdering::SequentiallyConsistent);
  EXPECT_EQ(Store->getSyncScopeID(), SyncScope::System);
  EXPECT_EQ(Store->getValueOperand(), AtomicStoreAtomicVal);
  EXPECT_EQ(Store->getPointerOperand(), PtrArg);
}

TEST_F(BuildBuiltinsTests, AtomicCmpxchg) {
  Value *AtomicSuccess = nullptr;
  ASSERT_THAT_EXPECTED(
      emitAtomicCompareExchangeBuiltin(
          /*AtomicPtr=*/PtrArg,
          /*ExpectedPtr=*/ExpectedArg,
          /*DesiredPtr=*/DesiredArg, /*TypeOrSize=*/Builder.getInt32Ty(),
          /*IsWeak*/ false,
          /*IsVolatile=*/false,
          /*SuccessMemorder=*/AtomicOrdering::SequentiallyConsistent,
          /*FailureMemorder=*/AtomicOrdering::SequentiallyConsistent,
          /*Scope=*/SyncScope::System,
          /*PrevPtr=*/nullptr,
          /*Align=*/{}, /*Builder=*/Builder,
          /*EmitOptions=*/AtomicEmitOptions(DL, TLI.get()),
          /*Name=*/"atomic_cmpxchg"),
      StoreResult(AtomicSuccess));
  EXPECT_FALSE(verifyModule(*M, &errs()));

  // clang-format off
  // %atomic_cmpxchg.cmpxchg.expected = load i32, ptr %expected_ptr, align 4
  // %atomic_cmpxchg.cmpxchg.desired = load i32, ptr %desired_ptr, align 4
  // %atomic_cmpxchg.cmpxchg.pair = cmpxchg ptr %atomic_ptr, i32 %atomic_cmpxchg.cmpxchg.expected, i32 %atomic_cmpxchg.cmpxchg.desired seq_cst seq_cst, align 1
  // %atomic_cmpxchg.cmpxchg.success = extractvalue { i32, i1 } %atomic_cmpxchg.cmpxchg.pair, 1
  // clang-format on

  // Follow use-def and load-store chains to discover instructions
  AtomicCmpXchgInst *AtomicCmpxchgCmpxchgPair =
      cast<AtomicCmpXchgInst>(getUniquePreviousStore(PtrArg, EntryBB));
  LoadInst *AtomicCmpxchgCmpxchgExpected =
      cast<LoadInst>(AtomicCmpxchgCmpxchgPair->getCompareOperand());
  LoadInst *AtomicCmpxchgCmpxchgDesired =
      cast<LoadInst>(AtomicCmpxchgCmpxchgPair->getNewValOperand());

  // %atomic_cmpxchg.cmpxchg.expected = load i32, ptr %expected_ptr, align 4
  EXPECT_EQ(AtomicCmpxchgCmpxchgExpected->getName(),
            "atomic_cmpxchg.cmpxchg.expected");
  EXPECT_EQ(AtomicCmpxchgCmpxchgExpected->getParent(), EntryBB);
  EXPECT_EQ(AtomicCmpxchgCmpxchgExpected->getType(), Type::getInt32Ty(Ctx));
  EXPECT_TRUE(AtomicCmpxchgCmpxchgExpected->isSimple());
  EXPECT_EQ(AtomicCmpxchgCmpxchgExpected->getPointerOperand(), ExpectedArg);

  // %atomic_cmpxchg.cmpxchg.desired = load i32, ptr %desired_ptr, align 4
  EXPECT_EQ(AtomicCmpxchgCmpxchgDesired->getName(),
            "atomic_cmpxchg.cmpxchg.desired");
  EXPECT_EQ(AtomicCmpxchgCmpxchgDesired->getParent(), EntryBB);
  EXPECT_EQ(AtomicCmpxchgCmpxchgDesired->getType(), Type::getInt32Ty(Ctx));
  EXPECT_TRUE(AtomicCmpxchgCmpxchgDesired->isSimple());
  EXPECT_EQ(AtomicCmpxchgCmpxchgDesired->getPointerOperand(), DesiredArg);

  // %atomic_cmpxchg.cmpxchg.pair = cmpxchg ptr %atomic_ptr, i32 %atomic_cm...
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getName(), "atomic_cmpxchg.cmpxchg.pair");
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getParent(), EntryBB);
  EXPECT_FALSE(AtomicCmpxchgCmpxchgPair->isVolatile());
  EXPECT_FALSE(AtomicCmpxchgCmpxchgPair->isWeak());
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getSuccessOrdering(),
            AtomicOrdering::SequentiallyConsistent);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getFailureOrdering(),
            AtomicOrdering::SequentiallyConsistent);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getSyncScopeID(), SyncScope::System);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getAlign(), 1);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getPointerOperand(), PtrArg);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getCompareOperand(),
            AtomicCmpxchgCmpxchgExpected);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getNewValOperand(),
            AtomicCmpxchgCmpxchgDesired);

  // %atomic_cmpxchg.cmpxchg.success = extractvalue { i32, i1 } %atomic_cmp...
  EXPECT_EQ(AtomicSuccess->getName(), "atomic_cmpxchg.cmpxchg.success");
  EXPECT_EQ(cast<Instruction>(AtomicSuccess)->getParent(), EntryBB);
  EXPECT_EQ(AtomicSuccess->getType(), Type::getInt1Ty(Ctx));
  EXPECT_EQ(cast<ExtractValueInst>(AtomicSuccess)->getNumIndices(), 1);
  EXPECT_EQ(cast<ExtractValueInst>(AtomicSuccess)->getIndices()[0], 1);
  EXPECT_EQ(cast<ExtractValueInst>(AtomicSuccess)->getAggregateOperand(),
            AtomicCmpxchgCmpxchgPair);
}

TEST_F(BuildBuiltinsTests, AtomicCmpxchg_SizedLibcall) {
  AtomicEmitOptions EO(DL, TLI.get());
  EO.AllowInstruction = false;

  Value *AtomicSuccess = nullptr;
  ASSERT_THAT_EXPECTED(
      emitAtomicCompareExchangeBuiltin(
          /*AtomicPtr=*/PtrArg,
          /*ExpectedPtr=*/ExpectedArg,
          /*DesiredPtr=*/DesiredArg, /*TypeOrSize=*/Builder.getInt32Ty(),
          /*IsWeak*/ false,
          /*IsVolatile=*/false,
          /*SuccessMemorder=*/AtomicOrdering::SequentiallyConsistent,
          /*FailureMemorder=*/AtomicOrdering::SequentiallyConsistent,
          /*Scope=*/SyncScope::System,
          /*PrevPtr=*/nullptr,
          /*Align=*/{}, /*Builder=*/Builder,
          /*EmitOptions=*/EO,
          /*Name=*/"atomic_cmpxchg"),
      StoreResult(AtomicSuccess));
  EXPECT_FALSE(verifyModule(*M, &errs()));

  // clang-format off
  // %__atomic_compare_exchange = call i8 @__atomic_compare_exchange(i64 4, ptr %atomic_ptr, ptr %expected_ptr, ptr %desired_ptr, i32 5, i32 5)
  // %atomic_cmpxchg.cmpxchg.success = icmp eq i8 %__atomic_compare_exchange, 0
  // clang-format on

  // Follow use-def and load-store chains to discover instructions
  CallInst *AtomicCompareExchange =
      cast<CallInst>(getUniquePreviousStore(PtrArg, EntryBB));

  // %__atomic_compare_exchange = call i8 @__atomic_compare_exchange(i64 4,...
  EXPECT_EQ(AtomicCompareExchange->getName(), "__atomic_compare_exchange");
  EXPECT_EQ(AtomicCompareExchange->getParent(), EntryBB);
  EXPECT_EQ(AtomicCompareExchange->getType(), Type::getInt8Ty(Ctx));
  EXPECT_EQ(AtomicCompareExchange->getName(), "__atomic_compare_exchange");
  EXPECT_FALSE(AtomicCompareExchange->isMustTailCall());
  EXPECT_FALSE(AtomicCompareExchange->isTailCall());
  EXPECT_EQ(AtomicCompareExchange->getCallingConv(), CallingConv::C);
  EXPECT_EQ(cast<ConstantInt>(AtomicCompareExchange->getArgOperand(0))
                ->getZExtValue(),
            4);
  EXPECT_EQ(AtomicCompareExchange->getArgOperand(1), PtrArg);
  EXPECT_EQ(AtomicCompareExchange->getArgOperand(2), ExpectedArg);
  EXPECT_EQ(AtomicCompareExchange->getArgOperand(3), DesiredArg);
  EXPECT_EQ(cast<ConstantInt>(AtomicCompareExchange->getArgOperand(4))
                ->getZExtValue(),
            5);
  EXPECT_EQ(cast<ConstantInt>(AtomicCompareExchange->getArgOperand(5))
                ->getZExtValue(),
            5);
  EXPECT_EQ(AtomicCompareExchange->getCalledFunction(),
            M->getFunction("__atomic_compare_exchange"));

  // %atomic_cmpxchg.cmpxchg.success = icmp eq i8 %__atomic_compare_exchang...
  EXPECT_EQ(AtomicSuccess->getName(), "atomic_cmpxchg.cmpxchg.success");
  EXPECT_EQ(cast<Instruction>(AtomicSuccess)->getParent(), EntryBB);
  EXPECT_EQ(AtomicSuccess->getType(), Type::getInt1Ty(Ctx));
  EXPECT_EQ(cast<Instruction>(AtomicSuccess)->getOperand(0),
            AtomicCompareExchange);
  EXPECT_EQ(cast<ConstantInt>(cast<Instruction>(AtomicSuccess)->getOperand(1))
                ->getZExtValue(),
            0);
}

TEST_F(BuildBuiltinsTests, AtomicCmpxchg_Libcall) {
  AtomicEmitOptions EO(DL, TLI.get());
  EO.AllowInstruction = false;
  EO.AllowSizedLibcall = false;

  Value *AtomicSuccess = nullptr;
  ASSERT_THAT_EXPECTED(
      emitAtomicCompareExchangeBuiltin(
          /*AtomicPtr=*/PtrArg,
          /*ExpectedPtr=*/ExpectedArg,
          /*DesiredPtr=*/DesiredArg, /*TypeOrSize=*/Builder.getInt32Ty(),
          /*IsWeak*/ false,
          /*IsVolatile=*/false,
          /*SuccessMemorder=*/AtomicOrdering::SequentiallyConsistent,
          /*FailureMemorder=*/AtomicOrdering::SequentiallyConsistent,
          /*Scope=*/SyncScope::System,
          /*PrevPtr=*/nullptr,
          /*Align=*/{}, /*Builder=*/Builder,
          /*EmitOptions=*/EO,
          /*Name=*/"atomic_cmpxchg"),
      StoreResult(AtomicSuccess));
  EXPECT_FALSE(verifyModule(*M, &errs()));

  // clang-format off
  // %__atomic_compare_exchange = call i8 @__atomic_compare_exchange(i64 4, ptr %atomic_ptr, ptr %expected_ptr, ptr %desired_ptr, i32 5, i32 5)
  // %atomic_cmpxchg.cmpxchg.success = icmp eq i8 %__atomic_compare_exchange, 0
  // clang-format on

  // Follow use-def and load-store chains to discover instructions
  CallInst *AtomicCompareExchange =
      cast<CallInst>(getUniquePreviousStore(PtrArg, EntryBB));

  // %__atomic_compare_exchange = call i8 @__atomic_compare_exchange(i64 4,...
  EXPECT_EQ(AtomicCompareExchange->getName(), "__atomic_compare_exchange");
  EXPECT_EQ(AtomicCompareExchange->getParent(), EntryBB);
  EXPECT_EQ(AtomicCompareExchange->getType(), Type::getInt8Ty(Ctx));
  EXPECT_EQ(AtomicCompareExchange->getName(), "__atomic_compare_exchange");
  EXPECT_FALSE(AtomicCompareExchange->isMustTailCall());
  EXPECT_FALSE(AtomicCompareExchange->isTailCall());
  EXPECT_EQ(AtomicCompareExchange->getCallingConv(), CallingConv::C);
  EXPECT_EQ(cast<ConstantInt>(AtomicCompareExchange->getArgOperand(0))
                ->getZExtValue(),
            4);
  EXPECT_EQ(AtomicCompareExchange->getArgOperand(1), PtrArg);
  EXPECT_EQ(AtomicCompareExchange->getArgOperand(2), ExpectedArg);
  EXPECT_EQ(AtomicCompareExchange->getArgOperand(3), DesiredArg);
  EXPECT_EQ(cast<ConstantInt>(AtomicCompareExchange->getArgOperand(4))
                ->getZExtValue(),
            5);
  EXPECT_EQ(cast<ConstantInt>(AtomicCompareExchange->getArgOperand(5))
                ->getZExtValue(),
            5);
  EXPECT_EQ(AtomicCompareExchange->getCalledFunction(),
            M->getFunction("__atomic_compare_exchange"));

  // %atomic_cmpxchg.cmpxchg.success = icmp eq i8 %__atomic_compare_exchang...
  EXPECT_EQ(AtomicSuccess->getName(), "atomic_cmpxchg.cmpxchg.success");
  EXPECT_EQ(cast<Instruction>(AtomicSuccess)->getParent(), EntryBB);
  EXPECT_EQ(AtomicSuccess->getType(), Type::getInt1Ty(Ctx));
  EXPECT_EQ(cast<Instruction>(AtomicSuccess)->getOperand(0),
            AtomicCompareExchange);
  EXPECT_EQ(cast<ConstantInt>(cast<Instruction>(AtomicSuccess)->getOperand(1))
                ->getZExtValue(),
            0);
}

TEST_F(BuildBuiltinsTests, AtomicCmpxchg_Weak) {
  Value *AtomicSuccess = nullptr;
  ASSERT_THAT_EXPECTED(
      emitAtomicCompareExchangeBuiltin(
          /*AtomicPtr=*/PtrArg,
          /*ExpectedPtr=*/ExpectedArg,
          /*DesiredPtr=*/DesiredArg, /*TypeOrSize=*/Builder.getInt32Ty(),
          /*IsWeak*/ true,
          /*IsVolatile=*/false,
          /*SuccessMemorder=*/AtomicOrdering::SequentiallyConsistent,
          /*FailureMemorder=*/AtomicOrdering::SequentiallyConsistent,
          /*Scope=*/SyncScope::System,
          /*PrevPtr=*/nullptr,
          /*Align=*/{}, /*Builder=*/Builder,
          /*EmitOptions=*/AtomicEmitOptions(DL, TLI.get()),
          /*Name=*/"atomic_cmpxchg"),
      StoreResult(AtomicSuccess));
  EXPECT_FALSE(verifyModule(*M, &errs()));

  // clang-format off
  // %atomic_cmpxchg.cmpxchg.expected = load i32, ptr %expected_ptr, align 4
  // %atomic_cmpxchg.cmpxchg.desired = load i32, ptr %desired_ptr, align 4
  // %atomic_cmpxchg.cmpxchg.pair = cmpxchg weak ptr %atomic_ptr, i32 %atomic_cmpxchg.cmpxchg.expected, i32 %atomic_cmpxchg.cmpxchg.desired seq_cst seq_cst, align 1
  // %atomic_cmpxchg.cmpxchg.success = extractvalue { i32, i1 } %atomic_cmpxchg.cmpxchg.pair, 1
  // clang-format on

  // Follow use-def and load-store chains to discover instructions
  AtomicCmpXchgInst *AtomicCmpxchgCmpxchgPair =
      cast<AtomicCmpXchgInst>(getUniquePreviousStore(PtrArg, EntryBB));
  LoadInst *AtomicCmpxchgCmpxchgExpected =
      cast<LoadInst>(AtomicCmpxchgCmpxchgPair->getCompareOperand());
  LoadInst *AtomicCmpxchgCmpxchgDesired =
      cast<LoadInst>(AtomicCmpxchgCmpxchgPair->getNewValOperand());

  // %atomic_cmpxchg.cmpxchg.expected = load i32, ptr %expected_ptr, align 4
  EXPECT_EQ(AtomicCmpxchgCmpxchgExpected->getName(),
            "atomic_cmpxchg.cmpxchg.expected");
  EXPECT_EQ(AtomicCmpxchgCmpxchgExpected->getParent(), EntryBB);
  EXPECT_EQ(AtomicCmpxchgCmpxchgExpected->getType(), Type::getInt32Ty(Ctx));
  EXPECT_TRUE(AtomicCmpxchgCmpxchgExpected->isSimple());
  EXPECT_EQ(AtomicCmpxchgCmpxchgExpected->getPointerOperand(), ExpectedArg);

  // %atomic_cmpxchg.cmpxchg.desired = load i32, ptr %desired_ptr, align 4
  EXPECT_EQ(AtomicCmpxchgCmpxchgDesired->getName(),
            "atomic_cmpxchg.cmpxchg.desired");
  EXPECT_EQ(AtomicCmpxchgCmpxchgDesired->getParent(), EntryBB);
  EXPECT_EQ(AtomicCmpxchgCmpxchgDesired->getType(), Type::getInt32Ty(Ctx));
  EXPECT_TRUE(AtomicCmpxchgCmpxchgDesired->isSimple());
  EXPECT_EQ(AtomicCmpxchgCmpxchgDesired->getPointerOperand(), DesiredArg);

  // %atomic_cmpxchg.cmpxchg.pair = cmpxchg weak ptr %atomic_ptr, i32 %atom...
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getName(), "atomic_cmpxchg.cmpxchg.pair");
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getParent(), EntryBB);
  EXPECT_FALSE(AtomicCmpxchgCmpxchgPair->isVolatile());
  EXPECT_TRUE(AtomicCmpxchgCmpxchgPair->isWeak());
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getSuccessOrdering(),
            AtomicOrdering::SequentiallyConsistent);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getFailureOrdering(),
            AtomicOrdering::SequentiallyConsistent);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getSyncScopeID(), SyncScope::System);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getAlign(), 1);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getPointerOperand(), PtrArg);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getCompareOperand(),
            AtomicCmpxchgCmpxchgExpected);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getNewValOperand(),
            AtomicCmpxchgCmpxchgDesired);

  // %atomic_cmpxchg.cmpxchg.success = extractvalue { i32, i1 } %atomic_cmp...
  EXPECT_EQ(AtomicSuccess->getName(), "atomic_cmpxchg.cmpxchg.success");
  EXPECT_EQ(cast<Instruction>(AtomicSuccess)->getParent(), EntryBB);
  EXPECT_EQ(AtomicSuccess->getType(), Type::getInt1Ty(Ctx));
  EXPECT_EQ(cast<ExtractValueInst>(AtomicSuccess)->getNumIndices(), 1);
  EXPECT_EQ(cast<ExtractValueInst>(AtomicSuccess)->getIndices()[0], 1);
  EXPECT_EQ(cast<ExtractValueInst>(AtomicSuccess)->getAggregateOperand(),
            AtomicCmpxchgCmpxchgPair);
}

TEST_F(BuildBuiltinsTests, AtomicCmpxchg_Volatile) {
  Value *AtomicSuccess = nullptr;
  ASSERT_THAT_EXPECTED(
      emitAtomicCompareExchangeBuiltin(
          /*AtomicPtr=*/PtrArg,
          /*ExpectedPtr=*/ExpectedArg,
          /*DesiredPtr=*/DesiredArg, /*TypeOrSize=*/Builder.getInt32Ty(),
          /*IsWeak*/ false,
          /*IsVolatile=*/true,
          /*SuccessMemorder=*/AtomicOrdering::SequentiallyConsistent,
          /*FailureMemorder=*/AtomicOrdering::SequentiallyConsistent,
          /*Scope=*/SyncScope::System,
          /*PrevPtr=*/nullptr,
          /*Align=*/{}, /*Builder=*/Builder,
          /*EmitOptions=*/AtomicEmitOptions(DL, TLI.get()),
          /*Name=*/"atomic_cmpxchg"),
      StoreResult(AtomicSuccess));
  EXPECT_FALSE(verifyModule(*M, &errs()));

  // clang-format off
  // %atomic_cmpxchg.cmpxchg.expected = load i32, ptr %expected_ptr, align 4
  // %atomic_cmpxchg.cmpxchg.desired = load i32, ptr %desired_ptr, align 4
  // %atomic_cmpxchg.cmpxchg.pair = cmpxchg volatile ptr %atomic_ptr, i32 %atomic_cmpxchg.cmpxchg.expected, i32 %atomic_cmpxchg.cmpxchg.desired seq_cst seq_cst, align 1
  // %atomic_cmpxchg.cmpxchg.success = extractvalue { i32, i1 } %atomic_cmpxchg.cmpxchg.pair, 1
  // clang-format on

  // Follow use-def and load-store chains to discover instructions
  AtomicCmpXchgInst *AtomicCmpxchgCmpxchgPair =
      cast<AtomicCmpXchgInst>(getUniquePreviousStore(PtrArg, EntryBB));
  LoadInst *AtomicCmpxchgCmpxchgExpected =
      cast<LoadInst>(AtomicCmpxchgCmpxchgPair->getCompareOperand());
  LoadInst *AtomicCmpxchgCmpxchgDesired =
      cast<LoadInst>(AtomicCmpxchgCmpxchgPair->getNewValOperand());

  // %atomic_cmpxchg.cmpxchg.expected = load i32, ptr %expected_ptr, align 4
  EXPECT_EQ(AtomicCmpxchgCmpxchgExpected->getName(),
            "atomic_cmpxchg.cmpxchg.expected");
  EXPECT_EQ(AtomicCmpxchgCmpxchgExpected->getParent(), EntryBB);
  EXPECT_EQ(AtomicCmpxchgCmpxchgExpected->getType(), Type::getInt32Ty(Ctx));
  EXPECT_TRUE(AtomicCmpxchgCmpxchgExpected->isSimple());
  EXPECT_EQ(AtomicCmpxchgCmpxchgExpected->getPointerOperand(), ExpectedArg);

  // %atomic_cmpxchg.cmpxchg.desired = load i32, ptr %desired_ptr, align 4
  EXPECT_EQ(AtomicCmpxchgCmpxchgDesired->getName(),
            "atomic_cmpxchg.cmpxchg.desired");
  EXPECT_EQ(AtomicCmpxchgCmpxchgDesired->getParent(), EntryBB);
  EXPECT_EQ(AtomicCmpxchgCmpxchgDesired->getType(), Type::getInt32Ty(Ctx));
  EXPECT_TRUE(AtomicCmpxchgCmpxchgDesired->isSimple());
  EXPECT_EQ(AtomicCmpxchgCmpxchgDesired->getPointerOperand(), DesiredArg);

  // %atomic_cmpxchg.cmpxchg.pair = cmpxchg volatile ptr %atomic_ptr, i32 %...
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getName(), "atomic_cmpxchg.cmpxchg.pair");
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getParent(), EntryBB);
  EXPECT_TRUE(AtomicCmpxchgCmpxchgPair->isVolatile());
  EXPECT_FALSE(AtomicCmpxchgCmpxchgPair->isWeak());
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getSuccessOrdering(),
            AtomicOrdering::SequentiallyConsistent);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getFailureOrdering(),
            AtomicOrdering::SequentiallyConsistent);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getSyncScopeID(), SyncScope::System);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getAlign(), 1);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getPointerOperand(), PtrArg);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getCompareOperand(),
            AtomicCmpxchgCmpxchgExpected);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getNewValOperand(),
            AtomicCmpxchgCmpxchgDesired);

  // %atomic_cmpxchg.cmpxchg.success = extractvalue { i32, i1 } %atomic_cmp...
  EXPECT_EQ(AtomicSuccess->getName(), "atomic_cmpxchg.cmpxchg.success");
  EXPECT_EQ(cast<Instruction>(AtomicSuccess)->getParent(), EntryBB);
  EXPECT_EQ(AtomicSuccess->getType(), Type::getInt1Ty(Ctx));
  EXPECT_EQ(cast<ExtractValueInst>(AtomicSuccess)->getNumIndices(), 1);
  EXPECT_EQ(cast<ExtractValueInst>(AtomicSuccess)->getIndices()[0], 1);
  EXPECT_EQ(cast<ExtractValueInst>(AtomicSuccess)->getAggregateOperand(),
            AtomicCmpxchgCmpxchgPair);
}

TEST_F(BuildBuiltinsTests, AtomicCmpxchg_Memorder) {
  Value *AtomicSuccess = nullptr;
  ASSERT_THAT_EXPECTED(emitAtomicCompareExchangeBuiltin(
                           /*AtomicPtr=*/PtrArg,
                           /*ExpectedPtr=*/ExpectedArg,
                           /*DesiredPtr=*/DesiredArg,
                           /*TypeOrSize=*/Builder.getInt32Ty(),
                           /*IsWeak*/ false,
                           /*IsVolatile=*/true,
                           /*SuccessMemorder=*/AtomicOrdering::AcquireRelease,
                           /*FailureMemorder=*/AtomicOrdering::Monotonic,
                           /*Scope=*/SyncScope::System,
                           /*PrevPtr=*/nullptr,
                           /*Align=*/{}, /*Builder=*/Builder,
                           /*EmitOptions=*/AtomicEmitOptions(DL, TLI.get()),
                           /*Name=*/"atomic_cmpxchg"),
                       StoreResult(AtomicSuccess));
  EXPECT_FALSE(verifyModule(*M, &errs()));

  // clang-format off
  // %atomic_cmpxchg.cmpxchg.expected = load i32, ptr %expected_ptr, align 4
  // %atomic_cmpxchg.cmpxchg.desired = load i32, ptr %desired_ptr, align 4
  // %atomic_cmpxchg.cmpxchg.pair = cmpxchg volatile ptr %atomic_ptr, i32 %atomic_cmpxchg.cmpxchg.expected, i32 %atomic_cmpxchg.cmpxchg.desired acq_rel monotonic, align 1
  // %atomic_cmpxchg.cmpxchg.success = extractvalue { i32, i1 } %atomic_cmpxchg.cmpxchg.pair, 1
  // clang-format on

  // Follow use-def and load-store chains to discover instructions
  AtomicCmpXchgInst *AtomicCmpxchgCmpxchgPair =
      cast<AtomicCmpXchgInst>(getUniquePreviousStore(PtrArg, EntryBB));
  LoadInst *AtomicCmpxchgCmpxchgExpected =
      cast<LoadInst>(AtomicCmpxchgCmpxchgPair->getCompareOperand());
  LoadInst *AtomicCmpxchgCmpxchgDesired =
      cast<LoadInst>(AtomicCmpxchgCmpxchgPair->getNewValOperand());

  // %atomic_cmpxchg.cmpxchg.expected = load i32, ptr %expected_ptr, align 4
  EXPECT_EQ(AtomicCmpxchgCmpxchgExpected->getName(),
            "atomic_cmpxchg.cmpxchg.expected");
  EXPECT_EQ(AtomicCmpxchgCmpxchgExpected->getParent(), EntryBB);
  EXPECT_EQ(AtomicCmpxchgCmpxchgExpected->getType(), Type::getInt32Ty(Ctx));
  EXPECT_TRUE(AtomicCmpxchgCmpxchgExpected->isSimple());
  EXPECT_EQ(AtomicCmpxchgCmpxchgExpected->getPointerOperand(), ExpectedArg);

  // %atomic_cmpxchg.cmpxchg.desired = load i32, ptr %desired_ptr, align 4
  EXPECT_EQ(AtomicCmpxchgCmpxchgDesired->getName(),
            "atomic_cmpxchg.cmpxchg.desired");
  EXPECT_EQ(AtomicCmpxchgCmpxchgDesired->getParent(), EntryBB);
  EXPECT_EQ(AtomicCmpxchgCmpxchgDesired->getType(), Type::getInt32Ty(Ctx));
  EXPECT_TRUE(AtomicCmpxchgCmpxchgDesired->isSimple());
  EXPECT_EQ(AtomicCmpxchgCmpxchgDesired->getPointerOperand(), DesiredArg);

  // %atomic_cmpxchg.cmpxchg.pair = cmpxchg volatile ptr %atomic_ptr, i32 %...
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getName(), "atomic_cmpxchg.cmpxchg.pair");
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getParent(), EntryBB);
  EXPECT_TRUE(AtomicCmpxchgCmpxchgPair->isVolatile());
  EXPECT_FALSE(AtomicCmpxchgCmpxchgPair->isWeak());
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getSuccessOrdering(),
            AtomicOrdering::AcquireRelease);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getFailureOrdering(),
            AtomicOrdering::Monotonic);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getSyncScopeID(), SyncScope::System);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getAlign(), 1);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getPointerOperand(), PtrArg);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getCompareOperand(),
            AtomicCmpxchgCmpxchgExpected);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getNewValOperand(),
            AtomicCmpxchgCmpxchgDesired);

  // %atomic_cmpxchg.cmpxchg.success = extractvalue { i32, i1 } %atomic_cmp...
  EXPECT_EQ(AtomicSuccess->getName(), "atomic_cmpxchg.cmpxchg.success");
  EXPECT_EQ(cast<Instruction>(AtomicSuccess)->getParent(), EntryBB);
  EXPECT_EQ(AtomicSuccess->getType(), Type::getInt1Ty(Ctx));
  EXPECT_EQ(cast<ExtractValueInst>(AtomicSuccess)->getNumIndices(), 1);
  EXPECT_EQ(cast<ExtractValueInst>(AtomicSuccess)->getIndices()[0], 1);
  EXPECT_EQ(cast<ExtractValueInst>(AtomicSuccess)->getAggregateOperand(),
            AtomicCmpxchgCmpxchgPair);
}

TEST_F(BuildBuiltinsTests, AtomicCmpxchg_Memorder_CABI) {
  Value *AtomicSuccess = nullptr;
  ASSERT_THAT_EXPECTED(emitAtomicCompareExchangeBuiltin(
                           /*AtomicPtr=*/PtrArg,
                           /*ExpectedPtr=*/ExpectedArg,
                           /*DesiredPtr=*/DesiredArg,
                           /*TypeOrSize=*/Builder.getInt32Ty(),
                           /*IsWeak*/ false,
                           /*IsVolatile=*/true,
                           /*SuccessMemorder=*/AtomicOrderingCABI::acq_rel,
                           /*FailureMemorder=*/AtomicOrderingCABI::relaxed,
                           /*Scope=*/SyncScope::System,
                           /*PrevPtr=*/nullptr,
                           /*Align=*/{}, /*Builder=*/Builder,
                           /*EmitOptions=*/AtomicEmitOptions(DL, TLI.get()),
                           /*Name=*/"atomic_cmpxchg"),
                       StoreResult(AtomicSuccess));
  EXPECT_FALSE(verifyModule(*M, &errs()));

  // clang-format off
  // %atomic_cmpxchg.cmpxchg.expected = load i32, ptr %expected_ptr, align 4
  // %atomic_cmpxchg.cmpxchg.desired = load i32, ptr %desired_ptr, align 4
  // %atomic_cmpxchg.cmpxchg.pair = cmpxchg volatile ptr %atomic_ptr, i32 %atomic_cmpxchg.cmpxchg.expected, i32 %atomic_cmpxchg.cmpxchg.desired acq_rel monotonic, align 1
  // %atomic_cmpxchg.cmpxchg.success = extractvalue { i32, i1 } %atomic_cmpxchg.cmpxchg.pair, 1
  // clang-format on

  // Follow use-def and load-store chains to discover instructions
  AtomicCmpXchgInst *AtomicCmpxchgCmpxchgPair =
      cast<AtomicCmpXchgInst>(getUniquePreviousStore(PtrArg, EntryBB));
  LoadInst *AtomicCmpxchgCmpxchgExpected =
      cast<LoadInst>(AtomicCmpxchgCmpxchgPair->getCompareOperand());
  LoadInst *AtomicCmpxchgCmpxchgDesired =
      cast<LoadInst>(AtomicCmpxchgCmpxchgPair->getNewValOperand());

  // %atomic_cmpxchg.cmpxchg.expected = load i32, ptr %expected_ptr, align 4
  EXPECT_EQ(AtomicCmpxchgCmpxchgExpected->getName(),
            "atomic_cmpxchg.cmpxchg.expected");
  EXPECT_EQ(AtomicCmpxchgCmpxchgExpected->getParent(), EntryBB);
  EXPECT_EQ(AtomicCmpxchgCmpxchgExpected->getType(), Type::getInt32Ty(Ctx));
  EXPECT_TRUE(AtomicCmpxchgCmpxchgExpected->isSimple());
  EXPECT_EQ(AtomicCmpxchgCmpxchgExpected->getPointerOperand(), ExpectedArg);

  // %atomic_cmpxchg.cmpxchg.desired = load i32, ptr %desired_ptr, align 4
  EXPECT_EQ(AtomicCmpxchgCmpxchgDesired->getName(),
            "atomic_cmpxchg.cmpxchg.desired");
  EXPECT_EQ(AtomicCmpxchgCmpxchgDesired->getParent(), EntryBB);
  EXPECT_EQ(AtomicCmpxchgCmpxchgDesired->getType(), Type::getInt32Ty(Ctx));
  EXPECT_TRUE(AtomicCmpxchgCmpxchgDesired->isSimple());
  EXPECT_EQ(AtomicCmpxchgCmpxchgDesired->getPointerOperand(), DesiredArg);

  // %atomic_cmpxchg.cmpxchg.pair = cmpxchg volatile ptr %atomic_ptr, i32 %...
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getName(), "atomic_cmpxchg.cmpxchg.pair");
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getParent(), EntryBB);
  EXPECT_TRUE(AtomicCmpxchgCmpxchgPair->isVolatile());
  EXPECT_FALSE(AtomicCmpxchgCmpxchgPair->isWeak());
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getSuccessOrdering(),
            AtomicOrdering::AcquireRelease);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getFailureOrdering(),
            AtomicOrdering::Monotonic);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getSyncScopeID(), SyncScope::System);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getAlign(), 1);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getPointerOperand(), PtrArg);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getCompareOperand(),
            AtomicCmpxchgCmpxchgExpected);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getNewValOperand(),
            AtomicCmpxchgCmpxchgDesired);

  // %atomic_cmpxchg.cmpxchg.success = extractvalue { i32, i1 } %atomic_cmp...
  EXPECT_EQ(AtomicSuccess->getName(), "atomic_cmpxchg.cmpxchg.success");
  EXPECT_EQ(cast<Instruction>(AtomicSuccess)->getParent(), EntryBB);
  EXPECT_EQ(AtomicSuccess->getType(), Type::getInt1Ty(Ctx));
  EXPECT_EQ(cast<ExtractValueInst>(AtomicSuccess)->getNumIndices(), 1);
  EXPECT_EQ(cast<ExtractValueInst>(AtomicSuccess)->getIndices()[0], 1);
  EXPECT_EQ(cast<ExtractValueInst>(AtomicSuccess)->getAggregateOperand(),
            AtomicCmpxchgCmpxchgPair);
}

TEST_F(BuildBuiltinsTests, AtomicCmpxchg_Switch) {
  Value *AtomicSuccess = nullptr;
  ASSERT_THAT_EXPECTED(emitAtomicCompareExchangeBuiltin(
                           /*AtomicPtr=*/PtrArg,
                           /*ExpectedPtr=*/ExpectedArg,
                           /*DesiredPtr=*/DesiredArg,
                           /*TypeOrSize=*/Builder.getInt32Ty(),
                           /*IsWeak*/ PredArg,
                           /*IsVolatile=*/true,
                           /*SuccessMemorder=*/MemorderArg,
                           /*FailureMemorder=*/MemorderArg,
                           /*Scope=*/SyncScope::System,
                           /*PrevPtr=*/nullptr,
                           /*Align=*/{}, /*Builder=*/Builder,
                           /*EmitOptions=*/AtomicEmitOptions(DL, TLI.get()),
                           /*Name=*/"atomic_cmpxchg"),
                       StoreResult(AtomicSuccess));
  BasicBlock *ExitBB = Builder.GetInsertBlock();
  EXPECT_FALSE(verifyModule(*M, &errs()));

  // clang-format off
  // entry:
  //   %atomic_cmpxchg.cmpxchg.expected = load i32, ptr %expected_ptr, align 4
  //   %atomic_cmpxchg.cmpxchg.desired = load i32, ptr %desired_ptr, align 4
  //   switch i1 %predarg, label %atomic_cmpxchg.cmpxchg.weak [
  //     i1 false, label %atomic_cmpxchg.cmpxchg.strong
  //   ]
  //
  // atomic_cmpxchg.cmpxchg.strong:                    ; preds = %entry
  //   switch i32 %memorderarg_success, label %atomic_cmpxchg.cmpxchg.monotonic [
  //     i32 1, label %atomic_cmpxchg.cmpxchg.acquire
  //     i32 2, label %atomic_cmpxchg.cmpxchg.acquire
  //     i32 3, label %atomic_cmpxchg.cmpxchg.release
  //     i32 4, label %atomic_cmpxchg.cmpxchg.acqrel
  //     i32 5, label %atomic_cmpxchg.cmpxchg.seqcst
  //   ]
  //
  // atomic_cmpxchg.cmpxchg.monotonic:                 ; preds = %atomic_cmpxchg.cmpxchg.strong
  //   switch i32 %memorderarg_success, label %atomic_cmpxchg.cmpxchg.monotonic_fail [
  //     i32 1, label %atomic_cmpxchg.cmpxchg.acquire_fail
  //     i32 2, label %atomic_cmpxchg.cmpxchg.acquire_fail
  //     i32 5, label %atomic_cmpxchg.cmpxchg.seqcst_fail
  //   ]
  //
  // atomic_cmpxchg.cmpxchg.monotonic_fail:            ; preds = %atomic_cmpxchg.cmpxchg.monotonic
  //   %atomic_cmpxchg.cmpxchg.pair = cmpxchg volatile ptr %atomic_ptr, i32 %atomic_cmpxchg.cmpxchg.expected, i32 %atomic_cmpxchg.cmpxchg.desired monotonic monotonic, align 1
  //   %atomic_cmpxchg.cmpxchg.success = extractvalue { i32, i1 } %atomic_cmpxchg.cmpxchg.pair, 1
  //   br label %atomic_cmpxchg.cmpxchg.failorder.continue
  //
  // atomic_cmpxchg.cmpxchg.acquire_fail:              ; preds = %atomic_cmpxchg.cmpxchg.monotonic, %atomic_cmpxchg.cmpxchg.monotonic
  //   %atomic_cmpxchg.cmpxchg.pair1 = cmpxchg volatile ptr %atomic_ptr, i32 %atomic_cmpxchg.cmpxchg.expected, i32 %atomic_cmpxchg.cmpxchg.desired monotonic acquire, align 1
  //   %atomic_cmpxchg.cmpxchg.success2 = extractvalue { i32, i1 } %atomic_cmpxchg.cmpxchg.pair1, 1
  //   br label %atomic_cmpxchg.cmpxchg.failorder.continue
  //
  // atomic_cmpxchg.cmpxchg.seqcst_fail:               ; preds = %atomic_cmpxchg.cmpxchg.monotonic
  //   %atomic_cmpxchg.cmpxchg.pair3 = cmpxchg volatile ptr %atomic_ptr, i32 %atomic_cmpxchg.cmpxchg.expected, i32 %atomic_cmpxchg.cmpxchg.desired monotonic seq_cst, align 1
  //   %atomic_cmpxchg.cmpxchg.success4 = extractvalue { i32, i1 } %atomic_cmpxchg.cmpxchg.pair3, 1
  //   br label %atomic_cmpxchg.cmpxchg.failorder.continue
  //
  // atomic_cmpxchg.cmpxchg.failorder.continue:        ; preds = %atomic_cmpxchg.cmpxchg.seqcst_fail, %atomic_cmpxchg.cmpxchg.acquire_fail, %atomic_cmpxchg.cmpxchg.monotonic_fail
  //   %atomic_cmpxchg.cmpxchg.failorder.success = phi i1 [ %atomic_cmpxchg.cmpxchg.success, %atomic_cmpxchg.cmpxchg.monotonic_fail ], [ %atomic_cmpxchg.cmpxchg.success2, %atomic_cmpxchg.cmpxchg.acquire_fail ], [ %atomic_cmpxchg.cmpxchg.success4, %atomic_cmpxchg.cmpxchg.seqcst_fail ]
  //   br label %atomic_cmpxchg.cmpxchg.memorder.continue
  //
  // atomic_cmpxchg.cmpxchg.acquire:                   ; preds = %atomic_cmpxchg.cmpxchg.strong, %atomic_cmpxchg.cmpxchg.strong
  //   switch i32 %memorderarg_success, label %atomic_cmpxchg.cmpxchg.monotonic_fail6 [
  //     i32 1, label %atomic_cmpxchg.cmpxchg.acquire_fail7
  //     i32 2, label %atomic_cmpxchg.cmpxchg.acquire_fail7
  //     i32 5, label %atomic_cmpxchg.cmpxchg.seqcst_fail8
  //   ]
  //
  // atomic_cmpxchg.cmpxchg.monotonic_fail6:           ; preds = %atomic_cmpxchg.cmpxchg.acquire
  //   %atomic_cmpxchg.cmpxchg.pair10 = cmpxchg volatile ptr %atomic_ptr, i32 %atomic_cmpxchg.cmpxchg.expected, i32 %atomic_cmpxchg.cmpxchg.desired acquire monotonic, align 1
  //   %atomic_cmpxchg.cmpxchg.success11 = extractvalue { i32, i1 } %atomic_cmpxchg.cmpxchg.pair10, 1
  //   br label %atomic_cmpxchg.cmpxchg.failorder.continue5
  //
  // atomic_cmpxchg.cmpxchg.acquire_fail7:             ; preds = %atomic_cmpxchg.cmpxchg.acquire, %atomic_cmpxchg.cmpxchg.acquire
  //   %atomic_cmpxchg.cmpxchg.pair12 = cmpxchg volatile ptr %atomic_ptr, i32 %atomic_cmpxchg.cmpxchg.expected, i32 %atomic_cmpxchg.cmpxchg.desired acquire acquire, align 1
  //   %atomic_cmpxchg.cmpxchg.success13 = extractvalue { i32, i1 } %atomic_cmpxchg.cmpxchg.pair12, 1
  //   br label %atomic_cmpxchg.cmpxchg.failorder.continue5
  //
  // atomic_cmpxchg.cmpxchg.seqcst_fail8:              ; preds = %atomic_cmpxchg.cmpxchg.acquire
  //   %atomic_cmpxchg.cmpxchg.pair14 = cmpxchg volatile ptr %atomic_ptr, i32 %atomic_cmpxchg.cmpxchg.expected, i32 %atomic_cmpxchg.cmpxchg.desired acquire seq_cst, align 1
  //   %atomic_cmpxchg.cmpxchg.success15 = extractvalue { i32, i1 } %atomic_cmpxchg.cmpxchg.pair14, 1
  //   br label %atomic_cmpxchg.cmpxchg.failorder.continue5
  //
  // atomic_cmpxchg.cmpxchg.failorder.continue5:       ; preds = %atomic_cmpxchg.cmpxchg.seqcst_fail8, %atomic_cmpxchg.cmpxchg.acquire_fail7, %atomic_cmpxchg.cmpxchg.monotonic_fail6
  //   %atomic_cmpxchg.cmpxchg.failorder.success9 = phi i1 [ %atomic_cmpxchg.cmpxchg.success11, %atomic_cmpxchg.cmpxchg.monotonic_fail6 ], [ %atomic_cmpxchg.cmpxchg.success13, %atomic_cmpxchg.cmpxchg.acquire_fail7 ], [ %atomic_cmpxchg.cmpxchg.success15, %atomic_cmpxchg.cmpxchg.seqcst_fail8 ]
  //   br label %atomic_cmpxchg.cmpxchg.memorder.continue
  //
  // atomic_cmpxchg.cmpxchg.release:                   ; preds = %atomic_cmpxchg.cmpxchg.strong
  //   switch i32 %memorderarg_success, label %atomic_cmpxchg.cmpxchg.monotonic_fail17 [
  //     i32 1, label %atomic_cmpxchg.cmpxchg.acquire_fail18
  //     i32 2, label %atomic_cmpxchg.cmpxchg.acquire_fail18
  //     i32 5, label %atomic_cmpxchg.cmpxchg.seqcst_fail19
  //   ]
  //
  // atomic_cmpxchg.cmpxchg.monotonic_fail17:          ; preds = %atomic_cmpxchg.cmpxchg.release
  //   %atomic_cmpxchg.cmpxchg.pair21 = cmpxchg volatile ptr %atomic_ptr, i32 %atomic_cmpxchg.cmpxchg.expected, i32 %atomic_cmpxchg.cmpxchg.desired release monotonic, align 1
  //   %atomic_cmpxchg.cmpxchg.success22 = extractvalue { i32, i1 } %atomic_cmpxchg.cmpxchg.pair21, 1
  //   br label %atomic_cmpxchg.cmpxchg.failorder.continue16
  //
  // atomic_cmpxchg.cmpxchg.acquire_fail18:            ; preds = %atomic_cmpxchg.cmpxchg.release, %atomic_cmpxchg.cmpxchg.release
  //   %atomic_cmpxchg.cmpxchg.pair23 = cmpxchg volatile ptr %atomic_ptr, i32 %atomic_cmpxchg.cmpxchg.expected, i32 %atomic_cmpxchg.cmpxchg.desired release acquire, align 1
  //   %atomic_cmpxchg.cmpxchg.success24 = extractvalue { i32, i1 } %atomic_cmpxchg.cmpxchg.pair23, 1
  //   br label %atomic_cmpxchg.cmpxchg.failorder.continue16
  //
  // atomic_cmpxchg.cmpxchg.seqcst_fail19:             ; preds = %atomic_cmpxchg.cmpxchg.release
  //   %atomic_cmpxchg.cmpxchg.pair25 = cmpxchg volatile ptr %atomic_ptr, i32 %atomic_cmpxchg.cmpxchg.expected, i32 %atomic_cmpxchg.cmpxchg.desired release seq_cst, align 1
  //   %atomic_cmpxchg.cmpxchg.success26 = extractvalue { i32, i1 } %atomic_cmpxchg.cmpxchg.pair25, 1
  //   br label %atomic_cmpxchg.cmpxchg.failorder.continue16
  //
  // atomic_cmpxchg.cmpxchg.failorder.continue16:      ; preds = %atomic_cmpxchg.cmpxchg.seqcst_fail19, %atomic_cmpxchg.cmpxchg.acquire_fail18, %atomic_cmpxchg.cmpxchg.monotonic_fail17
  //   %atomic_cmpxchg.cmpxchg.failorder.success20 = phi i1 [ %atomic_cmpxchg.cmpxchg.success22, %atomic_cmpxchg.cmpxchg.monotonic_fail17 ], [ %atomic_cmpxchg.cmpxchg.success24, %atomic_cmpxchg.cmpxchg.acquire_fail18 ], [ %atomic_cmpxchg.cmpxchg.success26, %atomic_cmpxchg.cmpxchg.seqcst_fail19 ]
  //   br label %atomic_cmpxchg.cmpxchg.memorder.continue
  //
  // atomic_cmpxchg.cmpxchg.acqrel:                    ; preds = %atomic_cmpxchg.cmpxchg.strong
  //   switch i32 %memorderarg_success, label %atomic_cmpxchg.cmpxchg.monotonic_fail28 [
  //     i32 1, label %atomic_cmpxchg.cmpxchg.acquire_fail29
  //     i32 2, label %atomic_cmpxchg.cmpxchg.acquire_fail29
  //     i32 5, label %atomic_cmpxchg.cmpxchg.seqcst_fail30
  //   ]
  //
  // atomic_cmpxchg.cmpxchg.monotonic_fail28:          ; preds = %atomic_cmpxchg.cmpxchg.acqrel
  //   %atomic_cmpxchg.cmpxchg.pair32 = cmpxchg volatile ptr %atomic_ptr, i32 %atomic_cmpxchg.cmpxchg.expected, i32 %atomic_cmpxchg.cmpxchg.desired acq_rel monotonic, align 1
  //   %atomic_cmpxchg.cmpxchg.success33 = extractvalue { i32, i1 } %atomic_cmpxchg.cmpxchg.pair32, 1
  //   br label %atomic_cmpxchg.cmpxchg.failorder.continue27
  //
  // atomic_cmpxchg.cmpxchg.acquire_fail29:            ; preds = %atomic_cmpxchg.cmpxchg.acqrel, %atomic_cmpxchg.cmpxchg.acqrel
  //   %atomic_cmpxchg.cmpxchg.pair34 = cmpxchg volatile ptr %atomic_ptr, i32 %atomic_cmpxchg.cmpxchg.expected, i32 %atomic_cmpxchg.cmpxchg.desired acq_rel acquire, align 1
  //   %atomic_cmpxchg.cmpxchg.success35 = extractvalue { i32, i1 } %atomic_cmpxchg.cmpxchg.pair34, 1
  //   br label %atomic_cmpxchg.cmpxchg.failorder.continue27
  //
  // atomic_cmpxchg.cmpxchg.seqcst_fail30:             ; preds = %atomic_cmpxchg.cmpxchg.acqrel
  //   %atomic_cmpxchg.cmpxchg.pair36 = cmpxchg volatile ptr %atomic_ptr, i32 %atomic_cmpxchg.cmpxchg.expected, i32 %atomic_cmpxchg.cmpxchg.desired acq_rel seq_cst, align 1
  //   %atomic_cmpxchg.cmpxchg.success37 = extractvalue { i32, i1 } %atomic_cmpxchg.cmpxchg.pair36, 1
  //   br label %atomic_cmpxchg.cmpxchg.failorder.continue27
  //
  // atomic_cmpxchg.cmpxchg.failorder.continue27:      ; preds = %atomic_cmpxchg.cmpxchg.seqcst_fail30, %atomic_cmpxchg.cmpxchg.acquire_fail29, %atomic_cmpxchg.cmpxchg.monotonic_fail28
  //   %atomic_cmpxchg.cmpxchg.failorder.success31 = phi i1 [ %atomic_cmpxchg.cmpxchg.success33, %atomic_cmpxchg.cmpxchg.monotonic_fail28 ], [ %atomic_cmpxchg.cmpxchg.success35, %atomic_cmpxchg.cmpxchg.acquire_fail29 ], [ %atomic_cmpxchg.cmpxchg.success37, %atomic_cmpxchg.cmpxchg.seqcst_fail30 ]
  //   br label %atomic_cmpxchg.cmpxchg.memorder.continue
  //
  // atomic_cmpxchg.cmpxchg.seqcst:                    ; preds = %atomic_cmpxchg.cmpxchg.strong
  //   switch i32 %memorderarg_success, label %atomic_cmpxchg.cmpxchg.monotonic_fail39 [
  //     i32 1, label %atomic_cmpxchg.cmpxchg.acquire_fail40
  //     i32 2, label %atomic_cmpxchg.cmpxchg.acquire_fail40
  //     i32 5, label %atomic_cmpxchg.cmpxchg.seqcst_fail41
  //   ]
  //
  // atomic_cmpxchg.cmpxchg.monotonic_fail39:          ; preds = %atomic_cmpxchg.cmpxchg.seqcst
  //   %atomic_cmpxchg.cmpxchg.pair43 = cmpxchg volatile ptr %atomic_ptr, i32 %atomic_cmpxchg.cmpxchg.expected, i32 %atomic_cmpxchg.cmpxchg.desired seq_cst monotonic, align 1
  //   %atomic_cmpxchg.cmpxchg.success44 = extractvalue { i32, i1 } %atomic_cmpxchg.cmpxchg.pair43, 1
  //   br label %atomic_cmpxchg.cmpxchg.failorder.continue38
  //
  // atomic_cmpxchg.cmpxchg.acquire_fail40:            ; preds = %atomic_cmpxchg.cmpxchg.seqcst, %atomic_cmpxchg.cmpxchg.seqcst
  //   %atomic_cmpxchg.cmpxchg.pair45 = cmpxchg volatile ptr %atomic_ptr, i32 %atomic_cmpxchg.cmpxchg.expected, i32 %atomic_cmpxchg.cmpxchg.desired seq_cst acquire, align 1
  //   %atomic_cmpxchg.cmpxchg.success46 = extractvalue { i32, i1 } %atomic_cmpxchg.cmpxchg.pair45, 1
  //   br label %atomic_cmpxchg.cmpxchg.failorder.continue38
  //
  // atomic_cmpxchg.cmpxchg.seqcst_fail41:             ; preds = %atomic_cmpxchg.cmpxchg.seqcst
  //   %atomic_cmpxchg.cmpxchg.pair47 = cmpxchg volatile ptr %atomic_ptr, i32 %atomic_cmpxchg.cmpxchg.expected, i32 %atomic_cmpxchg.cmpxchg.desired seq_cst seq_cst, align 1
  //   %atomic_cmpxchg.cmpxchg.success48 = extractvalue { i32, i1 } %atomic_cmpxchg.cmpxchg.pair47, 1
  //   br label %atomic_cmpxchg.cmpxchg.failorder.continue38
  //
  // atomic_cmpxchg.cmpxchg.failorder.continue38:      ; preds = %atomic_cmpxchg.cmpxchg.seqcst_fail41, %atomic_cmpxchg.cmpxchg.acquire_fail40, %atomic_cmpxchg.cmpxchg.monotonic_fail39
  //   %atomic_cmpxchg.cmpxchg.failorder.success42 = phi i1 [ %atomic_cmpxchg.cmpxchg.success44, %atomic_cmpxchg.cmpxchg.monotonic_fail39 ], [ %atomic_cmpxchg.cmpxchg.success46, %atomic_cmpxchg.cmpxchg.acquire_fail40 ], [ %atomic_cmpxchg.cmpxchg.success48, %atomic_cmpxchg.cmpxchg.seqcst_fail41 ]
  //   br label %atomic_cmpxchg.cmpxchg.memorder.continue
  //
  // atomic_cmpxchg.cmpxchg.memorder.continue:         ; preds = %atomic_cmpxchg.cmpxchg.failorder.continue38, %atomic_cmpxchg.cmpxchg.failorder.continue27, %atomic_cmpxchg.cmpxchg.failorder.continue16, %atomic_cmpxchg.cmpxchg.failorder.continue5, %atomic_cmpxchg.cmpxchg.failorder.continue
  //   %atomic_cmpxchg.cmpxchg.memorder.success = phi i1 [ %atomic_cmpxchg.cmpxchg.failorder.success, %atomic_cmpxchg.cmpxchg.failorder.continue ], [ %atomic_cmpxchg.cmpxchg.failorder.success9, %atomic_cmpxchg.cmpxchg.failorder.continue5 ], [ %atomic_cmpxchg.cmpxchg.failorder.success20, %atomic_cmpxchg.cmpxchg.failorder.continue16 ], [ %atomic_cmpxchg.cmpxchg.failorder.success31, %atomic_cmpxchg.cmpxchg.failorder.continue27 ], [ %atomic_cmpxchg.cmpxchg.failorder.success42, %atomic_cmpxchg.cmpxchg.failorder.continue38 ]
  //   br label %atomic_cmpxchg.cmpxchg.weak.continue
  //
  // atomic_cmpxchg.cmpxchg.weak:                      ; preds = %entry
  //   switch i32 %memorderarg_success, label %atomic_cmpxchg.cmpxchg.monotonic50 [
  //     i32 1, label %atomic_cmpxchg.cmpxchg.acquire51
  //     i32 2, label %atomic_cmpxchg.cmpxchg.acquire51
  //     i32 3, label %atomic_cmpxchg.cmpxchg.release52
  //     i32 4, label %atomic_cmpxchg.cmpxchg.acqrel53
  //     i32 5, label %atomic_cmpxchg.cmpxchg.seqcst54
  //   ]
  //
  // atomic_cmpxchg.cmpxchg.monotonic50:               ; preds = %atomic_cmpxchg.cmpxchg.weak
  //   switch i32 %memorderarg_success, label %atomic_cmpxchg.cmpxchg.monotonic_fail57 [
  //     i32 1, label %atomic_cmpxchg.cmpxchg.acquire_fail58
  //     i32 2, label %atomic_cmpxchg.cmpxchg.acquire_fail58
  //     i32 5, label %atomic_cmpxchg.cmpxchg.seqcst_fail59
  //   ]
  //
  // atomic_cmpxchg.cmpxchg.monotonic_fail57:          ; preds = %atomic_cmpxchg.cmpxchg.monotonic50
  //   %atomic_cmpxchg.cmpxchg.pair61 = cmpxchg weak volatile ptr %atomic_ptr, i32 %atomic_cmpxchg.cmpxchg.expected, i32 %atomic_cmpxchg.cmpxchg.desired monotonic monotonic, align 1
  //   %atomic_cmpxchg.cmpxchg.success62 = extractvalue { i32, i1 } %atomic_cmpxchg.cmpxchg.pair61, 1
  //   br label %atomic_cmpxchg.cmpxchg.failorder.continue56
  //
  // atomic_cmpxchg.cmpxchg.acquire_fail58:            ; preds = %atomic_cmpxchg.cmpxchg.monotonic50, %atomic_cmpxchg.cmpxchg.monotonic50
  //   %atomic_cmpxchg.cmpxchg.pair63 = cmpxchg weak volatile ptr %atomic_ptr, i32 %atomic_cmpxchg.cmpxchg.expected, i32 %atomic_cmpxchg.cmpxchg.desired monotonic acquire, align 1
  //   %atomic_cmpxchg.cmpxchg.success64 = extractvalue { i32, i1 } %atomic_cmpxchg.cmpxchg.pair63, 1
  //   br label %atomic_cmpxchg.cmpxchg.failorder.continue56
  //
  // atomic_cmpxchg.cmpxchg.seqcst_fail59:             ; preds = %atomic_cmpxchg.cmpxchg.monotonic50
  //   %atomic_cmpxchg.cmpxchg.pair65 = cmpxchg weak volatile ptr %atomic_ptr, i32 %atomic_cmpxchg.cmpxchg.expected, i32 %atomic_cmpxchg.cmpxchg.desired monotonic seq_cst, align 1
  //   %atomic_cmpxchg.cmpxchg.success66 = extractvalue { i32, i1 } %atomic_cmpxchg.cmpxchg.pair65, 1
  //   br label %atomic_cmpxchg.cmpxchg.failorder.continue56
  //
  // atomic_cmpxchg.cmpxchg.failorder.continue56:      ; preds = %atomic_cmpxchg.cmpxchg.seqcst_fail59, %atomic_cmpxchg.cmpxchg.acquire_fail58, %atomic_cmpxchg.cmpxchg.monotonic_fail57
  //   %atomic_cmpxchg.cmpxchg.failorder.success60 = phi i1 [ %atomic_cmpxchg.cmpxchg.success62, %atomic_cmpxchg.cmpxchg.monotonic_fail57 ], [ %atomic_cmpxchg.cmpxchg.success64, %atomic_cmpxchg.cmpxchg.acquire_fail58 ], [ %atomic_cmpxchg.cmpxchg.success66, %atomic_cmpxchg.cmpxchg.seqcst_fail59 ]
  //   br label %atomic_cmpxchg.cmpxchg.memorder.continue49
  //
  // atomic_cmpxchg.cmpxchg.acquire51:                 ; preds = %atomic_cmpxchg.cmpxchg.weak, %atomic_cmpxchg.cmpxchg.weak
  //   switch i32 %memorderarg_success, label %atomic_cmpxchg.cmpxchg.monotonic_fail68 [
  //     i32 1, label %atomic_cmpxchg.cmpxchg.acquire_fail69
  //     i32 2, label %atomic_cmpxchg.cmpxchg.acquire_fail69
  //     i32 5, label %atomic_cmpxchg.cmpxchg.seqcst_fail70
  //   ]
  //
  // atomic_cmpxchg.cmpxchg.monotonic_fail68:          ; preds = %atomic_cmpxchg.cmpxchg.acquire51
  //   %atomic_cmpxchg.cmpxchg.pair72 = cmpxchg weak volatile ptr %atomic_ptr, i32 %atomic_cmpxchg.cmpxchg.expected, i32 %atomic_cmpxchg.cmpxchg.desired acquire monotonic, align 1
  //   %atomic_cmpxchg.cmpxchg.success73 = extractvalue { i32, i1 } %atomic_cmpxchg.cmpxchg.pair72, 1
  //   br label %atomic_cmpxchg.cmpxchg.failorder.continue67
  //
  // atomic_cmpxchg.cmpxchg.acquire_fail69:            ; preds = %atomic_cmpxchg.cmpxchg.acquire51, %atomic_cmpxchg.cmpxchg.acquire51
  //   %atomic_cmpxchg.cmpxchg.pair74 = cmpxchg weak volatile ptr %atomic_ptr, i32 %atomic_cmpxchg.cmpxchg.expected, i32 %atomic_cmpxchg.cmpxchg.desired acquire acquire, align 1
  //   %atomic_cmpxchg.cmpxchg.success75 = extractvalue { i32, i1 } %atomic_cmpxchg.cmpxchg.pair74, 1
  //   br label %atomic_cmpxchg.cmpxchg.failorder.continue67
  //
  // atomic_cmpxchg.cmpxchg.seqcst_fail70:             ; preds = %atomic_cmpxchg.cmpxchg.acquire51
  //   %atomic_cmpxchg.cmpxchg.pair76 = cmpxchg weak volatile ptr %atomic_ptr, i32 %atomic_cmpxchg.cmpxchg.expected, i32 %atomic_cmpxchg.cmpxchg.desired acquire seq_cst, align 1
  //   %atomic_cmpxchg.cmpxchg.success77 = extractvalue { i32, i1 } %atomic_cmpxchg.cmpxchg.pair76, 1
  //   br label %atomic_cmpxchg.cmpxchg.failorder.continue67
  //
  // atomic_cmpxchg.cmpxchg.failorder.continue67:      ; preds = %atomic_cmpxchg.cmpxchg.seqcst_fail70, %atomic_cmpxchg.cmpxchg.acquire_fail69, %atomic_cmpxchg.cmpxchg.monotonic_fail68
  //   %atomic_cmpxchg.cmpxchg.failorder.success71 = phi i1 [ %atomic_cmpxchg.cmpxchg.success73, %atomic_cmpxchg.cmpxchg.monotonic_fail68 ], [ %atomic_cmpxchg.cmpxchg.success75, %atomic_cmpxchg.cmpxchg.acquire_fail69 ], [ %atomic_cmpxchg.cmpxchg.success77, %atomic_cmpxchg.cmpxchg.seqcst_fail70 ]
  //   br label %atomic_cmpxchg.cmpxchg.memorder.continue49
  //
  // atomic_cmpxchg.cmpxchg.release52:                 ; preds = %atomic_cmpxchg.cmpxchg.weak
  //   switch i32 %memorderarg_success, label %atomic_cmpxchg.cmpxchg.monotonic_fail79 [
  //     i32 1, label %atomic_cmpxchg.cmpxchg.acquire_fail80
  //     i32 2, label %atomic_cmpxchg.cmpxchg.acquire_fail80
  //     i32 5, label %atomic_cmpxchg.cmpxchg.seqcst_fail81
  //   ]
  //
  // atomic_cmpxchg.cmpxchg.monotonic_fail79:          ; preds = %atomic_cmpxchg.cmpxchg.release52
  //   %atomic_cmpxchg.cmpxchg.pair83 = cmpxchg weak volatile ptr %atomic_ptr, i32 %atomic_cmpxchg.cmpxchg.expected, i32 %atomic_cmpxchg.cmpxchg.desired release monotonic, align 1
  //   %atomic_cmpxchg.cmpxchg.success84 = extractvalue { i32, i1 } %atomic_cmpxchg.cmpxchg.pair83, 1
  //   br label %atomic_cmpxchg.cmpxchg.failorder.continue78
  //
  // atomic_cmpxchg.cmpxchg.acquire_fail80:            ; preds = %atomic_cmpxchg.cmpxchg.release52, %atomic_cmpxchg.cmpxchg.release52
  //   %atomic_cmpxchg.cmpxchg.pair85 = cmpxchg weak volatile ptr %atomic_ptr, i32 %atomic_cmpxchg.cmpxchg.expected, i32 %atomic_cmpxchg.cmpxchg.desired release acquire, align 1
  //   %atomic_cmpxchg.cmpxchg.success86 = extractvalue { i32, i1 } %atomic_cmpxchg.cmpxchg.pair85, 1
  //   br label %atomic_cmpxchg.cmpxchg.failorder.continue78
  //
  // atomic_cmpxchg.cmpxchg.seqcst_fail81:             ; preds = %atomic_cmpxchg.cmpxchg.release52
  //   %atomic_cmpxchg.cmpxchg.pair87 = cmpxchg weak volatile ptr %atomic_ptr, i32 %atomic_cmpxchg.cmpxchg.expected, i32 %atomic_cmpxchg.cmpxchg.desired release seq_cst, align 1
  //   %atomic_cmpxchg.cmpxchg.success88 = extractvalue { i32, i1 } %atomic_cmpxchg.cmpxchg.pair87, 1
  //   br label %atomic_cmpxchg.cmpxchg.failorder.continue78
  //
  // atomic_cmpxchg.cmpxchg.failorder.continue78:      ; preds = %atomic_cmpxchg.cmpxchg.seqcst_fail81, %atomic_cmpxchg.cmpxchg.acquire_fail80, %atomic_cmpxchg.cmpxchg.monotonic_fail79
  //   %atomic_cmpxchg.cmpxchg.failorder.success82 = phi i1 [ %atomic_cmpxchg.cmpxchg.success84, %atomic_cmpxchg.cmpxchg.monotonic_fail79 ], [ %atomic_cmpxchg.cmpxchg.success86, %atomic_cmpxchg.cmpxchg.acquire_fail80 ], [ %atomic_cmpxchg.cmpxchg.success88, %atomic_cmpxchg.cmpxchg.seqcst_fail81 ]
  //   br label %atomic_cmpxchg.cmpxchg.memorder.continue49
  //
  // atomic_cmpxchg.cmpxchg.acqrel53:                  ; preds = %atomic_cmpxchg.cmpxchg.weak
  //   switch i32 %memorderarg_success, label %atomic_cmpxchg.cmpxchg.monotonic_fail90 [
  //     i32 1, label %atomic_cmpxchg.cmpxchg.acquire_fail91
  //     i32 2, label %atomic_cmpxchg.cmpxchg.acquire_fail91
  //     i32 5, label %atomic_cmpxchg.cmpxchg.seqcst_fail92
  //   ]
  //
  // atomic_cmpxchg.cmpxchg.monotonic_fail90:          ; preds = %atomic_cmpxchg.cmpxchg.acqrel53
  //   %atomic_cmpxchg.cmpxchg.pair94 = cmpxchg weak volatile ptr %atomic_ptr, i32 %atomic_cmpxchg.cmpxchg.expected, i32 %atomic_cmpxchg.cmpxchg.desired acq_rel monotonic, align 1
  //   %atomic_cmpxchg.cmpxchg.success95 = extractvalue { i32, i1 } %atomic_cmpxchg.cmpxchg.pair94, 1
  //   br label %atomic_cmpxchg.cmpxchg.failorder.continue89
  //
  // atomic_cmpxchg.cmpxchg.acquire_fail91:            ; preds = %atomic_cmpxchg.cmpxchg.acqrel53, %atomic_cmpxchg.cmpxchg.acqrel53
  //   %atomic_cmpxchg.cmpxchg.pair96 = cmpxchg weak volatile ptr %atomic_ptr, i32 %atomic_cmpxchg.cmpxchg.expected, i32 %atomic_cmpxchg.cmpxchg.desired acq_rel acquire, align 1
  //   %atomic_cmpxchg.cmpxchg.success97 = extractvalue { i32, i1 } %atomic_cmpxchg.cmpxchg.pair96, 1
  //   br label %atomic_cmpxchg.cmpxchg.failorder.continue89
  //
  // atomic_cmpxchg.cmpxchg.seqcst_fail92:             ; preds = %atomic_cmpxchg.cmpxchg.acqrel53
  //   %atomic_cmpxchg.cmpxchg.pair98 = cmpxchg weak volatile ptr %atomic_ptr, i32 %atomic_cmpxchg.cmpxchg.expected, i32 %atomic_cmpxchg.cmpxchg.desired acq_rel seq_cst, align 1
  //   %atomic_cmpxchg.cmpxchg.success99 = extractvalue { i32, i1 } %atomic_cmpxchg.cmpxchg.pair98, 1
  //   br label %atomic_cmpxchg.cmpxchg.failorder.continue89
  //
  // atomic_cmpxchg.cmpxchg.failorder.continue89:      ; preds = %atomic_cmpxchg.cmpxchg.seqcst_fail92, %atomic_cmpxchg.cmpxchg.acquire_fail91, %atomic_cmpxchg.cmpxchg.monotonic_fail90
  //   %atomic_cmpxchg.cmpxchg.failorder.success93 = phi i1 [ %atomic_cmpxchg.cmpxchg.success95, %atomic_cmpxchg.cmpxchg.monotonic_fail90 ], [ %atomic_cmpxchg.cmpxchg.success97, %atomic_cmpxchg.cmpxchg.acquire_fail91 ], [ %atomic_cmpxchg.cmpxchg.success99, %atomic_cmpxchg.cmpxchg.seqcst_fail92 ]
  //   br label %atomic_cmpxchg.cmpxchg.memorder.continue49
  //
  // atomic_cmpxchg.cmpxchg.seqcst54:                  ; preds = %atomic_cmpxchg.cmpxchg.weak
  //   switch i32 %memorderarg_success, label %atomic_cmpxchg.cmpxchg.monotonic_fail101 [
  //     i32 1, label %atomic_cmpxchg.cmpxchg.acquire_fail102
  //     i32 2, label %atomic_cmpxchg.cmpxchg.acquire_fail102
  //     i32 5, label %atomic_cmpxchg.cmpxchg.seqcst_fail103
  //   ]
  //
  // atomic_cmpxchg.cmpxchg.monotonic_fail101:         ; preds = %atomic_cmpxchg.cmpxchg.seqcst54
  //   %atomic_cmpxchg.cmpxchg.pair105 = cmpxchg weak volatile ptr %atomic_ptr, i32 %atomic_cmpxchg.cmpxchg.expected, i32 %atomic_cmpxchg.cmpxchg.desired seq_cst monotonic, align 1
  //   %atomic_cmpxchg.cmpxchg.success106 = extractvalue { i32, i1 } %atomic_cmpxchg.cmpxchg.pair105, 1
  //   br label %atomic_cmpxchg.cmpxchg.failorder.continue100
  //
  // atomic_cmpxchg.cmpxchg.acquire_fail102:           ; preds = %atomic_cmpxchg.cmpxchg.seqcst54, %atomic_cmpxchg.cmpxchg.seqcst54
  //   %atomic_cmpxchg.cmpxchg.pair107 = cmpxchg weak volatile ptr %atomic_ptr, i32 %atomic_cmpxchg.cmpxchg.expected, i32 %atomic_cmpxchg.cmpxchg.desired seq_cst acquire, align 1
  //   %atomic_cmpxchg.cmpxchg.success108 = extractvalue { i32, i1 } %atomic_cmpxchg.cmpxchg.pair107, 1
  //   br label %atomic_cmpxchg.cmpxchg.failorder.continue100
  //
  // atomic_cmpxchg.cmpxchg.seqcst_fail103:            ; preds = %atomic_cmpxchg.cmpxchg.seqcst54
  //   %atomic_cmpxchg.cmpxchg.pair109 = cmpxchg weak volatile ptr %atomic_ptr, i32 %atomic_cmpxchg.cmpxchg.expected, i32 %atomic_cmpxchg.cmpxchg.desired seq_cst seq_cst, align 1
  //   %atomic_cmpxchg.cmpxchg.success110 = extractvalue { i32, i1 } %atomic_cmpxchg.cmpxchg.pair109, 1
  //   br label %atomic_cmpxchg.cmpxchg.failorder.continue100
  //
  // atomic_cmpxchg.cmpxchg.failorder.continue100:     ; preds = %atomic_cmpxchg.cmpxchg.seqcst_fail103, %atomic_cmpxchg.cmpxchg.acquire_fail102, %atomic_cmpxchg.cmpxchg.monotonic_fail101
  //   %atomic_cmpxchg.cmpxchg.failorder.success104 = phi i1 [ %atomic_cmpxchg.cmpxchg.success106, %atomic_cmpxchg.cmpxchg.monotonic_fail101 ], [ %atomic_cmpxchg.cmpxchg.success108, %atomic_cmpxchg.cmpxchg.acquire_fail102 ], [ %atomic_cmpxchg.cmpxchg.success110, %atomic_cmpxchg.cmpxchg.seqcst_fail103 ]
  //   br label %atomic_cmpxchg.cmpxchg.memorder.continue49
  //
  // atomic_cmpxchg.cmpxchg.memorder.continue49:       ; preds = %atomic_cmpxchg.cmpxchg.failorder.continue100, %atomic_cmpxchg.cmpxchg.failorder.continue89, %atomic_cmpxchg.cmpxchg.failorder.continue78, %atomic_cmpxchg.cmpxchg.failorder.continue67, %atomic_cmpxchg.cmpxchg.failorder.continue56
  //   %atomic_cmpxchg.cmpxchg.memorder.success55 = phi i1 [ %atomic_cmpxchg.cmpxchg.failorder.success60, %atomic_cmpxchg.cmpxchg.failorder.continue56 ], [ %atomic_cmpxchg.cmpxchg.failorder.success71, %atomic_cmpxchg.cmpxchg.failorder.continue67 ], [ %atomic_cmpxchg.cmpxchg.failorder.success82, %atomic_cmpxchg.cmpxchg.failorder.continue78 ], [ %atomic_cmpxchg.cmpxchg.failorder.success93, %atomic_cmpxchg.cmpxchg.failorder.continue89 ], [ %atomic_cmpxchg.cmpxchg.failorder.success104, %atomic_cmpxchg.cmpxchg.failorder.continue100 ]
  //   br label %atomic_cmpxchg.cmpxchg.weak.continue
  //
  // atomic_cmpxchg.cmpxchg.weak.continue:             ; preds = %atomic_cmpxchg.cmpxchg.memorder.continue49, %atomic_cmpxchg.cmpxchg.memorder.continue
  //   %atomic_cmpxchg.cmpxchg.isweak.success = phi i1 [ %atomic_cmpxchg.cmpxchg.memorder.success, %atomic_cmpxchg.cmpxchg.memorder.continue ], [ %atomic_cmpxchg.cmpxchg.memorder.success55, %atomic_cmpxchg.cmpxchg.memorder.continue49 ]
  //   ret void
  // clang-format on

  // Discover control flow graph
  SwitchInst *Switch1 = cast<SwitchInst>(EntryBB->getTerminator());
  BasicBlock *AtomicCmpxchgCmpxchgStrong =
      cast<BasicBlock>(Switch1->getSuccessor(1));
  BasicBlock *AtomicCmpxchgCmpxchgWeak =
      cast<BasicBlock>(Switch1->getDefaultDest());
  SwitchInst *Switch2 =
      cast<SwitchInst>(AtomicCmpxchgCmpxchgStrong->getTerminator());
  BasicBlock *AtomicCmpxchgCmpxchgAcquire =
      cast<BasicBlock>(Switch2->getSuccessor(1));
  BasicBlock *AtomicCmpxchgCmpxchgRelease =
      cast<BasicBlock>(Switch2->getSuccessor(3));
  BasicBlock *AtomicCmpxchgCmpxchgAcqrel =
      cast<BasicBlock>(Switch2->getSuccessor(4));
  BasicBlock *AtomicCmpxchgCmpxchgSeqcst =
      cast<BasicBlock>(Switch2->getSuccessor(5));
  BasicBlock *AtomicCmpxchgCmpxchgMonotonic =
      cast<BasicBlock>(Switch2->getDefaultDest());
  SwitchInst *Switch3 =
      cast<SwitchInst>(AtomicCmpxchgCmpxchgMonotonic->getTerminator());
  BasicBlock *AtomicCmpxchgCmpxchgAcquireFail =
      cast<BasicBlock>(Switch3->getSuccessor(1));
  BasicBlock *AtomicCmpxchgCmpxchgSeqcstFail =
      cast<BasicBlock>(Switch3->getSuccessor(3));
  BasicBlock *AtomicCmpxchgCmpxchgMonotonicFail =
      cast<BasicBlock>(Switch3->getDefaultDest());
  BranchInst *Branch1 =
      cast<BranchInst>(AtomicCmpxchgCmpxchgMonotonicFail->getTerminator());
  BasicBlock *AtomicCmpxchgCmpxchgFailorderContinue =
      cast<BasicBlock>(AtomicCmpxchgCmpxchgMonotonicFail->getUniqueSuccessor());
  BranchInst *Branch2 =
      cast<BranchInst>(AtomicCmpxchgCmpxchgAcquireFail->getTerminator());
  BranchInst *Branch3 =
      cast<BranchInst>(AtomicCmpxchgCmpxchgSeqcstFail->getTerminator());
  BranchInst *Branch4 =
      cast<BranchInst>(AtomicCmpxchgCmpxchgFailorderContinue->getTerminator());
  BasicBlock *AtomicCmpxchgCmpxchgMemorderContinue = cast<BasicBlock>(
      AtomicCmpxchgCmpxchgFailorderContinue->getUniqueSuccessor());
  SwitchInst *Switch4 =
      cast<SwitchInst>(AtomicCmpxchgCmpxchgAcquire->getTerminator());
  BasicBlock *AtomicCmpxchgCmpxchgAcquireFail7 =
      cast<BasicBlock>(Switch4->getSuccessor(1));
  BasicBlock *AtomicCmpxchgCmpxchgSeqcstFail8 =
      cast<BasicBlock>(Switch4->getSuccessor(3));
  BasicBlock *AtomicCmpxchgCmpxchgMonotonicFail6 =
      cast<BasicBlock>(Switch4->getDefaultDest());
  BranchInst *Branch5 =
      cast<BranchInst>(AtomicCmpxchgCmpxchgMonotonicFail6->getTerminator());
  BasicBlock *AtomicCmpxchgCmpxchgFailorderContinue5 = cast<BasicBlock>(
      AtomicCmpxchgCmpxchgMonotonicFail6->getUniqueSuccessor());
  BranchInst *Branch6 =
      cast<BranchInst>(AtomicCmpxchgCmpxchgAcquireFail7->getTerminator());
  BranchInst *Branch7 =
      cast<BranchInst>(AtomicCmpxchgCmpxchgSeqcstFail8->getTerminator());
  BranchInst *Branch8 =
      cast<BranchInst>(AtomicCmpxchgCmpxchgFailorderContinue5->getTerminator());
  SwitchInst *Switch5 =
      cast<SwitchInst>(AtomicCmpxchgCmpxchgRelease->getTerminator());
  BasicBlock *AtomicCmpxchgCmpxchgAcquireFail18 =
      cast<BasicBlock>(Switch5->getSuccessor(1));
  BasicBlock *AtomicCmpxchgCmpxchgSeqcstFail19 =
      cast<BasicBlock>(Switch5->getSuccessor(3));
  BasicBlock *AtomicCmpxchgCmpxchgMonotonicFail17 =
      cast<BasicBlock>(Switch5->getDefaultDest());
  BranchInst *Branch9 =
      cast<BranchInst>(AtomicCmpxchgCmpxchgMonotonicFail17->getTerminator());
  BasicBlock *AtomicCmpxchgCmpxchgFailorderContinue16 = cast<BasicBlock>(
      AtomicCmpxchgCmpxchgMonotonicFail17->getUniqueSuccessor());
  BranchInst *Branch10 =
      cast<BranchInst>(AtomicCmpxchgCmpxchgAcquireFail18->getTerminator());
  BranchInst *Branch11 =
      cast<BranchInst>(AtomicCmpxchgCmpxchgSeqcstFail19->getTerminator());
  BranchInst *Branch12 = cast<BranchInst>(
      AtomicCmpxchgCmpxchgFailorderContinue16->getTerminator());
  SwitchInst *Switch6 =
      cast<SwitchInst>(AtomicCmpxchgCmpxchgAcqrel->getTerminator());
  BasicBlock *AtomicCmpxchgCmpxchgAcquireFail29 =
      cast<BasicBlock>(Switch6->getSuccessor(1));
  BasicBlock *AtomicCmpxchgCmpxchgSeqcstFail30 =
      cast<BasicBlock>(Switch6->getSuccessor(3));
  BasicBlock *AtomicCmpxchgCmpxchgMonotonicFail28 =
      cast<BasicBlock>(Switch6->getDefaultDest());
  BranchInst *Branch13 =
      cast<BranchInst>(AtomicCmpxchgCmpxchgMonotonicFail28->getTerminator());
  BasicBlock *AtomicCmpxchgCmpxchgFailorderContinue27 = cast<BasicBlock>(
      AtomicCmpxchgCmpxchgMonotonicFail28->getUniqueSuccessor());
  BranchInst *Branch14 =
      cast<BranchInst>(AtomicCmpxchgCmpxchgAcquireFail29->getTerminator());
  BranchInst *Branch15 =
      cast<BranchInst>(AtomicCmpxchgCmpxchgSeqcstFail30->getTerminator());
  BranchInst *Branch16 = cast<BranchInst>(
      AtomicCmpxchgCmpxchgFailorderContinue27->getTerminator());
  SwitchInst *Switch7 =
      cast<SwitchInst>(AtomicCmpxchgCmpxchgSeqcst->getTerminator());
  BasicBlock *AtomicCmpxchgCmpxchgAcquireFail40 =
      cast<BasicBlock>(Switch7->getSuccessor(1));
  BasicBlock *AtomicCmpxchgCmpxchgSeqcstFail41 =
      cast<BasicBlock>(Switch7->getSuccessor(3));
  BasicBlock *AtomicCmpxchgCmpxchgMonotonicFail39 =
      cast<BasicBlock>(Switch7->getDefaultDest());
  BranchInst *Branch17 =
      cast<BranchInst>(AtomicCmpxchgCmpxchgMonotonicFail39->getTerminator());
  BasicBlock *AtomicCmpxchgCmpxchgFailorderContinue38 = cast<BasicBlock>(
      AtomicCmpxchgCmpxchgMonotonicFail39->getUniqueSuccessor());
  BranchInst *Branch18 =
      cast<BranchInst>(AtomicCmpxchgCmpxchgAcquireFail40->getTerminator());
  BranchInst *Branch19 =
      cast<BranchInst>(AtomicCmpxchgCmpxchgSeqcstFail41->getTerminator());
  BranchInst *Branch20 = cast<BranchInst>(
      AtomicCmpxchgCmpxchgFailorderContinue38->getTerminator());
  BranchInst *Branch21 =
      cast<BranchInst>(AtomicCmpxchgCmpxchgMemorderContinue->getTerminator());
  SwitchInst *Switch8 =
      cast<SwitchInst>(AtomicCmpxchgCmpxchgWeak->getTerminator());
  BasicBlock *AtomicCmpxchgCmpxchgAcquire51 =
      cast<BasicBlock>(Switch8->getSuccessor(1));
  BasicBlock *AtomicCmpxchgCmpxchgRelease52 =
      cast<BasicBlock>(Switch8->getSuccessor(3));
  BasicBlock *AtomicCmpxchgCmpxchgAcqrel53 =
      cast<BasicBlock>(Switch8->getSuccessor(4));
  BasicBlock *AtomicCmpxchgCmpxchgSeqcst54 =
      cast<BasicBlock>(Switch8->getSuccessor(5));
  BasicBlock *AtomicCmpxchgCmpxchgMonotonic50 =
      cast<BasicBlock>(Switch8->getDefaultDest());
  SwitchInst *Switch9 =
      cast<SwitchInst>(AtomicCmpxchgCmpxchgMonotonic50->getTerminator());
  BasicBlock *AtomicCmpxchgCmpxchgAcquireFail58 =
      cast<BasicBlock>(Switch9->getSuccessor(1));
  BasicBlock *AtomicCmpxchgCmpxchgSeqcstFail59 =
      cast<BasicBlock>(Switch9->getSuccessor(3));
  BasicBlock *AtomicCmpxchgCmpxchgMonotonicFail57 =
      cast<BasicBlock>(Switch9->getDefaultDest());
  BranchInst *Branch22 =
      cast<BranchInst>(AtomicCmpxchgCmpxchgMonotonicFail57->getTerminator());
  BasicBlock *AtomicCmpxchgCmpxchgFailorderContinue56 = cast<BasicBlock>(
      AtomicCmpxchgCmpxchgMonotonicFail57->getUniqueSuccessor());
  BranchInst *Branch23 =
      cast<BranchInst>(AtomicCmpxchgCmpxchgAcquireFail58->getTerminator());
  BranchInst *Branch24 =
      cast<BranchInst>(AtomicCmpxchgCmpxchgSeqcstFail59->getTerminator());
  BranchInst *Branch25 = cast<BranchInst>(
      AtomicCmpxchgCmpxchgFailorderContinue56->getTerminator());
  BasicBlock *AtomicCmpxchgCmpxchgMemorderContinue49 = cast<BasicBlock>(
      AtomicCmpxchgCmpxchgFailorderContinue56->getUniqueSuccessor());
  SwitchInst *Switch10 =
      cast<SwitchInst>(AtomicCmpxchgCmpxchgAcquire51->getTerminator());
  BasicBlock *AtomicCmpxchgCmpxchgAcquireFail69 =
      cast<BasicBlock>(Switch10->getSuccessor(1));
  BasicBlock *AtomicCmpxchgCmpxchgSeqcstFail70 =
      cast<BasicBlock>(Switch10->getSuccessor(3));
  BasicBlock *AtomicCmpxchgCmpxchgMonotonicFail68 =
      cast<BasicBlock>(Switch10->getDefaultDest());
  BranchInst *Branch26 =
      cast<BranchInst>(AtomicCmpxchgCmpxchgMonotonicFail68->getTerminator());
  BasicBlock *AtomicCmpxchgCmpxchgFailorderContinue67 = cast<BasicBlock>(
      AtomicCmpxchgCmpxchgMonotonicFail68->getUniqueSuccessor());
  BranchInst *Branch27 =
      cast<BranchInst>(AtomicCmpxchgCmpxchgAcquireFail69->getTerminator());
  BranchInst *Branch28 =
      cast<BranchInst>(AtomicCmpxchgCmpxchgSeqcstFail70->getTerminator());
  BranchInst *Branch29 = cast<BranchInst>(
      AtomicCmpxchgCmpxchgFailorderContinue67->getTerminator());
  SwitchInst *Switch11 =
      cast<SwitchInst>(AtomicCmpxchgCmpxchgRelease52->getTerminator());
  BasicBlock *AtomicCmpxchgCmpxchgAcquireFail80 =
      cast<BasicBlock>(Switch11->getSuccessor(1));
  BasicBlock *AtomicCmpxchgCmpxchgSeqcstFail81 =
      cast<BasicBlock>(Switch11->getSuccessor(3));
  BasicBlock *AtomicCmpxchgCmpxchgMonotonicFail79 =
      cast<BasicBlock>(Switch11->getDefaultDest());
  BranchInst *Branch30 =
      cast<BranchInst>(AtomicCmpxchgCmpxchgMonotonicFail79->getTerminator());
  BasicBlock *AtomicCmpxchgCmpxchgFailorderContinue78 = cast<BasicBlock>(
      AtomicCmpxchgCmpxchgMonotonicFail79->getUniqueSuccessor());
  BranchInst *Branch31 =
      cast<BranchInst>(AtomicCmpxchgCmpxchgAcquireFail80->getTerminator());
  BranchInst *Branch32 =
      cast<BranchInst>(AtomicCmpxchgCmpxchgSeqcstFail81->getTerminator());
  BranchInst *Branch33 = cast<BranchInst>(
      AtomicCmpxchgCmpxchgFailorderContinue78->getTerminator());
  SwitchInst *Switch12 =
      cast<SwitchInst>(AtomicCmpxchgCmpxchgAcqrel53->getTerminator());
  BasicBlock *AtomicCmpxchgCmpxchgAcquireFail91 =
      cast<BasicBlock>(Switch12->getSuccessor(1));
  BasicBlock *AtomicCmpxchgCmpxchgSeqcstFail92 =
      cast<BasicBlock>(Switch12->getSuccessor(3));
  BasicBlock *AtomicCmpxchgCmpxchgMonotonicFail90 =
      cast<BasicBlock>(Switch12->getDefaultDest());
  BranchInst *Branch34 =
      cast<BranchInst>(AtomicCmpxchgCmpxchgMonotonicFail90->getTerminator());
  BasicBlock *AtomicCmpxchgCmpxchgFailorderContinue89 = cast<BasicBlock>(
      AtomicCmpxchgCmpxchgMonotonicFail90->getUniqueSuccessor());
  BranchInst *Branch35 =
      cast<BranchInst>(AtomicCmpxchgCmpxchgAcquireFail91->getTerminator());
  BranchInst *Branch36 =
      cast<BranchInst>(AtomicCmpxchgCmpxchgSeqcstFail92->getTerminator());
  BranchInst *Branch37 = cast<BranchInst>(
      AtomicCmpxchgCmpxchgFailorderContinue89->getTerminator());
  SwitchInst *Switch13 =
      cast<SwitchInst>(AtomicCmpxchgCmpxchgSeqcst54->getTerminator());
  BasicBlock *AtomicCmpxchgCmpxchgAcquireFail102 =
      cast<BasicBlock>(Switch13->getSuccessor(1));
  BasicBlock *AtomicCmpxchgCmpxchgSeqcstFail103 =
      cast<BasicBlock>(Switch13->getSuccessor(3));
  BasicBlock *AtomicCmpxchgCmpxchgMonotonicFail101 =
      cast<BasicBlock>(Switch13->getDefaultDest());
  BranchInst *Branch38 =
      cast<BranchInst>(AtomicCmpxchgCmpxchgMonotonicFail101->getTerminator());
  BasicBlock *AtomicCmpxchgCmpxchgFailorderContinue100 = cast<BasicBlock>(
      AtomicCmpxchgCmpxchgMonotonicFail101->getUniqueSuccessor());
  BranchInst *Branch39 =
      cast<BranchInst>(AtomicCmpxchgCmpxchgAcquireFail102->getTerminator());
  BranchInst *Branch40 =
      cast<BranchInst>(AtomicCmpxchgCmpxchgSeqcstFail103->getTerminator());
  BranchInst *Branch41 = cast<BranchInst>(
      AtomicCmpxchgCmpxchgFailorderContinue100->getTerminator());
  BranchInst *Branch42 =
      cast<BranchInst>(AtomicCmpxchgCmpxchgMemorderContinue49->getTerminator());
  ReturnInst *Return = cast<ReturnInst>(ExitBB->getTerminator());

  // Follow use-def and load-store chains to discover instructions
  AtomicCmpXchgInst *AtomicCmpxchgCmpxchgPair109 = cast<AtomicCmpXchgInst>(
      getUniquePreviousStore(PtrArg, AtomicCmpxchgCmpxchgSeqcstFail103));
  LoadInst *AtomicCmpxchgCmpxchgExpected =
      cast<LoadInst>(AtomicCmpxchgCmpxchgPair109->getCompareOperand());
  LoadInst *AtomicCmpxchgCmpxchgDesired =
      cast<LoadInst>(AtomicCmpxchgCmpxchgPair109->getNewValOperand());
  AtomicCmpXchgInst *AtomicCmpxchgCmpxchgPair105 = cast<AtomicCmpXchgInst>(
      getUniquePreviousStore(PtrArg, AtomicCmpxchgCmpxchgMonotonicFail101));
  AtomicCmpXchgInst *AtomicCmpxchgCmpxchgPair87 = cast<AtomicCmpXchgInst>(
      getUniquePreviousStore(PtrArg, AtomicCmpxchgCmpxchgSeqcstFail81));
  AtomicCmpXchgInst *AtomicCmpxchgCmpxchgPair1 = cast<AtomicCmpXchgInst>(
      getUniquePreviousStore(PtrArg, AtomicCmpxchgCmpxchgAcquireFail));
  AtomicCmpXchgInst *AtomicCmpxchgCmpxchgPair10 = cast<AtomicCmpXchgInst>(
      getUniquePreviousStore(PtrArg, AtomicCmpxchgCmpxchgMonotonicFail6));
  AtomicCmpXchgInst *AtomicCmpxchgCmpxchgPair12 = cast<AtomicCmpXchgInst>(
      getUniquePreviousStore(PtrArg, AtomicCmpxchgCmpxchgAcquireFail7));
  AtomicCmpXchgInst *AtomicCmpxchgCmpxchgPair94 = cast<AtomicCmpXchgInst>(
      getUniquePreviousStore(PtrArg, AtomicCmpxchgCmpxchgMonotonicFail90));
  AtomicCmpXchgInst *AtomicCmpxchgCmpxchgPair72 = cast<AtomicCmpXchgInst>(
      getUniquePreviousStore(PtrArg, AtomicCmpxchgCmpxchgMonotonicFail68));
  AtomicCmpXchgInst *AtomicCmpxchgCmpxchgPair = cast<AtomicCmpXchgInst>(
      getUniquePreviousStore(PtrArg, AtomicCmpxchgCmpxchgMonotonicFail));
  AtomicCmpXchgInst *AtomicCmpxchgCmpxchgPair23 = cast<AtomicCmpXchgInst>(
      getUniquePreviousStore(PtrArg, AtomicCmpxchgCmpxchgAcquireFail18));
  AtomicCmpXchgInst *AtomicCmpxchgCmpxchgPair3 = cast<AtomicCmpXchgInst>(
      getUniquePreviousStore(PtrArg, AtomicCmpxchgCmpxchgSeqcstFail));
  AtomicCmpXchgInst *AtomicCmpxchgCmpxchgPair32 = cast<AtomicCmpXchgInst>(
      getUniquePreviousStore(PtrArg, AtomicCmpxchgCmpxchgMonotonicFail28));
  AtomicCmpXchgInst *AtomicCmpxchgCmpxchgPair34 = cast<AtomicCmpXchgInst>(
      getUniquePreviousStore(PtrArg, AtomicCmpxchgCmpxchgAcquireFail29));
  AtomicCmpXchgInst *AtomicCmpxchgCmpxchgPair36 = cast<AtomicCmpXchgInst>(
      getUniquePreviousStore(PtrArg, AtomicCmpxchgCmpxchgSeqcstFail30));
  AtomicCmpXchgInst *AtomicCmpxchgCmpxchgPair43 = cast<AtomicCmpXchgInst>(
      getUniquePreviousStore(PtrArg, AtomicCmpxchgCmpxchgMonotonicFail39));
  AtomicCmpXchgInst *AtomicCmpxchgCmpxchgPair45 = cast<AtomicCmpXchgInst>(
      getUniquePreviousStore(PtrArg, AtomicCmpxchgCmpxchgAcquireFail40));
  AtomicCmpXchgInst *AtomicCmpxchgCmpxchgPair98 = cast<AtomicCmpXchgInst>(
      getUniquePreviousStore(PtrArg, AtomicCmpxchgCmpxchgSeqcstFail92));
  AtomicCmpXchgInst *AtomicCmpxchgCmpxchgPair85 = cast<AtomicCmpXchgInst>(
      getUniquePreviousStore(PtrArg, AtomicCmpxchgCmpxchgAcquireFail80));
  AtomicCmpXchgInst *AtomicCmpxchgCmpxchgPair14 = cast<AtomicCmpXchgInst>(
      getUniquePreviousStore(PtrArg, AtomicCmpxchgCmpxchgSeqcstFail8));
  AtomicCmpXchgInst *AtomicCmpxchgCmpxchgPair61 = cast<AtomicCmpXchgInst>(
      getUniquePreviousStore(PtrArg, AtomicCmpxchgCmpxchgMonotonicFail57));
  AtomicCmpXchgInst *AtomicCmpxchgCmpxchgPair74 = cast<AtomicCmpXchgInst>(
      getUniquePreviousStore(PtrArg, AtomicCmpxchgCmpxchgAcquireFail69));
  AtomicCmpXchgInst *AtomicCmpxchgCmpxchgPair47 = cast<AtomicCmpXchgInst>(
      getUniquePreviousStore(PtrArg, AtomicCmpxchgCmpxchgSeqcstFail41));
  AtomicCmpXchgInst *AtomicCmpxchgCmpxchgPair83 = cast<AtomicCmpXchgInst>(
      getUniquePreviousStore(PtrArg, AtomicCmpxchgCmpxchgMonotonicFail79));
  AtomicCmpXchgInst *AtomicCmpxchgCmpxchgPair25 = cast<AtomicCmpXchgInst>(
      getUniquePreviousStore(PtrArg, AtomicCmpxchgCmpxchgSeqcstFail19));
  AtomicCmpXchgInst *AtomicCmpxchgCmpxchgPair96 = cast<AtomicCmpXchgInst>(
      getUniquePreviousStore(PtrArg, AtomicCmpxchgCmpxchgAcquireFail91));
  AtomicCmpXchgInst *AtomicCmpxchgCmpxchgPair65 = cast<AtomicCmpXchgInst>(
      getUniquePreviousStore(PtrArg, AtomicCmpxchgCmpxchgSeqcstFail59));
  AtomicCmpXchgInst *AtomicCmpxchgCmpxchgPair107 = cast<AtomicCmpXchgInst>(
      getUniquePreviousStore(PtrArg, AtomicCmpxchgCmpxchgAcquireFail102));
  AtomicCmpXchgInst *AtomicCmpxchgCmpxchgPair63 = cast<AtomicCmpXchgInst>(
      getUniquePreviousStore(PtrArg, AtomicCmpxchgCmpxchgAcquireFail58));
  AtomicCmpXchgInst *AtomicCmpxchgCmpxchgPair76 = cast<AtomicCmpXchgInst>(
      getUniquePreviousStore(PtrArg, AtomicCmpxchgCmpxchgSeqcstFail70));
  AtomicCmpXchgInst *AtomicCmpxchgCmpxchgPair21 = cast<AtomicCmpXchgInst>(
      getUniquePreviousStore(PtrArg, AtomicCmpxchgCmpxchgMonotonicFail17));

  // %atomic_cmpxchg.cmpxchg.expected = load i32, ptr %expected_ptr, align 4
  EXPECT_EQ(AtomicCmpxchgCmpxchgExpected->getName(),
            "atomic_cmpxchg.cmpxchg.expected");
  EXPECT_EQ(AtomicCmpxchgCmpxchgExpected->getParent(), EntryBB);
  EXPECT_EQ(AtomicCmpxchgCmpxchgExpected->getType(), Type::getInt32Ty(Ctx));
  EXPECT_TRUE(AtomicCmpxchgCmpxchgExpected->isSimple());
  EXPECT_EQ(AtomicCmpxchgCmpxchgExpected->getPointerOperand(), ExpectedArg);

  // %atomic_cmpxchg.cmpxchg.desired = load i32, ptr %desired_ptr, align 4
  EXPECT_EQ(AtomicCmpxchgCmpxchgDesired->getName(),
            "atomic_cmpxchg.cmpxchg.desired");
  EXPECT_EQ(AtomicCmpxchgCmpxchgDesired->getParent(), EntryBB);
  EXPECT_EQ(AtomicCmpxchgCmpxchgDesired->getType(), Type::getInt32Ty(Ctx));
  EXPECT_TRUE(AtomicCmpxchgCmpxchgDesired->isSimple());
  EXPECT_EQ(AtomicCmpxchgCmpxchgDesired->getPointerOperand(), DesiredArg);

  // switch i1 %predarg, label %atomic_cmpxchg.cmpxchg.weak [
  //   i1 false, label %atomic_cmpxchg.cmpxchg.strong
  // ]
  EXPECT_TRUE(Switch1->getName().empty());
  EXPECT_EQ(Switch1->getParent(), EntryBB);
  EXPECT_EQ(Switch1->getType(), Type::getVoidTy(Ctx));
  EXPECT_EQ(Switch1->getCondition(), PredArg);
  EXPECT_EQ(Switch1->getDefaultDest(), AtomicCmpxchgCmpxchgWeak);
  EXPECT_EQ(cast<ConstantInt>(Switch1->getOperand(2))->getZExtValue(), 0);
  EXPECT_EQ(Switch1->getOperand(3), AtomicCmpxchgCmpxchgStrong);

  // switch i32 %memorderarg_success, label %atomic_cmpxchg.cmpxchg.monoton...
  //   i32 1, label %atomic_cmpxchg.cmpxchg.acquire
  //   i32 2, label %atomic_cmpxchg.cmpxchg.acquire
  //   i32 3, label %atomic_cmpxchg.cmpxchg.release
  //   i32 4, label %atomic_cmpxchg.cmpxchg.acqrel
  //   i32 5, label %atomic_cmpxchg.cmpxchg.seqcst
  // ]
  EXPECT_TRUE(Switch2->getName().empty());
  EXPECT_EQ(Switch2->getParent(), AtomicCmpxchgCmpxchgStrong);
  EXPECT_EQ(Switch2->getType(), Type::getVoidTy(Ctx));
  EXPECT_EQ(Switch2->getCondition(), MemorderArg);
  EXPECT_EQ(Switch2->getDefaultDest(), AtomicCmpxchgCmpxchgMonotonic);
  EXPECT_EQ(cast<ConstantInt>(Switch2->getOperand(2))->getZExtValue(), 1);
  EXPECT_EQ(Switch2->getOperand(3), AtomicCmpxchgCmpxchgAcquire);
  EXPECT_EQ(cast<ConstantInt>(Switch2->getOperand(4))->getZExtValue(), 2);
  EXPECT_EQ(Switch2->getOperand(5), AtomicCmpxchgCmpxchgAcquire);
  EXPECT_EQ(cast<ConstantInt>(Switch2->getOperand(6))->getZExtValue(), 3);
  EXPECT_EQ(Switch2->getOperand(7), AtomicCmpxchgCmpxchgRelease);
  EXPECT_EQ(cast<ConstantInt>(Switch2->getOperand(8))->getZExtValue(), 4);
  EXPECT_EQ(Switch2->getOperand(9), AtomicCmpxchgCmpxchgAcqrel);
  EXPECT_EQ(cast<ConstantInt>(Switch2->getOperand(10))->getZExtValue(), 5);
  EXPECT_EQ(Switch2->getOperand(11), AtomicCmpxchgCmpxchgSeqcst);

  // switch i32 %memorderarg_success, label %atomic_cmpxchg.cmpxchg.monoton...
  //   i32 1, label %atomic_cmpxchg.cmpxchg.acquire_fail
  //   i32 2, label %atomic_cmpxchg.cmpxchg.acquire_fail
  //   i32 5, label %atomic_cmpxchg.cmpxchg.seqcst_fail
  // ]
  EXPECT_TRUE(Switch3->getName().empty());
  EXPECT_EQ(Switch3->getParent(), AtomicCmpxchgCmpxchgMonotonic);
  EXPECT_EQ(Switch3->getType(), Type::getVoidTy(Ctx));
  EXPECT_EQ(Switch3->getCondition(), MemorderArg);
  EXPECT_EQ(Switch3->getDefaultDest(), AtomicCmpxchgCmpxchgMonotonicFail);
  EXPECT_EQ(cast<ConstantInt>(Switch3->getOperand(2))->getZExtValue(), 1);
  EXPECT_EQ(Switch3->getOperand(3), AtomicCmpxchgCmpxchgAcquireFail);
  EXPECT_EQ(cast<ConstantInt>(Switch3->getOperand(4))->getZExtValue(), 2);
  EXPECT_EQ(Switch3->getOperand(5), AtomicCmpxchgCmpxchgAcquireFail);
  EXPECT_EQ(cast<ConstantInt>(Switch3->getOperand(6))->getZExtValue(), 5);
  EXPECT_EQ(Switch3->getOperand(7), AtomicCmpxchgCmpxchgSeqcstFail);

  // %atomic_cmpxchg.cmpxchg.pair = cmpxchg volatile ptr %atomic_ptr, i32 %...
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getName(), "atomic_cmpxchg.cmpxchg.pair");
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getParent(),
            AtomicCmpxchgCmpxchgMonotonicFail);
  EXPECT_TRUE(AtomicCmpxchgCmpxchgPair->isVolatile());
  EXPECT_FALSE(AtomicCmpxchgCmpxchgPair->isWeak());
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getSuccessOrdering(),
            AtomicOrdering::Monotonic);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getFailureOrdering(),
            AtomicOrdering::Monotonic);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getSyncScopeID(), SyncScope::System);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getAlign(), 1);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getPointerOperand(), PtrArg);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getCompareOperand(),
            AtomicCmpxchgCmpxchgExpected);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getNewValOperand(),
            AtomicCmpxchgCmpxchgDesired);

  // br label %atomic_cmpxchg.cmpxchg.failorder.continue
  EXPECT_TRUE(Branch1->getName().empty());
  EXPECT_EQ(Branch1->getParent(), AtomicCmpxchgCmpxchgMonotonicFail);
  EXPECT_EQ(Branch1->getType(), Type::getVoidTy(Ctx));
  EXPECT_TRUE(Branch1->isUnconditional());
  EXPECT_EQ(Branch1->getOperand(0), AtomicCmpxchgCmpxchgFailorderContinue);

  // %atomic_cmpxchg.cmpxchg.pair1 = cmpxchg volatile ptr %atomic_ptr, i32 ...
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair1->getName(),
            "atomic_cmpxchg.cmpxchg.pair1");
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair1->getParent(),
            AtomicCmpxchgCmpxchgAcquireFail);
  EXPECT_TRUE(AtomicCmpxchgCmpxchgPair1->isVolatile());
  EXPECT_FALSE(AtomicCmpxchgCmpxchgPair1->isWeak());
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair1->getSuccessOrdering(),
            AtomicOrdering::Monotonic);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair1->getFailureOrdering(),
            AtomicOrdering::Acquire);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair1->getSyncScopeID(), SyncScope::System);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair1->getAlign(), 1);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair1->getPointerOperand(), PtrArg);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair1->getCompareOperand(),
            AtomicCmpxchgCmpxchgExpected);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair1->getNewValOperand(),
            AtomicCmpxchgCmpxchgDesired);

  // br label %atomic_cmpxchg.cmpxchg.failorder.continue
  EXPECT_TRUE(Branch2->getName().empty());
  EXPECT_EQ(Branch2->getParent(), AtomicCmpxchgCmpxchgAcquireFail);
  EXPECT_EQ(Branch2->getType(), Type::getVoidTy(Ctx));
  EXPECT_TRUE(Branch2->isUnconditional());
  EXPECT_EQ(Branch2->getOperand(0), AtomicCmpxchgCmpxchgFailorderContinue);

  // %atomic_cmpxchg.cmpxchg.pair3 = cmpxchg volatile ptr %atomic_ptr, i32 ...
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair3->getName(),
            "atomic_cmpxchg.cmpxchg.pair3");
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair3->getParent(),
            AtomicCmpxchgCmpxchgSeqcstFail);
  EXPECT_TRUE(AtomicCmpxchgCmpxchgPair3->isVolatile());
  EXPECT_FALSE(AtomicCmpxchgCmpxchgPair3->isWeak());
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair3->getSuccessOrdering(),
            AtomicOrdering::Monotonic);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair3->getFailureOrdering(),
            AtomicOrdering::SequentiallyConsistent);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair3->getSyncScopeID(), SyncScope::System);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair3->getAlign(), 1);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair3->getPointerOperand(), PtrArg);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair3->getCompareOperand(),
            AtomicCmpxchgCmpxchgExpected);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair3->getNewValOperand(),
            AtomicCmpxchgCmpxchgDesired);

  // br label %atomic_cmpxchg.cmpxchg.failorder.continue
  EXPECT_TRUE(Branch3->getName().empty());
  EXPECT_EQ(Branch3->getParent(), AtomicCmpxchgCmpxchgSeqcstFail);
  EXPECT_EQ(Branch3->getType(), Type::getVoidTy(Ctx));
  EXPECT_TRUE(Branch3->isUnconditional());
  EXPECT_EQ(Branch3->getOperand(0), AtomicCmpxchgCmpxchgFailorderContinue);

  // br label %atomic_cmpxchg.cmpxchg.memorder.continue
  EXPECT_TRUE(Branch4->getName().empty());
  EXPECT_EQ(Branch4->getParent(), AtomicCmpxchgCmpxchgFailorderContinue);
  EXPECT_EQ(Branch4->getType(), Type::getVoidTy(Ctx));
  EXPECT_TRUE(Branch4->isUnconditional());
  EXPECT_EQ(Branch4->getOperand(0), AtomicCmpxchgCmpxchgMemorderContinue);

  // switch i32 %memorderarg_success, label %atomic_cmpxchg.cmpxchg.monoton...
  //   i32 1, label %atomic_cmpxchg.cmpxchg.acquire_fail7
  //   i32 2, label %atomic_cmpxchg.cmpxchg.acquire_fail7
  //   i32 5, label %atomic_cmpxchg.cmpxchg.seqcst_fail8
  // ]
  EXPECT_TRUE(Switch4->getName().empty());
  EXPECT_EQ(Switch4->getParent(), AtomicCmpxchgCmpxchgAcquire);
  EXPECT_EQ(Switch4->getType(), Type::getVoidTy(Ctx));
  EXPECT_EQ(Switch4->getCondition(), MemorderArg);
  EXPECT_EQ(Switch4->getDefaultDest(), AtomicCmpxchgCmpxchgMonotonicFail6);
  EXPECT_EQ(cast<ConstantInt>(Switch4->getOperand(2))->getZExtValue(), 1);
  EXPECT_EQ(Switch4->getOperand(3), AtomicCmpxchgCmpxchgAcquireFail7);
  EXPECT_EQ(cast<ConstantInt>(Switch4->getOperand(4))->getZExtValue(), 2);
  EXPECT_EQ(Switch4->getOperand(5), AtomicCmpxchgCmpxchgAcquireFail7);
  EXPECT_EQ(cast<ConstantInt>(Switch4->getOperand(6))->getZExtValue(), 5);
  EXPECT_EQ(Switch4->getOperand(7), AtomicCmpxchgCmpxchgSeqcstFail8);

  // %atomic_cmpxchg.cmpxchg.pair10 = cmpxchg volatile ptr %atomic_ptr, i32...
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair10->getName(),
            "atomic_cmpxchg.cmpxchg.pair10");
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair10->getParent(),
            AtomicCmpxchgCmpxchgMonotonicFail6);
  EXPECT_TRUE(AtomicCmpxchgCmpxchgPair10->isVolatile());
  EXPECT_FALSE(AtomicCmpxchgCmpxchgPair10->isWeak());
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair10->getSuccessOrdering(),
            AtomicOrdering::Acquire);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair10->getFailureOrdering(),
            AtomicOrdering::Monotonic);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair10->getSyncScopeID(), SyncScope::System);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair10->getAlign(), 1);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair10->getPointerOperand(), PtrArg);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair10->getCompareOperand(),
            AtomicCmpxchgCmpxchgExpected);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair10->getNewValOperand(),
            AtomicCmpxchgCmpxchgDesired);

  // br label %atomic_cmpxchg.cmpxchg.failorder.continue5
  EXPECT_TRUE(Branch5->getName().empty());
  EXPECT_EQ(Branch5->getParent(), AtomicCmpxchgCmpxchgMonotonicFail6);
  EXPECT_EQ(Branch5->getType(), Type::getVoidTy(Ctx));
  EXPECT_TRUE(Branch5->isUnconditional());
  EXPECT_EQ(Branch5->getOperand(0), AtomicCmpxchgCmpxchgFailorderContinue5);

  // %atomic_cmpxchg.cmpxchg.pair12 = cmpxchg volatile ptr %atomic_ptr, i32...
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair12->getName(),
            "atomic_cmpxchg.cmpxchg.pair12");
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair12->getParent(),
            AtomicCmpxchgCmpxchgAcquireFail7);
  EXPECT_TRUE(AtomicCmpxchgCmpxchgPair12->isVolatile());
  EXPECT_FALSE(AtomicCmpxchgCmpxchgPair12->isWeak());
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair12->getSuccessOrdering(),
            AtomicOrdering::Acquire);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair12->getFailureOrdering(),
            AtomicOrdering::Acquire);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair12->getSyncScopeID(), SyncScope::System);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair12->getAlign(), 1);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair12->getPointerOperand(), PtrArg);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair12->getCompareOperand(),
            AtomicCmpxchgCmpxchgExpected);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair12->getNewValOperand(),
            AtomicCmpxchgCmpxchgDesired);

  // br label %atomic_cmpxchg.cmpxchg.failorder.continue5
  EXPECT_TRUE(Branch6->getName().empty());
  EXPECT_EQ(Branch6->getParent(), AtomicCmpxchgCmpxchgAcquireFail7);
  EXPECT_EQ(Branch6->getType(), Type::getVoidTy(Ctx));
  EXPECT_TRUE(Branch6->isUnconditional());
  EXPECT_EQ(Branch6->getOperand(0), AtomicCmpxchgCmpxchgFailorderContinue5);

  // %atomic_cmpxchg.cmpxchg.pair14 = cmpxchg volatile ptr %atomic_ptr, i32...
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair14->getName(),
            "atomic_cmpxchg.cmpxchg.pair14");
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair14->getParent(),
            AtomicCmpxchgCmpxchgSeqcstFail8);
  EXPECT_TRUE(AtomicCmpxchgCmpxchgPair14->isVolatile());
  EXPECT_FALSE(AtomicCmpxchgCmpxchgPair14->isWeak());
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair14->getSuccessOrdering(),
            AtomicOrdering::Acquire);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair14->getFailureOrdering(),
            AtomicOrdering::SequentiallyConsistent);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair14->getSyncScopeID(), SyncScope::System);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair14->getAlign(), 1);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair14->getPointerOperand(), PtrArg);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair14->getCompareOperand(),
            AtomicCmpxchgCmpxchgExpected);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair14->getNewValOperand(),
            AtomicCmpxchgCmpxchgDesired);

  // br label %atomic_cmpxchg.cmpxchg.failorder.continue5
  EXPECT_TRUE(Branch7->getName().empty());
  EXPECT_EQ(Branch7->getParent(), AtomicCmpxchgCmpxchgSeqcstFail8);
  EXPECT_EQ(Branch7->getType(), Type::getVoidTy(Ctx));
  EXPECT_TRUE(Branch7->isUnconditional());
  EXPECT_EQ(Branch7->getOperand(0), AtomicCmpxchgCmpxchgFailorderContinue5);

  // br label %atomic_cmpxchg.cmpxchg.memorder.continue
  EXPECT_TRUE(Branch8->getName().empty());
  EXPECT_EQ(Branch8->getParent(), AtomicCmpxchgCmpxchgFailorderContinue5);
  EXPECT_EQ(Branch8->getType(), Type::getVoidTy(Ctx));
  EXPECT_TRUE(Branch8->isUnconditional());
  EXPECT_EQ(Branch8->getOperand(0), AtomicCmpxchgCmpxchgMemorderContinue);

  // switch i32 %memorderarg_success, label %atomic_cmpxchg.cmpxchg.monoton...
  //   i32 1, label %atomic_cmpxchg.cmpxchg.acquire_fail18
  //   i32 2, label %atomic_cmpxchg.cmpxchg.acquire_fail18
  //   i32 5, label %atomic_cmpxchg.cmpxchg.seqcst_fail19
  // ]
  EXPECT_TRUE(Switch5->getName().empty());
  EXPECT_EQ(Switch5->getParent(), AtomicCmpxchgCmpxchgRelease);
  EXPECT_EQ(Switch5->getType(), Type::getVoidTy(Ctx));
  EXPECT_EQ(Switch5->getCondition(), MemorderArg);
  EXPECT_EQ(Switch5->getDefaultDest(), AtomicCmpxchgCmpxchgMonotonicFail17);
  EXPECT_EQ(cast<ConstantInt>(Switch5->getOperand(2))->getZExtValue(), 1);
  EXPECT_EQ(Switch5->getOperand(3), AtomicCmpxchgCmpxchgAcquireFail18);
  EXPECT_EQ(cast<ConstantInt>(Switch5->getOperand(4))->getZExtValue(), 2);
  EXPECT_EQ(Switch5->getOperand(5), AtomicCmpxchgCmpxchgAcquireFail18);
  EXPECT_EQ(cast<ConstantInt>(Switch5->getOperand(6))->getZExtValue(), 5);
  EXPECT_EQ(Switch5->getOperand(7), AtomicCmpxchgCmpxchgSeqcstFail19);

  // %atomic_cmpxchg.cmpxchg.pair21 = cmpxchg volatile ptr %atomic_ptr, i32...
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair21->getName(),
            "atomic_cmpxchg.cmpxchg.pair21");
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair21->getParent(),
            AtomicCmpxchgCmpxchgMonotonicFail17);
  EXPECT_TRUE(AtomicCmpxchgCmpxchgPair21->isVolatile());
  EXPECT_FALSE(AtomicCmpxchgCmpxchgPair21->isWeak());
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair21->getSuccessOrdering(),
            AtomicOrdering::Release);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair21->getFailureOrdering(),
            AtomicOrdering::Monotonic);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair21->getSyncScopeID(), SyncScope::System);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair21->getAlign(), 1);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair21->getPointerOperand(), PtrArg);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair21->getCompareOperand(),
            AtomicCmpxchgCmpxchgExpected);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair21->getNewValOperand(),
            AtomicCmpxchgCmpxchgDesired);

  // br label %atomic_cmpxchg.cmpxchg.failorder.continue16
  EXPECT_TRUE(Branch9->getName().empty());
  EXPECT_EQ(Branch9->getParent(), AtomicCmpxchgCmpxchgMonotonicFail17);
  EXPECT_EQ(Branch9->getType(), Type::getVoidTy(Ctx));
  EXPECT_TRUE(Branch9->isUnconditional());
  EXPECT_EQ(Branch9->getOperand(0), AtomicCmpxchgCmpxchgFailorderContinue16);

  // %atomic_cmpxchg.cmpxchg.pair23 = cmpxchg volatile ptr %atomic_ptr, i32...
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair23->getName(),
            "atomic_cmpxchg.cmpxchg.pair23");
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair23->getParent(),
            AtomicCmpxchgCmpxchgAcquireFail18);
  EXPECT_TRUE(AtomicCmpxchgCmpxchgPair23->isVolatile());
  EXPECT_FALSE(AtomicCmpxchgCmpxchgPair23->isWeak());
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair23->getSuccessOrdering(),
            AtomicOrdering::Release);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair23->getFailureOrdering(),
            AtomicOrdering::Acquire);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair23->getSyncScopeID(), SyncScope::System);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair23->getAlign(), 1);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair23->getPointerOperand(), PtrArg);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair23->getCompareOperand(),
            AtomicCmpxchgCmpxchgExpected);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair23->getNewValOperand(),
            AtomicCmpxchgCmpxchgDesired);

  // br label %atomic_cmpxchg.cmpxchg.failorder.continue16
  EXPECT_TRUE(Branch10->getName().empty());
  EXPECT_EQ(Branch10->getParent(), AtomicCmpxchgCmpxchgAcquireFail18);
  EXPECT_EQ(Branch10->getType(), Type::getVoidTy(Ctx));
  EXPECT_TRUE(Branch10->isUnconditional());
  EXPECT_EQ(Branch10->getOperand(0), AtomicCmpxchgCmpxchgFailorderContinue16);

  // %atomic_cmpxchg.cmpxchg.pair25 = cmpxchg volatile ptr %atomic_ptr, i32...
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair25->getName(),
            "atomic_cmpxchg.cmpxchg.pair25");
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair25->getParent(),
            AtomicCmpxchgCmpxchgSeqcstFail19);
  EXPECT_TRUE(AtomicCmpxchgCmpxchgPair25->isVolatile());
  EXPECT_FALSE(AtomicCmpxchgCmpxchgPair25->isWeak());
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair25->getSuccessOrdering(),
            AtomicOrdering::Release);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair25->getFailureOrdering(),
            AtomicOrdering::SequentiallyConsistent);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair25->getSyncScopeID(), SyncScope::System);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair25->getAlign(), 1);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair25->getPointerOperand(), PtrArg);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair25->getCompareOperand(),
            AtomicCmpxchgCmpxchgExpected);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair25->getNewValOperand(),
            AtomicCmpxchgCmpxchgDesired);

  // br label %atomic_cmpxchg.cmpxchg.failorder.continue16
  EXPECT_TRUE(Branch11->getName().empty());
  EXPECT_EQ(Branch11->getParent(), AtomicCmpxchgCmpxchgSeqcstFail19);
  EXPECT_EQ(Branch11->getType(), Type::getVoidTy(Ctx));
  EXPECT_TRUE(Branch11->isUnconditional());
  EXPECT_EQ(Branch11->getOperand(0), AtomicCmpxchgCmpxchgFailorderContinue16);

  // br label %atomic_cmpxchg.cmpxchg.memorder.continue
  EXPECT_TRUE(Branch12->getName().empty());
  EXPECT_EQ(Branch12->getParent(), AtomicCmpxchgCmpxchgFailorderContinue16);
  EXPECT_EQ(Branch12->getType(), Type::getVoidTy(Ctx));
  EXPECT_TRUE(Branch12->isUnconditional());
  EXPECT_EQ(Branch12->getOperand(0), AtomicCmpxchgCmpxchgMemorderContinue);

  // switch i32 %memorderarg_success, label %atomic_cmpxchg.cmpxchg.monoton...
  //   i32 1, label %atomic_cmpxchg.cmpxchg.acquire_fail29
  //   i32 2, label %atomic_cmpxchg.cmpxchg.acquire_fail29
  //   i32 5, label %atomic_cmpxchg.cmpxchg.seqcst_fail30
  // ]
  EXPECT_TRUE(Switch6->getName().empty());
  EXPECT_EQ(Switch6->getParent(), AtomicCmpxchgCmpxchgAcqrel);
  EXPECT_EQ(Switch6->getType(), Type::getVoidTy(Ctx));
  EXPECT_EQ(Switch6->getCondition(), MemorderArg);
  EXPECT_EQ(Switch6->getDefaultDest(), AtomicCmpxchgCmpxchgMonotonicFail28);
  EXPECT_EQ(cast<ConstantInt>(Switch6->getOperand(2))->getZExtValue(), 1);
  EXPECT_EQ(Switch6->getOperand(3), AtomicCmpxchgCmpxchgAcquireFail29);
  EXPECT_EQ(cast<ConstantInt>(Switch6->getOperand(4))->getZExtValue(), 2);
  EXPECT_EQ(Switch6->getOperand(5), AtomicCmpxchgCmpxchgAcquireFail29);
  EXPECT_EQ(cast<ConstantInt>(Switch6->getOperand(6))->getZExtValue(), 5);
  EXPECT_EQ(Switch6->getOperand(7), AtomicCmpxchgCmpxchgSeqcstFail30);

  // %atomic_cmpxchg.cmpxchg.pair32 = cmpxchg volatile ptr %atomic_ptr, i32...
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair32->getName(),
            "atomic_cmpxchg.cmpxchg.pair32");
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair32->getParent(),
            AtomicCmpxchgCmpxchgMonotonicFail28);
  EXPECT_TRUE(AtomicCmpxchgCmpxchgPair32->isVolatile());
  EXPECT_FALSE(AtomicCmpxchgCmpxchgPair32->isWeak());
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair32->getSuccessOrdering(),
            AtomicOrdering::AcquireRelease);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair32->getFailureOrdering(),
            AtomicOrdering::Monotonic);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair32->getSyncScopeID(), SyncScope::System);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair32->getAlign(), 1);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair32->getPointerOperand(), PtrArg);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair32->getCompareOperand(),
            AtomicCmpxchgCmpxchgExpected);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair32->getNewValOperand(),
            AtomicCmpxchgCmpxchgDesired);

  // br label %atomic_cmpxchg.cmpxchg.failorder.continue27
  EXPECT_TRUE(Branch13->getName().empty());
  EXPECT_EQ(Branch13->getParent(), AtomicCmpxchgCmpxchgMonotonicFail28);
  EXPECT_EQ(Branch13->getType(), Type::getVoidTy(Ctx));
  EXPECT_TRUE(Branch13->isUnconditional());
  EXPECT_EQ(Branch13->getOperand(0), AtomicCmpxchgCmpxchgFailorderContinue27);

  // %atomic_cmpxchg.cmpxchg.pair34 = cmpxchg volatile ptr %atomic_ptr, i32...
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair34->getName(),
            "atomic_cmpxchg.cmpxchg.pair34");
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair34->getParent(),
            AtomicCmpxchgCmpxchgAcquireFail29);
  EXPECT_TRUE(AtomicCmpxchgCmpxchgPair34->isVolatile());
  EXPECT_FALSE(AtomicCmpxchgCmpxchgPair34->isWeak());
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair34->getSuccessOrdering(),
            AtomicOrdering::AcquireRelease);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair34->getFailureOrdering(),
            AtomicOrdering::Acquire);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair34->getSyncScopeID(), SyncScope::System);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair34->getAlign(), 1);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair34->getPointerOperand(), PtrArg);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair34->getCompareOperand(),
            AtomicCmpxchgCmpxchgExpected);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair34->getNewValOperand(),
            AtomicCmpxchgCmpxchgDesired);

  // br label %atomic_cmpxchg.cmpxchg.failorder.continue27
  EXPECT_TRUE(Branch14->getName().empty());
  EXPECT_EQ(Branch14->getParent(), AtomicCmpxchgCmpxchgAcquireFail29);
  EXPECT_EQ(Branch14->getType(), Type::getVoidTy(Ctx));
  EXPECT_TRUE(Branch14->isUnconditional());
  EXPECT_EQ(Branch14->getOperand(0), AtomicCmpxchgCmpxchgFailorderContinue27);

  // %atomic_cmpxchg.cmpxchg.pair36 = cmpxchg volatile ptr %atomic_ptr, i32...
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair36->getName(),
            "atomic_cmpxchg.cmpxchg.pair36");
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair36->getParent(),
            AtomicCmpxchgCmpxchgSeqcstFail30);
  EXPECT_TRUE(AtomicCmpxchgCmpxchgPair36->isVolatile());
  EXPECT_FALSE(AtomicCmpxchgCmpxchgPair36->isWeak());
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair36->getSuccessOrdering(),
            AtomicOrdering::AcquireRelease);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair36->getFailureOrdering(),
            AtomicOrdering::SequentiallyConsistent);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair36->getSyncScopeID(), SyncScope::System);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair36->getAlign(), 1);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair36->getPointerOperand(), PtrArg);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair36->getCompareOperand(),
            AtomicCmpxchgCmpxchgExpected);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair36->getNewValOperand(),
            AtomicCmpxchgCmpxchgDesired);

  // br label %atomic_cmpxchg.cmpxchg.failorder.continue27
  EXPECT_TRUE(Branch15->getName().empty());
  EXPECT_EQ(Branch15->getParent(), AtomicCmpxchgCmpxchgSeqcstFail30);
  EXPECT_EQ(Branch15->getType(), Type::getVoidTy(Ctx));
  EXPECT_TRUE(Branch15->isUnconditional());
  EXPECT_EQ(Branch15->getOperand(0), AtomicCmpxchgCmpxchgFailorderContinue27);

  // br label %atomic_cmpxchg.cmpxchg.memorder.continue
  EXPECT_TRUE(Branch16->getName().empty());
  EXPECT_EQ(Branch16->getParent(), AtomicCmpxchgCmpxchgFailorderContinue27);
  EXPECT_EQ(Branch16->getType(), Type::getVoidTy(Ctx));
  EXPECT_TRUE(Branch16->isUnconditional());
  EXPECT_EQ(Branch16->getOperand(0), AtomicCmpxchgCmpxchgMemorderContinue);

  // switch i32 %memorderarg_success, label %atomic_cmpxchg.cmpxchg.monoton...
  //   i32 1, label %atomic_cmpxchg.cmpxchg.acquire_fail40
  //   i32 2, label %atomic_cmpxchg.cmpxchg.acquire_fail40
  //   i32 5, label %atomic_cmpxchg.cmpxchg.seqcst_fail41
  // ]
  EXPECT_TRUE(Switch7->getName().empty());
  EXPECT_EQ(Switch7->getParent(), AtomicCmpxchgCmpxchgSeqcst);
  EXPECT_EQ(Switch7->getType(), Type::getVoidTy(Ctx));
  EXPECT_EQ(Switch7->getCondition(), MemorderArg);
  EXPECT_EQ(Switch7->getDefaultDest(), AtomicCmpxchgCmpxchgMonotonicFail39);
  EXPECT_EQ(cast<ConstantInt>(Switch7->getOperand(2))->getZExtValue(), 1);
  EXPECT_EQ(Switch7->getOperand(3), AtomicCmpxchgCmpxchgAcquireFail40);
  EXPECT_EQ(cast<ConstantInt>(Switch7->getOperand(4))->getZExtValue(), 2);
  EXPECT_EQ(Switch7->getOperand(5), AtomicCmpxchgCmpxchgAcquireFail40);
  EXPECT_EQ(cast<ConstantInt>(Switch7->getOperand(6))->getZExtValue(), 5);
  EXPECT_EQ(Switch7->getOperand(7), AtomicCmpxchgCmpxchgSeqcstFail41);

  // %atomic_cmpxchg.cmpxchg.pair43 = cmpxchg volatile ptr %atomic_ptr, i32...
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair43->getName(),
            "atomic_cmpxchg.cmpxchg.pair43");
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair43->getParent(),
            AtomicCmpxchgCmpxchgMonotonicFail39);
  EXPECT_TRUE(AtomicCmpxchgCmpxchgPair43->isVolatile());
  EXPECT_FALSE(AtomicCmpxchgCmpxchgPair43->isWeak());
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair43->getSuccessOrdering(),
            AtomicOrdering::SequentiallyConsistent);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair43->getFailureOrdering(),
            AtomicOrdering::Monotonic);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair43->getSyncScopeID(), SyncScope::System);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair43->getAlign(), 1);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair43->getPointerOperand(), PtrArg);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair43->getCompareOperand(),
            AtomicCmpxchgCmpxchgExpected);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair43->getNewValOperand(),
            AtomicCmpxchgCmpxchgDesired);

  // br label %atomic_cmpxchg.cmpxchg.failorder.continue38
  EXPECT_TRUE(Branch17->getName().empty());
  EXPECT_EQ(Branch17->getParent(), AtomicCmpxchgCmpxchgMonotonicFail39);
  EXPECT_EQ(Branch17->getType(), Type::getVoidTy(Ctx));
  EXPECT_TRUE(Branch17->isUnconditional());
  EXPECT_EQ(Branch17->getOperand(0), AtomicCmpxchgCmpxchgFailorderContinue38);

  // %atomic_cmpxchg.cmpxchg.pair45 = cmpxchg volatile ptr %atomic_ptr, i32...
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair45->getName(),
            "atomic_cmpxchg.cmpxchg.pair45");
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair45->getParent(),
            AtomicCmpxchgCmpxchgAcquireFail40);
  EXPECT_TRUE(AtomicCmpxchgCmpxchgPair45->isVolatile());
  EXPECT_FALSE(AtomicCmpxchgCmpxchgPair45->isWeak());
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair45->getSuccessOrdering(),
            AtomicOrdering::SequentiallyConsistent);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair45->getFailureOrdering(),
            AtomicOrdering::Acquire);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair45->getSyncScopeID(), SyncScope::System);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair45->getAlign(), 1);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair45->getPointerOperand(), PtrArg);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair45->getCompareOperand(),
            AtomicCmpxchgCmpxchgExpected);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair45->getNewValOperand(),
            AtomicCmpxchgCmpxchgDesired);

  // br label %atomic_cmpxchg.cmpxchg.failorder.continue38
  EXPECT_TRUE(Branch18->getName().empty());
  EXPECT_EQ(Branch18->getParent(), AtomicCmpxchgCmpxchgAcquireFail40);
  EXPECT_EQ(Branch18->getType(), Type::getVoidTy(Ctx));
  EXPECT_TRUE(Branch18->isUnconditional());
  EXPECT_EQ(Branch18->getOperand(0), AtomicCmpxchgCmpxchgFailorderContinue38);

  // %atomic_cmpxchg.cmpxchg.pair47 = cmpxchg volatile ptr %atomic_ptr, i32...
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair47->getName(),
            "atomic_cmpxchg.cmpxchg.pair47");
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair47->getParent(),
            AtomicCmpxchgCmpxchgSeqcstFail41);
  EXPECT_TRUE(AtomicCmpxchgCmpxchgPair47->isVolatile());
  EXPECT_FALSE(AtomicCmpxchgCmpxchgPair47->isWeak());
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair47->getSuccessOrdering(),
            AtomicOrdering::SequentiallyConsistent);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair47->getFailureOrdering(),
            AtomicOrdering::SequentiallyConsistent);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair47->getSyncScopeID(), SyncScope::System);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair47->getAlign(), 1);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair47->getPointerOperand(), PtrArg);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair47->getCompareOperand(),
            AtomicCmpxchgCmpxchgExpected);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair47->getNewValOperand(),
            AtomicCmpxchgCmpxchgDesired);

  // br label %atomic_cmpxchg.cmpxchg.failorder.continue38
  EXPECT_TRUE(Branch19->getName().empty());
  EXPECT_EQ(Branch19->getParent(), AtomicCmpxchgCmpxchgSeqcstFail41);
  EXPECT_EQ(Branch19->getType(), Type::getVoidTy(Ctx));
  EXPECT_TRUE(Branch19->isUnconditional());
  EXPECT_EQ(Branch19->getOperand(0), AtomicCmpxchgCmpxchgFailorderContinue38);

  // br label %atomic_cmpxchg.cmpxchg.memorder.continue
  EXPECT_TRUE(Branch20->getName().empty());
  EXPECT_EQ(Branch20->getParent(), AtomicCmpxchgCmpxchgFailorderContinue38);
  EXPECT_EQ(Branch20->getType(), Type::getVoidTy(Ctx));
  EXPECT_TRUE(Branch20->isUnconditional());
  EXPECT_EQ(Branch20->getOperand(0), AtomicCmpxchgCmpxchgMemorderContinue);

  // br label %atomic_cmpxchg.cmpxchg.weak.continue
  EXPECT_TRUE(Branch21->getName().empty());
  EXPECT_EQ(Branch21->getParent(), AtomicCmpxchgCmpxchgMemorderContinue);
  EXPECT_EQ(Branch21->getType(), Type::getVoidTy(Ctx));
  EXPECT_TRUE(Branch21->isUnconditional());
  EXPECT_EQ(Branch21->getOperand(0), ExitBB);

  // switch i32 %memorderarg_success, label %atomic_cmpxchg.cmpxchg.monoton...
  //   i32 1, label %atomic_cmpxchg.cmpxchg.acquire51
  //   i32 2, label %atomic_cmpxchg.cmpxchg.acquire51
  //   i32 3, label %atomic_cmpxchg.cmpxchg.release52
  //   i32 4, label %atomic_cmpxchg.cmpxchg.acqrel53
  //   i32 5, label %atomic_cmpxchg.cmpxchg.seqcst54
  // ]
  EXPECT_TRUE(Switch8->getName().empty());
  EXPECT_EQ(Switch8->getParent(), AtomicCmpxchgCmpxchgWeak);
  EXPECT_EQ(Switch8->getType(), Type::getVoidTy(Ctx));
  EXPECT_EQ(Switch8->getCondition(), MemorderArg);
  EXPECT_EQ(Switch8->getDefaultDest(), AtomicCmpxchgCmpxchgMonotonic50);
  EXPECT_EQ(cast<ConstantInt>(Switch8->getOperand(2))->getZExtValue(), 1);
  EXPECT_EQ(Switch8->getOperand(3), AtomicCmpxchgCmpxchgAcquire51);
  EXPECT_EQ(cast<ConstantInt>(Switch8->getOperand(4))->getZExtValue(), 2);
  EXPECT_EQ(Switch8->getOperand(5), AtomicCmpxchgCmpxchgAcquire51);
  EXPECT_EQ(cast<ConstantInt>(Switch8->getOperand(6))->getZExtValue(), 3);
  EXPECT_EQ(Switch8->getOperand(7), AtomicCmpxchgCmpxchgRelease52);
  EXPECT_EQ(cast<ConstantInt>(Switch8->getOperand(8))->getZExtValue(), 4);
  EXPECT_EQ(Switch8->getOperand(9), AtomicCmpxchgCmpxchgAcqrel53);
  EXPECT_EQ(cast<ConstantInt>(Switch8->getOperand(10))->getZExtValue(), 5);
  EXPECT_EQ(Switch8->getOperand(11), AtomicCmpxchgCmpxchgSeqcst54);

  // switch i32 %memorderarg_success, label %atomic_cmpxchg.cmpxchg.monoton...
  //   i32 1, label %atomic_cmpxchg.cmpxchg.acquire_fail58
  //   i32 2, label %atomic_cmpxchg.cmpxchg.acquire_fail58
  //   i32 5, label %atomic_cmpxchg.cmpxchg.seqcst_fail59
  // ]
  EXPECT_TRUE(Switch9->getName().empty());
  EXPECT_EQ(Switch9->getParent(), AtomicCmpxchgCmpxchgMonotonic50);
  EXPECT_EQ(Switch9->getType(), Type::getVoidTy(Ctx));
  EXPECT_EQ(Switch9->getCondition(), MemorderArg);
  EXPECT_EQ(Switch9->getDefaultDest(), AtomicCmpxchgCmpxchgMonotonicFail57);
  EXPECT_EQ(cast<ConstantInt>(Switch9->getOperand(2))->getZExtValue(), 1);
  EXPECT_EQ(Switch9->getOperand(3), AtomicCmpxchgCmpxchgAcquireFail58);
  EXPECT_EQ(cast<ConstantInt>(Switch9->getOperand(4))->getZExtValue(), 2);
  EXPECT_EQ(Switch9->getOperand(5), AtomicCmpxchgCmpxchgAcquireFail58);
  EXPECT_EQ(cast<ConstantInt>(Switch9->getOperand(6))->getZExtValue(), 5);
  EXPECT_EQ(Switch9->getOperand(7), AtomicCmpxchgCmpxchgSeqcstFail59);

  // %atomic_cmpxchg.cmpxchg.pair61 = cmpxchg weak volatile ptr %atomic_ptr...
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair61->getName(),
            "atomic_cmpxchg.cmpxchg.pair61");
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair61->getParent(),
            AtomicCmpxchgCmpxchgMonotonicFail57);
  EXPECT_TRUE(AtomicCmpxchgCmpxchgPair61->isVolatile());
  EXPECT_TRUE(AtomicCmpxchgCmpxchgPair61->isWeak());
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair61->getSuccessOrdering(),
            AtomicOrdering::Monotonic);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair61->getFailureOrdering(),
            AtomicOrdering::Monotonic);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair61->getSyncScopeID(), SyncScope::System);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair61->getAlign(), 1);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair61->getPointerOperand(), PtrArg);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair61->getCompareOperand(),
            AtomicCmpxchgCmpxchgExpected);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair61->getNewValOperand(),
            AtomicCmpxchgCmpxchgDesired);

  // br label %atomic_cmpxchg.cmpxchg.failorder.continue56
  EXPECT_TRUE(Branch22->getName().empty());
  EXPECT_EQ(Branch22->getParent(), AtomicCmpxchgCmpxchgMonotonicFail57);
  EXPECT_EQ(Branch22->getType(), Type::getVoidTy(Ctx));
  EXPECT_TRUE(Branch22->isUnconditional());
  EXPECT_EQ(Branch22->getOperand(0), AtomicCmpxchgCmpxchgFailorderContinue56);

  // %atomic_cmpxchg.cmpxchg.pair63 = cmpxchg weak volatile ptr %atomic_ptr...
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair63->getName(),
            "atomic_cmpxchg.cmpxchg.pair63");
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair63->getParent(),
            AtomicCmpxchgCmpxchgAcquireFail58);
  EXPECT_TRUE(AtomicCmpxchgCmpxchgPair63->isVolatile());
  EXPECT_TRUE(AtomicCmpxchgCmpxchgPair63->isWeak());
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair63->getSuccessOrdering(),
            AtomicOrdering::Monotonic);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair63->getFailureOrdering(),
            AtomicOrdering::Acquire);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair63->getSyncScopeID(), SyncScope::System);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair63->getAlign(), 1);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair63->getPointerOperand(), PtrArg);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair63->getCompareOperand(),
            AtomicCmpxchgCmpxchgExpected);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair63->getNewValOperand(),
            AtomicCmpxchgCmpxchgDesired);

  // br label %atomic_cmpxchg.cmpxchg.failorder.continue56
  EXPECT_TRUE(Branch23->getName().empty());
  EXPECT_EQ(Branch23->getParent(), AtomicCmpxchgCmpxchgAcquireFail58);
  EXPECT_EQ(Branch23->getType(), Type::getVoidTy(Ctx));
  EXPECT_TRUE(Branch23->isUnconditional());
  EXPECT_EQ(Branch23->getOperand(0), AtomicCmpxchgCmpxchgFailorderContinue56);

  // %atomic_cmpxchg.cmpxchg.pair65 = cmpxchg weak volatile ptr %atomic_ptr...
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair65->getName(),
            "atomic_cmpxchg.cmpxchg.pair65");
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair65->getParent(),
            AtomicCmpxchgCmpxchgSeqcstFail59);
  EXPECT_TRUE(AtomicCmpxchgCmpxchgPair65->isVolatile());
  EXPECT_TRUE(AtomicCmpxchgCmpxchgPair65->isWeak());
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair65->getSuccessOrdering(),
            AtomicOrdering::Monotonic);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair65->getFailureOrdering(),
            AtomicOrdering::SequentiallyConsistent);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair65->getSyncScopeID(), SyncScope::System);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair65->getAlign(), 1);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair65->getPointerOperand(), PtrArg);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair65->getCompareOperand(),
            AtomicCmpxchgCmpxchgExpected);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair65->getNewValOperand(),
            AtomicCmpxchgCmpxchgDesired);

  // br label %atomic_cmpxchg.cmpxchg.failorder.continue56
  EXPECT_TRUE(Branch24->getName().empty());
  EXPECT_EQ(Branch24->getParent(), AtomicCmpxchgCmpxchgSeqcstFail59);
  EXPECT_EQ(Branch24->getType(), Type::getVoidTy(Ctx));
  EXPECT_TRUE(Branch24->isUnconditional());
  EXPECT_EQ(Branch24->getOperand(0), AtomicCmpxchgCmpxchgFailorderContinue56);

  // br label %atomic_cmpxchg.cmpxchg.memorder.continue49
  EXPECT_TRUE(Branch25->getName().empty());
  EXPECT_EQ(Branch25->getParent(), AtomicCmpxchgCmpxchgFailorderContinue56);
  EXPECT_EQ(Branch25->getType(), Type::getVoidTy(Ctx));
  EXPECT_TRUE(Branch25->isUnconditional());
  EXPECT_EQ(Branch25->getOperand(0), AtomicCmpxchgCmpxchgMemorderContinue49);

  // switch i32 %memorderarg_success, label %atomic_cmpxchg.cmpxchg.monoton...
  //   i32 1, label %atomic_cmpxchg.cmpxchg.acquire_fail69
  //   i32 2, label %atomic_cmpxchg.cmpxchg.acquire_fail69
  //   i32 5, label %atomic_cmpxchg.cmpxchg.seqcst_fail70
  // ]
  EXPECT_TRUE(Switch10->getName().empty());
  EXPECT_EQ(Switch10->getParent(), AtomicCmpxchgCmpxchgAcquire51);
  EXPECT_EQ(Switch10->getType(), Type::getVoidTy(Ctx));
  EXPECT_EQ(Switch10->getCondition(), MemorderArg);
  EXPECT_EQ(Switch10->getDefaultDest(), AtomicCmpxchgCmpxchgMonotonicFail68);
  EXPECT_EQ(cast<ConstantInt>(Switch10->getOperand(2))->getZExtValue(), 1);
  EXPECT_EQ(Switch10->getOperand(3), AtomicCmpxchgCmpxchgAcquireFail69);
  EXPECT_EQ(cast<ConstantInt>(Switch10->getOperand(4))->getZExtValue(), 2);
  EXPECT_EQ(Switch10->getOperand(5), AtomicCmpxchgCmpxchgAcquireFail69);
  EXPECT_EQ(cast<ConstantInt>(Switch10->getOperand(6))->getZExtValue(), 5);
  EXPECT_EQ(Switch10->getOperand(7), AtomicCmpxchgCmpxchgSeqcstFail70);

  // %atomic_cmpxchg.cmpxchg.pair72 = cmpxchg weak volatile ptr %atomic_ptr...
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair72->getName(),
            "atomic_cmpxchg.cmpxchg.pair72");
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair72->getParent(),
            AtomicCmpxchgCmpxchgMonotonicFail68);
  EXPECT_TRUE(AtomicCmpxchgCmpxchgPair72->isVolatile());
  EXPECT_TRUE(AtomicCmpxchgCmpxchgPair72->isWeak());
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair72->getSuccessOrdering(),
            AtomicOrdering::Acquire);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair72->getFailureOrdering(),
            AtomicOrdering::Monotonic);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair72->getSyncScopeID(), SyncScope::System);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair72->getAlign(), 1);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair72->getPointerOperand(), PtrArg);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair72->getCompareOperand(),
            AtomicCmpxchgCmpxchgExpected);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair72->getNewValOperand(),
            AtomicCmpxchgCmpxchgDesired);

  // br label %atomic_cmpxchg.cmpxchg.failorder.continue67
  EXPECT_TRUE(Branch26->getName().empty());
  EXPECT_EQ(Branch26->getParent(), AtomicCmpxchgCmpxchgMonotonicFail68);
  EXPECT_EQ(Branch26->getType(), Type::getVoidTy(Ctx));
  EXPECT_TRUE(Branch26->isUnconditional());
  EXPECT_EQ(Branch26->getOperand(0), AtomicCmpxchgCmpxchgFailorderContinue67);

  // %atomic_cmpxchg.cmpxchg.pair74 = cmpxchg weak volatile ptr %atomic_ptr...
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair74->getName(),
            "atomic_cmpxchg.cmpxchg.pair74");
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair74->getParent(),
            AtomicCmpxchgCmpxchgAcquireFail69);
  EXPECT_TRUE(AtomicCmpxchgCmpxchgPair74->isVolatile());
  EXPECT_TRUE(AtomicCmpxchgCmpxchgPair74->isWeak());
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair74->getSuccessOrdering(),
            AtomicOrdering::Acquire);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair74->getFailureOrdering(),
            AtomicOrdering::Acquire);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair74->getSyncScopeID(), SyncScope::System);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair74->getAlign(), 1);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair74->getPointerOperand(), PtrArg);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair74->getCompareOperand(),
            AtomicCmpxchgCmpxchgExpected);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair74->getNewValOperand(),
            AtomicCmpxchgCmpxchgDesired);

  // br label %atomic_cmpxchg.cmpxchg.failorder.continue67
  EXPECT_TRUE(Branch27->getName().empty());
  EXPECT_EQ(Branch27->getParent(), AtomicCmpxchgCmpxchgAcquireFail69);
  EXPECT_EQ(Branch27->getType(), Type::getVoidTy(Ctx));
  EXPECT_TRUE(Branch27->isUnconditional());
  EXPECT_EQ(Branch27->getOperand(0), AtomicCmpxchgCmpxchgFailorderContinue67);

  // %atomic_cmpxchg.cmpxchg.pair76 = cmpxchg weak volatile ptr %atomic_ptr...
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair76->getName(),
            "atomic_cmpxchg.cmpxchg.pair76");
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair76->getParent(),
            AtomicCmpxchgCmpxchgSeqcstFail70);
  EXPECT_TRUE(AtomicCmpxchgCmpxchgPair76->isVolatile());
  EXPECT_TRUE(AtomicCmpxchgCmpxchgPair76->isWeak());
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair76->getSuccessOrdering(),
            AtomicOrdering::Acquire);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair76->getFailureOrdering(),
            AtomicOrdering::SequentiallyConsistent);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair76->getSyncScopeID(), SyncScope::System);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair76->getAlign(), 1);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair76->getPointerOperand(), PtrArg);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair76->getCompareOperand(),
            AtomicCmpxchgCmpxchgExpected);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair76->getNewValOperand(),
            AtomicCmpxchgCmpxchgDesired);

  // br label %atomic_cmpxchg.cmpxchg.failorder.continue67
  EXPECT_TRUE(Branch28->getName().empty());
  EXPECT_EQ(Branch28->getParent(), AtomicCmpxchgCmpxchgSeqcstFail70);
  EXPECT_EQ(Branch28->getType(), Type::getVoidTy(Ctx));
  EXPECT_TRUE(Branch28->isUnconditional());
  EXPECT_EQ(Branch28->getOperand(0), AtomicCmpxchgCmpxchgFailorderContinue67);

  // br label %atomic_cmpxchg.cmpxchg.memorder.continue49
  EXPECT_TRUE(Branch29->getName().empty());
  EXPECT_EQ(Branch29->getParent(), AtomicCmpxchgCmpxchgFailorderContinue67);
  EXPECT_EQ(Branch29->getType(), Type::getVoidTy(Ctx));
  EXPECT_TRUE(Branch29->isUnconditional());
  EXPECT_EQ(Branch29->getOperand(0), AtomicCmpxchgCmpxchgMemorderContinue49);

  // switch i32 %memorderarg_success, label %atomic_cmpxchg.cmpxchg.monoton...
  //   i32 1, label %atomic_cmpxchg.cmpxchg.acquire_fail80
  //   i32 2, label %atomic_cmpxchg.cmpxchg.acquire_fail80
  //   i32 5, label %atomic_cmpxchg.cmpxchg.seqcst_fail81
  // ]
  EXPECT_TRUE(Switch11->getName().empty());
  EXPECT_EQ(Switch11->getParent(), AtomicCmpxchgCmpxchgRelease52);
  EXPECT_EQ(Switch11->getType(), Type::getVoidTy(Ctx));
  EXPECT_EQ(Switch11->getCondition(), MemorderArg);
  EXPECT_EQ(Switch11->getDefaultDest(), AtomicCmpxchgCmpxchgMonotonicFail79);
  EXPECT_EQ(cast<ConstantInt>(Switch11->getOperand(2))->getZExtValue(), 1);
  EXPECT_EQ(Switch11->getOperand(3), AtomicCmpxchgCmpxchgAcquireFail80);
  EXPECT_EQ(cast<ConstantInt>(Switch11->getOperand(4))->getZExtValue(), 2);
  EXPECT_EQ(Switch11->getOperand(5), AtomicCmpxchgCmpxchgAcquireFail80);
  EXPECT_EQ(cast<ConstantInt>(Switch11->getOperand(6))->getZExtValue(), 5);
  EXPECT_EQ(Switch11->getOperand(7), AtomicCmpxchgCmpxchgSeqcstFail81);

  // %atomic_cmpxchg.cmpxchg.pair83 = cmpxchg weak volatile ptr %atomic_ptr...
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair83->getName(),
            "atomic_cmpxchg.cmpxchg.pair83");
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair83->getParent(),
            AtomicCmpxchgCmpxchgMonotonicFail79);
  EXPECT_TRUE(AtomicCmpxchgCmpxchgPair83->isVolatile());
  EXPECT_TRUE(AtomicCmpxchgCmpxchgPair83->isWeak());
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair83->getSuccessOrdering(),
            AtomicOrdering::Release);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair83->getFailureOrdering(),
            AtomicOrdering::Monotonic);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair83->getSyncScopeID(), SyncScope::System);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair83->getAlign(), 1);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair83->getPointerOperand(), PtrArg);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair83->getCompareOperand(),
            AtomicCmpxchgCmpxchgExpected);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair83->getNewValOperand(),
            AtomicCmpxchgCmpxchgDesired);

  // br label %atomic_cmpxchg.cmpxchg.failorder.continue78
  EXPECT_TRUE(Branch30->getName().empty());
  EXPECT_EQ(Branch30->getParent(), AtomicCmpxchgCmpxchgMonotonicFail79);
  EXPECT_EQ(Branch30->getType(), Type::getVoidTy(Ctx));
  EXPECT_TRUE(Branch30->isUnconditional());
  EXPECT_EQ(Branch30->getOperand(0), AtomicCmpxchgCmpxchgFailorderContinue78);

  // %atomic_cmpxchg.cmpxchg.pair85 = cmpxchg weak volatile ptr %atomic_ptr...
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair85->getName(),
            "atomic_cmpxchg.cmpxchg.pair85");
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair85->getParent(),
            AtomicCmpxchgCmpxchgAcquireFail80);
  EXPECT_TRUE(AtomicCmpxchgCmpxchgPair85->isVolatile());
  EXPECT_TRUE(AtomicCmpxchgCmpxchgPair85->isWeak());
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair85->getSuccessOrdering(),
            AtomicOrdering::Release);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair85->getFailureOrdering(),
            AtomicOrdering::Acquire);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair85->getSyncScopeID(), SyncScope::System);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair85->getAlign(), 1);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair85->getPointerOperand(), PtrArg);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair85->getCompareOperand(),
            AtomicCmpxchgCmpxchgExpected);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair85->getNewValOperand(),
            AtomicCmpxchgCmpxchgDesired);

  // br label %atomic_cmpxchg.cmpxchg.failorder.continue78
  EXPECT_TRUE(Branch31->getName().empty());
  EXPECT_EQ(Branch31->getParent(), AtomicCmpxchgCmpxchgAcquireFail80);
  EXPECT_EQ(Branch31->getType(), Type::getVoidTy(Ctx));
  EXPECT_TRUE(Branch31->isUnconditional());
  EXPECT_EQ(Branch31->getOperand(0), AtomicCmpxchgCmpxchgFailorderContinue78);

  // %atomic_cmpxchg.cmpxchg.pair87 = cmpxchg weak volatile ptr %atomic_ptr...
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair87->getName(),
            "atomic_cmpxchg.cmpxchg.pair87");
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair87->getParent(),
            AtomicCmpxchgCmpxchgSeqcstFail81);
  EXPECT_TRUE(AtomicCmpxchgCmpxchgPair87->isVolatile());
  EXPECT_TRUE(AtomicCmpxchgCmpxchgPair87->isWeak());
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair87->getSuccessOrdering(),
            AtomicOrdering::Release);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair87->getFailureOrdering(),
            AtomicOrdering::SequentiallyConsistent);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair87->getSyncScopeID(), SyncScope::System);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair87->getAlign(), 1);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair87->getPointerOperand(), PtrArg);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair87->getCompareOperand(),
            AtomicCmpxchgCmpxchgExpected);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair87->getNewValOperand(),
            AtomicCmpxchgCmpxchgDesired);

  // br label %atomic_cmpxchg.cmpxchg.failorder.continue78
  EXPECT_TRUE(Branch32->getName().empty());
  EXPECT_EQ(Branch32->getParent(), AtomicCmpxchgCmpxchgSeqcstFail81);
  EXPECT_EQ(Branch32->getType(), Type::getVoidTy(Ctx));
  EXPECT_TRUE(Branch32->isUnconditional());
  EXPECT_EQ(Branch32->getOperand(0), AtomicCmpxchgCmpxchgFailorderContinue78);

  // br label %atomic_cmpxchg.cmpxchg.memorder.continue49
  EXPECT_TRUE(Branch33->getName().empty());
  EXPECT_EQ(Branch33->getParent(), AtomicCmpxchgCmpxchgFailorderContinue78);
  EXPECT_EQ(Branch33->getType(), Type::getVoidTy(Ctx));
  EXPECT_TRUE(Branch33->isUnconditional());
  EXPECT_EQ(Branch33->getOperand(0), AtomicCmpxchgCmpxchgMemorderContinue49);

  // switch i32 %memorderarg_success, label %atomic_cmpxchg.cmpxchg.monoton...
  //   i32 1, label %atomic_cmpxchg.cmpxchg.acquire_fail91
  //   i32 2, label %atomic_cmpxchg.cmpxchg.acquire_fail91
  //   i32 5, label %atomic_cmpxchg.cmpxchg.seqcst_fail92
  // ]
  EXPECT_TRUE(Switch12->getName().empty());
  EXPECT_EQ(Switch12->getParent(), AtomicCmpxchgCmpxchgAcqrel53);
  EXPECT_EQ(Switch12->getType(), Type::getVoidTy(Ctx));
  EXPECT_EQ(Switch12->getCondition(), MemorderArg);
  EXPECT_EQ(Switch12->getDefaultDest(), AtomicCmpxchgCmpxchgMonotonicFail90);
  EXPECT_EQ(cast<ConstantInt>(Switch12->getOperand(2))->getZExtValue(), 1);
  EXPECT_EQ(Switch12->getOperand(3), AtomicCmpxchgCmpxchgAcquireFail91);
  EXPECT_EQ(cast<ConstantInt>(Switch12->getOperand(4))->getZExtValue(), 2);
  EXPECT_EQ(Switch12->getOperand(5), AtomicCmpxchgCmpxchgAcquireFail91);
  EXPECT_EQ(cast<ConstantInt>(Switch12->getOperand(6))->getZExtValue(), 5);
  EXPECT_EQ(Switch12->getOperand(7), AtomicCmpxchgCmpxchgSeqcstFail92);

  // %atomic_cmpxchg.cmpxchg.pair94 = cmpxchg weak volatile ptr %atomic_ptr...
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair94->getName(),
            "atomic_cmpxchg.cmpxchg.pair94");
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair94->getParent(),
            AtomicCmpxchgCmpxchgMonotonicFail90);
  EXPECT_TRUE(AtomicCmpxchgCmpxchgPair94->isVolatile());
  EXPECT_TRUE(AtomicCmpxchgCmpxchgPair94->isWeak());
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair94->getSuccessOrdering(),
            AtomicOrdering::AcquireRelease);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair94->getFailureOrdering(),
            AtomicOrdering::Monotonic);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair94->getSyncScopeID(), SyncScope::System);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair94->getAlign(), 1);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair94->getPointerOperand(), PtrArg);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair94->getCompareOperand(),
            AtomicCmpxchgCmpxchgExpected);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair94->getNewValOperand(),
            AtomicCmpxchgCmpxchgDesired);

  // br label %atomic_cmpxchg.cmpxchg.failorder.continue89
  EXPECT_TRUE(Branch34->getName().empty());
  EXPECT_EQ(Branch34->getParent(), AtomicCmpxchgCmpxchgMonotonicFail90);
  EXPECT_EQ(Branch34->getType(), Type::getVoidTy(Ctx));
  EXPECT_TRUE(Branch34->isUnconditional());
  EXPECT_EQ(Branch34->getOperand(0), AtomicCmpxchgCmpxchgFailorderContinue89);

  // %atomic_cmpxchg.cmpxchg.pair96 = cmpxchg weak volatile ptr %atomic_ptr...
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair96->getName(),
            "atomic_cmpxchg.cmpxchg.pair96");
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair96->getParent(),
            AtomicCmpxchgCmpxchgAcquireFail91);
  EXPECT_TRUE(AtomicCmpxchgCmpxchgPair96->isVolatile());
  EXPECT_TRUE(AtomicCmpxchgCmpxchgPair96->isWeak());
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair96->getSuccessOrdering(),
            AtomicOrdering::AcquireRelease);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair96->getFailureOrdering(),
            AtomicOrdering::Acquire);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair96->getSyncScopeID(), SyncScope::System);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair96->getAlign(), 1);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair96->getPointerOperand(), PtrArg);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair96->getCompareOperand(),
            AtomicCmpxchgCmpxchgExpected);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair96->getNewValOperand(),
            AtomicCmpxchgCmpxchgDesired);

  // br label %atomic_cmpxchg.cmpxchg.failorder.continue89
  EXPECT_TRUE(Branch35->getName().empty());
  EXPECT_EQ(Branch35->getParent(), AtomicCmpxchgCmpxchgAcquireFail91);
  EXPECT_EQ(Branch35->getType(), Type::getVoidTy(Ctx));
  EXPECT_TRUE(Branch35->isUnconditional());
  EXPECT_EQ(Branch35->getOperand(0), AtomicCmpxchgCmpxchgFailorderContinue89);

  // %atomic_cmpxchg.cmpxchg.pair98 = cmpxchg weak volatile ptr %atomic_ptr...
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair98->getName(),
            "atomic_cmpxchg.cmpxchg.pair98");
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair98->getParent(),
            AtomicCmpxchgCmpxchgSeqcstFail92);
  EXPECT_TRUE(AtomicCmpxchgCmpxchgPair98->isVolatile());
  EXPECT_TRUE(AtomicCmpxchgCmpxchgPair98->isWeak());
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair98->getSuccessOrdering(),
            AtomicOrdering::AcquireRelease);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair98->getFailureOrdering(),
            AtomicOrdering::SequentiallyConsistent);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair98->getSyncScopeID(), SyncScope::System);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair98->getAlign(), 1);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair98->getPointerOperand(), PtrArg);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair98->getCompareOperand(),
            AtomicCmpxchgCmpxchgExpected);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair98->getNewValOperand(),
            AtomicCmpxchgCmpxchgDesired);

  // br label %atomic_cmpxchg.cmpxchg.failorder.continue89
  EXPECT_TRUE(Branch36->getName().empty());
  EXPECT_EQ(Branch36->getParent(), AtomicCmpxchgCmpxchgSeqcstFail92);
  EXPECT_EQ(Branch36->getType(), Type::getVoidTy(Ctx));
  EXPECT_TRUE(Branch36->isUnconditional());
  EXPECT_EQ(Branch36->getOperand(0), AtomicCmpxchgCmpxchgFailorderContinue89);

  // br label %atomic_cmpxchg.cmpxchg.memorder.continue49
  EXPECT_TRUE(Branch37->getName().empty());
  EXPECT_EQ(Branch37->getParent(), AtomicCmpxchgCmpxchgFailorderContinue89);
  EXPECT_EQ(Branch37->getType(), Type::getVoidTy(Ctx));
  EXPECT_TRUE(Branch37->isUnconditional());
  EXPECT_EQ(Branch37->getOperand(0), AtomicCmpxchgCmpxchgMemorderContinue49);

  // switch i32 %memorderarg_success, label %atomic_cmpxchg.cmpxchg.monoton...
  //   i32 1, label %atomic_cmpxchg.cmpxchg.acquire_fail102
  //   i32 2, label %atomic_cmpxchg.cmpxchg.acquire_fail102
  //   i32 5, label %atomic_cmpxchg.cmpxchg.seqcst_fail103
  // ]
  EXPECT_TRUE(Switch13->getName().empty());
  EXPECT_EQ(Switch13->getParent(), AtomicCmpxchgCmpxchgSeqcst54);
  EXPECT_EQ(Switch13->getType(), Type::getVoidTy(Ctx));
  EXPECT_EQ(Switch13->getCondition(), MemorderArg);
  EXPECT_EQ(Switch13->getDefaultDest(), AtomicCmpxchgCmpxchgMonotonicFail101);
  EXPECT_EQ(cast<ConstantInt>(Switch13->getOperand(2))->getZExtValue(), 1);
  EXPECT_EQ(Switch13->getOperand(3), AtomicCmpxchgCmpxchgAcquireFail102);
  EXPECT_EQ(cast<ConstantInt>(Switch13->getOperand(4))->getZExtValue(), 2);
  EXPECT_EQ(Switch13->getOperand(5), AtomicCmpxchgCmpxchgAcquireFail102);
  EXPECT_EQ(cast<ConstantInt>(Switch13->getOperand(6))->getZExtValue(), 5);
  EXPECT_EQ(Switch13->getOperand(7), AtomicCmpxchgCmpxchgSeqcstFail103);

  // %atomic_cmpxchg.cmpxchg.pair105 = cmpxchg weak volatile ptr %atomic_pt...
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair105->getName(),
            "atomic_cmpxchg.cmpxchg.pair105");
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair105->getParent(),
            AtomicCmpxchgCmpxchgMonotonicFail101);
  EXPECT_TRUE(AtomicCmpxchgCmpxchgPair105->isVolatile());
  EXPECT_TRUE(AtomicCmpxchgCmpxchgPair105->isWeak());
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair105->getSuccessOrdering(),
            AtomicOrdering::SequentiallyConsistent);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair105->getFailureOrdering(),
            AtomicOrdering::Monotonic);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair105->getSyncScopeID(), SyncScope::System);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair105->getAlign(), 1);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair105->getPointerOperand(), PtrArg);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair105->getCompareOperand(),
            AtomicCmpxchgCmpxchgExpected);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair105->getNewValOperand(),
            AtomicCmpxchgCmpxchgDesired);

  // br label %atomic_cmpxchg.cmpxchg.failorder.continue100
  EXPECT_TRUE(Branch38->getName().empty());
  EXPECT_EQ(Branch38->getParent(), AtomicCmpxchgCmpxchgMonotonicFail101);
  EXPECT_EQ(Branch38->getType(), Type::getVoidTy(Ctx));
  EXPECT_TRUE(Branch38->isUnconditional());
  EXPECT_EQ(Branch38->getOperand(0), AtomicCmpxchgCmpxchgFailorderContinue100);

  // %atomic_cmpxchg.cmpxchg.pair107 = cmpxchg weak volatile ptr %atomic_pt...
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair107->getName(),
            "atomic_cmpxchg.cmpxchg.pair107");
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair107->getParent(),
            AtomicCmpxchgCmpxchgAcquireFail102);
  EXPECT_TRUE(AtomicCmpxchgCmpxchgPair107->isVolatile());
  EXPECT_TRUE(AtomicCmpxchgCmpxchgPair107->isWeak());
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair107->getSuccessOrdering(),
            AtomicOrdering::SequentiallyConsistent);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair107->getFailureOrdering(),
            AtomicOrdering::Acquire);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair107->getSyncScopeID(), SyncScope::System);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair107->getAlign(), 1);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair107->getPointerOperand(), PtrArg);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair107->getCompareOperand(),
            AtomicCmpxchgCmpxchgExpected);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair107->getNewValOperand(),
            AtomicCmpxchgCmpxchgDesired);

  // br label %atomic_cmpxchg.cmpxchg.failorder.continue100
  EXPECT_TRUE(Branch39->getName().empty());
  EXPECT_EQ(Branch39->getParent(), AtomicCmpxchgCmpxchgAcquireFail102);
  EXPECT_EQ(Branch39->getType(), Type::getVoidTy(Ctx));
  EXPECT_TRUE(Branch39->isUnconditional());
  EXPECT_EQ(Branch39->getOperand(0), AtomicCmpxchgCmpxchgFailorderContinue100);

  // %atomic_cmpxchg.cmpxchg.pair109 = cmpxchg weak volatile ptr %atomic_pt...
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair109->getName(),
            "atomic_cmpxchg.cmpxchg.pair109");
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair109->getParent(),
            AtomicCmpxchgCmpxchgSeqcstFail103);
  EXPECT_TRUE(AtomicCmpxchgCmpxchgPair109->isVolatile());
  EXPECT_TRUE(AtomicCmpxchgCmpxchgPair109->isWeak());
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair109->getSuccessOrdering(),
            AtomicOrdering::SequentiallyConsistent);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair109->getFailureOrdering(),
            AtomicOrdering::SequentiallyConsistent);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair109->getSyncScopeID(), SyncScope::System);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair109->getAlign(), 1);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair109->getPointerOperand(), PtrArg);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair109->getCompareOperand(),
            AtomicCmpxchgCmpxchgExpected);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair109->getNewValOperand(),
            AtomicCmpxchgCmpxchgDesired);

  // br label %atomic_cmpxchg.cmpxchg.failorder.continue100
  EXPECT_TRUE(Branch40->getName().empty());
  EXPECT_EQ(Branch40->getParent(), AtomicCmpxchgCmpxchgSeqcstFail103);
  EXPECT_EQ(Branch40->getType(), Type::getVoidTy(Ctx));
  EXPECT_TRUE(Branch40->isUnconditional());
  EXPECT_EQ(Branch40->getOperand(0), AtomicCmpxchgCmpxchgFailorderContinue100);

  // br label %atomic_cmpxchg.cmpxchg.memorder.continue49
  EXPECT_TRUE(Branch41->getName().empty());
  EXPECT_EQ(Branch41->getParent(), AtomicCmpxchgCmpxchgFailorderContinue100);
  EXPECT_EQ(Branch41->getType(), Type::getVoidTy(Ctx));
  EXPECT_TRUE(Branch41->isUnconditional());
  EXPECT_EQ(Branch41->getOperand(0), AtomicCmpxchgCmpxchgMemorderContinue49);

  // br label %atomic_cmpxchg.cmpxchg.weak.continue
  EXPECT_TRUE(Branch42->getName().empty());
  EXPECT_EQ(Branch42->getParent(), AtomicCmpxchgCmpxchgMemorderContinue49);
  EXPECT_EQ(Branch42->getType(), Type::getVoidTy(Ctx));
  EXPECT_TRUE(Branch42->isUnconditional());
  EXPECT_EQ(Branch42->getOperand(0), ExitBB);

  // %atomic_cmpxchg.cmpxchg.isweak.success = phi i1 [ %atomic_cmpxchg.cmpx...
  EXPECT_EQ(AtomicSuccess->getName(), "atomic_cmpxchg.cmpxchg.isweak.success");
  EXPECT_EQ(cast<Instruction>(AtomicSuccess)->getParent(), ExitBB);
  EXPECT_EQ(AtomicSuccess->getType(), Type::getInt1Ty(Ctx));
  EXPECT_TRUE(isa<PHINode>(cast<Instruction>(AtomicSuccess)->getOperand(0)));
  EXPECT_TRUE(isa<PHINode>(cast<Instruction>(AtomicSuccess)->getOperand(1)));

  // ret void
  EXPECT_TRUE(Return->getName().empty());
  EXPECT_EQ(Return->getParent(), ExitBB);
  EXPECT_EQ(Return->getType(), Type::getVoidTy(Ctx));
}

TEST_F(BuildBuiltinsTests, AtomicCmpxchg_SyncScope) {
  Value *AtomicSuccess = nullptr;
  ASSERT_THAT_EXPECTED(
      emitAtomicCompareExchangeBuiltin(
          /*AtomicPtr=*/PtrArg,
          /*ExpectedPtr=*/ExpectedArg,
          /*DesiredPtr=*/DesiredArg, /*TypeOrSize=*/Builder.getInt32Ty(),
          /*IsWeak*/ false,
          /*IsVolatile=*/false,
          /*SuccessMemorder=*/AtomicOrdering::SequentiallyConsistent,
          /*FailureMemorder=*/AtomicOrdering::SequentiallyConsistent,
          /*Scope=*/SyncScope::SingleThread,
          /*PrevPtr=*/nullptr,
          /*Align=*/{}, /*Builder=*/Builder,
          /*EmitOptions=*/AtomicEmitOptions(DL, TLI.get()),
          /*Name=*/"atomic_cmpxchg"),
      StoreResult(AtomicSuccess));
  EXPECT_FALSE(verifyModule(*M, &errs()));

  // clang-format off
  // %atomic_cmpxchg.cmpxchg.expected = load i32, ptr %expected_ptr, align 4
  // %atomic_cmpxchg.cmpxchg.desired = load i32, ptr %desired_ptr, align 4
  // %atomic_cmpxchg.cmpxchg.pair = cmpxchg ptr %atomic_ptr, i32 %atomic_cmpxchg.cmpxchg.expected, i32 %atomic_cmpxchg.cmpxchg.desired syncscope("singlethread") seq_cst seq_cst, align 1
  // %atomic_cmpxchg.cmpxchg.success = extractvalue { i32, i1 } %atomic_cmpxchg.cmpxchg.pair, 1
  // clang-format on

  // Follow use-def and load-store chains to discover instructions
  AtomicCmpXchgInst *AtomicCmpxchgCmpxchgPair =
      cast<AtomicCmpXchgInst>(getUniquePreviousStore(PtrArg, EntryBB));
  LoadInst *AtomicCmpxchgCmpxchgExpected =
      cast<LoadInst>(AtomicCmpxchgCmpxchgPair->getCompareOperand());
  LoadInst *AtomicCmpxchgCmpxchgDesired =
      cast<LoadInst>(AtomicCmpxchgCmpxchgPair->getNewValOperand());

  // %atomic_cmpxchg.cmpxchg.expected = load i32, ptr %expected_ptr, align 4
  EXPECT_EQ(AtomicCmpxchgCmpxchgExpected->getName(),
            "atomic_cmpxchg.cmpxchg.expected");
  EXPECT_EQ(AtomicCmpxchgCmpxchgExpected->getParent(), EntryBB);
  EXPECT_EQ(AtomicCmpxchgCmpxchgExpected->getType(), Type::getInt32Ty(Ctx));
  EXPECT_TRUE(AtomicCmpxchgCmpxchgExpected->isSimple());
  EXPECT_EQ(AtomicCmpxchgCmpxchgExpected->getPointerOperand(), ExpectedArg);

  // %atomic_cmpxchg.cmpxchg.desired = load i32, ptr %desired_ptr, align 4
  EXPECT_EQ(AtomicCmpxchgCmpxchgDesired->getName(),
            "atomic_cmpxchg.cmpxchg.desired");
  EXPECT_EQ(AtomicCmpxchgCmpxchgDesired->getParent(), EntryBB);
  EXPECT_EQ(AtomicCmpxchgCmpxchgDesired->getType(), Type::getInt32Ty(Ctx));
  EXPECT_TRUE(AtomicCmpxchgCmpxchgDesired->isSimple());
  EXPECT_EQ(AtomicCmpxchgCmpxchgDesired->getPointerOperand(), DesiredArg);

  // %atomic_cmpxchg.cmpxchg.pair = cmpxchg ptr %atomic_ptr, i32 %atomic_cm...
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getName(), "atomic_cmpxchg.cmpxchg.pair");
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getParent(), EntryBB);
  EXPECT_FALSE(AtomicCmpxchgCmpxchgPair->isVolatile());
  EXPECT_FALSE(AtomicCmpxchgCmpxchgPair->isWeak());
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getSuccessOrdering(),
            AtomicOrdering::SequentiallyConsistent);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getFailureOrdering(),
            AtomicOrdering::SequentiallyConsistent);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getSyncScopeID(),
            SyncScope::SingleThread);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getAlign(), 1);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getPointerOperand(), PtrArg);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getCompareOperand(),
            AtomicCmpxchgCmpxchgExpected);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getNewValOperand(),
            AtomicCmpxchgCmpxchgDesired);

  // %atomic_cmpxchg.cmpxchg.success = extractvalue { i32, i1 } %atomic_cmp...
  EXPECT_EQ(AtomicSuccess->getName(), "atomic_cmpxchg.cmpxchg.success");
  EXPECT_EQ(cast<Instruction>(AtomicSuccess)->getParent(), EntryBB);
  EXPECT_EQ(AtomicSuccess->getType(), Type::getInt1Ty(Ctx));
  EXPECT_EQ(cast<ExtractValueInst>(AtomicSuccess)->getNumIndices(), 1);
  EXPECT_EQ(cast<ExtractValueInst>(AtomicSuccess)->getIndices()[0], 1);
  EXPECT_EQ(cast<ExtractValueInst>(AtomicSuccess)->getAggregateOperand(),
            AtomicCmpxchgCmpxchgPair);
}

TEST_F(BuildBuiltinsTests, AtomicCmpxchg_Float) {
  Value *AtomicSuccess = nullptr;
  ASSERT_THAT_EXPECTED(
      emitAtomicCompareExchangeBuiltin(
          /*AtomicPtr=*/PtrArg,
          /*ExpectedPtr=*/ExpectedArg,
          /*DesiredPtr=*/DesiredArg, /*TypeOrSize=*/Builder.getFloatTy(),
          /*IsWeak*/ false,
          /*IsVolatile=*/false,
          /*SuccessMemorder=*/AtomicOrdering::SequentiallyConsistent,
          /*FailureMemorder=*/AtomicOrdering::SequentiallyConsistent,
          /*Scope=*/SyncScope::System,
          /*PrevPtr=*/nullptr,
          /*Align=*/{}, /*Builder=*/Builder,
          /*EmitOptions=*/AtomicEmitOptions(DL, TLI.get()),
          /*Name=*/"atomic_cmpxchg"),
      StoreResult(AtomicSuccess));
  EXPECT_FALSE(verifyModule(*M, &errs()));

  // clang-format off
  // %atomic_cmpxchg.cmpxchg.expected = load i32, ptr %expected_ptr, align 4
  // %atomic_cmpxchg.cmpxchg.desired = load i32, ptr %desired_ptr, align 4
  // %atomic_cmpxchg.cmpxchg.pair = cmpxchg ptr %atomic_ptr, i32 %atomic_cmpxchg.cmpxchg.expected, i32 %atomic_cmpxchg.cmpxchg.desired seq_cst seq_cst, align 1
  // %atomic_cmpxchg.cmpxchg.success = extractvalue { i32, i1 } %atomic_cmpxchg.cmpxchg.pair, 1
  // clang-format on

  // Follow use-def and load-store chains to discover instructions
  AtomicCmpXchgInst *AtomicCmpxchgCmpxchgPair =
      cast<AtomicCmpXchgInst>(getUniquePreviousStore(PtrArg, EntryBB));
  LoadInst *AtomicCmpxchgCmpxchgExpected =
      cast<LoadInst>(AtomicCmpxchgCmpxchgPair->getCompareOperand());
  LoadInst *AtomicCmpxchgCmpxchgDesired =
      cast<LoadInst>(AtomicCmpxchgCmpxchgPair->getNewValOperand());

  // %atomic_cmpxchg.cmpxchg.expected = load i32, ptr %expected_ptr, align 4
  EXPECT_EQ(AtomicCmpxchgCmpxchgExpected->getName(),
            "atomic_cmpxchg.cmpxchg.expected");
  EXPECT_EQ(AtomicCmpxchgCmpxchgExpected->getParent(), EntryBB);
  EXPECT_EQ(AtomicCmpxchgCmpxchgExpected->getType(), Type::getInt32Ty(Ctx));
  EXPECT_TRUE(AtomicCmpxchgCmpxchgExpected->isSimple());
  EXPECT_EQ(AtomicCmpxchgCmpxchgExpected->getPointerOperand(), ExpectedArg);

  // %atomic_cmpxchg.cmpxchg.desired = load i32, ptr %desired_ptr, align 4
  EXPECT_EQ(AtomicCmpxchgCmpxchgDesired->getName(),
            "atomic_cmpxchg.cmpxchg.desired");
  EXPECT_EQ(AtomicCmpxchgCmpxchgDesired->getParent(), EntryBB);
  EXPECT_EQ(AtomicCmpxchgCmpxchgDesired->getType(), Type::getInt32Ty(Ctx));
  EXPECT_TRUE(AtomicCmpxchgCmpxchgDesired->isSimple());
  EXPECT_EQ(AtomicCmpxchgCmpxchgDesired->getPointerOperand(), DesiredArg);

  // %atomic_cmpxchg.cmpxchg.pair = cmpxchg ptr %atomic_ptr, i32 %atomic_cm...
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getName(), "atomic_cmpxchg.cmpxchg.pair");
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getParent(), EntryBB);
  EXPECT_FALSE(AtomicCmpxchgCmpxchgPair->isVolatile());
  EXPECT_FALSE(AtomicCmpxchgCmpxchgPair->isWeak());
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getSuccessOrdering(),
            AtomicOrdering::SequentiallyConsistent);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getFailureOrdering(),
            AtomicOrdering::SequentiallyConsistent);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getSyncScopeID(), SyncScope::System);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getAlign(), 1);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getPointerOperand(), PtrArg);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getCompareOperand(),
            AtomicCmpxchgCmpxchgExpected);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getNewValOperand(),
            AtomicCmpxchgCmpxchgDesired);

  // %atomic_cmpxchg.cmpxchg.success = extractvalue { i32, i1 } %atomic_cmp...
  EXPECT_EQ(AtomicSuccess->getName(), "atomic_cmpxchg.cmpxchg.success");
  EXPECT_EQ(cast<Instruction>(AtomicSuccess)->getParent(), EntryBB);
  EXPECT_EQ(AtomicSuccess->getType(), Type::getInt1Ty(Ctx));
  EXPECT_EQ(cast<ExtractValueInst>(AtomicSuccess)->getNumIndices(), 1);
  EXPECT_EQ(cast<ExtractValueInst>(AtomicSuccess)->getIndices()[0], 1);
  EXPECT_EQ(cast<ExtractValueInst>(AtomicSuccess)->getAggregateOperand(),
            AtomicCmpxchgCmpxchgPair);
}

TEST_F(BuildBuiltinsTests, AtomicCmpxchg_FP80) {
  Value *AtomicSuccess = nullptr;
  ASSERT_THAT_EXPECTED(
      emitAtomicCompareExchangeBuiltin(
          /*AtomicPtr=*/PtrArg,
          /*ExpectedPtr=*/ExpectedArg,
          /*DesiredPtr=*/DesiredArg, /*TypeOrSize=*/Type::getX86_FP80Ty(Ctx),
          /*IsWeak*/ false,
          /*IsVolatile=*/false,
          /*SuccessMemorder=*/AtomicOrdering::SequentiallyConsistent,
          /*FailureMemorder=*/AtomicOrdering::SequentiallyConsistent,
          /*Scope=*/SyncScope::System,
          /*PrevPtr=*/nullptr,
          /*Align=*/{}, /*Builder=*/Builder,
          /*EmitOptions=*/AtomicEmitOptions(DL, TLI.get()),
          /*Name=*/"atomic_cmpxchg"),
      StoreResult(AtomicSuccess));
  EXPECT_FALSE(verifyModule(*M, &errs()));

  // clang-format off
  // %__atomic_compare_exchange = call i8 @__atomic_compare_exchange(i64 10, ptr %atomic_ptr, ptr %expected_ptr, ptr %desired_ptr, i32 5, i32 5)
  // %atomic_cmpxchg.cmpxchg.success = icmp eq i8 %__atomic_compare_exchange, 0
  // clang-format on

  // Follow use-def and load-store chains to discover instructions
  CallInst *AtomicCompareExchange =
      cast<CallInst>(getUniquePreviousStore(PtrArg, EntryBB));

  // %__atomic_compare_exchange = call i8 @__atomic_compare_exchange(i64 10...
  EXPECT_EQ(AtomicCompareExchange->getName(), "__atomic_compare_exchange");
  EXPECT_EQ(AtomicCompareExchange->getParent(), EntryBB);
  EXPECT_EQ(AtomicCompareExchange->getType(), Type::getInt8Ty(Ctx));
  EXPECT_EQ(AtomicCompareExchange->getName(), "__atomic_compare_exchange");
  EXPECT_FALSE(AtomicCompareExchange->isMustTailCall());
  EXPECT_FALSE(AtomicCompareExchange->isTailCall());
  EXPECT_EQ(AtomicCompareExchange->getCallingConv(), CallingConv::C);
  EXPECT_EQ(cast<ConstantInt>(AtomicCompareExchange->getArgOperand(0))
                ->getZExtValue(),
            10);
  EXPECT_EQ(AtomicCompareExchange->getArgOperand(1), PtrArg);
  EXPECT_EQ(AtomicCompareExchange->getArgOperand(2), ExpectedArg);
  EXPECT_EQ(AtomicCompareExchange->getArgOperand(3), DesiredArg);
  EXPECT_EQ(cast<ConstantInt>(AtomicCompareExchange->getArgOperand(4))
                ->getZExtValue(),
            5);
  EXPECT_EQ(cast<ConstantInt>(AtomicCompareExchange->getArgOperand(5))
                ->getZExtValue(),
            5);
  EXPECT_EQ(AtomicCompareExchange->getCalledFunction(),
            M->getFunction("__atomic_compare_exchange"));

  // %atomic_cmpxchg.cmpxchg.success = icmp eq i8 %__atomic_compare_exchang...
  EXPECT_EQ(AtomicSuccess->getName(), "atomic_cmpxchg.cmpxchg.success");
  EXPECT_EQ(cast<Instruction>(AtomicSuccess)->getParent(), EntryBB);
  EXPECT_EQ(AtomicSuccess->getType(), Type::getInt1Ty(Ctx));
  EXPECT_EQ(cast<Instruction>(AtomicSuccess)->getOperand(0),
            AtomicCompareExchange);
  EXPECT_EQ(cast<ConstantInt>(cast<Instruction>(AtomicSuccess)->getOperand(1))
                ->getZExtValue(),
            0);
}

TEST_F(BuildBuiltinsTests, AtomicCmpxchg_Ptr) {
  Value *AtomicSuccess = nullptr;
  ASSERT_THAT_EXPECTED(
      emitAtomicCompareExchangeBuiltin(
          /*AtomicPtr=*/PtrArg,
          /*ExpectedPtr=*/ExpectedArg,
          /*DesiredPtr=*/DesiredArg, /*TypeOrSize=*/Builder.getPtrTy(),
          /*IsWeak*/ false,
          /*IsVolatile=*/false,
          /*SuccessMemorder=*/AtomicOrdering::SequentiallyConsistent,
          /*FailureMemorder=*/AtomicOrdering::SequentiallyConsistent,
          /*Scope=*/SyncScope::System,
          /*PrevPtr=*/nullptr,
          /*Align=*/{}, /*Builder=*/Builder,
          /*EmitOptions=*/AtomicEmitOptions(DL, TLI.get()),
          /*Name=*/"atomic_cmpxchg"),
      StoreResult(AtomicSuccess));
  EXPECT_FALSE(verifyModule(*M, &errs()));

  // clang-format off
  // %atomic_cmpxchg.cmpxchg.expected = load ptr, ptr %expected_ptr, align 8
  // %atomic_cmpxchg.cmpxchg.desired = load ptr, ptr %desired_ptr, align 8
  // %atomic_cmpxchg.cmpxchg.pair = cmpxchg ptr %atomic_ptr, ptr %atomic_cmpxchg.cmpxchg.expected, ptr %atomic_cmpxchg.cmpxchg.desired seq_cst seq_cst, align 1
  // %atomic_cmpxchg.cmpxchg.success = extractvalue { ptr, i1 } %atomic_cmpxchg.cmpxchg.pair, 1
  // clang-format on

  // Follow use-def and load-store chains to discover instructions
  AtomicCmpXchgInst *AtomicCmpxchgCmpxchgPair =
      cast<AtomicCmpXchgInst>(getUniquePreviousStore(PtrArg, EntryBB));
  LoadInst *AtomicCmpxchgCmpxchgExpected =
      cast<LoadInst>(AtomicCmpxchgCmpxchgPair->getCompareOperand());
  LoadInst *AtomicCmpxchgCmpxchgDesired =
      cast<LoadInst>(AtomicCmpxchgCmpxchgPair->getNewValOperand());

  // %atomic_cmpxchg.cmpxchg.expected = load ptr, ptr %expected_ptr, align 8
  EXPECT_EQ(AtomicCmpxchgCmpxchgExpected->getName(),
            "atomic_cmpxchg.cmpxchg.expected");
  EXPECT_EQ(AtomicCmpxchgCmpxchgExpected->getParent(), EntryBB);
  EXPECT_EQ(AtomicCmpxchgCmpxchgExpected->getType(), PointerType::get(Ctx, 0));
  EXPECT_TRUE(AtomicCmpxchgCmpxchgExpected->isSimple());
  EXPECT_EQ(AtomicCmpxchgCmpxchgExpected->getPointerOperand(), ExpectedArg);

  // %atomic_cmpxchg.cmpxchg.desired = load ptr, ptr %desired_ptr, align 8
  EXPECT_EQ(AtomicCmpxchgCmpxchgDesired->getName(),
            "atomic_cmpxchg.cmpxchg.desired");
  EXPECT_EQ(AtomicCmpxchgCmpxchgDesired->getParent(), EntryBB);
  EXPECT_EQ(AtomicCmpxchgCmpxchgDesired->getType(), PointerType::get(Ctx, 0));
  EXPECT_TRUE(AtomicCmpxchgCmpxchgDesired->isSimple());
  EXPECT_EQ(AtomicCmpxchgCmpxchgDesired->getPointerOperand(), DesiredArg);

  // %atomic_cmpxchg.cmpxchg.pair = cmpxchg ptr %atomic_ptr, ptr %atomic_cm...
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getName(), "atomic_cmpxchg.cmpxchg.pair");
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getParent(), EntryBB);
  EXPECT_FALSE(AtomicCmpxchgCmpxchgPair->isVolatile());
  EXPECT_FALSE(AtomicCmpxchgCmpxchgPair->isWeak());
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getSuccessOrdering(),
            AtomicOrdering::SequentiallyConsistent);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getFailureOrdering(),
            AtomicOrdering::SequentiallyConsistent);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getSyncScopeID(), SyncScope::System);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getAlign(), 1);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getPointerOperand(), PtrArg);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getCompareOperand(),
            AtomicCmpxchgCmpxchgExpected);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getNewValOperand(),
            AtomicCmpxchgCmpxchgDesired);

  // %atomic_cmpxchg.cmpxchg.success = extractvalue { ptr, i1 } %atomic_cmp...
  EXPECT_EQ(AtomicSuccess->getName(), "atomic_cmpxchg.cmpxchg.success");
  EXPECT_EQ(cast<Instruction>(AtomicSuccess)->getParent(), EntryBB);
  EXPECT_EQ(AtomicSuccess->getType(), Type::getInt1Ty(Ctx));
  EXPECT_EQ(cast<ExtractValueInst>(AtomicSuccess)->getNumIndices(), 1);
  EXPECT_EQ(cast<ExtractValueInst>(AtomicSuccess)->getIndices()[0], 1);
  EXPECT_EQ(cast<ExtractValueInst>(AtomicSuccess)->getAggregateOperand(),
            AtomicCmpxchgCmpxchgPair);
}

TEST_F(BuildBuiltinsTests, AtomicCmpxchg_Struct) {
  // A struct that is small enough to be covered with a single instruction
  StructType *STy =
      StructType::get(Ctx, {Builder.getFloatTy(), Builder.getFloatTy()});

  Value *AtomicSuccess = nullptr;
  ASSERT_THAT_EXPECTED(
      emitAtomicCompareExchangeBuiltin(
          /*AtomicPtr=*/PtrArg,
          /*ExpectedPtr=*/ExpectedArg,
          /*DesiredPtr=*/DesiredArg, /*TypeOrSize=*/STy,
          /*IsWeak*/ false,
          /*IsVolatile=*/false,
          /*SuccessMemorder=*/AtomicOrdering::SequentiallyConsistent,
          /*FailureMemorder=*/AtomicOrdering::SequentiallyConsistent,
          /*Scope=*/SyncScope::System,
          /*PrevPtr=*/nullptr,
          /*Align=*/{}, /*Builder=*/Builder,
          /*EmitOptions=*/AtomicEmitOptions(DL, TLI.get()),
          /*Name=*/"atomic_cmpxchg"),
      StoreResult(AtomicSuccess));
  EXPECT_FALSE(verifyModule(*M, &errs()));

  // clang-format off
  // %atomic_cmpxchg.cmpxchg.expected = load i64, ptr %expected_ptr, align 4
  // %atomic_cmpxchg.cmpxchg.desired = load i64, ptr %desired_ptr, align 4
  // %atomic_cmpxchg.cmpxchg.pair = cmpxchg ptr %atomic_ptr, i64 %atomic_cmpxchg.cmpxchg.expected, i64 %atomic_cmpxchg.cmpxchg.desired seq_cst seq_cst, align 1
  // %atomic_cmpxchg.cmpxchg.success = extractvalue { i64, i1 } %atomic_cmpxchg.cmpxchg.pair, 1
  // clang-format on

  // Follow use-def and load-store chains to discover instructions
  AtomicCmpXchgInst *AtomicCmpxchgCmpxchgPair =
      cast<AtomicCmpXchgInst>(getUniquePreviousStore(PtrArg, EntryBB));
  LoadInst *AtomicCmpxchgCmpxchgExpected =
      cast<LoadInst>(AtomicCmpxchgCmpxchgPair->getCompareOperand());
  LoadInst *AtomicCmpxchgCmpxchgDesired =
      cast<LoadInst>(AtomicCmpxchgCmpxchgPair->getNewValOperand());

  // %atomic_cmpxchg.cmpxchg.expected = load i64, ptr %expected_ptr, align 4
  EXPECT_EQ(AtomicCmpxchgCmpxchgExpected->getName(),
            "atomic_cmpxchg.cmpxchg.expected");
  EXPECT_EQ(AtomicCmpxchgCmpxchgExpected->getParent(), EntryBB);
  EXPECT_EQ(AtomicCmpxchgCmpxchgExpected->getType(), Type::getInt64Ty(Ctx));
  EXPECT_TRUE(AtomicCmpxchgCmpxchgExpected->isSimple());
  EXPECT_EQ(AtomicCmpxchgCmpxchgExpected->getPointerOperand(), ExpectedArg);

  // %atomic_cmpxchg.cmpxchg.desired = load i64, ptr %desired_ptr, align 4
  EXPECT_EQ(AtomicCmpxchgCmpxchgDesired->getName(),
            "atomic_cmpxchg.cmpxchg.desired");
  EXPECT_EQ(AtomicCmpxchgCmpxchgDesired->getParent(), EntryBB);
  EXPECT_EQ(AtomicCmpxchgCmpxchgDesired->getType(), Type::getInt64Ty(Ctx));
  EXPECT_TRUE(AtomicCmpxchgCmpxchgDesired->isSimple());
  EXPECT_EQ(AtomicCmpxchgCmpxchgDesired->getPointerOperand(), DesiredArg);

  // %atomic_cmpxchg.cmpxchg.pair = cmpxchg ptr %atomic_ptr, i64 %atomic_cm...
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getName(), "atomic_cmpxchg.cmpxchg.pair");
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getParent(), EntryBB);
  EXPECT_FALSE(AtomicCmpxchgCmpxchgPair->isVolatile());
  EXPECT_FALSE(AtomicCmpxchgCmpxchgPair->isWeak());
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getSuccessOrdering(),
            AtomicOrdering::SequentiallyConsistent);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getFailureOrdering(),
            AtomicOrdering::SequentiallyConsistent);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getSyncScopeID(), SyncScope::System);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getAlign(), 1);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getPointerOperand(), PtrArg);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getCompareOperand(),
            AtomicCmpxchgCmpxchgExpected);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getNewValOperand(),
            AtomicCmpxchgCmpxchgDesired);

  // %atomic_cmpxchg.cmpxchg.success = extractvalue { i64, i1 } %atomic_cmp...
  EXPECT_EQ(AtomicSuccess->getName(), "atomic_cmpxchg.cmpxchg.success");
  EXPECT_EQ(cast<Instruction>(AtomicSuccess)->getParent(), EntryBB);
  EXPECT_EQ(AtomicSuccess->getType(), Type::getInt1Ty(Ctx));
  EXPECT_EQ(cast<ExtractValueInst>(AtomicSuccess)->getNumIndices(), 1);
  EXPECT_EQ(cast<ExtractValueInst>(AtomicSuccess)->getIndices()[0], 1);
  EXPECT_EQ(cast<ExtractValueInst>(AtomicSuccess)->getAggregateOperand(),
            AtomicCmpxchgCmpxchgPair);
}

TEST_F(BuildBuiltinsTests, AtomicCmpxchg_Array) {
  // A type that is too large for atomic instructions
  ArrayType *ATy = ArrayType::get(Builder.getFloatTy(), 19);

  Value *AtomicSuccess = nullptr;
  ASSERT_THAT_EXPECTED(
      emitAtomicCompareExchangeBuiltin(
          /*AtomicPtr=*/PtrArg,
          /*ExpectedPtr=*/ExpectedArg,
          /*DesiredPtr=*/DesiredArg, /*TypeOrSize=*/ATy,
          /*IsWeak*/ false,
          /*IsVolatile=*/false,
          /*SuccessMemorder=*/AtomicOrdering::SequentiallyConsistent,
          /*FailureMemorder=*/AtomicOrdering::SequentiallyConsistent,
          /*Scope=*/SyncScope::System,
          /*PrevPtr=*/nullptr,
          /*Align=*/{}, /*Builder=*/Builder,
          /*EmitOptions=*/AtomicEmitOptions(DL, TLI.get()),
          /*Name=*/"atomic_cmpxchg"),
      StoreResult(AtomicSuccess));
  EXPECT_FALSE(verifyModule(*M, &errs()));

  // clang-format off
  // %__atomic_compare_exchange = call i8 @__atomic_compare_exchange(i64 76, ptr %atomic_ptr, ptr %expected_ptr, ptr %desired_ptr, i32 5, i32 5)
  // %atomic_cmpxchg.cmpxchg.success = icmp eq i8 %__atomic_compare_exchange, 0
  // clang-format on

  // Follow use-def and load-store chains to discover instructions
  CallInst *AtomicCompareExchange =
      cast<CallInst>(getUniquePreviousStore(PtrArg, EntryBB));

  // %__atomic_compare_exchange = call i8 @__atomic_compare_exchange(i64 76...
  EXPECT_EQ(AtomicCompareExchange->getName(), "__atomic_compare_exchange");
  EXPECT_EQ(AtomicCompareExchange->getParent(), EntryBB);
  EXPECT_EQ(AtomicCompareExchange->getType(), Type::getInt8Ty(Ctx));
  EXPECT_EQ(AtomicCompareExchange->getName(), "__atomic_compare_exchange");
  EXPECT_FALSE(AtomicCompareExchange->isMustTailCall());
  EXPECT_FALSE(AtomicCompareExchange->isTailCall());
  EXPECT_EQ(AtomicCompareExchange->getCallingConv(), CallingConv::C);
  EXPECT_EQ(cast<ConstantInt>(AtomicCompareExchange->getArgOperand(0))
                ->getZExtValue(),
            76);
  EXPECT_EQ(AtomicCompareExchange->getArgOperand(1), PtrArg);
  EXPECT_EQ(AtomicCompareExchange->getArgOperand(2), ExpectedArg);
  EXPECT_EQ(AtomicCompareExchange->getArgOperand(3), DesiredArg);
  EXPECT_EQ(cast<ConstantInt>(AtomicCompareExchange->getArgOperand(4))
                ->getZExtValue(),
            5);
  EXPECT_EQ(cast<ConstantInt>(AtomicCompareExchange->getArgOperand(5))
                ->getZExtValue(),
            5);
  EXPECT_EQ(AtomicCompareExchange->getCalledFunction(),
            M->getFunction("__atomic_compare_exchange"));

  // %atomic_cmpxchg.cmpxchg.success = icmp eq i8 %__atomic_compare_exchang...
  EXPECT_EQ(AtomicSuccess->getName(), "atomic_cmpxchg.cmpxchg.success");
  EXPECT_EQ(cast<Instruction>(AtomicSuccess)->getParent(), EntryBB);
  EXPECT_EQ(AtomicSuccess->getType(), Type::getInt1Ty(Ctx));
  EXPECT_EQ(cast<Instruction>(AtomicSuccess)->getOperand(0),
            AtomicCompareExchange);
  EXPECT_EQ(cast<ConstantInt>(cast<Instruction>(AtomicSuccess)->getOperand(1))
                ->getZExtValue(),
            0);
}

TEST_F(BuildBuiltinsTests, AtomicCmpxchg_Array_NoLibatomic) {
  // Use a triple that does not support libatomic (according to
  // initializeLibCalls in TargetLibraryInfo.cpp)
  Triple T("x86_64-scei-ps4");
  TLII.reset(new TargetLibraryInfoImpl(T));
  TLI.reset(new TargetLibraryInfo(*TLII));

  // A type that is too large for atomic instructions
  ArrayType *ATy = ArrayType::get(Builder.getFloatTy(), 19);

  ASSERT_THAT_EXPECTED(
      emitAtomicCompareExchangeBuiltin(
          /*AtomicPtr=*/PtrArg,
          /*ExpectedPtr=*/ExpectedArg,
          /*DesiredPtr=*/DesiredArg, /*TypeOrSize=*/ATy,
          /*IsWeak*/ false,
          /*IsVolatile=*/false,
          /*SuccessMemorder=*/AtomicOrdering::SequentiallyConsistent,
          /*FailureMemorder=*/AtomicOrdering::SequentiallyConsistent,
          /*Scope=*/SyncScope::SingleThread,
          /*PrevPtr=*/nullptr,
          /*Align=*/{}, /*Builder=*/Builder,
          /*EmitOptions=*/AtomicEmitOptions(DL, TLI.get()),
          /*Name=*/"atomic_cmpxchg"),
      FailedWithMessage("__atomic_compare_exchange builtin not supported by "
                        "any available means"));
}

TEST_F(BuildBuiltinsTests, AtomicCmpxchg_DataSize) {
  Value *AtomicSuccess = nullptr;
  ASSERT_THAT_EXPECTED(
      emitAtomicCompareExchangeBuiltin(
          /*AtomicPtr=*/PtrArg,
          /*ExpectedPtr=*/ExpectedArg,
          /*DesiredPtr=*/DesiredArg, /*TypeOrSize=*/static_cast<uint64_t>(6),
          /*IsWeak*/ false,
          /*IsVolatile=*/false,
          /*SuccessMemorder=*/AtomicOrdering::SequentiallyConsistent,
          /*FailureMemorder=*/AtomicOrdering::SequentiallyConsistent,
          /*Scope=*/SyncScope::System,
          /*PrevPtr=*/nullptr,
          /*Align=*/{}, /*Builder=*/Builder,
          /*EmitOptions=*/AtomicEmitOptions(DL, TLI.get()),
          /*Name=*/"atomic_cmpxchg"),
      StoreResult(AtomicSuccess));
  EXPECT_FALSE(verifyModule(*M, &errs()));

  // clang-format off
  // %__atomic_compare_exchange = call i8 @__atomic_compare_exchange(i64 6, ptr %atomic_ptr, ptr %expected_ptr, ptr %desired_ptr, i32 5, i32 5)
  // %atomic_cmpxchg.cmpxchg.success = icmp eq i8 %__atomic_compare_exchange, 0
  // clang-format on

  // Follow use-def and load-store chains to discover instructions
  CallInst *AtomicCompareExchange =
      cast<CallInst>(getUniquePreviousStore(PtrArg, EntryBB));

  // %__atomic_compare_exchange = call i8 @__atomic_compare_exchange(i64 6,...
  EXPECT_EQ(AtomicCompareExchange->getName(), "__atomic_compare_exchange");
  EXPECT_EQ(AtomicCompareExchange->getParent(), EntryBB);
  EXPECT_EQ(AtomicCompareExchange->getType(), Type::getInt8Ty(Ctx));
  EXPECT_EQ(AtomicCompareExchange->getName(), "__atomic_compare_exchange");
  EXPECT_FALSE(AtomicCompareExchange->isMustTailCall());
  EXPECT_FALSE(AtomicCompareExchange->isTailCall());
  EXPECT_EQ(AtomicCompareExchange->getCallingConv(), CallingConv::C);
  EXPECT_EQ(cast<ConstantInt>(AtomicCompareExchange->getArgOperand(0))
                ->getZExtValue(),
            6);
  EXPECT_EQ(AtomicCompareExchange->getArgOperand(1), PtrArg);
  EXPECT_EQ(AtomicCompareExchange->getArgOperand(2), ExpectedArg);
  EXPECT_EQ(AtomicCompareExchange->getArgOperand(3), DesiredArg);
  EXPECT_EQ(cast<ConstantInt>(AtomicCompareExchange->getArgOperand(4))
                ->getZExtValue(),
            5);
  EXPECT_EQ(cast<ConstantInt>(AtomicCompareExchange->getArgOperand(5))
                ->getZExtValue(),
            5);
  EXPECT_EQ(AtomicCompareExchange->getCalledFunction(),
            M->getFunction("__atomic_compare_exchange"));

  // %atomic_cmpxchg.cmpxchg.success = icmp eq i8 %__atomic_compare_exchang...
  EXPECT_EQ(AtomicSuccess->getName(), "atomic_cmpxchg.cmpxchg.success");
  EXPECT_EQ(cast<Instruction>(AtomicSuccess)->getParent(), EntryBB);
  EXPECT_EQ(AtomicSuccess->getType(), Type::getInt1Ty(Ctx));
  EXPECT_EQ(cast<Instruction>(AtomicSuccess)->getOperand(0),
            AtomicCompareExchange);
  EXPECT_EQ(cast<ConstantInt>(cast<Instruction>(AtomicSuccess)->getOperand(1))
                ->getZExtValue(),
            0);
}

TEST_F(BuildBuiltinsTests, AtomicCmpxchg_Align) {
  Value *AtomicSuccess = nullptr;
  ASSERT_THAT_EXPECTED(
      emitAtomicCompareExchangeBuiltin(
          /*AtomicPtr=*/PtrArg,
          /*ExpectedPtr=*/ExpectedArg,
          /*DesiredPtr=*/DesiredArg, /*TypeOrSize=*/Builder.getFloatTy(),
          /*IsWeak*/ false,
          /*IsVolatile=*/false,
          /*SuccessMemorder=*/AtomicOrdering::SequentiallyConsistent,
          /*FailureMemorder=*/AtomicOrdering::SequentiallyConsistent,
          /*Scope=*/SyncScope::System,
          /*PrevPtr=*/nullptr,
          /*Align=*/Align(8), /*Builder=*/Builder,
          /*EmitOptions=*/AtomicEmitOptions(DL, TLI.get()),
          /*Name=*/"atomic_cmpxchg"),
      StoreResult(AtomicSuccess));
  EXPECT_FALSE(verifyModule(*M, &errs()));

  // clang-format off
  // %atomic_cmpxchg.cmpxchg.expected = load i32, ptr %expected_ptr, align 4
  // %atomic_cmpxchg.cmpxchg.desired = load i32, ptr %desired_ptr, align 4
  // %atomic_cmpxchg.cmpxchg.pair = cmpxchg ptr %atomic_ptr, i32 %atomic_cmpxchg.cmpxchg.expected, i32 %atomic_cmpxchg.cmpxchg.desired seq_cst seq_cst, align 8
  // %atomic_cmpxchg.cmpxchg.success = extractvalue { i32, i1 } %atomic_cmpxchg.cmpxchg.pair, 1
  // clang-format on

  // Follow use-def and load-store chains to discover instructions
  AtomicCmpXchgInst *AtomicCmpxchgCmpxchgPair =
      cast<AtomicCmpXchgInst>(getUniquePreviousStore(PtrArg, EntryBB));
  LoadInst *AtomicCmpxchgCmpxchgExpected =
      cast<LoadInst>(AtomicCmpxchgCmpxchgPair->getCompareOperand());
  LoadInst *AtomicCmpxchgCmpxchgDesired =
      cast<LoadInst>(AtomicCmpxchgCmpxchgPair->getNewValOperand());

  // %atomic_cmpxchg.cmpxchg.expected = load i32, ptr %expected_ptr, align 4
  EXPECT_EQ(AtomicCmpxchgCmpxchgExpected->getName(),
            "atomic_cmpxchg.cmpxchg.expected");
  EXPECT_EQ(AtomicCmpxchgCmpxchgExpected->getParent(), EntryBB);
  EXPECT_EQ(AtomicCmpxchgCmpxchgExpected->getType(), Type::getInt32Ty(Ctx));
  EXPECT_TRUE(AtomicCmpxchgCmpxchgExpected->isSimple());
  EXPECT_EQ(AtomicCmpxchgCmpxchgExpected->getPointerOperand(), ExpectedArg);

  // %atomic_cmpxchg.cmpxchg.desired = load i32, ptr %desired_ptr, align 4
  EXPECT_EQ(AtomicCmpxchgCmpxchgDesired->getName(),
            "atomic_cmpxchg.cmpxchg.desired");
  EXPECT_EQ(AtomicCmpxchgCmpxchgDesired->getParent(), EntryBB);
  EXPECT_EQ(AtomicCmpxchgCmpxchgDesired->getType(), Type::getInt32Ty(Ctx));
  EXPECT_TRUE(AtomicCmpxchgCmpxchgDesired->isSimple());
  EXPECT_EQ(AtomicCmpxchgCmpxchgDesired->getPointerOperand(), DesiredArg);

  // %atomic_cmpxchg.cmpxchg.pair = cmpxchg ptr %atomic_ptr, i32 %atomic_cm...
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getName(), "atomic_cmpxchg.cmpxchg.pair");
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getParent(), EntryBB);
  EXPECT_FALSE(AtomicCmpxchgCmpxchgPair->isVolatile());
  EXPECT_FALSE(AtomicCmpxchgCmpxchgPair->isWeak());
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getSuccessOrdering(),
            AtomicOrdering::SequentiallyConsistent);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getFailureOrdering(),
            AtomicOrdering::SequentiallyConsistent);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getSyncScopeID(), SyncScope::System);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getAlign(), 8);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getPointerOperand(), PtrArg);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getCompareOperand(),
            AtomicCmpxchgCmpxchgExpected);
  EXPECT_EQ(AtomicCmpxchgCmpxchgPair->getNewValOperand(),
            AtomicCmpxchgCmpxchgDesired);

  // %atomic_cmpxchg.cmpxchg.success = extractvalue { i32, i1 } %atomic_cmp...
  EXPECT_EQ(AtomicSuccess->getName(), "atomic_cmpxchg.cmpxchg.success");
  EXPECT_EQ(cast<Instruction>(AtomicSuccess)->getParent(), EntryBB);
  EXPECT_EQ(AtomicSuccess->getType(), Type::getInt1Ty(Ctx));
  EXPECT_EQ(cast<ExtractValueInst>(AtomicSuccess)->getNumIndices(), 1);
  EXPECT_EQ(cast<ExtractValueInst>(AtomicSuccess)->getIndices()[0], 1);
  EXPECT_EQ(cast<ExtractValueInst>(AtomicSuccess)->getAggregateOperand(),
            AtomicCmpxchgCmpxchgPair);
}

} // namespace
