//===- TrackerTest.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Module.h"
#include "llvm/SandboxIR/Function.h"
#include "llvm/SandboxIR/Instruction.h"
#include "llvm/Support/SourceMgr.h"
#include "gmock/gmock-matchers.h"
#include "gtest/gtest.h"

using namespace llvm;

struct TrackerTest : public testing::Test {
  LLVMContext C;
  std::unique_ptr<Module> M;

  void parseIR(LLVMContext &C, const char *IR) {
    SMDiagnostic Err;
    M = parseAssemblyString(IR, Err, C);
    if (!M)
      Err.print("TrackerTest", errs());
  }
  BasicBlock *getBasicBlockByName(Function &F, StringRef Name) {
    for (BasicBlock &BB : F)
      if (BB.getName() == Name)
        return &BB;
    llvm_unreachable("Expected to find basic block!");
  }
};

TEST_F(TrackerTest, SetOperand) {
  parseIR(C, R"IR(
define void @foo(ptr %ptr) {
  %gep0 = getelementptr float, ptr %ptr, i32 0
  %gep1 = getelementptr float, ptr %ptr, i32 1
  %ld0 = load float, ptr %gep0
  store float undef, ptr %gep0
  ret void
}
)IR");
  Function &LLVMF = *M->getFunction("foo");
  sandboxir::Context Ctx(C);
  auto *F = Ctx.createFunction(&LLVMF);
  auto *BB = &*F->begin();
  auto &Tracker = Ctx.getTracker();
  Tracker.save();
  auto It = BB->begin();
  auto *Gep0 = &*It++;
  auto *Gep1 = &*It++;
  auto *Ld = &*It++;
  auto *St = &*It++;
  St->setOperand(0, Ld);
  St->setOperand(1, Gep1);
  Ld->setOperand(0, Gep1);
  EXPECT_EQ(St->getOperand(0), Ld);
  EXPECT_EQ(St->getOperand(1), Gep1);
  EXPECT_EQ(Ld->getOperand(0), Gep1);

  Ctx.getTracker().revert();
  EXPECT_NE(St->getOperand(0), Ld);
  EXPECT_EQ(St->getOperand(1), Gep0);
  EXPECT_EQ(Ld->getOperand(0), Gep0);
}

TEST_F(TrackerTest, SetUse) {
  parseIR(C, R"IR(
define void @foo(ptr %ptr, i8 %arg) {
  %ld = load i8, ptr %ptr
  %add = add i8 %ld, %arg
  ret void
}
)IR");
  Function &LLVMF = *M->getFunction("foo");
  sandboxir::Context Ctx(C);
  auto *F = Ctx.createFunction(&LLVMF);
  unsigned ArgIdx = 0;
  auto *Arg0 = F->getArg(ArgIdx++);
  auto *BB = &*F->begin();
  auto &Tracker = Ctx.getTracker();
  Tracker.save();
  auto It = BB->begin();
  auto *Ld = &*It++;
  auto *Add = &*It++;

  Ctx.save();
  sandboxir::Use Use = Add->getOperandUse(0);
  Use.set(Arg0);
  EXPECT_EQ(Add->getOperand(0), Arg0);
  Ctx.revert();
  EXPECT_EQ(Add->getOperand(0), Ld);
}

TEST_F(TrackerTest, SwapOperands) {
  parseIR(C, R"IR(
define void @foo(i1 %cond) {
 bb0:
   br i1 %cond, label %bb1, label %bb2
 bb1:
   ret void
 bb2:
   ret void
}
)IR");
  Function &LLVMF = *M->getFunction("foo");
  sandboxir::Context Ctx(C);
  Ctx.createFunction(&LLVMF);
  auto *BB0 = cast<sandboxir::BasicBlock>(
      Ctx.getValue(getBasicBlockByName(LLVMF, "bb0")));
  auto *BB1 = cast<sandboxir::BasicBlock>(
      Ctx.getValue(getBasicBlockByName(LLVMF, "bb1")));
  auto *BB2 = cast<sandboxir::BasicBlock>(
      Ctx.getValue(getBasicBlockByName(LLVMF, "bb2")));
  auto &Tracker = Ctx.getTracker();
  Tracker.save();
  auto It = BB0->begin();
  auto *Br = cast<sandboxir::BranchInst>(&*It++);

  unsigned SuccIdx = 0;
  SmallVector<sandboxir::BasicBlock *> ExpectedSuccs({BB2, BB1});
  for (auto *Succ : Br->successors())
    EXPECT_EQ(Succ, ExpectedSuccs[SuccIdx++]);

  // This calls User::swapOperandsInternal() internally.
  Br->swapSuccessors();

  SuccIdx = 0;
  for (auto *Succ : reverse(Br->successors()))
    EXPECT_EQ(Succ, ExpectedSuccs[SuccIdx++]);

  Ctx.getTracker().revert();
  SuccIdx = 0;
  for (auto *Succ : Br->successors())
    EXPECT_EQ(Succ, ExpectedSuccs[SuccIdx++]);
}

TEST_F(TrackerTest, RUWIf_RAUW_RUOW) {
  parseIR(C, R"IR(
define void @foo(ptr %ptr) {
  %ld0 = load float, ptr %ptr
  %ld1 = load float, ptr %ptr
  store float %ld0, ptr %ptr
  store float %ld0, ptr %ptr
  ret void
}
)IR");
  llvm::Function &LLVMF = *M->getFunction("foo");
  sandboxir::Context Ctx(C);
  llvm::BasicBlock *LLVMBB = &*LLVMF.begin();
  Ctx.createFunction(&LLVMF);
  auto *BB = cast<sandboxir::BasicBlock>(Ctx.getValue(LLVMBB));
  auto It = BB->begin();
  sandboxir::Instruction *Ld0 = &*It++;
  sandboxir::Instruction *Ld1 = &*It++;
  sandboxir::Instruction *St0 = &*It++;
  sandboxir::Instruction *St1 = &*It++;
  Ctx.save();
  // Check RUWIf when the lambda returns false.
  Ld0->replaceUsesWithIf(Ld1, [](const sandboxir::Use &Use) { return false; });
  EXPECT_EQ(St0->getOperand(0), Ld0);
  EXPECT_EQ(St1->getOperand(0), Ld0);

  // Check RUWIf when the lambda returns true.
  Ld0->replaceUsesWithIf(Ld1, [](const sandboxir::Use &Use) { return true; });
  EXPECT_EQ(St0->getOperand(0), Ld1);
  EXPECT_EQ(St1->getOperand(0), Ld1);
  Ctx.revert();
  EXPECT_EQ(St0->getOperand(0), Ld0);
  EXPECT_EQ(St1->getOperand(0), Ld0);

  // Check RUWIf user == St0.
  Ctx.save();
  Ld0->replaceUsesWithIf(
      Ld1, [St0](const sandboxir::Use &Use) { return Use.getUser() == St0; });
  EXPECT_EQ(St0->getOperand(0), Ld1);
  EXPECT_EQ(St1->getOperand(0), Ld0);
  Ctx.revert();
  EXPECT_EQ(St0->getOperand(0), Ld0);
  EXPECT_EQ(St1->getOperand(0), Ld0);

  // Check RUWIf user == St1.
  Ctx.save();
  Ld0->replaceUsesWithIf(
      Ld1, [St1](const sandboxir::Use &Use) { return Use.getUser() == St1; });
  EXPECT_EQ(St0->getOperand(0), Ld0);
  EXPECT_EQ(St1->getOperand(0), Ld1);
  Ctx.revert();
  EXPECT_EQ(St0->getOperand(0), Ld0);
  EXPECT_EQ(St1->getOperand(0), Ld0);

  // Check RAUW.
  Ctx.save();
  Ld1->replaceAllUsesWith(Ld0);
  EXPECT_EQ(St0->getOperand(0), Ld0);
  EXPECT_EQ(St1->getOperand(0), Ld0);
  Ctx.revert();
  EXPECT_EQ(St0->getOperand(0), Ld0);
  EXPECT_EQ(St1->getOperand(0), Ld0);

  // Check RUOW.
  Ctx.save();
  St0->replaceUsesOfWith(Ld0, Ld1);
  EXPECT_EQ(St0->getOperand(0), Ld1);
  Ctx.revert();
  EXPECT_EQ(St0->getOperand(0), Ld0);

  // Check accept().
  Ctx.save();
  St0->replaceUsesOfWith(Ld0, Ld1);
  EXPECT_EQ(St0->getOperand(0), Ld1);
  Ctx.accept();
  EXPECT_EQ(St0->getOperand(0), Ld1);
}

// TODO: Test multi-instruction patterns.
TEST_F(TrackerTest, EraseFromParent) {
  parseIR(C, R"IR(
define void @foo(i32 %v1) {
  %add0 = add i32 %v1, %v1
  %add1 = add i32 %add0, %v1
  ret void
}
)IR");
  Function &LLVMF = *M->getFunction("foo");
  sandboxir::Context Ctx(C);

  auto *F = Ctx.createFunction(&LLVMF);
  auto *BB = &*F->begin();
  auto It = BB->begin();
  sandboxir::Instruction *Add0 = &*It++;
  sandboxir::Instruction *Add1 = &*It++;
  sandboxir::Instruction *Ret = &*It++;

  Ctx.save();
  // Check erase.
  Add1->eraseFromParent();
  It = BB->begin();
  EXPECT_EQ(&*It++, Add0);
  EXPECT_EQ(&*It++, Ret);
  EXPECT_EQ(It, BB->end());
  EXPECT_EQ(Add0->getNumUses(), 0u);

  // Check revert().
  Ctx.revert();
  It = BB->begin();
  EXPECT_EQ(&*It++, Add0);
  EXPECT_EQ(&*It++, Add1);
  EXPECT_EQ(&*It++, Ret);
  EXPECT_EQ(It, BB->end());
  EXPECT_EQ(Add1->getOperand(0), Add0);

  // Same for the last instruction in the block.
  Ctx.save();
  Ret->eraseFromParent();
  It = BB->begin();
  EXPECT_EQ(&*It++, Add0);
  EXPECT_EQ(&*It++, Add1);
  EXPECT_EQ(It, BB->end());
  Ctx.revert();
  It = BB->begin();
  EXPECT_EQ(&*It++, Add0);
  EXPECT_EQ(&*It++, Add1);
  EXPECT_EQ(&*It++, Ret);
  EXPECT_EQ(It, BB->end());
}

// TODO: Test multi-instruction patterns.
TEST_F(TrackerTest, RemoveFromParent) {
  parseIR(C, R"IR(
define i32 @foo(i32 %arg) {
  %add0 = add i32 %arg, %arg
  %add1 = add i32 %add0, %arg
  ret i32 %add1
}
)IR");
  Function &LLVMF = *M->getFunction("foo");
  sandboxir::Context Ctx(C);

  auto *F = Ctx.createFunction(&LLVMF);
  auto *Arg = F->getArg(0);
  auto *BB = &*F->begin();
  auto It = BB->begin();
  sandboxir::Instruction *Add0 = &*It++;
  sandboxir::Instruction *Add1 = &*It++;
  sandboxir::Instruction *Ret = &*It++;

  Ctx.save();
  // Check removeFromParent().
  Add1->removeFromParent();
  It = BB->begin();
  EXPECT_EQ(&*It++, Add0);
  EXPECT_EQ(&*It++, Ret);
  EXPECT_EQ(It, BB->end());
  // Removed instruction still be connected to operands and users.
  EXPECT_EQ(Add1->getOperand(0), Add0);
  EXPECT_EQ(Add1->getOperand(1), Arg);
  EXPECT_EQ(Add0->getNumUses(), 1u);

  // Check revert().
  Ctx.revert();
  It = BB->begin();
  EXPECT_EQ(&*It++, Add0);
  EXPECT_EQ(&*It++, Add1);
  EXPECT_EQ(&*It++, Ret);
  EXPECT_EQ(It, BB->end());
  EXPECT_EQ(Add1->getOperand(0), Add0);

  // Same for the last instruction in the block.
  Ctx.save();
  Ret->removeFromParent();
  It = BB->begin();
  EXPECT_EQ(&*It++, Add0);
  EXPECT_EQ(&*It++, Add1);
  EXPECT_EQ(It, BB->end());
  EXPECT_EQ(Ret->getOperand(0), Add1);
  Ctx.revert();
  It = BB->begin();
  EXPECT_EQ(&*It++, Add0);
  EXPECT_EQ(&*It++, Add1);
  EXPECT_EQ(&*It++, Ret);
  EXPECT_EQ(It, BB->end());
}

// TODO: Test multi-instruction patterns.
TEST_F(TrackerTest, MoveInstr) {
  parseIR(C, R"IR(
define i32 @foo(i32 %arg) {
  %add0 = add i32 %arg, %arg
  %add1 = add i32 %add0, %arg
  ret i32 %add1
}
)IR");
  Function &LLVMF = *M->getFunction("foo");
  sandboxir::Context Ctx(C);

  auto *F = Ctx.createFunction(&LLVMF);
  auto *BB = &*F->begin();
  auto It = BB->begin();
  sandboxir::Instruction *Add0 = &*It++;
  sandboxir::Instruction *Add1 = &*It++;
  sandboxir::Instruction *Ret = &*It++;

  // Check moveBefore(Instruction *) with tracking enabled.
  Ctx.save();
  Add1->moveBefore(Add0);
  It = BB->begin();
  EXPECT_EQ(&*It++, Add1);
  EXPECT_EQ(&*It++, Add0);
  EXPECT_EQ(&*It++, Ret);
  EXPECT_EQ(It, BB->end());
  // Check revert().
  Ctx.revert();
  It = BB->begin();
  EXPECT_EQ(&*It++, Add0);
  EXPECT_EQ(&*It++, Add1);
  EXPECT_EQ(&*It++, Ret);
  EXPECT_EQ(It, BB->end());

  // Same for the last instruction in the block.
  Ctx.save();
  Ret->moveBefore(Add0);
  It = BB->begin();
  EXPECT_EQ(&*It++, Ret);
  EXPECT_EQ(&*It++, Add0);
  EXPECT_EQ(&*It++, Add1);
  EXPECT_EQ(It, BB->end());
  Ctx.revert();
  It = BB->begin();
  EXPECT_EQ(&*It++, Add0);
  EXPECT_EQ(&*It++, Add1);
  EXPECT_EQ(&*It++, Ret);
  EXPECT_EQ(It, BB->end());

  // Check moveBefore(BasicBlock &, BasicBlock::iterator) with tracking enabled.
  Ctx.save();
  Add1->moveBefore(*BB, Add0->getIterator());
  It = BB->begin();
  EXPECT_EQ(&*It++, Add1);
  EXPECT_EQ(&*It++, Add0);
  EXPECT_EQ(&*It++, Ret);
  EXPECT_EQ(It, BB->end());
  // Check revert().
  Ctx.revert();
  It = BB->begin();
  EXPECT_EQ(&*It++, Add0);
  EXPECT_EQ(&*It++, Add1);
  EXPECT_EQ(&*It++, Ret);
  EXPECT_EQ(It, BB->end());

  // Same for the last instruction in the block.
  Ctx.save();
  Ret->moveBefore(*BB, Add0->getIterator());
  It = BB->begin();
  EXPECT_EQ(&*It++, Ret);
  EXPECT_EQ(&*It++, Add0);
  EXPECT_EQ(&*It++, Add1);
  EXPECT_EQ(It, BB->end());
  // Check revert().
  Ctx.revert();
  It = BB->begin();
  EXPECT_EQ(&*It++, Add0);
  EXPECT_EQ(&*It++, Add1);
  EXPECT_EQ(&*It++, Ret);
  EXPECT_EQ(It, BB->end());

  // Check moveAfter(Instruction *) with tracking enabled.
  Ctx.save();
  Add0->moveAfter(Add1);
  It = BB->begin();
  EXPECT_EQ(&*It++, Add1);
  EXPECT_EQ(&*It++, Add0);
  EXPECT_EQ(&*It++, Ret);
  EXPECT_EQ(It, BB->end());
  // Check revert().
  Ctx.revert();
  It = BB->begin();
  EXPECT_EQ(&*It++, Add0);
  EXPECT_EQ(&*It++, Add1);
  EXPECT_EQ(&*It++, Ret);
  EXPECT_EQ(It, BB->end());

  // Same for the last instruction in the block.
  Ctx.save();
  Ret->moveAfter(Add0);
  It = BB->begin();
  EXPECT_EQ(&*It++, Add0);
  EXPECT_EQ(&*It++, Ret);
  EXPECT_EQ(&*It++, Add1);
  EXPECT_EQ(It, BB->end());
  // Check revert().
  Ctx.revert();
  It = BB->begin();
  EXPECT_EQ(&*It++, Add0);
  EXPECT_EQ(&*It++, Add1);
  EXPECT_EQ(&*It++, Ret);
  EXPECT_EQ(It, BB->end());
}

// TODO: Test multi-instruction patterns.
TEST_F(TrackerTest, InsertIntoBB) {
  parseIR(C, R"IR(
define void @foo(i32 %arg) {
  %add0 = add i32 %arg, %arg
  ret void
}
)IR");
  Function &LLVMF = *M->getFunction("foo");
  sandboxir::Context Ctx(C);

  auto *F = Ctx.createFunction(&LLVMF);
  auto *BB = &*F->begin();
  auto It = BB->begin();
  sandboxir::Instruction *Add0 = &*It++;
  sandboxir::Instruction *Ret = &*It++;
  // Detach `Add0` before we save.
  Add0->removeFromParent();

  // Check insertBefore(Instruction *) with tracking enabled.
  Ctx.save();
  Add0->insertBefore(Ret);
  It = BB->begin();
  EXPECT_EQ(&*It++, Add0);
  EXPECT_EQ(&*It++, Ret);
  EXPECT_EQ(It, BB->end());
  // Check revert().
  Ctx.revert();
  It = BB->begin();
  EXPECT_EQ(&*It++, Ret);
  EXPECT_EQ(It, BB->end());

  // Check insertAfter(Instruction *) with tracking enabled.
  Ctx.save();
  Add0->insertAfter(Ret);
  It = BB->begin();
  EXPECT_EQ(&*It++, Ret);
  EXPECT_EQ(&*It++, Add0);
  EXPECT_EQ(It, BB->end());
  // Check revert().
  Ctx.revert();
  It = BB->begin();
  EXPECT_EQ(&*It++, Ret);
  EXPECT_EQ(It, BB->end());

  // Check insertInto(BasicBlock *, BasicBlock::iterator) with tracking enabled.
  Ctx.save();
  Add0->insertInto(BB, Ret->getIterator());
  It = BB->begin();
  EXPECT_EQ(&*It++, Add0);
  EXPECT_EQ(&*It++, Ret);
  EXPECT_EQ(It, BB->end());
  // Check revert().
  Ctx.revert();
  It = BB->begin();
  EXPECT_EQ(&*It++, Ret);
  EXPECT_EQ(It, BB->end());

  // To make sure we don't leak memory insert `Add0` back into the BB before the
  // end of the test.
  Add0->insertBefore(Ret);
}

// TODO: Test multi-instruction patterns.
TEST_F(TrackerTest, CreateAndInsertInst) {
  parseIR(C, R"IR(
define void @foo(ptr %ptr) {
  %ld = load i8, ptr %ptr, align 64
  ret void
}
)IR");
  Function &LLVMF = *M->getFunction("foo");
  sandboxir::Context Ctx(C);

  auto *F = Ctx.createFunction(&LLVMF);
  auto *Ptr = F->getArg(0);
  auto *BB = &*F->begin();
  auto It = BB->begin();
  auto *Ld = cast<sandboxir::LoadInst>(&*It++);
  auto *Ret = &*It++;

  Ctx.save();
  // Check create(InsertBefore) with tracking enabled.
  sandboxir::LoadInst *NewLd = sandboxir::LoadInst::create(
      Ld->getType(), Ptr, Align(8),
      /*InsertBefore=*/Ld->getIterator(), Ctx, "NewLd");
  It = BB->begin();
  EXPECT_EQ(&*It++, NewLd);
  EXPECT_EQ(&*It++, Ld);
  EXPECT_EQ(&*It++, Ret);
  EXPECT_EQ(It, BB->end());
  // Check revert().
  Ctx.revert();
  It = BB->begin();
  EXPECT_EQ(&*It++, Ld);
  EXPECT_EQ(&*It++, Ret);
  EXPECT_EQ(It, BB->end());
}

TEST_F(TrackerTest, FenceInstSetters) {
  parseIR(C, R"IR(
define void @foo() {
  fence syncscope("singlethread") seq_cst
  ret void
}
)IR");
  llvm::Function *LLVMF = &*M->getFunction("foo");
  sandboxir::Context Ctx(C);
  sandboxir::Function *F = Ctx.createFunction(LLVMF);
  auto *BB = &*F->begin();
  auto It = BB->begin();
  auto *Fence = cast<sandboxir::FenceInst>(&*It++);

  // Check setOrdering().
  auto OrigOrdering = Fence->getOrdering();
  auto NewOrdering = AtomicOrdering::Release;
  EXPECT_NE(NewOrdering, OrigOrdering);
  Ctx.save();
  Fence->setOrdering(NewOrdering);
  EXPECT_EQ(Fence->getOrdering(), NewOrdering);
  Ctx.revert();
  EXPECT_EQ(Fence->getOrdering(), OrigOrdering);
  // Check setSyncScopeID().
  auto OrigSSID = Fence->getSyncScopeID();
  auto NewSSID = SyncScope::System;
  EXPECT_NE(NewSSID, OrigSSID);
  Ctx.save();
  Fence->setSyncScopeID(NewSSID);
  EXPECT_EQ(Fence->getSyncScopeID(), NewSSID);
  Ctx.revert();
  EXPECT_EQ(Fence->getSyncScopeID(), OrigSSID);
}

TEST_F(TrackerTest, CallBaseSetters) {
  parseIR(C, R"IR(
declare void @bar1(i8)
declare void @bar2(i8)

define void @foo(i8 %arg0, i8 %arg1) {
  call void @bar1(i8 %arg0)
  ret void
}
)IR");
  Function &LLVMF = *M->getFunction("foo");
  sandboxir::Context Ctx(C);

  auto *F = Ctx.createFunction(&LLVMF);
  unsigned ArgIdx = 0;
  auto *Arg0 = F->getArg(ArgIdx++);
  auto *Arg1 = F->getArg(ArgIdx++);
  auto *BB = &*F->begin();
  auto It = BB->begin();
  auto *Call = cast<sandboxir::CallBase>(&*It++);
  [[maybe_unused]] auto *Ret = cast<sandboxir::ReturnInst>(&*It++);

  // Check setArgOperand().
  Ctx.save();
  Call->setArgOperand(0, Arg1);
  EXPECT_EQ(Call->getArgOperand(0), Arg1);
  Ctx.revert();
  EXPECT_EQ(Call->getArgOperand(0), Arg0);

  auto *Bar1F = Call->getCalledFunction();
  auto *Bar2F = Ctx.createFunction(M->getFunction("bar2"));

  // Check setCalledOperand().
  Ctx.save();
  Call->setCalledOperand(Bar2F);
  EXPECT_EQ(Call->getCalledOperand(), Bar2F);
  Ctx.revert();
  EXPECT_EQ(Call->getCalledOperand(), Bar1F);

  // Check setCalledFunction().
  Ctx.save();
  Call->setCalledFunction(Bar2F);
  EXPECT_EQ(Call->getCalledFunction(), Bar2F);
  Ctx.revert();
  EXPECT_EQ(Call->getCalledFunction(), Bar1F);
}

TEST_F(TrackerTest, InvokeSetters) {
  parseIR(C, R"IR(
define void @foo(i8 %arg) {
 bb0:
   invoke i8 @foo(i8 %arg) to label %normal_bb
                       unwind label %exception_bb
 normal_bb:
   ret void
 exception_bb:
   ret void
 other_bb:
   ret void
}
)IR");
  Function &LLVMF = *M->getFunction("foo");
  sandboxir::Context Ctx(C);
  [[maybe_unused]] auto &F = *Ctx.createFunction(&LLVMF);
  auto *BB0 = cast<sandboxir::BasicBlock>(
      Ctx.getValue(getBasicBlockByName(LLVMF, "bb0")));
  auto *NormalBB = cast<sandboxir::BasicBlock>(
      Ctx.getValue(getBasicBlockByName(LLVMF, "normal_bb")));
  auto *ExceptionBB = cast<sandboxir::BasicBlock>(
      Ctx.getValue(getBasicBlockByName(LLVMF, "exception_bb")));
  auto *OtherBB = cast<sandboxir::BasicBlock>(
      Ctx.getValue(getBasicBlockByName(LLVMF, "other_bb")));
  auto It = BB0->begin();
  auto *Invoke = cast<sandboxir::InvokeInst>(&*It++);

  // Check setNormalDest().
  Ctx.save();
  Invoke->setNormalDest(OtherBB);
  EXPECT_EQ(Invoke->getNormalDest(), OtherBB);
  Ctx.revert();
  EXPECT_EQ(Invoke->getNormalDest(), NormalBB);

  // Check setUnwindDest().
  Ctx.save();
  Invoke->setUnwindDest(OtherBB);
  EXPECT_EQ(Invoke->getUnwindDest(), OtherBB);
  Ctx.revert();
  EXPECT_EQ(Invoke->getUnwindDest(), ExceptionBB);

  // Check setSuccessor().
  Ctx.save();
  Invoke->setSuccessor(0, OtherBB);
  EXPECT_EQ(Invoke->getSuccessor(0), OtherBB);
  Ctx.revert();
  EXPECT_EQ(Invoke->getSuccessor(0), NormalBB);

  Ctx.save();
  Invoke->setSuccessor(1, OtherBB);
  EXPECT_EQ(Invoke->getSuccessor(1), OtherBB);
  Ctx.revert();
  EXPECT_EQ(Invoke->getSuccessor(1), ExceptionBB);
}

TEST_F(TrackerTest, CatchSwitchInst) {
  parseIR(C, R"IR(
define void @foo(i32 %cond0, i32 %cond1) {
  bb0:
    %cs0 = catchswitch within none [label %handler0, label %handler1] unwind to caller
  bb1:
    %cs1 = catchswitch within %cs0 [label %handler0, label %handler1] unwind label %cleanup
  handler0:
    ret void
  handler1:
    ret void
  cleanup:
    ret void
}
)IR");
  Function &LLVMF = *M->getFunction("foo");

  sandboxir::Context Ctx(C);
  [[maybe_unused]] auto &F = *Ctx.createFunction(&LLVMF);
  auto *BB0 = cast<sandboxir::BasicBlock>(
      Ctx.getValue(getBasicBlockByName(LLVMF, "bb0")));
  auto *BB1 = cast<sandboxir::BasicBlock>(
      Ctx.getValue(getBasicBlockByName(LLVMF, "bb1")));
  auto *Handler0 = cast<sandboxir::BasicBlock>(
      Ctx.getValue(getBasicBlockByName(LLVMF, "handler0")));
  auto *Handler1 = cast<sandboxir::BasicBlock>(
      Ctx.getValue(getBasicBlockByName(LLVMF, "handler1")));
  auto *CS0 = cast<sandboxir::CatchSwitchInst>(&*BB0->begin());
  auto *CS1 = cast<sandboxir::CatchSwitchInst>(&*BB1->begin());

  // Check setParentPad().
  auto *OrigPad = CS0->getParentPad();
  auto *NewPad = CS1;
  EXPECT_NE(NewPad, OrigPad);
  Ctx.save();
  CS0->setParentPad(NewPad);
  EXPECT_EQ(CS0->getParentPad(), NewPad);
  Ctx.revert();
  EXPECT_EQ(CS0->getParentPad(), OrigPad);
  // Check setUnwindDest().
  auto *OrigUnwindDest = CS1->getUnwindDest();
  auto *NewUnwindDest = BB0;
  EXPECT_NE(NewUnwindDest, OrigUnwindDest);
  Ctx.save();
  CS1->setUnwindDest(NewUnwindDest);
  EXPECT_EQ(CS1->getUnwindDest(), NewUnwindDest);
  Ctx.revert();
  EXPECT_EQ(CS1->getUnwindDest(), OrigUnwindDest);
  // Check setSuccessor().
  auto *OrigSuccessor = CS0->getSuccessor(0);
  auto *NewSuccessor = BB0;
  EXPECT_NE(NewSuccessor, OrigSuccessor);
  Ctx.save();
  CS0->setSuccessor(0, NewSuccessor);
  EXPECT_EQ(CS0->getSuccessor(0), NewSuccessor);
  Ctx.revert();
  EXPECT_EQ(CS0->getSuccessor(0), OrigSuccessor);
  // Check addHandler().
  Ctx.save();
  CS0->addHandler(BB0);
  EXPECT_EQ(CS0->getNumHandlers(), 3u);
  Ctx.revert();
  EXPECT_EQ(CS0->getNumHandlers(), 2u);
  auto HIt = CS0->handler_begin();
  EXPECT_EQ(*HIt++, Handler0);
  EXPECT_EQ(*HIt++, Handler1);
}

TEST_F(TrackerTest, LandingPadInstSetters) {
  parseIR(C, R"IR(
define void @foo() {
entry:
  invoke void @foo()
      to label %bb unwind label %unwind
unwind:
  %lpad = landingpad { ptr, i32 }
            catch ptr null
  ret void
bb:
  ret void
}
)IR");
  Function &LLVMF = *M->getFunction("foo");
  auto *LLVMUnwind = getBasicBlockByName(LLVMF, "unwind");

  sandboxir::Context Ctx(C);
  [[maybe_unused]] auto &F = *Ctx.createFunction(&LLVMF);
  auto *Unwind = cast<sandboxir::BasicBlock>(Ctx.getValue(LLVMUnwind));
  auto It = Unwind->begin();
  auto *LPad = cast<sandboxir::LandingPadInst>(&*It++);
  [[maybe_unused]] auto *Ret = cast<sandboxir::ReturnInst>(&*It++);

  // Check setCleanup().
  auto OrigIsCleanup = LPad->isCleanup();
  auto NewIsCleanup = true;
  EXPECT_NE(NewIsCleanup, OrigIsCleanup);
  Ctx.save();
  LPad->setCleanup(NewIsCleanup);
  EXPECT_EQ(LPad->isCleanup(), NewIsCleanup);
  Ctx.revert();
  EXPECT_EQ(LPad->isCleanup(), OrigIsCleanup);
}

TEST_F(TrackerTest, CatchReturnInstSetters) {
  parseIR(C, R"IR(
define void @foo() {
dispatch:
  %cs = catchswitch within none [label %catch] unwind to caller
catch:
  %catchpad = catchpad within %cs [ptr @foo]
  catchret from %catchpad to label %continue
continue:
  ret void
catch2:
  %catchpad2 = catchpad within %cs [ptr @foo]
  ret void
}
)IR");
  Function &LLVMF = *M->getFunction("foo");
  BasicBlock *LLVMCatch = getBasicBlockByName(LLVMF, "catch");
  auto LLVMIt = LLVMCatch->begin();
  [[maybe_unused]] auto *LLVMCP = cast<llvm::CatchPadInst>(&*LLVMIt++);

  sandboxir::Context Ctx(C);
  [[maybe_unused]] auto &F = *Ctx.createFunction(&LLVMF);
  auto *Catch = cast<sandboxir::BasicBlock>(Ctx.getValue(LLVMCatch));
  auto *Catch2 = cast<sandboxir::BasicBlock>(
      Ctx.getValue(getBasicBlockByName(LLVMF, "catch2")));
  auto It = Catch->begin();
  [[maybe_unused]] auto *CP = cast<sandboxir::CatchPadInst>(&*It++);
  auto *CR = cast<sandboxir::CatchReturnInst>(&*It++);
  auto *CP2 = cast<sandboxir::CatchPadInst>(&*Catch2->begin());

  // Check setCatchPad().
  auto *OrigCP = CR->getCatchPad();
  auto *NewCP = CP2;
  EXPECT_NE(NewCP, OrigCP);
  Ctx.save();
  CR->setCatchPad(NewCP);
  EXPECT_EQ(CR->getCatchPad(), NewCP);
  Ctx.revert();
  EXPECT_EQ(CR->getCatchPad(), OrigCP);
  // Check setSuccessor().
  auto *OrigSucc = CR->getSuccessor();
  auto *NewSucc = Catch;
  EXPECT_NE(NewSucc, OrigSucc);
  Ctx.save();
  CR->setSuccessor(NewSucc);
  EXPECT_EQ(CR->getSuccessor(), NewSucc);
  Ctx.revert();
  EXPECT_EQ(CR->getSuccessor(), OrigSucc);
}

TEST_F(TrackerTest, CleanupReturnInstSetters) {
  parseIR(C, R"IR(
define void @foo() {
dispatch:
  invoke void @foo()
              to label %throw unwind label %cleanup
throw:
  ret void
cleanup:
  %cleanuppad = cleanuppad within none []
  cleanupret from %cleanuppad unwind label %cleanup2
cleanup2:
  %cleanuppad2 = cleanuppad within none []
  ret void
}
)IR");
  Function &LLVMF = *M->getFunction("foo");
  BasicBlock *LLVMCleanup = getBasicBlockByName(LLVMF, "cleanup");

  sandboxir::Context Ctx(C);
  [[maybe_unused]] auto &F = *Ctx.createFunction(&LLVMF);
  auto *Throw = cast<sandboxir::BasicBlock>(
      Ctx.getValue(getBasicBlockByName(LLVMF, "throw")));
  auto *Cleanup = cast<sandboxir::BasicBlock>(Ctx.getValue(LLVMCleanup));
  auto *Cleanup2 = cast<sandboxir::BasicBlock>(
      Ctx.getValue(getBasicBlockByName(LLVMF, "cleanup2")));
  auto It = Cleanup->begin();
  [[maybe_unused]] auto *CP = cast<sandboxir::CleanupPadInst>(&*It++);
  auto *CRI = cast<sandboxir::CleanupReturnInst>(&*It++);
  auto *CP2 = cast<sandboxir::CleanupPadInst>(&*Cleanup2->begin());

  // Check setCleanupPad().
  auto *OrigCleanupPad = CRI->getCleanupPad();
  auto *NewCleanupPad = CP2;
  EXPECT_NE(NewCleanupPad, OrigCleanupPad);
  Ctx.save();
  CRI->setCleanupPad(NewCleanupPad);
  EXPECT_EQ(CRI->getCleanupPad(), NewCleanupPad);
  Ctx.revert();
  EXPECT_EQ(CRI->getCleanupPad(), OrigCleanupPad);
  // Check setUnwindDest().
  auto *OrigUnwindDest = CRI->getUnwindDest();
  auto *NewUnwindDest = Throw;
  EXPECT_NE(NewUnwindDest, OrigUnwindDest);
  Ctx.save();
  CRI->setUnwindDest(NewUnwindDest);
  EXPECT_EQ(CRI->getUnwindDest(), NewUnwindDest);
  Ctx.revert();
  EXPECT_EQ(CRI->getUnwindDest(), OrigUnwindDest);
}

TEST_F(TrackerTest, SwitchInstSetters) {
  parseIR(C, R"IR(
define void @foo(i32 %cond0, i32 %cond1) {
  entry:
    switch i32 %cond0, label %default [ i32 0, label %bb0
                                        i32 1, label %bb1 ]
  bb0:
    ret void
  bb1:
    ret void
  default:
    ret void
}
)IR");
  Function &LLVMF = *M->getFunction("foo");
  auto *LLVMEntry = getBasicBlockByName(LLVMF, "entry");

  sandboxir::Context Ctx(C);
  auto &F = *Ctx.createFunction(&LLVMF);
  auto *Cond1 = F.getArg(1);
  auto *Entry = cast<sandboxir::BasicBlock>(Ctx.getValue(LLVMEntry));
  auto *BB0 = cast<sandboxir::BasicBlock>(
      Ctx.getValue(getBasicBlockByName(LLVMF, "bb0")));
  auto *BB1 = cast<sandboxir::BasicBlock>(
      Ctx.getValue(getBasicBlockByName(LLVMF, "bb1")));
  auto *Switch = cast<sandboxir::SwitchInst>(&*Entry->begin());

  // Check setCondition().
  auto *OrigCond = Switch->getCondition();
  auto *NewCond = Cond1;
  EXPECT_NE(NewCond, OrigCond);
  Ctx.save();
  Switch->setCondition(NewCond);
  EXPECT_EQ(Switch->getCondition(), NewCond);
  Ctx.revert();
  EXPECT_EQ(Switch->getCondition(), OrigCond);
  // Check setDefaultDest().
  auto *OrigDefaultDest = Switch->getDefaultDest();
  auto *NewDefaultDest = Entry;
  EXPECT_NE(NewDefaultDest, OrigDefaultDest);
  Ctx.save();
  Switch->setDefaultDest(NewDefaultDest);
  EXPECT_EQ(Switch->getDefaultDest(), NewDefaultDest);
  Ctx.revert();
  EXPECT_EQ(Switch->getDefaultDest(), OrigDefaultDest);
  // Check setSuccessor().
  auto *OrigSucc = Switch->getSuccessor(0);
  auto *NewSucc = Entry;
  EXPECT_NE(NewSucc, OrigSucc);
  Ctx.save();
  Switch->setSuccessor(0, NewSucc);
  EXPECT_EQ(Switch->getSuccessor(0), NewSucc);
  Ctx.revert();
  EXPECT_EQ(Switch->getSuccessor(0), OrigSucc);
  // Check addCase().
  auto *Zero = sandboxir::ConstantInt::get(sandboxir::Type::getInt32Ty(Ctx), 0);
  auto *One = sandboxir::ConstantInt::get(sandboxir::Type::getInt32Ty(Ctx), 1);
  auto *FortyTwo =
      sandboxir::ConstantInt::get(sandboxir::Type::getInt32Ty(Ctx), 42);
  Ctx.save();
  Switch->addCase(FortyTwo, Entry);
  EXPECT_EQ(Switch->getNumCases(), 3u);
  EXPECT_EQ(Switch->findCaseDest(Entry), FortyTwo);
  EXPECT_EQ(Switch->findCaseValue(FortyTwo)->getCaseSuccessor(), Entry);
  EXPECT_EQ(Switch->findCaseDest(BB0), Zero);
  EXPECT_EQ(Switch->findCaseDest(BB1), One);
  Ctx.revert();
  EXPECT_EQ(Switch->getNumCases(), 2u);
  EXPECT_EQ(Switch->findCaseDest(BB0), Zero);
  EXPECT_EQ(Switch->findCaseDest(BB1), One);
  // Check removeCase().
  Ctx.save();
  Switch->removeCase(Switch->findCaseValue(Zero));
  EXPECT_EQ(Switch->getNumCases(), 1u);
  EXPECT_EQ(Switch->findCaseDest(BB1), One);
  Ctx.revert();
  EXPECT_EQ(Switch->getNumCases(), 2u);
  EXPECT_EQ(Switch->findCaseDest(BB0), Zero);
  EXPECT_EQ(Switch->findCaseDest(BB1), One);
}

TEST_F(TrackerTest, SwitchInstPreservesSuccesorOrder) {
  parseIR(C, R"IR(
define void @foo(i32 %cond0) {
  entry:
    switch i32 %cond0, label %default [ i32 0, label %bb0
                                        i32 1, label %bb1
                                        i32 2, label %bb2 ]
  bb0:
    ret void
  bb1:
    ret void
  bb2:
    ret void
  default:
    ret void
}
)IR");
  Function &LLVMF = *M->getFunction("foo");
  auto *LLVMEntry = getBasicBlockByName(LLVMF, "entry");

  sandboxir::Context Ctx(C);
  [[maybe_unused]] auto &F = *Ctx.createFunction(&LLVMF);
  auto *Entry = cast<sandboxir::BasicBlock>(Ctx.getValue(LLVMEntry));
  auto *BB0 = cast<sandboxir::BasicBlock>(
      Ctx.getValue(getBasicBlockByName(LLVMF, "bb0")));
  auto *BB1 = cast<sandboxir::BasicBlock>(
      Ctx.getValue(getBasicBlockByName(LLVMF, "bb1")));
  auto *BB2 = cast<sandboxir::BasicBlock>(
      Ctx.getValue(getBasicBlockByName(LLVMF, "bb2")));
  auto *Switch = cast<sandboxir::SwitchInst>(&*Entry->begin());

  auto *DefaultDest = Switch->getDefaultDest();
  auto *Zero = sandboxir::ConstantInt::get(sandboxir::Type::getInt32Ty(Ctx), 0);
  auto *One = sandboxir::ConstantInt::get(sandboxir::Type::getInt32Ty(Ctx), 1);
  auto *Two = sandboxir::ConstantInt::get(sandboxir::Type::getInt32Ty(Ctx), 2);

  // Check that we can properly revert a removeCase multiple positions apart
  // from the end of the operand list.
  Ctx.save();
  Switch->removeCase(Switch->findCaseValue(Zero));
  EXPECT_EQ(Switch->getNumCases(), 2u);
  Ctx.revert();
  EXPECT_EQ(Switch->getNumCases(), 3u);
  EXPECT_EQ(Switch->findCaseDest(BB0), Zero);
  EXPECT_EQ(Switch->findCaseDest(BB1), One);
  EXPECT_EQ(Switch->findCaseDest(BB2), Two);
  EXPECT_EQ(Switch->getSuccessor(0), DefaultDest);
  EXPECT_EQ(Switch->getSuccessor(1), BB0);
  EXPECT_EQ(Switch->getSuccessor(2), BB1);
  EXPECT_EQ(Switch->getSuccessor(3), BB2);

  // Check that we can properly revert a removeCase of the last case.
  Ctx.save();
  Switch->removeCase(Switch->findCaseValue(Two));
  EXPECT_EQ(Switch->getNumCases(), 2u);
  Ctx.revert();
  EXPECT_EQ(Switch->getNumCases(), 3u);
  EXPECT_EQ(Switch->findCaseDest(BB0), Zero);
  EXPECT_EQ(Switch->findCaseDest(BB1), One);
  EXPECT_EQ(Switch->findCaseDest(BB2), Two);
  EXPECT_EQ(Switch->getSuccessor(0), DefaultDest);
  EXPECT_EQ(Switch->getSuccessor(1), BB0);
  EXPECT_EQ(Switch->getSuccessor(2), BB1);
  EXPECT_EQ(Switch->getSuccessor(3), BB2);

  // Check order is preserved after reverting multiple removeCase invocations.
  Ctx.save();
  Switch->removeCase(Switch->findCaseValue(One));
  Switch->removeCase(Switch->findCaseValue(Zero));
  Switch->removeCase(Switch->findCaseValue(Two));
  EXPECT_EQ(Switch->getNumCases(), 0u);
  Ctx.revert();
  EXPECT_EQ(Switch->getNumCases(), 3u);
  EXPECT_EQ(Switch->findCaseDest(BB0), Zero);
  EXPECT_EQ(Switch->findCaseDest(BB1), One);
  EXPECT_EQ(Switch->findCaseDest(BB2), Two);
  EXPECT_EQ(Switch->getSuccessor(0), DefaultDest);
  EXPECT_EQ(Switch->getSuccessor(1), BB0);
  EXPECT_EQ(Switch->getSuccessor(2), BB1);
  EXPECT_EQ(Switch->getSuccessor(3), BB2);
}

TEST_F(TrackerTest, SelectInst) {
  parseIR(C, R"IR(
define void @foo(i1 %c0, i8 %v0, i8 %v1) {
  %sel = select i1 %c0, i8 %v0, i8 %v1
  ret void
}
)IR");
  llvm::Function *LLVMF = &*M->getFunction("foo");
  sandboxir::Context Ctx(C);
  sandboxir::Function *F = Ctx.createFunction(LLVMF);
  auto *V0 = F->getArg(1);
  auto *V1 = F->getArg(2);
  auto *BB = &*F->begin();
  auto It = BB->begin();
  auto *Select = cast<sandboxir::SelectInst>(&*It++);

  // Check tracking for swapValues.
  Ctx.save();
  Select->swapValues();
  EXPECT_EQ(Select->getTrueValue(), V1);
  EXPECT_EQ(Select->getFalseValue(), V0);
  Ctx.revert();
  EXPECT_EQ(Select->getTrueValue(), V0);
  EXPECT_EQ(Select->getFalseValue(), V1);
}

TEST_F(TrackerTest, ShuffleVectorInst) {
  parseIR(C, R"IR(
define void @foo(<2 x i8> %v1, <2 x i8> %v2) {
  %shuf = shufflevector <2 x i8> %v1, <2 x i8> %v2, <2 x i32> <i32 1, i32 2>
  ret void
}
)IR");
  Function &LLVMF = *M->getFunction("foo");
  sandboxir::Context Ctx(C);

  auto *F = Ctx.createFunction(&LLVMF);
  auto *BB = &*F->begin();
  auto It = BB->begin();
  auto *SVI = cast<sandboxir::ShuffleVectorInst>(&*It++);

  // Check setShuffleMask.
  SmallVector<int, 2> OrigMask(SVI->getShuffleMask());
  Ctx.save();
  SVI->setShuffleMask(ArrayRef<int>({0, 0}));
  EXPECT_NE(SVI->getShuffleMask(), ArrayRef<int>(OrigMask));
  Ctx.revert();
  EXPECT_EQ(SVI->getShuffleMask(), ArrayRef<int>(OrigMask));

  // Check commute.
  auto *Op0 = SVI->getOperand(0);
  auto *Op1 = SVI->getOperand(1);
  Ctx.save();
  SVI->commute();
  EXPECT_EQ(SVI->getOperand(0), Op1);
  EXPECT_EQ(SVI->getOperand(1), Op0);
  EXPECT_NE(SVI->getShuffleMask(), ArrayRef<int>(OrigMask));
  Ctx.revert();
  EXPECT_EQ(SVI->getOperand(0), Op0);
  EXPECT_EQ(SVI->getOperand(1), Op1);
  EXPECT_EQ(SVI->getShuffleMask(), ArrayRef<int>(OrigMask));
}

TEST_F(TrackerTest, PossiblyDisjointInstSetters) {
  parseIR(C, R"IR(
define void @foo(i8 %arg0, i8 %arg1) {
  %or = or i8 %arg0, %arg1
  ret void
}
)IR");
  Function &LLVMF = *M->getFunction("foo");
  sandboxir::Context Ctx(C);

  auto &F = *Ctx.createFunction(&LLVMF);
  auto *BB = &*F.begin();
  auto It = BB->begin();
  auto *PDI = cast<sandboxir::PossiblyDisjointInst>(&*It++);

  // Check setIsDisjoint().
  auto OrigIsDisjoint = PDI->isDisjoint();
  auto NewIsDisjoint = true;
  EXPECT_NE(NewIsDisjoint, OrigIsDisjoint);
  Ctx.save();
  PDI->setIsDisjoint(NewIsDisjoint);
  EXPECT_EQ(PDI->isDisjoint(), NewIsDisjoint);
  Ctx.revert();
  EXPECT_EQ(PDI->isDisjoint(), OrigIsDisjoint);
}

TEST_F(TrackerTest, PossiblyNonNegInstSetters) {
  parseIR(C, R"IR(
define void @foo(i32 %arg) {
  %zext = zext i32 %arg to i64
  ret void
}
)IR");
  Function &LLVMF = *M->getFunction("foo");
  sandboxir::Context Ctx(C);

  auto &F = *Ctx.createFunction(&LLVMF);
  auto *BB = &*F.begin();
  auto It = BB->begin();
  auto *PNNI = cast<sandboxir::PossiblyNonNegInst>(&*It++);

  // Check setNonNeg().
  auto OrigNonNeg = PNNI->hasNonNeg();
  auto NewNonNeg = true;
  EXPECT_NE(NewNonNeg, OrigNonNeg);
  Ctx.save();
  PNNI->setNonNeg(NewNonNeg);
  EXPECT_EQ(PNNI->hasNonNeg(), NewNonNeg);
  Ctx.revert();
  EXPECT_EQ(PNNI->hasNonNeg(), OrigNonNeg);
}

TEST_F(TrackerTest, AtomicRMWSetters) {
  parseIR(C, R"IR(
define void @foo(ptr %ptr, i8 %arg) {
  %atomicrmw = atomicrmw add ptr %ptr, i8 %arg acquire, align 128
  ret void
}
)IR");
  Function &LLVMF = *M->getFunction("foo");
  sandboxir::Context Ctx(C);
  auto &F = *Ctx.createFunction(&LLVMF);
  auto *BB = &*F.begin();
  auto It = BB->begin();
  auto *RMW = cast<sandboxir::AtomicRMWInst>(&*It++);

  // Check setAlignment().
  Ctx.save();
  auto OrigAlign = RMW->getAlign();
  Align NewAlign(1024);
  EXPECT_NE(NewAlign, OrigAlign);
  RMW->setAlignment(NewAlign);
  EXPECT_EQ(RMW->getAlign(), NewAlign);
  Ctx.revert();
  EXPECT_EQ(RMW->getAlign(), OrigAlign);

  // Check setVolatile().
  Ctx.save();
  auto OrigIsVolatile = RMW->isVolatile();
  bool NewIsVolatile = true;
  EXPECT_NE(NewIsVolatile, OrigIsVolatile);
  RMW->setVolatile(NewIsVolatile);
  EXPECT_EQ(RMW->isVolatile(), NewIsVolatile);
  Ctx.revert();
  EXPECT_EQ(RMW->isVolatile(), OrigIsVolatile);

  // Check setOrdering().
  Ctx.save();
  auto OrigOrdering = RMW->getOrdering();
  auto NewOrdering = AtomicOrdering::SequentiallyConsistent;
  EXPECT_NE(NewOrdering, OrigOrdering);
  RMW->setOrdering(NewOrdering);
  EXPECT_EQ(RMW->getOrdering(), NewOrdering);
  Ctx.revert();
  EXPECT_EQ(RMW->getOrdering(), OrigOrdering);

  // Check setSyncScopeID().
  Ctx.save();
  auto OrigSSID = RMW->getSyncScopeID();
  auto NewSSID = SyncScope::SingleThread;
  EXPECT_NE(NewSSID, OrigSSID);
  RMW->setSyncScopeID(NewSSID);
  EXPECT_EQ(RMW->getSyncScopeID(), NewSSID);
  Ctx.revert();
  EXPECT_EQ(RMW->getSyncScopeID(), OrigSSID);
}

TEST_F(TrackerTest, AtomicCmpXchgSetters) {
  parseIR(C, R"IR(
define void @foo(ptr %ptr, i8 %cmp, i8 %new) {
  %cmpxchg = cmpxchg ptr %ptr, i8 %cmp, i8 %new monotonic monotonic, align 128
  ret void
}
)IR");
  Function &LLVMF = *M->getFunction("foo");
  sandboxir::Context Ctx(C);
  auto &F = *Ctx.createFunction(&LLVMF);
  auto *BB = &*F.begin();
  auto It = BB->begin();
  auto *CmpXchg = cast<sandboxir::AtomicCmpXchgInst>(&*It++);

  // Check setAlignment().
  Ctx.save();
  auto OrigAlign = CmpXchg->getAlign();
  Align NewAlign(1024);
  EXPECT_NE(NewAlign, OrigAlign);
  CmpXchg->setAlignment(NewAlign);
  EXPECT_EQ(CmpXchg->getAlign(), NewAlign);
  Ctx.revert();
  EXPECT_EQ(CmpXchg->getAlign(), OrigAlign);

  // Check setVolatile().
  Ctx.save();
  auto OrigIsVolatile = CmpXchg->isVolatile();
  bool NewIsVolatile = true;
  EXPECT_NE(NewIsVolatile, OrigIsVolatile);
  CmpXchg->setVolatile(NewIsVolatile);
  EXPECT_EQ(CmpXchg->isVolatile(), NewIsVolatile);
  Ctx.revert();
  EXPECT_EQ(CmpXchg->isVolatile(), OrigIsVolatile);

  // Check setWeak().
  Ctx.save();
  auto OrigIsWeak = CmpXchg->isWeak();
  bool NewIsWeak = true;
  EXPECT_NE(NewIsWeak, OrigIsWeak);
  CmpXchg->setWeak(NewIsWeak);
  EXPECT_EQ(CmpXchg->isWeak(), NewIsWeak);
  Ctx.revert();
  EXPECT_EQ(CmpXchg->isWeak(), OrigIsWeak);

  // Check setSuccessOrdering().
  Ctx.save();
  auto OrigSuccessOrdering = CmpXchg->getSuccessOrdering();
  auto NewSuccessOrdering = AtomicOrdering::SequentiallyConsistent;
  EXPECT_NE(NewSuccessOrdering, OrigSuccessOrdering);
  CmpXchg->setSuccessOrdering(NewSuccessOrdering);
  EXPECT_EQ(CmpXchg->getSuccessOrdering(), NewSuccessOrdering);
  Ctx.revert();
  EXPECT_EQ(CmpXchg->getSuccessOrdering(), OrigSuccessOrdering);

  // Check setFailureOrdering().
  Ctx.save();
  auto OrigFailureOrdering = CmpXchg->getFailureOrdering();
  auto NewFailureOrdering = AtomicOrdering::SequentiallyConsistent;
  EXPECT_NE(NewFailureOrdering, OrigFailureOrdering);
  CmpXchg->setFailureOrdering(NewFailureOrdering);
  EXPECT_EQ(CmpXchg->getFailureOrdering(), NewFailureOrdering);
  Ctx.revert();
  EXPECT_EQ(CmpXchg->getFailureOrdering(), OrigFailureOrdering);

  // Check setSyncScopeID().
  Ctx.save();
  auto OrigSSID = CmpXchg->getSyncScopeID();
  auto NewSSID = SyncScope::SingleThread;
  EXPECT_NE(NewSSID, OrigSSID);
  CmpXchg->setSyncScopeID(NewSSID);
  EXPECT_EQ(CmpXchg->getSyncScopeID(), NewSSID);
  Ctx.revert();
  EXPECT_EQ(CmpXchg->getSyncScopeID(), OrigSSID);
}

TEST_F(TrackerTest, AllocaInstSetters) {
  parseIR(C, R"IR(
define void @foo(i8 %arg) {
  %alloca = alloca i32, align 64
  ret void
}
)IR");
  Function &LLVMF = *M->getFunction("foo");
  sandboxir::Context Ctx(C);
  auto &F = *Ctx.createFunction(&LLVMF);
  auto *BB = &*F.begin();
  auto It = BB->begin();
  auto *Alloca = cast<sandboxir::AllocaInst>(&*It++);

  // Check setAllocatedType().
  Ctx.save();
  auto *OrigTy = Alloca->getAllocatedType();
  auto *NewTy = sandboxir::Type::getInt64Ty(Ctx);
  EXPECT_NE(NewTy, OrigTy);
  Alloca->setAllocatedType(NewTy);
  EXPECT_EQ(Alloca->getAllocatedType(), NewTy);
  Ctx.revert();
  EXPECT_EQ(Alloca->getAllocatedType(), OrigTy);

  // Check setAlignment().
  Ctx.save();
  auto OrigAlign = Alloca->getAlign();
  Align NewAlign(128);
  EXPECT_NE(NewAlign, OrigAlign);
  Alloca->setAlignment(NewAlign);
  EXPECT_EQ(Alloca->getAlign(), NewAlign);
  Ctx.revert();
  EXPECT_EQ(Alloca->getAlign(), OrigAlign);

  // Check setUsedWithInAlloca().
  Ctx.save();
  auto OrigWIA = Alloca->isUsedWithInAlloca();
  bool NewWIA = true;
  EXPECT_NE(NewWIA, OrigWIA);
  Alloca->setUsedWithInAlloca(NewWIA);
  EXPECT_EQ(Alloca->isUsedWithInAlloca(), NewWIA);
  Ctx.revert();
  EXPECT_EQ(Alloca->isUsedWithInAlloca(), OrigWIA);
}

TEST_F(TrackerTest, CallBrSetters) {
  parseIR(C, R"IR(
define void @foo(i8 %arg) {
 bb0:
   callbr void @foo(i8 %arg)
               to label %bb1 [label %bb2]
 bb1:
   ret void
 bb2:
   ret void
 other_bb:
   ret void
}
)IR");
  Function &LLVMF = *M->getFunction("foo");
  sandboxir::Context Ctx(C);
  [[maybe_unused]] auto &F = *Ctx.createFunction(&LLVMF);
  auto *BB0 = cast<sandboxir::BasicBlock>(
      Ctx.getValue(getBasicBlockByName(LLVMF, "bb0")));
  auto *OtherBB = cast<sandboxir::BasicBlock>(
      Ctx.getValue(getBasicBlockByName(LLVMF, "other_bb")));
  auto It = BB0->begin();
  auto *CallBr = cast<sandboxir::CallBrInst>(&*It++);
  // Check setDefaultDest().
  Ctx.save();
  auto *OrigDefaultDest = CallBr->getDefaultDest();
  CallBr->setDefaultDest(OtherBB);
  EXPECT_EQ(CallBr->getDefaultDest(), OtherBB);
  Ctx.revert();
  EXPECT_EQ(CallBr->getDefaultDest(), OrigDefaultDest);

  // Check setIndirectDest().
  Ctx.save();
  auto *OrigIndirectDest = CallBr->getIndirectDest(0);
  CallBr->setIndirectDest(0, OtherBB);
  EXPECT_EQ(CallBr->getIndirectDest(0), OtherBB);
  Ctx.revert();
  EXPECT_EQ(CallBr->getIndirectDest(0), OrigIndirectDest);
}

TEST_F(TrackerTest, FuncletPadInstSetters) {
  parseIR(C, R"IR(
define void @foo() {
dispatch:
  %cs = catchswitch within none [label %handler0] unwind to caller
handler0:
  %catchpad = catchpad within %cs [ptr @foo]
  ret void
handler1:
  %cleanuppad = cleanuppad within %cs [ptr @foo]
  ret void
bb:
  ret void
}
)IR");
  Function &LLVMF = *M->getFunction("foo");
  sandboxir::Context Ctx(C);
  [[maybe_unused]] auto &F = *Ctx.createFunction(&LLVMF);
  auto *Dispatch = cast<sandboxir::BasicBlock>(
      Ctx.getValue(getBasicBlockByName(LLVMF, "dispatch")));
  auto *Handler0 = cast<sandboxir::BasicBlock>(
      Ctx.getValue(getBasicBlockByName(LLVMF, "handler0")));
  auto *Handler1 = cast<sandboxir::BasicBlock>(
      Ctx.getValue(getBasicBlockByName(LLVMF, "handler1")));
  auto *CP = cast<sandboxir::CatchPadInst>(&*Handler0->begin());
  auto *CLP = cast<sandboxir::CleanupPadInst>(&*Handler1->begin());

  for (auto *FPI : {static_cast<sandboxir::FuncletPadInst *>(CP),
                    static_cast<sandboxir::FuncletPadInst *>(CLP)}) {
    // Check setParentPad().
    auto *OrigParentPad = FPI->getParentPad();
    auto *NewParentPad = Dispatch;
    EXPECT_NE(NewParentPad, OrigParentPad);
    Ctx.save();
    FPI->setParentPad(NewParentPad);
    EXPECT_EQ(FPI->getParentPad(), NewParentPad);
    Ctx.revert();
    EXPECT_EQ(FPI->getParentPad(), OrigParentPad);

    // Check setArgOperand().
    auto *OrigArgOperand = FPI->getArgOperand(0);
    auto *NewArgOperand = Dispatch;
    EXPECT_NE(NewArgOperand, OrigArgOperand);
    Ctx.save();
    FPI->setArgOperand(0, NewArgOperand);
    EXPECT_EQ(FPI->getArgOperand(0), NewArgOperand);
    Ctx.revert();
    EXPECT_EQ(FPI->getArgOperand(0), OrigArgOperand);
  }
}

TEST_F(TrackerTest, PHINodeSetters) {
  parseIR(C, R"IR(
define void @foo(i8 %arg0, i8 %arg1, i8 %arg2) {
bb0:
  br label %bb2

bb1:
  %phi = phi i8 [ %arg0, %bb0 ], [ %arg1, %bb1 ]
  br label %bb1

bb2:
  ret void
}
)IR");
  Function &LLVMF = *M->getFunction("foo");
  sandboxir::Context Ctx(C);
  auto &F = *Ctx.createFunction(&LLVMF);
  unsigned ArgIdx = 0;
  auto *Arg0 = F.getArg(ArgIdx++);
  auto *Arg1 = F.getArg(ArgIdx++);
  auto *Arg2 = F.getArg(ArgIdx++);
  auto *BB0 = cast<sandboxir::BasicBlock>(
      Ctx.getValue(getBasicBlockByName(LLVMF, "bb0")));
  auto *BB1 = cast<sandboxir::BasicBlock>(
      Ctx.getValue(getBasicBlockByName(LLVMF, "bb1")));
  auto *BB2 = cast<sandboxir::BasicBlock>(
      Ctx.getValue(getBasicBlockByName(LLVMF, "bb2")));
  auto *PHI = cast<sandboxir::PHINode>(&*BB1->begin());

  // Check setIncomingValue().
  Ctx.save();
  EXPECT_EQ(PHI->getIncomingValue(0), Arg0);
  PHI->setIncomingValue(0, Arg2);
  EXPECT_EQ(PHI->getIncomingValue(0), Arg2);
  Ctx.revert();
  EXPECT_EQ(PHI->getIncomingValue(0), Arg0);
  EXPECT_EQ(PHI->getNumIncomingValues(), 2u);
  EXPECT_EQ(PHI->getIncomingBlock(0), BB0);
  EXPECT_EQ(PHI->getIncomingValue(0), Arg0);
  EXPECT_EQ(PHI->getIncomingBlock(1), BB1);
  EXPECT_EQ(PHI->getIncomingValue(1), Arg1);

  // Check setIncomingBlock().
  Ctx.save();
  EXPECT_EQ(PHI->getIncomingBlock(0), BB0);
  PHI->setIncomingBlock(0, BB2);
  EXPECT_EQ(PHI->getIncomingBlock(0), BB2);
  Ctx.revert();
  EXPECT_EQ(PHI->getIncomingBlock(0), BB0);
  EXPECT_EQ(PHI->getNumIncomingValues(), 2u);
  EXPECT_EQ(PHI->getIncomingBlock(0), BB0);
  EXPECT_EQ(PHI->getIncomingValue(0), Arg0);
  EXPECT_EQ(PHI->getIncomingBlock(1), BB1);
  EXPECT_EQ(PHI->getIncomingValue(1), Arg1);

  // Check addIncoming().
  Ctx.save();
  EXPECT_EQ(PHI->getNumIncomingValues(), 2u);
  PHI->addIncoming(Arg1, BB2);
  EXPECT_EQ(PHI->getNumIncomingValues(), 3u);
  EXPECT_EQ(PHI->getIncomingBlock(2), BB2);
  EXPECT_EQ(PHI->getIncomingValue(2), Arg1);
  Ctx.revert();
  EXPECT_EQ(PHI->getNumIncomingValues(), 2u);
  EXPECT_EQ(PHI->getIncomingBlock(0), BB0);
  EXPECT_EQ(PHI->getIncomingValue(0), Arg0);
  EXPECT_EQ(PHI->getIncomingBlock(1), BB1);
  EXPECT_EQ(PHI->getIncomingValue(1), Arg1);

  // Check removeIncomingValue(1).
  Ctx.save();
  PHI->removeIncomingValue(1);
  EXPECT_EQ(PHI->getNumIncomingValues(), 1u);
  EXPECT_EQ(PHI->getIncomingBlock(0), BB0);
  EXPECT_EQ(PHI->getIncomingValue(0), Arg0);
  Ctx.revert();
  EXPECT_EQ(PHI->getNumIncomingValues(), 2u);
  EXPECT_EQ(PHI->getIncomingBlock(0), BB0);
  EXPECT_EQ(PHI->getIncomingValue(0), Arg0);
  EXPECT_EQ(PHI->getIncomingBlock(1), BB1);
  EXPECT_EQ(PHI->getIncomingValue(1), Arg1);

  // Check removeIncomingValue(0).
  Ctx.save();
  PHI->removeIncomingValue(0u);
  EXPECT_EQ(PHI->getNumIncomingValues(), 1u);
  EXPECT_EQ(PHI->getIncomingBlock(0), BB1);
  EXPECT_EQ(PHI->getIncomingValue(0), Arg1);
  Ctx.revert();
  EXPECT_EQ(PHI->getNumIncomingValues(), 2u);
  EXPECT_EQ(PHI->getIncomingBlock(0), BB0);
  EXPECT_EQ(PHI->getIncomingValue(0), Arg0);
  EXPECT_EQ(PHI->getIncomingBlock(1), BB1);
  EXPECT_EQ(PHI->getIncomingValue(1), Arg1);

  // Check removeIncomingValueIf(FromBB1).
  Ctx.save();
  PHI->removeIncomingValueIf(
      [&](unsigned Idx) { return PHI->getIncomingBlock(Idx) == BB1; });
  EXPECT_EQ(PHI->getNumIncomingValues(), 1u);
  Ctx.revert();
  EXPECT_EQ(PHI->getNumIncomingValues(), 2u);
  EXPECT_EQ(PHI->getIncomingBlock(0), BB0);
  EXPECT_EQ(PHI->getIncomingBlock(1), BB1);
  // Check removeIncomingValue() remove all.
  Ctx.save();
  PHI->removeIncomingValue(0u);
  EXPECT_EQ(PHI->getNumIncomingValues(), 1u);
  EXPECT_EQ(PHI->getIncomingBlock(0), BB1);
  EXPECT_EQ(PHI->getIncomingValue(0), Arg1);
  PHI->removeIncomingValue(0u);
  EXPECT_EQ(PHI->getNumIncomingValues(), 0u);
  Ctx.revert();
  EXPECT_EQ(PHI->getNumIncomingValues(), 2u);
  EXPECT_EQ(PHI->getIncomingBlock(0), BB0);
  EXPECT_EQ(PHI->getIncomingValue(0), Arg0);
  EXPECT_EQ(PHI->getIncomingBlock(1), BB1);
  EXPECT_EQ(PHI->getIncomingValue(1), Arg1);

  // Check removeIncomingValue(BasicBlock *).
  Ctx.save();
  PHI->removeIncomingValue(BB1);
  EXPECT_EQ(PHI->getNumIncomingValues(), 1u);
  EXPECT_EQ(PHI->getIncomingBlock(0), BB0);
  EXPECT_EQ(PHI->getIncomingValue(0), Arg0);
  Ctx.revert();
  EXPECT_EQ(PHI->getNumIncomingValues(), 2u);
  EXPECT_EQ(PHI->getIncomingBlock(0), BB0);
  EXPECT_EQ(PHI->getIncomingValue(0), Arg0);
  EXPECT_EQ(PHI->getIncomingBlock(1), BB1);
  EXPECT_EQ(PHI->getIncomingValue(1), Arg1);
}

void checkCmpInst(sandboxir::Context &Ctx, sandboxir::CmpInst *Cmp) {
  Ctx.save();
  auto OrigP = Cmp->getPredicate();
  auto NewP = Cmp->getSwappedPredicate();
  Cmp->setPredicate(NewP);
  EXPECT_EQ(Cmp->getPredicate(), NewP);
  Ctx.revert();
  EXPECT_EQ(Cmp->getPredicate(), OrigP);

  Ctx.save();
  auto OrigOp0 = Cmp->getOperand(0);
  auto OrigOp1 = Cmp->getOperand(1);
  Cmp->swapOperands();
  EXPECT_EQ(Cmp->getPredicate(), NewP);
  EXPECT_EQ(Cmp->getOperand(0), OrigOp1);
  EXPECT_EQ(Cmp->getOperand(1), OrigOp0);
  Ctx.revert();
  EXPECT_EQ(Cmp->getPredicate(), OrigP);
  EXPECT_EQ(Cmp->getOperand(0), OrigOp0);
  EXPECT_EQ(Cmp->getOperand(1), OrigOp1);
}

TEST_F(TrackerTest, CmpInst) {
  SCOPED_TRACE("TrackerTest sandboxir::CmpInst tests");
  parseIR(C, R"IR(
define void @foo(i64 %i0, i64 %i1, float %f0, float %f1) {
  %foeq = fcmp ogt float %f0, %f1
  %ioeq = icmp uge i64 %i0, %i1

  ret void
}
)IR");
  Function &LLVMF = *M->getFunction("foo");
  sandboxir::Context Ctx(C);
  auto &F = *Ctx.createFunction(&LLVMF);
  auto *BB = &*F.begin();
  auto It = BB->begin();
  auto *FCmp = cast<sandboxir::CmpInst>(&*It++);
  checkCmpInst(Ctx, FCmp);
  auto *ICmp = cast<sandboxir::CmpInst>(&*It++);
  checkCmpInst(Ctx, ICmp);
}

TEST_F(TrackerTest, GlobalValueSetters) {
  parseIR(C, R"IR(
define void @foo() {
  call void @foo()
  ret void
}
)IR");
  Function &LLVMF = *M->getFunction("foo");
  sandboxir::Context Ctx(C);

  auto &F = *Ctx.createFunction(&LLVMF);
  auto *BB = &*F.begin();
  auto *Call = cast<sandboxir::CallInst>(&*BB->begin());

  auto *GV = cast<sandboxir::GlobalValue>(Call->getCalledOperand());
  // Check setUnnamedAddr().
  auto OrigUnnamedAddr = GV->getUnnamedAddr();
  auto NewUnnamedAddr = sandboxir::GlobalValue::UnnamedAddr::Global;
  EXPECT_NE(NewUnnamedAddr, OrigUnnamedAddr);
  Ctx.save();
  GV->setUnnamedAddr(NewUnnamedAddr);
  EXPECT_EQ(GV->getUnnamedAddr(), NewUnnamedAddr);
  Ctx.revert();
  EXPECT_EQ(GV->getUnnamedAddr(), OrigUnnamedAddr);

  // Check setVisibility().
  auto OrigVisibility = GV->getVisibility();
  auto NewVisibility =
      sandboxir::GlobalValue::VisibilityTypes::ProtectedVisibility;
  EXPECT_NE(NewVisibility, OrigVisibility);
  Ctx.save();
  GV->setVisibility(NewVisibility);
  EXPECT_EQ(GV->getVisibility(), NewVisibility);
  Ctx.revert();
  EXPECT_EQ(GV->getVisibility(), OrigVisibility);
}

TEST_F(TrackerTest, GlobalIFuncSetters) {
  parseIR(C, R"IR(
declare external void @bar()
@ifunc = ifunc void(), ptr @foo
define void @foo() {
  call void @ifunc()
  call void @bar()
  ret void
}
)IR");
  Function &LLVMF = *M->getFunction("foo");
  sandboxir::Context Ctx(C);

  auto &F = *Ctx.createFunction(&LLVMF);
  auto *BB = &*F.begin();
  auto It = BB->begin();
  auto *Call0 = cast<sandboxir::CallInst>(&*It++);
  auto *Call1 = cast<sandboxir::CallInst>(&*It++);
  // Check classof(), creation.
  auto *IFunc = cast<sandboxir::GlobalIFunc>(Call0->getCalledOperand());
  auto *Bar = cast<sandboxir::Function>(Call1->getCalledOperand());
  // Check setResolver().
  auto *OrigResolver = IFunc->getResolver();
  auto *NewResolver = Bar;
  EXPECT_NE(NewResolver, OrigResolver);
  Ctx.save();
  IFunc->setResolver(NewResolver);
  EXPECT_EQ(IFunc->getResolver(), NewResolver);
  Ctx.revert();
  EXPECT_EQ(IFunc->getResolver(), OrigResolver);
}

TEST_F(TrackerTest, GlobalVariableSetters) {
  parseIR(C, R"IR(
@glob0 = global i32 42
@glob1 = global i32 43
define void @foo() {
  %ld0 = load i32, ptr @glob0
  %ld1 = load i32, ptr @glob1
  ret void
}
)IR");
  Function &LLVMF = *M->getFunction("foo");
  sandboxir::Context Ctx(C);

  auto &F = *Ctx.createFunction(&LLVMF);
  auto *BB = &*F.begin();
  auto It = BB->begin();
  auto *Ld0 = cast<sandboxir::LoadInst>(&*It++);
  auto *Ld1 = cast<sandboxir::LoadInst>(&*It++);
  // Check classof(), creation.
  auto *GV0 = cast<sandboxir::GlobalVariable>(Ld0->getPointerOperand());
  auto *GV1 = cast<sandboxir::GlobalVariable>(Ld1->getPointerOperand());
  // Check setInitializer().
  auto *OrigInitializer = GV0->getInitializer();
  auto *NewInitializer = GV1->getInitializer();
  EXPECT_NE(NewInitializer, OrigInitializer);
  Ctx.save();
  GV0->setInitializer(NewInitializer);
  EXPECT_EQ(GV0->getInitializer(), NewInitializer);
  Ctx.revert();
  EXPECT_EQ(GV0->getInitializer(), OrigInitializer);
  // Check setConstant().
  bool OrigIsConstant = GV0->isConstant();
  bool NewIsConstant = !OrigIsConstant;
  Ctx.save();
  GV0->setConstant(NewIsConstant);
  EXPECT_EQ(GV0->isConstant(), NewIsConstant);
  Ctx.revert();
  EXPECT_EQ(GV0->isConstant(), OrigIsConstant);
  // Check setExternallyInitialized().
  bool OrigIsExtInit = GV0->isExternallyInitialized();
  bool NewIsExtInit = !OrigIsExtInit;
  Ctx.save();
  GV0->setExternallyInitialized(NewIsExtInit);
  EXPECT_EQ(GV0->isExternallyInitialized(), NewIsExtInit);
  Ctx.revert();
  EXPECT_EQ(GV0->isExternallyInitialized(), OrigIsExtInit);
}

TEST_F(TrackerTest, GlobalAliasSetters) {
  parseIR(C, R"IR(
@alias = dso_local alias void(), ptr @foo
declare void @bar();
define void @foo() {
  call void @alias()
  call void @bar()
  ret void
}
)IR");
  Function &LLVMF = *M->getFunction("foo");
  sandboxir::Context Ctx(C);

  auto &F = *Ctx.createFunction(&LLVMF);
  auto *BB = &*F.begin();
  auto It = BB->begin();
  auto *Call0 = cast<sandboxir::CallInst>(&*It++);
  auto *Call1 = cast<sandboxir::CallInst>(&*It++);
  auto *Callee1 = cast<sandboxir::Constant>(Call1->getCalledOperand());
  auto *Alias = cast<sandboxir::GlobalAlias>(Call0->getCalledOperand());
  // Check setAliasee().
  auto *OrigAliasee = Alias->getAliasee();
  auto *NewAliasee = Callee1;
  EXPECT_NE(NewAliasee, OrigAliasee);
  Ctx.save();
  Alias->setAliasee(NewAliasee);
  EXPECT_EQ(Alias->getAliasee(), NewAliasee);
  Ctx.revert();
  EXPECT_EQ(Alias->getAliasee(), OrigAliasee);
}

TEST_F(TrackerTest, SetVolatile) {
  parseIR(C, R"IR(
define void @foo(ptr %arg0, i8 %val) {
  %ld = load i8, ptr %arg0, align 64
  store i8 %val, ptr %arg0, align 64
  ret void
}
)IR");
  Function &LLVMF = *M->getFunction("foo");
  sandboxir::Context Ctx(C);

  auto *F = Ctx.createFunction(&LLVMF);
  auto *BB = &*F->begin();
  auto It = BB->begin();
  auto *Load = cast<sandboxir::LoadInst>(&*It++);
  auto *Store = cast<sandboxir::StoreInst>(&*It++);

  EXPECT_FALSE(Load->isVolatile());
  Ctx.save();
  Load->setVolatile(true);
  EXPECT_TRUE(Load->isVolatile());
  Ctx.revert();
  EXPECT_FALSE(Load->isVolatile());

  EXPECT_FALSE(Store->isVolatile());
  Ctx.save();
  Store->setVolatile(true);
  EXPECT_TRUE(Store->isVolatile());
  Ctx.revert();
  EXPECT_FALSE(Store->isVolatile());
}

TEST_F(TrackerTest, Flags) {
  parseIR(C, R"IR(
define void @foo(i32 %arg, float %farg) {
  %add = add i32 %arg, %arg
  %fadd = fadd float %farg, %farg
  %udiv = udiv i32 %arg, %arg
  ret void
}
)IR");
  Function &LLVMF = *M->getFunction("foo");
  sandboxir::Context Ctx(C);
  auto &F = *Ctx.createFunction(&LLVMF);
  auto *BB = &*F.begin();
  auto It = BB->begin();
  auto *Add = &*It++;
  auto *FAdd = &*It++;
  auto *UDiv = &*It++;

#define CHECK_FLAG(I, GETTER, SETTER)                                          \
  {                                                                            \
    Ctx.save();                                                                \
    bool OrigFlag = I->GETTER();                                               \
    bool NewFlag = !OrigFlag;                                                  \
    I->SETTER(NewFlag);                                                        \
    EXPECT_EQ(I->GETTER(), NewFlag);                                           \
    Ctx.revert();                                                              \
    EXPECT_EQ(I->GETTER(), OrigFlag);                                          \
  }

  CHECK_FLAG(Add, hasNoUnsignedWrap, setHasNoUnsignedWrap);
  CHECK_FLAG(Add, hasNoSignedWrap, setHasNoSignedWrap);
  CHECK_FLAG(FAdd, isFast, setFast);
  CHECK_FLAG(FAdd, hasAllowReassoc, setHasAllowReassoc);
  CHECK_FLAG(UDiv, isExact, setIsExact);
  CHECK_FLAG(FAdd, hasNoNaNs, setHasNoNaNs);
  CHECK_FLAG(FAdd, hasNoInfs, setHasNoInfs);
  CHECK_FLAG(FAdd, hasNoSignedZeros, setHasNoSignedZeros);
  CHECK_FLAG(FAdd, hasAllowReciprocal, setHasAllowReciprocal);
  CHECK_FLAG(FAdd, hasAllowContract, setHasAllowContract);
  CHECK_FLAG(FAdd, hasApproxFunc, setHasApproxFunc);

  // Check setFastMathFlags().
  FastMathFlags OrigFMF = FAdd->getFastMathFlags();
  FastMathFlags NewFMF;
  NewFMF.setAllowReassoc(true);
  EXPECT_TRUE(NewFMF != OrigFMF);

  Ctx.save();
  FAdd->setFastMathFlags(NewFMF);
  EXPECT_FALSE(FAdd->getFastMathFlags() != NewFMF);
  Ctx.revert();
  EXPECT_FALSE(FAdd->getFastMathFlags() != OrigFMF);

  // Check copyFastMathFlags().
  Ctx.save();
  FAdd->copyFastMathFlags(NewFMF);
  EXPECT_FALSE(FAdd->getFastMathFlags() != NewFMF);
  Ctx.revert();
  EXPECT_FALSE(FAdd->getFastMathFlags() != OrigFMF);
}

// IRSnapshotChecker is only defined in debug mode.
#ifndef NDEBUG

TEST_F(TrackerTest, IRSnapshotCheckerNoChanges) {
  parseIR(C, R"IR(
define i32 @foo(i32 %arg) {
  %add0 = add i32 %arg, %arg
  ret i32 %add0
}
)IR");
  Function &LLVMF = *M->getFunction("foo");
  sandboxir::Context Ctx(C);

  [[maybe_unused]] auto *F = Ctx.createFunction(&LLVMF);
  sandboxir::IRSnapshotChecker Checker(Ctx);
  Checker.save();
  Checker.expectNoDiff();
}

TEST_F(TrackerTest, IRSnapshotCheckerDiesWithUnexpectedChanges) {
  parseIR(C, R"IR(
define i32 @foo(i32 %arg) {
  %add0 = add i32 %arg, %arg
  %add1 = add i32 %add0, %arg
  ret i32 %add1
}
)IR");
  Function &LLVMF = *M->getFunction("foo");
  sandboxir::Context Ctx(C);

  auto *F = Ctx.createFunction(&LLVMF);
  auto *BB = &*F->begin();
  auto It = BB->begin();
  sandboxir::Instruction *Add0 = &*It++;
  sandboxir::Instruction *Add1 = &*It++;
  sandboxir::IRSnapshotChecker Checker(Ctx);
  Checker.save();
  Add1->setOperand(1, Add0);
  EXPECT_DEATH(Checker.expectNoDiff(), "Found IR difference");
}

TEST_F(TrackerTest, IRSnapshotCheckerSaveMultipleTimes) {
  parseIR(C, R"IR(
define i32 @foo(i32 %arg) {
  %add0 = add i32 %arg, %arg
  %add1 = add i32 %add0, %arg
  ret i32 %add1
}
)IR");
  Function &LLVMF = *M->getFunction("foo");
  sandboxir::Context Ctx(C);

  auto *F = Ctx.createFunction(&LLVMF);
  auto *BB = &*F->begin();
  auto It = BB->begin();
  sandboxir::Instruction *Add0 = &*It++;
  sandboxir::Instruction *Add1 = &*It++;
  sandboxir::IRSnapshotChecker Checker(Ctx);
  Checker.save();
  Add1->setOperand(1, Add0);
  // Now IR differs from the last snapshot. Let's take a new snapshot.
  Checker.save();
  // The new snapshot should have replaced the old one, so this should succeed.
  Checker.expectNoDiff();
}

#endif // NDEBUG
