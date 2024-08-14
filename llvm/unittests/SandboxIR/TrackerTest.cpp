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
#include "llvm/SandboxIR/SandboxIR.h"
#include "llvm/Support/SourceMgr.h"
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
  sandboxir::LoadInst *NewLd =
      sandboxir::LoadInst::create(Ld->getType(), Ptr, Align(8),
                                  /*InsertBefore=*/Ld, Ctx, "NewLd");
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
  auto *NewTy = Type::getInt64Ty(C);
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
