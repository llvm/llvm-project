//===- SSAUpdaterBulk.cpp - Unit tests for SSAUpdaterBulk -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/SSAUpdaterBulk.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace llvm;

TEST(SSAUpdaterBulk, SimpleMerge) {
  SSAUpdaterBulk Updater;
  LLVMContext C;
  Module M("SSAUpdaterTest", C);
  IRBuilder<> B(C);
  Type *I32Ty = B.getInt32Ty();
  auto *F = Function::Create(FunctionType::get(B.getVoidTy(), {I32Ty}, false),
                             GlobalValue::ExternalLinkage, "F", &M);

  // Generate a simple program:
  //   if:
  //     br i1 true, label %true, label %false
  //   true:
  //     %1 = add i32 %0, 1
  //     %2 = sub i32 %0, 2
  //     br label %merge
  //   false:
  //     %3 = add i32 %0, 3
  //     %4 = sub i32 %0, 4
  //     br label %merge
  //   merge:
  //     %5 = add i32 %1, 5
  //     %6 = add i32 %3, 6
  //     %7 = add i32 %2, %4
  //     %8 = sub i32 %2, %4
  Argument *FirstArg = &*(F->arg_begin());
  BasicBlock *IfBB = BasicBlock::Create(C, "if", F);
  BasicBlock *TrueBB = BasicBlock::Create(C, "true", F);
  BasicBlock *FalseBB = BasicBlock::Create(C, "false", F);
  BasicBlock *MergeBB = BasicBlock::Create(C, "merge", F);

  B.SetInsertPoint(IfBB);
  B.CreateCondBr(B.getTrue(), TrueBB, FalseBB);

  B.SetInsertPoint(TrueBB);
  Value *AddOp1 = B.CreateAdd(FirstArg, ConstantInt::get(I32Ty, 1));
  Value *SubOp1 = B.CreateSub(FirstArg, ConstantInt::get(I32Ty, 2));
  B.CreateBr(MergeBB);

  B.SetInsertPoint(FalseBB);
  Value *AddOp2 = B.CreateAdd(FirstArg, ConstantInt::get(I32Ty, 3));
  Value *SubOp2 = B.CreateSub(FirstArg, ConstantInt::get(I32Ty, 4));
  B.CreateBr(MergeBB);

  B.SetInsertPoint(MergeBB, MergeBB->begin());
  auto *I1 = cast<Instruction>(B.CreateAdd(AddOp1, ConstantInt::get(I32Ty, 5)));
  auto *I2 = cast<Instruction>(B.CreateAdd(AddOp2, ConstantInt::get(I32Ty, 6)));
  auto *I3 = cast<Instruction>(B.CreateAdd(SubOp1, SubOp2));
  auto *I4 = cast<Instruction>(B.CreateSub(SubOp1, SubOp2));

  // Now rewrite uses in instructions %5, %6, %7. They need to use a phi, which
  // SSAUpdater should insert into %merge.
  // Intentionally don't touch %8 to see that SSAUpdater only changes
  // instructions that were explicitly specified.
  unsigned VarNum = Updater.AddVariable("a", I32Ty);
  Updater.AddAvailableValue(VarNum, TrueBB, AddOp1);
  Updater.AddAvailableValue(VarNum, FalseBB, AddOp2);
  Updater.AddUse(VarNum, &I1->getOperandUse(0));
  Updater.AddUse(VarNum, &I2->getOperandUse(0));

  VarNum = Updater.AddVariable("b", I32Ty);
  Updater.AddAvailableValue(VarNum, TrueBB, SubOp1);
  Updater.AddAvailableValue(VarNum, FalseBB, SubOp2);
  Updater.AddUse(VarNum, &I3->getOperandUse(0));
  Updater.AddUse(VarNum, &I3->getOperandUse(1));

  DominatorTree DT(*F);
  Updater.RewriteAllUses(&DT);

  // Check how %5 and %6 were rewritten.
  PHINode *UpdatePhiA = dyn_cast_or_null<PHINode>(I1->getOperand(0));
  EXPECT_NE(UpdatePhiA, nullptr);
  EXPECT_EQ(UpdatePhiA->getIncomingValueForBlock(TrueBB), AddOp1);
  EXPECT_EQ(UpdatePhiA->getIncomingValueForBlock(FalseBB), AddOp2);
  EXPECT_EQ(UpdatePhiA, dyn_cast_or_null<PHINode>(I1->getOperand(0)));

  // Check how %7 was rewritten.
  PHINode *UpdatePhiB = dyn_cast_or_null<PHINode>(I3->getOperand(0));
  EXPECT_EQ(UpdatePhiB->getIncomingValueForBlock(TrueBB), SubOp1);
  EXPECT_EQ(UpdatePhiB->getIncomingValueForBlock(FalseBB), SubOp2);
  EXPECT_EQ(UpdatePhiB, dyn_cast_or_null<PHINode>(I3->getOperand(1)));

  // Check that %8 was kept untouched.
  EXPECT_EQ(I4->getOperand(0), SubOp1);
  EXPECT_EQ(I4->getOperand(1), SubOp2);
}

TEST(SSAUpdaterBulk, Irreducible) {
  SSAUpdaterBulk Updater;
  LLVMContext C;
  Module M("SSAUpdaterTest", C);
  IRBuilder<> B(C);
  Type *I32Ty = B.getInt32Ty();
  auto *F = Function::Create(FunctionType::get(B.getVoidTy(), {I32Ty}, false),
                             GlobalValue::ExternalLinkage, "F", &M);

  // Generate a small program with a multi-entry loop:
  //     if:
  //       %1 = add i32 %0, 1
  //       br i1 true, label %loopmain, label %loopstart
  //
  //     loopstart:
  //       %2 = add i32 %0, 2
  //       br label %loopmain
  //
  //     loopmain:
  //       %3 = add i32 %1, 3
  //       br i1 true, label %loopstart, label %afterloop
  //
  //     afterloop:
  //       %4 = add i32 %2, 4
  //       ret i32 %0
  Argument *FirstArg = &*F->arg_begin();
  BasicBlock *IfBB = BasicBlock::Create(C, "if", F);
  BasicBlock *LoopStartBB = BasicBlock::Create(C, "loopstart", F);
  BasicBlock *LoopMainBB = BasicBlock::Create(C, "loopmain", F);
  BasicBlock *AfterLoopBB = BasicBlock::Create(C, "afterloop", F);

  B.SetInsertPoint(IfBB);
  Value *AddOp1 = B.CreateAdd(FirstArg, ConstantInt::get(I32Ty, 1));
  B.CreateCondBr(B.getTrue(), LoopMainBB, LoopStartBB);

  B.SetInsertPoint(LoopStartBB);
  Value *AddOp2 = B.CreateAdd(FirstArg, ConstantInt::get(I32Ty, 2));
  B.CreateBr(LoopMainBB);

  B.SetInsertPoint(LoopMainBB);
  auto *I1 = cast<Instruction>(B.CreateAdd(AddOp1, ConstantInt::get(I32Ty, 3)));
  B.CreateCondBr(B.getTrue(), LoopStartBB, AfterLoopBB);

  B.SetInsertPoint(AfterLoopBB);
  auto *I2 = cast<Instruction>(B.CreateAdd(AddOp2, ConstantInt::get(I32Ty, 4)));
  ReturnInst *Return = B.CreateRet(FirstArg);

  // Now rewrite uses in instructions %3, %4, and 'ret i32 %0'. Only %4 needs a
  // new phi, others should be able to work with existing values.
  // The phi for %4 should be inserted into LoopMainBB and should look like
  // this:
  //   %b = phi i32 [ %2, %loopstart ], [ undef, %if ]
  // No other rewrites should be made.

  // Add use in %3.
  unsigned VarNum = Updater.AddVariable("c", I32Ty);
  Updater.AddAvailableValue(VarNum, IfBB, AddOp1);
  Updater.AddUse(VarNum, &I1->getOperandUse(0));

  // Add use in %4.
  VarNum = Updater.AddVariable("b", I32Ty);
  Updater.AddAvailableValue(VarNum, LoopStartBB, AddOp2);
  Updater.AddUse(VarNum, &I2->getOperandUse(0));

  // Add use in the return instruction.
  VarNum = Updater.AddVariable("a", I32Ty);
  Updater.AddAvailableValue(VarNum, &F->getEntryBlock(), FirstArg);
  Updater.AddUse(VarNum, &Return->getOperandUse(0));

  // Save all inserted phis into a vector.
  SmallVector<PHINode *, 8> Inserted;
  DominatorTree DT(*F);
  Updater.RewriteAllUses(&DT, &Inserted);

  // Only one phi should have been inserted.
  EXPECT_EQ(Inserted.size(), 1u);

  // I1 and Return should use the same values as they used before.
  EXPECT_EQ(I1->getOperand(0), AddOp1);
  EXPECT_EQ(Return->getOperand(0), FirstArg);

  // I2 should use the new phi.
  PHINode *UpdatePhi = dyn_cast_or_null<PHINode>(I2->getOperand(0));
  EXPECT_NE(UpdatePhi, nullptr);
  EXPECT_EQ(UpdatePhi->getIncomingValueForBlock(LoopStartBB), AddOp2);
  EXPECT_EQ(UpdatePhi->getIncomingValueForBlock(IfBB), UndefValue::get(I32Ty));
}

TEST(SSAUpdaterBulk, SingleBBLoop) {
  const char *IR = R"(
      define void @main() {
      entry:
          br label %loop
      loop:
          %i = add i32 0, 1
          %cmp = icmp slt i32 %i, 42
          br i1 %cmp, label %loop, label %exit
      exit:
          ret void
      }
  )";

  llvm::LLVMContext Context;
  llvm::SMDiagnostic Err;
  std::unique_ptr<llvm::Module> M = llvm::parseAssemblyString(IR, Err, Context);
  ASSERT_NE(M, nullptr) << "Failed to parse IR: " << Err.getMessage();

  Function *F = M->getFunction("main");
  auto *Entry = &F->getEntryBlock();
  auto *Loop = Entry->getSingleSuccessor();
  auto *I = &Loop->front();

  // Rewrite first operand of "%i = add i32 0, 1" to use incoming values entry:0
  // or loop:%i (that is the value of %i from the previous iteration).
  SSAUpdaterBulk Updater;
  Type *I32Ty = Type::getInt32Ty(Context);
  unsigned PrevI = Updater.AddVariable("i.prev", I32Ty);
  Updater.AddAvailableValue(PrevI, Entry, ConstantInt::get(I32Ty, 0));
  Updater.AddAvailableValue(PrevI, Loop, I);
  Updater.AddUse(PrevI, &I->getOperandUse(0));

  SmallVector<PHINode *, 1> Inserted;
  DominatorTree DT(*F);
  Updater.RewriteAllUses(&DT, &Inserted);

#if 0 // Enable for debugging.
  Loop->dump();
  // Output:
  // loop: ; preds = %loop, %entry
  //   %i.prev = phi i32 [ %i, %loop ], [ 0, %entry ]
  //   %i = add i32 %i.prev, 1
  //   %cmp = icmp slt i32 %i, 42
  //   br i1 %cmp, label %loop, label %exit
#endif

  ASSERT_EQ(Inserted.size(), 1u);
  PHINode *Phi = Inserted[0];
  EXPECT_EQ(Phi, dyn_cast<PHINode>(I->getOperand(0)));
  EXPECT_EQ(Phi->getIncomingValueForBlock(Entry), ConstantInt::get(I32Ty, 0));
  EXPECT_EQ(Phi->getIncomingValueForBlock(Loop), I);
}

TEST(SSAUpdaterBulk, TwoBBLoop) {
  const char *IR = R"(
      define void @main() {
      entry:
          br label %loop_header
      loop_header:
          br label %loop
      loop:
          %i = add i32 0, 1
          %cmp = icmp slt i32 %i, 42
          br i1 %cmp, label %loop_header, label %exit
      exit:
          ret void
      }
  )";

  llvm::LLVMContext Context;
  llvm::SMDiagnostic Err;
  std::unique_ptr<llvm::Module> M = llvm::parseAssemblyString(IR, Err, Context);
  ASSERT_NE(M, nullptr) << "Failed to parse IR: " << Err.getMessage();

  Function *F = M->getFunction("main");
  auto *Entry = &F->getEntryBlock();
  auto *LoopHdr = Entry->getSingleSuccessor();
  auto *Loop = LoopHdr->getSingleSuccessor();
  auto *I = &Loop->front();

  // Rewrite first operand of "%i = add i32 0, 1" to use incoming values entry:0
  // or loop:%i (that is the value of %i from the previous iteration).
  SSAUpdaterBulk Updater;
  Type *I32Ty = Type::getInt32Ty(Context);
  unsigned PrevI = Updater.AddVariable("i.prev", I32Ty);
  Updater.AddAvailableValue(PrevI, Entry, ConstantInt::get(I32Ty, 0));
  Updater.AddAvailableValue(PrevI, Loop, I);
  Updater.AddUse(PrevI, &I->getOperandUse(0));

  SmallVector<PHINode *, 1> Inserted;
  DominatorTree DT(*F);
  Updater.RewriteAllUses(&DT, &Inserted);

#if 0 // Enable for debugging.
  LoopHdr->dump();
  Loop->dump();
  // Output:
  // loop_header:                                      ; preds = %loop, %entry
  //   %i.prev = phi i32 [ %i, %loop ], [ 0, %entry ]
  //   br label %loop
  // loop:                                             ; preds = %loop_header
  //   %i = add i32 %i.prev, 1
  //   %cmp = icmp slt i32 %i, 42
  //   br i1 %cmp, label %loop_header, label %exit
#endif

  ASSERT_EQ(Inserted.size(), 1u);
  PHINode *Phi = Inserted[0];
  EXPECT_EQ(Phi, dyn_cast<PHINode>(I->getOperand(0)));
  EXPECT_EQ(Phi->getParent(), LoopHdr);
  EXPECT_EQ(Phi->getIncomingValueForBlock(Entry), ConstantInt::get(I32Ty, 0));
  EXPECT_EQ(Phi->getIncomingValueForBlock(Loop), I);
}

TEST(SSAUpdaterBulk, SimplifyPHIs) {
  const char *IR = R"(
      define void @main(i32 %val, i1 %cond) {
      entry:
          br i1 %cond, label %left, label %right
      left:
          %add = add i32 %val, 1
          br label %exit
      right:
          %sub = sub i32 %val, 1
          br label %exit
      exit:
          %phi = phi i32 [ %sub, %right ], [ %add, %left ]
          %cmp = icmp slt i32 0, 42
          ret void
      }
  )";

  llvm::LLVMContext Context;
  llvm::SMDiagnostic Err;
  std::unique_ptr<llvm::Module> M = llvm::parseAssemblyString(IR, Err, Context);
  ASSERT_NE(M, nullptr) << "Failed to parse IR: " << Err.getMessage();

  Function *F = M->getFunction("main");
  auto *Entry = &F->getEntryBlock();
  auto *Left = Entry->getTerminator()->getSuccessor(0);
  auto *Right = Entry->getTerminator()->getSuccessor(1);
  auto *Exit = Left->getSingleSuccessor();
  auto *Val = &*F->arg_begin();
  auto *Phi = &Exit->front();
  auto *Cmp = &*std::next(Exit->begin());
  auto *Add = &Left->front();
  auto *Sub = &Right->front();

  SSAUpdaterBulk Updater;
  Type *I32Ty = Type::getInt32Ty(Context);

  // Use %val directly instead of creating a phi.
  unsigned ValVar = Updater.AddVariable("Val", I32Ty);
  Updater.AddAvailableValue(ValVar, Left, Val);
  Updater.AddAvailableValue(ValVar, Right, Val);
  Updater.AddUse(ValVar, &Cmp->getOperandUse(0));

  // Use existing %phi for %add and %sub values.
  unsigned AddSubVar = Updater.AddVariable("AddSub", I32Ty);
  Updater.AddAvailableValue(AddSubVar, Left, Add);
  Updater.AddAvailableValue(AddSubVar, Right, Sub);
  Updater.AddUse(AddSubVar, &Cmp->getOperandUse(1));

  auto ExitSizeBefore = Exit->size();
  DominatorTree DT(*F);
  Updater.RewriteAndOptimizeAllUses(DT);

  //  Output for Exit->dump():
  //  exit:                                             ; preds = %right, %left
  //    %phi = phi i32 [ %sub, %right ], [ %add, %left ]
  //    %cmp = icmp slt i32 %val, %phi
  //    ret void

  ASSERT_EQ(Exit->size(), ExitSizeBefore);
  ASSERT_EQ(&Exit->front(), Phi);
  EXPECT_EQ(Val, Cmp->getOperand(0));
  EXPECT_EQ(Phi, Cmp->getOperand(1));
}

bool EliminateNewDuplicatePHINodes(BasicBlock *BB,
                                   BasicBlock::phi_iterator FirstExistingPN);

// Helper to run both versions on the same input.
static void RunEliminateNewDuplicatePHINode(
    const char *AsmText,
    std::function<void(BasicBlock &,
                       bool(BasicBlock *BB, BasicBlock::phi_iterator))>
        Check) {
  LLVMContext C;

  SMDiagnostic Err;
  std::unique_ptr<Module> M = parseAssemblyString(AsmText, Err, C);
  if (!M) {
    Err.print("UtilsTests", errs());
    return;
  }

  Function *F = M->getFunction("main");
  auto BBIt = std::find_if(F->begin(), F->end(), [](const BasicBlock &Block) {
    return Block.getName() == "testbb";
  });
  ASSERT_NE(BBIt, F->end());
  Check(*BBIt, EliminateNewDuplicatePHINodes);
}

static BasicBlock::phi_iterator getPhiIt(BasicBlock &BB, unsigned Idx) {
  return std::next(BB.phis().begin(), Idx);
}

static PHINode *getPhi(BasicBlock &BB, unsigned Idx) {
  return &*getPhiIt(BB, Idx);
}

static int getNumPHIs(BasicBlock &BB) {
  return std::distance(BB.phis().begin(), BB.phis().end());
}

TEST(SSAUpdaterBulk, EliminateNewDuplicatePHINodes_OrderExisting) {
  RunEliminateNewDuplicatePHINode(R"(
      define void @main() {
      entry:
          br label %testbb
      testbb:
          %np0 = phi i32 [ 1, %entry ]
          %np1 = phi i32 [ 1, %entry ]
          %ep0 = phi i32 [ 1, %entry ]
          %ep1 = phi i32 [ 1, %entry ]
          %u = add i32 %np0, %np1
          ret void
      }
  )", [](BasicBlock &BB, auto *ENDPN) {
    AssertingVH<PHINode> EP0 = getPhi(BB, 2);
    AssertingVH<PHINode> EP1 = getPhi(BB, 3);
    EXPECT_TRUE(ENDPN(&BB, getPhiIt(BB, 2)));
    // Expected:
    //   %ep0 = phi i32 [ 1, %entry ]
    //   %ep1 = phi i32 [ 1, %entry ]
    //   %u = add i32 %ep0, %ep0
    EXPECT_EQ(getNumPHIs(BB), 2);
    Instruction &Add = *BB.getFirstNonPHIIt();
    EXPECT_EQ(Add.getOperand(0), EP0);
    EXPECT_EQ(Add.getOperand(1), EP0);
    (void)EP1; // Avoid "unused" warning.
  });
}

TEST(SSAUpdaterBulk, EliminateNewDuplicatePHINodes_OrderNew) {
  RunEliminateNewDuplicatePHINode(R"(
      define void @main() {
      entry:
          br label %testbb
      testbb:
          %np0 = phi i32 [ 1, %entry ]
          %np1 = phi i32 [ 1, %entry ]
          %ep0 = phi i32 [ 2, %entry ]
          %ep1 = phi i32 [ 2, %entry ]
          %u = add i32 %np0, %np1
          ret void
      }
  )", [](BasicBlock &BB, auto *ENDPN) {
    AssertingVH<PHINode> NP0 = getPhi(BB, 0);
    AssertingVH<PHINode> EP0 = getPhi(BB, 2);
    AssertingVH<PHINode> EP1 = getPhi(BB, 3);
    EXPECT_TRUE(ENDPN(&BB, getPhiIt(BB, 2)));
    // Expected:
    //   %np0 = phi i32 [ 1, %entry ]
    //   %ep0 = phi i32 [ 2, %entry ]
    //   %ep1 = phi i32 [ 2, %entry ]
    //   %u = add i32 %np0, %np0
    EXPECT_EQ(getNumPHIs(BB), 3);
    Instruction &Add = *BB.getFirstNonPHIIt();
    EXPECT_EQ(Add.getOperand(0), NP0);
    EXPECT_EQ(Add.getOperand(1), NP0);
    (void)EP0;
    (void)EP1; // Avoid "unused" warning.
  });
}

TEST(SSAUpdaterBulk, EliminateNewDuplicatePHINodes_NewRefExisting) {
  RunEliminateNewDuplicatePHINode(R"(
      define void @main() {
      entry:
          br label %testbb
      testbb:
          %np0 = phi i32 [ 1, %entry ], [ %ep0, %testbb ]
          %np1 = phi i32 [ 1, %entry ], [ %ep1, %testbb ]
          %ep0 = phi i32 [ 1, %entry ], [ %ep0, %testbb ]
          %ep1 = phi i32 [ 1, %entry ], [ %ep1, %testbb ]
          %u = add i32 %np0, %np1
          br label %testbb
      }
  )", [](BasicBlock &BB, auto *ENDPN) {
    AssertingVH<PHINode> EP0 = getPhi(BB, 2);
    AssertingVH<PHINode> EP1 = getPhi(BB, 3);
    EXPECT_TRUE(ENDPN(&BB, getPhiIt(BB, 2)));
    // Expected:
    //   %ep0 = phi i32 [ 1, %entry ], [ %ep0, %testbb ]
    //   %ep1 = phi i32 [ 1, %entry ], [ %ep1, %testbb ]
    //   %u = add i32 %ep0, %ep1
    EXPECT_EQ(getNumPHIs(BB), 2);
    Instruction &Add = *BB.getFirstNonPHIIt();
    EXPECT_EQ(Add.getOperand(0), EP0);
    EXPECT_EQ(Add.getOperand(1), EP1);
  });
}

TEST(SSAUpdaterBulk, EliminateNewDuplicatePHINodes_ExistingRefNew) {
  RunEliminateNewDuplicatePHINode(R"(
      define void @main() {
      entry:
          br label %testbb
      testbb:
          %np0 = phi i32 [ 1, %entry ], [ %np0, %testbb ]
          %np1 = phi i32 [ 1, %entry ], [ %np1, %testbb ]
          %ep0 = phi i32 [ 1, %entry ], [ %np0, %testbb ]
          %ep1 = phi i32 [ 1, %entry ], [ %np1, %testbb ]
          %u = add i32 %np0, %np1
          br label %testbb
      }
  )", [](BasicBlock &BB, auto *ENDPN) {
    AssertingVH<PHINode> EP0 = getPhi(BB, 2);
    AssertingVH<PHINode> EP1 = getPhi(BB, 3);
    EXPECT_TRUE(ENDPN(&BB, getPhiIt(BB, 2)));
    // Expected:
    //   %ep0 = phi i32 [ 1, %entry ], [ %ep0, %testbb ]
    //   %ep1 = phi i32 [ 1, %entry ], [ %ep1, %testbb ]
    //   %u = add i32 %ep0, %ep1
    EXPECT_EQ(getNumPHIs(BB), 2);
    Instruction &Add = *BB.getFirstNonPHIIt();
    EXPECT_EQ(Add.getOperand(0), EP0);
    EXPECT_EQ(Add.getOperand(1), EP1);
  });
}
