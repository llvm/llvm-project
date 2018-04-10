//===- SSAUpdaterBulk.cpp - Unit tests for SSAUpdaterBulk -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
  Updater.AddVariable(0, "a", I32Ty);
  Updater.AddAvailableValue(0, TrueBB, AddOp1);
  Updater.AddAvailableValue(0, FalseBB, AddOp2);
  Updater.AddUse(0, &I1->getOperandUse(0));
  Updater.AddUse(0, &I2->getOperandUse(0));

  Updater.AddVariable(1, "b", I32Ty);
  Updater.AddAvailableValue(1, TrueBB, SubOp1);
  Updater.AddAvailableValue(1, FalseBB, SubOp2);
  Updater.AddUse(1, &I3->getOperandUse(0));
  Updater.AddUse(1, &I3->getOperandUse(1));

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
  Updater.AddVariable(0, "c", I32Ty);
  Updater.AddAvailableValue(0, IfBB, AddOp1);
  Updater.AddUse(0, &I1->getOperandUse(0));

  // Add use in %4.
  Updater.AddVariable(1, "b", I32Ty);
  Updater.AddAvailableValue(1, LoopStartBB, AddOp2);
  Updater.AddUse(1, &I2->getOperandUse(0));

  // Add use in the return instruction.
  Updater.AddVariable(2, "a", I32Ty);
  Updater.AddAvailableValue(2, &F->getEntryBlock(), FirstArg);
  Updater.AddUse(2, &Return->getOperandUse(0));

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
