//===- IntegerDivision.cpp - Unit tests for the integer division code -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "llvm/BasicBlock.h"
#include "llvm/GlobalValue.h"
#include "llvm/Function.h"
#include "llvm/IRBuilder.h"
#include "llvm/Module.h"
#include "llvm/Transforms/Utils/IntegerDivision.h"

using namespace llvm;

namespace {

TEST(IntegerDivision, SDiv) {
  LLVMContext &C(getGlobalContext());
  Module M("test division", C);
  IRBuilder<> Builder(C);

  SmallVector<Type*, 2> ArgTys(2, Builder.getInt32Ty());
  Function *F = Function::Create(FunctionType::get(Builder.getInt32Ty(),
                                                   ArgTys, false),
                                 GlobalValue::ExternalLinkage, "F", &M);
  assert(F->getArgumentList().size() == 2);

  BasicBlock *BB = BasicBlock::Create(C, "", F);
  Builder.SetInsertPoint(BB);

  Function::arg_iterator AI = F->arg_begin();
  Value *A = AI++;
  Value *B = AI++;

  Value *Div = Builder.CreateSDiv(A, B);
  EXPECT_TRUE(BB->front().getOpcode() == Instruction::SDiv);

  Value *Ret = Builder.CreateRet(Div);

  expandDivision(cast<BinaryOperator>(Div));
  EXPECT_TRUE(BB->front().getOpcode() == Instruction::AShr);

  Instruction* Quotient = dyn_cast<Instruction>(cast<User>(Ret)->getOperand(0));
  EXPECT_TRUE(Quotient && Quotient->getOpcode() == Instruction::Sub);
}

TEST(IntegerDivision, UDiv) {
  LLVMContext &C(getGlobalContext());
  Module M("test division", C);
  IRBuilder<> Builder(C);

  SmallVector<Type*, 2> ArgTys(2, Builder.getInt32Ty());
  Function *F = Function::Create(FunctionType::get(Builder.getInt32Ty(),
                                                   ArgTys, false),
                                 GlobalValue::ExternalLinkage, "F", &M);
  assert(F->getArgumentList().size() == 2);

  BasicBlock *BB = BasicBlock::Create(C, "", F);
  Builder.SetInsertPoint(BB);

  Function::arg_iterator AI = F->arg_begin();
  Value *A = AI++;
  Value *B = AI++;

  Value *Div = Builder.CreateUDiv(A, B);
  EXPECT_TRUE(BB->front().getOpcode() == Instruction::UDiv);

  Value *Ret = Builder.CreateRet(Div);

  expandDivision(cast<BinaryOperator>(Div));
  EXPECT_TRUE(BB->front().getOpcode() == Instruction::ICmp);

  Instruction* Quotient = dyn_cast<Instruction>(cast<User>(Ret)->getOperand(0));
  EXPECT_TRUE(Quotient && Quotient->getOpcode() == Instruction::PHI);
}

TEST(IntegerDivision, SRem) {
  LLVMContext &C(getGlobalContext());
  Module M("test remainder", C);
  IRBuilder<> Builder(C);

  SmallVector<Type*, 2> ArgTys(2, Builder.getInt32Ty());
  Function *F = Function::Create(FunctionType::get(Builder.getInt32Ty(),
                                                   ArgTys, false),
                                 GlobalValue::ExternalLinkage, "F", &M);
  assert(F->getArgumentList().size() == 2);

  BasicBlock *BB = BasicBlock::Create(C, "", F);
  Builder.SetInsertPoint(BB);

  Function::arg_iterator AI = F->arg_begin();
  Value *A = AI++;
  Value *B = AI++;

  Value *Rem = Builder.CreateSRem(A, B);
  EXPECT_TRUE(BB->front().getOpcode() == Instruction::SRem);

  Value *Ret = Builder.CreateRet(Rem);

  expandRemainder(cast<BinaryOperator>(Rem));
  EXPECT_TRUE(BB->front().getOpcode() == Instruction::AShr);

  Instruction* Remainder = dyn_cast<Instruction>(cast<User>(Ret)->getOperand(0));
  EXPECT_TRUE(Remainder && Remainder->getOpcode() == Instruction::Sub);
}

TEST(IntegerDivision, URem) {
  LLVMContext &C(getGlobalContext());
  Module M("test remainder", C);
  IRBuilder<> Builder(C);

  SmallVector<Type*, 2> ArgTys(2, Builder.getInt32Ty());
  Function *F = Function::Create(FunctionType::get(Builder.getInt32Ty(),
                                                   ArgTys, false),
                                 GlobalValue::ExternalLinkage, "F", &M);
  assert(F->getArgumentList().size() == 2);

  BasicBlock *BB = BasicBlock::Create(C, "", F);
  Builder.SetInsertPoint(BB);

  Function::arg_iterator AI = F->arg_begin();
  Value *A = AI++;
  Value *B = AI++;

  Value *Rem = Builder.CreateURem(A, B);
  EXPECT_TRUE(BB->front().getOpcode() == Instruction::URem);

  Value *Ret = Builder.CreateRet(Rem);

  expandRemainder(cast<BinaryOperator>(Rem));
  EXPECT_TRUE(BB->front().getOpcode() == Instruction::ICmp);

  Instruction* Remainder = dyn_cast<Instruction>(cast<User>(Ret)->getOperand(0));
  EXPECT_TRUE(Remainder && Remainder->getOpcode() == Instruction::Sub);
}

}
