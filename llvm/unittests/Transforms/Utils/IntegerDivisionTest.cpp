//===- IntegerDivision.cpp - Unit tests for the integer division code -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/IntegerDivision.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {


TEST(IntegerDivision, SDiv) {
  LLVMContext C;
  Module M("test division", C);
  IRBuilder<> Builder(C);

  SmallVector<Type*, 2> ArgTys(2, Builder.getInt32Ty());
  Function *F = Function::Create(FunctionType::get(Builder.getInt32Ty(),
                                                   ArgTys, false),
                                 GlobalValue::ExternalLinkage, "F", &M);
  assert(F->arg_size() == 2);

  BasicBlock *BB = BasicBlock::Create(C, "", F);
  Builder.SetInsertPoint(BB);

  Function::arg_iterator AI = F->arg_begin();
  Value *A = &*AI++;
  Value *B = &*AI++;

  Value *Div = Builder.CreateSDiv(A, B);
  EXPECT_TRUE(BB->front().getOpcode() == Instruction::SDiv);

  Value *Ret = Builder.CreateRet(Div);

  expandDivision(cast<BinaryOperator>(Div));
  EXPECT_TRUE(BB->front().getOpcode() == Instruction::Freeze);

  Instruction* Quotient = dyn_cast<Instruction>(cast<User>(Ret)->getOperand(0));
  EXPECT_TRUE(Quotient && Quotient->getOpcode() == Instruction::Sub);
}

TEST(IntegerDivision, UDiv) {
  LLVMContext C;
  Module M("test division", C);
  IRBuilder<> Builder(C);

  SmallVector<Type*, 2> ArgTys(2, Builder.getInt32Ty());
  Function *F = Function::Create(FunctionType::get(Builder.getInt32Ty(),
                                                   ArgTys, false),
                                 GlobalValue::ExternalLinkage, "F", &M);
  assert(F->arg_size() == 2);

  BasicBlock *BB = BasicBlock::Create(C, "", F);
  Builder.SetInsertPoint(BB);

  Function::arg_iterator AI = F->arg_begin();
  Value *A = &*AI++;
  Value *B = &*AI++;

  Value *Div = Builder.CreateUDiv(A, B);
  EXPECT_TRUE(BB->front().getOpcode() == Instruction::UDiv);

  Value *Ret = Builder.CreateRet(Div);

  expandDivision(cast<BinaryOperator>(Div));
  EXPECT_TRUE(BB->front().getOpcode() == Instruction::Freeze);

  Instruction* Quotient = dyn_cast<Instruction>(cast<User>(Ret)->getOperand(0));
  EXPECT_TRUE(Quotient && Quotient->getOpcode() == Instruction::PHI);
}

TEST(IntegerDivision, SRem) {
  LLVMContext C;
  Module M("test remainder", C);
  IRBuilder<> Builder(C);

  SmallVector<Type*, 2> ArgTys(2, Builder.getInt32Ty());
  Function *F = Function::Create(FunctionType::get(Builder.getInt32Ty(),
                                                   ArgTys, false),
                                 GlobalValue::ExternalLinkage, "F", &M);
  assert(F->arg_size() == 2);

  BasicBlock *BB = BasicBlock::Create(C, "", F);
  Builder.SetInsertPoint(BB);

  Function::arg_iterator AI = F->arg_begin();
  Value *A = &*AI++;
  Value *B = &*AI++;

  Value *Rem = Builder.CreateSRem(A, B);
  EXPECT_TRUE(BB->front().getOpcode() == Instruction::SRem);

  Value *Ret = Builder.CreateRet(Rem);

  expandRemainder(cast<BinaryOperator>(Rem));
  EXPECT_TRUE(BB->front().getOpcode() == Instruction::Freeze);

  Instruction* Remainder = dyn_cast<Instruction>(cast<User>(Ret)->getOperand(0));
  EXPECT_TRUE(Remainder && Remainder->getOpcode() == Instruction::Sub);
}

TEST(IntegerDivision, URem) {
  LLVMContext C;
  Module M("test remainder", C);
  IRBuilder<> Builder(C);

  SmallVector<Type*, 2> ArgTys(2, Builder.getInt32Ty());
  Function *F = Function::Create(FunctionType::get(Builder.getInt32Ty(),
                                                   ArgTys, false),
                                 GlobalValue::ExternalLinkage, "F", &M);
  assert(F->arg_size() == 2);

  BasicBlock *BB = BasicBlock::Create(C, "", F);
  Builder.SetInsertPoint(BB);

  Function::arg_iterator AI = F->arg_begin();
  Value *A = &*AI++;
  Value *B = &*AI++;

  Value *Rem = Builder.CreateURem(A, B);
  EXPECT_TRUE(BB->front().getOpcode() == Instruction::URem);

  Value *Ret = Builder.CreateRet(Rem);

  expandRemainder(cast<BinaryOperator>(Rem));
  EXPECT_TRUE(BB->front().getOpcode() == Instruction::Freeze);

  Instruction* Remainder = dyn_cast<Instruction>(cast<User>(Ret)->getOperand(0));
  EXPECT_TRUE(Remainder && Remainder->getOpcode() == Instruction::Sub);
}


TEST(IntegerDivision, SDiv64) {
  LLVMContext C;
  Module M("test division", C);
  IRBuilder<> Builder(C);

  SmallVector<Type*, 2> ArgTys(2, Builder.getInt64Ty());
  Function *F = Function::Create(FunctionType::get(Builder.getInt64Ty(),
                                                   ArgTys, false),
                                 GlobalValue::ExternalLinkage, "F", &M);
  assert(F->arg_size() == 2);

  BasicBlock *BB = BasicBlock::Create(C, "", F);
  Builder.SetInsertPoint(BB);

  Function::arg_iterator AI = F->arg_begin();
  Value *A = &*AI++;
  Value *B = &*AI++;

  Value *Div = Builder.CreateSDiv(A, B);
  EXPECT_TRUE(BB->front().getOpcode() == Instruction::SDiv);

  Value *Ret = Builder.CreateRet(Div);

  expandDivision(cast<BinaryOperator>(Div));
  EXPECT_TRUE(BB->front().getOpcode() == Instruction::Freeze);

  Instruction* Quotient = dyn_cast<Instruction>(cast<User>(Ret)->getOperand(0));
  EXPECT_TRUE(Quotient && Quotient->getOpcode() == Instruction::Sub);
}

TEST(IntegerDivision, UDiv64) {
  LLVMContext C;
  Module M("test division", C);
  IRBuilder<> Builder(C);

  SmallVector<Type*, 2> ArgTys(2, Builder.getInt64Ty());
  Function *F = Function::Create(FunctionType::get(Builder.getInt64Ty(),
                                                   ArgTys, false),
                                 GlobalValue::ExternalLinkage, "F", &M);
  assert(F->arg_size() == 2);

  BasicBlock *BB = BasicBlock::Create(C, "", F);
  Builder.SetInsertPoint(BB);

  Function::arg_iterator AI = F->arg_begin();
  Value *A = &*AI++;
  Value *B = &*AI++;

  Value *Div = Builder.CreateUDiv(A, B);
  EXPECT_TRUE(BB->front().getOpcode() == Instruction::UDiv);

  Value *Ret = Builder.CreateRet(Div);

  expandDivision(cast<BinaryOperator>(Div));
  EXPECT_TRUE(BB->front().getOpcode() == Instruction::Freeze);

  Instruction* Quotient = dyn_cast<Instruction>(cast<User>(Ret)->getOperand(0));
  EXPECT_TRUE(Quotient && Quotient->getOpcode() == Instruction::PHI);
}

TEST(IntegerDivision, SRem64) {
  LLVMContext C;
  Module M("test remainder", C);
  IRBuilder<> Builder(C);

  SmallVector<Type*, 2> ArgTys(2, Builder.getInt64Ty());
  Function *F = Function::Create(FunctionType::get(Builder.getInt64Ty(),
                                                   ArgTys, false),
                                 GlobalValue::ExternalLinkage, "F", &M);
  assert(F->arg_size() == 2);

  BasicBlock *BB = BasicBlock::Create(C, "", F);
  Builder.SetInsertPoint(BB);

  Function::arg_iterator AI = F->arg_begin();
  Value *A = &*AI++;
  Value *B = &*AI++;

  Value *Rem = Builder.CreateSRem(A, B);
  EXPECT_TRUE(BB->front().getOpcode() == Instruction::SRem);

  Value *Ret = Builder.CreateRet(Rem);

  expandRemainder(cast<BinaryOperator>(Rem));
  EXPECT_TRUE(BB->front().getOpcode() == Instruction::Freeze);

  Instruction* Remainder = dyn_cast<Instruction>(cast<User>(Ret)->getOperand(0));
  EXPECT_TRUE(Remainder && Remainder->getOpcode() == Instruction::Sub);
}

TEST(IntegerDivision, URem64) {
  LLVMContext C;
  Module M("test remainder", C);
  IRBuilder<> Builder(C);

  SmallVector<Type*, 2> ArgTys(2, Builder.getInt64Ty());
  Function *F = Function::Create(FunctionType::get(Builder.getInt64Ty(),
                                                   ArgTys, false),
                                 GlobalValue::ExternalLinkage, "F", &M);
  assert(F->arg_size() == 2);

  BasicBlock *BB = BasicBlock::Create(C, "", F);
  Builder.SetInsertPoint(BB);

  Function::arg_iterator AI = F->arg_begin();
  Value *A = &*AI++;
  Value *B = &*AI++;

  Value *Rem = Builder.CreateURem(A, B);
  EXPECT_TRUE(BB->front().getOpcode() == Instruction::URem);

  Value *Ret = Builder.CreateRet(Rem);

  expandRemainder(cast<BinaryOperator>(Rem));
  EXPECT_TRUE(BB->front().getOpcode() == Instruction::Freeze);

  Instruction* Remainder = dyn_cast<Instruction>(cast<User>(Ret)->getOperand(0));
  EXPECT_TRUE(Remainder && Remainder->getOpcode() == Instruction::Sub);
}

}
