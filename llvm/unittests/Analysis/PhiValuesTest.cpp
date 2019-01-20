//===- PhiValuesTest.cpp - PhiValues unit tests ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/PhiValues.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "gtest/gtest.h"

using namespace llvm;

TEST(PhiValuesTest, SimplePhi) {
  LLVMContext C;
  Module M("PhiValuesTest", C);

  Type *VoidTy = Type::getVoidTy(C);
  Type *I1Ty = Type::getInt1Ty(C);
  Type *I32Ty = Type::getInt32Ty(C);
  Type *I32PtrTy = Type::getInt32PtrTy(C);

  // Create a function with phis that do not have other phis as incoming values
  Function *F = cast<Function>(M.getOrInsertFunction("f", FunctionType::get(VoidTy, false)));

  BasicBlock *Entry = BasicBlock::Create(C, "entry", F);
  BasicBlock *If = BasicBlock::Create(C, "if", F);
  BasicBlock *Else = BasicBlock::Create(C, "else", F);
  BasicBlock *Then = BasicBlock::Create(C, "then", F);
  BranchInst::Create(If, Else, UndefValue::get(I1Ty), Entry);
  BranchInst::Create(Then, If);
  BranchInst::Create(Then, Else);

  Value *Val1 = new LoadInst(UndefValue::get(I32PtrTy), "val1", Entry);
  Value *Val2 = new LoadInst(UndefValue::get(I32PtrTy), "val2", Entry);
  Value *Val3 = new LoadInst(UndefValue::get(I32PtrTy), "val3", Entry);
  Value *Val4 = new LoadInst(UndefValue::get(I32PtrTy), "val4", Entry);

  PHINode *Phi1 = PHINode::Create(I32Ty, 2, "phi1", Then);
  Phi1->addIncoming(Val1, If);
  Phi1->addIncoming(Val2, Else);
  PHINode *Phi2 = PHINode::Create(I32Ty, 2, "phi2", Then);
  Phi2->addIncoming(Val1, If);
  Phi2->addIncoming(Val3, Else);

  PhiValues PV(*F);
  PhiValues::ValueSet Vals;

  // Check that simple usage works
  Vals = PV.getValuesForPhi(Phi1);
  EXPECT_EQ(Vals.size(), 2u);
  EXPECT_TRUE(Vals.count(Val1));
  EXPECT_TRUE(Vals.count(Val2));
  Vals = PV.getValuesForPhi(Phi2);
  EXPECT_EQ(Vals.size(), 2u);
  EXPECT_TRUE(Vals.count(Val1));
  EXPECT_TRUE(Vals.count(Val3));

  // Check that values are updated when one value is replaced with another
  Val1->replaceAllUsesWith(Val4);
  PV.invalidateValue(Val1);
  Vals = PV.getValuesForPhi(Phi1);
  EXPECT_EQ(Vals.size(), 2u);
  EXPECT_TRUE(Vals.count(Val4));
  EXPECT_TRUE(Vals.count(Val2));
  Vals = PV.getValuesForPhi(Phi2);
  EXPECT_EQ(Vals.size(), 2u);
  EXPECT_TRUE(Vals.count(Val4));
  EXPECT_TRUE(Vals.count(Val3));

  // Check that setting in incoming value directly updates the values
  Phi1->setIncomingValue(0, Val1);
  PV.invalidateValue(Phi1);
  Vals = PV.getValuesForPhi(Phi1);
  EXPECT_EQ(Vals.size(), 2u);
  EXPECT_TRUE(Vals.count(Val1));
  EXPECT_TRUE(Vals.count(Val2));
}

TEST(PhiValuesTest, DependentPhi) {
  LLVMContext C;
  Module M("PhiValuesTest", C);

  Type *VoidTy = Type::getVoidTy(C);
  Type *I1Ty = Type::getInt1Ty(C);
  Type *I32Ty = Type::getInt32Ty(C);
  Type *I32PtrTy = Type::getInt32PtrTy(C);

  // Create a function with a phi that has another phi as an incoming value
  Function *F = cast<Function>(M.getOrInsertFunction("f", FunctionType::get(VoidTy, false)));

  BasicBlock *Entry = BasicBlock::Create(C, "entry", F);
  BasicBlock *If1 = BasicBlock::Create(C, "if1", F);
  BasicBlock *Else1 = BasicBlock::Create(C, "else1", F);
  BasicBlock *Then = BasicBlock::Create(C, "then", F);
  BasicBlock *If2 = BasicBlock::Create(C, "if2", F);
  BasicBlock *Else2 = BasicBlock::Create(C, "else2", F);
  BasicBlock *End = BasicBlock::Create(C, "then", F);
  BranchInst::Create(If1, Else1, UndefValue::get(I1Ty), Entry);
  BranchInst::Create(Then, If1);
  BranchInst::Create(Then, Else1);
  BranchInst::Create(If2, Else2, UndefValue::get(I1Ty), Then);
  BranchInst::Create(End, If2);
  BranchInst::Create(End, Else2);

  Value *Val1 = new LoadInst(UndefValue::get(I32PtrTy), "val1", Entry);
  Value *Val2 = new LoadInst(UndefValue::get(I32PtrTy), "val2", Entry);
  Value *Val3 = new LoadInst(UndefValue::get(I32PtrTy), "val3", Entry);
  Value *Val4 = new LoadInst(UndefValue::get(I32PtrTy), "val4", Entry);

  PHINode *Phi1 = PHINode::Create(I32Ty, 2, "phi1", Then);
  Phi1->addIncoming(Val1, If1);
  Phi1->addIncoming(Val2, Else1);
  PHINode *Phi2 = PHINode::Create(I32Ty, 2, "phi2", Then);
  Phi2->addIncoming(Val2, If1);
  Phi2->addIncoming(Val3, Else1);
  PHINode *Phi3 = PHINode::Create(I32Ty, 2, "phi3", End);
  Phi3->addIncoming(Phi1, If2);
  Phi3->addIncoming(Val3, Else2);

  PhiValues PV(*F);
  PhiValues::ValueSet Vals;

  // Check that simple usage works
  Vals = PV.getValuesForPhi(Phi1);
  EXPECT_EQ(Vals.size(), 2u);
  EXPECT_TRUE(Vals.count(Val1));
  EXPECT_TRUE(Vals.count(Val2));
  Vals = PV.getValuesForPhi(Phi2);
  EXPECT_EQ(Vals.size(), 2u);
  EXPECT_TRUE(Vals.count(Val2));
  EXPECT_TRUE(Vals.count(Val3));
  Vals = PV.getValuesForPhi(Phi3);
  EXPECT_EQ(Vals.size(), 3u);
  EXPECT_TRUE(Vals.count(Val1));
  EXPECT_TRUE(Vals.count(Val2));
  EXPECT_TRUE(Vals.count(Val3));

  // Check that changing an incoming value in the dependent phi changes the depending phi
  Phi1->setIncomingValue(0, Val4);
  PV.invalidateValue(Phi1);
  Vals = PV.getValuesForPhi(Phi1);
  EXPECT_EQ(Vals.size(), 2u);
  EXPECT_TRUE(Vals.count(Val4));
  EXPECT_TRUE(Vals.count(Val2));
  Vals = PV.getValuesForPhi(Phi2);
  EXPECT_EQ(Vals.size(), 2u);
  EXPECT_TRUE(Vals.count(Val2));
  EXPECT_TRUE(Vals.count(Val3));
  Vals = PV.getValuesForPhi(Phi3);
  EXPECT_EQ(Vals.size(), 3u);
  EXPECT_TRUE(Vals.count(Val4));
  EXPECT_TRUE(Vals.count(Val2));
  EXPECT_TRUE(Vals.count(Val3));

  // Check that replacing an incoming phi with a value works
  Phi3->setIncomingValue(0, Val1);
  PV.invalidateValue(Phi3);
  Vals = PV.getValuesForPhi(Phi1);
  EXPECT_EQ(Vals.size(), 2u);
  EXPECT_TRUE(Vals.count(Val4));
  EXPECT_TRUE(Vals.count(Val2));
  Vals = PV.getValuesForPhi(Phi2);
  EXPECT_EQ(Vals.size(), 2u);
  EXPECT_TRUE(Vals.count(Val2));
  EXPECT_TRUE(Vals.count(Val3));
  Vals = PV.getValuesForPhi(Phi3);
  EXPECT_EQ(Vals.size(), 2u);
  EXPECT_TRUE(Vals.count(Val1));
  EXPECT_TRUE(Vals.count(Val3));

  // Check that adding a phi as an incoming value works
  Phi3->setIncomingValue(1, Phi2);
  PV.invalidateValue(Phi3);
  Vals = PV.getValuesForPhi(Phi1);
  EXPECT_EQ(Vals.size(), 2u);
  EXPECT_TRUE(Vals.count(Val4));
  EXPECT_TRUE(Vals.count(Val2));
  Vals = PV.getValuesForPhi(Phi2);
  EXPECT_EQ(Vals.size(), 2u);
  EXPECT_TRUE(Vals.count(Val2));
  EXPECT_TRUE(Vals.count(Val3));
  Vals = PV.getValuesForPhi(Phi3);
  EXPECT_EQ(Vals.size(), 3u);
  EXPECT_TRUE(Vals.count(Val1));
  EXPECT_TRUE(Vals.count(Val2));
  EXPECT_TRUE(Vals.count(Val3));

  // Check that replacing an incoming phi then deleting it works
  Phi3->setIncomingValue(1, Val2);
  PV.invalidateValue(Phi2);
  Phi2->eraseFromParent();
  PV.invalidateValue(Phi3);
  Vals = PV.getValuesForPhi(Phi1);
  EXPECT_EQ(Vals.size(), 2u);
  EXPECT_TRUE(Vals.count(Val4));
  EXPECT_TRUE(Vals.count(Val2));
  Vals = PV.getValuesForPhi(Phi3);
  EXPECT_EQ(Vals.size(), 2u);
  EXPECT_TRUE(Vals.count(Val1));
  EXPECT_TRUE(Vals.count(Val2));
}
