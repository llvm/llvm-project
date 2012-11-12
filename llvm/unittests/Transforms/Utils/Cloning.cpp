//===- Cloning.cpp - Unit tests for the Cloner ----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "llvm/Argument.h"
#include "llvm/Constant.h"
#include "llvm/Instructions.h"
#include "llvm/LLVMContext.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/STLExtras.h"

using namespace llvm;

namespace {

class CloneInstruction : public ::testing::Test {
protected:
  virtual void SetUp() {
    V = NULL;
  }

  template <typename T>
  T *clone(T *V1) {
    Value *V2 = V1->clone();
    Orig.insert(V1);
    Clones.insert(V2);
    return cast<T>(V2);
  }

  void eraseClones() {
    DeleteContainerPointers(Clones);
  }

  virtual void TearDown() {
    eraseClones();
    DeleteContainerPointers(Orig);
    delete V;
  }

  SmallPtrSet<Value *, 4> Orig;   // Erase on exit
  SmallPtrSet<Value *, 4> Clones; // Erase in eraseClones

  LLVMContext context;
  Value *V;
};

TEST_F(CloneInstruction, OverflowBits) {
  V = new Argument(Type::getInt32Ty(context));

  BinaryOperator *Add = BinaryOperator::Create(Instruction::Add, V, V);
  BinaryOperator *Sub = BinaryOperator::Create(Instruction::Sub, V, V);
  BinaryOperator *Mul = BinaryOperator::Create(Instruction::Mul, V, V);

  BinaryOperator *AddClone = this->clone(Add);
  BinaryOperator *SubClone = this->clone(Sub);
  BinaryOperator *MulClone = this->clone(Mul);

  EXPECT_FALSE(AddClone->hasNoUnsignedWrap());
  EXPECT_FALSE(AddClone->hasNoSignedWrap());
  EXPECT_FALSE(SubClone->hasNoUnsignedWrap());
  EXPECT_FALSE(SubClone->hasNoSignedWrap());
  EXPECT_FALSE(MulClone->hasNoUnsignedWrap());
  EXPECT_FALSE(MulClone->hasNoSignedWrap());

  eraseClones();

  Add->setHasNoUnsignedWrap();
  Sub->setHasNoUnsignedWrap();
  Mul->setHasNoUnsignedWrap();

  AddClone = this->clone(Add);
  SubClone = this->clone(Sub);
  MulClone = this->clone(Mul);

  EXPECT_TRUE(AddClone->hasNoUnsignedWrap());
  EXPECT_FALSE(AddClone->hasNoSignedWrap());
  EXPECT_TRUE(SubClone->hasNoUnsignedWrap());
  EXPECT_FALSE(SubClone->hasNoSignedWrap());
  EXPECT_TRUE(MulClone->hasNoUnsignedWrap());
  EXPECT_FALSE(MulClone->hasNoSignedWrap());

  eraseClones();

  Add->setHasNoSignedWrap();
  Sub->setHasNoSignedWrap();
  Mul->setHasNoSignedWrap();

  AddClone = this->clone(Add);
  SubClone = this->clone(Sub);
  MulClone = this->clone(Mul);

  EXPECT_TRUE(AddClone->hasNoUnsignedWrap());
  EXPECT_TRUE(AddClone->hasNoSignedWrap());
  EXPECT_TRUE(SubClone->hasNoUnsignedWrap());
  EXPECT_TRUE(SubClone->hasNoSignedWrap());
  EXPECT_TRUE(MulClone->hasNoUnsignedWrap());
  EXPECT_TRUE(MulClone->hasNoSignedWrap());

  eraseClones();

  Add->setHasNoUnsignedWrap(false);
  Sub->setHasNoUnsignedWrap(false);
  Mul->setHasNoUnsignedWrap(false);

  AddClone = this->clone(Add);
  SubClone = this->clone(Sub);
  MulClone = this->clone(Mul);

  EXPECT_FALSE(AddClone->hasNoUnsignedWrap());
  EXPECT_TRUE(AddClone->hasNoSignedWrap());
  EXPECT_FALSE(SubClone->hasNoUnsignedWrap());
  EXPECT_TRUE(SubClone->hasNoSignedWrap());
  EXPECT_FALSE(MulClone->hasNoUnsignedWrap());
  EXPECT_TRUE(MulClone->hasNoSignedWrap());
}

TEST_F(CloneInstruction, Inbounds) {
  V = new Argument(Type::getInt32PtrTy(context));

  Constant *Z = Constant::getNullValue(Type::getInt32Ty(context));
  std::vector<Value *> ops;
  ops.push_back(Z);
  GetElementPtrInst *GEP = GetElementPtrInst::Create(V, ops);
  EXPECT_FALSE(this->clone(GEP)->isInBounds());

  GEP->setIsInBounds();
  EXPECT_TRUE(this->clone(GEP)->isInBounds());
}

TEST_F(CloneInstruction, Exact) {
  V = new Argument(Type::getInt32Ty(context));

  BinaryOperator *SDiv = BinaryOperator::Create(Instruction::SDiv, V, V);
  EXPECT_FALSE(this->clone(SDiv)->isExact());

  SDiv->setIsExact(true);
  EXPECT_TRUE(this->clone(SDiv)->isExact());
}

}
