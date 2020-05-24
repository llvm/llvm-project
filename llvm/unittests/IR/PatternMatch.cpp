//===---- llvm/unittest/IR/PatternMatch.cpp - PatternMatch unit tests ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/PatternMatch.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/NoFolder.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/Type.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::PatternMatch;

namespace {

struct PatternMatchTest : ::testing::Test {
  LLVMContext Ctx;
  std::unique_ptr<Module> M;
  Function *F;
  BasicBlock *BB;
  IRBuilder<NoFolder> IRB;

  PatternMatchTest()
      : M(new Module("PatternMatchTestModule", Ctx)),
        F(Function::Create(
            FunctionType::get(Type::getVoidTy(Ctx), /* IsVarArg */ false),
            Function::ExternalLinkage, "f", M.get())),
        BB(BasicBlock::Create(Ctx, "entry", F)), IRB(BB) {}
};

TEST_F(PatternMatchTest, OneUse) {
  // Build up a little tree of values:
  //
  //   One  = (1 + 2) + 42
  //   Two  = One + 42
  //   Leaf = (Two + 8) + (Two + 13)
  Value *One = IRB.CreateAdd(IRB.CreateAdd(IRB.getInt32(1), IRB.getInt32(2)),
                             IRB.getInt32(42));
  Value *Two = IRB.CreateAdd(One, IRB.getInt32(42));
  Value *Leaf = IRB.CreateAdd(IRB.CreateAdd(Two, IRB.getInt32(8)),
                              IRB.CreateAdd(Two, IRB.getInt32(13)));
  Value *V;

  EXPECT_TRUE(m_OneUse(m_Value(V)).match(One));
  EXPECT_EQ(One, V);

  EXPECT_FALSE(m_OneUse(m_Value()).match(Two));
  EXPECT_FALSE(m_OneUse(m_Value()).match(Leaf));
}

TEST_F(PatternMatchTest, SpecificIntEQ) {
  Type *IntTy = IRB.getInt32Ty();
  unsigned BitWidth = IntTy->getScalarSizeInBits();

  Value *Zero = ConstantInt::get(IntTy, 0);
  Value *One = ConstantInt::get(IntTy, 1);
  Value *NegOne = ConstantInt::get(IntTy, -1);

  EXPECT_TRUE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_EQ, APInt(BitWidth, 0))
          .match(Zero));
  EXPECT_FALSE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_EQ, APInt(BitWidth, 0))
          .match(One));
  EXPECT_FALSE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_EQ, APInt(BitWidth, 0))
          .match(NegOne));

  EXPECT_FALSE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_EQ, APInt(BitWidth, 1))
          .match(Zero));
  EXPECT_TRUE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_EQ, APInt(BitWidth, 1))
          .match(One));
  EXPECT_FALSE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_EQ, APInt(BitWidth, 1))
          .match(NegOne));

  EXPECT_FALSE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_EQ, APInt(BitWidth, -1))
          .match(Zero));
  EXPECT_FALSE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_EQ, APInt(BitWidth, -1))
          .match(One));
  EXPECT_TRUE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_EQ, APInt(BitWidth, -1))
          .match(NegOne));
}

TEST_F(PatternMatchTest, SpecificIntNE) {
  Type *IntTy = IRB.getInt32Ty();
  unsigned BitWidth = IntTy->getScalarSizeInBits();

  Value *Zero = ConstantInt::get(IntTy, 0);
  Value *One = ConstantInt::get(IntTy, 1);
  Value *NegOne = ConstantInt::get(IntTy, -1);

  EXPECT_FALSE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_NE, APInt(BitWidth, 0))
          .match(Zero));
  EXPECT_TRUE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_NE, APInt(BitWidth, 0))
          .match(One));
  EXPECT_TRUE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_NE, APInt(BitWidth, 0))
          .match(NegOne));

  EXPECT_TRUE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_NE, APInt(BitWidth, 1))
          .match(Zero));
  EXPECT_FALSE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_NE, APInt(BitWidth, 1))
          .match(One));
  EXPECT_TRUE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_NE, APInt(BitWidth, 1))
          .match(NegOne));

  EXPECT_TRUE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_NE, APInt(BitWidth, -1))
          .match(Zero));
  EXPECT_TRUE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_NE, APInt(BitWidth, -1))
          .match(One));
  EXPECT_FALSE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_NE, APInt(BitWidth, -1))
          .match(NegOne));
}

TEST_F(PatternMatchTest, SpecificIntUGT) {
  Type *IntTy = IRB.getInt32Ty();
  unsigned BitWidth = IntTy->getScalarSizeInBits();

  Value *Zero = ConstantInt::get(IntTy, 0);
  Value *One = ConstantInt::get(IntTy, 1);
  Value *NegOne = ConstantInt::get(IntTy, -1);

  EXPECT_FALSE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_UGT, APInt(BitWidth, 0))
          .match(Zero));
  EXPECT_TRUE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_UGT, APInt(BitWidth, 0))
          .match(One));
  EXPECT_TRUE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_UGT, APInt(BitWidth, 0))
          .match(NegOne));

  EXPECT_FALSE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_UGT, APInt(BitWidth, 1))
          .match(Zero));
  EXPECT_FALSE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_UGT, APInt(BitWidth, 1))
          .match(One));
  EXPECT_TRUE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_UGT, APInt(BitWidth, 1))
          .match(NegOne));

  EXPECT_FALSE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_UGT, APInt(BitWidth, -1))
          .match(Zero));
  EXPECT_FALSE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_UGT, APInt(BitWidth, -1))
          .match(One));
  EXPECT_FALSE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_UGT, APInt(BitWidth, -1))
          .match(NegOne));
}

TEST_F(PatternMatchTest, SignbitZeroChecks) {
  Type *IntTy = IRB.getInt32Ty();

  Value *Zero = ConstantInt::get(IntTy, 0);
  Value *One = ConstantInt::get(IntTy, 1);
  Value *NegOne = ConstantInt::get(IntTy, -1);

  EXPECT_TRUE(m_Negative().match(NegOne));
  EXPECT_FALSE(m_NonNegative().match(NegOne));
  EXPECT_FALSE(m_StrictlyPositive().match(NegOne));
  EXPECT_TRUE(m_NonPositive().match(NegOne));

  EXPECT_FALSE(m_Negative().match(Zero));
  EXPECT_TRUE(m_NonNegative().match(Zero));
  EXPECT_FALSE(m_StrictlyPositive().match(Zero));
  EXPECT_TRUE(m_NonPositive().match(Zero));

  EXPECT_FALSE(m_Negative().match(One));
  EXPECT_TRUE(m_NonNegative().match(One));
  EXPECT_TRUE(m_StrictlyPositive().match(One));
  EXPECT_FALSE(m_NonPositive().match(One));
}

TEST_F(PatternMatchTest, SpecificIntUGE) {
  Type *IntTy = IRB.getInt32Ty();
  unsigned BitWidth = IntTy->getScalarSizeInBits();

  Value *Zero = ConstantInt::get(IntTy, 0);
  Value *One = ConstantInt::get(IntTy, 1);
  Value *NegOne = ConstantInt::get(IntTy, -1);

  EXPECT_TRUE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_UGE, APInt(BitWidth, 0))
          .match(Zero));
  EXPECT_TRUE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_UGE, APInt(BitWidth, 0))
          .match(One));
  EXPECT_TRUE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_UGE, APInt(BitWidth, 0))
          .match(NegOne));

  EXPECT_FALSE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_UGE, APInt(BitWidth, 1))
          .match(Zero));
  EXPECT_TRUE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_UGE, APInt(BitWidth, 1))
          .match(One));
  EXPECT_TRUE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_UGE, APInt(BitWidth, 1))
          .match(NegOne));

  EXPECT_FALSE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_UGE, APInt(BitWidth, -1))
          .match(Zero));
  EXPECT_FALSE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_UGE, APInt(BitWidth, -1))
          .match(One));
  EXPECT_TRUE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_UGE, APInt(BitWidth, -1))
          .match(NegOne));
}

TEST_F(PatternMatchTest, SpecificIntULT) {
  Type *IntTy = IRB.getInt32Ty();
  unsigned BitWidth = IntTy->getScalarSizeInBits();

  Value *Zero = ConstantInt::get(IntTy, 0);
  Value *One = ConstantInt::get(IntTy, 1);
  Value *NegOne = ConstantInt::get(IntTy, -1);

  EXPECT_FALSE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_ULT, APInt(BitWidth, 0))
          .match(Zero));
  EXPECT_FALSE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_ULT, APInt(BitWidth, 0))
          .match(One));
  EXPECT_FALSE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_ULT, APInt(BitWidth, 0))
          .match(NegOne));

  EXPECT_TRUE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_ULT, APInt(BitWidth, 1))
          .match(Zero));
  EXPECT_FALSE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_ULT, APInt(BitWidth, 1))
          .match(One));
  EXPECT_FALSE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_ULT, APInt(BitWidth, 1))
          .match(NegOne));

  EXPECT_TRUE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_ULT, APInt(BitWidth, -1))
          .match(Zero));
  EXPECT_TRUE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_ULT, APInt(BitWidth, -1))
          .match(One));
  EXPECT_FALSE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_ULT, APInt(BitWidth, -1))
          .match(NegOne));
}

TEST_F(PatternMatchTest, SpecificIntULE) {
  Type *IntTy = IRB.getInt32Ty();
  unsigned BitWidth = IntTy->getScalarSizeInBits();

  Value *Zero = ConstantInt::get(IntTy, 0);
  Value *One = ConstantInt::get(IntTy, 1);
  Value *NegOne = ConstantInt::get(IntTy, -1);

  EXPECT_TRUE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_ULE, APInt(BitWidth, 0))
          .match(Zero));
  EXPECT_FALSE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_ULE, APInt(BitWidth, 0))
          .match(One));
  EXPECT_FALSE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_ULE, APInt(BitWidth, 0))
          .match(NegOne));

  EXPECT_TRUE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_ULE, APInt(BitWidth, 1))
          .match(Zero));
  EXPECT_TRUE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_ULE, APInt(BitWidth, 1))
          .match(One));
  EXPECT_FALSE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_ULE, APInt(BitWidth, 1))
          .match(NegOne));

  EXPECT_TRUE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_ULE, APInt(BitWidth, -1))
          .match(Zero));
  EXPECT_TRUE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_ULE, APInt(BitWidth, -1))
          .match(One));
  EXPECT_TRUE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_ULE, APInt(BitWidth, -1))
          .match(NegOne));
}

TEST_F(PatternMatchTest, SpecificIntSGT) {
  Type *IntTy = IRB.getInt32Ty();
  unsigned BitWidth = IntTy->getScalarSizeInBits();

  Value *Zero = ConstantInt::get(IntTy, 0);
  Value *One = ConstantInt::get(IntTy, 1);
  Value *NegOne = ConstantInt::get(IntTy, -1);

  EXPECT_FALSE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_SGT, APInt(BitWidth, 0))
          .match(Zero));
  EXPECT_TRUE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_SGT, APInt(BitWidth, 0))
          .match(One));
  EXPECT_FALSE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_SGT, APInt(BitWidth, 0))
          .match(NegOne));

  EXPECT_FALSE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_SGT, APInt(BitWidth, 1))
          .match(Zero));
  EXPECT_FALSE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_SGT, APInt(BitWidth, 1))
          .match(One));
  EXPECT_FALSE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_SGT, APInt(BitWidth, 1))
          .match(NegOne));

  EXPECT_TRUE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_SGT, APInt(BitWidth, -1))
          .match(Zero));
  EXPECT_TRUE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_SGT, APInt(BitWidth, -1))
          .match(One));
  EXPECT_FALSE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_SGT, APInt(BitWidth, -1))
          .match(NegOne));
}

TEST_F(PatternMatchTest, SpecificIntSGE) {
  Type *IntTy = IRB.getInt32Ty();
  unsigned BitWidth = IntTy->getScalarSizeInBits();

  Value *Zero = ConstantInt::get(IntTy, 0);
  Value *One = ConstantInt::get(IntTy, 1);
  Value *NegOne = ConstantInt::get(IntTy, -1);

  EXPECT_TRUE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_SGE, APInt(BitWidth, 0))
          .match(Zero));
  EXPECT_TRUE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_SGE, APInt(BitWidth, 0))
          .match(One));
  EXPECT_FALSE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_SGE, APInt(BitWidth, 0))
          .match(NegOne));

  EXPECT_FALSE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_SGE, APInt(BitWidth, 1))
          .match(Zero));
  EXPECT_TRUE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_SGE, APInt(BitWidth, 1))
          .match(One));
  EXPECT_FALSE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_SGE, APInt(BitWidth, 1))
          .match(NegOne));

  EXPECT_TRUE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_SGE, APInt(BitWidth, -1))
          .match(Zero));
  EXPECT_TRUE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_SGE, APInt(BitWidth, -1))
          .match(One));
  EXPECT_TRUE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_SGE, APInt(BitWidth, -1))
          .match(NegOne));
}

TEST_F(PatternMatchTest, SpecificIntSLT) {
  Type *IntTy = IRB.getInt32Ty();
  unsigned BitWidth = IntTy->getScalarSizeInBits();

  Value *Zero = ConstantInt::get(IntTy, 0);
  Value *One = ConstantInt::get(IntTy, 1);
  Value *NegOne = ConstantInt::get(IntTy, -1);

  EXPECT_FALSE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_SLT, APInt(BitWidth, 0))
          .match(Zero));
  EXPECT_FALSE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_SLT, APInt(BitWidth, 0))
          .match(One));
  EXPECT_TRUE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_SLT, APInt(BitWidth, 0))
          .match(NegOne));

  EXPECT_TRUE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_SLT, APInt(BitWidth, 1))
          .match(Zero));
  EXPECT_FALSE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_SLT, APInt(BitWidth, 1))
          .match(One));
  EXPECT_TRUE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_SLT, APInt(BitWidth, 1))
          .match(NegOne));

  EXPECT_FALSE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_SLT, APInt(BitWidth, -1))
          .match(Zero));
  EXPECT_FALSE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_SLT, APInt(BitWidth, -1))
          .match(One));
  EXPECT_FALSE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_SLT, APInt(BitWidth, -1))
          .match(NegOne));
}

TEST_F(PatternMatchTest, SpecificIntSLE) {
  Type *IntTy = IRB.getInt32Ty();
  unsigned BitWidth = IntTy->getScalarSizeInBits();

  Value *Zero = ConstantInt::get(IntTy, 0);
  Value *One = ConstantInt::get(IntTy, 1);
  Value *NegOne = ConstantInt::get(IntTy, -1);

  EXPECT_TRUE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_SLE, APInt(BitWidth, 0))
          .match(Zero));
  EXPECT_FALSE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_SLE, APInt(BitWidth, 0))
          .match(One));
  EXPECT_TRUE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_SLE, APInt(BitWidth, 0))
          .match(NegOne));

  EXPECT_TRUE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_SLE, APInt(BitWidth, 1))
          .match(Zero));
  EXPECT_TRUE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_SLE, APInt(BitWidth, 1))
          .match(One));
  EXPECT_TRUE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_SLE, APInt(BitWidth, 1))
          .match(NegOne));

  EXPECT_FALSE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_SLE, APInt(BitWidth, -1))
          .match(Zero));
  EXPECT_FALSE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_SLE, APInt(BitWidth, -1))
          .match(One));
  EXPECT_TRUE(
      m_SpecificInt_ICMP(ICmpInst::Predicate::ICMP_SLE, APInt(BitWidth, -1))
          .match(NegOne));
}

TEST_F(PatternMatchTest, Unless) {
  Value *X = IRB.CreateAdd(IRB.getInt32(1), IRB.getInt32(0));

  EXPECT_TRUE(m_Add(m_One(), m_Zero()).match(X));
  EXPECT_FALSE(m_Add(m_Zero(), m_One()).match(X));

  EXPECT_FALSE(m_Unless(m_Add(m_One(), m_Zero())).match(X));
  EXPECT_TRUE(m_Unless(m_Add(m_Zero(), m_One())).match(X));

  EXPECT_TRUE(m_c_Add(m_One(), m_Zero()).match(X));
  EXPECT_TRUE(m_c_Add(m_Zero(), m_One()).match(X));

  EXPECT_FALSE(m_Unless(m_c_Add(m_One(), m_Zero())).match(X));
  EXPECT_FALSE(m_Unless(m_c_Add(m_Zero(), m_One())).match(X));
}

TEST_F(PatternMatchTest, ZExtSExtSelf) {
  LLVMContext &Ctx = IRB.getContext();

  Value *One32 = IRB.getInt32(1);
  Value *One64Z = IRB.CreateZExt(One32, IntegerType::getInt64Ty(Ctx));
  Value *One64S = IRB.CreateSExt(One32, IntegerType::getInt64Ty(Ctx));

  EXPECT_TRUE(m_One().match(One32));
  EXPECT_FALSE(m_One().match(One64Z));
  EXPECT_FALSE(m_One().match(One64S));

  EXPECT_FALSE(m_ZExt(m_One()).match(One32));
  EXPECT_TRUE(m_ZExt(m_One()).match(One64Z));
  EXPECT_FALSE(m_ZExt(m_One()).match(One64S));

  EXPECT_FALSE(m_SExt(m_One()).match(One32));
  EXPECT_FALSE(m_SExt(m_One()).match(One64Z));
  EXPECT_TRUE(m_SExt(m_One()).match(One64S));

  EXPECT_TRUE(m_ZExtOrSelf(m_One()).match(One32));
  EXPECT_TRUE(m_ZExtOrSelf(m_One()).match(One64Z));
  EXPECT_FALSE(m_ZExtOrSelf(m_One()).match(One64S));

  EXPECT_TRUE(m_SExtOrSelf(m_One()).match(One32));
  EXPECT_FALSE(m_SExtOrSelf(m_One()).match(One64Z));
  EXPECT_TRUE(m_SExtOrSelf(m_One()).match(One64S));

  EXPECT_FALSE(m_ZExtOrSExt(m_One()).match(One32));
  EXPECT_TRUE(m_ZExtOrSExt(m_One()).match(One64Z));
  EXPECT_TRUE(m_ZExtOrSExt(m_One()).match(One64S));

  EXPECT_TRUE(m_ZExtOrSExtOrSelf(m_One()).match(One32));
  EXPECT_TRUE(m_ZExtOrSExtOrSelf(m_One()).match(One64Z));
  EXPECT_TRUE(m_ZExtOrSExtOrSelf(m_One()).match(One64S));
}

TEST_F(PatternMatchTest, Power2) {
  Value *C128 = IRB.getInt32(128);
  Value *CNeg128 = ConstantExpr::getNeg(cast<Constant>(C128));

  EXPECT_TRUE(m_Power2().match(C128));
  EXPECT_FALSE(m_Power2().match(CNeg128));

  EXPECT_FALSE(m_NegatedPower2().match(C128));
  EXPECT_TRUE(m_NegatedPower2().match(CNeg128));

  Value *CIntMin = IRB.getInt64(APSInt::getSignedMinValue(64).getSExtValue());
  Value *CNegIntMin = ConstantExpr::getNeg(cast<Constant>(CIntMin));

  EXPECT_TRUE(m_Power2().match(CIntMin));
  EXPECT_TRUE(m_Power2().match(CNegIntMin));

  EXPECT_TRUE(m_NegatedPower2().match(CIntMin));
  EXPECT_TRUE(m_NegatedPower2().match(CNegIntMin));
}

TEST_F(PatternMatchTest, CommutativeDeferredValue) {
  Value *X = IRB.getInt32(1);
  Value *Y = IRB.getInt32(2);

  {
    Value *tX = X;
    EXPECT_TRUE(match(X, m_Deferred(tX)));
    EXPECT_FALSE(match(Y, m_Deferred(tX)));
  }
  {
    const Value *tX = X;
    EXPECT_TRUE(match(X, m_Deferred(tX)));
    EXPECT_FALSE(match(Y, m_Deferred(tX)));
  }
  {
    Value *const tX = X;
    EXPECT_TRUE(match(X, m_Deferred(tX)));
    EXPECT_FALSE(match(Y, m_Deferred(tX)));
  }
  {
    const Value *const tX = X;
    EXPECT_TRUE(match(X, m_Deferred(tX)));
    EXPECT_FALSE(match(Y, m_Deferred(tX)));
  }

  {
    Value *tX = nullptr;
    EXPECT_TRUE(match(IRB.CreateAnd(X, X), m_And(m_Value(tX), m_Deferred(tX))));
    EXPECT_EQ(tX, X);
  }
  {
    Value *tX = nullptr;
    EXPECT_FALSE(
        match(IRB.CreateAnd(X, Y), m_c_And(m_Value(tX), m_Deferred(tX))));
  }

  auto checkMatch = [X, Y](Value *Pattern) {
    Value *tX = nullptr, *tY = nullptr;
    EXPECT_TRUE(match(
        Pattern, m_c_And(m_Value(tX), m_c_And(m_Deferred(tX), m_Value(tY)))));
    EXPECT_EQ(tX, X);
    EXPECT_EQ(tY, Y);
  };

  checkMatch(IRB.CreateAnd(X, IRB.CreateAnd(X, Y)));
  checkMatch(IRB.CreateAnd(X, IRB.CreateAnd(Y, X)));
  checkMatch(IRB.CreateAnd(IRB.CreateAnd(X, Y), X));
  checkMatch(IRB.CreateAnd(IRB.CreateAnd(Y, X), X));
}

TEST_F(PatternMatchTest, FloatingPointOrderedMin) {
  Type *FltTy = IRB.getFloatTy();
  Value *L = ConstantFP::get(FltTy, 1.0);
  Value *R = ConstantFP::get(FltTy, 2.0);
  Value *MatchL, *MatchR;

  // Test OLT.
  EXPECT_TRUE(m_OrdFMin(m_Value(MatchL), m_Value(MatchR))
                  .match(IRB.CreateSelect(IRB.CreateFCmpOLT(L, R), L, R)));
  EXPECT_EQ(L, MatchL);
  EXPECT_EQ(R, MatchR);

  // Test OLE.
  EXPECT_TRUE(m_OrdFMin(m_Value(MatchL), m_Value(MatchR))
                  .match(IRB.CreateSelect(IRB.CreateFCmpOLE(L, R), L, R)));
  EXPECT_EQ(L, MatchL);
  EXPECT_EQ(R, MatchR);

  // Test no match on OGE.
  EXPECT_FALSE(m_OrdFMin(m_Value(MatchL), m_Value(MatchR))
                   .match(IRB.CreateSelect(IRB.CreateFCmpOGE(L, R), L, R)));

  // Test no match on OGT.
  EXPECT_FALSE(m_OrdFMin(m_Value(MatchL), m_Value(MatchR))
                   .match(IRB.CreateSelect(IRB.CreateFCmpOGT(L, R), L, R)));

  // Test inverted selects. Note, that this "inverts" the ordering, e.g.:
  // %cmp = fcmp oge L, R
  // %min = select %cmp R, L
  // Given L == NaN
  // the above is expanded to %cmp == false ==> %min = L
  // which is true for UnordFMin, not OrdFMin, so test that:

  // [OU]GE with inverted select.
  EXPECT_FALSE(m_OrdFMin(m_Value(MatchL), m_Value(MatchR))
                  .match(IRB.CreateSelect(IRB.CreateFCmpOGE(L, R), R, L)));
  EXPECT_TRUE(m_OrdFMin(m_Value(MatchL), m_Value(MatchR))
                  .match(IRB.CreateSelect(IRB.CreateFCmpUGE(L, R), R, L)));
  EXPECT_EQ(L, MatchL);
  EXPECT_EQ(R, MatchR);

  // [OU]GT with inverted select.
  EXPECT_FALSE(m_OrdFMin(m_Value(MatchL), m_Value(MatchR))
                  .match(IRB.CreateSelect(IRB.CreateFCmpOGT(L, R), R, L)));
  EXPECT_TRUE(m_OrdFMin(m_Value(MatchL), m_Value(MatchR))
                  .match(IRB.CreateSelect(IRB.CreateFCmpUGT(L, R), R, L)));
  EXPECT_EQ(L, MatchL);
  EXPECT_EQ(R, MatchR);
}

TEST_F(PatternMatchTest, FloatingPointOrderedMax) {
  Type *FltTy = IRB.getFloatTy();
  Value *L = ConstantFP::get(FltTy, 1.0);
  Value *R = ConstantFP::get(FltTy, 2.0);
  Value *MatchL, *MatchR;

  // Test OGT.
  EXPECT_TRUE(m_OrdFMax(m_Value(MatchL), m_Value(MatchR))
                  .match(IRB.CreateSelect(IRB.CreateFCmpOGT(L, R), L, R)));
  EXPECT_EQ(L, MatchL);
  EXPECT_EQ(R, MatchR);

  // Test OGE.
  EXPECT_TRUE(m_OrdFMax(m_Value(MatchL), m_Value(MatchR))
                  .match(IRB.CreateSelect(IRB.CreateFCmpOGE(L, R), L, R)));
  EXPECT_EQ(L, MatchL);
  EXPECT_EQ(R, MatchR);

  // Test no match on OLE.
  EXPECT_FALSE(m_OrdFMax(m_Value(MatchL), m_Value(MatchR))
                   .match(IRB.CreateSelect(IRB.CreateFCmpOLE(L, R), L, R)));

  // Test no match on OLT.
  EXPECT_FALSE(m_OrdFMax(m_Value(MatchL), m_Value(MatchR))
                   .match(IRB.CreateSelect(IRB.CreateFCmpOLT(L, R), L, R)));


  // Test inverted selects. Note, that this "inverts" the ordering, e.g.:
  // %cmp = fcmp ole L, R
  // %max = select %cmp, R, L
  // Given L == NaN,
  // the above is expanded to %cmp == false ==> %max == L
  // which is true for UnordFMax, not OrdFMax, so test that:

  // [OU]LE with inverted select.
  EXPECT_FALSE(m_OrdFMax(m_Value(MatchL), m_Value(MatchR))
                   .match(IRB.CreateSelect(IRB.CreateFCmpOLE(L, R), R, L)));
  EXPECT_TRUE(m_OrdFMax(m_Value(MatchL), m_Value(MatchR))
                  .match(IRB.CreateSelect(IRB.CreateFCmpULE(L, R), R, L)));
  EXPECT_EQ(L, MatchL);
  EXPECT_EQ(R, MatchR);

  // [OUT]LT with inverted select.
  EXPECT_FALSE(m_OrdFMax(m_Value(MatchL), m_Value(MatchR))
                   .match(IRB.CreateSelect(IRB.CreateFCmpOLT(L, R), R, L)));
  EXPECT_TRUE(m_OrdFMax(m_Value(MatchL), m_Value(MatchR))
                  .match(IRB.CreateSelect(IRB.CreateFCmpULT(L, R), R, L)));
  EXPECT_EQ(L, MatchL);
  EXPECT_EQ(R, MatchR);
}

TEST_F(PatternMatchTest, FloatingPointUnorderedMin) {
  Type *FltTy = IRB.getFloatTy();
  Value *L = ConstantFP::get(FltTy, 1.0);
  Value *R = ConstantFP::get(FltTy, 2.0);
  Value *MatchL, *MatchR;

  // Test ULT.
  EXPECT_TRUE(m_UnordFMin(m_Value(MatchL), m_Value(MatchR))
                  .match(IRB.CreateSelect(IRB.CreateFCmpULT(L, R), L, R)));
  EXPECT_EQ(L, MatchL);
  EXPECT_EQ(R, MatchR);

  // Test ULE.
  EXPECT_TRUE(m_UnordFMin(m_Value(MatchL), m_Value(MatchR))
                  .match(IRB.CreateSelect(IRB.CreateFCmpULE(L, R), L, R)));
  EXPECT_EQ(L, MatchL);
  EXPECT_EQ(R, MatchR);

  // Test no match on UGE.
  EXPECT_FALSE(m_UnordFMin(m_Value(MatchL), m_Value(MatchR))
                   .match(IRB.CreateSelect(IRB.CreateFCmpUGE(L, R), L, R)));

  // Test no match on UGT.
  EXPECT_FALSE(m_UnordFMin(m_Value(MatchL), m_Value(MatchR))
                   .match(IRB.CreateSelect(IRB.CreateFCmpUGT(L, R), L, R)));

  // Test inverted selects. Note, that this "inverts" the ordering, e.g.:
  // %cmp = fcmp uge L, R
  // %min = select %cmp R, L
  // Given L == NaN
  // the above is expanded to %cmp == true ==> %min = R
  // which is true for OrdFMin, not UnordFMin, so test that:

  // [UO]GE with inverted select.
  EXPECT_FALSE(m_UnordFMin(m_Value(MatchL), m_Value(MatchR))
                  .match(IRB.CreateSelect(IRB.CreateFCmpUGE(L, R), R, L)));
  EXPECT_TRUE(m_UnordFMin(m_Value(MatchL), m_Value(MatchR))
                  .match(IRB.CreateSelect(IRB.CreateFCmpOGE(L, R), R, L)));
  EXPECT_EQ(L, MatchL);
  EXPECT_EQ(R, MatchR);

  // [UO]GT with inverted select.
  EXPECT_FALSE(m_UnordFMin(m_Value(MatchL), m_Value(MatchR))
                  .match(IRB.CreateSelect(IRB.CreateFCmpUGT(L, R), R, L)));
  EXPECT_TRUE(m_UnordFMin(m_Value(MatchL), m_Value(MatchR))
                  .match(IRB.CreateSelect(IRB.CreateFCmpOGT(L, R), R, L)));
  EXPECT_EQ(L, MatchL);
  EXPECT_EQ(R, MatchR);
}

TEST_F(PatternMatchTest, FloatingPointUnorderedMax) {
  Type *FltTy = IRB.getFloatTy();
  Value *L = ConstantFP::get(FltTy, 1.0);
  Value *R = ConstantFP::get(FltTy, 2.0);
  Value *MatchL, *MatchR;

  // Test UGT.
  EXPECT_TRUE(m_UnordFMax(m_Value(MatchL), m_Value(MatchR))
                  .match(IRB.CreateSelect(IRB.CreateFCmpUGT(L, R), L, R)));
  EXPECT_EQ(L, MatchL);
  EXPECT_EQ(R, MatchR);

  // Test UGE.
  EXPECT_TRUE(m_UnordFMax(m_Value(MatchL), m_Value(MatchR))
                  .match(IRB.CreateSelect(IRB.CreateFCmpUGE(L, R), L, R)));
  EXPECT_EQ(L, MatchL);
  EXPECT_EQ(R, MatchR);

  // Test no match on ULE.
  EXPECT_FALSE(m_UnordFMax(m_Value(MatchL), m_Value(MatchR))
                   .match(IRB.CreateSelect(IRB.CreateFCmpULE(L, R), L, R)));

  // Test no match on ULT.
  EXPECT_FALSE(m_UnordFMax(m_Value(MatchL), m_Value(MatchR))
                   .match(IRB.CreateSelect(IRB.CreateFCmpULT(L, R), L, R)));

  // Test inverted selects. Note, that this "inverts" the ordering, e.g.:
  // %cmp = fcmp ule L, R
  // %max = select %cmp R, L
  // Given L == NaN
  // the above is expanded to %cmp == true ==> %max = R
  // which is true for OrdFMax, not UnordFMax, so test that:

  // [UO]LE with inverted select.
  EXPECT_FALSE(m_UnordFMax(m_Value(MatchL), m_Value(MatchR))
                  .match(IRB.CreateSelect(IRB.CreateFCmpULE(L, R), R, L)));
  EXPECT_TRUE(m_UnordFMax(m_Value(MatchL), m_Value(MatchR))
                  .match(IRB.CreateSelect(IRB.CreateFCmpOLE(L, R), R, L)));
  EXPECT_EQ(L, MatchL);
  EXPECT_EQ(R, MatchR);

  // [UO]LT with inverted select.
  EXPECT_FALSE(m_UnordFMax(m_Value(MatchL), m_Value(MatchR))
                  .match(IRB.CreateSelect(IRB.CreateFCmpULT(L, R), R, L)));
  EXPECT_TRUE(m_UnordFMax(m_Value(MatchL), m_Value(MatchR))
                  .match(IRB.CreateSelect(IRB.CreateFCmpOLT(L, R), R, L)));
  EXPECT_EQ(L, MatchL);
  EXPECT_EQ(R, MatchR);
}

TEST_F(PatternMatchTest, OverflowingBinOps) {
  Value *L = IRB.getInt32(1);
  Value *R = IRB.getInt32(2);
  Value *MatchL, *MatchR;

  EXPECT_TRUE(
      m_NSWAdd(m_Value(MatchL), m_Value(MatchR)).match(IRB.CreateNSWAdd(L, R)));
  EXPECT_EQ(L, MatchL);
  EXPECT_EQ(R, MatchR);
  MatchL = MatchR = nullptr;
  EXPECT_TRUE(
      m_NSWSub(m_Value(MatchL), m_Value(MatchR)).match(IRB.CreateNSWSub(L, R)));
  EXPECT_EQ(L, MatchL);
  EXPECT_EQ(R, MatchR);
  MatchL = MatchR = nullptr;
  EXPECT_TRUE(
      m_NSWMul(m_Value(MatchL), m_Value(MatchR)).match(IRB.CreateNSWMul(L, R)));
  EXPECT_EQ(L, MatchL);
  EXPECT_EQ(R, MatchR);
  MatchL = MatchR = nullptr;
  EXPECT_TRUE(m_NSWShl(m_Value(MatchL), m_Value(MatchR)).match(
      IRB.CreateShl(L, R, "", /* NUW */ false, /* NSW */ true)));
  EXPECT_EQ(L, MatchL);
  EXPECT_EQ(R, MatchR);

  EXPECT_TRUE(
      m_NUWAdd(m_Value(MatchL), m_Value(MatchR)).match(IRB.CreateNUWAdd(L, R)));
  EXPECT_EQ(L, MatchL);
  EXPECT_EQ(R, MatchR);
  MatchL = MatchR = nullptr;
  EXPECT_TRUE(
      m_NUWSub(m_Value(MatchL), m_Value(MatchR)).match(IRB.CreateNUWSub(L, R)));
  EXPECT_EQ(L, MatchL);
  EXPECT_EQ(R, MatchR);
  MatchL = MatchR = nullptr;
  EXPECT_TRUE(
      m_NUWMul(m_Value(MatchL), m_Value(MatchR)).match(IRB.CreateNUWMul(L, R)));
  EXPECT_EQ(L, MatchL);
  EXPECT_EQ(R, MatchR);
  MatchL = MatchR = nullptr;
  EXPECT_TRUE(m_NUWShl(m_Value(MatchL), m_Value(MatchR)).match(
      IRB.CreateShl(L, R, "", /* NUW */ true, /* NSW */ false)));
  EXPECT_EQ(L, MatchL);
  EXPECT_EQ(R, MatchR);

  EXPECT_FALSE(m_NSWAdd(m_Value(), m_Value()).match(IRB.CreateAdd(L, R)));
  EXPECT_FALSE(m_NSWAdd(m_Value(), m_Value()).match(IRB.CreateNUWAdd(L, R)));
  EXPECT_FALSE(m_NSWAdd(m_Value(), m_Value()).match(IRB.CreateNSWSub(L, R)));
  EXPECT_FALSE(m_NSWSub(m_Value(), m_Value()).match(IRB.CreateSub(L, R)));
  EXPECT_FALSE(m_NSWSub(m_Value(), m_Value()).match(IRB.CreateNUWSub(L, R)));
  EXPECT_FALSE(m_NSWSub(m_Value(), m_Value()).match(IRB.CreateNSWAdd(L, R)));
  EXPECT_FALSE(m_NSWMul(m_Value(), m_Value()).match(IRB.CreateMul(L, R)));
  EXPECT_FALSE(m_NSWMul(m_Value(), m_Value()).match(IRB.CreateNUWMul(L, R)));
  EXPECT_FALSE(m_NSWMul(m_Value(), m_Value()).match(IRB.CreateNSWAdd(L, R)));
  EXPECT_FALSE(m_NSWShl(m_Value(), m_Value()).match(IRB.CreateShl(L, R)));
  EXPECT_FALSE(m_NSWShl(m_Value(), m_Value()).match(
      IRB.CreateShl(L, R, "", /* NUW */ true, /* NSW */ false)));
  EXPECT_FALSE(m_NSWShl(m_Value(), m_Value()).match(IRB.CreateNSWAdd(L, R)));

  EXPECT_FALSE(m_NUWAdd(m_Value(), m_Value()).match(IRB.CreateAdd(L, R)));
  EXPECT_FALSE(m_NUWAdd(m_Value(), m_Value()).match(IRB.CreateNSWAdd(L, R)));
  EXPECT_FALSE(m_NUWAdd(m_Value(), m_Value()).match(IRB.CreateNUWSub(L, R)));
  EXPECT_FALSE(m_NUWSub(m_Value(), m_Value()).match(IRB.CreateSub(L, R)));
  EXPECT_FALSE(m_NUWSub(m_Value(), m_Value()).match(IRB.CreateNSWSub(L, R)));
  EXPECT_FALSE(m_NUWSub(m_Value(), m_Value()).match(IRB.CreateNUWAdd(L, R)));
  EXPECT_FALSE(m_NUWMul(m_Value(), m_Value()).match(IRB.CreateMul(L, R)));
  EXPECT_FALSE(m_NUWMul(m_Value(), m_Value()).match(IRB.CreateNSWMul(L, R)));
  EXPECT_FALSE(m_NUWMul(m_Value(), m_Value()).match(IRB.CreateNUWAdd(L, R)));
  EXPECT_FALSE(m_NUWShl(m_Value(), m_Value()).match(IRB.CreateShl(L, R)));
  EXPECT_FALSE(m_NUWShl(m_Value(), m_Value()).match(
      IRB.CreateShl(L, R, "", /* NUW */ false, /* NSW */ true)));
  EXPECT_FALSE(m_NUWShl(m_Value(), m_Value()).match(IRB.CreateNUWAdd(L, R)));
}

TEST_F(PatternMatchTest, LoadStoreOps) {
  // Create this load/store sequence:
  //
  //  %p = alloca i32*
  //  %0 = load i32*, i32** %p
  //  store i32 42, i32* %0

  Value *Alloca = IRB.CreateAlloca(IRB.getInt32Ty());
  Value *LoadInst = IRB.CreateLoad(IRB.getInt32Ty(), Alloca);
  Value *FourtyTwo = IRB.getInt32(42);
  Value *StoreInst = IRB.CreateStore(FourtyTwo, Alloca);
  Value *MatchLoad, *MatchStoreVal, *MatchStorePointer;

  EXPECT_TRUE(m_Load(m_Value(MatchLoad)).match(LoadInst));
  EXPECT_EQ(Alloca, MatchLoad);

  EXPECT_TRUE(m_Load(m_Specific(Alloca)).match(LoadInst));

  EXPECT_FALSE(m_Load(m_Value(MatchLoad)).match(Alloca));

  EXPECT_TRUE(m_Store(m_Value(MatchStoreVal), m_Value(MatchStorePointer))
                .match(StoreInst));
  EXPECT_EQ(FourtyTwo, MatchStoreVal);
  EXPECT_EQ(Alloca, MatchStorePointer);

  EXPECT_FALSE(m_Store(m_Value(MatchStoreVal), m_Value(MatchStorePointer))
                .match(Alloca));

  EXPECT_TRUE(m_Store(m_SpecificInt(42), m_Specific(Alloca))
                .match(StoreInst));
  EXPECT_FALSE(m_Store(m_SpecificInt(42), m_Specific(FourtyTwo))
                .match(StoreInst));
  EXPECT_FALSE(m_Store(m_SpecificInt(43), m_Specific(Alloca))
                .match(StoreInst));
}

TEST_F(PatternMatchTest, VectorOps) {
  // Build up small tree of vector operations
  //
  //   Val = 0 + 1
  //   Val2 = Val + 3
  //   VI1 = insertelement <2 x i8> undef, i8 1, i32 0 = <1, undef>
  //   VI2 = insertelement <2 x i8> %VI1, i8 %Val2, i8 %Val = <1, 4>
  //   VI3 = insertelement <2 x i8> %VI1, i8 %Val2, i32 1 = <1, 4>
  //   VI4 = insertelement <2 x i8> %VI1, i8 2, i8 %Val = <1, 2>
  //
  //   SI1 = shufflevector <2 x i8> %VI1, <2 x i8> undef, zeroinitializer
  //   SI2 = shufflevector <2 x i8> %VI3, <2 x i8> %VI4, <2 x i8> <i8 0, i8 2>
  //   SI3 = shufflevector <2 x i8> %VI3, <2 x i8> undef, zeroinitializer
  //   SI4 = shufflevector <2 x i8> %VI4, <2 x i8> undef, zeroinitializer
  //
  //   SP1 = VectorSplat(2, i8 2)
  //   SP2 = VectorSplat(2, i8 %Val)
  Type *VecTy = VectorType::get(IRB.getInt8Ty(), 2);
  Type *i32 = IRB.getInt32Ty();
  Type *i32VecTy = VectorType::get(i32, 2);

  Value *Val = IRB.CreateAdd(IRB.getInt8(0), IRB.getInt8(1));
  Value *Val2 = IRB.CreateAdd(Val, IRB.getInt8(3));

  SmallVector<Constant *, 2> VecElemIdxs;
  VecElemIdxs.push_back(ConstantInt::get(i32, 0));
  VecElemIdxs.push_back(ConstantInt::get(i32, 2));
  auto *IdxVec = ConstantVector::get(VecElemIdxs);

  Value *UndefVec = UndefValue::get(VecTy);
  Value *VI1 = IRB.CreateInsertElement(UndefVec, IRB.getInt8(1), (uint64_t)0);
  Value *VI2 = IRB.CreateInsertElement(VI1, Val2, Val);
  Value *VI3 = IRB.CreateInsertElement(VI1, Val2, (uint64_t)1);
  Value *VI4 = IRB.CreateInsertElement(VI1, IRB.getInt8(2), Val);

  Value *EX1 = IRB.CreateExtractElement(VI4, Val);
  Value *EX2 = IRB.CreateExtractElement(VI4, (uint64_t)0);
  Value *EX3 = IRB.CreateExtractElement(IdxVec, (uint64_t)1);

  Value *Zero = ConstantAggregateZero::get(i32VecTy);
  Value *SI1 = IRB.CreateShuffleVector(VI1, UndefVec, Zero);
  Value *SI2 = IRB.CreateShuffleVector(VI3, VI4, IdxVec);
  Value *SI3 = IRB.CreateShuffleVector(VI3, UndefVec, Zero);
  Value *SI4 = IRB.CreateShuffleVector(VI4, UndefVec, Zero);

  Value *SP1 = IRB.CreateVectorSplat(2, IRB.getInt8(2));
  Value *SP2 = IRB.CreateVectorSplat(2, Val);

  Value *A = nullptr, *B = nullptr, *C = nullptr;

  // Test matching insertelement
  EXPECT_TRUE(match(VI1, m_InsertElt(m_Value(), m_Value(), m_Value())));
  EXPECT_TRUE(
      match(VI1, m_InsertElt(m_Undef(), m_ConstantInt(), m_ConstantInt())));
  EXPECT_TRUE(
      match(VI1, m_InsertElt(m_Undef(), m_ConstantInt(), m_Zero())));
  EXPECT_TRUE(
      match(VI1, m_InsertElt(m_Undef(), m_SpecificInt(1), m_Zero())));
  EXPECT_TRUE(match(VI2, m_InsertElt(m_Value(), m_Value(), m_Value())));
  EXPECT_FALSE(
      match(VI2, m_InsertElt(m_Value(), m_Value(), m_ConstantInt())));
  EXPECT_FALSE(
      match(VI2, m_InsertElt(m_Value(), m_ConstantInt(), m_Value())));
  EXPECT_FALSE(match(VI2, m_InsertElt(m_Constant(), m_Value(), m_Value())));
  EXPECT_TRUE(match(VI3, m_InsertElt(m_Value(A), m_Value(B), m_Value(C))));
  EXPECT_TRUE(A == VI1);
  EXPECT_TRUE(B == Val2);
  EXPECT_TRUE(isa<ConstantInt>(C));
  A = B = C = nullptr; // reset

  // Test matching extractelement
  EXPECT_TRUE(match(EX1, m_ExtractElt(m_Value(A), m_Value(B))));
  EXPECT_TRUE(A == VI4);
  EXPECT_TRUE(B == Val);
  A = B = C = nullptr; // reset
  EXPECT_FALSE(match(EX1, m_ExtractElt(m_Value(), m_ConstantInt())));
  EXPECT_TRUE(match(EX2, m_ExtractElt(m_Value(), m_ConstantInt())));
  EXPECT_TRUE(match(EX3, m_ExtractElt(m_Constant(), m_ConstantInt())));

  // Test matching shufflevector
  ArrayRef<int> Mask;
  EXPECT_TRUE(match(SI1, m_Shuffle(m_Value(), m_Undef(), m_ZeroMask())));
  EXPECT_TRUE(match(SI2, m_Shuffle(m_Value(A), m_Value(B), m_Mask(Mask))));
  EXPECT_TRUE(A == VI3);
  EXPECT_TRUE(B == VI4);
  A = B = C = nullptr; // reset

  // Test matching the vector splat pattern
  EXPECT_TRUE(match(
      SI1,
      m_Shuffle(m_InsertElt(m_Undef(), m_SpecificInt(1), m_Zero()),
                m_Undef(), m_ZeroMask())));
  EXPECT_FALSE(match(
      SI3, m_Shuffle(m_InsertElt(m_Undef(), m_Value(), m_Zero()),
                     m_Undef(), m_ZeroMask())));
  EXPECT_FALSE(match(
      SI4, m_Shuffle(m_InsertElt(m_Undef(), m_Value(), m_Zero()),
                     m_Undef(), m_ZeroMask())));
  EXPECT_TRUE(match(
      SP1,
      m_Shuffle(m_InsertElt(m_Undef(), m_SpecificInt(2), m_Zero()),
                m_Undef(), m_ZeroMask())));
  EXPECT_TRUE(match(
      SP2, m_Shuffle(m_InsertElt(m_Undef(), m_Value(A), m_Zero()),
                     m_Undef(), m_ZeroMask())));
  EXPECT_TRUE(A == Val);
}

TEST_F(PatternMatchTest, VectorUndefInt) {
  Type *ScalarTy = IRB.getInt8Ty();
  Type *VectorTy = VectorType::get(ScalarTy, 4);
  Constant *ScalarUndef = UndefValue::get(ScalarTy);
  Constant *VectorUndef = UndefValue::get(VectorTy);
  Constant *ScalarZero = Constant::getNullValue(ScalarTy);
  Constant *VectorZero = Constant::getNullValue(VectorTy);

  SmallVector<Constant *, 4> Elems;
  Elems.push_back(ScalarUndef);
  Elems.push_back(ScalarZero);
  Elems.push_back(ScalarUndef);
  Elems.push_back(ScalarZero);
  Constant *VectorZeroUndef = ConstantVector::get(Elems);

  EXPECT_TRUE(match(ScalarUndef, m_Undef()));
  EXPECT_TRUE(match(VectorUndef, m_Undef()));
  EXPECT_FALSE(match(ScalarZero, m_Undef()));
  EXPECT_FALSE(match(VectorZero, m_Undef()));
  EXPECT_FALSE(match(VectorZeroUndef, m_Undef()));

  EXPECT_FALSE(match(ScalarUndef, m_Zero()));
  EXPECT_FALSE(match(VectorUndef, m_Zero()));
  EXPECT_TRUE(match(ScalarZero, m_Zero()));
  EXPECT_TRUE(match(VectorZero, m_Zero()));
  EXPECT_TRUE(match(VectorZeroUndef, m_Zero()));

  const APInt *C;
  // Regardless of whether undefs are allowed,
  // a fully undef constant does not match.
  EXPECT_FALSE(match(ScalarUndef, m_APInt(C)));
  EXPECT_FALSE(match(ScalarUndef, m_APIntForbidUndef(C)));
  EXPECT_FALSE(match(ScalarUndef, m_APIntAllowUndef(C)));
  EXPECT_FALSE(match(VectorUndef, m_APInt(C)));
  EXPECT_FALSE(match(VectorUndef, m_APIntForbidUndef(C)));
  EXPECT_FALSE(match(VectorUndef, m_APIntAllowUndef(C)));

  // We can always match simple constants and simple splats.
  C = nullptr;
  EXPECT_TRUE(match(ScalarZero, m_APInt(C)));
  EXPECT_TRUE(C->isNullValue());
  C = nullptr;
  EXPECT_TRUE(match(ScalarZero, m_APIntForbidUndef(C)));
  EXPECT_TRUE(C->isNullValue());
  C = nullptr;
  EXPECT_TRUE(match(ScalarZero, m_APIntAllowUndef(C)));
  EXPECT_TRUE(C->isNullValue());
  C = nullptr;
  EXPECT_TRUE(match(VectorZero, m_APInt(C)));
  EXPECT_TRUE(C->isNullValue());
  C = nullptr;
  EXPECT_TRUE(match(VectorZero, m_APIntForbidUndef(C)));
  EXPECT_TRUE(C->isNullValue());
  C = nullptr;
  EXPECT_TRUE(match(VectorZero, m_APIntAllowUndef(C)));
  EXPECT_TRUE(C->isNullValue());

  // Whether splats with undef can be matched depends on the matcher.
  EXPECT_FALSE(match(VectorZeroUndef, m_APInt(C)));
  EXPECT_FALSE(match(VectorZeroUndef, m_APIntForbidUndef(C)));
  C = nullptr;
  EXPECT_TRUE(match(VectorZeroUndef, m_APIntAllowUndef(C)));
  EXPECT_TRUE(C->isNullValue());
}

TEST_F(PatternMatchTest, VectorUndefFloat) {
  Type *ScalarTy = IRB.getFloatTy();
  Type *VectorTy = VectorType::get(ScalarTy, 4);
  Constant *ScalarUndef = UndefValue::get(ScalarTy);
  Constant *VectorUndef = UndefValue::get(VectorTy);
  Constant *ScalarZero = Constant::getNullValue(ScalarTy);
  Constant *VectorZero = Constant::getNullValue(VectorTy);
  Constant *ScalarPosInf = ConstantFP::getInfinity(ScalarTy, false);
  Constant *ScalarNegInf = ConstantFP::getInfinity(ScalarTy, true);

  SmallVector<Constant *, 4> Elems;
  Elems.push_back(ScalarUndef);
  Elems.push_back(ScalarZero);
  Elems.push_back(ScalarUndef);
  Elems.push_back(ScalarZero);
  Constant *VectorZeroUndef = ConstantVector::get(Elems);

  SmallVector<Constant *, 4> InfElems;
  InfElems.push_back(ScalarPosInf);
  InfElems.push_back(ScalarNegInf);
  InfElems.push_back(ScalarUndef);
  InfElems.push_back(ScalarPosInf);
  Constant *VectorInfUndef = ConstantVector::get(InfElems);

  EXPECT_TRUE(match(ScalarUndef, m_Undef()));
  EXPECT_TRUE(match(VectorUndef, m_Undef()));
  EXPECT_FALSE(match(ScalarZero, m_Undef()));
  EXPECT_FALSE(match(VectorZero, m_Undef()));
  EXPECT_FALSE(match(VectorZeroUndef, m_Undef()));

  EXPECT_FALSE(match(ScalarUndef, m_AnyZeroFP()));
  EXPECT_FALSE(match(VectorUndef, m_AnyZeroFP()));
  EXPECT_TRUE(match(ScalarZero, m_AnyZeroFP()));
  EXPECT_TRUE(match(VectorZero, m_AnyZeroFP()));
  EXPECT_TRUE(match(VectorZeroUndef, m_AnyZeroFP()));

  EXPECT_FALSE(match(ScalarUndef, m_Inf()));
  EXPECT_FALSE(match(VectorUndef, m_Inf()));
  EXPECT_FALSE(match(VectorZeroUndef, m_Inf()));
  EXPECT_TRUE(match(ScalarPosInf, m_Inf()));
  EXPECT_TRUE(match(ScalarNegInf, m_Inf()));
  EXPECT_TRUE(match(VectorInfUndef, m_Inf()));

  const APFloat *C;
  // Regardless of whether undefs are allowed,
  // a fully undef constant does not match.
  EXPECT_FALSE(match(ScalarUndef, m_APFloat(C)));
  EXPECT_FALSE(match(ScalarUndef, m_APFloatForbidUndef(C)));
  EXPECT_FALSE(match(ScalarUndef, m_APFloatAllowUndef(C)));
  EXPECT_FALSE(match(VectorUndef, m_APFloat(C)));
  EXPECT_FALSE(match(VectorUndef, m_APFloatForbidUndef(C)));
  EXPECT_FALSE(match(VectorUndef, m_APFloatAllowUndef(C)));

  // We can always match simple constants and simple splats.
  C = nullptr;
  EXPECT_TRUE(match(ScalarZero, m_APFloat(C)));
  EXPECT_TRUE(C->isZero());
  C = nullptr;
  EXPECT_TRUE(match(ScalarZero, m_APFloatForbidUndef(C)));
  EXPECT_TRUE(C->isZero());
  C = nullptr;
  EXPECT_TRUE(match(ScalarZero, m_APFloatAllowUndef(C)));
  EXPECT_TRUE(C->isZero());
  C = nullptr;
  EXPECT_TRUE(match(VectorZero, m_APFloat(C)));
  EXPECT_TRUE(C->isZero());
  C = nullptr;
  EXPECT_TRUE(match(VectorZero, m_APFloatForbidUndef(C)));
  EXPECT_TRUE(C->isZero());
  C = nullptr;
  EXPECT_TRUE(match(VectorZero, m_APFloatAllowUndef(C)));
  EXPECT_TRUE(C->isZero());

  // Whether splats with undef can be matched depends on the matcher.
  EXPECT_FALSE(match(VectorZeroUndef, m_APFloat(C)));
  EXPECT_FALSE(match(VectorZeroUndef, m_APFloatForbidUndef(C)));
  C = nullptr;
  EXPECT_TRUE(match(VectorZeroUndef, m_APFloatAllowUndef(C)));
  EXPECT_TRUE(C->isZero());
}

TEST_F(PatternMatchTest, FloatingPointFNeg) {
  Type *FltTy = IRB.getFloatTy();
  Value *One = ConstantFP::get(FltTy, 1.0);
  Value *Z = ConstantFP::get(FltTy, 0.0);
  Value *NZ = ConstantFP::get(FltTy, -0.0);
  Value *V = IRB.CreateFNeg(One);
  Value *V1 = IRB.CreateFSub(NZ, One);
  Value *V2 = IRB.CreateFSub(Z, One);
  Value *V3 = IRB.CreateFAdd(NZ, One);
  Value *Match;

  // Test FNeg(1.0)
  EXPECT_TRUE(match(V, m_FNeg(m_Value(Match))));
  EXPECT_EQ(One, Match);

  // Test FSub(-0.0, 1.0)
  EXPECT_TRUE(match(V1, m_FNeg(m_Value(Match))));
  EXPECT_EQ(One, Match);

  // Test FSub(0.0, 1.0)
  EXPECT_FALSE(match(V2, m_FNeg(m_Value(Match))));
  cast<Instruction>(V2)->setHasNoSignedZeros(true);
  EXPECT_TRUE(match(V2, m_FNeg(m_Value(Match))));
  EXPECT_EQ(One, Match);

  // Test FAdd(-0.0, 1.0)
  EXPECT_FALSE(match(V3, m_FNeg(m_Value(Match))));
}

TEST_F(PatternMatchTest, CondBranchTest) {
  BasicBlock *TrueBB = BasicBlock::Create(Ctx, "TrueBB", F);
  BasicBlock *FalseBB = BasicBlock::Create(Ctx, "FalseBB", F);
  Value *Br1 = IRB.CreateCondBr(IRB.getTrue(), TrueBB, FalseBB);

  EXPECT_TRUE(match(Br1, m_Br(m_Value(), m_BasicBlock(), m_BasicBlock())));

  BasicBlock *A, *B;
  EXPECT_TRUE(match(Br1, m_Br(m_Value(), m_BasicBlock(A), m_BasicBlock(B))));
  EXPECT_EQ(TrueBB, A);
  EXPECT_EQ(FalseBB, B);

  EXPECT_FALSE(
      match(Br1, m_Br(m_Value(), m_SpecificBB(FalseBB), m_BasicBlock())));
  EXPECT_FALSE(
      match(Br1, m_Br(m_Value(), m_BasicBlock(), m_SpecificBB(TrueBB))));
  EXPECT_FALSE(
      match(Br1, m_Br(m_Value(), m_SpecificBB(FalseBB), m_BasicBlock(TrueBB))));
  EXPECT_TRUE(
      match(Br1, m_Br(m_Value(), m_SpecificBB(TrueBB), m_BasicBlock(FalseBB))));

  // Check we can use m_Deferred with branches.
  EXPECT_FALSE(match(Br1, m_Br(m_Value(), m_BasicBlock(A), m_Deferred(A))));
  Value *Br2 = IRB.CreateCondBr(IRB.getTrue(), TrueBB, TrueBB);
  A = nullptr;
  EXPECT_TRUE(match(Br2, m_Br(m_Value(), m_BasicBlock(A), m_Deferred(A))));
}

TEST_F(PatternMatchTest, WithOverflowInst) {
  Value *Add = IRB.CreateBinaryIntrinsic(Intrinsic::uadd_with_overflow,
                                         IRB.getInt32(0), IRB.getInt32(0));
  Value *Add0 = IRB.CreateExtractValue(Add, 0);
  Value *Add1 = IRB.CreateExtractValue(Add, 1);

  EXPECT_TRUE(match(Add0, m_ExtractValue<0>(m_Value())));
  EXPECT_FALSE(match(Add0, m_ExtractValue<1>(m_Value())));
  EXPECT_FALSE(match(Add1, m_ExtractValue<0>(m_Value())));
  EXPECT_TRUE(match(Add1, m_ExtractValue<1>(m_Value())));
  EXPECT_FALSE(match(Add, m_ExtractValue<1>(m_Value())));
  EXPECT_FALSE(match(Add, m_ExtractValue<1>(m_Value())));

  WithOverflowInst *WOI;
  EXPECT_FALSE(match(Add0, m_WithOverflowInst(WOI)));
  EXPECT_FALSE(match(Add1, m_WithOverflowInst(WOI)));
  EXPECT_TRUE(match(Add, m_WithOverflowInst(WOI)));

  EXPECT_TRUE(match(Add0, m_ExtractValue<0>(m_WithOverflowInst(WOI))));
  EXPECT_EQ(Add, WOI);
  EXPECT_TRUE(match(Add1, m_ExtractValue<1>(m_WithOverflowInst(WOI))));
  EXPECT_EQ(Add, WOI);
}

TEST_F(PatternMatchTest, IntrinsicMatcher) {
  Value *Name = IRB.CreateAlloca(IRB.getInt8Ty());
  Value *Hash = IRB.getInt64(0);
  Value *Num = IRB.getInt32(1);
  Value *Index = IRB.getInt32(2);
  Value *Step = IRB.getInt64(3);

  Value *Ops[] = {Name, Hash, Num, Index, Step};
  Module *M = BB->getParent()->getParent();
  Function *TheFn =
      Intrinsic::getDeclaration(M, Intrinsic::instrprof_increment_step);

  Value *Intrinsic5 = CallInst::Create(TheFn, Ops, "", BB);

  // Match without capturing.
  EXPECT_TRUE(match(
      Intrinsic5, m_Intrinsic<Intrinsic::instrprof_increment_step>(
                      m_Value(), m_Value(), m_Value(), m_Value(), m_Value())));
  EXPECT_FALSE(match(
      Intrinsic5, m_Intrinsic<Intrinsic::memmove>(
                      m_Value(), m_Value(), m_Value(), m_Value(), m_Value())));

  // Match with capturing.
  Value *Arg1 = nullptr;
  Value *Arg2 = nullptr;
  Value *Arg3 = nullptr;
  Value *Arg4 = nullptr;
  Value *Arg5 = nullptr;
  EXPECT_TRUE(
      match(Intrinsic5, m_Intrinsic<Intrinsic::instrprof_increment_step>(
                            m_Value(Arg1), m_Value(Arg2), m_Value(Arg3),
                            m_Value(Arg4), m_Value(Arg5))));
  EXPECT_EQ(Arg1, Name);
  EXPECT_EQ(Arg2, Hash);
  EXPECT_EQ(Arg3, Num);
  EXPECT_EQ(Arg4, Index);
  EXPECT_EQ(Arg5, Step);

  // Match specific second argument.
  EXPECT_TRUE(
      match(Intrinsic5,
            m_Intrinsic<Intrinsic::instrprof_increment_step>(
                m_Value(), m_SpecificInt(0), m_Value(), m_Value(), m_Value())));
  EXPECT_FALSE(
      match(Intrinsic5, m_Intrinsic<Intrinsic::instrprof_increment_step>(
                            m_Value(), m_SpecificInt(10), m_Value(), m_Value(),
                            m_Value())));

  // Match specific third argument.
  EXPECT_TRUE(
      match(Intrinsic5,
            m_Intrinsic<Intrinsic::instrprof_increment_step>(
                m_Value(), m_Value(), m_SpecificInt(1), m_Value(), m_Value())));
  EXPECT_FALSE(
      match(Intrinsic5, m_Intrinsic<Intrinsic::instrprof_increment_step>(
                            m_Value(), m_Value(), m_SpecificInt(10), m_Value(),
                            m_Value())));

  // Match specific fourth argument.
  EXPECT_TRUE(
      match(Intrinsic5,
            m_Intrinsic<Intrinsic::instrprof_increment_step>(
                m_Value(), m_Value(), m_Value(), m_SpecificInt(2), m_Value())));
  EXPECT_FALSE(
      match(Intrinsic5, m_Intrinsic<Intrinsic::instrprof_increment_step>(
                            m_Value(), m_Value(), m_Value(), m_SpecificInt(10),
                            m_Value())));

  // Match specific fifth argument.
  EXPECT_TRUE(
      match(Intrinsic5,
            m_Intrinsic<Intrinsic::instrprof_increment_step>(
                m_Value(), m_Value(), m_Value(), m_Value(), m_SpecificInt(3))));
  EXPECT_FALSE(
      match(Intrinsic5, m_Intrinsic<Intrinsic::instrprof_increment_step>(
                            m_Value(), m_Value(), m_Value(), m_Value(),
                            m_SpecificInt(10))));
}

template <typename T> struct MutableConstTest : PatternMatchTest { };

typedef ::testing::Types<std::tuple<Value*, Instruction*>,
                         std::tuple<const Value*, const Instruction *>>
    MutableConstTestTypes;
TYPED_TEST_CASE(MutableConstTest, MutableConstTestTypes);

TYPED_TEST(MutableConstTest, ICmp) {
  auto &IRB = PatternMatchTest::IRB;

  typedef std::tuple_element_t<0, TypeParam> ValueType;
  typedef std::tuple_element_t<1, TypeParam> InstructionType;

  Value *L = IRB.getInt32(1);
  Value *R = IRB.getInt32(2);
  ICmpInst::Predicate Pred = ICmpInst::ICMP_UGT;

  ValueType MatchL;
  ValueType MatchR;
  ICmpInst::Predicate MatchPred;

  EXPECT_TRUE(m_ICmp(MatchPred, m_Value(MatchL), m_Value(MatchR))
              .match((InstructionType)IRB.CreateICmp(Pred, L, R)));
  EXPECT_EQ(L, MatchL);
  EXPECT_EQ(R, MatchR);
}

} // anonymous namespace.
