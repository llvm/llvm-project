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

TEST_F(PatternMatchTest, Not) {
  Value *C1 = IRB.getInt32(1);
  Value *C2 = IRB.getInt32(2);
  Value *C3 = IRB.getInt32(3);
  Instruction *Not = BinaryOperator::CreateXor(C1, C2);

  // When `m_Not` does not match the `not` itself,
  // it should not try to apply the inner matcher.
  Value *Val = C3;
  EXPECT_FALSE(m_Not(m_Value(Val)).match(Not));
  EXPECT_EQ(Val, C3);
  Not->deleteValue();
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
  Type *VecTy = FixedVectorType::get(IRB.getInt8Ty(), 2);
  Type *i32 = IRB.getInt32Ty();
  Type *i32VecTy = FixedVectorType::get(i32, 2);

  Value *Val = IRB.CreateAdd(IRB.getInt8(0), IRB.getInt8(1));
  Value *Val2 = IRB.CreateAdd(Val, IRB.getInt8(3));

  SmallVector<Constant *, 2> VecElemIdxs;
  VecElemIdxs.push_back(ConstantInt::get(i32, 0));
  VecElemIdxs.push_back(ConstantInt::get(i32, 2));
  auto *IdxVec = ConstantVector::get(VecElemIdxs);

  Value *VI1 = IRB.CreateInsertElement(VecTy, IRB.getInt8(1), (uint64_t)0);
  Value *VI2 = IRB.CreateInsertElement(VI1, Val2, Val);
  Value *VI3 = IRB.CreateInsertElement(VI1, Val2, (uint64_t)1);
  Value *VI4 = IRB.CreateInsertElement(VI1, IRB.getInt8(2), Val);

  Value *EX1 = IRB.CreateExtractElement(VI4, Val);
  Value *EX2 = IRB.CreateExtractElement(VI4, (uint64_t)0);
  Value *EX3 = IRB.CreateExtractElement(IdxVec, (uint64_t)1);

  Constant *Zero = ConstantAggregateZero::get(i32VecTy);
  SmallVector<int, 16> ZeroMask;
  ShuffleVectorInst::getShuffleMask(Zero, ZeroMask);

  Value *SI1 = IRB.CreateShuffleVector(VI1, ZeroMask);
  Value *SI2 = IRB.CreateShuffleVector(VI3, VI4, IdxVec);
  Value *SI3 = IRB.CreateShuffleVector(VI3, ZeroMask);
  Value *SI4 = IRB.CreateShuffleVector(VI4, ZeroMask);

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

TEST_F(PatternMatchTest, UndefPoisonMix) {
  Type *ScalarTy = IRB.getInt8Ty();
  ArrayType *ArrTy = ArrayType::get(ScalarTy, 2);
  StructType *StTy = StructType::get(ScalarTy, ScalarTy);
  StructType *StTy2 = StructType::get(ScalarTy, StTy);
  StructType *StTy3 = StructType::get(StTy, ScalarTy);
  Constant *Zero = ConstantInt::getNullValue(ScalarTy);
  UndefValue *U = UndefValue::get(ScalarTy);
  UndefValue *P = PoisonValue::get(ScalarTy);

  EXPECT_TRUE(match(ConstantVector::get({U, P}), m_Undef()));
  EXPECT_TRUE(match(ConstantVector::get({P, U}), m_Undef()));

  EXPECT_TRUE(match(ConstantArray::get(ArrTy, {U, P}), m_Undef()));
  EXPECT_TRUE(match(ConstantArray::get(ArrTy, {P, U}), m_Undef()));

  auto *UP = ConstantStruct::get(StTy, {U, P});
  EXPECT_TRUE(match(ConstantStruct::get(StTy2, {U, UP}), m_Undef()));
  EXPECT_TRUE(match(ConstantStruct::get(StTy2, {P, UP}), m_Undef()));
  EXPECT_TRUE(match(ConstantStruct::get(StTy3, {UP, U}), m_Undef()));
  EXPECT_TRUE(match(ConstantStruct::get(StTy3, {UP, P}), m_Undef()));

  EXPECT_FALSE(match(ConstantStruct::get(StTy, {U, Zero}), m_Undef()));
  EXPECT_FALSE(match(ConstantStruct::get(StTy, {Zero, U}), m_Undef()));
  EXPECT_FALSE(match(ConstantStruct::get(StTy, {P, Zero}), m_Undef()));
  EXPECT_FALSE(match(ConstantStruct::get(StTy, {Zero, P}), m_Undef()));

  EXPECT_FALSE(match(ConstantStruct::get(StTy2, {Zero, UP}), m_Undef()));
  EXPECT_FALSE(match(ConstantStruct::get(StTy3, {UP, Zero}), m_Undef()));
}

TEST_F(PatternMatchTest, VectorUndefInt) {
  Type *ScalarTy = IRB.getInt8Ty();
  Type *VectorTy = FixedVectorType::get(ScalarTy, 4);
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
  EXPECT_TRUE(C->isZero());
  C = nullptr;
  EXPECT_TRUE(match(ScalarZero, m_APIntForbidUndef(C)));
  EXPECT_TRUE(C->isZero());
  C = nullptr;
  EXPECT_TRUE(match(ScalarZero, m_APIntAllowUndef(C)));
  EXPECT_TRUE(C->isZero());
  C = nullptr;
  EXPECT_TRUE(match(VectorZero, m_APInt(C)));
  EXPECT_TRUE(C->isZero());
  C = nullptr;
  EXPECT_TRUE(match(VectorZero, m_APIntForbidUndef(C)));
  EXPECT_TRUE(C->isZero());
  C = nullptr;
  EXPECT_TRUE(match(VectorZero, m_APIntAllowUndef(C)));
  EXPECT_TRUE(C->isZero());

  // Whether splats with undef can be matched depends on the matcher.
  EXPECT_FALSE(match(VectorZeroUndef, m_APInt(C)));
  EXPECT_FALSE(match(VectorZeroUndef, m_APIntForbidUndef(C)));
  C = nullptr;
  EXPECT_TRUE(match(VectorZeroUndef, m_APIntAllowUndef(C)));
  EXPECT_TRUE(C->isZero());
}

TEST_F(PatternMatchTest, VectorUndefFloat) {
  Type *ScalarTy = IRB.getFloatTy();
  Type *VectorTy = FixedVectorType::get(ScalarTy, 4);
  Constant *ScalarUndef = UndefValue::get(ScalarTy);
  Constant *VectorUndef = UndefValue::get(VectorTy);
  Constant *ScalarZero = Constant::getNullValue(ScalarTy);
  Constant *VectorZero = Constant::getNullValue(VectorTy);
  Constant *ScalarPosInf = ConstantFP::getInfinity(ScalarTy, false);
  Constant *ScalarNegInf = ConstantFP::getInfinity(ScalarTy, true);
  Constant *ScalarNaN = ConstantFP::getNaN(ScalarTy, true);

  Constant *VectorZeroUndef =
      ConstantVector::get({ScalarUndef, ScalarZero, ScalarUndef, ScalarZero});

  Constant *VectorInfUndef = ConstantVector::get(
      {ScalarPosInf, ScalarNegInf, ScalarUndef, ScalarPosInf});

  Constant *VectorNaNUndef =
      ConstantVector::get({ScalarUndef, ScalarNaN, ScalarNaN, ScalarNaN});

  EXPECT_TRUE(match(ScalarUndef, m_Undef()));
  EXPECT_TRUE(match(VectorUndef, m_Undef()));
  EXPECT_FALSE(match(ScalarZero, m_Undef()));
  EXPECT_FALSE(match(VectorZero, m_Undef()));
  EXPECT_FALSE(match(VectorZeroUndef, m_Undef()));
  EXPECT_FALSE(match(VectorInfUndef, m_Undef()));
  EXPECT_FALSE(match(VectorNaNUndef, m_Undef()));

  EXPECT_FALSE(match(ScalarUndef, m_AnyZeroFP()));
  EXPECT_FALSE(match(VectorUndef, m_AnyZeroFP()));
  EXPECT_TRUE(match(ScalarZero, m_AnyZeroFP()));
  EXPECT_TRUE(match(VectorZero, m_AnyZeroFP()));
  EXPECT_TRUE(match(VectorZeroUndef, m_AnyZeroFP()));
  EXPECT_FALSE(match(VectorInfUndef, m_AnyZeroFP()));
  EXPECT_FALSE(match(VectorNaNUndef, m_AnyZeroFP()));

  EXPECT_FALSE(match(ScalarUndef, m_NaN()));
  EXPECT_FALSE(match(VectorUndef, m_NaN()));
  EXPECT_FALSE(match(VectorZeroUndef, m_NaN()));
  EXPECT_FALSE(match(ScalarPosInf, m_NaN()));
  EXPECT_FALSE(match(ScalarNegInf, m_NaN()));
  EXPECT_TRUE(match(ScalarNaN, m_NaN()));
  EXPECT_FALSE(match(VectorInfUndef, m_NaN()));
  EXPECT_TRUE(match(VectorNaNUndef, m_NaN()));

  EXPECT_FALSE(match(ScalarUndef, m_NonNaN()));
  EXPECT_FALSE(match(VectorUndef, m_NonNaN()));
  EXPECT_TRUE(match(VectorZeroUndef, m_NonNaN()));
  EXPECT_TRUE(match(ScalarPosInf, m_NonNaN()));
  EXPECT_TRUE(match(ScalarNegInf, m_NonNaN()));
  EXPECT_FALSE(match(ScalarNaN, m_NonNaN()));
  EXPECT_TRUE(match(VectorInfUndef, m_NonNaN()));
  EXPECT_FALSE(match(VectorNaNUndef, m_NonNaN()));

  EXPECT_FALSE(match(ScalarUndef, m_Inf()));
  EXPECT_FALSE(match(VectorUndef, m_Inf()));
  EXPECT_FALSE(match(VectorZeroUndef, m_Inf()));
  EXPECT_TRUE(match(ScalarPosInf, m_Inf()));
  EXPECT_TRUE(match(ScalarNegInf, m_Inf()));
  EXPECT_FALSE(match(ScalarNaN, m_Inf()));
  EXPECT_TRUE(match(VectorInfUndef, m_Inf()));
  EXPECT_FALSE(match(VectorNaNUndef, m_Inf()));

  EXPECT_FALSE(match(ScalarUndef, m_NonInf()));
  EXPECT_FALSE(match(VectorUndef, m_NonInf()));
  EXPECT_TRUE(match(VectorZeroUndef, m_NonInf()));
  EXPECT_FALSE(match(ScalarPosInf, m_NonInf()));
  EXPECT_FALSE(match(ScalarNegInf, m_NonInf()));
  EXPECT_TRUE(match(ScalarNaN, m_NonInf()));
  EXPECT_FALSE(match(VectorInfUndef, m_NonInf()));
  EXPECT_TRUE(match(VectorNaNUndef, m_NonInf()));

  EXPECT_FALSE(match(ScalarUndef, m_Finite()));
  EXPECT_FALSE(match(VectorUndef, m_Finite()));
  EXPECT_TRUE(match(VectorZeroUndef, m_Finite()));
  EXPECT_FALSE(match(ScalarPosInf, m_Finite()));
  EXPECT_FALSE(match(ScalarNegInf, m_Finite()));
  EXPECT_FALSE(match(ScalarNaN, m_Finite()));
  EXPECT_FALSE(match(VectorInfUndef, m_Finite()));
  EXPECT_FALSE(match(VectorNaNUndef, m_Finite()));

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
  C = nullptr;
  EXPECT_TRUE(match(VectorZeroUndef, m_Finite(C)));
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

TEST_F(PatternMatchTest, MinMaxIntrinsics) {
  Type *Ty = IRB.getInt32Ty();
  Value *L = ConstantInt::get(Ty, 1);
  Value *R = ConstantInt::get(Ty, 2);
  Value *MatchL, *MatchR;

  // Check for intrinsic ID match and capture of operands.
  EXPECT_TRUE(m_SMax(m_Value(MatchL), m_Value(MatchR))
                  .match(IRB.CreateBinaryIntrinsic(Intrinsic::smax, L, R)));
  EXPECT_EQ(L, MatchL);
  EXPECT_EQ(R, MatchR);

  EXPECT_TRUE(m_SMin(m_Value(MatchL), m_Value(MatchR))
                  .match(IRB.CreateBinaryIntrinsic(Intrinsic::smin, L, R)));
  EXPECT_EQ(L, MatchL);
  EXPECT_EQ(R, MatchR);

  EXPECT_TRUE(m_UMax(m_Value(MatchL), m_Value(MatchR))
                  .match(IRB.CreateBinaryIntrinsic(Intrinsic::umax, L, R)));
  EXPECT_EQ(L, MatchL);
  EXPECT_EQ(R, MatchR);

  EXPECT_TRUE(m_UMin(m_Value(MatchL), m_Value(MatchR))
                  .match(IRB.CreateBinaryIntrinsic(Intrinsic::umin, L, R)));
  EXPECT_EQ(L, MatchL);
  EXPECT_EQ(R, MatchR);

  // Check for intrinsic ID mismatch.
  EXPECT_FALSE(m_SMax(m_Value(MatchL), m_Value(MatchR))
                  .match(IRB.CreateBinaryIntrinsic(Intrinsic::smin, L, R)));
  EXPECT_FALSE(m_SMin(m_Value(MatchL), m_Value(MatchR))
                  .match(IRB.CreateBinaryIntrinsic(Intrinsic::umax, L, R)));
  EXPECT_FALSE(m_UMax(m_Value(MatchL), m_Value(MatchR))
                  .match(IRB.CreateBinaryIntrinsic(Intrinsic::umin, L, R)));
  EXPECT_FALSE(m_UMin(m_Value(MatchL), m_Value(MatchR))
                  .match(IRB.CreateBinaryIntrinsic(Intrinsic::smax, L, R)));
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

namespace {

struct is_unsigned_zero_pred {
  bool isValue(const APInt &C) { return C.isZero(); }
};

struct is_float_zero_pred {
  bool isValue(const APFloat &C) { return C.isZero(); }
};

template <typename T> struct always_true_pred {
  bool isValue(const T &) { return true; }
};

template <typename T> struct always_false_pred {
  bool isValue(const T &) { return false; }
};

struct is_unsigned_max_pred {
  bool isValue(const APInt &C) { return C.isMaxValue(); }
};

struct is_float_nan_pred {
  bool isValue(const APFloat &C) { return C.isNaN(); }
};

} // namespace

TEST_F(PatternMatchTest, ConstantPredicateType) {

  // Scalar integer
  APInt U32Max = APInt::getAllOnes(32);
  APInt U32Zero = APInt::getZero(32);
  APInt U32DeadBeef(32, 0xDEADBEEF);

  Type *U32Ty = Type::getInt32Ty(Ctx);

  Constant *CU32Max = Constant::getIntegerValue(U32Ty, U32Max);
  Constant *CU32Zero = Constant::getIntegerValue(U32Ty, U32Zero);
  Constant *CU32DeadBeef = Constant::getIntegerValue(U32Ty, U32DeadBeef);

  EXPECT_TRUE(match(CU32Max, cst_pred_ty<is_unsigned_max_pred>()));
  EXPECT_FALSE(match(CU32Max, cst_pred_ty<is_unsigned_zero_pred>()));
  EXPECT_TRUE(match(CU32Max, cst_pred_ty<always_true_pred<APInt>>()));
  EXPECT_FALSE(match(CU32Max, cst_pred_ty<always_false_pred<APInt>>()));

  EXPECT_FALSE(match(CU32Zero, cst_pred_ty<is_unsigned_max_pred>()));
  EXPECT_TRUE(match(CU32Zero, cst_pred_ty<is_unsigned_zero_pred>()));
  EXPECT_TRUE(match(CU32Zero, cst_pred_ty<always_true_pred<APInt>>()));
  EXPECT_FALSE(match(CU32Zero, cst_pred_ty<always_false_pred<APInt>>()));

  EXPECT_FALSE(match(CU32DeadBeef, cst_pred_ty<is_unsigned_max_pred>()));
  EXPECT_FALSE(match(CU32DeadBeef, cst_pred_ty<is_unsigned_zero_pred>()));
  EXPECT_TRUE(match(CU32DeadBeef, cst_pred_ty<always_true_pred<APInt>>()));
  EXPECT_FALSE(match(CU32DeadBeef, cst_pred_ty<always_false_pred<APInt>>()));

  // Scalar float
  APFloat F32NaN = APFloat::getNaN(APFloat::IEEEsingle());
  APFloat F32Zero = APFloat::getZero(APFloat::IEEEsingle());
  APFloat F32Pi(3.14f);

  Type *F32Ty = Type::getFloatTy(Ctx);

  Constant *CF32NaN = ConstantFP::get(F32Ty, F32NaN);
  Constant *CF32Zero = ConstantFP::get(F32Ty, F32Zero);
  Constant *CF32Pi = ConstantFP::get(F32Ty, F32Pi);

  EXPECT_TRUE(match(CF32NaN, cstfp_pred_ty<is_float_nan_pred>()));
  EXPECT_FALSE(match(CF32NaN, cstfp_pred_ty<is_float_zero_pred>()));
  EXPECT_TRUE(match(CF32NaN, cstfp_pred_ty<always_true_pred<APFloat>>()));
  EXPECT_FALSE(match(CF32NaN, cstfp_pred_ty<always_false_pred<APFloat>>()));

  EXPECT_FALSE(match(CF32Zero, cstfp_pred_ty<is_float_nan_pred>()));
  EXPECT_TRUE(match(CF32Zero, cstfp_pred_ty<is_float_zero_pred>()));
  EXPECT_TRUE(match(CF32Zero, cstfp_pred_ty<always_true_pred<APFloat>>()));
  EXPECT_FALSE(match(CF32Zero, cstfp_pred_ty<always_false_pred<APFloat>>()));

  EXPECT_FALSE(match(CF32Pi, cstfp_pred_ty<is_float_nan_pred>()));
  EXPECT_FALSE(match(CF32Pi, cstfp_pred_ty<is_float_zero_pred>()));
  EXPECT_TRUE(match(CF32Pi, cstfp_pred_ty<always_true_pred<APFloat>>()));
  EXPECT_FALSE(match(CF32Pi, cstfp_pred_ty<always_false_pred<APFloat>>()));

  auto FixedEC = ElementCount::getFixed(4);
  auto ScalableEC = ElementCount::getScalable(4);

  // Vector splat

  for (auto EC : {FixedEC, ScalableEC}) {
    // integer

    Constant *CSplatU32Max = ConstantVector::getSplat(EC, CU32Max);
    Constant *CSplatU32Zero = ConstantVector::getSplat(EC, CU32Zero);
    Constant *CSplatU32DeadBeef = ConstantVector::getSplat(EC, CU32DeadBeef);

    EXPECT_TRUE(match(CSplatU32Max, cst_pred_ty<is_unsigned_max_pred>()));
    EXPECT_FALSE(match(CSplatU32Max, cst_pred_ty<is_unsigned_zero_pred>()));
    EXPECT_TRUE(match(CSplatU32Max, cst_pred_ty<always_true_pred<APInt>>()));
    EXPECT_FALSE(match(CSplatU32Max, cst_pred_ty<always_false_pred<APInt>>()));

    EXPECT_FALSE(match(CSplatU32Zero, cst_pred_ty<is_unsigned_max_pred>()));
    EXPECT_TRUE(match(CSplatU32Zero, cst_pred_ty<is_unsigned_zero_pred>()));
    EXPECT_TRUE(match(CSplatU32Zero, cst_pred_ty<always_true_pred<APInt>>()));
    EXPECT_FALSE(match(CSplatU32Zero, cst_pred_ty<always_false_pred<APInt>>()));

    EXPECT_FALSE(match(CSplatU32DeadBeef, cst_pred_ty<is_unsigned_max_pred>()));
    EXPECT_FALSE(
        match(CSplatU32DeadBeef, cst_pred_ty<is_unsigned_zero_pred>()));
    EXPECT_TRUE(
        match(CSplatU32DeadBeef, cst_pred_ty<always_true_pred<APInt>>()));
    EXPECT_FALSE(
        match(CSplatU32DeadBeef, cst_pred_ty<always_false_pred<APInt>>()));

    // float

    Constant *CSplatF32NaN = ConstantVector::getSplat(EC, CF32NaN);
    Constant *CSplatF32Zero = ConstantVector::getSplat(EC, CF32Zero);
    Constant *CSplatF32Pi = ConstantVector::getSplat(EC, CF32Pi);

    EXPECT_TRUE(match(CSplatF32NaN, cstfp_pred_ty<is_float_nan_pred>()));
    EXPECT_FALSE(match(CSplatF32NaN, cstfp_pred_ty<is_float_zero_pred>()));
    EXPECT_TRUE(
        match(CSplatF32NaN, cstfp_pred_ty<always_true_pred<APFloat>>()));
    EXPECT_FALSE(
        match(CSplatF32NaN, cstfp_pred_ty<always_false_pred<APFloat>>()));

    EXPECT_FALSE(match(CSplatF32Zero, cstfp_pred_ty<is_float_nan_pred>()));
    EXPECT_TRUE(match(CSplatF32Zero, cstfp_pred_ty<is_float_zero_pred>()));
    EXPECT_TRUE(
        match(CSplatF32Zero, cstfp_pred_ty<always_true_pred<APFloat>>()));
    EXPECT_FALSE(
        match(CSplatF32Zero, cstfp_pred_ty<always_false_pred<APFloat>>()));

    EXPECT_FALSE(match(CSplatF32Pi, cstfp_pred_ty<is_float_nan_pred>()));
    EXPECT_FALSE(match(CSplatF32Pi, cstfp_pred_ty<is_float_zero_pred>()));
    EXPECT_TRUE(match(CSplatF32Pi, cstfp_pred_ty<always_true_pred<APFloat>>()));
    EXPECT_FALSE(
        match(CSplatF32Pi, cstfp_pred_ty<always_false_pred<APFloat>>()));
  }

  // Int arbitrary vector

  Constant *CMixedU32 = ConstantVector::get({CU32Max, CU32Zero, CU32DeadBeef});
  Constant *CU32Undef = UndefValue::get(U32Ty);
  Constant *CU32MaxWithUndef =
      ConstantVector::get({CU32Undef, CU32Max, CU32Undef});

  EXPECT_FALSE(match(CMixedU32, cst_pred_ty<is_unsigned_max_pred>()));
  EXPECT_FALSE(match(CMixedU32, cst_pred_ty<is_unsigned_zero_pred>()));
  EXPECT_TRUE(match(CMixedU32, cst_pred_ty<always_true_pred<APInt>>()));
  EXPECT_FALSE(match(CMixedU32, cst_pred_ty<always_false_pred<APInt>>()));

  EXPECT_TRUE(match(CU32MaxWithUndef, cst_pred_ty<is_unsigned_max_pred>()));
  EXPECT_FALSE(match(CU32MaxWithUndef, cst_pred_ty<is_unsigned_zero_pred>()));
  EXPECT_TRUE(match(CU32MaxWithUndef, cst_pred_ty<always_true_pred<APInt>>()));
  EXPECT_FALSE(
      match(CU32MaxWithUndef, cst_pred_ty<always_false_pred<APInt>>()));

  // Float arbitrary vector

  Constant *CMixedF32 = ConstantVector::get({CF32NaN, CF32Zero, CF32Pi});
  Constant *CF32Undef = UndefValue::get(F32Ty);
  Constant *CF32NaNWithUndef =
      ConstantVector::get({CF32Undef, CF32NaN, CF32Undef});

  EXPECT_FALSE(match(CMixedF32, cstfp_pred_ty<is_float_nan_pred>()));
  EXPECT_FALSE(match(CMixedF32, cstfp_pred_ty<is_float_zero_pred>()));
  EXPECT_TRUE(match(CMixedF32, cstfp_pred_ty<always_true_pred<APFloat>>()));
  EXPECT_FALSE(match(CMixedF32, cstfp_pred_ty<always_false_pred<APFloat>>()));

  EXPECT_TRUE(match(CF32NaNWithUndef, cstfp_pred_ty<is_float_nan_pred>()));
  EXPECT_FALSE(match(CF32NaNWithUndef, cstfp_pred_ty<is_float_zero_pred>()));
  EXPECT_TRUE(
      match(CF32NaNWithUndef, cstfp_pred_ty<always_true_pred<APFloat>>()));
  EXPECT_FALSE(
      match(CF32NaNWithUndef, cstfp_pred_ty<always_false_pred<APFloat>>()));
}

TEST_F(PatternMatchTest, InsertValue) {
  Type *StructTy = StructType::create(IRB.getContext(),
                                      {IRB.getInt32Ty(), IRB.getInt64Ty()});
  Value *Ins0 =
      IRB.CreateInsertValue(UndefValue::get(StructTy), IRB.getInt32(20), 0);
  Value *Ins1 = IRB.CreateInsertValue(Ins0, IRB.getInt64(90), 1);

  EXPECT_TRUE(match(Ins0, m_InsertValue<0>(m_Value(), m_Value())));
  EXPECT_FALSE(match(Ins0, m_InsertValue<1>(m_Value(), m_Value())));
  EXPECT_FALSE(match(Ins1, m_InsertValue<0>(m_Value(), m_Value())));
  EXPECT_TRUE(match(Ins1, m_InsertValue<1>(m_Value(), m_Value())));

  EXPECT_TRUE(match(Ins0, m_InsertValue<0>(m_Undef(), m_SpecificInt(20))));
  EXPECT_FALSE(match(Ins0, m_InsertValue<0>(m_Undef(), m_SpecificInt(0))));

  EXPECT_TRUE(
      match(Ins1, m_InsertValue<1>(m_InsertValue<0>(m_Value(), m_Value()),
                                   m_SpecificInt(90))));
  EXPECT_FALSE(match(IRB.getInt64(99), m_InsertValue<0>(m_Value(), m_Value())));
}

TEST_F(PatternMatchTest, LogicalSelects) {
  Value *Alloca = IRB.CreateAlloca(IRB.getInt1Ty());
  Value *X = IRB.CreateLoad(IRB.getInt1Ty(), Alloca);
  Value *Y = IRB.CreateLoad(IRB.getInt1Ty(), Alloca);
  Constant *T = IRB.getInt1(true);
  Constant *F = IRB.getInt1(false);
  Value *And = IRB.CreateSelect(X, Y, F);
  Value *Or = IRB.CreateSelect(X, T, Y);

  // Logical and:
  // Check basic no-capture logic - opcode and constant must match.
  EXPECT_TRUE(match(And, m_LogicalAnd(m_Value(), m_Value())));
  EXPECT_TRUE(match(And, m_c_LogicalAnd(m_Value(), m_Value())));
  EXPECT_FALSE(match(And, m_LogicalOr(m_Value(), m_Value())));
  EXPECT_FALSE(match(And, m_c_LogicalOr(m_Value(), m_Value())));

  // Check with captures.
  EXPECT_TRUE(match(And, m_LogicalAnd(m_Specific(X), m_Value())));
  EXPECT_TRUE(match(And, m_LogicalAnd(m_Value(), m_Specific(Y))));
  EXPECT_TRUE(match(And, m_LogicalAnd(m_Specific(X), m_Specific(Y))));

  EXPECT_FALSE(match(And, m_LogicalAnd(m_Specific(Y), m_Value())));
  EXPECT_FALSE(match(And, m_LogicalAnd(m_Value(), m_Specific(X))));
  EXPECT_FALSE(match(And, m_LogicalAnd(m_Specific(Y), m_Specific(X))));

  EXPECT_FALSE(match(And, m_LogicalAnd(m_Specific(X), m_Specific(X))));
  EXPECT_FALSE(match(And, m_LogicalAnd(m_Specific(Y), m_Specific(Y))));

  // Check captures for commutative match.
  EXPECT_TRUE(match(And, m_c_LogicalAnd(m_Specific(X), m_Value())));
  EXPECT_TRUE(match(And, m_c_LogicalAnd(m_Value(), m_Specific(Y))));
  EXPECT_TRUE(match(And, m_c_LogicalAnd(m_Specific(X), m_Specific(Y))));

  EXPECT_TRUE(match(And, m_c_LogicalAnd(m_Specific(Y), m_Value())));
  EXPECT_TRUE(match(And, m_c_LogicalAnd(m_Value(), m_Specific(X))));
  EXPECT_TRUE(match(And, m_c_LogicalAnd(m_Specific(Y), m_Specific(X))));

  EXPECT_FALSE(match(And, m_c_LogicalAnd(m_Specific(X), m_Specific(X))));
  EXPECT_FALSE(match(And, m_c_LogicalAnd(m_Specific(Y), m_Specific(Y))));

  // Logical or:
  // Check basic no-capture logic - opcode and constant must match.
  EXPECT_TRUE(match(Or, m_LogicalOr(m_Value(), m_Value())));
  EXPECT_TRUE(match(Or, m_c_LogicalOr(m_Value(), m_Value())));
  EXPECT_FALSE(match(Or, m_LogicalAnd(m_Value(), m_Value())));
  EXPECT_FALSE(match(Or, m_c_LogicalAnd(m_Value(), m_Value())));

  // Check with captures.
  EXPECT_TRUE(match(Or, m_LogicalOr(m_Specific(X), m_Value())));
  EXPECT_TRUE(match(Or, m_LogicalOr(m_Value(), m_Specific(Y))));
  EXPECT_TRUE(match(Or, m_LogicalOr(m_Specific(X), m_Specific(Y))));

  EXPECT_FALSE(match(Or, m_LogicalOr(m_Specific(Y), m_Value())));
  EXPECT_FALSE(match(Or, m_LogicalOr(m_Value(), m_Specific(X))));
  EXPECT_FALSE(match(Or, m_LogicalOr(m_Specific(Y), m_Specific(X))));

  EXPECT_FALSE(match(Or, m_LogicalOr(m_Specific(X), m_Specific(X))));
  EXPECT_FALSE(match(Or, m_LogicalOr(m_Specific(Y), m_Specific(Y))));

  // Check captures for commutative match.
  EXPECT_TRUE(match(Or, m_c_LogicalOr(m_Specific(X), m_Value())));
  EXPECT_TRUE(match(Or, m_c_LogicalOr(m_Value(), m_Specific(Y))));
  EXPECT_TRUE(match(Or, m_c_LogicalOr(m_Specific(X), m_Specific(Y))));

  EXPECT_TRUE(match(Or, m_c_LogicalOr(m_Specific(Y), m_Value())));
  EXPECT_TRUE(match(Or, m_c_LogicalOr(m_Value(), m_Specific(X))));
  EXPECT_TRUE(match(Or, m_c_LogicalOr(m_Specific(Y), m_Specific(X))));

  EXPECT_FALSE(match(Or, m_c_LogicalOr(m_Specific(X), m_Specific(X))));
  EXPECT_FALSE(match(Or, m_c_LogicalOr(m_Specific(Y), m_Specific(Y))));
}

TEST_F(PatternMatchTest, VectorLogicalSelects) {
  Type *i1 = IRB.getInt1Ty();
  Type *v3i1 = FixedVectorType::get(i1, 3);

  Value *Alloca = IRB.CreateAlloca(i1);
  Value *AllocaVec = IRB.CreateAlloca(v3i1);
  Value *Scalar = IRB.CreateLoad(i1, Alloca);
  Value *Vector = IRB.CreateLoad(v3i1, AllocaVec);
  Constant *F = Constant::getNullValue(v3i1);
  Constant *T = Constant::getAllOnesValue(v3i1);

  // select <3 x i1> Vector, <3 x i1> Vector, <3 x i1> <i1 0, i1 0, i1 0>
  Value *VecAnd = IRB.CreateSelect(Vector, Vector, F);

  // select i1 Scalar, <3 x i1> Vector, <3 x i1> <i1 0, i1 0, i1 0>
  Value *MixedTypeAnd = IRB.CreateSelect(Scalar, Vector, F);

  // select <3 x i1> Vector, <3 x i1> <i1 1, i1 1, i1 1>, <3 x i1> Vector
  Value *VecOr = IRB.CreateSelect(Vector, T, Vector);

  // select i1 Scalar, <3 x i1> <i1 1, i1 1, i1 1>, <3 x i1> Vector
  Value *MixedTypeOr = IRB.CreateSelect(Scalar, T, Vector);

  // We allow matching a real vector logical select,
  // but not a scalar select of vector bools.
  EXPECT_TRUE(match(VecAnd, m_LogicalAnd(m_Value(), m_Value())));
  EXPECT_FALSE(match(MixedTypeAnd, m_LogicalAnd(m_Value(), m_Value())));
  EXPECT_TRUE(match(VecOr, m_LogicalOr(m_Value(), m_Value())));
  EXPECT_FALSE(match(MixedTypeOr, m_LogicalOr(m_Value(), m_Value())));
}

TEST_F(PatternMatchTest, VScale) {
  DataLayout DL = M->getDataLayout();

  Type *VecTy = ScalableVectorType::get(IRB.getInt8Ty(), 1);
  Value *NullPtrVec =
      Constant::getNullValue(PointerType::getUnqual(VecTy->getContext()));
  Value *GEP = IRB.CreateGEP(VecTy, NullPtrVec, IRB.getInt64(1));
  Value *PtrToInt = IRB.CreatePtrToInt(GEP, DL.getIntPtrType(GEP->getType()));
  EXPECT_TRUE(match(PtrToInt, m_VScale()));

  Type *VecTy2 = ScalableVectorType::get(IRB.getInt8Ty(), 2);
  Value *NullPtrVec2 =
      Constant::getNullValue(PointerType::getUnqual(VecTy2->getContext()));
  Value *GEP2 = IRB.CreateGEP(VecTy, NullPtrVec2, IRB.getInt64(1));
  Value *PtrToInt2 =
      IRB.CreatePtrToInt(GEP2, DL.getIntPtrType(GEP2->getType()));
  EXPECT_TRUE(match(PtrToInt2, m_VScale()));
}

TEST_F(PatternMatchTest, NotForbidUndef) {
  Type *ScalarTy = IRB.getInt8Ty();
  Type *VectorTy = FixedVectorType::get(ScalarTy, 3);
  Constant *ScalarUndef = UndefValue::get(ScalarTy);
  Constant *ScalarOnes = Constant::getAllOnesValue(ScalarTy);
  Constant *VectorZero = Constant::getNullValue(VectorTy);
  Constant *VectorOnes = Constant::getAllOnesValue(VectorTy);

  SmallVector<Constant *, 3> MixedElems;
  MixedElems.push_back(ScalarOnes);
  MixedElems.push_back(ScalarOnes);
  MixedElems.push_back(ScalarUndef);
  Constant *VectorMixed = ConstantVector::get(MixedElems);

  Value *Not = IRB.CreateXor(VectorZero, VectorOnes);
  Value *X;
  EXPECT_TRUE(match(Not, m_Not(m_Value())));
  EXPECT_TRUE(match(Not, m_NotForbidUndef(m_Value(X))));
  EXPECT_TRUE(match(X, m_Zero()));

  Value *NotCommute = IRB.CreateXor(VectorOnes, VectorZero);
  Value *Y;
  EXPECT_TRUE(match(NotCommute, m_Not(m_Value())));
  EXPECT_TRUE(match(NotCommute, m_NotForbidUndef(m_Value(Y))));
  EXPECT_TRUE(match(Y, m_Zero()));

  Value *NotWithUndefs = IRB.CreateXor(VectorZero, VectorMixed);
  EXPECT_TRUE(match(NotWithUndefs, m_Not(m_Value())));
  EXPECT_FALSE(match(NotWithUndefs, m_NotForbidUndef(m_Value())));

  Value *NotWithUndefsCommute = IRB.CreateXor(VectorMixed, VectorZero);
  EXPECT_TRUE(match(NotWithUndefsCommute, m_Not(m_Value())));
  EXPECT_FALSE(match(NotWithUndefsCommute, m_NotForbidUndef(m_Value(X))));
}

template <typename T> struct MutableConstTest : PatternMatchTest { };

typedef ::testing::Types<std::tuple<Value*, Instruction*>,
                         std::tuple<const Value*, const Instruction *>>
    MutableConstTestTypes;
TYPED_TEST_SUITE(MutableConstTest, MutableConstTestTypes, );

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

TEST_F(PatternMatchTest, ConstExpr) {
  Constant *G =
      M->getOrInsertGlobal("dummy", PointerType::getUnqual(IRB.getInt32Ty()));
  Constant *S = ConstantExpr::getPtrToInt(G, IRB.getInt32Ty());
  Type *VecTy = FixedVectorType::get(IRB.getInt32Ty(), 2);
  PoisonValue *P = PoisonValue::get(VecTy);
  Constant *V = ConstantExpr::getInsertElement(P, S, IRB.getInt32(0));

  // The match succeeds on a constant that is a constant expression itself
  // or a constant that contains a constant expression.
  EXPECT_TRUE(match(S, m_ConstantExpr()));
  EXPECT_TRUE(match(V, m_ConstantExpr()));
}

} // anonymous namespace.
