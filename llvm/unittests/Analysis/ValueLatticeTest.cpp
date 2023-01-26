//===- ValueLatticeTest.cpp - ScalarEvolution unit tests --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/ValueLattice.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/ConstantRange.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "gtest/gtest.h"

namespace llvm {
namespace {

// We use this fixture to ensure that we clean up ScalarEvolution before
// deleting the PassManager.
class ValueLatticeTest : public testing::Test {
protected:
  LLVMContext Context;
  DataLayout DL = DataLayout("");
};

TEST_F(ValueLatticeTest, ValueLatticeGetters) {
  auto I32Ty = IntegerType::get(Context, 32);
  auto *C1 = ConstantInt::get(I32Ty, 1);

  EXPECT_TRUE(ValueLatticeElement::get(C1).isConstantRange());
  EXPECT_TRUE(
      ValueLatticeElement::getRange({C1->getValue()}).isConstantRange());
  EXPECT_TRUE(ValueLatticeElement::getOverdefined().isOverdefined());

  auto FloatTy = Type::getFloatTy(Context);
  auto *C2 = ConstantFP::get(FloatTy, 1.1);
  EXPECT_TRUE(ValueLatticeElement::get(C2).isConstant());
  EXPECT_TRUE(ValueLatticeElement::getNot(C2).isNotConstant());
}

TEST_F(ValueLatticeTest, MarkConstantRange) {
  auto LV1 =
      ValueLatticeElement::getRange({APInt(32, 10, true), APInt(32, 20, true)});

  // Test markConstantRange() with an equal range.
  EXPECT_FALSE(
      LV1.markConstantRange({APInt(32, 10, true), APInt(32, 20, true)}));

  // Test markConstantRange() with supersets of existing range.
  EXPECT_TRUE(LV1.markConstantRange({APInt(32, 5, true), APInt(32, 20, true)}));
  EXPECT_EQ(LV1.getConstantRange().getLower().getLimitedValue(), 5U);
  EXPECT_EQ(LV1.getConstantRange().getUpper().getLimitedValue(), 20U);
  EXPECT_TRUE(LV1.markConstantRange({APInt(32, 5, true), APInt(32, 23, true)}));
  EXPECT_EQ(LV1.getConstantRange().getLower().getLimitedValue(), 5U);
  EXPECT_EQ(LV1.getConstantRange().getUpper().getLimitedValue(), 23U);
}

TEST_F(ValueLatticeTest, MergeIn) {
  auto I32Ty = IntegerType::get(Context, 32);
  auto *C1 = ConstantInt::get(I32Ty, 1);

  // Merge to lattice values with equal integer constant.
  auto LV1 = ValueLatticeElement::get(C1);
  EXPECT_FALSE(LV1.mergeIn(ValueLatticeElement::get(C1)));
  EXPECT_TRUE(LV1.isConstantRange());
  EXPECT_EQ(LV1.asConstantInteger()->getLimitedValue(), 1U);

  // Merge LV1 with different integer constant.
  EXPECT_TRUE(
      LV1.mergeIn(ValueLatticeElement::get(ConstantInt::get(I32Ty, 99))));
  EXPECT_TRUE(LV1.isConstantRange());
  EXPECT_EQ(LV1.getConstantRange().getLower().getLimitedValue(), 1U);
  EXPECT_EQ(LV1.getConstantRange().getUpper().getLimitedValue(), 100U);

  // Merge constant range with same constant range.
  EXPECT_FALSE(LV1.mergeIn(LV1));
  EXPECT_TRUE(LV1.isConstantRange());
  EXPECT_EQ(LV1.getConstantRange().getLower().getLimitedValue(), 1U);
  EXPECT_EQ(LV1.getConstantRange().getUpper().getLimitedValue(), 100U);

  // Merge LV1 in undefined value.
  ValueLatticeElement LV2;
  EXPECT_TRUE(LV2.mergeIn(LV1));
  EXPECT_TRUE(LV1.isConstantRange());
  EXPECT_EQ(LV1.getConstantRange().getLower().getLimitedValue(), 1U);
  EXPECT_EQ(LV1.getConstantRange().getUpper().getLimitedValue(), 100U);
  EXPECT_TRUE(LV2.isConstantRange());
  EXPECT_EQ(LV2.getConstantRange().getLower().getLimitedValue(), 1U);
  EXPECT_EQ(LV2.getConstantRange().getUpper().getLimitedValue(), 100U);

  // Merge LV1 with overdefined.
  EXPECT_TRUE(LV1.mergeIn(ValueLatticeElement::getOverdefined()));
  EXPECT_TRUE(LV1.isOverdefined());

  // Merge overdefined with overdefined.
  EXPECT_FALSE(LV1.mergeIn(ValueLatticeElement::getOverdefined()));
  EXPECT_TRUE(LV1.isOverdefined());
}

TEST_F(ValueLatticeTest, getCompareIntegers) {
  auto *I32Ty = IntegerType::get(Context, 32);
  auto *I1Ty = IntegerType::get(Context, 1);
  auto *C1 = ConstantInt::get(I32Ty, 1);
  auto LV1 = ValueLatticeElement::get(C1);

  // Check getCompare for equal integer constants.
  EXPECT_TRUE(LV1.getCompare(CmpInst::ICMP_EQ, I1Ty, LV1, DL)->isOneValue());
  EXPECT_TRUE(LV1.getCompare(CmpInst::ICMP_SGE, I1Ty, LV1, DL)->isOneValue());
  EXPECT_TRUE(LV1.getCompare(CmpInst::ICMP_SLE, I1Ty, LV1, DL)->isOneValue());
  EXPECT_TRUE(LV1.getCompare(CmpInst::ICMP_NE, I1Ty, LV1, DL)->isZeroValue());
  EXPECT_TRUE(LV1.getCompare(CmpInst::ICMP_SLT, I1Ty, LV1, DL)->isZeroValue());
  EXPECT_TRUE(LV1.getCompare(CmpInst::ICMP_SGT, I1Ty, LV1, DL)->isZeroValue());

  auto LV2 =
      ValueLatticeElement::getRange({APInt(32, 10, true), APInt(32, 20, true)});
  // Check getCompare with distinct integer ranges.
  EXPECT_TRUE(LV1.getCompare(CmpInst::ICMP_SLT, I1Ty, LV2, DL)->isOneValue());
  EXPECT_TRUE(LV1.getCompare(CmpInst::ICMP_SLE, I1Ty, LV2, DL)->isOneValue());
  EXPECT_TRUE(LV1.getCompare(CmpInst::ICMP_NE, I1Ty, LV2, DL)->isOneValue());
  EXPECT_TRUE(LV1.getCompare(CmpInst::ICMP_EQ, I1Ty, LV2, DL)->isZeroValue());
  EXPECT_TRUE(LV1.getCompare(CmpInst::ICMP_SGE, I1Ty, LV2, DL)->isZeroValue());
  EXPECT_TRUE(LV1.getCompare(CmpInst::ICMP_SGT, I1Ty, LV2, DL)->isZeroValue());

  auto LV3 =
      ValueLatticeElement::getRange({APInt(32, 15, true), APInt(32, 19, true)});
  // Check getCompare with a subset integer ranges.
  EXPECT_EQ(LV2.getCompare(CmpInst::ICMP_SLT, I1Ty, LV3, DL), nullptr);
  EXPECT_EQ(LV2.getCompare(CmpInst::ICMP_SLE, I1Ty, LV3, DL), nullptr);
  EXPECT_EQ(LV2.getCompare(CmpInst::ICMP_NE, I1Ty, LV3, DL), nullptr);
  EXPECT_EQ(LV2.getCompare(CmpInst::ICMP_EQ, I1Ty, LV3, DL), nullptr);
  EXPECT_EQ(LV2.getCompare(CmpInst::ICMP_SGE, I1Ty, LV3, DL), nullptr);
  EXPECT_EQ(LV2.getCompare(CmpInst::ICMP_SGT, I1Ty, LV3, DL), nullptr);

  auto LV4 =
      ValueLatticeElement::getRange({APInt(32, 15, true), APInt(32, 25, true)});
  // Check getCompare with overlapping integer ranges.
  EXPECT_EQ(LV3.getCompare(CmpInst::ICMP_SLT, I1Ty, LV4, DL), nullptr);
  EXPECT_EQ(LV3.getCompare(CmpInst::ICMP_SLE, I1Ty, LV4, DL), nullptr);
  EXPECT_EQ(LV3.getCompare(CmpInst::ICMP_NE, I1Ty, LV4, DL), nullptr);
  EXPECT_EQ(LV3.getCompare(CmpInst::ICMP_EQ, I1Ty, LV4, DL), nullptr);
  EXPECT_EQ(LV3.getCompare(CmpInst::ICMP_SGE, I1Ty, LV4, DL), nullptr);
  EXPECT_EQ(LV3.getCompare(CmpInst::ICMP_SGT, I1Ty, LV4, DL), nullptr);
}

TEST_F(ValueLatticeTest, getCompareFloat) {
  auto *FloatTy = IntegerType::getFloatTy(Context);
  auto *I1Ty = IntegerType::get(Context, 1);
  auto *C1 = ConstantFP::get(FloatTy, 1.0);
  auto LV1 = ValueLatticeElement::get(C1);
  auto LV2 = ValueLatticeElement::get(C1);

  // Check getCompare for equal floating point constants.
  EXPECT_TRUE(LV1.getCompare(CmpInst::FCMP_OEQ, I1Ty, LV2, DL)->isOneValue());
  EXPECT_TRUE(LV1.getCompare(CmpInst::FCMP_OGE, I1Ty, LV2, DL)->isOneValue());
  EXPECT_TRUE(LV1.getCompare(CmpInst::FCMP_OLE, I1Ty, LV2, DL)->isOneValue());
  EXPECT_TRUE(LV1.getCompare(CmpInst::FCMP_ONE, I1Ty, LV2, DL)->isZeroValue());
  EXPECT_TRUE(LV1.getCompare(CmpInst::FCMP_OLT, I1Ty, LV2, DL)->isZeroValue());
  EXPECT_TRUE(LV1.getCompare(CmpInst::FCMP_OGT, I1Ty, LV2, DL)->isZeroValue());

  EXPECT_TRUE(
      LV1.mergeIn(ValueLatticeElement::get(ConstantFP::get(FloatTy, 2.2))));
  EXPECT_EQ(LV1.getCompare(CmpInst::FCMP_OEQ, I1Ty, LV2, DL), nullptr);
  EXPECT_EQ(LV1.getCompare(CmpInst::FCMP_OGE, I1Ty, LV2, DL), nullptr);
  EXPECT_EQ(LV1.getCompare(CmpInst::FCMP_OLE, I1Ty, LV2, DL), nullptr);
  EXPECT_EQ(LV1.getCompare(CmpInst::FCMP_ONE, I1Ty, LV2, DL), nullptr);
  EXPECT_EQ(LV1.getCompare(CmpInst::FCMP_OLT, I1Ty, LV2, DL), nullptr);
  EXPECT_EQ(LV1.getCompare(CmpInst::FCMP_OGT, I1Ty, LV2, DL), nullptr);
}

TEST_F(ValueLatticeTest, getCompareUndef) {
  auto *I32Ty = IntegerType::get(Context, 32);
  auto *I1Ty = IntegerType::get(Context, 1);

  // TODO: These results can be improved.
  auto LV1 = ValueLatticeElement::get(UndefValue::get(I32Ty));
  auto LV2 =
      ValueLatticeElement::getRange({APInt(32, 10, true), APInt(32, 20, true)});
  EXPECT_EQ(LV1.getCompare(CmpInst::ICMP_SLT, I1Ty, LV2, DL), nullptr);
  EXPECT_EQ(LV1.getCompare(CmpInst::ICMP_SLE, I1Ty, LV2, DL), nullptr);
  EXPECT_EQ(LV1.getCompare(CmpInst::ICMP_NE, I1Ty, LV2, DL), nullptr);
  EXPECT_EQ(LV1.getCompare(CmpInst::ICMP_EQ, I1Ty, LV2, DL), nullptr);
  EXPECT_EQ(LV1.getCompare(CmpInst::ICMP_SGE, I1Ty, LV2, DL), nullptr);
  EXPECT_EQ(LV1.getCompare(CmpInst::ICMP_SGT, I1Ty, LV2, DL), nullptr);

  auto *FloatTy = IntegerType::getFloatTy(Context);
  auto LV3 = ValueLatticeElement::get(ConstantFP::get(FloatTy, 1.0));
  EXPECT_EQ(LV1.getCompare(CmpInst::FCMP_OEQ, I1Ty, LV3, DL), nullptr);
  EXPECT_EQ(LV1.getCompare(CmpInst::FCMP_OGE, I1Ty, LV3, DL), nullptr);
  EXPECT_EQ(LV1.getCompare(CmpInst::FCMP_OLE, I1Ty, LV3, DL), nullptr);
  EXPECT_EQ(LV1.getCompare(CmpInst::FCMP_ONE, I1Ty, LV3, DL), nullptr);
  EXPECT_EQ(LV1.getCompare(CmpInst::FCMP_OLT, I1Ty, LV3, DL), nullptr);
  EXPECT_EQ(LV1.getCompare(CmpInst::FCMP_OGT, I1Ty, LV3, DL), nullptr);
}

} // end anonymous namespace
} // end namespace llvm
