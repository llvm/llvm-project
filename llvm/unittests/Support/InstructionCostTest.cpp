//===- InstructionCostTest.cpp - InstructionCost tests --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/InstructionCost.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

struct CostTest : public testing::Test {
  CostTest() {}
};

} // namespace

TEST_F(CostTest, Operators) {
  InstructionCost VThree = 3;
  InstructionCost VNegTwo = -2;
  InstructionCost VSix = 6;
  InstructionCost IThreeA = InstructionCost::getInvalid(3);
  InstructionCost IThreeB = InstructionCost::getInvalid(3);
  InstructionCost TmpCost;

  EXPECT_NE(VThree, VNegTwo);
  EXPECT_GT(VThree, VNegTwo);
  EXPECT_NE(VThree, IThreeA);
  EXPECT_EQ(IThreeA, IThreeB);
  EXPECT_GE(IThreeA, VNegTwo);
  EXPECT_LT(VSix, IThreeA);
  EXPECT_EQ(VSix - IThreeA, IThreeB);
  EXPECT_EQ(VThree - VNegTwo, 5);
  EXPECT_EQ(VThree * VNegTwo, -6);
  EXPECT_EQ(VSix / VThree, 2);

  EXPECT_FALSE(IThreeA.isValid());
  EXPECT_EQ(IThreeA.getState(), InstructionCost::Invalid);

  TmpCost = VThree + IThreeA;
  EXPECT_FALSE(TmpCost.isValid());

  // Test increments, decrements
  EXPECT_EQ(++VThree, 4);
  EXPECT_EQ(VThree++, 4);
  EXPECT_EQ(VThree, 5);
  EXPECT_EQ(--VThree, 4);
  EXPECT_EQ(VThree--, 4);
  EXPECT_EQ(VThree, 3);

  TmpCost = VThree * IThreeA;
  EXPECT_FALSE(TmpCost.isValid());

  // Test value extraction
  EXPECT_EQ(*(VThree.getValue()), 3);
  EXPECT_EQ(IThreeA.getValue(), None);

  EXPECT_EQ(std::min(VThree, VNegTwo), -2);
  EXPECT_EQ(std::max(VThree, VSix), 6);
}
