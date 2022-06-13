//
// Created by tanmay on 6/13/22.
//

#include "llvm/Transforms/ErrorAnalysis/AtomicCondition/AtomicCondition.h"
#include "gtest/gtest.h"

TEST(AtomicCondition, BinaryOperationAdd) {
  EXPECT_FLOAT_EQ(acBinaryAdd(2.0, 4.0, 1), 0.33333333);
  EXPECT_DOUBLE_EQ(acBinaryAdd(2.0, 4.0, 1), 0.33333333333333333);

  EXPECT_FLOAT_EQ(acBinaryAdd(2.0, 4.0, 2), 0.66666666);
  EXPECT_DOUBLE_EQ(acBinaryAdd(2.0, 4.0, 2), 0.66666666666666666);
}

TEST(AtomicCondition, BinaryOperationSub) {
  EXPECT_FLOAT_EQ(acBinarySub(2.0, 4.0, 1), 1.0);
  EXPECT_DOUBLE_EQ(acBinarySub(2.0, 4.0, 1), 1.0);

  EXPECT_FLOAT_EQ(acBinarySub(2.0, 4.0, 2), 2.0);
  EXPECT_DOUBLE_EQ(acBinarySub(2.0, 4.0, 2), 2.0);
}

TEST(AtomicCondition, BinaryOperationMul) {
  EXPECT_FLOAT_EQ(acBinaryMul(2.0, 4.0, 1), 1.0);
  EXPECT_DOUBLE_EQ(acBinaryMul(2.0, 4.0, 1), 1.0);

  EXPECT_FLOAT_EQ(acBinaryMul(2.0, 4.0, 2), 1.0);
  EXPECT_DOUBLE_EQ(acBinaryMul(2.0, 4.0, 2), 1.0);
}

TEST(AtomicCondition, BinaryOperationDiv) {
  EXPECT_FLOAT_EQ(acBinaryDiv(2.0, 4.0, 1), 1.0);
  EXPECT_DOUBLE_EQ(acBinaryDiv(2.0, 4.0, 1), 1.0);

  EXPECT_FLOAT_EQ(acBinaryMul(2.0, 4.0, 2), 1.0);
  EXPECT_DOUBLE_EQ(acBinaryMul(2.0, 4.0, 2), 1.0);
}
