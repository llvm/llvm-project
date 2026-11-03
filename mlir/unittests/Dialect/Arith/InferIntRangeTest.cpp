//===- InferIntRangeTest.cpp - Arith integer range inference tests
//----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "gtest/gtest.h"
#include <array>

using namespace mlir;

namespace {
class SelectIntRangeTest : public ::testing::Test {
protected:
  SelectIntRangeTest() : builder(&context), loc(builder.getUnknownLoc()) {
    context.loadDialect<arith::ArithDialect>();
    builder.setInsertionPointToStart(&block);
  }

  arith::SelectOp createSelect(Type type) {
    Value cond = block.addArgument(builder.getI1Type(), loc);
    Value lhs = block.addArgument(type, loc);
    Value rhs = block.addArgument(type, loc);
    return arith::SelectOp::create(builder, loc, cond, lhs, rhs);
  }

  MLIRContext context;
  OpBuilder builder;
  Location loc;
  Block block;
};

TEST_F(SelectIntRangeTest, WaitForBothArms) {
  auto select = createSelect(builder.getI32Type());
  IntegerValueRange seven(ConstantIntRanges::constant(APInt(32, 7)));
  IntegerValueRange nine(ConstantIntRanges::constant(APInt(32, 9)));
  for (bool conditionReady : {false, true}) {
    for (unsigned readyArms = 0; readyArms < 3; ++readyArms) {
      std::array<IntegerValueRange, 3> operands{
          conditionReady ? IntegerValueRange(ConstantIntRanges::maxRange(1))
                         : IntegerValueRange{},
          readyArms & 1 ? seven : IntegerValueRange{},
          readyArms & 2 ? nine : IntegerValueRange{}};
      IntegerValueRange result;
      auto joinResult = [&](Value value, const IntegerValueRange &range) {
        EXPECT_EQ(value, select.getResult());
        result = IntegerValueRange::join(result, range);
      };
      select.inferResultRangesFromOptional(operands, joinResult);
      EXPECT_TRUE(result.isUninitialized());

      operands[1] = seven;
      operands[2] = nine;
      select.inferResultRangesFromOptional(operands, joinResult);
      EXPECT_EQ(result, IntegerValueRange::join(seven, nine));
    }
  }
}

TEST_F(SelectIntRangeTest, KnownConditionNeedsOnlySelectedArm) {
  auto select = createSelect(builder.getI32Type());
  IntegerValueRange seven(ConstantIntRanges::constant(APInt(32, 7)));
  for (bool condition : {false, true}) {
    std::array<IntegerValueRange, 3> operands{
        IntegerValueRange(ConstantIntRanges::constant(APInt(1, condition))),
        condition ? seven : IntegerValueRange{},
        condition ? IntegerValueRange{} : seven};
    IntegerValueRange result;
    select.inferResultRangesFromOptional(
        operands,
        [&](Value, const IntegerValueRange &range) { result = range; });
    EXPECT_EQ(result, seven);
  }
}

TEST_F(SelectIntRangeTest, PoisonConditionNeedsNoArms) {
  auto select = createSelect(builder.getI32Type());
  std::array<IntegerValueRange, 3> operands{
      IntegerValueRange(ConstantIntRanges::empty(1)), IntegerValueRange{},
      IntegerValueRange{}};
  IntegerValueRange result;
  select.inferResultRangesFromOptional(
      operands, [&](Value, const IntegerValueRange &range) { result = range; });
  EXPECT_EQ(result, IntegerValueRange(ConstantIntRanges::empty(32)));
}

TEST_F(SelectIntRangeTest, NonIntegerHasNoRange) {
  auto select = createSelect(builder.getF32Type());
  std::array<IntegerValueRange, 3> operands;
  EXPECT_TRUE(
      IntegerValueRange::getMaxRange(select.getResult()).isUninitialized());
  select.inferResultRangesFromOptional(
      operands, [&](Value, const IntegerValueRange &) {
        ADD_FAILURE() << "Non-integer results have no integer range";
      });
}
TEST_F(SelectIntRangeTest, NonIntegerLatticeIgnoresIntegerRange) {
  dataflow::IntegerValueRangeLattice integer(
      block.addArgument(builder.getI32Type(), loc));
  dataflow::IntegerValueRangeLattice nonInteger(
      block.addArgument(builder.getF32Type(), loc));
  EXPECT_EQ(integer.join(
                IntegerValueRange(ConstantIntRanges::constant(APInt(32, 7)))),
            ChangeResult::Change);
  EXPECT_EQ(nonInteger.join(integer), ChangeResult::NoChange);
  EXPECT_TRUE(nonInteger.getValue().isUninitialized());
}

TEST_F(SelectIntRangeTest, NonIntegerSourceDoesNotActAsPendingInteger) {
  dataflow::IntegerValueRangeLattice integer(
      block.addArgument(builder.getI32Type(), loc));
  dataflow::IntegerValueRangeLattice nonInteger(
      block.addArgument(builder.getF32Type(), loc));
  EXPECT_EQ(integer.join(
                IntegerValueRange(ConstantIntRanges::constant(APInt(32, 7)))),
            ChangeResult::Change);
  EXPECT_EQ(integer.join(nonInteger), ChangeResult::Change);
  EXPECT_EQ(integer.getValue(),
            IntegerValueRange(ConstantIntRanges::maxRange(32)));
}
} // namespace
