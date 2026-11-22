//===- ArithUtilsTest.cpp - Arith utils unit tests ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "gtest/gtest.h"

using namespace mlir;

namespace {

class ArithUtilsTest : public ::testing::Test {
protected:
  ArithUtilsTest() : builder(&context), loc(builder.getUnknownLoc()) {
    context.loadDialect<arith::ArithDialect>();
    builder.setInsertionPointToStart(&block);
  }

  void expectConvertFOp(Type inputType, Type resultType) {
    Value input = block.addArgument(inputType, loc);
    Value result = convertScalarToDtype(builder, loc, input, resultType,
                                        /*isUnsignedCast=*/false);

    EXPECT_EQ(result.getType(), resultType);
    auto convertOp = result.getDefiningOp<arith::ConvertFOp>();
    ASSERT_TRUE(convertOp);
    EXPECT_EQ(convertOp.getIn(), input);
  }

  MLIRContext context;
  OpBuilder builder;
  Location loc;
  Block block;
};

TEST_F(ArithUtilsTest, ConvertF16ToBF16) {
  expectConvertFOp(builder.getF16Type(), builder.getBF16Type());
}

TEST_F(ArithUtilsTest, ConvertBF16ToF16) {
  expectConvertFOp(builder.getBF16Type(), builder.getF16Type());
}

TEST_F(ArithUtilsTest, InferExpandShapeOutputShapeStaticInput) {
  // The input shape can be more static than the expanded type: the constant
  // input dim must be materialized as a value for the dynamic output dim.
  RankedTensorType expandedType =
      RankedTensorType::get({ShapedType::kDynamic, 3}, builder.getF32Type());
  std::optional<SmallVector<OpFoldResult>> outputShape =
      inferExpandShapeOutputShape(
          builder, loc, expandedType,
          {ReassociationIndices{0}, ReassociationIndices{1}},
          {builder.getIndexAttr(2), builder.getIndexAttr(3)});
  ASSERT_TRUE(outputShape.has_value());
  ASSERT_EQ(outputShape->size(), 2u);
  EXPECT_EQ(getConstantIntValue((*outputShape)[0]), 2);
  EXPECT_EQ(getConstantIntValue((*outputShape)[1]), 3);
}

TEST_F(ArithUtilsTest, InferExpandShapeOutputShapeNonDivisibleInput) {
  // tensor<3xf32> cannot be expanded into tensor<?x2xf32>, so the dynamic
  // output dim size cannot be inferred.
  RankedTensorType expandedType =
      RankedTensorType::get({ShapedType::kDynamic, 2}, builder.getF32Type());
  EXPECT_FALSE(inferExpandShapeOutputShape(builder, loc, expandedType,
                                           {ReassociationIndices{0, 1}},
                                           {builder.getIndexAttr(3)}));
}
} // namespace
