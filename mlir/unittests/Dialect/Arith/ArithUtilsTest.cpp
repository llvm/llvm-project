//===- ArithUtilsTest.cpp - Arith utils unit tests ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/Utils/Utils.h"
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

} // namespace
