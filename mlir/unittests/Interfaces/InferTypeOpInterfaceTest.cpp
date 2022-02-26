//===- InferTypeOpInterfaceTest.cpp - Unit Test for type interface --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Parser.h"

#include <gtest/gtest.h>

using namespace mlir;

class ValueShapeRangeTest : public testing::Test {
protected:
  void SetUp() override {
    const char *ir = R"MLIR(
      func @map(%arg : tensor<1xi64>) {
        %0 = arith.constant dense<[10]> : tensor<1xi64>
        %1 = arith.addi %arg, %0 : tensor<1xi64>
        return
      }
    )MLIR";

    registry.insert<func::FuncDialect, arith::ArithmeticDialect>();
    ctx.appendDialectRegistry(registry);
    module = parseSourceString(ir, &ctx);
    mapFn = cast<FuncOp>(module->front());
  }

  // Create ValueShapeRange on the arith.addi operation.
  ValueShapeRange addiRange() {
    auto &fnBody = mapFn.body();
    return std::next(fnBody.front().begin())->getOperands();
  }

  DialectRegistry registry;
  MLIRContext ctx;
  OwningOpRef<ModuleOp> module;
  FuncOp mapFn;
};

TEST_F(ValueShapeRangeTest, ShapesFromValues) {
  ValueShapeRange range = addiRange();

  EXPECT_FALSE(range.getValueAsShape(0));
  ASSERT_TRUE(range.getValueAsShape(1));
  EXPECT_TRUE(range.getValueAsShape(1).hasRank());
  EXPECT_EQ(range.getValueAsShape(1).getRank(), 1);
  EXPECT_EQ(range.getValueAsShape(1).getDimSize(0), 10);
  EXPECT_EQ(range.getShape(1).getRank(), 1);
  EXPECT_EQ(range.getShape(1).getDimSize(0), 1);
}

TEST_F(ValueShapeRangeTest, MapValuesToShapes) {
  ValueShapeRange range = addiRange();
  ShapedTypeComponents fixed(SmallVector<int64_t>{30});
  auto mapping = [&](Value val) -> ShapeAdaptor {
    if (val == mapFn.getArgument(0))
      return &fixed;
    return nullptr;
  };
  range.setValueToShapeMapping(mapping);

  ASSERT_TRUE(range.getValueAsShape(0));
  EXPECT_TRUE(range.getValueAsShape(0).hasRank());
  EXPECT_EQ(range.getValueAsShape(0).getRank(), 1);
  EXPECT_EQ(range.getValueAsShape(0).getDimSize(0), 30);
  ASSERT_TRUE(range.getValueAsShape(1));
  EXPECT_TRUE(range.getValueAsShape(1).hasRank());
  EXPECT_EQ(range.getValueAsShape(1).getRank(), 1);
  EXPECT_EQ(range.getValueAsShape(1).getDimSize(0), 10);
}

TEST_F(ValueShapeRangeTest, SettingShapes) {
  ShapedTypeComponents shape(SmallVector<int64_t>{10, 20});
  ValueShapeRange range = addiRange();
  auto mapping = [&](Value val) -> ShapeAdaptor {
    if (val == mapFn.getArgument(0))
      return &shape;
    return nullptr;
  };
  range.setOperandShapeMapping(mapping);

  ASSERT_TRUE(range.getShape(0));
  EXPECT_EQ(range.getShape(0).getRank(), 2);
  EXPECT_EQ(range.getShape(0).getDimSize(0), 10);
  EXPECT_EQ(range.getShape(0).getDimSize(1), 20);
  ASSERT_TRUE(range.getShape(1));
  EXPECT_EQ(range.getShape(1).getRank(), 1);
  EXPECT_EQ(range.getShape(1).getDimSize(0), 1);
  EXPECT_FALSE(range.getShape(2));
}
