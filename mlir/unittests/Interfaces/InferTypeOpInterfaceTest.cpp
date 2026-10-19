//===- InferTypeOpInterfaceTest.cpp - Unit Test for type interface --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Dialect/Arith/IR/ArithDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Parser/Parser.h"

#include <gtest/gtest.h>

using namespace mlir;

TEST(ShapedTypeComponentsTest, RankedScalarAndUnrankedCopies) {
  ShapedTypeComponents unranked;
  ShapedTypeComponents scalar(ArrayRef<int64_t>{});
  EXPECT_FALSE(unranked.hasRank());
  EXPECT_TRUE(scalar.hasRank());
  EXPECT_TRUE(scalar.getDims().empty());

  unranked = scalar;
  EXPECT_TRUE(unranked.hasRank());
  unranked = ShapedTypeComponents();
  EXPECT_FALSE(unranked.hasRank());
}

TEST(ShapedTypeComponentsTest, HeapBackedMovesAndShapeAdaptor) {
  SmallVector<int64_t> dimensions(16, 7);
  SmallVector<int64_t> input = dimensions;
  int64_t *data = input.data();
  ShapedTypeComponents shape(std::move(input));
  EXPECT_EQ(shape.getDims().data(), data);
  ShapedTypeComponents moved(std::move(shape));
  EXPECT_TRUE(moved.hasRank());
  EXPECT_EQ(moved.getDims(), ArrayRef<int64_t>(dimensions));

  ShapedTypeComponents result;
  ShapeAdaptor(moved).getDims(result);
  EXPECT_TRUE(result.hasRank());
  EXPECT_EQ(result.getDims(), ArrayRef<int64_t>(dimensions));
  SmallVector<int64_t> output;
  ShapeAdaptor(result).getDims(output);
  EXPECT_EQ(output, dimensions);
}

class ValueShapeRangeTest : public testing::Test {
protected:
  void SetUp() override {
    const char *ir = R"MLIR(
      func.func @map(%arg : tensor<1xi64>) {
        %0 = arith.constant dense<[10]> : tensor<1xi64>
        %1 = arith.addi %arg, %0 : tensor<1xi64>
        return
      }
    )MLIR";

    registry.insert<func::FuncDialect, arith::ArithDialect>();
    ctx.appendDialectRegistry(registry);
    module = parseSourceString<ModuleOp>(ir, &ctx);
    assert(module);
    mapFn = cast<func::FuncOp>(module->front());
  }

  // Create ValueShapeRange on the arith.addi operation.
  ValueShapeRange addiRange() {
    auto &fnBody = mapFn.getBody();
    return std::next(fnBody.front().begin())->getOperands();
  }

  DialectRegistry registry;
  MLIRContext ctx;
  OwningOpRef<ModuleOp> module;
  func::FuncOp mapFn;
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
