//===- SPIRVOpsTest.cpp - SPIR-V Op Tests ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "gtest/gtest.h"

using namespace mlir;

TEST(SPIRVOpsTest, ConstantZeroAndOneOfArray) {
  MLIRContext context;
  context.loadDialect<spirv::SPIRVDialect>();
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  Block block;
  builder.setInsertionPointToEnd(&block);

  auto intArray = spirv::ArrayType::get(builder.getI32Type(), 3);
  Attribute zeroAttr = builder.getI32IntegerAttr(0);
  spirv::ConstantOp zero = spirv::ConstantOp::getZero(intArray, loc, builder);
  EXPECT_EQ(zero.getType(), intArray);
  EXPECT_EQ(zero.getValue(),
            builder.getArrayAttr({zeroAttr, zeroAttr, zeroAttr}));
  EXPECT_TRUE(succeeded(verify(zero)));

  auto floatArray = spirv::ArrayType::get(builder.getF32Type(), 2);
  Attribute oneAttr = builder.getF32FloatAttr(1.0);
  spirv::ConstantOp one = spirv::ConstantOp::getOne(floatArray, loc, builder);
  EXPECT_EQ(one.getType(), floatArray);
  EXPECT_EQ(one.getValue(), builder.getArrayAttr({oneAttr, oneAttr}));
  EXPECT_TRUE(succeeded(verify(one)));
}

TEST(SPIRVOpsTest, ConstantOneOfNestedBoolArray) {
  MLIRContext context;
  context.loadDialect<spirv::SPIRVDialect>();
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  Block block;
  builder.setInsertionPointToEnd(&block);

  auto inner = spirv::ArrayType::get(builder.getI1Type(), 2);
  auto outer = spirv::ArrayType::get(inner, 2);
  Attribute trueAttr = builder.getBoolAttr(true);
  Attribute innerAttr = builder.getArrayAttr({trueAttr, trueAttr});
  spirv::ConstantOp one = spirv::ConstantOp::getOne(outer, loc, builder);
  EXPECT_EQ(one.getType(), outer);
  EXPECT_EQ(one.getValue(), builder.getArrayAttr({innerAttr, innerAttr}));
  EXPECT_TRUE(succeeded(verify(one)));
}
