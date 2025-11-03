//===- mlir/unittest/IR/ValueTest.cpp - Value unit tests ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Value.h"
#include "../../test/lib/Dialect/Test/TestDialect.h"
#include "../../test/lib/Dialect/Test/TestOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "gtest/gtest.h"

using namespace mlir;

static Operation *createOp(MLIRContext *context, ArrayRef<Value> operands = {},
                           ArrayRef<Type> resultTypes = {},
                           unsigned int numRegions = 0) {
  context->allowUnregisteredDialects();
  return Operation::create(UnknownLoc::get(context),
                           OperationName("foo.bar", context), resultTypes,
                           operands, NamedAttrList(), nullptr, {}, numRegions);
}

namespace {

TEST(ValueTest, getNumUses) {
  MLIRContext context;
  Builder builder(&context);

  Operation *op0 =
      createOp(&context, /*operands=*/{}, builder.getIntegerType(16));

  Value v0 = op0->getResult(0);
  EXPECT_EQ(v0.getNumUses(), (unsigned)0);

  Operation *op1 = createOp(&context, {v0}, builder.getIntegerType(16));
  EXPECT_EQ(v0.getNumUses(), (unsigned)1);

  Operation *op2 = createOp(&context, {v0, v0}, builder.getIntegerType(16));
  EXPECT_EQ(v0.getNumUses(), (unsigned)3);

  op2->destroy();
  op1->destroy();
  op0->destroy();
}

TEST(ValueTest, hasNUses) {
  MLIRContext context;
  Builder builder(&context);

  Operation *op0 =
      createOp(&context, /*operands=*/{}, builder.getIntegerType(16));
  Value v0 = op0->getResult(0);
  EXPECT_TRUE(v0.hasNUses(0));
  EXPECT_FALSE(v0.hasNUses(1));

  Operation *op1 = createOp(&context, {v0}, builder.getIntegerType(16));
  EXPECT_FALSE(v0.hasNUses(0));
  EXPECT_TRUE(v0.hasNUses(1));

  Operation *op2 = createOp(&context, {v0, v0}, builder.getIntegerType(16));
  EXPECT_FALSE(v0.hasNUses(0));
  EXPECT_FALSE(v0.hasNUses(1));
  EXPECT_TRUE(v0.hasNUses(3));

  op2->destroy();
  op1->destroy();
  op0->destroy();
}

TEST(ValueTest, hasNUsesOrMore) {
  MLIRContext context;
  Builder builder(&context);

  Operation *op0 =
      createOp(&context, /*operands=*/{}, builder.getIntegerType(16));
  Value v0 = op0->getResult(0);
  EXPECT_TRUE(v0.hasNUsesOrMore(0));
  EXPECT_FALSE(v0.hasNUsesOrMore(1));

  Operation *op1 = createOp(&context, {v0}, builder.getIntegerType(16));
  EXPECT_TRUE(v0.hasNUsesOrMore(0));
  EXPECT_TRUE(v0.hasNUsesOrMore(1));
  EXPECT_FALSE(v0.hasNUsesOrMore(2));

  Operation *op2 = createOp(&context, {v0, v0}, builder.getIntegerType(16));
  EXPECT_TRUE(v0.hasNUsesOrMore(0));
  EXPECT_TRUE(v0.hasNUsesOrMore(1));
  EXPECT_TRUE(v0.hasNUsesOrMore(3));
  EXPECT_FALSE(v0.hasNUsesOrMore(4));

  op2->destroy();
  op1->destroy();
  op0->destroy();
}

} // end anonymous namespace
