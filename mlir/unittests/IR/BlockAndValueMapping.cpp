//===- BlockAndValueMapping.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "gtest/gtest.h"

#include "../../test/lib/Dialect/Test/TestDialect.h"

using namespace mlir;

TEST(BlockAndValueMapping, TypedValue) {
  MLIRContext context;

  context.loadDialect<test::TestDialect>();

  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();

  Value i64Val = builder.create<test::TestOpConstant>(
      loc, builder.getI64Type(), builder.getI64IntegerAttr(0));
  Value f64Val = builder.create<test::TestOpConstant>(
      loc, builder.getF64Type(), builder.getF64FloatAttr(0.0));

  BlockAndValueMapping mapping;
  mapping.map(i64Val, f64Val);
  TypedValue<IntegerType> typedI64Val = i64Val;
  mapping.lookup(typedI64Val);
}
