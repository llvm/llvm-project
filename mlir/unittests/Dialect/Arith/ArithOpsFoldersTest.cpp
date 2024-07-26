//===- ArithOpsFoldersTest.cpp - unit tests for arith op folders ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "gtest/gtest.h"

using namespace mlir;

namespace {
// Tests a regression that made `BitcastOp::fold` crash on invalid input IR, see
// #100743,
TEST(BitcastOpTest, FoldInteger) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect>();
  auto loc = UnknownLoc::get(&context);
  auto module = ModuleOp::create(loc);
  OpBuilder builder(module.getBodyRegion());
  Value i32Val = builder.create<arith::ConstantOp>(
      loc, builder.getI32Type(), builder.getI32IntegerAttr(0));
  // This would create an invalid op: `bitcast` can't cast different bitwidths.
  builder.createOrFold<arith::BitcastOp>(loc, builder.getI64Type(), i32Val);
  ASSERT_TRUE(failed(verify(module)));
}

// Tests a regression that made `BitcastOp::fold` crash on invalid input IR, see
// #100743,
TEST(BitcastOpTest, FoldFloat) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect>();
  auto loc = UnknownLoc::get(&context);
  auto module = ModuleOp::create(loc);
  OpBuilder builder(module.getBodyRegion());
  Value f32Val = builder.create<arith::ConstantOp>(loc, builder.getF32Type(),
                                                   builder.getF32FloatAttr(0));
  // This would create an invalid op: `bitcast` can't cast different bitwidths.
  builder.createOrFold<arith::BitcastOp>(loc, builder.getF64Type(), f32Val);
  ASSERT_TRUE(failed(verify(module)));
}
} // namespace
