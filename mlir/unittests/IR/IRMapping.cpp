//===- IRMapping.cpp --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Builders.h"
#include "gtest/gtest.h"

#include "../../test/lib/Dialect/Test/TestDialect.h"
#include "../../test/lib/Dialect/Test/TestOps.h"

using namespace mlir;

TEST(IRMapping, TypedValue) {
  MLIRContext context;

  context.loadDialect<test::TestDialect>();

  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();

  Block block;
  builder.setInsertionPointToEnd(&block);

  Value i64Val = builder.create<test::TestOpConstant>(
      loc, builder.getI64Type(), builder.getI64IntegerAttr(0));
  Value f64Val = builder.create<test::TestOpConstant>(
      loc, builder.getF64Type(), builder.getF64FloatAttr(0.0));

  IRMapping mapping;
  mapping.map(i64Val, f64Val);
  auto typedI64Val = cast<TypedValue<IntegerType>>(i64Val);
  EXPECT_EQ(mapping.lookup(typedI64Val), f64Val);
}

TEST(IRMapping, OperationClone) {
  MLIRContext ctx;
  ctx.allowUnregisteredDialects();

  OperationState state(UnknownLoc::get(&ctx), "no_results");
  Operation *noResultsOp = Operation::create(state);

  OperationState owner(UnknownLoc::get(&ctx), "owner");
  owner.addRegion()->emplaceBlock().push_back(noResultsOp);
  OwningOpRef<Operation *> ownerOp = Operation::create(owner);

  IRMapping irMap;
  OwningOpRef<Operation *> clonedOwnerOp = (*ownerOp)->clone(irMap);

  EXPECT_EQ(irMap.lookupOrNull(*ownerOp), *clonedOwnerOp);
  EXPECT_EQ(irMap.lookupOrNull(noResultsOp),
            &(*clonedOwnerOp)->getRegion(0).front().front());
}
