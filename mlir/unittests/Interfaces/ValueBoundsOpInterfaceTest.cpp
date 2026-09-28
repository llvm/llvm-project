//===- ValueBoundsOpInterfaceTest.cpp - ValueBounds unit tests -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include <gtest/gtest.h>

using namespace mlir;

namespace {

TEST(ValueBoundsOpInterfaceTest, EquivalentSlices) {
  MLIRContext context;
  context.loadDialect<func::FuncDialect>();
  OpBuilder builder(&context);

  auto function = func::FuncOp::create(
      builder.getUnknownLoc(), "test",
      builder.getFunctionType({builder.getIndexType(), builder.getIndexType()},
                              {}));
  Block *entryBlock = function.addEntryBlock();
  Value dynamicValue = entryBlock->getArgument(0);
  Value otherDynamicValue = entryBlock->getArgument(1);

  OpFoldResult zero = builder.getIndexAttr(0);
  OpFoldResult one = builder.getIndexAttr(1);
  OpFoldResult four = builder.getIndexAttr(4);
  OpFoldResult five = builder.getIndexAttr(5);

  HyperrectangularSlice slice({zero, dynamicValue}, {four, dynamicValue},
                              {one, one});
  HyperrectangularSlice identicalSlice({zero, dynamicValue},
                                       {four, dynamicValue}, {one, one});
  FailureOr<bool> equivalent = ValueBoundsConstraintSet::areEquivalentSlices(
      &context, slice, identicalSlice);
  ASSERT_TRUE(succeeded(equivalent));
  EXPECT_TRUE(*equivalent);

  // Identical components must not hide a later non-equivalent component.
  HyperrectangularSlice differentSlice({zero, dynamicValue},
                                       {five, dynamicValue}, {one, one});
  equivalent = ValueBoundsConstraintSet::areEquivalentSlices(&context, slice,
                                                             differentSlice);
  ASSERT_TRUE(succeeded(equivalent));
  EXPECT_FALSE(*equivalent);

  // Distinct dynamic values continue to use the ValueBounds fallback.
  HyperrectangularSlice unknownSlice({zero, otherDynamicValue},
                                     {four, dynamicValue}, {one, one});
  equivalent = ValueBoundsConstraintSet::areEquivalentSlices(&context, slice,
                                                             unknownSlice);
  EXPECT_TRUE(failed(equivalent));
}

} // namespace
