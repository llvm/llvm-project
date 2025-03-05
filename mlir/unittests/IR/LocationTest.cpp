//===- LocationTest.cpp - unit tests for affine map API -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Location.h"
#include "mlir/IR/Builders.h"
#include "gtest/gtest.h"

using namespace mlir;

// Check that we only walk *locations* and not non-location attributes.
TEST(LocationTest, Walk) {
  MLIRContext ctx;
  Builder builder(&ctx);
  BoolAttr trueAttr = builder.getBoolAttr(true);

  Location loc1 = FileLineColLoc::get(builder.getStringAttr("foo"), 1, 2);
  Location loc2 = FileLineColLoc::get(builder.getStringAttr("foo"), 3, 4);
  Location fused = builder.getFusedLoc({loc1, loc2}, trueAttr);

  SmallVector<Attribute> visited;
  fused->walk([&](Location l) {
    visited.push_back(LocationAttr(l));
    return WalkResult::advance();
  });

  EXPECT_EQ(llvm::ArrayRef(visited), ArrayRef<Attribute>({fused, loc1, loc2}));
}

// Check that we skip location attrs nested under a non-location attr.
TEST(LocationTest, SkipNested) {
  MLIRContext ctx;
  Builder builder(&ctx);

  Location loc1 = FileLineColLoc::get(builder.getStringAttr("foo"), 1, 2);
  Location loc2 = FileLineColLoc::get(builder.getStringAttr("foo"), 3, 4);
  Location loc3 = FileLineColLoc::get(builder.getStringAttr("bar"), 1, 2);
  Location loc4 = FileLineColLoc::get(builder.getStringAttr("bar"), 3, 4);
  ArrayAttr arr = builder.getArrayAttr({loc3, loc4});
  Location fused = builder.getFusedLoc({loc1, loc2}, arr);

  SmallVector<Attribute> visited;
  fused->walk([&](Location l) {
    visited.push_back(LocationAttr(l));
    return WalkResult::advance();
  });

  EXPECT_EQ(llvm::ArrayRef(visited), ArrayRef<Attribute>({fused, loc1, loc2}));
}
