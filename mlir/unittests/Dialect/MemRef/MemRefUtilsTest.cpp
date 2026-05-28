//===- MemRefUtilsTest.cpp - MemRef utils unit tests ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "gtest/gtest.h"

using namespace mlir;

TEST(MemRefUtilsTest, IsStaticShapeAndContiguousRowMajor) {
  MLIRContext ctx;
  OpBuilder builder(&ctx);

  // Rank-0 memrefs are scalar values and should return before stride analysis.
  EXPECT_TRUE(memref::isStaticShapeAndContiguousRowMajor(
      MemRefType::get({}, builder.getI32Type())));
}

TEST(MemRefUtilsTest, ComputeSuffixProductIRBlockEmptySizes) {
  MLIRContext ctx;
  OpBuilder builder(&ctx);

  // Empty size lists should not underflow the reverse loop's initial index.
  EXPECT_TRUE(
      memref::computeSuffixProductIRBlock(builder.getUnknownLoc(), builder, {})
          .empty());
}
