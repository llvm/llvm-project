//===- StructuredOpsUtilsTest.cpp - StructuredOpsUtils unit tests ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace mlir;
using testing::Not;
using testing::Truly;

namespace {

TEST(isRowMajorMatmul, Simple) {
  MLIRContext context;

  AffineExpr m, n, k;
  bindDims(&context, m, n, k);
  auto mapA = AffineMapAttr::get(AffineMap::get(3, 0, {m, k}, &context));
  auto mapB = AffineMapAttr::get(AffineMap::get(3, 0, {k, n}, &context));
  auto mapC = AffineMapAttr::get(AffineMap::get(3, 0, {m, n}, &context));
  auto maps = ArrayAttr::get(&context, {mapA, mapB, mapC});

  EXPECT_THAT(maps, Truly(isRowMajorMatmul));
}

TEST(isRowMajorMatmul, BindingShifted) {
  MLIRContext context;

  AffineExpr m, n, k;
  bindDims(&context, k, m, n); // bind in different order
  auto mapA = AffineMapAttr::get(AffineMap::get(3, 0, {m, k}, &context));
  auto mapB = AffineMapAttr::get(AffineMap::get(3, 0, {k, n}, &context));
  auto mapC = AffineMapAttr::get(AffineMap::get(3, 0, {m, n}, &context));
  auto maps = ArrayAttr::get(&context, {mapA, mapB, mapC});

  EXPECT_THAT(maps, Truly(isRowMajorMatmul));
}

TEST(isRowMajorMatmul, BindingSwapped) {
  MLIRContext context;

  AffineExpr m, n, k;
  bindDims(&context, k, n, m); // bind in different order
  auto mapA = AffineMapAttr::get(AffineMap::get(3, 0, {m, k}, &context));
  auto mapB = AffineMapAttr::get(AffineMap::get(3, 0, {k, n}, &context));
  auto mapC = AffineMapAttr::get(AffineMap::get(3, 0, {m, n}, &context));
  auto maps = ArrayAttr::get(&context, {mapA, mapB, mapC});

  EXPECT_THAT(maps, Truly(isRowMajorMatmul));
}

TEST(isRowMajorMatmul, ColumnMajor) {
  MLIRContext context;

  AffineExpr m, n, k;
  bindDims(&context, m, n, k);
  auto mapA = AffineMapAttr::get(AffineMap::get(3, 0, {k, n}, &context));
  auto mapB = AffineMapAttr::get(AffineMap::get(3, 0, {m, k}, &context));
  auto mapC = AffineMapAttr::get(AffineMap::get(3, 0, {m, n}, &context));
  auto maps = ArrayAttr::get(&context, {mapA, mapB, mapC});

  EXPECT_THAT(maps, Not(Truly(isRowMajorMatmul)));
}

TEST(isRowMajorMatmul, FirstInputSwapped) {
  MLIRContext context;

  AffineExpr m, n, k;
  bindDims(&context, m, n, k);
  auto mapA = AffineMapAttr::get(AffineMap::get(3, 0, {k, m}, &context));
  auto mapB = AffineMapAttr::get(AffineMap::get(3, 0, {k, n}, &context));
  auto mapC = AffineMapAttr::get(AffineMap::get(3, 0, {m, n}, &context));
  auto maps = ArrayAttr::get(&context, {mapA, mapB, mapC});

  EXPECT_THAT(maps, Not(Truly(isRowMajorMatmul)));
}

TEST(isRowMajorMatmul, TooFewMaps) {
  MLIRContext context;

  AffineExpr m, n, k;
  bindDims(&context, m, n, k);
  auto mapA = AffineMapAttr::get(AffineMap::get(3, 0, {m, k}, &context));
  auto mapB = AffineMapAttr::get(AffineMap::get(3, 0, {k, n}, &context));
  auto maps = ArrayAttr::get(&context, {mapA, mapB});

  EXPECT_THAT(maps, Not(Truly(isRowMajorMatmul)));
}

TEST(isRowMajorMatmul, TooManyMaps) {
  MLIRContext context;

  AffineExpr m, n, k;
  bindDims(&context, m, n, k);
  auto mapA = AffineMapAttr::get(AffineMap::get(3, 0, {m, k}, &context));
  auto mapB = AffineMapAttr::get(AffineMap::get(3, 0, {k, n}, &context));
  auto mapC = AffineMapAttr::get(AffineMap::get(3, 0, {m, n}, &context));
  auto mapD = AffineMapAttr::get(AffineMap::get(3, 0, {k, m}, &context));

  auto maps = ArrayAttr::get(&context, {mapA, mapB, mapC, mapD});

  EXPECT_THAT(maps, Not(Truly(isRowMajorMatmul)));
}

TEST(isRowMajorMatmul, TooFewOutputs) {
  MLIRContext context;

  AffineExpr m, n, k;
  bindDims(&context, m, n, k);
  auto mapA = AffineMapAttr::get(AffineMap::get(3, 0, {m}, &context));
  auto mapB = AffineMapAttr::get(AffineMap::get(3, 0, {k, n}, &context));
  auto mapC = AffineMapAttr::get(AffineMap::get(3, 0, {m, n}, &context));
  auto maps = ArrayAttr::get(&context, {mapA, mapB, mapC});

  EXPECT_THAT(maps, Not(Truly(isRowMajorMatmul)));
}

TEST(isColumnMajorMatmul, Simple) {
  MLIRContext context;

  AffineExpr m, n, k;
  bindDims(&context, m, n, k);
  auto mapA = AffineMapAttr::get(AffineMap::get(3, 0, {k, n}, &context));
  auto mapB = AffineMapAttr::get(AffineMap::get(3, 0, {m, k}, &context));
  auto mapC = AffineMapAttr::get(AffineMap::get(3, 0, {m, n}, &context));
  auto maps = ArrayAttr::get(&context, {mapA, mapB, mapC});

  EXPECT_THAT(maps, Truly(isColumnMajorMatmul));
}

TEST(isColumnMajorMatmul, BindingShifted) {
  MLIRContext context;

  AffineExpr m, n, k;
  bindDims(&context, k, m, n); // bind in different order
  auto mapA = AffineMapAttr::get(AffineMap::get(3, 0, {k, n}, &context));
  auto mapB = AffineMapAttr::get(AffineMap::get(3, 0, {m, k}, &context));
  auto mapC = AffineMapAttr::get(AffineMap::get(3, 0, {m, n}, &context));
  auto maps = ArrayAttr::get(&context, {mapA, mapB, mapC});

  EXPECT_THAT(maps, Truly(isColumnMajorMatmul));
}

TEST(isColumnMajorMatmul, BindingSwapped) {
  MLIRContext context;

  AffineExpr m, n, k;
  bindDims(&context, k, n, m); // bind in different order
  auto mapA = AffineMapAttr::get(AffineMap::get(3, 0, {k, n}, &context));
  auto mapB = AffineMapAttr::get(AffineMap::get(3, 0, {m, k}, &context));
  auto mapC = AffineMapAttr::get(AffineMap::get(3, 0, {m, n}, &context));
  auto maps = ArrayAttr::get(&context, {mapA, mapB, mapC});

  EXPECT_THAT(maps, Truly(isColumnMajorMatmul));
}

TEST(isColumnMajorMatmul, RowMajor) {
  MLIRContext context;

  AffineExpr m, n, k;
  bindDims(&context, m, n, k);
  auto mapA = AffineMapAttr::get(AffineMap::get(3, 0, {m, k}, &context));
  auto mapB = AffineMapAttr::get(AffineMap::get(3, 0, {k, n}, &context));
  auto mapC = AffineMapAttr::get(AffineMap::get(3, 0, {m, n}, &context));
  auto maps = ArrayAttr::get(&context, {mapA, mapB, mapC});

  EXPECT_THAT(maps, Not(Truly(isColumnMajorMatmul)));
}

TEST(isColumnMajorMatmul, FirstInputSwapped) {
  MLIRContext context;

  AffineExpr m, n, k;
  bindDims(&context, m, n, k);
  auto mapA = AffineMapAttr::get(AffineMap::get(3, 0, {n, k}, &context));
  auto mapB = AffineMapAttr::get(AffineMap::get(3, 0, {m, k}, &context));
  auto mapC = AffineMapAttr::get(AffineMap::get(3, 0, {m, n}, &context));
  auto maps = ArrayAttr::get(&context, {mapA, mapB, mapC});

  EXPECT_THAT(maps, Not(Truly(isColumnMajorMatmul)));
}

TEST(isRowMajorBatchMatmul, Simple) {
  MLIRContext context;

  AffineExpr batch, m, n, k;
  bindDims(&context, batch, m, n, k);
  auto mapA = AffineMapAttr::get(AffineMap::get(4, 0, {batch, m, k}, &context));
  auto mapB = AffineMapAttr::get(AffineMap::get(4, 0, {batch, k, n}, &context));
  auto mapC = AffineMapAttr::get(AffineMap::get(4, 0, {batch, m, n}, &context));
  auto maps = ArrayAttr::get(&context, {mapA, mapB, mapC});

  EXPECT_THAT(maps, Truly(isRowMajorBatchMatmul));
}

TEST(isRowMajorBatchMatmul, BindingShifted) {
  MLIRContext context;

  AffineExpr batch, m, n, k;
  bindDims(&context, k, batch, m, n); // bind in different order
  auto mapA = AffineMapAttr::get(AffineMap::get(4, 0, {batch, m, k}, &context));
  auto mapB = AffineMapAttr::get(AffineMap::get(4, 0, {batch, k, n}, &context));
  auto mapC = AffineMapAttr::get(AffineMap::get(4, 0, {batch, m, n}, &context));
  auto maps = ArrayAttr::get(&context, {mapA, mapB, mapC});

  EXPECT_THAT(maps, Truly(isRowMajorBatchMatmul));
}

TEST(isRowMajorBatchMatmul, BindingSwapped) {
  MLIRContext context;

  AffineExpr batch, m, n, k;
  bindDims(&context, batch, k, n, m); // bind in different order
  auto mapA = AffineMapAttr::get(AffineMap::get(4, 0, {batch, m, k}, &context));
  auto mapB = AffineMapAttr::get(AffineMap::get(4, 0, {batch, k, n}, &context));
  auto mapC = AffineMapAttr::get(AffineMap::get(4, 0, {batch, m, n}, &context));
  auto maps = ArrayAttr::get(&context, {mapA, mapB, mapC});

  EXPECT_THAT(maps, Truly(isRowMajorBatchMatmul));
}

TEST(isRowMajorBatchMatmul, FirstInputSwapped) {
  MLIRContext context;

  AffineExpr batch, m, n, k;
  bindDims(&context, batch, m, n, k);
  auto mapA = AffineMapAttr::get(AffineMap::get(4, 0, {batch, k, m}, &context));
  auto mapB = AffineMapAttr::get(AffineMap::get(4, 0, {batch, k, n}, &context));
  auto mapC = AffineMapAttr::get(AffineMap::get(4, 0, {batch, m, n}, &context));
  auto maps = ArrayAttr::get(&context, {mapA, mapB, mapC});

  EXPECT_THAT(maps, Not(Truly(isRowMajorBatchMatmul)));
}

} // namespace
