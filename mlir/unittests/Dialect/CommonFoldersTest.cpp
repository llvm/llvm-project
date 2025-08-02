//===- CommonFoldersTest.cpp - tests for folder-pattern helper templates --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/CommonFolders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/Casting.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace mlir {
namespace {

using ::llvm::APFloat;
using ::llvm::APInt;
using ::mlir::constFoldBinaryOp;
using ::mlir::DenseElementsAttr;
using ::mlir::Float32Type;
using ::mlir::FloatAttr;
using ::mlir::IntegerAttr;
using ::mlir::IntegerType;
using ::mlir::MLIRContext;
using ::mlir::RankedTensorType;
using ::testing::ElementsAre;

APInt floatLessThan(APFloat lhs, APFloat rhs) { return APInt(1, lhs < rhs); }

TEST(CommonFoldersTest, FoldFloatComparisonToBoolean) {
  MLIRContext context;
  auto vector4xf32 = RankedTensorType::get({4}, Float32Type::get(&context));

  auto lhs = DenseElementsAttr::get(vector4xf32, {-12.9f, 0.0f, 42.5f, -0.01f});
  auto rhs = DenseElementsAttr::get(vector4xf32, {0.0f, 0.0f, 0.0f, 0.0f});

  auto result = llvm::dyn_cast<DenseElementsAttr>(
      constFoldBinaryOp<FloatAttr, FloatAttr::ValueType, void, IntegerAttr>(
          {lhs, rhs}, RankedTensorType::get({4}, IntegerType::get(&context, 1)),
          floatLessThan));
  ASSERT_TRUE(result);

  auto resultElementType = result.getElementType();
  EXPECT_TRUE(resultElementType.isInteger(1));

  const APInt i1True = APInt(1, true);
  const APInt i1False = APInt(1, false);

  EXPECT_THAT(result.getValues<APInt>(),
              ElementsAre(i1True, i1False, i1False, i1True));
}

} // namespace
} // namespace mlir
