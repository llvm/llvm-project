//===- MLIRContextResetTest.cpp - Tests for transient MLIRContext ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/Support/ThreadPool.h"
#include "gtest/gtest.h"

using namespace mlir;

namespace {

TEST(MLIRContextResetTest, BasicTypeTransientScopeAndReset) {
  MLIRContext ctx;
  OpBuilder builder(&ctx);

  // Base types created before transient scope.
  Type i32Type = builder.getI32Type();
  Type f32Type = builder.getF32Type();
  Type baseVectorType = VectorType::get({2, 2}, f32Type);

  EXPECT_FALSE(ctx.isInTransientScope());
  ctx.beginTransientScope();
  EXPECT_TRUE(ctx.isInTransientScope());

  // Base types remain identical under transient scope.
  EXPECT_EQ(i32Type, builder.getI32Type());
  EXPECT_EQ(f32Type, builder.getF32Type());
  EXPECT_EQ(baseVectorType, VectorType::get({2, 2}, f32Type));

  // Create transient types in transient scope.
  Type transientVectorType = VectorType::get({4, 8}, i32Type);
  Type transientTupleType = TupleType::get(&ctx, {i32Type, baseVectorType});

  // Verify uniquing in transient state.
  EXPECT_EQ(transientVectorType, VectorType::get({4, 8}, i32Type));
  EXPECT_EQ(transientTupleType,
            TupleType::get(&ctx, {i32Type, baseVectorType}));

  // End transient scope back to base state.
  ctx.endTransientScope();
  EXPECT_FALSE(ctx.isInTransientScope());

  // Verify base types are completely preserved.
  EXPECT_EQ(i32Type, builder.getI32Type());
  EXPECT_EQ(f32Type, builder.getF32Type());
  EXPECT_EQ(baseVectorType, VectorType::get({2, 2}, f32Type));

  // Types can be cleanly recreated post-reset.
  Type postResetVectorType = VectorType::get({4, 8}, i32Type);
  EXPECT_NE(postResetVectorType, Type());
  EXPECT_EQ(postResetVectorType, VectorType::get({4, 8}, i32Type));
}

TEST(MLIRContextResetTest, AttributeTransientScopeAndReset) {
  MLIRContext ctx;
  OpBuilder builder(&ctx);

  // Base attributes.
  StringAttr baseStr = builder.getStringAttr("base_identifier");
  UnitAttr baseUnit = builder.getUnitAttr();
  IntegerAttr baseInt = builder.getI32IntegerAttr(42);
  DistinctAttr baseDistinct = DistinctAttr::create(baseUnit);

  ctx.beginTransientScope();
  EXPECT_TRUE(ctx.isInTransientScope());

  // Transient attributes.
  StringAttr transientStr = builder.getStringAttr("transient_identifier");
  IntegerAttr transientInt = builder.getI32IntegerAttr(100);
  ArrayAttr transientArray = builder.getArrayAttr({baseInt, transientInt});
  DictionaryAttr transientDict =
      builder.getDictionaryAttr({builder.getNamedAttr("key", transientStr)});
  DistinctAttr transientDistinct = DistinctAttr::create(transientInt);

  EXPECT_EQ(transientStr, builder.getStringAttr("transient_identifier"));
  EXPECT_EQ(transientArray, builder.getArrayAttr({baseInt, transientInt}));
  EXPECT_EQ(transientDict, builder.getDictionaryAttr(
                               {builder.getNamedAttr("key", transientStr)}));
  EXPECT_EQ(transientDistinct.getReferencedAttr(), transientInt);

  // End transient scope.
  ctx.endTransientScope();
  EXPECT_FALSE(ctx.isInTransientScope());

  // Verify base attributes.
  EXPECT_EQ(baseStr, builder.getStringAttr("base_identifier"));
  EXPECT_EQ(baseUnit, builder.getUnitAttr());
  EXPECT_EQ(baseInt, builder.getI32IntegerAttr(42));
  EXPECT_NE(baseDistinct, DistinctAttr());

  // Re-create attributes post-reset.
  StringAttr postResetStr = builder.getStringAttr("transient_identifier");
  EXPECT_EQ(postResetStr.getValue(), "transient_identifier");
}

TEST(MLIRContextResetTest, AffineTransientScopeAndReset) {
  MLIRContext ctx;

  // Base affine expression and map.
  AffineExpr d0 = getAffineDimExpr(0, &ctx);
  AffineExpr d1 = getAffineDimExpr(1, &ctx);
  AffineMap baseMap = AffineMap::get(2, 0, {d0 + d1}, &ctx);

  ctx.beginTransientScope();

  // Transient affine expressions and maps.
  AffineExpr c42 = getAffineConstantExpr(42, &ctx);
  AffineMap transientMap = AffineMap::get(2, 0, {d0 * 4 + d1 + c42}, &ctx);

  EXPECT_EQ(transientMap, AffineMap::get(2, 0, {d0 * 4 + d1 + c42}, &ctx));

  ctx.endTransientScope();

  // Verify base map.
  EXPECT_EQ(baseMap, AffineMap::get(2, 0, {d0 + d1}, &ctx));

  // Recreate map post reset.
  AffineExpr newC42 = getAffineConstantExpr(42, &ctx);
  AffineMap newMap = AffineMap::get(2, 0, {d0 * 4 + d1 + newC42}, &ctx);
  EXPECT_EQ(newMap.getNumResults(), 1u);
}

TEST(MLIRContextResetTest, UnregisteredOperationPruning) {
  MLIRContext ctx;
  ctx.allowUnregisteredDialects(true);

  // Base unregistered op.
  OperationName baseOpName("custom_base.op", &ctx);

  ctx.beginTransientScope();

  // Transient unregistered op.
  OperationName transientOpName("custom_transient.op", &ctx);
  EXPECT_EQ(transientOpName.getStringRef(), "custom_transient.op");

  ctx.endTransientScope();

  // Base op is still valid and lookup succeeds.
  OperationName baseOpNameAfter("custom_base.op", &ctx);
  EXPECT_EQ(baseOpName, baseOpNameAfter);
}

TEST(MLIRContextResetTest, MultithreadedTransientScopeAndReset) {
  MLIRContext ctx;
  OpBuilder builder(&ctx);

  Type i32Type = builder.getI32Type();

  ctx.beginTransientScope();

  // Allocate in parallel across multiple threads during transient scope.
  llvm::DefaultThreadPool pool;
  for (int i = 0; i < 20; ++i) {
    pool.async([&ctx, i, i32Type]() {
      for (int j = 0; j < 50; ++j) {
        (void)VectorType::get({i + 1, j + 1}, i32Type);
        (void)StringAttr::get(&ctx, "thread_str_" + std::to_string(i) + "_" +
                                        std::to_string(j));
      }
    });
  }
  pool.wait();

  // End transient scope back to base.
  ctx.endTransientScope();

  // Allocate again in parallel across multiple threads.
  for (int i = 0; i < 20; ++i) {
    pool.async([i, i32Type]() {
      for (int j = 0; j < 50; ++j) {
        Type ty = VectorType::get({i + 1, j + 1}, i32Type);
        EXPECT_TRUE(isa<VectorType>(ty));
      }
    });
  }
  pool.wait();

  EXPECT_EQ(i32Type, builder.getI32Type());
}

} // namespace
