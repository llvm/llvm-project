//===- BuildOnlyExtensionTest.cpp - unit test for transform extensions ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::transform;

namespace {
class Extension : public TransformDialectExtension<Extension> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(Extension)

  using Base::Base;
  void init() { declareGeneratedDialect<func::FuncDialect>(); }
};
} // end namespace

TEST(BuildOnlyExtensionTest, buildOnlyExtension) {
  // Register the build-only version of the transform dialect extension. The
  // func dialect is declared as generated so it should not be loaded along with
  // the transform dialect.
  DialectRegistry registry;
  registry.addExtensions<BuildOnly<Extension>>();
  MLIRContext ctx(registry);
  ctx.getOrLoadDialect<TransformDialect>();
  ASSERT_FALSE(ctx.getLoadedDialect<func::FuncDialect>());
}

TEST(BuildOnlyExtensionTest, buildAndApplyExtension) {
  // Register the full version of the transform dialect extension. The func
  // dialect should be loaded along with the transform dialect.
  DialectRegistry registry;
  registry.addExtensions<Extension>();
  MLIRContext ctx(registry);
  ctx.getOrLoadDialect<TransformDialect>();
  ASSERT_TRUE(ctx.getLoadedDialect<func::FuncDialect>());
}
