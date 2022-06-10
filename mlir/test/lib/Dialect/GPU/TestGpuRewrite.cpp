//===- TestAllReduceLowering.cpp - Test gpu.all_reduce lowering -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains test passes for lowering the gpu.all_reduce op.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {
struct TestGpuRewritePass
    : public PassWrapper<TestGpuRewritePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestGpuRewritePass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithmeticDialect, func::FuncDialect,
                    memref::MemRefDialect>();
  }
  StringRef getArgument() const final { return "test-gpu-rewrite"; }
  StringRef getDescription() const final {
    return "Applies all rewrite patterns within the GPU dialect.";
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateGpuRewritePatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};
} // namespace

namespace mlir {
void registerTestAllReduceLoweringPass() {
  PassRegistration<TestGpuRewritePass>();
}
} // namespace mlir
