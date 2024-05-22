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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {
struct TestGpuRewritePass
    : public PassWrapper<TestGpuRewritePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestGpuRewritePass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect, index::IndexDialect,
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

struct TestGpuSubgroupReduceLoweringPass
    : public PassWrapper<TestGpuSubgroupReduceLoweringPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      TestGpuSubgroupReduceLoweringPass)

  TestGpuSubgroupReduceLoweringPass() = default;
  TestGpuSubgroupReduceLoweringPass(
      const TestGpuSubgroupReduceLoweringPass &pass)
      : PassWrapper(pass) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, vector::VectorDialect>();
  }

  StringRef getArgument() const final {
    return "test-gpu-subgroup-reduce-lowering";
  }

  StringRef getDescription() const final {
    return "Applies gpu.subgroup_reduce lowering patterns.";
  }

  Option<bool> expandToShuffles{
      *this, "expand-to-shuffles",
      llvm::cl::desc("Expand subgroup_reduce ops to shuffle ops."),
      llvm::cl::init(false)};

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());

    // Since both pattern sets match on the same ops, set higher benefit to
    // perform fewer failing matches.
    populateGpuBreakDownSubgrupReducePatterns(patterns,
                                              /*maxShuffleBitwidth=*/32,
                                              PatternBenefit(2));
    if (expandToShuffles)
      populateGpuLowerSubgroupReduceToShufflePattenrs(
          patterns, /*subgroupSize=*/32, /*shuffleBitwidth=*/32);

    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};
} // namespace

namespace mlir {
void registerTestGpuLoweringPasses() {
  PassRegistration<TestGpuRewritePass>();
  PassRegistration<TestGpuSubgroupReduceLoweringPass>();
}
} // namespace mlir
