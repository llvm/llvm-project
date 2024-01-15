//===- ResolveConditionalExecution.cpp - Resolve conditional exec ops ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the `gpu-resolve-conditional-execution` pass.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::gpu;

namespace mlir {
#define GEN_PASS_DEF_GPURESOLVECONDITIONALEXECUTIONPASS
#include "mlir/Dialect/GPU/Transforms/Passes.h.inc"
} // namespace mlir

namespace {
class GpuResolveConditionalExecutionPass
    : public impl::GpuResolveConditionalExecutionPassBase<
          GpuResolveConditionalExecutionPass> {
public:
  using Base::Base;
  void runOnOperation() final;
};
} // namespace

void GpuResolveConditionalExecutionPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  mlir::populateGpuConditionalExecutionPatterns(patterns);
  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    return signalPassFailure();
}

namespace {
struct GpuConditionalExecutionOpRewriter
    : public OpRewritePattern<ConditionalExecutionOp> {
  using OpRewritePattern<ConditionalExecutionOp>::OpRewritePattern;
  // Check whether the operation is inside a device execution context.
  bool isDevice(Operation *op) const {
    while ((op = op->getParentOp()))
      if (isa<GPUFuncOp, LaunchOp, GPUModuleOp>(op))
        return true;
    return false;
  }
  LogicalResult matchAndRewrite(ConditionalExecutionOp op,
                                PatternRewriter &rewriter) const override {
    bool isDev = isDevice(op);
    // Remove the op if the device region is empty and we are in a device
    // context.
    if (isDev && op.getDeviceRegion().empty()) {
      rewriter.eraseOp(op);
      return success();
    }
    // Remove the op if the host region is empty and we are in a host context.
    if (!isDev && op.getHostRegion().empty()) {
      rewriter.eraseOp(op);
      return success();
    }
    // Replace `ConditionalExecutionOp` with a `scf::ExecuteRegionOp`.
    auto execRegionOp = rewriter.create<scf::ExecuteRegionOp>(
        op.getLoc(), op.getResults().getTypes());
    if (isDev)
      rewriter.inlineRegionBefore(op.getDeviceRegion(),
                                  execRegionOp.getRegion(),
                                  execRegionOp.getRegion().begin());
    else
      rewriter.inlineRegionBefore(op.getHostRegion(), execRegionOp.getRegion(),
                                  execRegionOp.getRegion().begin());
    rewriter.eraseOp(op);
    // This is safe because `ConditionalExecutionOp` always terminates with
    // `gpu::YieldOp`
    auto yieldOp =
        dyn_cast<YieldOp>(execRegionOp.getRegion().back().getTerminator());
    rewriter.setInsertionPoint(yieldOp);
    rewriter.replaceOpWithNewOp<scf::YieldOp>(yieldOp, yieldOp.getValues());
    return success();
  }
};
} // namespace

void mlir::populateGpuConditionalExecutionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<GpuConditionalExecutionOpRewriter>(patterns.getContext());
}
