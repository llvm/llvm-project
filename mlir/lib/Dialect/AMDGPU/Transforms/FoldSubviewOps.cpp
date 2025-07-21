//===- FoldSubviewOps.cpp - AMDGPU fold subview ops ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/AMDGPU/Transforms/Passes.h"

#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Affine/ViewLikeInterfaceUtils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::amdgpu {
#define GEN_PASS_DEF_AMDGPUFOLDSUBVIEWOPSPASS
#include "mlir/Dialect/AMDGPU/Transforms/Passes.h.inc"
} // namespace mlir::amdgpu

using namespace mlir;
using namespace mlir::amdgpu;

namespace {
struct AmdgpuFoldSubviewOpsPass
    : public amdgpu::impl::AmdgpuFoldSubviewOpsPassBase<
          AmdgpuFoldSubviewOpsPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateAmdgpuFoldSubviewOpsPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

struct FoldSubviewIntoGatherToLDSOp : public OpRewritePattern<GatherToLDSOp> {
  using OpRewritePattern<GatherToLDSOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(GatherToLDSOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Check if the source is a subview operation:
    auto subviewOp = dyn_cast<memref::SubViewOp>(op.getSrc().getDefiningOp());
    if (!subviewOp)
      return rewriter.notifyMatchFailure(
          loc, "GatherToLDSOp folding is currently supported only when the source is a SubviewOp. This is one specific pattern, and other scenarios may be added in the future.");

    SmallVector<Value> sourceIndices;
    mlir::affine::resolveIndicesIntoOpWithOffsetsAndStrides(
        rewriter, loc, subviewOp.getMixedOffsets(), subviewOp.getMixedStrides(),
        subviewOp.getDroppedDims(), op.getSrcIndices(), sourceIndices);

    rewriter.replaceOpWithNewOp<GatherToLDSOp>(
        op, subviewOp.getSource(), sourceIndices, op.getDst(),
        op.getDstIndices(), op.getTransferType());

    return success();
  }
};
} // namespace

void mlir::amdgpu::populateAmdgpuFoldSubviewOpsPatterns(
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<FoldSubviewIntoGatherToLDSOp>(patterns.getContext(), benefit);
}
