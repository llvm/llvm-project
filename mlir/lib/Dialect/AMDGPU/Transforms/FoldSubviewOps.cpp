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
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"

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

struct FoldSubviewIntoGatherToLDSOp final : OpRewritePattern<GatherToLDSOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(GatherToLDSOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value memrefSource;
    SmallVector<Value> sourceIndices;
    llvm::TypeSwitch<Operation *>(op.getSrc().getDefiningOp())
        .Case<memref::SubViewOp>([&](memref::SubViewOp subviewOp) {
          // If the source is a SubViewOp, we can directly rewrite the
          // GatherToLDSOp.
          mlir::affine::resolveIndicesIntoOpWithOffsetsAndStrides(
              rewriter, loc, subviewOp.getMixedOffsets(),
              subviewOp.getMixedStrides(), subviewOp.getDroppedDims(),
              op.getSrcIndices(), sourceIndices);
          memrefSource = subviewOp.getSource();
        })
        .Case<memref::ExpandShapeOp>([&](memref::ExpandShapeOp expandShapeOp) {
          mlir::memref::resolveSourceIndicesExpandShape(
              loc, rewriter, expandShapeOp, op.getSrcIndices(), sourceIndices,
              false);
          memrefSource = expandShapeOp.getViewSource();
        })
        .Case<memref::CollapseShapeOp>(
            [&](memref::CollapseShapeOp collapseShapeOp) {
              mlir::memref::resolveSourceIndicesCollapseShape(
                  loc, rewriter, collapseShapeOp, op.getSrcIndices(),
                  sourceIndices);
              memrefSource = collapseShapeOp.getViewSource();
            });

    if (!memrefSource)
      return failure();

    rewriter.replaceOpWithNewOp<GatherToLDSOp>(op, memrefSource, sourceIndices,
                                               op.getDst(), op.getDstIndices(),
                                               op.getTransferType());

    return success();
  }
};
} // namespace

void mlir::amdgpu::populateAmdgpuFoldSubviewOpsPatterns(
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<FoldSubviewIntoGatherToLDSOp>(patterns.getContext(), benefit);
}
