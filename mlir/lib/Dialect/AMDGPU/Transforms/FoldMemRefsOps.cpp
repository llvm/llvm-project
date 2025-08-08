//===- FoldSubviewOps.cpp - AMDGPU fold subview ops -----------------------===//
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
#include "mlir/Transforms/WalkPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir::amdgpu {
#define GEN_PASS_DEF_AMDGPUFOLDMEMREFOPSPASS
#include "mlir/Dialect/AMDGPU/Transforms/Passes.h.inc"

struct AmdgpuFoldMemRefOpsPass final
    : amdgpu::impl::AmdgpuFoldMemRefOpsPassBase<AmdgpuFoldMemRefOpsPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateAmdgpuFoldMemRefOpsPatterns(patterns);
    walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};

struct FoldMemRefOpsIntoGatherToLDSOp final : OpRewritePattern<GatherToLDSOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(GatherToLDSOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value memrefSource;
    SmallVector<Value> sourceIndices;
    auto foldResult =
        llvm::TypeSwitch<Operation *, LogicalResult>(
            op.getSrc().getDefiningOp())
            .Case<memref::SubViewOp>([&](memref::SubViewOp subviewOp) {
              // If the source is a SubViewOp, we can directly rewrite the
              // GatherToLDSOp.
              mlir::affine::resolveIndicesIntoOpWithOffsetsAndStrides(
                  rewriter, loc, subviewOp.getMixedOffsets(),
                  subviewOp.getMixedStrides(), subviewOp.getDroppedDims(),
                  op.getSrcIndices(), sourceIndices);
              memrefSource = subviewOp.getSource();
              return success();
            })
            .Case<memref::ExpandShapeOp>(
                [&](memref::ExpandShapeOp expandShapeOp) {
                  if (failed(mlir::memref::resolveSourceIndicesExpandShape(
                          loc, rewriter, expandShapeOp, op.getSrcIndices(),
                          sourceIndices, false))) {
                    return failure();
                  }
                  memrefSource = expandShapeOp.getViewSource();
                  return success();
                })
            .Case<memref::CollapseShapeOp>(
                [&](memref::CollapseShapeOp collapseShapeOp) {
                  if (failed(mlir::memref::resolveSourceIndicesCollapseShape(
                          loc, rewriter, collapseShapeOp, op.getSrcIndices(),
                          sourceIndices))) {
                    return failure();
                  }
                  memrefSource = collapseShapeOp.getViewSource();
                  return success();
                })
            .Default([&](Operation *op) {
              // If the source is not a SubViewOp, ExpandShapeOp, or
              // CollapseShapeOp, we cannot fold the GatherToLDSOp.
              return rewriter.notifyMatchFailure(
                  op,
                  "source producer is not one of SubViewOp, ExpandShapeOp, or "
                  "CollapseShapeOp");
            });

    if (failed(foldResult)) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<GatherToLDSOp>(op, memrefSource, sourceIndices,
                                               op.getDst(), op.getDstIndices(),
                                               op.getTransferType());

    return success();
  }
};

void populateAmdgpuFoldMemRefOpsPatterns(RewritePatternSet &patterns,
                                         PatternBenefit benefit) {
  patterns.add<FoldMemRefOpsIntoGatherToLDSOp>(patterns.getContext(), benefit);
}
} // namespace mlir::amdgpu
