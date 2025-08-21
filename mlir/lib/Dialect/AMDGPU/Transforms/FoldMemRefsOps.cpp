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

static LogicalResult foldMemrefViewOp(PatternRewriter &rewriter, Location loc,
                                      Value view, mlir::OperandRange indices,
                                      SmallVectorImpl<Value> &resolvedIndices,
                                      Value &memrefBase, StringRef role) {
  Operation *defOp = view.getDefiningOp();
  if (!defOp) {
    return failure();
  }
  return llvm::TypeSwitch<Operation *, LogicalResult>(defOp)
      .Case<memref::SubViewOp>([&](memref::SubViewOp subviewOp) {
        mlir::affine::resolveIndicesIntoOpWithOffsetsAndStrides(
            rewriter, loc, subviewOp.getMixedOffsets(),
            subviewOp.getMixedStrides(), subviewOp.getDroppedDims(), indices,
            resolvedIndices);
        memrefBase = subviewOp.getSource();
        return success();
      })
      .Case<memref::ExpandShapeOp>([&](memref::ExpandShapeOp expandShapeOp) {
        if (failed(mlir::memref::resolveSourceIndicesExpandShape(
                loc, rewriter, expandShapeOp, indices, resolvedIndices,
                false))) {
          return failure();
        }
        memrefBase = expandShapeOp.getViewSource();
        return success();
      })
      .Case<memref::CollapseShapeOp>(
          [&](memref::CollapseShapeOp collapseShapeOp) {
            if (failed(mlir::memref::resolveSourceIndicesCollapseShape(
                    loc, rewriter, collapseShapeOp, indices,
                    resolvedIndices))) {
              return failure();
            }
            memrefBase = collapseShapeOp.getViewSource();
            return success();
          })
      .Default([&](Operation *op) {
        return rewriter.notifyMatchFailure(
            op, (role + " producer is not one of SubViewOp, ExpandShapeOp, or "
                        "CollapseShapeOp")
                    .str());
      });
}

struct FoldMemRefOpsIntoGatherToLDSOp final : OpRewritePattern<GatherToLDSOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(GatherToLDSOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    SmallVector<Value> sourceIndices, destIndices;
    Value memrefSource, memrefDest;

    auto foldSrcResult =
        foldMemrefViewOp(rewriter, loc, op.getSrc(), op.getSrcIndices(),
                         sourceIndices, memrefSource, "source");

    if (failed(foldSrcResult)) {
      memrefSource = op.getSrc();
      sourceIndices = op.getSrcIndices();
    }

    auto foldDstResult =
        foldMemrefViewOp(rewriter, loc, op.getDst(), op.getDstIndices(),
                         destIndices, memrefDest, "destination");

    if (failed(foldDstResult)) {
      memrefDest = op.getDst();
      destIndices = op.getDstIndices();
    }

    rewriter.replaceOpWithNewOp<GatherToLDSOp>(op, memrefSource, sourceIndices,
                                               memrefDest, destIndices,
                                               op.getTransferType());

    return success();
  }
};

void populateAmdgpuFoldMemRefOpsPatterns(RewritePatternSet &patterns,
                                         PatternBenefit benefit) {
  patterns.add<FoldMemRefOpsIntoGatherToLDSOp>(patterns.getContext(), benefit);
}
} // namespace mlir::amdgpu
