//===- FoldMemRefsOps.cpp - AMDGPU fold memref ops ------------------------===//
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
      .Case([&](memref::SubViewOp subviewOp) {
        mlir::affine::resolveIndicesIntoOpWithOffsetsAndStrides(
            rewriter, loc, subviewOp.getMixedOffsets(),
            subviewOp.getMixedStrides(), subviewOp.getDroppedDims(), indices,
            resolvedIndices);
        memrefBase = subviewOp.getSource();
        return success();
      })
      .Case([&](memref::ExpandShapeOp expandShapeOp) {
        mlir::memref::resolveSourceIndicesExpandShape(
            loc, rewriter, expandShapeOp, indices, resolvedIndices, false);
        memrefBase = expandShapeOp.getViewSource();
        return success();
      })
      .Case([&](memref::CollapseShapeOp collapseShapeOp) {
        mlir::memref::resolveSourceIndicesCollapseShape(
            loc, rewriter, collapseShapeOp, indices, resolvedIndices);
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

static LogicalResult
foldMemrefViewOps(PatternRewriter &rewriter, Operation *op, Value source,
                  mlir::OperandRange sourceIndices, Value dest,
                  mlir::OperandRange destIndices,
                  SmallVectorImpl<Value> &resolvedSourceIndices,
                  SmallVectorImpl<Value> &resolvedDestIndices,
                  Value &sourceBase, Value &destBase) {
  Location loc = op->getLoc();

  LogicalResult didFoldSource =
      foldMemrefViewOp(rewriter, loc, source, sourceIndices,
                       resolvedSourceIndices, sourceBase, "source");
  if (failed(didFoldSource)) {
    sourceBase = source;
    resolvedSourceIndices.assign(sourceIndices.begin(), sourceIndices.end());
  }

  LogicalResult didFoldDest =
      foldMemrefViewOp(rewriter, loc, dest, destIndices, resolvedDestIndices,
                       destBase, "destination");
  if (failed(didFoldDest)) {
    destBase = dest;
    resolvedDestIndices.assign(destIndices.begin(), destIndices.end());
  }

  if (failed(didFoldSource) && failed(didFoldDest))
    return rewriter.notifyMatchFailure(op, "no fold found");

  return success();
}

struct FoldMemRefOpsIntoGatherToLDSOp final : OpRewritePattern<GatherToLDSOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(GatherToLDSOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> sourceIndices, destIndices;
    Value memrefSource, memrefDest;

    if (failed(foldMemrefViewOps(rewriter, op, op.getSrc(), op.getSrcIndices(),
                                 op.getDst(), op.getDstIndices(), sourceIndices,
                                 destIndices, memrefSource, memrefDest)))
      return failure();

    rewriter.replaceOpWithNewOp<GatherToLDSOp>(
        op, memrefSource, sourceIndices, memrefDest, destIndices,
        op.getTransferType(), op.getAsync());

    return success();
  }
};

struct FoldMemRefOpsIntoGlobalLoadAsyncToLDSOp final
    : OpRewritePattern<GlobalLoadAsyncToLDSOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(GlobalLoadAsyncToLDSOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> sourceIndices, destIndices;
    Value memrefSource, memrefDest;

    if (failed(foldMemrefViewOps(rewriter, op, op.getSrc(), op.getSrcIndices(),
                                 op.getDst(), op.getDstIndices(), sourceIndices,
                                 destIndices, memrefSource, memrefDest)))
      return failure();

    rewriter.replaceOpWithNewOp<GlobalLoadAsyncToLDSOp>(
        op, memrefSource, sourceIndices, memrefDest, destIndices,
        op.getTransferType(), op.getMask());

    return success();
  }
};

template <typename OpTy>
struct FoldMemRefOpsIntoDmaBaseOp final : OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    SmallVector<Value> globalIndices, ldsIndices;
    Value globalBase, ldsBase;

    LogicalResult didFoldGlobal =
        foldMemrefViewOp(rewriter, loc, op.getGlobal(), op.getGlobalIndices(),
                         globalIndices, globalBase, "global");
    if (failed(didFoldGlobal)) {
      globalBase = op.getGlobal();
      globalIndices = op.getGlobalIndices();
    }

    LogicalResult didFoldLds =
        foldMemrefViewOp(rewriter, loc, op.getLds(), op.getLdsIndices(),
                         ldsIndices, ldsBase, "lds");
    if (failed(didFoldLds)) {
      ldsBase = op.getLds();
      ldsIndices = op.getLdsIndices();
    }

    if (failed(didFoldGlobal) && failed(didFoldLds))
      return rewriter.notifyMatchFailure(op, "no fold found");

    rewriter.replaceOpWithNewOp<OpTy>(op, op.getBase().getType(), globalBase,
                                      globalIndices, ldsBase, ldsIndices);
    return success();
  }
};

struct FoldMemRefOpsIntoTransposeLoadOp final
    : OpRewritePattern<TransposeLoadOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(TransposeLoadOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> sourceIndices;
    Value memrefSource;

    if (failed(foldMemrefViewOp(rewriter, op.getLoc(), op.getSrc(),
                                op.getSrcIndices(), sourceIndices, memrefSource,
                                "source")))
      return failure();

    rewriter.replaceOpWithNewOp<TransposeLoadOp>(op, op.getResult().getType(),
                                                 memrefSource, sourceIndices);
    return success();
  }
};

void populateAmdgpuFoldMemRefOpsPatterns(RewritePatternSet &patterns,
                                         PatternBenefit benefit) {
  patterns.add<FoldMemRefOpsIntoGatherToLDSOp,
               FoldMemRefOpsIntoGlobalLoadAsyncToLDSOp,
               FoldMemRefOpsIntoDmaBaseOp<MakeDmaBaseOp>,
               FoldMemRefOpsIntoDmaBaseOp<MakeGatherDmaBaseOp>,
               FoldMemRefOpsIntoTransposeLoadOp>(patterns.getContext(),
                                                 benefit);
}
} // namespace mlir::amdgpu
