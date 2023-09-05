//===- DecomposeMemrefs.cpp - Decompose memrefs pass implementation -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements decompose memrefs pass.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_GPUDECOMPOSEMEMREFSPASS
#include "mlir/Dialect/GPU/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

static void setInsertionPointToStart(OpBuilder &builder, Value val) {
  if (auto parentOp = val.getDefiningOp()) {
    builder.setInsertionPointAfter(parentOp);
  } else {
    builder.setInsertionPointToStart(val.getParentBlock());
  }
}

static bool isInsideLaunch(Operation *op) {
  return op->getParentOfType<gpu::LaunchOp>();
}

static std::tuple<Value, OpFoldResult, SmallVector<OpFoldResult>>
getFlatOffsetAndStrides(OpBuilder &rewriter, Location loc, Value source,
                        ArrayRef<OpFoldResult> subOffsets,
                        ArrayRef<OpFoldResult> subStrides = std::nullopt) {
  auto sourceType = cast<MemRefType>(source.getType());
  auto sourceRank = static_cast<unsigned>(sourceType.getRank());

  memref::ExtractStridedMetadataOp newExtractStridedMetadata;
  {
    OpBuilder::InsertionGuard g(rewriter);
    setInsertionPointToStart(rewriter, source);
    newExtractStridedMetadata =
        rewriter.create<memref::ExtractStridedMetadataOp>(loc, source);
  }

  auto &&[sourceStrides, sourceOffset] = getStridesAndOffset(sourceType);

  auto getDim = [&](int64_t dim, Value dimVal) -> OpFoldResult {
    return ShapedType::isDynamic(dim) ? getAsOpFoldResult(dimVal)
                                      : rewriter.getIndexAttr(dim);
  };

  OpFoldResult origOffset =
      getDim(sourceOffset, newExtractStridedMetadata.getOffset());
  ValueRange sourceStridesVals = newExtractStridedMetadata.getStrides();

  SmallVector<OpFoldResult> origStrides;
  origStrides.reserve(sourceRank);

  SmallVector<OpFoldResult> strides;
  strides.reserve(sourceRank);

  AffineExpr s0 = rewriter.getAffineSymbolExpr(0);
  AffineExpr s1 = rewriter.getAffineSymbolExpr(1);
  for (auto i : llvm::seq(0u, sourceRank)) {
    OpFoldResult origStride = getDim(sourceStrides[i], sourceStridesVals[i]);

    if (!subStrides.empty()) {
      strides.push_back(affine::makeComposedFoldedAffineApply(
          rewriter, loc, s0 * s1, {subStrides[i], origStride}));
    }

    origStrides.emplace_back(origStride);
  }

  auto &&[expr, values] =
      computeLinearIndex(origOffset, origStrides, subOffsets);
  OpFoldResult finalOffset =
      affine::makeComposedFoldedAffineApply(rewriter, loc, expr, values);
  return {newExtractStridedMetadata.getBaseBuffer(), finalOffset, strides};
}

static Value getFlatMemref(OpBuilder &rewriter, Location loc, Value source,
                           ValueRange offsets) {
  SmallVector<OpFoldResult> offsetsTemp = getAsOpFoldResult(offsets);
  auto &&[base, offset, ignore] =
      getFlatOffsetAndStrides(rewriter, loc, source, offsetsTemp);
  auto retType = cast<MemRefType>(base.getType());
  return rewriter.create<memref::ReinterpretCastOp>(loc, retType, base, offset,
                                                    std::nullopt, std::nullopt);
}

static bool needFlatten(Value val) {
  auto type = cast<MemRefType>(val.getType());
  return type.getRank() != 0;
}

static bool checkLayout(Value val) {
  auto type = cast<MemRefType>(val.getType());
  return type.getLayout().isIdentity() ||
         isa<StridedLayoutAttr>(type.getLayout());
}

namespace {
struct FlattenLoad : public OpRewritePattern<memref::LoadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::LoadOp op,
                                PatternRewriter &rewriter) const override {
    if (!isInsideLaunch(op))
      return rewriter.notifyMatchFailure(op, "not inside gpu.launch");

    Value memref = op.getMemref();
    if (!needFlatten(memref))
      return rewriter.notifyMatchFailure(op, "nothing to do");

    if (!checkLayout(memref))
      return rewriter.notifyMatchFailure(op, "unsupported layout");

    Location loc = op.getLoc();
    Value flatMemref = getFlatMemref(rewriter, loc, memref, op.getIndices());
    rewriter.replaceOpWithNewOp<memref::LoadOp>(op, flatMemref);
    return success();
  }
};

struct FlattenStore : public OpRewritePattern<memref::StoreOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::StoreOp op,
                                PatternRewriter &rewriter) const override {
    if (!isInsideLaunch(op))
      return rewriter.notifyMatchFailure(op, "not inside gpu.launch");

    Value memref = op.getMemref();
    if (!needFlatten(memref))
      return rewriter.notifyMatchFailure(op, "nothing to do");

    if (!checkLayout(memref))
      return rewriter.notifyMatchFailure(op, "unsupported layout");

    Location loc = op.getLoc();
    Value flatMemref = getFlatMemref(rewriter, loc, memref, op.getIndices());
    Value value = op.getValue();
    rewriter.replaceOpWithNewOp<memref::StoreOp>(op, value, flatMemref);
    return success();
  }
};

struct FlattenSubview : public OpRewritePattern<memref::SubViewOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::SubViewOp op,
                                PatternRewriter &rewriter) const override {
    if (!isInsideLaunch(op))
      return rewriter.notifyMatchFailure(op, "not inside gpu.launch");

    Value memref = op.getSource();
    if (!needFlatten(memref))
      return rewriter.notifyMatchFailure(op, "nothing to do");

    if (!checkLayout(memref))
      return rewriter.notifyMatchFailure(op, "unsupported layout");

    Location loc = op.getLoc();
    SmallVector<OpFoldResult> subOffsets = op.getMixedOffsets();
    SmallVector<OpFoldResult> subSizes = op.getMixedSizes();
    SmallVector<OpFoldResult> subStrides = op.getMixedStrides();
    auto &&[base, finalOffset, strides] =
        getFlatOffsetAndStrides(rewriter, loc, memref, subOffsets, subStrides);

    auto srcType = cast<MemRefType>(memref.getType());
    auto resultType = cast<MemRefType>(op.getType());
    unsigned subRank = static_cast<unsigned>(resultType.getRank());

    llvm::SmallBitVector droppedDims = op.getDroppedDims();

    SmallVector<OpFoldResult> finalSizes;
    finalSizes.reserve(subRank);

    SmallVector<OpFoldResult> finalStrides;
    finalStrides.reserve(subRank);

    for (auto i : llvm::seq(0u, static_cast<unsigned>(srcType.getRank()))) {
      if (droppedDims.test(i))
        continue;

      finalSizes.push_back(subSizes[i]);
      finalStrides.push_back(strides[i]);
    }

    rewriter.replaceOpWithNewOp<memref::ReinterpretCastOp>(
        op, resultType, base, finalOffset, finalSizes, finalStrides);
    return success();
  }
};

struct GpuDecomposeMemrefsPass
    : public impl::GpuDecomposeMemrefsPassBase<GpuDecomposeMemrefsPass> {

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());

    populateGpuDecomposeMemrefsPatterns(patterns);

    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

void mlir::populateGpuDecomposeMemrefsPatterns(RewritePatternSet &patterns) {
  patterns.insert<FlattenLoad, FlattenStore, FlattenSubview>(
      patterns.getContext());
}

std::unique_ptr<Pass> mlir::createGpuDecomposeMemrefsPass() {
  return std::make_unique<GpuDecomposeMemrefsPass>();
}
