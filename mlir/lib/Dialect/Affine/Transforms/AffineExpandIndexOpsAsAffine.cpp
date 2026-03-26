//===- AffineExpandIndexOpsAsAffine.cpp - Expand index ops to apply pass --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to expand affine index ops into one or more more
// fundamental operations.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Transforms/Transforms.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace affine {
#define GEN_PASS_DEF_AFFINEEXPANDINDEXOPSASAFFINE
#include "mlir/Dialect/Affine/Transforms/Passes.h.inc"
} // namespace affine
} // namespace mlir

using namespace mlir;
using namespace mlir::affine;

namespace {
/// Lowers `affine.delinearize_index` into a sequence of division and remainder
/// operations via affine.apply. For vector types, unrolls to per-element
/// scalar affine.apply operations.
struct LowerDelinearizeIndexOps
    : public OpRewritePattern<AffineDelinearizeIndexOp> {
  using OpRewritePattern<AffineDelinearizeIndexOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(AffineDelinearizeIndexOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value linearIndex = op.getLinearIndex();
    auto vecTy = dyn_cast<VectorType>(linearIndex.getType());

    // Scalar case: use the existing affine lowering path.
    if (!vecTy) {
      FailureOr<SmallVector<Value>> multiIndex =
          delinearizeIndex(rewriter, loc, linearIndex, op.getEffectiveBasis(),
                           /*hasOuterBound=*/false);
      if (failed(multiIndex))
        return failure();
      rewriter.replaceOp(op, *multiIndex);
      return success();
    }

    // Vector case: unroll to per-element scalar affine.apply operations.
    if (vecTy.isScalable())
      return rewriter.notifyMatchFailure(op, "scalable vectors not supported");

    int64_t numElems = vecTy.getNumElements();
    unsigned numResults = op.getNumResults();

    // Initialize result vectors with a poison/undef-like value.
    SmallVector<Value> resultVecs(numResults);
    Value poison = ub::PoisonOp::create(rewriter, loc, vecTy);
    for (unsigned r = 0; r < numResults; ++r)
      resultVecs[r] = poison;

    for (int64_t i = 0; i < numElems; ++i) {
      // Extract scalar element.
      Value idx = arith::ConstantIndexOp::create(rewriter, loc, i);
      Value scalar = vector::ExtractOp::create(rewriter, loc, linearIndex, idx);

      // Apply scalar delinearization.
      FailureOr<SmallVector<Value>> scalarResults =
          delinearizeIndex(rewriter, loc, scalar, op.getEffectiveBasis(),
                           /*hasOuterBound=*/false);
      if (failed(scalarResults))
        return failure();

      // Insert results back into vectors.
      for (unsigned r = 0; r < numResults; ++r)
        resultVecs[r] = vector::InsertOp::create(
            rewriter, loc, (*scalarResults)[r], resultVecs[r], idx);
    }

    rewriter.replaceOp(op, resultVecs);
    return success();
  }
};

/// Lowers `affine.linearize_index` into a sequence of multiplications and
/// additions via affine.apply. For vector types, unrolls to per-element
/// scalar affine.apply operations.
struct LowerLinearizeIndexOps final : OpRewritePattern<AffineLinearizeIndexOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AffineLinearizeIndexOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto vecTy = dyn_cast<VectorType>(op.getLinearIndex().getType());

    // Scalar case: use the existing affine lowering path.
    if (!vecTy) {
      // Should be folded away, included here for safety.
      if (op.getMultiIndex().empty()) {
        rewriter.replaceOpWithNewOp<arith::ConstantIndexOp>(op, 0);
        return success();
      }

      SmallVector<OpFoldResult> multiIndex =
          getAsOpFoldResult(op.getMultiIndex());
      OpFoldResult linearIndex =
          linearizeIndex(rewriter, loc, multiIndex, op.getMixedBasis());
      Value linearIndexValue =
          getValueOrCreateConstantIntOp(rewriter, loc, linearIndex);
      rewriter.replaceOp(op, linearIndexValue);
      return success();
    }

    // Vector case: unroll to per-element scalar affine.apply operations.
    if (vecTy.isScalable())
      return rewriter.notifyMatchFailure(op, "scalable vectors not supported");

    int64_t numElems = vecTy.getNumElements();
    ValueRange multiIndex = op.getMultiIndex();

    Value result = ub::PoisonOp::create(rewriter, loc, vecTy);

    for (int64_t i = 0; i < numElems; ++i) {
      Value idx = arith::ConstantIndexOp::create(rewriter, loc, i);

      // Extract scalar elements from each multi_index vector.
      SmallVector<OpFoldResult> scalarIndices;
      for (Value vec : multiIndex)
        scalarIndices.push_back(
            vector::ExtractOp::create(rewriter, loc, vec, idx).getResult());

      // Apply scalar linearization.
      OpFoldResult linearIndex =
          linearizeIndex(rewriter, loc, scalarIndices, op.getMixedBasis());
      Value scalarResult =
          getValueOrCreateConstantIntOp(rewriter, loc, linearIndex);

      // Insert result back into vector.
      result =
          vector::InsertOp::create(rewriter, loc, scalarResult, result, idx);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

class ExpandAffineIndexOpsAsAffinePass
    : public affine::impl::AffineExpandIndexOpsAsAffineBase<
          ExpandAffineIndexOpsAsAffinePass> {
public:
  ExpandAffineIndexOpsAsAffinePass() = default;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    populateAffineExpandIndexOpsAsAffinePatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

void mlir::affine::populateAffineExpandIndexOpsAsAffinePatterns(
    RewritePatternSet &patterns) {
  patterns.insert<LowerDelinearizeIndexOps, LowerLinearizeIndexOps>(
      patterns.getContext());
}

std::unique_ptr<Pass> mlir::affine::createAffineExpandIndexOpsAsAffinePass() {
  return std::make_unique<ExpandAffineIndexOpsAsAffinePass>();
}
