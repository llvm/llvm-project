//===- LowerVectorMask.cpp - Lower 'vector.mask' operation ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements target-independent rewrites and utilitites to lower the
// 'vector.mask' operation.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "lower-vector-mask"

namespace mlir {
namespace vector {
#define GEN_PASS_DEF_LOWERVECTORMASKPASS
#include "mlir/Dialect/Vector/Transforms/Passes.h.inc"
} // namespace vector
} // namespace mlir

using namespace mlir;
using namespace mlir::vector;

namespace {

/// The `MaskOpRewritePattern` implements a pattern that follows a two-fold
/// matching:
///   1. It matches a `vector.mask` operation.
///   2. It invokes `matchAndRewriteMaskableOp` on `MaskableOpInterface` nested
///      in the matched `vector.mask` operation.
///
/// It is required that the replacement op in the pattern replaces the
/// `vector.mask` operation and not the nested `MaskableOpInterface`. This
/// approach allows having patterns that "stop" at every `vector.mask` operation
/// and actually match the traits of its the nested `MaskableOpInterface`.
template <class SourceOp>
struct MaskOpRewritePattern : OpRewritePattern<MaskOp> {
  using OpRewritePattern<MaskOp>::OpRewritePattern;

private:
  LogicalResult matchAndRewrite(MaskOp maskOp,
                                PatternRewriter &rewriter) const final {
    MaskableOpInterface maskableOp = maskOp.getMaskableOp();
    SourceOp sourceOp = dyn_cast<SourceOp>(maskableOp.getOperation());
    if (!sourceOp)
      return failure();

    return matchAndRewriteMaskableOp(sourceOp, maskOp, rewriter);
  }

protected:
  virtual LogicalResult
  matchAndRewriteMaskableOp(SourceOp sourceOp, MaskingOpInterface maskingOp,
                            PatternRewriter &rewriter) const = 0;
};

/// Lowers a masked `vector.transfer_read` operation.
struct MaskedTransferReadOpPattern
    : public MaskOpRewritePattern<TransferReadOp> {
public:
  using MaskOpRewritePattern<TransferReadOp>::MaskOpRewritePattern;

  LogicalResult
  matchAndRewriteMaskableOp(TransferReadOp readOp, MaskingOpInterface maskingOp,
                            PatternRewriter &rewriter) const override {
    // TODO: The 'vector.mask' passthru is a vector and 'vector.transfer_read'
    // expects a scalar. We could only lower one to the other for cases where
    // the passthru is a broadcast of a scalar.
    if (maskingOp.hasPassthru())
      return rewriter.notifyMatchFailure(
          maskingOp, "Can't lower passthru to vector.transfer_read");

    // Replace the `vector.mask` operation.
    rewriter.replaceOpWithNewOp<TransferReadOp>(
        maskingOp.getOperation(), readOp.getVectorType(), readOp.getSource(),
        readOp.getIndices(), readOp.getPermutationMap(), readOp.getPadding(),
        maskingOp.getMask(), readOp.getInBounds().value_or(ArrayAttr()));
    return success();
  }
};

/// Lowers a masked `vector.transfer_write` operation.
struct MaskedTransferWriteOpPattern
    : public MaskOpRewritePattern<TransferWriteOp> {
public:
  using MaskOpRewritePattern<TransferWriteOp>::MaskOpRewritePattern;

  LogicalResult
  matchAndRewriteMaskableOp(TransferWriteOp writeOp,
                            MaskingOpInterface maskingOp,
                            PatternRewriter &rewriter) const override {
    Type resultType =
        writeOp.getResult() ? writeOp.getResult().getType() : Type();

    // Replace the `vector.mask` operation.
    rewriter.replaceOpWithNewOp<TransferWriteOp>(
        maskingOp.getOperation(), resultType, writeOp.getVector(),
        writeOp.getSource(), writeOp.getIndices(), writeOp.getPermutationMap(),
        maskingOp.getMask(), writeOp.getInBounds().value_or(ArrayAttr()));
    return success();
  }
};

/// Populates instances of `MaskOpRewritePattern` to lower masked operations
/// with `vector.mask`. Patterns should rewrite the `vector.mask` operation and
/// not its nested `MaskableOpInterface`.
void populateVectorMaskLoweringPatternsForSideEffectingOps(
    RewritePatternSet &patterns) {
  patterns.add<MaskedTransferReadOpPattern, MaskedTransferWriteOpPattern>(
      patterns.getContext());
}

struct LowerVectorMaskPass
    : public vector::impl::LowerVectorMaskPassBase<LowerVectorMaskPass> {
  using Base::Base;

  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *context = op->getContext();

    RewritePatternSet loweringPatterns(context);
    populateVectorMaskLoweringPatternsForSideEffectingOps(loweringPatterns);

    if (failed(applyPatternsAndFoldGreedily(op->getRegions(),
                                            std::move(loweringPatterns))))
      signalPassFailure();
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect>();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::vector::createLowerVectorMaskPass() {
  return std::make_unique<LowerVectorMaskPass>();
}
