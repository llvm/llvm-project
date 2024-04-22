//===- PackMatmul.cpp - Linalg matmul packing -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/Passes.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

#include <optional>

namespace mlir {
#define GEN_PASS_DEF_LINALGPACKMATMUL
#include "mlir/Dialect/Linalg/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::linalg;

static std::optional<int64_t> getConstantRange(const Range &range) {
  std::optional<int64_t> stride = getConstantIntValue(range.stride);
  if (!stride || *stride != 1)
    return std::nullopt;
  std::optional<int64_t> offset = getConstantIntValue(range.offset);
  if (!offset)
    return std::nullopt;
  std::optional<int64_t> size = getConstantIntValue(range.size);
  if (!size)
    return std::nullopt;
  return (*size - *offset);
}

static bool validateFullTilesOnDims(TilingInterface tileOp,
                                    ArrayRef<OpFoldResult> tiles,
                                    ArrayRef<size_t> dims) {
  if (dims.size() != tiles.size() || tiles.empty())
    return false;

  OpBuilder builder(tileOp);
  OpBuilder::InsertionGuard guard(builder);
  SmallVector<Range> iterationDomain =
      cast<TilingInterface>(tileOp.getOperation()).getIterationDomain(builder);

  for (auto dim : llvm::enumerate(dims)) {
    if (dim.value() >= iterationDomain.size())
      return false;

    auto tileSize = getConstantIntValue(tiles[dim.index()]);
    auto rangeOnDim = getConstantRange(iterationDomain[dim.value()]);

    // If the tile factor or the range are non-constant, the tile size is
    // considered to be invalid.
    if (!tileSize || !rangeOnDim)
      return false;

    // The dimension must be fully divisible by the tile.
    if (*rangeOnDim % *tileSize != 0)
      return false;
  }

  return true;
}

static FailureOr<linalg::LinalgOp>
packMatmulOp(RewriterBase &rewriter, linalg::LinalgOp matmulOp,
             ArrayRef<OpFoldResult> mnkTiles) {
  if (!(isa<linalg::MatmulOp>(matmulOp) ||
        isa<linalg::BatchMatmulOp>(matmulOp))) {
    return rewriter.notifyMatchFailure(matmulOp, "not a matmul-like operation");
  }

  if (mnkTiles.size() != 3)
    return rewriter.notifyMatchFailure(matmulOp, "require 3 tile factors");

  if (matmulOp.hasDynamicShape())
    return rewriter.notifyMatchFailure(matmulOp, "require static shape");

  if (matmulOp.hasPureBufferSemantics())
    return rewriter.notifyMatchFailure(matmulOp, "require tensor semantics");

  SmallVector<size_t, 3> dims{0, 1, 2};
  // Skip the batch dimension if present.
  bool isBatchMatmulOp = isa<linalg::BatchMatmulOp>(matmulOp);
  if (isBatchMatmulOp)
    dims = {1, 2, 3};

  if (!validateFullTilesOnDims(cast<TilingInterface>(matmulOp.getOperation()),
                               mnkTiles, dims)) {
    return rewriter.notifyMatchFailure(matmulOp,
                                       "expect packing full tiles only");
  }

  OpBuilder::InsertionGuard guard(rewriter);
  // The op is replaced, we need to set the insertion point after it.
  rewriter.setInsertionPointAfter(matmulOp);

  auto packedCanonicalMatmul = packMatmulGreedily(
      rewriter, matmulOp, mnkTiles, /*mnkPaddedSizesNextMultipleOf=*/{},
      /*mnkOrder=*/{0, 1, 2});
  if (failed(packedCanonicalMatmul))
    return failure();

  assert(packedCanonicalMatmul->packOps.size() == 3 && "failed matmul packing");
  assert(packedCanonicalMatmul->unPackOps.size() == 1 &&
         "failed matmul unpacking");

  SmallVector<int64_t> innerPerm = {1, 0};
  SmallVector<int64_t> outerPerm = {1, 0};
  // Leave the batch dimension as is.
  if (isBatchMatmulOp)
    outerPerm = {0, 2, 1};

  auto packedMatmul =
      packTranspose(rewriter, packedCanonicalMatmul->packOps[1],
                    packedCanonicalMatmul->packedLinalgOp,
                    /*maybeUnPackOp=*/nullptr, outerPerm, innerPerm);
  if (failed(packedMatmul))
    return failure();

  return packedMatmul->transposedLinalgOp;
}

namespace {
template <typename OpTy>
struct PackMatmul : public OpRewritePattern<OpTy> {
  PackMatmul(MLIRContext *context, ArrayRef<int64_t> blockFactors,
             PatternBenefit benefit = 1)
      : OpRewritePattern<OpTy>(context, benefit), blockFactors(blockFactors) {}

  LogicalResult matchAndRewrite(OpTy matmulOp,
                                PatternRewriter &rewriter) const override {
    if (blockFactors.empty())
      return failure();
    auto packedMatmul =
        packMatmulOp(rewriter, matmulOp,
                     getAsOpFoldResult(rewriter.getI64ArrayAttr(blockFactors)));
    if (failed(packedMatmul))
      return failure();
    return success();
  }

private:
  SmallVector<int64_t> blockFactors;
};

// Entry point for packing matmul operations.
// Pack MatmulOp as following:
// [MB][NB][mb][nb] += [MB][KB][mb][kb] * [NB][KB][kb][nb]
// Pack a BatchMatmulOp as following:
// [B][MB][NB][mb][nb] += [B][MB][KB][mb][kb] * [B][NB][KB][kb][nb]
struct LinalgPackMatmul : public impl::LinalgPackMatmulBase<LinalgPackMatmul> {
  using LinalgPackMatmulBase::LinalgPackMatmulBase;

  void runOnOperation() override {
    Operation *op = getOperation();
    RewritePatternSet patterns(&getContext());
    linalg::populatePackMatmulPatterns(patterns, blockFactors);
    if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

void linalg::populatePackMatmulPatterns(RewritePatternSet &patterns,
                                        ArrayRef<int64_t> blockFactors) {
  patterns.add<PackMatmul<linalg::MatmulOp>, PackMatmul<linalg::BatchMatmulOp>>(
      patterns.getContext(), blockFactors);
}
