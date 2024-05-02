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
                                    ArrayRef<int64_t> dims) {
  if (dims.size() != tiles.size() || tiles.empty())
    return false;

  OpBuilder builder(tileOp);
  OpBuilder::InsertionGuard guard(builder);
  SmallVector<Range> iterationDomain =
      cast<TilingInterface>(tileOp.getOperation()).getIterationDomain(builder);

  for (auto dim : llvm::enumerate(dims)) {
    if (dim.value() >= static_cast<int64_t>(iterationDomain.size()))
      return false;

    std::optional<int64_t> tileSize = getConstantIntValue(tiles[dim.index()]);
    std::optional<int64_t> rangeOnDim =
        getConstantRange(iterationDomain[dim.value()]);

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

FailureOr<PackResult>
linalg::packMatmulOp(RewriterBase &rewriter, linalg::LinalgOp matmulOp,
                     const ControlPackMatmulFn &controlPackMatmul) {
  if (!(isa<linalg::MatmulOp>(matmulOp) ||
        isa<linalg::BatchMatmulOp>(matmulOp) ||
        isa<linalg::MatmulTransposeAOp>(matmulOp) ||
        isa<linalg::MatmulTransposeBOp>(matmulOp) ||
        isa<linalg::BatchMatmulTransposeAOp>(matmulOp) ||
        isa<linalg::BatchMatmulTransposeBOp>(matmulOp))) {
    return rewriter.notifyMatchFailure(matmulOp, "not a matmul-like operation");
  }

  if (matmulOp.hasDynamicShape())
    return rewriter.notifyMatchFailure(matmulOp, "require static shape");

  if (matmulOp.hasPureBufferSemantics())
    return rewriter.notifyMatchFailure(matmulOp, "require tensor semantics");

  std::optional<PackMatmulOptions> options = controlPackMatmul(matmulOp);
  if (!options)
    return rewriter.notifyMatchFailure(matmulOp, "invalid packing options");

  if (options->blockFactors.size() != 3)
    return rewriter.notifyMatchFailure(matmulOp, "require 3 tile factors");

  SmallVector<OpFoldResult> mnkTiles =
      getAsOpFoldResult(rewriter.getI64ArrayAttr(options->blockFactors));

  SmallVector<int64_t, 3> dims{options->mnkOrder};
  // Skip the batch dimension if present.
  bool isBatchMatmulOp = isa<linalg::BatchMatmulOp>(matmulOp) ||
                         isa<linalg::BatchMatmulTransposeAOp>(matmulOp) ||
                         isa<linalg::BatchMatmulTransposeBOp>(matmulOp);
  if (isBatchMatmulOp) {
    // Offset all dimensions.
    for (size_t i = 0; i < dims.size(); i++)
      ++dims[i];
  }

  if (!options->allowPadding &&
      !validateFullTilesOnDims(cast<TilingInterface>(matmulOp.getOperation()),
                               mnkTiles, dims)) {
    return rewriter.notifyMatchFailure(matmulOp,
                                       "expect packing full tiles only");
  }

  bool isTransposedRhs = isa<linalg::MatmulTransposeBOp>(matmulOp) ||
                         isa<linalg::BatchMatmulTransposeBOp>(matmulOp);

  OpBuilder::InsertionGuard guard(rewriter);
  // The op is replaced, we need to set the insertion point after it.
  rewriter.setInsertionPointAfter(matmulOp);

  // Pack the matmul operation into blocked layout with two levels of
  // subdivision:
  //   - major 2D blocks - outer dimensions, consist of minor blocks
  //   - minor 2D blocks - inner dimensions, consist of scalar elements
  FailureOr<PackResult> packedCanonicalMatmul = packMatmulGreedily(
      rewriter, matmulOp, mnkTiles, options->mnkPaddedSizesNextMultipleOf,
      options->mnkOrder);
  if (failed(packedCanonicalMatmul))
    return failure();

  assert(packedCanonicalMatmul->packOps.size() == 3 && "failed matmul packing");
  assert(packedCanonicalMatmul->unPackOps.size() == 1 &&
         "failed matmul unpacking");

  SmallVector<int64_t> innerPerm{1, 0};
  SmallVector<int64_t> outerPerm{1, 0};
  // No need to block transpose if the RHS matrix is already transposed.
  if (isTransposedRhs)
    outerPerm = {0, 1};

  // Leave the batch dimension as is.
  if (isBatchMatmulOp) {
    // Account for the batch dimension.
    SmallVector<int64_t> newOuterPerms{0};
    // Offset all permutations.
    for (auto perm : outerPerm)
      newOuterPerms.push_back(++perm);
    outerPerm = newOuterPerms;
  }

  // Block transpose the packed matmul i.e., transpose the outer dimensions
  // layout of the RHS matrix. The inner dimensions (minor blocks) remain
  // unchanged.
  FailureOr<PackTransposeResult> packedMatmul =
      packTranspose(rewriter, packedCanonicalMatmul->packOps[1],
                    packedCanonicalMatmul->packedLinalgOp,
                    /*maybeUnPackOp=*/nullptr, outerPerm, innerPerm);
  if (failed(packedMatmul))
    return failure();

  packedCanonicalMatmul->packedLinalgOp = packedMatmul->transposedLinalgOp;

  return packedCanonicalMatmul;
}

namespace {
template <typename OpTy>
struct PackMatmul : public OpRewritePattern<OpTy> {
  PackMatmul(MLIRContext *context, ControlPackMatmulFn fun,
             PatternBenefit benefit = 1)
      : OpRewritePattern<OpTy>(context, benefit), controlFn(std::move(fun)) {}

  LogicalResult matchAndRewrite(OpTy matmulOp,
                                PatternRewriter &rewriter) const override {
    FailureOr<PackResult> packedMatmul =
        packMatmulOp(rewriter, matmulOp, controlFn);
    if (failed(packedMatmul))
      return failure();
    return success();
  }

private:
  ControlPackMatmulFn controlFn;
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

    ControlPackMatmulFn controlFn =
        [&](linalg::LinalgOp op) -> PackMatmulOptions {
      PackMatmulOptions options;
      options.blockFactors = SmallVector<int64_t>{*blockFactors};
      if (!mnkOrder.empty())
        options.mnkOrder = SmallVector<int64_t>{*mnkOrder};
      options.mnkPaddedSizesNextMultipleOf =
          SmallVector<int64_t>{*mnkPaddedSizesNextMultipleOf};
      options.allowPadding = allowPadding;
      return options;
    };

    linalg::populatePackMatmulPatterns(patterns, controlFn);
    if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

void linalg::populatePackMatmulPatterns(RewritePatternSet &patterns,
                                        const ControlPackMatmulFn &controlFn) {
  patterns.add<PackMatmul<linalg::MatmulOp>, PackMatmul<linalg::BatchMatmulOp>,
               PackMatmul<linalg::MatmulTransposeAOp>,
               PackMatmul<linalg::BatchMatmulTransposeAOp>,
               PackMatmul<linalg::MatmulTransposeBOp>,
               PackMatmul<linalg::BatchMatmulTransposeBOp>>(
      patterns.getContext(), controlFn);
}
