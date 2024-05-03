//===- BlockPackMatmul.cpp - Linalg matmul block packing ------------------===//
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
#define GEN_PASS_DEF_LINALGBLOCKPACKMATMUL
#include "mlir/Dialect/Linalg/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::linalg;

/// Return constant range span or nullopt, otherwise.
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

/// Return true if all dimensions are fully divisible by the respective tiles.
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

/// Pack a matmul operation into blocked 4D layout.
FailureOr<PackResult>
linalg::blockPackMatmulOp(RewriterBase &rewriter, linalg::LinalgOp matmulOp,
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

  bool isTransposedLhs = isa<linalg::MatmulTransposeAOp>(matmulOp) ||
                         isa<linalg::BatchMatmulTransposeAOp>(matmulOp);
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

  auto applyBatchDim = [&](ArrayRef<int64_t> perms) -> SmallVector<int64_t> {
    // Account for the batch dimension.
    SmallVector<int64_t> newPerms{0};
    // Offset all permutations.
    for (auto perm : perms)
      newPerms.push_back(++perm);
    return newPerms;
  };

  // If needed, block transpose the packed matmul i.e., transpose the outer
  // dimensions. The inner dimensions (minor blocks) remain unchanged.
  if (isTransposedLhs) {
    // The inner blocks' layout is already correctly enforced by the initial
    // packing.
    SmallVector<int64_t> lhsInnerPerm{0, 1};
    // Only block transpose the outer dimensions for LHS matrix.
    SmallVector<int64_t> lhsOuterPerm{1, 0};
    // Leave the batch dimension as is.
    if (isBatchMatmulOp)
      lhsOuterPerm = applyBatchDim(lhsOuterPerm);

    FailureOr<PackTransposeResult> packedMatmul =
        packTranspose(rewriter, packedCanonicalMatmul->packOps[0],
                      packedCanonicalMatmul->packedLinalgOp,
                      /*maybeUnPackOp=*/nullptr, lhsOuterPerm, lhsInnerPerm);
    if (failed(packedMatmul))
      return failure();

    packedCanonicalMatmul->packOps[0] = packedMatmul->transposedPackOp;
    packedCanonicalMatmul->packedLinalgOp = packedMatmul->transposedLinalgOp;
  }

  // Transpose the layout of the inner dimension (minor blocks).
  SmallVector<int64_t> rhsInnerPerm{1, 0};
  // Block transpose the RHS matrix i.e., transpose the outer dimensions.
  SmallVector<int64_t> rhsOuterPerm{1, 0};
  // No need to block transpose if the RHS matrix is already transposed.
  if (isTransposedRhs)
    rhsOuterPerm = {0, 1};
  // Leave the batch dimension as is.
  if (isBatchMatmulOp)
    rhsOuterPerm = applyBatchDim(rhsOuterPerm);

  FailureOr<PackTransposeResult> packedMatmul =
      packTranspose(rewriter, packedCanonicalMatmul->packOps[1],
                    packedCanonicalMatmul->packedLinalgOp,
                    /*maybeUnPackOp=*/nullptr, rhsOuterPerm, rhsInnerPerm);
  if (failed(packedMatmul))
    return failure();

  packedCanonicalMatmul->packOps[1] = packedMatmul->transposedPackOp;
  packedCanonicalMatmul->packedLinalgOp = packedMatmul->transposedLinalgOp;

  return packedCanonicalMatmul;
}

namespace {
template <typename OpTy>
struct BlockPackMatmul : public OpRewritePattern<OpTy> {
  BlockPackMatmul(MLIRContext *context, ControlPackMatmulFn fun,
                  PatternBenefit benefit = 1)
      : OpRewritePattern<OpTy>(context, benefit), controlFn(std::move(fun)) {}

  LogicalResult matchAndRewrite(OpTy matmulOp,
                                PatternRewriter &rewriter) const override {
    FailureOr<PackResult> packedMatmul =
        blockPackMatmulOp(rewriter, matmulOp, controlFn);
    if (failed(packedMatmul))
      return failure();
    return success();
  }

private:
  ControlPackMatmulFn controlFn;
};

/// Convert linalg matmul ops to block layout and back.
struct LinalgBlockPackMatmul
    : public impl::LinalgBlockPackMatmulBase<LinalgBlockPackMatmul> {
  using LinalgBlockPackMatmulBase::LinalgBlockPackMatmulBase;

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

    linalg::populateBlockPackMatmulPatterns(patterns, controlFn);
    if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

void linalg::populateBlockPackMatmulPatterns(
    RewritePatternSet &patterns, const ControlPackMatmulFn &controlFn) {
  patterns.add<BlockPackMatmul<linalg::MatmulOp>,
               BlockPackMatmul<linalg::BatchMatmulOp>,
               BlockPackMatmul<linalg::MatmulTransposeAOp>,
               BlockPackMatmul<linalg::BatchMatmulTransposeAOp>,
               BlockPackMatmul<linalg::MatmulTransposeBOp>,
               BlockPackMatmul<linalg::BatchMatmulTransposeBOp>>(
      patterns.getContext(), controlFn);
}
