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
static bool validateFullTilesOnDims(linalg::LinalgOp linalgOp,
                                    ArrayRef<OpFoldResult> tiles,
                                    ArrayRef<int64_t> dims) {
  if (dims.size() != tiles.size() || tiles.empty())
    return false;

  FailureOr<ContractionDimensions> contractDims =
      inferContractionDims(linalgOp);
  if (failed(contractDims))
    return false;
  unsigned batchDimsOffset = contractDims->batch.size();

  // Skip the batch dimension if present.
  // Offset all dimensions accordingly.
  SmallVector<int64_t, 3> offsetDims(dims);
  for (size_t i = 0; i < offsetDims.size(); i++)
    offsetDims[i] += batchDimsOffset;

  auto tileOp = cast<TilingInterface>(linalgOp.getOperation());
  OpBuilder builder(tileOp);
  OpBuilder::InsertionGuard guard(builder);
  SmallVector<Range> iterationDomain = tileOp.getIterationDomain(builder);

  for (auto dim : llvm::enumerate(offsetDims)) {
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

/// Return failure or packed matmul with one of its operands transposed.
static FailureOr<PackTransposeResult>
transposePackedMatmul(RewriterBase &rewriter, linalg::LinalgOp linalgOp,
                      tensor::PackOp packOp, AffineMap operandMap,
                      ArrayRef<unsigned> blocksStartDimPos,
                      bool transposeOuterBlocks, bool transposeInnerBlocks) {
  assert(operandMap.getNumDims() >= 4 &&
         "expected at least 4D prepacked matmul");
  assert(blocksStartDimPos.size() >= 2 &&
         "expected starting outer and inner block positions");

  // Bias toward innermost dimensions.
  unsigned outerBlockPos = operandMap.getNumResults() - 4;
  unsigned innerBlockPos = operandMap.getNumResults() - 2;

  // Transpose control options define the desired block and element layout.
  // Block transposition (outer dimensions) or element transposition (inner
  // dimensions) may not be necessary depending on the original matmul data
  // layout.
  bool isOuterTransposed =
      operandMap.getDimPosition(outerBlockPos) != blocksStartDimPos.end()[-2];
  bool isInnerTransposed =
      operandMap.getDimPosition(innerBlockPos) != blocksStartDimPos.back();

  // Transpose only the dimensions that need that to conform to the provided
  // transpotion settings.
  SmallVector<int64_t> innerPerm = {0, 1};
  if (isInnerTransposed != transposeInnerBlocks)
    innerPerm = {1, 0};
  SmallVector<int64_t> outerPerm = {0, 1};
  if (isOuterTransposed != transposeOuterBlocks)
    outerPerm = {1, 0};

  // Leave the outer dimensions, like batch, unchanged by offsetting all
  // outer dimensions permutations.
  SmallVector<int64_t> offsetPerms;
  for (auto i : llvm::seq(0u, outerBlockPos))
    offsetPerms.push_back(i);
  for (auto perm : outerPerm)
    offsetPerms.push_back(perm + outerBlockPos);
  outerPerm = offsetPerms;

  FailureOr<PackTransposeResult> packTransposedMatmul =
      packTranspose(rewriter, packOp, linalgOp,
                    /*maybeUnPackOp=*/nullptr, outerPerm, innerPerm);

  return packTransposedMatmul;
}

/// Pack a matmul operation into blocked 4D layout.
FailureOr<PackResult>
linalg::blockPackMatmul(RewriterBase &rewriter, linalg::LinalgOp linalgOp,
                        const ControlBlockPackMatmulFn &controlPackMatmul) {
  if (linalgOp.hasPureBufferSemantics())
    return rewriter.notifyMatchFailure(linalgOp, "require tensor semantics");

  std::optional<BlockPackMatmulOptions> options = controlPackMatmul(linalgOp);
  if (!options)
    return rewriter.notifyMatchFailure(linalgOp, "invalid packing options");

  if (options->blockFactors.size() != 3)
    return rewriter.notifyMatchFailure(linalgOp, "require 3 tile factors");

  SmallVector<OpFoldResult> mnkTiles =
      getAsOpFoldResult(rewriter.getI64ArrayAttr(options->blockFactors));

  // If padding is disabled, make sure that dimensions can be packed cleanly.
  if (!options->allowPadding &&
      !validateFullTilesOnDims(linalgOp, mnkTiles, options->mnkOrder)) {
    return rewriter.notifyMatchFailure(linalgOp,
                                       "expect packing full tiles only");
  }

  OpBuilder::InsertionGuard guard(rewriter);
  // The op is replaced, we need to set the insertion point after it.
  rewriter.setInsertionPointAfter(linalgOp);

  // Pack the matmul operation into blocked layout with two levels of
  // subdivision:
  //   - major 2D blocks - outer dimensions, consist of minor blocks
  //   - minor 2D blocks - inner dimensions, consist of scalar elements
  FailureOr<PackResult> packedMatmul = packMatmulGreedily(
      rewriter, linalgOp, mnkTiles, options->mnkPaddedSizesNextMultipleOf,
      options->mnkOrder);
  if (failed(packedMatmul))
    return failure();

  assert(packedMatmul->packOps.size() == 3 &&
         "invalid number of pack ops after matmul packing");
  assert(packedMatmul->unPackOps.size() == 1 &&
         "invalid number of unpack ops after matmul packing");

  FailureOr<ContractionDimensions> contractDims =
      inferContractionDims(packedMatmul->packedLinalgOp);
  if (failed(contractDims))
    return failure();

  auto genericOp =
      dyn_cast<linalg::GenericOp>(packedMatmul->packedLinalgOp.getOperation());
  SmallVector<AffineMap> maps = genericOp.getIndexingMapsArray();

  // Transpose LHS matrix according to the options.
  FailureOr<PackTransposeResult> packedLhs = transposePackedMatmul(
      rewriter, packedMatmul->packedLinalgOp, packedMatmul->packOps[0], maps[0],
      contractDims->m, options->lhsTransposeOuterBlocks,
      options->lhsTransposeInnerBlocks);
  if (failed(packedLhs))
    return failure();

  // Update results.
  packedMatmul->packOps[0] = packedLhs->transposedPackOp;
  packedMatmul->packedLinalgOp = packedLhs->transposedLinalgOp;

  // Transpose RHS matrix according to the options.
  FailureOr<PackTransposeResult> packedRhs = transposePackedMatmul(
      rewriter, packedMatmul->packedLinalgOp, packedMatmul->packOps[1], maps[1],
      contractDims->k, options->rhsTransposeOuterBlocks,
      options->rhsTransposeInnerBlocks);
  if (failed(packedRhs))
    return failure();

  // Update results.
  packedMatmul->packOps[1] = packedRhs->transposedPackOp;
  packedMatmul->packedLinalgOp = packedRhs->transposedLinalgOp;

  return packedMatmul;
}

namespace {
template <typename OpTy>
struct BlockPackMatmul : public OpRewritePattern<OpTy> {
  BlockPackMatmul(MLIRContext *context, ControlBlockPackMatmulFn fun,
                  PatternBenefit benefit = 1)
      : OpRewritePattern<OpTy>(context, benefit), controlFn(std::move(fun)) {}

  LogicalResult matchAndRewrite(OpTy linalgOp,
                                PatternRewriter &rewriter) const override {
    FailureOr<PackResult> packedMatmul =
        blockPackMatmul(rewriter, linalgOp, controlFn);
    if (failed(packedMatmul))
      return failure();
    return success();
  }

private:
  ControlBlockPackMatmulFn controlFn;
};

template <>
struct BlockPackMatmul<linalg::GenericOp>
    : public OpRewritePattern<linalg::GenericOp> {
  BlockPackMatmul(MLIRContext *context, ControlBlockPackMatmulFn fun,
                  PatternBenefit benefit = 1)
      : OpRewritePattern<linalg::GenericOp>(context, benefit),
        controlFn(std::move(fun)) {}

  LogicalResult matchAndRewrite(linalg::GenericOp linalgOp,
                                PatternRewriter &rewriter) const override {
    // Match suitable generics.
    if (!linalg::isaContractionOpInterface(linalgOp)) {
      return rewriter.notifyMatchFailure(linalgOp, "not a contraction");
    }

    using MapList = ArrayRef<ArrayRef<AffineExpr>>;
    auto infer = [&](MapList m) {
      return AffineMap::inferFromExprList(m, linalgOp.getContext());
    };

    AffineExpr i, j, k;
    bindDims(linalgOp->getContext(), i, j, k);
    SmallVector<AffineMap> maps = linalgOp.getIndexingMapsArray();

    // For now, only match simple matmuls.
    if (!(maps == infer({{i, k}, {k, j}, {i, j}}) ||
          maps == infer({{k, i}, {k, j}, {i, j}}) ||
          maps == infer({{i, k}, {j, k}, {i, j}}))) {
      return rewriter.notifyMatchFailure(linalgOp, "not a suitable matmul");
    }

    FailureOr<PackResult> packedMatmul =
        blockPackMatmul(rewriter, linalgOp, controlFn);
    if (failed(packedMatmul))
      return failure();
    return success();
  }

private:
  ControlBlockPackMatmulFn controlFn;
};

/// Convert linalg matmul ops to block layout and back.
struct LinalgBlockPackMatmul
    : public impl::LinalgBlockPackMatmulBase<LinalgBlockPackMatmul> {
  using LinalgBlockPackMatmulBase::LinalgBlockPackMatmulBase;

  void runOnOperation() override {
    Operation *op = getOperation();
    RewritePatternSet patterns(&getContext());

    ControlBlockPackMatmulFn controlFn =
        [&](linalg::LinalgOp op) -> BlockPackMatmulOptions {
      BlockPackMatmulOptions options;
      options.blockFactors = SmallVector<int64_t>{*blockFactors};
      options.allowPadding = allowPadding;
      options.mnkPaddedSizesNextMultipleOf =
          SmallVector<int64_t>{*mnkPaddedSizesNextMultipleOf};
      if (!mnkOrder.empty())
        options.mnkOrder = SmallVector<int64_t>{*mnkOrder};
      options.lhsTransposeOuterBlocks = lhsTransposeOuterBlocks;
      options.lhsTransposeInnerBlocks = lhsTransposeInnerBlocks;
      options.rhsTransposeOuterBlocks = rhsTransposeOuterBlocks;
      options.rhsTransposeInnerBlocks = rhsTransposeInnerBlocks;
      return options;
    };

    linalg::populateBlockPackMatmulPatterns(patterns, controlFn);
    if (failed(applyPatternsGreedily(op, std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

void linalg::populateBlockPackMatmulPatterns(
    RewritePatternSet &patterns, const ControlBlockPackMatmulFn &controlFn) {
  patterns.add<BlockPackMatmul<linalg::GenericOp>,
               BlockPackMatmul<linalg::MatmulOp>,
               BlockPackMatmul<linalg::BatchMatmulOp>,
               BlockPackMatmul<linalg::MatmulTransposeAOp>,
               BlockPackMatmul<linalg::BatchMatmulTransposeAOp>,
               BlockPackMatmul<linalg::MatmulTransposeBOp>,
               BlockPackMatmul<linalg::BatchMatmulTransposeBOp>>(
      patterns.getContext(), controlFn);
}
