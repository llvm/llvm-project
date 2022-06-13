//===- Tiling.cpp - Implementation of tiling using TilingInterface -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the tiling using TilingInterface.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/TileUsingInterface.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tile-using-interface"

using namespace mlir;

scf::SCFTilingOptions &
scf::SCFTilingOptions::setTileSizes(ArrayRef<int64_t> ts) {
  assert(!tileSizeComputationFunction && "tile sizes already set");
  SmallVector<int64_t, 4> tileSizes(ts.begin(), ts.end());
  tileSizeComputationFunction = [tileSizes](OpBuilder &b, Operation *op) {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(
        &op->getParentOfType<func::FuncOp>().getBody().front());
    return llvm::to_vector<4>(map_range(tileSizes, [&](int64_t s) {
      Value v = b.create<arith::ConstantIndexOp>(op->getLoc(), s);
      return v;
    }));
  };
  return *this;
}

/// Generate an empty loop nest that represents the tiled loop nest shell.
/// - `loopRanges` specifies the lb, ub and step of the untiled iteration space.
/// - `tileSizeVals` is the tile sizes to use. Zero represent untiled loops.
/// - In `offsets` and `sizes` return the multi-dimensional offset and size of
/// the
///   tile processed within the inner most loop.
static SmallVector<scf::ForOp>
generateTileLoopNest(OpBuilder &builder, Location loc,
                     ArrayRef<Range> loopRanges, ArrayRef<Value> tileSizeVals,
                     SmallVector<OpFoldResult> &offsets,
                     SmallVector<OpFoldResult> &sizes) {
  assert(!loopRanges.empty() && "expected at least one loop range");
  assert(loopRanges.size() == tileSizeVals.size() &&
         "expected as many tile sizes as loop ranges");
  OpBuilder::InsertionGuard guard(builder);
  SmallVector<scf::ForOp> loops;
  offsets.resize(loopRanges.size());
  sizes.resize(loopRanges.size());

  // The tile size to use (to avoid out of bounds access) is  minimum of
  // `tileSize` and `ub - iv`, where `iv` is the induction variable
  // of the tiled loop.
  AffineExpr s0, s1, d0;
  bindDims(builder.getContext(), d0);
  bindSymbols(builder.getContext(), s0, s1);
  AffineMap minMap = AffineMap::get(1, 2, {s0, s1 - d0}, builder.getContext());

  for (auto loopRange : llvm::enumerate(loopRanges)) {
    // No loops if tile size is zero. Set offset and size to the loop
    // offset and size.
    if (matchPattern(tileSizeVals[loopRange.index()], m_Zero())) {
      offsets[loopRange.index()] = loopRange.value().offset;
      sizes[loopRange.index()] = loopRange.value().size;
      continue;
    }

    auto loop = builder.create<scf::ForOp>(
        loc, loopRange.value().offset, loopRange.value().size,
        tileSizeVals[loopRange.index()], ValueRange{},
        [&](OpBuilder &bodyBuilder, Location bodyLoc, Value iv,
            ValueRange /*iterArgs*/) {
          Value boundedTileSize = builder.create<AffineMinOp>(
              bodyLoc, minMap,
              ValueRange{iv, tileSizeVals[loopRange.index()],
                         loopRange.value().size});
          sizes[loopRange.index()] = boundedTileSize;
          builder.create<scf::YieldOp>(loc);
        });
    offsets[loopRange.index()] = loop.getInductionVar();
    loops.push_back(loop);
    builder.setInsertionPoint(loop.getBody()->getTerminator());
  }
  return loops;
}

scf::TileUsingSCFForOp::TileUsingSCFForOp(MLIRContext *context,
                                          scf::SCFTilingOptions options,
                                          PatternBenefit benefit)
    : OpInterfaceRewritePattern<TilingInterface>(context, benefit),
      options(std::move(options)) {}

scf::TileUsingSCFForOp::TileUsingSCFForOp(StringRef opName,
                                          MLIRContext *context,
                                          scf::SCFTilingOptions options,
                                          PatternBenefit benefit)
    : OpInterfaceRewritePattern<TilingInterface>(context, benefit),
      options(std::move(options)) {}

FailureOr<scf::SCFTilingResult>
scf::TileUsingSCFForOp::returningMatchAndRewrite(
    TilingInterface op, PatternRewriter &rewriter) const {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfter(op);

  if (!options.tileSizeComputationFunction) {
    return rewriter.notifyMatchFailure(
        op, "missing tile size computation function");
  }

  // 1. Get the range of the loops that are represented by the operation.
  SmallVector<Range> iterationDomain = op.getIterationDomain(rewriter);
  size_t numLoops = iterationDomain.size();
  if (numLoops == 0) {
    return rewriter.notifyMatchFailure(
        op, "unable to tile op with no iteration domain");
  }

  // 2. Materialize the tile sizes. Enforce the convention that "tiling by zero"
  // skips tiling a particular dimension. This convention is significantly
  // simpler to handle instead of adjusting affine maps to account for missing
  // dimensions.
  SmallVector<Value, 4> tileSizeVector =
      options.tileSizeComputationFunction(rewriter, op);
  if (tileSizeVector.size() < iterationDomain.size()) {
    auto zero = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
    tileSizeVector.append(numLoops - tileSizeVector.size(), zero);
  }

  scf::SCFTilingResult tilingResult;
  SmallVector<OpFoldResult> offsets, sizes;
  {
    // 3. Materialize an empty loop nest that iterates over the tiles. These
    // loops for now do not return any values even if the original operation has
    // results.
    tilingResult.loops = generateTileLoopNest(
        rewriter, op.getLoc(), iterationDomain, tileSizeVector, offsets, sizes);

    LLVM_DEBUG({
      if (!tilingResult.loops.empty()) {
        llvm::errs() << "LoopNest shell :\n";
        tilingResult.loops.front().dump();
        llvm::errs() << "\n";
      }
    });

    // 4. Generate the tiled implementation within the inner most loop.
    if (!tilingResult.loops.empty())
      rewriter.setInsertionPoint(
          tilingResult.loops.back().getBody()->getTerminator());
    SmallVector<Operation *> tiledImplementation = op.getTiledImplementation(
        rewriter, op.getDestinationOperands(rewriter), offsets, sizes, true);
    if (tiledImplementation.size() != 1) {
      return rewriter.notifyMatchFailure(
          op, "expected tiled implementation to return a single op");
    }
    tilingResult.tiledOp = tiledImplementation[0];

    LLVM_DEBUG({
      if (!tilingResult.loops.empty()) {
        llvm::errs() << "After tiled implementation :\n";
        tilingResult.loops.front().dump();
        llvm::errs() << "\n";
      }
    });
  }

  if (op->getNumResults() == 0) {
    rewriter.eraseOp(op);
    return tilingResult;
  }

  // 5. If the original operations has results, modify the loop nest to yield
  // the replacement values.
  SmallVector<Value> replacements;
  if (tilingResult.loops.empty()) {
    // 5a. If there were no loops, the tiled implementation results are the
    // replacements.
    rewriter.replaceOp(op, tilingResult.tiledOp->getResults());
    return tilingResult;
  }

  // 5b. `scf.for` with tensor semantics requires the loop nest to yield the
  // replacement values using destructive updates. Use the `TilingInterface`
  // to get the position of the result tiles and use that to generate the
  // destructive update pattern, i.e.,
  //
  // ```mlir
  // scf.for %iv0 = ... {
  //   %0 = tiled_op
  // }
  // ```
  //
  // is transformed to
  //
  // ```mlir
  // %result = scf.for %iv0 = ... iter_args(%arg = %init) -> .. {
  //   %0 = tiled_op
  //   %1 = tensor.insert_slice %0 into %arg[..] [..] [..]
  //   scf.yield %1
  // }
  // ```
  NewYieldValueFn yieldValueFn =
      [&](OpBuilder &b, Location loc,
          ArrayRef<BlockArgument> newBBArgs) -> SmallVector<Value> {
    SmallVector<Value> yieldedValues;
    Attribute one = b.getIndexAttr(1);
    for (auto resultNum : llvm::seq<unsigned>(0, op->getNumResults())) {
      SmallVector<OpFoldResult> resultTileOffsets, resultTileSizes;
      if (failed(op.getResultTilePosition(b, resultNum, offsets, sizes,
                                          resultTileOffsets,
                                          resultTileSizes))) {
        op.emitOpError("unable to get position of result ")
            << resultNum << " of the tiled implementation";
        return {};
      }
      SmallVector<OpFoldResult> resultTileStrides(resultTileOffsets.size(),
                                                  one);
      Value yieldedValue = b.create<tensor::InsertSliceOp>(
          op->getLoc(), tilingResult.tiledOp->getResult(resultNum),
          newBBArgs[resultNum], resultTileOffsets, resultTileSizes,
          resultTileStrides);
      yieldedValues.push_back(yieldedValue);
    }
    return yieldedValues;
  };
  SmallVector<scf::ForOp> newLoops = replaceLoopNestWithNewYields(
      rewriter, tilingResult.loops, op.getDestinationOperands(rewriter),
      yieldValueFn);
  for (auto loop : llvm::enumerate(tilingResult.loops)) {
    rewriter.eraseOp(loop.value());
    tilingResult.loops[loop.index()] = newLoops[loop.index()];
  }
  rewriter.replaceOp(op, tilingResult.loops.front().getResults());
  return tilingResult;
}
