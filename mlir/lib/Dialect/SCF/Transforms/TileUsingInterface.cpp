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

#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include <optional>

#define DEBUG_TYPE "tile-using-interface"

using namespace mlir;

scf::SCFTilingOptions &
scf::SCFTilingOptions::setTileSizes(ArrayRef<OpFoldResult> ts) {
  assert(!tileSizeComputationFunction && "tile sizes already set");
  auto tileSizes = llvm::to_vector(ts);
  tileSizeComputationFunction = [tileSizes](OpBuilder &b, Operation *op) {
    return tileSizes;
  };
  return *this;
}

scf::SCFTilingOptions &
scf::SCFTilingOptions::setMaxNumTiles(ArrayRef<OpFoldResult> mnt) {
  assert(!maxNumTilesComputationFunction && "max num tiles already set");
  auto maxNumTiles = llvm::to_vector(mnt);
  maxNumTilesComputationFunction = [maxNumTiles](OpBuilder &b, Operation *op) {
    return maxNumTiles;
  };
  return *this;
}

/// Helper method to adjust the interchange vector to match the iteration
/// domain.
static SmallVector<int64_t>
fillInterchangeVector(ArrayRef<int64_t> interchangeVector,
                      size_t iterationDomainSize) {
  SmallVector<int64_t> filledVector = llvm::to_vector(interchangeVector);
  if (filledVector.size() < iterationDomainSize) {
    auto range = llvm::seq<int64_t>(filledVector.size(), iterationDomainSize);
    filledVector.append(range.begin(), range.end());
  }
  if (filledVector.size() > iterationDomainSize)
    filledVector.resize(iterationDomainSize);
  return filledVector;
}

//===----------------------------------------------------------------------===//
// tileUsingSCF implementation.
//===----------------------------------------------------------------------===//

/// Verify the tile size options are set in a consistent manner.
static LogicalResult
verifyTileSizeOptions(RewriterBase &rewriter, Location loc,
                      const scf::SCFTilingOptions &options) {
  if (!options.tileSizeComputationFunction &&
      !options.maxNumTilesComputationFunction) {
    return rewriter.notifyMatchFailure(
        loc, "at least one of tile size computation function or max num tiles "
             "computation must be specified.");
  }
  if (options.tileSizeComputationFunction &&
      options.maxNumTilesComputationFunction) {
    return rewriter.notifyMatchFailure(
        loc, "only one of tile size computation function or max num tiles "
             "computation function can be specified");
  }

  // If specified, check that the interchange vector is a permutation.
  if (!options.interchangeVector.empty()) {
    if (!isPermutationVector(options.interchangeVector)) {
      return rewriter.notifyMatchFailure(
          loc, "invalid intechange vector, not a permutation of the entire "
               "iteration space");
    }
  }
  return success();
}

/// Compute the tile sizes and num tiles values. The `numTiles`
/// is empty if the `maxNumTilesComputationFunction` is not specified.
static std::tuple<SmallVector<OpFoldResult>, SmallVector<OpFoldResult>>
getTileSizesAndNumTiles(RewriterBase &rewriter, TilingInterface op,
                        ArrayRef<Range> iterationDomain,
                        const scf::SCFTilingOptions &options) {
  SmallVector<OpFoldResult> tileSizes, numTiles;

  // Enforce the convention that "tiling by zero"
  // skips tiling a particular dimension. This convention is significantly
  // simpler to handle instead of adjusting affine maps to account for missing
  // dimensions.
  auto numLoops = iterationDomain.size();
  if (options.tileSizeComputationFunction) {
    tileSizes = options.tileSizeComputationFunction(rewriter, op);
    tileSizes.resize(numLoops, rewriter.getIndexAttr(0));
    return {tileSizes, numTiles};
  }

  assert(options.maxNumTilesComputationFunction &&
         "expected at least one of tile sizes cpomputation function or max num "
         "tiles computation function");
  // Enforce the convention that "maxNumTiles to zero"
  // skips tiling a particular dimension. This convention is significantly
  // simpler to handle instead of adjusting affine maps to account for missing
  // dimensions.
  SmallVector<OpFoldResult> maxNumTiles =
      options.maxNumTilesComputationFunction(rewriter, op);
  maxNumTiles.resize(numLoops, rewriter.getIndexAttr(0));

  // Use the maxNumTiles to compute the tile sizes as
  // - niters = ceilDiv(ub - lb, step)
  // - tileSize = ceilDiv(niters, maxNumTiles)
  AffineExpr s0, s1, s2, s3;
  bindSymbols(rewriter.getContext(), s0, s1, s2, s3);
  AffineExpr numIters = (s1 - s0).ceilDiv(s2);
  AffineExpr tileSizeExpr = numIters.ceilDiv(s3);
  tileSizes.resize(numLoops, rewriter.getIndexAttr(0));
  for (auto [index, maxNumTile] : llvm::enumerate(maxNumTiles)) {
    if (isConstantIntValue(maxNumTile, 0))
      continue;

    tileSizes[index] = affine::makeComposedFoldedAffineApply(
        rewriter, op.getLoc(), tileSizeExpr,
        {iterationDomain[index].offset, iterationDomain[index].size,
         iterationDomain[index].stride, maxNumTile});
  }

  return {tileSizes, maxNumTiles};
}

// Check if `stride` evenly divides the trip count `size - offset`.
static bool tileDividesIterationDomain(Range loopRange) {
  std::optional<int64_t> offsetAsInt = getConstantIntValue(loopRange.offset);
  if (!offsetAsInt)
    return false;
  std::optional<int64_t> sizeAsInt = getConstantIntValue(loopRange.size);
  if (!sizeAsInt)
    return false;
  std::optional<int64_t> strideAsInt = getConstantIntValue(loopRange.stride);
  if (!strideAsInt)
    return false;
  return ((sizeAsInt.value() - offsetAsInt.value()) % strideAsInt.value() == 0);
}

/// Returns the bounded tile size given the current `iv`, `loopRange` and
/// `tileSize`, i.e., `min(tileSize, range.end() - iv)`.
static OpFoldResult getBoundedTileSize(OpBuilder &b, Location loc,
                                       Range loopRange, Value iv,
                                       OpFoldResult tileSize) {
  std::optional<int64_t> ts = getConstantIntValue(tileSize);
  if (ts && ts.value() == 1)
    return tileSize;

  if (tileDividesIterationDomain(
          Range{loopRange.offset, loopRange.size, tileSize}))
    return tileSize;

  // The tile size to use (to avoid out of bounds access) is  minimum of
  // `tileSize` and `ub - iv`, where `iv` is the induction variable of the tiled
  // loop.
  AffineExpr s0, s1, d0;
  bindDims(b.getContext(), d0);
  bindSymbols(b.getContext(), s0, s1);
  AffineMap minMap = AffineMap::get(1, 2, {s0, s1 - d0}, b.getContext());
  Value size = getValueOrCreateConstantIndexOp(b, loc, loopRange.size);
  return affine::makeComposedFoldedAffineMin(
      b, loc, minMap, SmallVector<OpFoldResult>{iv, tileSize, size});
}

/// Compute the tile offsets and sizes.
static std::tuple<SmallVector<OpFoldResult>, SmallVector<OpFoldResult>>
getTileOffsetAndSizes(RewriterBase &rewriter, Location loc, ValueRange ivs,
                      ArrayRef<Range> iterationDomain,
                      ArrayRef<OpFoldResult> tileSizes, bool isLoopNormalized) {
  SmallVector<OpFoldResult> offsets, sizes;
  int materializedLoopNum = 0;

  AffineExpr d0, s0, s1, s2;
  AffineExpr offsetExpr;
  if (isLoopNormalized) {
    bindDims(rewriter.getContext(), d0);
    bindSymbols(rewriter.getContext(), s0, s1, s2);
    offsetExpr = s0 + d0 * s1 * s2;
  }

  for (auto [tileSize, loopRange] :
       llvm::zip_equal(tileSizes, iterationDomain)) {
    if (isConstantIntValue(tileSize, 0)) {
      offsets.push_back(loopRange.offset);
      sizes.push_back(loopRange.size);
      continue;
    }
    // If loop is normalized, the offset is (lb + iv * step * tileSize)
    Value iv = ivs[materializedLoopNum++];
    OpFoldResult offset;
    if (isLoopNormalized) {
      offset = affine::makeComposedFoldedAffineApply(
          rewriter, loc, offsetExpr,
          ArrayRef<OpFoldResult>{iv, loopRange.offset, loopRange.stride,
                                 tileSize});
    } else {
      offset = getAsOpFoldResult(iv);
    }
    offsets.push_back(offset);
    sizes.push_back(getBoundedTileSize(rewriter, loc, loopRange, iv, tileSize));
  }
  return {offsets, sizes};
}

/// A function that allows returning additional yielded values during
/// `yieldTiledValuesAndReplace`.
/// - `ivs` induction variable for the loop.
/// - `newBbArgs` basic block arguments corresponding to newly added iter_args.
/// - `tiledValues` the tiled values to return. Must be of same size as
///   `newbbArgs`, each element of this array is inserted into the corresponding
///   element in `newbbArgs`.
/// - `resultOffsets` is of the same size as `tiledValues` and represents
///   the offsets to use when inserting corresponding element from `tiledValues`
///   into the element from `newBbArgs`.
/// - `resultSizes` is of the same size as `tiledValues` and represents
///   the size of the corresponding element from `tiledValues` inserted into
///   the element from `newBbArgs`.
/// In case the method needs to return `failure()` the method is expected
/// to clean up any inserted operations.
using YieldTiledValuesFn = std::function<LogicalResult(
    RewriterBase &rewriter, Location loc, ValueRange ivs, ValueRange newBbArgs,
    SmallVector<Value> &tiledValues,
    SmallVector<SmallVector<OpFoldResult>> &resultOffsets,
    SmallVector<SmallVector<OpFoldResult>> &resultSizes)>;

/// Clones the operation and updates the destination if the operation
/// implements the `DestinationStyleOpInterface`.
static Operation *cloneOpAndUpdateDestinationArgs(RewriterBase &rewriter,
                                                  Operation *op,
                                                  ValueRange newDestArgs) {
  Operation *clonedOp = rewriter.clone(*op);
  if (newDestArgs.empty())
    return clonedOp;
  if (auto destinationStyleOp = dyn_cast<DestinationStyleOpInterface>(clonedOp))
    destinationStyleOp.getDpsInitsMutable().assign(newDestArgs);
  return clonedOp;
}

/// Generate the tile-loop nest using `scf.for` operation.
/// - `loopRanges` specifies the lb, ub and step of the untiled iteration space.
/// - `tileSizes` is the tile sizes to use. Zero represent untiled loops.
/// - `destinationTensors` are the init values to use for the outer most loop.
/// - `yieldTiledValuesFn` is called to generated the loop body of the inner
/// most
///    loop.
/// - `loops` is an in-out parameter into which the generated loops are
///    populated.
static LogicalResult generateLoopNestUsingForOp(
    RewriterBase &rewriter, Location loc, ArrayRef<Range> loopRanges,
    ArrayRef<OpFoldResult> tileSizes, ArrayRef<OpFoldResult> numTiles,
    ValueRange destinationTensors, YieldTiledValuesFn yieldTiledValuesFn,
    SmallVector<LoopLikeOpInterface> &loops) {
  assert(!loopRanges.empty() && "unexpected empty loop ranges");
  assert(loopRanges.size() == tileSizes.size() &&
         "expected as many tile sizes as loop ranges");
  OpBuilder::InsertionGuard guard(rewriter);
  SmallVector<Value> ivs;

  Value zero, one;
  if (!numTiles.empty()) {
    zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    ;
    one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  }

  for (auto [index, loopRange, tileSize] :
       llvm::enumerate(loopRanges, tileSizes)) {
    // No loops if tile size is zero. Set offset and size to the loop
    // offset and size.
    if (isConstantIntValue(tileSize, 0))
      continue;

    Value lb, ub, step;
    if (numTiles.empty()) {
      lb = getValueOrCreateConstantIndexOp(rewriter, loc, loopRange.offset);
      ub = getValueOrCreateConstantIndexOp(rewriter, loc, loopRange.size);
      step = getValueOrCreateConstantIndexOp(rewriter, loc, tileSize);
    } else {
      lb = zero;
      ub = getValueOrCreateConstantIndexOp(rewriter, loc, numTiles[index]);
      step = one;
    }
    auto loop =
        rewriter.create<scf::ForOp>(loc, lb, ub, step, destinationTensors,
                                    [](OpBuilder &bodyBuilder, Location bodyLoc,
                                       Value iv, ValueRange /*iterArgs*/) {});
    loops.push_back(loop);
    ivs.push_back(loop.getInductionVar());
    rewriter.setInsertionPointToEnd(loop.getBody());
    destinationTensors = loop.getRegionIterArgs();
  }

  SmallVector<Value> tiledResults;
  SmallVector<SmallVector<OpFoldResult>> resultOffsets, resultSizes;
  if (failed(yieldTiledValuesFn(rewriter, loc, ivs, destinationTensors,
                                tiledResults, resultOffsets, resultSizes))) {
    return rewriter.notifyMatchFailure(
        loc, "failed to generate inner tile loop body");
  }
  if (loops.empty())
    return success();

  assert(tiledResults.size() == destinationTensors.size() &&
         "Number of results of body should be equal to number of iter args");

  // 6. Yield all the results of the tiled operation.
  SmallVector<Value> yieldedValues;
  for (auto [tiledValue, destinationTensor, resultOffset, resultSize] :
       llvm::zip_equal(tiledResults, destinationTensors, resultOffsets,
                       resultSizes)) {
    SmallVector<OpFoldResult> resultStride(resultOffset.size(),
                                           rewriter.getIndexAttr(1));
    auto insertSlice = rewriter.create<tensor::InsertSliceOp>(
        loc, tiledValue, destinationTensor, resultOffset, resultSize,
        resultStride);
    yieldedValues.push_back(insertSlice);
  }
  rewriter.create<scf::YieldOp>(loc, yieldedValues);

  // Add the scf.yield operations for all the outer loops.
  for (auto [outerLoop, innerLoop] :
       llvm::zip_equal(MutableArrayRef(loops).drop_back(),
                       MutableArrayRef(loops).drop_front())) {
    rewriter.setInsertionPointToEnd(
        cast<scf::ForOp>(outerLoop.getOperation()).getBody());
    rewriter.create<scf::YieldOp>(outerLoop.getLoc(), innerLoop->getResults());
  }
  return success();
}

/// Generate the tile-loop nest using `scf.forall` operation.
/// - `loopRanges` specifies the lb, ub and step of the untiled iteration space.
/// - `tileSizes` is the tile sizes to use. Zero represent untiled loops.
/// - `destinationTensors` are the init values to use for the outer most loop.
/// - `mappingVector` is the mapping attributes to use for loop construction.
///   Can be empty.
/// - `yieldTiledValuesFn` is called to generated the loop body of the inner
/// most
///    loop.
/// - `loops` is an in-out parameter into which the generated loops are
///    populated.
static LogicalResult generateLoopNestUsingForallOp(
    RewriterBase &rewriter, Location loc, ArrayRef<Range> loopRanges,
    ArrayRef<OpFoldResult> tileSizes, ArrayRef<OpFoldResult> numTiles,
    ArrayRef<Attribute> mappingVector, ValueRange destinationTensors,
    YieldTiledValuesFn tiledBodyFn, SmallVector<LoopLikeOpInterface> &loops) {
  assert(!loopRanges.empty() && "unexpected empty loop ranges");
  assert(loopRanges.size() == tileSizes.size() &&
         "expected as many tile sizes as loop ranges");
  assert((numTiles.empty() || numTiles.size() == loopRanges.size()) &&
         "expected max number of tiles to be either empty or equal to number "
         "of loops");
  OpBuilder::InsertionGuard guard(rewriter);
  SmallVector<OpFoldResult> offsets(loopRanges.size()),
      sizes(loopRanges.size());

  std::optional<ArrayAttr> mappingAttr;
  if (!mappingVector.empty())
    mappingAttr = rewriter.getArrayAttr(mappingVector);

  scf::ForallOp forallOp;
  SmallVector<OpFoldResult> lbs, ubs, steps;
  if (numTiles.empty()) {
    for (auto [tileSize, loopRange] : llvm::zip_equal(tileSizes, loopRanges)) {
      if (isConstantIntValue(tileSize, 0))
        continue;
      lbs.push_back(loopRange.offset);
      ubs.push_back(loopRange.size);
      steps.push_back(tileSize);
    }
    assert(!lbs.empty() && "Expected at least one loop range");
    forallOp = rewriter.create<scf::ForallOp>(loc, lbs, ubs, steps,
                                              destinationTensors, mappingAttr);
  } else {
    SmallVector<OpFoldResult> numThreads;
    for (auto maxNumTile : numTiles) {
      if (!isConstantIntValue(maxNumTile, 0))
        numThreads.push_back(maxNumTile);
    }
    forallOp = rewriter.create<scf::ForallOp>(loc, numThreads,
                                              destinationTensors, mappingAttr);
  }
  loops.push_back(forallOp);

  rewriter.setInsertionPoint(forallOp.getTerminator());
  destinationTensors = forallOp.getRegionOutArgs();

  SmallVector<Value> tiledResults;
  SmallVector<SmallVector<OpFoldResult>> resultOffsets, resultSizes;
  if (failed(tiledBodyFn(rewriter, loc, forallOp.getInductionVars(),
                         destinationTensors, tiledResults, resultOffsets,
                         resultSizes)))
    return rewriter.notifyMatchFailure(loc, "failed to generate loop body");

  rewriter.setInsertionPointToEnd(forallOp.getTerminator().getBody());
  for (auto [tiledValue, destinationTensor, resultOffset, resultSize] :
       llvm::zip_equal(tiledResults, destinationTensors, resultOffsets,
                       resultSizes)) {
    SmallVector<OpFoldResult> resultStride(resultOffset.size(),
                                           rewriter.getIndexAttr(1));

    rewriter.create<tensor::ParallelInsertSliceOp>(
        loc, tiledValue, destinationTensor, resultOffset, resultSize,
        resultStride);
  }
  return success();
}

/// Generate the tile-loop nest using the loop construct specifed in `options`.
/// - `options`: Tiling options specified.
/// - `loopRanges` specifies the lb, ub and step of the untiled iteration space.
/// - `tileSizes` is the tile sizes to use. Zero represent untiled loops.
/// - `destinationTensors` are the init values to use for the outer most loop.
/// - `yieldTiledValuesFn` is called to generated the loop body of the inner
/// most
///    loop.
/// - `loops` is an in-out parameter into which the generated loops are
///    populated.
static LogicalResult generateLoopNest(
    RewriterBase &rewriter, Location loc, const scf::SCFTilingOptions &options,
    ArrayRef<Range> loopRanges, ArrayRef<OpFoldResult> tileSizes,
    ArrayRef<OpFoldResult> numTiles, ValueRange destinationTensors,
    YieldTiledValuesFn tiledBodyFn, SmallVector<LoopLikeOpInterface> &loops) {
  // If the tile sizes are all zero, no loops are generated. Just call the
  // callback function to handle untiled case.
  if (llvm::all_of(tileSizes, isZeroIndex)) {
    SmallVector<Value> tiledResults;
    SmallVector<SmallVector<OpFoldResult>> resultOffsets, resultSizes;
    return tiledBodyFn(rewriter, loc, ValueRange{}, destinationTensors,
                       tiledResults, resultOffsets, resultSizes);
  }
  if (options.loopType == scf::SCFTilingOptions::LoopType::ForOp) {
    return generateLoopNestUsingForOp(rewriter, loc, loopRanges, tileSizes,
                                      numTiles, destinationTensors, tiledBodyFn,
                                      loops);
  }
  if (options.loopType == scf::SCFTilingOptions::LoopType::ForallOp) {
    return generateLoopNestUsingForallOp(
        rewriter, loc, loopRanges, tileSizes, numTiles, options.mappingVector,
        destinationTensors, tiledBodyFn, loops);
  }
  return rewriter.notifyMatchFailure(loc, "unhandled loop type");
}

/// Append the specified additional `newInitOperands` operands to the
/// loops existing `init` operands (or similar), and replace `loopOp` with
/// the new loop that has the additional init operands. The loop body of
/// this loop is moved over to the new loop. `yieldTiledValuesFn`
/// is called to get the new tiled values returned, and the offset
/// and sizes at which the tiled value is inserted into the
/// new region iter_args that correspond to the newly added init operands.
template <typename LoopType>
FailureOr<LoopLikeOpInterface>
yieldTiledValuesAndReplaceLoop(LoopType loopOp, RewriterBase &rewriter,
                               ValueRange newInitOperands,
                               YieldTiledValuesFn yieldTiledValuesFn) {
  return rewriter.notifyMatchFailure(loopOp, "unhandled loop type");
}

/// Implementation of `yieldTiledValuesAndReplaceLoop` for `scf.for`.
template <>
FailureOr<LoopLikeOpInterface> yieldTiledValuesAndReplaceLoop<scf::ForOp>(
    scf::ForOp loopOp, RewriterBase &rewriter, ValueRange newInitOperands,
    YieldTiledValuesFn yieldTiledValuesFn) {
  OpBuilder::InsertionGuard g(rewriter);
  Location loc = loopOp.getLoc();
  rewriter.setInsertionPoint(loopOp);

  auto inits = llvm::to_vector(loopOp.getInitArgs());
  inits.append(newInitOperands.begin(), newInitOperands.end());
  auto newLoop = rewriter.create<scf::ForOp>(
      loc, loopOp.getLowerBound(), loopOp.getUpperBound(), loopOp.getStep(),
      inits, [](OpBuilder &, Location, Value, ValueRange) {});

  // Move the loop body to the new op.
  Block *loopBody = loopOp.getBody();
  Block *newLoopBody = newLoop.getBody();
  rewriter.mergeBlocks(
      loopBody, newLoopBody,
      newLoopBody->getArguments().take_front(loopBody->getNumArguments()));

  auto yieldOp = cast<scf::YieldOp>(newLoopBody->getTerminator());
  rewriter.setInsertionPoint(yieldOp);

  SmallVector<Value> tiledValues;
  SmallVector<SmallVector<OpFoldResult>> resultOffsets, resultSizes;
  ValueRange newRegionIterArgs =
      newLoop.getRegionIterArgs().take_back(newInitOperands.size());
  if (failed(yieldTiledValuesFn(rewriter, loc, newLoop.getInductionVar(),
                                newRegionIterArgs, tiledValues, resultOffsets,
                                resultSizes))) {
    rewriter.eraseOp(newLoop);
    return rewriter.notifyMatchFailure(loopOp, "failed to get tiled values");
  }

  SmallVector<Value> newYieldValues = llvm::to_vector(yieldOp.getOperands());
  for (auto [tiledValue, regionIterArg, resultOffset, resultSize] :
       llvm::zip_equal(tiledValues, newRegionIterArgs, resultOffsets,
                       resultSizes)) {
    SmallVector<OpFoldResult> resultStride(resultOffset.size(),
                                           rewriter.getIndexAttr(1));
    Value insert = rewriter.create<tensor::InsertSliceOp>(
        yieldOp->getLoc(), tiledValue, regionIterArg, resultOffset, resultSize,
        resultStride);
    newYieldValues.push_back(insert);
  }

  rewriter.replaceOpWithNewOp<scf::YieldOp>(yieldOp, newYieldValues);
  rewriter.replaceOp(loopOp,
                     newLoop->getResults().take_front(loopOp.getNumResults()));
  return cast<LoopLikeOpInterface>(newLoop.getOperation());
}

/// Implementation of `yieldTiledValuesAndReplaceLoop` for `scf.forall`
template <>
FailureOr<LoopLikeOpInterface> yieldTiledValuesAndReplaceLoop<scf::ForallOp>(
    scf::ForallOp loopOp, RewriterBase &rewriter, ValueRange newInitOperands,
    YieldTiledValuesFn yieldTiledValuesFn) {
  OpBuilder::InsertionGuard g(rewriter);
  Location loc = loopOp.getLoc();
  rewriter.setInsertionPoint(loopOp);
  auto inits = llvm::to_vector(loopOp.getOutputs());
  inits.append(newInitOperands.begin(), newInitOperands.end());
  auto newLoop = rewriter.create<scf::ForallOp>(
      loc, loopOp.getMixedLowerBound(), loopOp.getMixedUpperBound(),
      loopOp.getMixedStep(), inits, loopOp.getMapping(),
      [](OpBuilder &, Location, ValueRange) {});

  // Move the region of the current block to the newly created op.
  Block *loopBody = loopOp.getBody();
  Block *newLoopBody = newLoop.getBody();
  rewriter.mergeBlocks(
      loopBody, newLoopBody,
      newLoopBody->getArguments().take_front(loopBody->getNumArguments()));

  auto terminator = cast<scf::InParallelOp>(newLoopBody->getTerminator());
  rewriter.setInsertionPoint(terminator);
  SmallVector<Value> tiledValues;
  SmallVector<SmallVector<OpFoldResult>> resultOffsets, resultSizes;
  ValueRange regionIterArgs =
      newLoop.getRegionIterArgs().take_back(newInitOperands.size());
  if (failed(yieldTiledValuesFn(rewriter, loc, newLoop.getInductionVars(),
                                regionIterArgs, tiledValues, resultOffsets,
                                resultSizes))) {
    rewriter.eraseOp(newLoop);
    return rewriter.notifyMatchFailure(loopOp,
                                       "failed to get yielded tiled values");
  }

  // Update the terminator.
  rewriter.setInsertionPointToEnd(terminator.getBody());

  for (auto [tiledValue, iterArg, resultOffset, resultSize] : llvm::zip_equal(
           tiledValues, regionIterArgs, resultOffsets, resultSizes)) {
    SmallVector<OpFoldResult> resultStride(resultOffset.size(),
                                           rewriter.getIndexAttr(1));
    rewriter.create<tensor::ParallelInsertSliceOp>(
        terminator.getLoc(), tiledValue, iterArg, resultOffset, resultSize,
        resultStride);
  }

  rewriter.replaceOp(loopOp,
                     newLoop->getResults().take_front(loopOp.getNumResults()));
  return cast<LoopLikeOpInterface>(newLoop.getOperation());
}

/// Implementation of `yieldTiledValuesAndReplaceLoop` for
/// `LoopLikeOpInterface`, that just dispatches to the implementation for each
/// supported loop type.
FailureOr<LoopLikeOpInterface> yieldTiledValuesAndReplaceLoop(
    LoopLikeOpInterface loopLikeOp, RewriterBase &rewriter,
    ValueRange newInitOperands, YieldTiledValuesFn yieldTiledValuesFn) {
  return TypeSwitch<Operation *, FailureOr<LoopLikeOpInterface>>(
             loopLikeOp.getOperation())
      .Case<scf::ForOp, scf::ForallOp>(
          [&](auto loopOp) -> FailureOr<LoopLikeOpInterface> {
            return yieldTiledValuesAndReplaceLoop(
                loopOp, rewriter, newInitOperands, yieldTiledValuesFn);
          })
      .Default([&](auto loopOp) -> FailureOr<LoopLikeOpInterface> {
        return rewriter.notifyMatchFailure(loopOp, "unhandled loop type");
      });
}

/// Method to add new init values to a loop nest. Updates `loops` in-place with
/// new loops that use the `newInitValues`.
/// The outer-loops are updated to yield the new result values of the inner
/// loop. For the innermost loop, the call back `getNewYields` is invoked to get
/// the additional values to yield form the innermost loop.
static LogicalResult addInitOperandsToLoopNest(
    RewriterBase &rewriter, MutableArrayRef<LoopLikeOpInterface> loops,
    ValueRange newInitValues, YieldTiledValuesFn getNewTiledYieldsFn) {
  SmallVector<scf::ForOp> newLoops;
  if (loops.empty())
    return success();
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(loops.front());

  SmallVector<Value> ivs;
  for (auto &loop : loops.drop_back()) {
    rewriter.setInsertionPoint(loop);

    // if loops.size() > 1 we assume that scf.for is used for the loops.
    auto forLoop = cast<scf::ForOp>(loop.getOperation());

    // Create a new loop with the new init values for this loop.
    SmallVector<Value> newInits = llvm::to_vector(forLoop.getInitArgs());
    newInits.append(newInitValues.begin(), newInitValues.end());
    auto newLoop = rewriter.create<scf::ForOp>(
        forLoop.getLoc(), forLoop.getLowerBound(), forLoop.getUpperBound(),
        forLoop.getStep(), newInits,
        [&](OpBuilder &b, Location loc, Value iv, ValueRange iterArgs) {});

    // Merge the body of the new loop with the body of the old loops.
    SmallVector<Value> sourceBlockArgs;
    sourceBlockArgs.push_back(newLoop.getInductionVar());
    auto newRegionIterArgs = newLoop.getRegionIterArgs();
    sourceBlockArgs.append(
        newRegionIterArgs.begin(),
        std::next(newRegionIterArgs.begin(), forLoop.getNumResults()));
    rewriter.mergeBlocks(forLoop.getBody(), newLoop.getBody(), sourceBlockArgs);
    rewriter.replaceOp(
        forLoop, newLoop.getResults().take_front(forLoop.getNumResults()));
    loop = newLoop;
    ivs.push_back(newLoop.getInductionVar());
    newInitValues = newLoop.getRegionIterArgs().take_back(newInitValues.size());
  }

  // Update the loop body of the innermost loop to get new yield values.
  LoopLikeOpInterface innerMostLoop = loops.back();
  FailureOr<LoopLikeOpInterface> newInnerMostLoop =
      yieldTiledValuesAndReplaceLoop(innerMostLoop, rewriter, newInitValues,
                                     getNewTiledYieldsFn);

  if (failed(newInnerMostLoop))
    return innerMostLoop.emitOpError("failed to return additional yields");
  loops.back() = newInnerMostLoop.value();

  // Make all other loops except the innermost loops yield the values returned
  // by the inner loop.
  for (auto [outerLoop, innerLoop] :
       llvm::zip_equal(loops.drop_back(), loops.drop_front())) {
    // Again assume that all the outer loops are scf.for operations.
    auto outerForLoop = cast<scf::ForOp>(outerLoop);
    auto outerLoopYield =
        cast<scf::YieldOp>(outerForLoop.getBody()->getTerminator());
    SmallVector<Value> newYields =
        llvm::to_vector(outerLoopYield.getOperands());
    ValueRange additionalYields =
        innerLoop->getResults().take_back(newInitValues.size());
    newYields.append(additionalYields.begin(), additionalYields.end());
    rewriter.setInsertionPoint(outerLoopYield);
    rewriter.replaceOpWithNewOp<scf::YieldOp>(outerLoopYield, newYields);
  }
  return success();
}

/// Implementation of tiling transformation of `op` that implements the
/// `TilingInterface` using `scf.for` to iterate over the tiles.
FailureOr<scf::SCFTilingResult>
mlir::scf::tileUsingSCF(RewriterBase &rewriter, TilingInterface op,
                        const scf::SCFTilingOptions &options) {
  if (failed(verifyTileSizeOptions(rewriter, op.getLoc(), options))) {
    return failure();
  }

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfter(op);

  // 1. Get the range of the loops that are represented by the operation.
  SmallVector<Range> iterationDomain = op.getIterationDomain(rewriter);

  // 2. Materialize the tile sizes or max num tiles;
  SmallVector<OpFoldResult> tileSizes, numTiles;
  std::tie(tileSizes, numTiles) =
      getTileSizesAndNumTiles(rewriter, op, iterationDomain, options);

  // 3. If there is an interchange specified, permute the iteration domain and
  // the tile sizes.
  SmallVector<int64_t> interchangeVector;
  if (!options.interchangeVector.empty()) {
    interchangeVector = fillInterchangeVector(options.interchangeVector,
                                              iterationDomain.size());
    assert(isPermutationVector(interchangeVector) &&
           "expected interchange vector to be a permutation");

    applyPermutationToVector(iterationDomain, interchangeVector);
    applyPermutationToVector(tileSizes, interchangeVector);
    if (!numTiles.empty())
      applyPermutationToVector(numTiles, interchangeVector);
  }

  FailureOr<TilingResult> tilingResult;
  // 4. Define the lambda function used later to generate the body of the
  // innermost tiled loop.
  YieldTiledValuesFn innerYieldTiledValuesFn =
      [&](RewriterBase &rewriter, Location loc, ValueRange ivs,
          ValueRange regionIterArgs, SmallVector<Value> &tiledResults,
          SmallVector<SmallVector<OpFoldResult>> &resultOffsets,
          SmallVector<SmallVector<OpFoldResult>> &resultSizes)
      -> LogicalResult {
    // 4a. Compute the `offsets` and `sizes` to use for tiling.
    SmallVector<OpFoldResult> offsets, sizes;
    std::tie(offsets, sizes) = getTileOffsetAndSizes(
        rewriter, loc, ivs, iterationDomain, tileSizes, !numTiles.empty());

    // 4b. If interchange was provided, apply inverse of the interchange
    //     to get back the offsets/sizes in the order to be specified.
    if (!interchangeVector.empty()) {
      auto inversePermutation = invertPermutationVector(interchangeVector);
      applyPermutationToVector(offsets, inversePermutation);
      applyPermutationToVector(sizes, inversePermutation);
    }

    // 5. Generate the tiled implementation within the inner most loop.

    // 5a. Clone the operation within the loop body.
    auto clonedOp = cast<TilingInterface>(
        cloneOpAndUpdateDestinationArgs(rewriter, op, regionIterArgs));

    // 5b. Early return cloned op if tiling is not happening. We can not return
    // the original op because it could lead to
    // `rewriter.replaceOp(op, op->getResults())` and users would get crash.
    if (llvm::all_of(tileSizes, isZeroIndex)) {
      tiledResults.append(clonedOp->result_begin(), clonedOp->result_end());
      tilingResult =
          TilingResult{/*tiledOps=*/{clonedOp}, clonedOp->getResults()};
      return success();
    }

    // 5c. Tile the cloned operation.
    tilingResult = clonedOp.getTiledImplementation(rewriter, offsets, sizes);
    if (failed(tilingResult)) {
      rewriter.eraseOp(clonedOp);
      return op.emitOpError("faild to tile operation");
    }

    // 5d. Delete the cloned operation.
    rewriter.eraseOp(clonedOp);

    // 5e. Compute the offsets at which the result values are to be inserted
    //     back into its destinations.
    for (auto [index, tiledValue] :
         llvm::enumerate(tilingResult->tiledValues)) {
      tiledResults.push_back(tiledValue);
      SmallVector<OpFoldResult> resultOffset, resultSize;
      if (failed(op.getResultTilePosition(rewriter, index, offsets, sizes,
                                          resultOffset, resultSize))) {
        for (auto op : tilingResult->tiledOps) {
          rewriter.eraseOp(op);
        }
        return rewriter.notifyMatchFailure(
            op, "failed to get slice of result produced");
      }
      resultOffsets.emplace_back(std::move(resultOffset));
      resultSizes.emplace_back(std::move(resultSize));
    }

    return success();
  };

  // 6. Find the destination tensors to use for the operation.
  SmallVector<Value> destinationTensors;
  if (failed(tensor::getOrCreateDestinations(rewriter, op.getLoc(), op,
                                             destinationTensors))) {
    return rewriter.notifyMatchFailure(op,
                                       "unable to create destination tensors");
  }

  // 7. Generate the tiled loops nest using the callback defined above.
  SmallVector<LoopLikeOpInterface> loops;
  if (failed(generateLoopNest(rewriter, op.getLoc(), options, iterationDomain,
                              tileSizes, numTiles, destinationTensors,
                              innerYieldTiledValuesFn, loops)))
    return op.emitOpError("failed to generate tiling loops");
  assert(succeeded(tilingResult) &&
         "expected tiling result to be computed after loop generation");

  // If loops are empty, the tiled op is used as the replacement for the untiled
  // op.
  if (loops.empty()) {
    return scf::SCFTilingResult{tilingResult->tiledOps, loops,
                                tilingResult->tiledValues};
  }

  SmallVector<Value> replacements = llvm::map_to_vector(
      loops.front()->getResults(), [](OpResult r) -> Value { return r; });
  return scf::SCFTilingResult{tilingResult->tiledOps, loops, replacements};
}

FailureOr<scf::SCFReductionTilingResult>
mlir::scf::tileReductionUsingScf(RewriterBase &b,
                                 PartialReductionOpInterface op,
                                 ArrayRef<OpFoldResult> tileSizes) {
  Location loc = op.getLoc();
  // Ops implementing PartialReductionOpInterface are expected to implement
  // TilingInterface.
  auto tilingInterfaceOp = cast<TilingInterface>(op.getOperation());
  SmallVector<Range> iterationDomain = tilingInterfaceOp.getIterationDomain(b);
  auto tileSizesVector = llvm::to_vector(tileSizes);
  if (tileSizesVector.size() < iterationDomain.size()) {
    auto zero = b.getIndexAttr(0);
    tileSizesVector.append(iterationDomain.size() - tileSizesVector.size(),
                           zero);
  }
  SmallVector<utils::IteratorType> iterators =
      tilingInterfaceOp.getLoopIteratorTypes();

  SmallVector<int> reductionDims;
  for (auto [idx, iteratorType] :
       llvm::enumerate(tilingInterfaceOp.getLoopIteratorTypes())) {
    if (iteratorType == utils::IteratorType::reduction)
      reductionDims.push_back(idx);
  }

  // 2. create the inital tensor value.
  FailureOr<SmallVector<Value>> maybeInitTensors =
      op.generateInitialTensorForPartialReduction(b, loc, tileSizesVector,
                                                  reductionDims);
  if (failed(maybeInitTensors)) {
    return b.notifyMatchFailure(op, "Failed to create initial tensors.");
  }
  SmallVector<Value> &initTensors = maybeInitTensors.value();

  // 3. Define the callback to use for generating the inner most tile loop body.
  SmallVector<Operation *> parallelTiledOps;
  auto innerYieldTiledValuesFn =
      [&](RewriterBase &rewriter, Location loc, ValueRange ivs,
          ValueRange regionIterArgs, SmallVector<Value> &tiledResult,
          SmallVector<SmallVector<OpFoldResult>> &resultOffsets,
          SmallVector<SmallVector<OpFoldResult>> &resultSizes)
      -> LogicalResult {
    SmallVector<OpFoldResult> offsets, sizes;
    {
      int materializedLoopNum = 0;
      for (auto [tileSize, loopRange] :
           llvm::zip_equal(tileSizesVector, iterationDomain)) {
        if (isConstantIntValue(tileSize, 0)) {
          offsets.push_back(loopRange.offset);
          sizes.push_back(loopRange.size);
          continue;
        }
        Value iv = ivs[materializedLoopNum++];
        offsets.push_back(iv);
        sizes.push_back(
            getBoundedTileSize(rewriter, loc, loopRange, iv, tileSize));
      }
    }

    // 4a. Clone the operation.
    {
      auto clonedOp = cast<PartialReductionOpInterface>(
          cloneOpAndUpdateDestinationArgs(b, op, regionIterArgs));

      // 4b. Tile the cloned operation.
      FailureOr<TilingResult> partialTilingResult =
          clonedOp.tileToPartialReduction(b, loc, regionIterArgs, offsets,
                                          sizes, reductionDims);
      if (failed(partialTilingResult)) {
        return failure();
      }
      std::swap(parallelTiledOps, partialTilingResult->tiledOps);
      std::swap(tiledResult, partialTilingResult->tiledValues);

      // 4c. Delete the cloned operation.
      b.eraseOp(clonedOp);
    }

    // 4d. Compute the offsets and sizes needed to insert the result of the
    // tiled value back into destination before yielding the destination.
    for (auto result : tiledResult) {
      SmallVector<OpFoldResult> outOffsets(offsets.size(), b.getIndexAttr(0));
      resultOffsets.emplace_back(std::move(outOffsets));

      SmallVector<OpFoldResult> outSizes;
      for (size_t i = 0; i < offsets.size(); i++) {
        outSizes.push_back(tensor::getMixedSize(b, loc, result, i));
      }
      resultSizes.emplace_back(std::move(outSizes));
    }
    return success();
  };

  // 5. Generate the tiled implementation using the destination tensors.
  SmallVector<LoopLikeOpInterface> loops;
  scf::SCFTilingOptions options;
  options.setLoopType(scf::SCFTilingOptions::LoopType::ForOp);
  if (failed(generateLoopNest(b, loc, options, iterationDomain, tileSizesVector,
                              /*numTiles=*/ArrayRef<OpFoldResult>{},
                              initTensors, innerYieldTiledValuesFn, loops)))
    return b.notifyMatchFailure(op, "failed to tile for parallel reduction");

  SmallVector<Value> replacements = llvm::map_to_vector(
      loops.front()->getResults(), [](OpResult r) -> Value { return r; });

  // 5. Apply the merge reduction to combine all the partial values.
  b.setInsertionPointAfter(*loops.begin());
  FailureOr<MergeResult> mergeResult =
      op.mergeReductions(b, loc, replacements, reductionDims);
  if (failed(mergeResult)) {
    return failure();
  }
  b.replaceOp(op, mergeResult->replacements);

  SCFReductionTilingResult reductionTilingResult;
  std::swap(reductionTilingResult.parallelTiledOps, parallelTiledOps);
  std::swap(reductionTilingResult.mergeOps, mergeResult->mergeOps);
  std::swap(reductionTilingResult.initialValues, initTensors);
  std::swap(reductionTilingResult.loops, loops);
  std::swap(reductionTilingResult.replacements, mergeResult->replacements);

  return reductionTilingResult;
}

//===----------------------------------------------------------------------===//
// tileConsumerAndFuseProducersUsingSCF implementation.
//===----------------------------------------------------------------------===//

/// Return the untiled producer whose slice is used in a tiled consumer. The
/// method traverses the tile loop nest (`loops`) if needed, and returns the
/// `iter_args` of the outer most that is encountered. Traversing the iter_args
/// indicates that this is a destination operand of the consumer. If there was
/// no loop traversal needed, the second value of the returned tuple is empty.
static std::tuple<OpResult, std::optional<OpOperand *>>
getUntiledProducerFromSliceSource(OpOperand *source,
                                  ArrayRef<LoopLikeOpInterface> loops) {
  std::optional<OpOperand *> destinationIterArg;
  auto loopIt = loops.rbegin();
  while (auto iterArg = dyn_cast<BlockArgument>(source->get())) {
    auto loop = *loopIt;
    if (iterArg.getOwner()->getParentOp() != loop)
      break;
    source = loop.getTiedLoopInit(iterArg);
    loopIt++;
  }
  if (loopIt == loops.rend())
    destinationIterArg = source;
  return {dyn_cast<OpResult>(source->get()), destinationIterArg};
}

/// Implementation of fusing producer of a single slice by computing the
/// slice of the producer in-place.
std::optional<scf::SCFFuseProducerOfSliceResult>
mlir::scf::tileAndFuseProducerOfSlice(
    RewriterBase &rewriter, tensor::ExtractSliceOp candidateSliceOp,
    MutableArrayRef<LoopLikeOpInterface> loops) {
  // 1. Get the producer of the source (potentially walking through
  // `iter_args` of nested `scf.for`)
  auto [fusableProducer, destinationInitArg] =
      getUntiledProducerFromSliceSource(&candidateSliceOp.getSourceMutable(),
                                        loops);
  if (!fusableProducer)
    return std::nullopt;
  unsigned resultNumber = fusableProducer.getResultNumber();

  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(candidateSliceOp);

  // 2. Clone the fused producer
  // 2a. Compute the destination operands to use for the cloned operation.
  SmallVector<Value> origDestinationTensors, clonedOpDestinationTensors;
  Operation *fusableProducerOp = fusableProducer.getOwner();
  if (isa<DestinationStyleOpInterface>(fusableProducerOp) &&
      failed(tensor::getOrCreateDestinations(
          rewriter, fusableProducerOp->getLoc(), fusableProducerOp,
          origDestinationTensors)))
    return std::nullopt;

  clonedOpDestinationTensors = origDestinationTensors;
  if (destinationInitArg &&
      isa<DestinationStyleOpInterface>(fusableProducerOp)) {
    // 2b. If the producer is also destination style, then to maintain the
    // destination passing style, update the destination of the producer to be
    // the source of the slice.
    clonedOpDestinationTensors[resultNumber] = candidateSliceOp.getSource();
  }
  // 2c. Clone the fused producer.
  Operation *clonedProducerOp = cloneOpAndUpdateDestinationArgs(
      rewriter, fusableProducerOp, clonedOpDestinationTensors);
  // 2d. Update the source of the candidateSlice to be the cloned producer.
  //     Easier to just clone the slice with different source since replacements
  //     and DCE of cloned ops becomes easier
  SmallVector<Value> candidateSliceOpOperands =
      llvm::to_vector(candidateSliceOp->getOperands());
  candidateSliceOpOperands[0] = clonedProducerOp->getResult(resultNumber);
  tensor::ExtractSliceOp clonedCandidateSliceOp =
      mlir::clone(rewriter, candidateSliceOp,
                  candidateSliceOp->getResultTypes(), candidateSliceOpOperands);

  // 3. Generate the tiled implementation of the producer of the source
  FailureOr<TilingResult> tileAndFuseResult =
      tensor::replaceExtractSliceWithTiledProducer(
          rewriter, clonedCandidateSliceOp,
          clonedProducerOp->getResult(resultNumber));
  if (failed(tileAndFuseResult))
    return std::nullopt;
  // Note: Do not delete the candidateSliceOp, since its passed in from the
  // caller.
  rewriter.replaceAllUsesWith(candidateSliceOp,
                              tileAndFuseResult->tiledValues[0]);
  rewriter.eraseOp(clonedCandidateSliceOp);
  rewriter.eraseOp(clonedProducerOp);

  // 3. If the slice is for a destination operand, for example,
  //
  // ```mlir
  // %0 = linalg.init
  // %1 = linalg.fill .. outs(%0 : )
  // %2 = scf.for .. iter_args(%arg0 = %1) {
  //   %3 = scf.for .. iter_args(%arg1 = %arg0) {
  //     %4 = tensor.extract_slice %arg1 [..]
  //     .. = linalg.matmul .. outs(%4 : )
  //   }
  // }
  // ```
  //
  // the IR is currently
  //
  // ```
  // %0 = linalg.init
  // %1 = linalg.fill
  // %2 = scf.for .. iter_args(%arg0 = %1 /* incorrect value */ ) {
  //   %3 = scf.for .. iter_args(%arg1 = %arg0) {
  //     %4 = tensor.extract_slice %arg1[..]
  //     %5 = linalg.fill .. outs(%4 : )
  //     .. = linalg.matmul .. outs(%5 : )
  //   }
  // }
  // ```
  //
  // The untiled `linalg.fill` is still used as the `init_value` since it
  // was originally a destination operand of the untiled `linalg.matmul`.
  // When fusing an operand that is a destination operand, the iter_arg of
  // the outer most loop should be changed to use the destination of the
  // fused operation. With this the IR will be.
  //
  // ```
  // %0 = linalg.init
  // %1 = scf.for .. iter_args(%arg0 = %0 /* corrected value */ ) {
  //   %2 = scf.for .. iter_args(%arg1 = %arg0) {
  //     %3 = tensor.extract_slice %arg1[..]
  //     %4 = linalg.fill .. outs(%3 : )
  //     .. = linalg.matmul .. outs(%4 : )
  //   }
  // }
  // ```
  if (destinationInitArg &&
      isa<DestinationStyleOpInterface>(fusableProducerOp) && !loops.empty()) {
    loops.front()
        ->getOpOperands()[destinationInitArg.value()->getOperandNumber()]
        .set(origDestinationTensors[resultNumber]);
  }
  return scf::SCFFuseProducerOfSliceResult{fusableProducer,
                                           tileAndFuseResult->tiledValues[0],
                                           tileAndFuseResult->tiledOps};
}

/// Reconstruct the fused producer from within the tiled-and-fused code.
LogicalResult mlir::scf::yieldReplacementForFusedProducer(
    RewriterBase &rewriter, tensor::ExtractSliceOp sliceOp,
    scf::SCFFuseProducerOfSliceResult fusedProducerInfo,
    MutableArrayRef<LoopLikeOpInterface> loops,
    ArrayRef<unsigned> yieldResultNumber) {
  if (loops.empty())
    return success();

  Operation *originalOwner = fusedProducerInfo.origProducer.getOwner(),
            *tiledOwner = fusedProducerInfo.tiledOps[0];

  Location loc = originalOwner->getLoc();
  // a. collect all init Value to be appended
  SmallVector<unsigned> initNumberList =
      yieldResultNumber.empty() ? llvm::to_vector(llvm::seq<unsigned>(
                                      0, originalOwner->getNumResults()))
                                : llvm::to_vector(yieldResultNumber);
  SmallVector<Value> initValueList;
  for (const auto &resultNumber : initNumberList) {
    FailureOr<Value> initValue = tensor::getOrCreateDestination(
        rewriter, loc, originalOwner->getResult(resultNumber));
    if (succeeded(initValue)) {
      initValueList.push_back(initValue.value());
    } else {
      return failure();
    }
  }

  YieldTiledValuesFn newYieldValuesFn =
      [&](RewriterBase &innerRewriter, Location loc, ValueRange /*ivs*/,
          ValueRange newRegionIterArgs, SmallVector<Value> &tiledResult,
          SmallVector<SmallVector<OpFoldResult>> &tiledOffset,
          SmallVector<SmallVector<OpFoldResult>> &tiledSizes) -> LogicalResult {
    OpBuilder::InsertionGuard g(innerRewriter);

    // get sliceOp tile information
    SmallVector<OpFoldResult> sliceOffset = sliceOp.getMixedOffsets(),
                              sliceSizes = sliceOp.getMixedSizes();

    // expect all strides of sliceOp being 1
    if (llvm::any_of(sliceOp.getMixedStrides(), [](OpFoldResult ofr) {
          return !isConstantIntValue(ofr, 1);
        }))
      return failure();

    unsigned sliceResultNumber =
        fusedProducerInfo.origProducer.getResultNumber();

    auto tilableOp = cast<TilingInterface>(originalOwner);
    // b. get iterDomain Offset and Sizes based on sliceOp tile
    SmallVector<OpFoldResult> iterDomainOffset, iterDomainSizes;
    // skip tensor.pack/unpack/pad, which expects single opResult
    if (tilableOp->getNumResults() > 1 &&
        failed(tilableOp.getIterationDomainTileFromResultTile(
            rewriter, sliceResultNumber, sliceOffset, sliceSizes,
            iterDomainOffset, iterDomainSizes))) {
      // In theory, it is unnecessary to raise an error here. Actually although
      // it fails to reconstruct the result tensor, it should not broke current
      // fusion anyway. The reason why we must return failure currently is that
      // the callback function `newYieldValuesFn` will be called after new init
      // operand(s) has already been appended. It will take more refactoring to
      // make sure the init operands are added consistently in the future. For
      // more details, please refer to:
      // https://github.com/llvm/llvm-project/pull/93144#discussion_r1643760814
      return failure();
    }

    // c. calculate offsets and sizes info of all OpResults respectively based
    // on iteration Domain Tile
    SmallVector<SmallVector<OpFoldResult>> offsetList, sizesList;
    for (const auto &resultNumber : initNumberList) {
      if (resultNumber == sliceResultNumber) {
        offsetList.push_back(sliceOffset);
        sizesList.push_back(sliceSizes);
      } else {
        assert(!iterDomainOffset.empty() && !iterDomainSizes.empty());
        // infer result tile according to the iteration domain tile
        SmallVector<OpFoldResult> offset, sizes;
        if (failed(tilableOp.getResultTilePosition(
                rewriter, resultNumber, iterDomainOffset, iterDomainSizes,
                offset, sizes))) {
          return failure();
        }
        offsetList.push_back(offset);
        sizesList.push_back(sizes);
      }
    }

    // d. create `extract_slice` for `iter_args` for DPS operation if necessary
    if (auto tiledDestStyleOp =
            dyn_cast<DestinationStyleOpInterface>(tiledOwner)) {
      rewriter.setInsertionPoint(tiledDestStyleOp);
      for (const auto &&[index, newRegionArg] :
           llvm::enumerate(newRegionIterArgs)) {
        auto destSlice = rewriter.create<tensor::ExtractSliceOp>(
            loc, newRegionArg, offsetList[index], sizesList[index],
            SmallVector<OpFoldResult>(offsetList[index].size(),
                                      rewriter.getIndexAttr(1)));
        unsigned resultNumber = initNumberList[index];
        rewriter.modifyOpInPlace(tiledDestStyleOp, [&]() {
          tiledDestStyleOp.getDpsInitsMutable()[resultNumber].set(destSlice);
        });
      }
    }

    // e. prepare tiled offset and sizes for later `insert_slice` creation by
    // caller
    Block *block = rewriter.getInsertionPoint()->getBlock();
    rewriter.setInsertionPoint(block->getTerminator());
    for (const auto &&[index, resultNumber] : llvm::enumerate(initNumberList)) {
      tiledResult.push_back(tiledOwner->getResult(resultNumber));
      tiledOffset.emplace_back(offsetList[index]);
      tiledSizes.emplace_back(sizesList[index]);
    }
    return success();
  };

  return addInitOperandsToLoopNest(rewriter, loops, initValueList,
                                   newYieldValuesFn);
}

/// Implementation of tile consumer and fuse producer greedily.
FailureOr<scf::SCFTileAndFuseResult>
mlir::scf::tileConsumerAndFuseProducersUsingSCF(
    RewriterBase &rewriter, TilingInterface consumer,
    const scf::SCFTileAndFuseOptions &options) {
  // This transformation is only valid for ops that return values (i.e. not
  // valid to use with operations that have memref operands).
  if (!consumer->getNumResults()) {
    return rewriter.notifyMatchFailure(
        consumer, "invalid pattern for op with no results");
  }

  // 1. First tile the consumer.
  SetVector<Operation *> fusedProducers, tiledAndFusedOps;
  llvm::SmallDenseMap<Value, size_t> origProducerToLoopResultNum;

  FailureOr<scf::SCFTilingResult> tilingResult =
      tileUsingSCF(rewriter, consumer, options.tilingOptions);

  if (failed(tilingResult))
    return rewriter.notifyMatchFailure(consumer, "failed to tile consumer");
  for (auto *tiledOp : tilingResult->tiledOps)
    tiledAndFusedOps.insert(tiledOp);

  // If there are no loops generated, fusion is immaterial.
  auto &loops = tilingResult->loops;
  if (loops.empty()) {
    DenseMap<Value, Value> replacements;
    for (auto [origVal, replacement] :
         llvm::zip_equal(consumer->getResults(), tilingResult->replacements)) {
      replacements[origVal] = replacement;
    }
    return scf::SCFTileAndFuseResult{fusedProducers, tiledAndFusedOps, loops,
                                     replacements};
  }

  // To keep track of replacements for now just record the map from the original
  // untiled value to the result number of the for loop. Since the loop gets
  // potentially replaced during fusion, keeping the value directly wont work.
  DenseMap<Value, size_t> origValToResultNumber;
  for (auto [index, result] : llvm::enumerate(consumer->getResults())) {
    origValToResultNumber[result] = index;
  }

  // 2. Typically, the operands of the tiled operation are slices of the
  //    operands of the untiled operation. These are expressed in IR using
  //    `tensor.extract_slice` operations with source being the operands of the
  //    untiled operation. Create a worklist of these `tensor.extract_slice`
  //    operations. If the producers of the source of the `tensor.extract_slice`
  //    can be tiled such that the tiled value is generated in-place, that
  //    effectively tiles + fuses the operations.
  auto addCandidateSlices = [](Operation *fusedOp,
                               std::deque<tensor::ExtractSliceOp> &candidates) {
    for (Value operand : fusedOp->getOperands())
      if (auto sliceOp = operand.getDefiningOp<tensor::ExtractSliceOp>())
        candidates.push_back(sliceOp);
  };

  std::deque<tensor::ExtractSliceOp> candidates;
  addCandidateSlices(tiledAndFusedOps.back(), candidates);
  OpBuilder::InsertionGuard g(rewriter);
  while (!candidates.empty()) {
    // Traverse the slices in BFS fashion.
    tensor::ExtractSliceOp candidateSliceOp = candidates.front();
    candidates.pop_front();

    // Find the original producer of the slice.
    auto [fusableProducer, destinationInitArg] =
        getUntiledProducerFromSliceSource(&candidateSliceOp.getSourceMutable(),
                                          loops);
    if (!fusableProducer)
      continue;

    auto [fuseSlice, yieldReplacement] = options.fusionControlFn(
        candidateSliceOp, fusableProducer, destinationInitArg.has_value());
    if (!fuseSlice)
      continue;

    // The operands of the fused producer might themselved be slices of
    // values produced by operations that implement the `TilingInterface`.
    // Add these operations to the worklist.
    std::optional<scf::SCFFuseProducerOfSliceResult> fusedResult =
        tileAndFuseProducerOfSlice(rewriter, candidateSliceOp, loops);
    if (!fusedResult)
      continue;

    if (yieldReplacement) {
      // Reconstruct and yield all opResult of fusableProducerOp by default. The
      // caller can specific which one to yield by designating optional argument
      // named `yieldResultNumber` of `yieldReplacementForFusedProducer`.
      Operation *fusableProducerOp = fusableProducer.getOwner();
      if (failed(yieldReplacementForFusedProducer(
              rewriter, candidateSliceOp, fusedResult.value(), loops))) {
        return rewriter.notifyMatchFailure(
            fusableProducerOp, "failed to replacement value for this "
                               "operation from within the tiled loop");
      }
      for (auto [index, result] :
           llvm::enumerate(fusableProducerOp->getResults())) {
        origValToResultNumber[result] = loops.front()->getNumResults() -
                                        fusableProducerOp->getNumResults() +
                                        index;
      }
    }

    if (Operation *tiledAndFusedOp =
            fusedResult->tiledAndFusedProducer.getDefiningOp()) {
      fusedProducers.insert(fusedResult->origProducer.getDefiningOp());
      tiledAndFusedOps.insert(tiledAndFusedOp);
      addCandidateSlices(tiledAndFusedOp, candidates);
    }
  }

  DenseMap<Value, Value> replacements;
  for (auto [origVal, resultNumber] : origValToResultNumber) {
    replacements[origVal] = loops.front()->getResult(resultNumber);
  }

  return scf::SCFTileAndFuseResult{fusedProducers, tiledAndFusedOps, loops,
                                   replacements};
}

//===----------------------------------------------------------------------===//
// tileAndFuseConsumerUsingSCF implementation.
//===----------------------------------------------------------------------===//

/// A utility function that checks whether the only use of the result of a
/// tensor.insert_slice op is in a scf.yield op.
static LogicalResult
checkAssumptionForFusingConsumer(tensor::InsertSliceOp candidateSliceOp) {
  Value result = candidateSliceOp.getResult();
  Value::use_range uses = result.getUses();
  if (!llvm::hasSingleElement(uses)) {
    LLVM_DEBUG(llvm::dbgs() << "Too many uses of the candidate slice op\n");
    return failure();
  }
  OpOperand &operandUse = (*uses.begin());
  Operation *userOp = operandUse.getOwner();
  if (!isa<scf::YieldOp>(userOp)) {
    LLVM_DEBUG(llvm::dbgs()
               << "Expected scf.yield to be the only user, but got -> "
               << (*userOp));
    return failure();
  }
  if (result.getDefiningOp()->getBlock() != userOp->getBlock()) {
    LLVM_DEBUG(llvm::dbgs() << "Expected tensor.insert_slice and scf.yield to "
                               "be in the same block\n");
    return failure();
  }
  return success();
}

/// Fetches the OpOperand of the only user (and use) of the value `val` which
/// implements `TilingInterface` and `DestinationStyleOpInterface`. Returns
/// failure otherwise.
static FailureOr<OpOperand *> getConsumerFromUses(Value val,
                                                  Block *containingOpBlock) {
  // Step 1. Check that the value has exactly one use.
  if (!llvm::hasSingleElement(val.getUses()))
    return failure();
  // Step 2. Get uses.
  OpOperand &operand = (*val.getUses().begin());
  Operation *consumerOp = operand.getOwner();
  // TODO: We have to init result of consumer before scf.for, use
  //       DestinationStyleOpInterface to get result shape from init for now.
  //       Add support for other op such as op has InferTypeOpInterface.
  if (!isa<TilingInterface>(consumerOp) ||
      !isa<DestinationStyleOpInterface>(consumerOp))
    return failure();
  if (containingOpBlock != consumerOp->getBlock())
    return failure();
  return &operand;
}

/// Fetch the untiled consumer of a scf.for's result which is yielded by a
/// tensor.insert_slice. This function makes the following assumptions :
/// 1.  tensor.insert_slice has scf.yield as its only user.
/// 2.  scf.for's corresponding result has only one use.
static FailureOr<OpOperand *>
getUntiledConsumerFromSlice(tensor::InsertSliceOp candidateSliceOp) {
  if (failed(checkAssumptionForFusingConsumer(candidateSliceOp)))
    return failure();
  Value sliceResult = candidateSliceOp.getResult();
  // Step 1. Fetch the corresponding output.
  OpOperand &yieldOpOperand = (*sliceResult.getUses().begin());
  unsigned resultNumber = yieldOpOperand.getOperandNumber();
  // Step 2. Check containing op is scf.for.
  Operation *containingOp = candidateSliceOp->getParentOp();
  auto forOp = dyn_cast<scf::ForOp>(containingOp);
  if (!forOp)
    return failure();
  Value resultingValue = forOp->getResult(resultNumber);

  return getConsumerFromUses(resultingValue, containingOp->getBlock());
}

/// Fetch the first untiled consumer of a scf.forall's result which is yielded
/// by a tensor.parallel_insert_slice.
static FailureOr<OpOperand *>
getUntiledConsumerFromSlice(tensor::ParallelInsertSliceOp candidateSliceOp) {
  // Step 1. Fetch the corresponding output
  Value sliceDest = candidateSliceOp.getDest();
  auto iterArg = dyn_cast<BlockArgument>(sliceDest);
  if (!iterArg)
    return failure();
  Operation *containingOp = iterArg.getOwner()->getParentOp();
  if (containingOp != candidateSliceOp->getParentOp()->getParentOp())
    return failure();
  // Step 2. Check that the containing op is scf.forall.
  auto forallOp = dyn_cast<scf::ForallOp>(containingOp);
  if (!forallOp)
    return failure();
  Value resultingValue =
      forallOp.getTiedOpResult(forallOp.getTiedOpOperand(iterArg));

  return getConsumerFromUses(resultingValue, containingOp->getBlock());
}

/// This utility currently checks whether the loop either :-
/// 1. Yields exactly one result.
/// 2. Has consumer op as its first user and other users to be in the same
/// containing block as that of consumer op's. Currently we clone the loop op
/// right before the consumer op in order to maintain a valid def-use chain.
/// This utility thus helps ensuring that no invalid IR is formed due to the
/// same.
static LogicalResult checkAssumptionForLoop(Operation *loopOp,
                                            Operation *consumerOp) {
  // Check if the loop op yields one result.
  if (loopOp->getNumResults() == 1)
    return success();
  // Check if the consumerOp is the first user of the loopOp and if other users
  // are in the same containing block as that of consumer op's.
  Block *parentBlock = consumerOp->getBlock();
  for (Operation *userOp : loopOp->getUsers()) {
    if (userOp == consumerOp)
      continue;
    if (parentBlock != userOp->getBlock() ||
        !consumerOp->isBeforeInBlock(userOp))
      return failure();
  }
  return success();
}

/// A utility to fetch an untiled consumer of
/// tensor.insert_slice/tensor.parallel_insert_slice.
static FailureOr<OpOperand *> getUntiledConsumerFromSlice(Operation *sliceOp) {
  if (auto insertSlice = dyn_cast<tensor::InsertSliceOp>(sliceOp)) {
    return getUntiledConsumerFromSlice(insertSlice);
  } else if (auto parallelInsertSlice =
                 dyn_cast<tensor::ParallelInsertSliceOp>(sliceOp)) {
    return getUntiledConsumerFromSlice(parallelInsertSlice);
  } else {
    return failure();
  }
}

/// After fusing consumer into scf.for we want to modify the scf.yield operation
/// to reflect the same by returning the values yielded by the tiled consumer.
static void
fixTerminatorSCFYield(RewriterBase &rewriter, scf::ForOp newForOp,
                      TilingResult &tilingResult,
                      ArrayRef<SmallVector<OpFoldResult>> &resultOffsets,
                      ArrayRef<SmallVector<OpFoldResult>> &resultSizes,
                      ArrayRef<BlockArgument> bbArgs) {
  scf::YieldOp oldTerminatorOp =
      cast<scf::YieldOp>(newForOp.getBody()->getTerminator());
  unsigned totalOldResults = oldTerminatorOp->getNumResults();
  unsigned totalTiledResults = tilingResult.tiledOps[0]->getNumResults();
  SmallVector<Value> newYieldOperands;
  newYieldOperands.reserve(totalOldResults + totalTiledResults);
  for (auto oldResult : oldTerminatorOp.getResults()) {
    newYieldOperands.push_back(oldResult);
  }
  rewriter.setInsertionPointAfter(oldTerminatorOp);
  Location loc = newForOp.getLoc();
  for (auto [tiledResult, bbArg, resultOffset, resultSize] :
       llvm::zip_equal(tilingResult.tiledOps[0]->getResults(), bbArgs,
                       resultOffsets, resultSizes)) {
    SmallVector<OpFoldResult> strides(resultOffset.size(),
                                      rewriter.getIndexAttr(1));
    Value newInsertSliceOp = rewriter.create<tensor::InsertSliceOp>(
        loc, tiledResult, bbArg, resultOffset, resultSize, strides);
    newYieldOperands.push_back(newInsertSliceOp);
  }
  rewriter.create<scf::YieldOp>(loc, newYieldOperands);
  rewriter.eraseOp(oldTerminatorOp);
}

/// After fusing consumer into scf.forall we want to yield each of the resulting
/// values by the tiled consumer within scf.forall.in_parallel region.
static void
fixTerminatorSCFInParallel(RewriterBase &rewriter, scf::ForallOp newForallOp,
                           SmallVector<Value> tiledResults,
                           ArrayRef<SmallVector<OpFoldResult>> &resultOffsets,
                           ArrayRef<SmallVector<OpFoldResult>> &resultSizes,
                           ArrayRef<BlockArgument> bbArgs) {
  scf::InParallelOp newTerminatorOp = newForallOp.getTerminator();
  rewriter.setInsertionPointToStart(newTerminatorOp.getBody());
  Location firstYieldOpLoc =
      (*(newTerminatorOp.getYieldingOps().begin())).getLoc();
  for (auto [tiledResult, bbArg, resultOffset, resultSize] :
       llvm::zip_equal(tiledResults, bbArgs, resultOffsets, resultSizes)) {
    SmallVector<OpFoldResult> strides(resultOffset.size(),
                                      rewriter.getIndexAttr(1));
    rewriter.create<tensor::ParallelInsertSliceOp>(
        firstYieldOpLoc, tiledResult, bbArg, resultOffset, resultSize, strides);
  }
}

/// Implementation of fusing consumer of a single slice by computing the
/// slice of the consumer in-place for scf loop.
FailureOr<scf::SCFFuseConsumerOfSliceResult>
mlir::scf::tileAndFuseConsumerOfSlice(RewriterBase &rewriter,
                                      Operation *candidateSliceOp) {
  if (!isa<tensor::InsertSliceOp, tensor::ParallelInsertSliceOp>(
          candidateSliceOp))
    return failure();

  bool isInsertSliceOp = isa<tensor::InsertSliceOp>(candidateSliceOp);

  // 1. Get the consumer of scf.for for the result yielded by
  // tensor.insert_slice/parallel_insert_slice.
  FailureOr<OpOperand *> maybeConsumerOpOperand =
      getUntiledConsumerFromSlice(candidateSliceOp);
  if (failed(maybeConsumerOpOperand)) {
    return rewriter.notifyMatchFailure(candidateSliceOp,
                                       "could not fetch consumer to fuse");
  }
  OpOperand *consumerOpOperand = *maybeConsumerOpOperand;
  Operation *consumerOp = consumerOpOperand->getOwner();
  unsigned operandNumber = consumerOpOperand->getOperandNumber();
  unsigned resultNumber = 0;
  if (auto producerResult = dyn_cast<OpResult>(consumerOpOperand->get())) {
    resultNumber = producerResult.getResultNumber();
  } else {
    return rewriter.notifyMatchFailure(
        consumerOp, "consumer op's operand doesn't seem to be an OpResult");
  }

  Operation *oldLoopOp = nullptr;
  SmallVector<Value> newOuts;
  Block *oldLoopBody = nullptr;
  unsigned initSize = 0;
  unsigned rank = 1;
  if (isInsertSliceOp) {
    auto forOp = candidateSliceOp->getParentOfType<scf::ForOp>();
    oldLoopOp = forOp;
    llvm::append_range(newOuts, forOp.getInits());
    oldLoopBody = forOp.getBody();
    initSize = forOp.getInits().size();
  } else {
    auto forallOp = candidateSliceOp->getParentOfType<scf::ForallOp>();
    oldLoopOp = forallOp;
    llvm::append_range(newOuts, forallOp.getOutputs());
    oldLoopBody = forallOp.getBody();
    initSize = forallOp.getOutputs().size();
    rank = forallOp.getRank();
  }

  if (failed(checkAssumptionForLoop(oldLoopOp, consumerOp))) {
    return rewriter.notifyMatchFailure(
        oldLoopOp, "containing loop op should either yield just one value or "
                   "have the consumer op as its first user");
  }

  OpBuilder::InsertionGuard g(rewriter);

  // 2. Check consumer is not using scf loop's output as init.
  auto dstOp = cast<DestinationStyleOpInterface>(consumerOp);
  SmallVector<Value> dpsInits =
      llvm::map_to_vector(dstOp.getDpsInits(), [](Value v) { return v; });
  if (llvm::is_contained(dpsInits, oldLoopOp->getResult(resultNumber))) {
    return rewriter.notifyMatchFailure(
        consumerOp,
        "consumer op taking the result of scf.for as init is not supported");
  }
  newOuts.append(dpsInits);

  Location loc = oldLoopOp->getLoc();

  // 3. Create new scf loop op.
  rewriter.setInsertionPoint(consumerOp);
  Operation *newLoopOp = nullptr;
  Block *newLoopBody = nullptr;
  if (isInsertSliceOp) {
    auto forOp = cast<scf::ForOp>(oldLoopOp);
    auto newForOp = rewriter.create<scf::ForOp>(loc, forOp.getLowerBound(),
                                                forOp.getUpperBound(),
                                                forOp.getStep(), newOuts);
    newLoopOp = newForOp;
    newLoopBody = newForOp.getBody();
  } else {
    auto forallOp = cast<scf::ForallOp>(oldLoopOp);
    auto newForallOp = rewriter.create<scf::ForallOp>(
        loc, forallOp.getMixedLowerBound(), forallOp.getMixedUpperBound(),
        forallOp.getMixedStep(), newOuts, forallOp.getMapping());
    newLoopOp = newForallOp;
    rewriter.eraseOp(newForallOp.getTerminator());
    newLoopBody = newForallOp.getBody();
  }

  // 4. Move the loop body to the new op.
  unsigned oldNumArguments = oldLoopBody->getNumArguments();
  rewriter.mergeBlocks(oldLoopBody, newLoopBody,
                       newLoopBody->getArguments().take_front(oldNumArguments));

  // 5. Set insertion point before terminator op of the loop and create a new
  // tensor.insert_slice. In the scf.for case this is a clone of the
  // candidateSliceOp whereas in the scf.forall case this is created from the
  // operands of tensor.parallel_insert_slice.
  tensor::InsertSliceOp clonedInsertSliceOp;
  if (auto sliceOp =
          dyn_cast<tensor::ParallelInsertSliceOp>(candidateSliceOp)) {
    auto newForallOp = cast<scf::ForallOp>(newLoopOp);
    rewriter.setInsertionPoint(newForallOp.getTerminator());
    clonedInsertSliceOp = rewriter.create<tensor::InsertSliceOp>(
        loc, sliceOp.getSource(), sliceOp.getDest(), sliceOp.getMixedOffsets(),
        sliceOp.getMixedSizes(), sliceOp.getMixedStrides());
  } else {
    rewriter.setInsertionPoint(candidateSliceOp);
    clonedInsertSliceOp =
        cast<tensor::InsertSliceOp>(rewriter.clone(*candidateSliceOp));
  }

  // 6.a. Clone consumer op.
  auto newForOpBlockArgsForConsumerDest =
      newLoopBody->getArguments().drop_front(oldNumArguments);
  auto clonedConsumerOp = cast<TilingInterface>(cloneOpAndUpdateDestinationArgs(
      rewriter, consumerOp, newForOpBlockArgsForConsumerDest));

  // 6.b. Replace all uses of the loop result with the result of the cloned
  // tensor.insert_slice.
  OpOperand &operandToReplace = clonedConsumerOp->getOpOperand(operandNumber);
  rewriter.modifyOpInPlace(clonedConsumerOp, [&]() {
    operandToReplace.set(clonedInsertSliceOp.getResult());
  });

  // 7 - Perform tiling of the cloned consumer and replace the operand at
  // `operandNumber` with the source of the cloned tensor.insert_slice op.
  auto ossSliceOp =
      cast<OffsetSizeAndStrideOpInterface>(clonedInsertSliceOp.getOperation());
  FailureOr<TilingResult> tileAndFuseResult =
      tensor::replaceInsertSliceWithTiledConsumer(
          rewriter, ossSliceOp, clonedConsumerOp->getOpOperand(operandNumber));
  if (failed(tileAndFuseResult)) {
    return failure();
  }
  rewriter.replaceAllUsesWith(
      tileAndFuseResult->tiledOps[0]->getOperand(operandNumber),
      clonedInsertSliceOp.getSource());

  // 8 - Extract offset/sizes/strides required to create the
  // tensor.insert_slice/parallel_insert_slice for each result of the consumer.
  SmallVector<OpFoldResult> offsets = ossSliceOp.getMixedOffsets();
  SmallVector<OpFoldResult> sizes = ossSliceOp.getMixedSizes();
  SmallVector<OpFoldResult> strides = ossSliceOp.getMixedStrides();

  // 9. Check all insert stride is 1.
  if (llvm::any_of(strides, [](OpFoldResult stride) {
        return !isConstantIntValue(stride, 1);
      })) {
    return rewriter.notifyMatchFailure(
        candidateSliceOp, "containingOp's result yield with stride");
  }

  // 10. Try to get iter domain position from input position.
  SmallVector<OpFoldResult> iterDomainOffsets, iterDomainSizes;
  if (failed(clonedConsumerOp.getIterationDomainTileFromOperandTile(
          rewriter, operandNumber, offsets, sizes, iterDomainOffsets,
          iterDomainSizes))) {
    return rewriter.notifyMatchFailure(
        clonedConsumerOp, "can't get iter domain position from input position");
  }

  // 11. Try to fetch the offset and size for all results of the cloned
  // consumer. This would then be used to form the corresponding
  // tensor.insert_slice/parallel_insert_slice later.
  unsigned totalNumResultsOfConsumer = clonedConsumerOp->getNumResults();
  SmallVector<SmallVector<OpFoldResult>> resultOffsets(
      totalNumResultsOfConsumer);
  SmallVector<SmallVector<OpFoldResult>> resultSizes(totalNumResultsOfConsumer);
  for (auto [idx, v] : llvm::enumerate(clonedConsumerOp->getResults())) {
    if (failed(clonedConsumerOp.getResultTilePosition(
            rewriter, idx, iterDomainOffsets, iterDomainSizes,
            resultOffsets[idx], resultSizes[idx]))) {
      return rewriter.notifyMatchFailure(
          clonedConsumerOp,
          "can't get result domain position from iter domain position");
    }
  }

  auto arrayRefOffsets = ArrayRef<SmallVector<OpFoldResult>>(resultOffsets);
  auto arrayRefSizes = ArrayRef<SmallVector<OpFoldResult>>(resultSizes);
  if (isInsertSliceOp) {
    auto newForOp = cast<scf::ForOp>(newLoopOp);
    fixTerminatorSCFYield(
        rewriter, newForOp, *tileAndFuseResult, arrayRefOffsets, arrayRefSizes,
        newForOp.getBody()->getArguments().drop_front(1 + initSize));
  } else {
    auto newForallOp = cast<scf::ForallOp>(newLoopOp);
    fixTerminatorSCFInParallel(
        rewriter, newForallOp, tileAndFuseResult->tiledOps[0]->getResults(),
        arrayRefOffsets, arrayRefSizes,
        newForallOp.getBody()->getArguments().drop_front(rank + initSize));
  }

  // 12. Replace the result of scf loop and consumer op with new loop's results.
  for (auto &&[oldResult, newResult] :
       llvm::zip_first(oldLoopOp->getResults(), newLoopOp->getResults())) {
    rewriter.replaceAllUsesWith(oldResult, newResult);
  }

  for (auto &&[oldResult, newResult] :
       llvm::zip(consumerOp->getResults(),
                 newLoopOp->getResults().drop_front(initSize))) {
    rewriter.replaceAllUsesWith(oldResult, newResult);
  }

  // 13. Need to erase the old scf loop and the cloned consumer op.
  rewriter.eraseOp(oldLoopOp);
  rewriter.eraseOp(clonedConsumerOp);

  return scf::SCFFuseConsumerOfSliceResult{
      consumerOpOperand,
      &(tileAndFuseResult->tiledOps[0]->getOpOperand(operandNumber)),
      tileAndFuseResult->tiledOps};
}

//===----------------------------------------------------------------------===//
// lowerToLoopsUsingSCFForOp implementation.
//===----------------------------------------------------------------------===//

FailureOr<SmallVector<scf::ForOp>>
mlir::scf::lowerToLoopsUsingSCFForOp(RewriterBase &rewriter,
                                     TilingInterface op) {
  // TODO: Handle cases where the op has results if needed.
  if (op->getNumResults() > 0) {
    return rewriter.notifyMatchFailure(
        op, "unable to lower to loops operations with return values");
  }

  SmallVector<Range> domain = op.getIterationDomain(rewriter);
  SmallVector<Value> ivs;
  SmallVector<scf::ForOp> loops;
  Location loc = op.getLoc();
  for (auto loopRange : domain) {
    Value offsetVal =
        getValueOrCreateConstantIndexOp(rewriter, loc, loopRange.offset);
    Value sizeVal =
        getValueOrCreateConstantIndexOp(rewriter, loc, loopRange.size);
    Value strideVal =
        getValueOrCreateConstantIndexOp(rewriter, loc, loopRange.stride);
    auto loop = rewriter.create<scf::ForOp>(op.getLoc(), offsetVal, sizeVal,
                                            strideVal, ValueRange{});
    loops.push_back(loop);
    ivs.push_back(loop.getInductionVar());
    rewriter.setInsertionPoint(loop.getBody()->getTerminator());
  }
  if (failed(op.generateScalarImplementation(rewriter, op.getLoc(), ivs))) {
    return failure();
  }
  return loops;
}
