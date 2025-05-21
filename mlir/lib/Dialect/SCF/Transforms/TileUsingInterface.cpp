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

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
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
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/ScopeExit.h"
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
scf::SCFTilingOptions::setNumThreads(ArrayRef<OpFoldResult> nt) {
  assert(!numThreadsComputationFunction && "num tiles already set");
  auto numThreads = llvm::to_vector(nt);
  numThreadsComputationFunction = [numThreads](OpBuilder &b, Operation *op) {
    return numThreads;
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
  // Specifying number of threads is only supported on `scf.forall` op.
  if (options.numThreadsComputationFunction &&
      options.loopType != scf::SCFTilingOptions::LoopType::ForallOp) {
    return rewriter.notifyMatchFailure(
        loc, "number of threads can only by specified when loop type is "
             "set to use `scf.forall`");
  }

  // If specified, check that the interchange vector is a permutation.
  if (!options.interchangeVector.empty()) {
    if (!isPermutationVector(options.interchangeVector)) {
      return rewriter.notifyMatchFailure(
          loc, "invalid interchange vector, not a permutation of the entire "
               "iteration space");
    }
  }
  return success();
}

/// Method to instantiate the tile sizes and/or number of threads specified
/// by the user.
static std::tuple<SmallVector<OpFoldResult>, SmallVector<OpFoldResult>>
getUserTileSizesAndNumThreads(RewriterBase &rewriter, TilingInterface op,
                              ArrayRef<Range> iterationDomain,
                              const scf::SCFTilingOptions &options) {
  OpFoldResult zero = rewriter.getIndexAttr(0);
  SmallVector<OpFoldResult> tileSizes, numThreads;
  size_t numLoops = iterationDomain.size();

  // Check whether the number of tiles to use is specified.
  if (options.numThreadsComputationFunction) {
    numThreads = options.numThreadsComputationFunction(rewriter, op);
    numThreads.resize(numLoops, zero);

    // If the number of tiles is also specified, use that.
    if (options.tileSizeComputationFunction) {
      tileSizes = options.tileSizeComputationFunction(rewriter, op);
      tileSizes.resize(numLoops, zero);
      return {tileSizes, numThreads};
    }

    // Compute the tile sizes from the iteration domain and number
    // of tiles as follows
    // - niters = ceilDiv(ub - lb, step)
    // - tileSize = ceilDiv(niters, numThreads)
    AffineExpr s0, s1, s2;
    bindSymbols(rewriter.getContext(), s0, s1, s2);
    // TODO: The step here is assumed to be 1.
    AffineExpr numItersExpr = (s1 - s0);
    AffineExpr tileSizeExpr = numItersExpr.ceilDiv(s2);
    tileSizes.resize(numLoops, zero);
    for (auto [index, range, nt] :
         llvm::enumerate(iterationDomain, numThreads)) {
      if (isZeroInteger(nt))
        continue;

      tileSizes[index] = affine::makeComposedFoldedAffineApply(
          rewriter, op.getLoc(), tileSizeExpr, {range.offset, range.size, nt});
    }
    tileSizes.resize(numLoops, zero);
    return {tileSizes, numThreads};
  }

  // Enforce the convention that "tiling by zero"
  // skips tiling a particular dimension. This convention is significantly
  // simpler to handle instead of adjusting affine maps to account for missing
  // dimensions.
  assert(options.tileSizeComputationFunction &&
         "expected tile sizes to be specified");
  tileSizes = options.tileSizeComputationFunction(rewriter, op);
  tileSizes.resize(numLoops, zero);

  return {tileSizes, numThreads};
}

/// Checks if any of the tiled loops are not parallel.
static void checkSafeToTileToForall(TilingInterface op,
                                    ArrayRef<OpFoldResult> tileSizes,
                                    ArrayRef<OpFoldResult> numThreads) {
  auto iterators = op.getLoopIteratorTypes();
  assert(iterators.size() == tileSizes.size() &&
         "expected as many tile size values as number of loops");
  assert((numThreads.empty() || (numThreads.size() == iterators.size())) &&
         "when specified, expected number of threads to use for each loop");

  for (auto [index, iterator, tileSize] :
       llvm::enumerate(iterators, tileSizes)) {
    // If num threads is specified, check that it is greater than one only for
    // parallel dimensions.
    if (!numThreads.empty()) {
      if (std::optional<int64_t> constNumThreads =
              getConstantIntValue(numThreads[index])) {
        if (constNumThreads.value() > 1 &&
            iterator != utils::IteratorType::parallel) {
          op.emitWarning() << "tiling is not thread safe at axis #" << index;
        }
      }
      continue;
    }

    if (std::optional<int64_t> constTileSize = getConstantIntValue(tileSize)) {
      if (constTileSize.value() > 0 &&
          iterator != utils::IteratorType::parallel) {
        op.emitWarning() << "tiling is not thread safe at axis #" << index;
      }
    }
  }
}

/// Check if `stride` evenly divides the trip count `size - offset`.
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

/// Returns the bounded tile size given the current `offset`, `loopRange` and
/// `tileSize`, i.e., `min(tileSize, range.end() - offset)`.
static OpFoldResult getBoundedTileSize(OpBuilder &b, Location loc,
                                       Range loopRange, OpFoldResult offset,
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
  AffineMap minMap = AffineMap::get(1, 2, {s0 - d0, s1}, b.getContext());
  Value size = getValueOrCreateConstantIndexOp(b, loc, loopRange.size);
  return affine::makeComposedFoldedAffineMin(
      b, loc, minMap, SmallVector<OpFoldResult>{offset, size, tileSize});
}

/// Returns true if the maximum tile offset `tileSize * numThreads-1` is less
/// than `iterationSize`.
static bool canOmitTileOffsetInBoundsCheck(OpFoldResult tileSize,
                                           OpFoldResult numThreads,
                                           OpFoldResult iterationSize) {
  std::optional<int64_t> tileSizeConst = getConstantIntValue(tileSize);
  std::optional<int64_t> numThreadsConst = getConstantIntValue(numThreads);
  std::optional<int64_t> iterSizeConst = getConstantIntValue(iterationSize);
  if (!tileSizeConst || !numThreadsConst || !iterSizeConst)
    return false;
  return *tileSizeConst * (*numThreadsConst - 1) < *iterSizeConst;
}

/// Compute the `OpFoldResult`s that represents the multi-dimensional
/// `offset`s and `size`s of the tile of the iteration space that the
/// innermost loop body of the generated tiled loops corresponds to.
static std::tuple<SmallVector<OpFoldResult>, SmallVector<OpFoldResult>>
getTileOffsetAndSizes(RewriterBase &rewriter, Location loc, ValueRange ivs,
                      ArrayRef<Range> iterationDomain,
                      ArrayRef<OpFoldResult> tileSizes,
                      ArrayRef<OpFoldResult> numThreads) {
  SmallVector<OpFoldResult> offsets, sizes;
  int materializedLoopNum = 0;

  if (!numThreads.empty()) {
    AffineExpr d0, d1, s0, s1;
    AffineExpr offsetExpr, residualTileSizeExpr;
    bindDims(rewriter.getContext(), d0, d1);
    bindSymbols(rewriter.getContext(), s0, s1);
    offsetExpr = d0 + d1 * s0;
    residualTileSizeExpr = s1 - (d0 + d1 * s0);

    for (auto [nt, tileSize, loopRange] :
         llvm::zip_equal(numThreads, tileSizes, iterationDomain)) {

      // Non-tiled cases, set the offset and size to the
      // `loopRange.offset/size`.
      if (isZeroInteger(nt)) {
        offsets.push_back(loopRange.offset);
        sizes.push_back(loopRange.size);
        continue;
      }

      Value iv = ivs[materializedLoopNum++];
      OpFoldResult offset = affine::makeComposedFoldedAffineApply(
          rewriter, loc, offsetExpr,
          ArrayRef<OpFoldResult>{loopRange.offset, iv, tileSize});
      OpFoldResult residualTileSize = affine::makeComposedFoldedAffineApply(
          rewriter, loc, residualTileSizeExpr,
          {loopRange.offset, nt, tileSize, loopRange.size});

      OpFoldResult size = tileSize;
      if (!isZeroInteger(residualTileSize)) {
        OpFoldResult sizeMinusOffsetPerThread =
            affine::makeComposedFoldedAffineApply(rewriter, loc, s0 - d0,
                                                  {offset, loopRange.size});
        size = affine::makeComposedFoldedAffineMin(
            rewriter, loc,
            AffineMap::getMultiDimIdentityMap(2, rewriter.getContext()),
            {sizeMinusOffsetPerThread, tileSize});
      }

      // Consider the case where the original loop was `[0, 100)`.
      // If number of threads are `7`, the tile size would be computed as
      // `ceilDiv(100, 7) = 15`. For the last thread (thread_id = 6)
      // - `offset = 0 + 6 * 15 = 105`
      // - `tileSize = min(15, 100 - 105) = -5`
      // To avoid negative tile sizes, we need to do a further
      // `nonNegativeTileSize = affine.max(0, tileSize)`.
      // This `max` can be avoided if
      //  `offset + tileSize * (numThreads - 1) < (ub - lb)`
      if (!canOmitTileOffsetInBoundsCheck(tileSize, nt, loopRange.size)) {
        AffineMap maxMap =
            AffineMap::getMultiDimIdentityMap(2, rewriter.getContext());
        size = affine::makeComposedFoldedAffineMax(
            rewriter, loc, maxMap, {rewriter.getIndexAttr(0), size});
      }

      offsets.push_back(offset);
      sizes.push_back(size);
    }
    return {offsets, sizes};
  } else {
    for (auto [tileSize, loopRange] :
         llvm::zip_equal(tileSizes, iterationDomain)) {

      // Non-tiled cases, set the offset and size to the
      // `loopRange.offset/size`.
      if (isZeroInteger(tileSize)) {
        offsets.push_back(loopRange.offset);
        sizes.push_back(loopRange.size);
        continue;
      }

      Value iv = ivs[materializedLoopNum++];
      OpFoldResult offset = getAsOpFoldResult(iv);
      offsets.push_back(offset);
      OpFoldResult size =
          getBoundedTileSize(rewriter, loc, loopRange, offset, tileSize);
      sizes.push_back(size);
    }
    return {offsets, sizes};
  }
}

/// Function to return the bounds of the loops to be generated.
static std::tuple<SmallVector<OpFoldResult>, SmallVector<OpFoldResult>,
                  SmallVector<OpFoldResult>>
getLoopBounds(RewriterBase &rewriter, Location loc, ArrayRef<Range> loopRanges,
              ArrayRef<OpFoldResult> tileSizes) {
  SmallVector<OpFoldResult> lbs, ubs, steps;
  for (auto [loopRange, tileSize] : llvm::zip_equal(loopRanges, tileSizes)) {
    // No loop if the tile size is 0.
    if (isZeroInteger(tileSize))
      continue;
    lbs.push_back(loopRange.offset);
    ubs.push_back(loopRange.size);
    steps.push_back(tileSize);
  }
  return {lbs, ubs, steps};
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
    ArrayRef<OpFoldResult> tileSizes, ValueRange destinationTensors,
    YieldTiledValuesFn yieldTiledValuesFn,
    SmallVector<LoopLikeOpInterface> &loops) {
  assert(!loopRanges.empty() && "unexpected empty loop ranges");
  assert(loopRanges.size() == tileSizes.size() &&
         "expected as many tile sizes as loop ranges");
  OpBuilder::InsertionGuard guard(rewriter);

  SmallVector<OpFoldResult> lbs, ubs, steps;
  std::tie(lbs, ubs, steps) =
      getLoopBounds(rewriter, loc, loopRanges, tileSizes);
  SmallVector<Value> lbVals =
      getValueOrCreateConstantIndexOp(rewriter, loc, lbs);
  SmallVector<Value> ubVals =
      getValueOrCreateConstantIndexOp(rewriter, loc, ubs);
  SmallVector<Value> stepVals =
      getValueOrCreateConstantIndexOp(rewriter, loc, steps);

  SmallVector<Value> ivs;
  for (auto [lb, ub, step] : llvm::zip_equal(lbVals, ubVals, stepVals)) {
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
    ArrayRef<OpFoldResult> tileSizes, ArrayRef<OpFoldResult> numThreads,
    ArrayRef<Attribute> mappingVector, ValueRange destinationTensors,
    YieldTiledValuesFn tiledBodyFn, SmallVector<LoopLikeOpInterface> &loops) {
  assert(!loopRanges.empty() && "unexpected empty loop ranges");
  assert(loopRanges.size() == tileSizes.size() &&
         "expected as many tile sizes as loop ranges");
  OpBuilder::InsertionGuard guard(rewriter);

  std::optional<ArrayAttr> mappingAttr;
  if (!mappingVector.empty())
    mappingAttr = rewriter.getArrayAttr(mappingVector);

  scf::ForallOp forallOp;
  bool useNumThreads = !numThreads.empty();

  if (useNumThreads) {
    // Prune the zero numthreads.
    SmallVector<OpFoldResult> nonZeroNumThreads;
    for (auto nt : numThreads) {
      if (isZeroInteger(nt))
        continue;
      nonZeroNumThreads.push_back(nt);
    }
    forallOp = rewriter.create<scf::ForallOp>(loc, nonZeroNumThreads,
                                              destinationTensors, mappingAttr);
  } else {
    SmallVector<OpFoldResult> lbs, ubs, steps;
    std::tie(lbs, ubs, steps) =
        getLoopBounds(rewriter, loc, loopRanges, tileSizes);
    forallOp = rewriter.create<scf::ForallOp>(loc, lbs, ubs, steps,
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
    ArrayRef<OpFoldResult> numThreads, ValueRange destinationTensors,
    YieldTiledValuesFn tiledBodyFn, SmallVector<LoopLikeOpInterface> &loops) {
  // If the tile sizes are all zero, no loops are generated. Just call the
  // callback function to handle untiled case.
  if (llvm::all_of(tileSizes, isZeroInteger)) {
    SmallVector<Value> tiledResults;
    SmallVector<SmallVector<OpFoldResult>> resultOffsets, resultSizes;
    return tiledBodyFn(rewriter, loc, ValueRange{}, destinationTensors,
                       tiledResults, resultOffsets, resultSizes);
  }
  if (options.loopType == scf::SCFTilingOptions::LoopType::ForOp) {
    return generateLoopNestUsingForOp(rewriter, loc, loopRanges, tileSizes,
                                      destinationTensors, tiledBodyFn, loops);
  }
  if (options.loopType == scf::SCFTilingOptions::LoopType::ForallOp) {
    return generateLoopNestUsingForallOp(
        rewriter, loc, loopRanges, tileSizes, numThreads, options.mappingVector,
        destinationTensors, tiledBodyFn, loops);
  }
  return rewriter.notifyMatchFailure(loc, "unhandled loop type");
}

static FailureOr<SmallVector<Value>>
createInitialTensorsForTiling(RewriterBase &rewriter, TilingInterface op,
                              ArrayRef<OpFoldResult> tileSizes,
                              const scf::SCFTilingOptions &options) {
  SmallVector<Value> initTensors;
  Location loc = op->getLoc();
  switch (options.reductionStrategy) {
  case scf::SCFTilingOptions::ReductionTilingStrategy::FullReduction:
    if (failed(tensor::getOrCreateDestinations(rewriter, loc, op, initTensors)))
      return failure();
    return initTensors;
  case scf::SCFTilingOptions::ReductionTilingStrategy::
      PartialReductionOuterReduction: {
    auto redOp = dyn_cast<PartialReductionOpInterface>(op.getOperation());
    if (!redOp) {
      return rewriter.notifyMatchFailure(
          op, "PartialReductionOuterReduction tiling strategy is only supported"
              "for operations implementing PartialReductionOpInterface");
    }
    // Get reduction dimensions.
    // TODO: PartialReductionOpInterface should really query TilingInterface
    // itself and find reduction dimensions.
    SmallVector<int> reductionDims;
    for (auto [idx, iteratorType] :
         llvm::enumerate(op.getLoopIteratorTypes())) {
      if (iteratorType == utils::IteratorType::reduction)
        reductionDims.push_back(idx);
    }
    return redOp.generateInitialTensorForPartialReduction(
        rewriter, loc, tileSizes, reductionDims);
  }
  default:
    return rewriter.notifyMatchFailure(op,
                                       "unhandled reduction tiling strategy");
  }
}

static FailureOr<TilingResult>
getTiledImplementation(RewriterBase &rewriter, TilingInterface op,
                       ValueRange regionIterArg, ArrayRef<OpFoldResult> offsets,
                       ArrayRef<OpFoldResult> sizes,
                       const scf::SCFTilingOptions &options) {
  switch (options.reductionStrategy) {
  case scf::SCFTilingOptions::ReductionTilingStrategy::FullReduction:
    return op.getTiledImplementation(rewriter, offsets, sizes);
  case scf::SCFTilingOptions::ReductionTilingStrategy::
      PartialReductionOuterReduction: {
    auto redOp = dyn_cast<PartialReductionOpInterface>(op.getOperation());
    if (!redOp) {
      return rewriter.notifyMatchFailure(
          op, "PartialReductionOuterReduction tiling strategy is only "
              "supported for operations "
              "implementing PartialReductionOpInterface");
    }
    // Get reduction dimensions.
    // TODO: PartialReductionOpInterface should really query TilingInterface
    // itself and find reduction dimensions.
    SmallVector<int> reductionDims;
    for (auto [idx, iteratorType] :
         llvm::enumerate(op.getLoopIteratorTypes())) {
      if (iteratorType == utils::IteratorType::reduction)
        reductionDims.push_back(idx);
    }
    return redOp.tileToPartialReduction(rewriter, op.getLoc(), regionIterArg,
                                        offsets, sizes, reductionDims);
  }
  default:
    return rewriter.notifyMatchFailure(op,
                                       "unhandled reduction tiling strategy");
  }
}

static LogicalResult
getResultTilePosition(RewriterBase &rewriter, int64_t index, Value tiledResult,
                      TilingInterface op, ArrayRef<OpFoldResult> offsets,
                      ArrayRef<OpFoldResult> sizes,
                      SmallVector<OpFoldResult> &resultOffset,
                      SmallVector<OpFoldResult> &resultSize,
                      const scf::SCFTilingOptions &options) {

  switch (options.reductionStrategy) {
  case scf::SCFTilingOptions::ReductionTilingStrategy::FullReduction:
    return op.getResultTilePosition(rewriter, index, offsets, sizes,
                                    resultOffset, resultSize);
  case scf::SCFTilingOptions::ReductionTilingStrategy::
      PartialReductionOuterReduction: {
    auto redOp = dyn_cast<PartialReductionOpInterface>(op.getOperation());
    if (!redOp) {
      return rewriter.notifyMatchFailure(
          op, "PartialReductionOuterReduction tiling strategy is only supported"
              "for operations implementing PartialReductionOpInterface");
    }
    // Get reduction dimensions.
    // TODO: PartialReductionOpInterface should really query TilingInterface
    // itself and find reduction dimensions.
    SmallVector<int> reductionDims;
    for (auto [idx, iteratorType] :
         llvm::enumerate(op.getLoopIteratorTypes())) {
      if (iteratorType == utils::IteratorType::reduction)
        reductionDims.push_back(idx);
    }
    return redOp.getPartialResultTilePosition(rewriter, index, offsets, sizes,
                                              resultOffset, resultSize,
                                              reductionDims);
  }
  default:
    return rewriter.notifyMatchFailure(op,
                                       "unhandled reduction tiling strategy");
  }
}

static FailureOr<MergeResult>
mergeTilingResults(RewriterBase &rewriter, TilingInterface op,
                   ValueRange partialResults,
                   const scf::SCFTilingOptions &options) {
  switch (options.reductionStrategy) {
  case scf::SCFTilingOptions::ReductionTilingStrategy::FullReduction:
    // No need to merge results for reduction tiling strategy.
    return MergeResult{{}, partialResults};
  case scf::SCFTilingOptions::ReductionTilingStrategy::
      PartialReductionOuterReduction: {
    auto redOp = dyn_cast<PartialReductionOpInterface>(op.getOperation());
    if (!redOp) {
      return rewriter.notifyMatchFailure(
          op, "PartialReductionOuterReduction tiling strategy is only "
              "supported for operations "
              "implementing PartialReductionOpInterface");
    }
    // Get reduction dimensions.
    // TODO: PartialReductionOpInterface should really query TilingInterface
    // itself and find reduction dimensions.
    SmallVector<int> reductionDims;
    for (auto [idx, iteratorType] :
         llvm::enumerate(op.getLoopIteratorTypes())) {
      if (iteratorType == utils::IteratorType::reduction)
        reductionDims.push_back(idx);
    }
    return redOp.mergeReductions(rewriter, op.getLoc(), partialResults,
                                 reductionDims);
  }
  default:
    return rewriter.notifyMatchFailure(op,
                                       "unhandled reduction tiling strategy");
  }
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

/// Method to add new init values to a loop nest. Updates `loops` in-place
/// with new loops that use the `newInitValues`. The outer-loops are updated
/// to yield the new result values of the inner loop. For the innermost loop,
/// the call back `getNewYields` is invoked to get the additional values to
/// yield form the innermost loop.
static LogicalResult addInitOperandsToLoopNest(
    RewriterBase &rewriter, MutableArrayRef<LoopLikeOpInterface> loops,
    ValueRange newInitValues, YieldTiledValuesFn getNewTiledYieldsFn) {
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

  // 2. Materialize the tile sizes and/or number of threads;
  SmallVector<OpFoldResult> tileSizes, numThreads;
  std::tie(tileSizes, numThreads) =
      getUserTileSizesAndNumThreads(rewriter, op, iterationDomain, options);

  // Check if it is safe to tile. This is hold over from previous iterations
  // of tile to for-all. Consider dropping it.
  if (options.loopType == scf::SCFTilingOptions::LoopType::ForallOp) {
    checkSafeToTileToForall(op, tileSizes, numThreads);
  }

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
    if (!numThreads.empty())
      applyPermutationToVector(numThreads, interchangeVector);
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
        rewriter, loc, ivs, iterationDomain, tileSizes, numThreads);

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

    // 5b. Early return cloned op if tiling is not happening. We can not
    // return the original op because it could lead to `rewriter.replaceOp(op,
    // op->getResults())` and users would get crash.
    if (llvm::all_of(tileSizes, isZeroInteger)) {
      tiledResults.append(clonedOp->result_begin(), clonedOp->result_end());
      tilingResult =
          TilingResult{/*tiledOps=*/{clonedOp}, clonedOp->getResults(),
                       /*generatedSlices=*/{}};
      return success();
    }

    // 5c. Tile the cloned operation.
    tilingResult = getTiledImplementation(rewriter, clonedOp, regionIterArgs,
                                          offsets, sizes, options);
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
      if (failed(getResultTilePosition(rewriter, index, tiledValue, op, offsets,
                                       sizes, resultOffset, resultSize,
                                       options))) {
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
  FailureOr<SmallVector<Value>> maybeInits =
      createInitialTensorsForTiling(rewriter, op, tileSizes, options);
  if (failed(maybeInits)) {
    return rewriter.notifyMatchFailure(
        op, "unable to create initial tensors for tiling");
  }
  SmallVector<Value> &initTensors = maybeInits.value();

  // 7. Generate the tiled loops nest using the callback defined above.
  SmallVector<LoopLikeOpInterface> loops;
  if (failed(generateLoopNest(rewriter, op.getLoc(), options, iterationDomain,
                              tileSizes, numThreads, initTensors,
                              innerYieldTiledValuesFn, loops)))
    return op.emitOpError("failed to generate tiling loops");
  assert(succeeded(tilingResult) &&
         "expected tiling result to be computed after loop generation");

  SmallVector<Value> partialResults;
  if (loops.empty()) {
    // If loops are empty, the tiled op is used as the replacement for the
    // untiled op.
    partialResults = tilingResult->tiledValues;
  } else {
    partialResults = llvm::map_to_vector(loops.front()->getResults(),
                                         [](OpResult r) -> Value { return r; });
  }

  FailureOr<MergeResult> mergeResult =
      mergeTilingResults(rewriter, op, partialResults, options);
  if (failed(mergeResult)) {
    return rewriter.notifyMatchFailure(
        op, "Failed to merge partial results from tiling");
  }

  return scf::SCFTilingResult{tilingResult->tiledOps, initTensors, loops,
                              mergeResult.value(),
                              tilingResult->generatedSlices};
}

FailureOr<scf::SCFTilingResult>
mlir::scf::tileReductionUsingScf(RewriterBase &b,
                                 PartialReductionOpInterface op,
                                 ArrayRef<OpFoldResult> tileSizes) {
  SCFTilingOptions options;
  options.setLoopType(SCFTilingOptions::LoopType::ForOp);
  options.setReductionTilingStrategy(SCFTilingOptions::ReductionTilingStrategy::
                                         PartialReductionOuterReduction);
  options.setTileSizes(tileSizes);

  TilingInterface tilingInterfaceOp =
      dyn_cast<TilingInterface>(op.getOperation());
  if (!tilingInterfaceOp) {
    return b.notifyMatchFailure(
        op,
        "Operation implementing PartialReductionOpInterface should implement "
        "TilingInterface");
  }

  return tileUsingSCF(b, tilingInterfaceOp, options);
}

//===----------------------------------------------------------------------===//
// tileConsumerAndFuseProducersUsingSCF implementation.
//===----------------------------------------------------------------------===//

/// Return the untiled producer whose slice is used in a tiled consumer. The
/// method traverses the tile loop nest (`loops`) if needed, and returns the
/// `iter_args` of the outer most that is encountered. Traversing the
/// iter_args indicates that this is a destination operand of the consumer. If
/// there was no loop traversal needed, the second value of the returned tuple
/// is empty.
static std::tuple<OpResult, std::optional<OpOperand *>>
getUntiledProducerFromSliceSource(OpOperand *source,
                                  ArrayRef<LoopLikeOpInterface> loops) {
  std::optional<OpOperand *> destinationIterArg;
  assert(!loops.empty() && "expected non empty loops container");
  auto loopIt = loops.rbegin();
  while (loopIt != loops.rend() && isa<BlockArgument>(source->get())) {
    auto iterArg = cast<BlockArgument>(source->get());
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
  //     Easier to just clone the slice with different source since
  //     replacements and DCE of cloned ops becomes easier
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
  return scf::SCFFuseProducerOfSliceResult{
      fusableProducer, tileAndFuseResult->tiledValues[0],
      tileAndFuseResult->tiledOps, tileAndFuseResult->generatedSlices};
}

/// Reconstruct the fused producer from within the tiled-and-fused code.
FailureOr<SmallVector<Operation *>> mlir::scf::yieldReplacementForFusedProducer(
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

  SmallVector<Operation *> generatedSlices;
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
    if (!llvm::all_of(sliceOp.getMixedStrides(), isOneInteger))
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
      // In theory, it is unnecessary to raise an error here. Actually
      // although it fails to reconstruct the result tensor, it should not
      // broke current fusion anyway. The reason why we must return failure
      // currently is that the callback function `newYieldValuesFn` will be
      // called after new init operand(s) has already been appended. It will
      // take more refactoring to make sure the init operands are added
      // consistently in the future. For more details, please refer to:
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

    // d. create `extract_slice` for `iter_args` for DPS operation if
    // necessary
    if (auto tiledDestStyleOp =
            dyn_cast<DestinationStyleOpInterface>(tiledOwner)) {
      rewriter.setInsertionPoint(tiledDestStyleOp);
      for (const auto &&[index, newRegionArg] :
           llvm::enumerate(newRegionIterArgs)) {
        auto destSlice = rewriter.create<tensor::ExtractSliceOp>(
            loc, newRegionArg, offsetList[index], sizesList[index],
            SmallVector<OpFoldResult>(offsetList[index].size(),
                                      rewriter.getIndexAttr(1)));
        generatedSlices.push_back(destSlice);
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

  if (failed(addInitOperandsToLoopNest(rewriter, loops, initValueList,
                                       newYieldValuesFn))) {
    return failure();
  }
  return generatedSlices;
}

namespace {

//===----------------------------------------------------------------------===//
// SliceTrackingListener
//===----------------------------------------------------------------------===//

/// This class is a listener for tracking the insertion and removal of
/// `tensor.extract_slice` ops in a worklist. This can be used in a greedy
/// fusion algorithm to apply cleanup patterns in between fusion steps.
class SliceTrackingListener : public RewriterBase::Listener {
public:
  explicit SliceTrackingListener(
      std::optional<FrozenRewritePatternSet> patterns);
  SliceTrackingListener() = default;

  /// Adds the given list of operations to the worklist, and if present,
  /// applies the list of `patterns` to the newly added operations. This only
  /// processes the given operations and any newly inserted ones by the
  /// pattern set.
  LogicalResult insertAndApplyPatterns(ArrayRef<Operation *> newOps);

  /// Add to the new operation worklist if it is an extract_slice.
  void notifyOperationInserted(Operation *op,
                               OpBuilder::InsertPoint previous) override;

  /// Shared helper for operation removal from the worklist.
  void removeOp(Operation *op);

  /// Remove the operation from the worklist.
  void notifyOperationErased(Operation *op) override;

  /// Remove the operation from the worklist.
  void notifyOperationReplaced(Operation *op, ValueRange replacement) override;

  /// The worklist for this transformation keeps track of the slices to visit
  /// next for fusion.
  std::deque<tensor::ExtractSliceOp> worklist;

private:
  /// Optional pattern set to apply when adding new operations to the
  /// worklist.
  std::optional<FrozenRewritePatternSet> patterns = std::nullopt;
};

SliceTrackingListener::SliceTrackingListener(
    std::optional<FrozenRewritePatternSet> p) {
  patterns = std::move(p);
}

LogicalResult
SliceTrackingListener::insertAndApplyPatterns(ArrayRef<Operation *> ops) {
  for (Operation *op : ops) {
    if (auto slice = dyn_cast<tensor::ExtractSliceOp>(op))
      worklist.push_back(slice);
  }

  if (!patterns)
    return success();

  return applyOpPatternsGreedily(
      ops, patterns.value(),
      GreedyRewriteConfig().setListener(this).setStrictness(
          GreedyRewriteStrictness::ExistingAndNewOps));
}

void SliceTrackingListener::notifyOperationInserted(
    Operation *op, OpBuilder::InsertPoint previous) {
  auto slice = dyn_cast<tensor::ExtractSliceOp>(op);
  if (!slice)
    return;
  worklist.push_back(slice);
}

// Scan the worklist for the given op and remove it if present. The
// expectation is for the worklist to be small and for removal to be
// relatively rare.
void SliceTrackingListener::removeOp(Operation *op) {
  if (!isa<tensor::ExtractSliceOp>(op))
    return;
  auto iter = worklist.begin();
  while (iter != worklist.end()) {
    if (*iter == op)
      break;
    iter++;
  }
  if (iter == worklist.end())
    return;

  worklist.erase(iter);
}

void SliceTrackingListener::notifyOperationErased(Operation *op) {
  removeOp(op);
}

void SliceTrackingListener::notifyOperationReplaced(Operation *op,
                                                    ValueRange replacement) {
  removeOp(op);
}

//===----------------------------------------------------------------------===//
// ReplacementListener
//===----------------------------------------------------------------------===//

/// Listener that tracks updates replacements for values which can be mutated.
/// This listener runs on top of the existing listener for the rewriter,
/// to make sure external users can still run listeners.
class ReplacementListener : public RewriterBase::ForwardingListener {
public:
  ReplacementListener(DenseMap<Value, Value> &replacements,
                      OpBuilder::Listener *listener)
      : ForwardingListener(listener), replacements(replacements) {}

  void updateReplacementValues(ValueRange origValues,
                               ValueRange replaceValues) {
    // This can probably be written better, but just iterates over the map
    // and the new replacements for now.
    for (auto &[key, val] : replacements) {
      for (auto [orig, replace] : llvm::zip_equal(origValues, replaceValues)) {
        if (val == orig) {
          val = replace;
        }
      }
    }
  }

  void notifyOperationReplaced(Operation *op, Operation *newOp) override {
    ForwardingListener::notifyOperationReplaced(op, newOp);
    updateReplacementValues(op->getResults(), newOp->getResults());
  }

  void notifyOperationReplaced(Operation *op, ValueRange values) override {
    ForwardingListener::notifyOperationReplaced(op, values);
    updateReplacementValues(op->getResults(), values);
  }

private:
  DenseMap<Value, Value> &replacements;
};

} // namespace

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

  FailureOr<scf::SCFTilingResult> tilingResult =
      tileUsingSCF(rewriter, consumer, options.tilingOptions);

  if (failed(tilingResult))
    return rewriter.notifyMatchFailure(consumer, "failed to tile consumer");
  tiledAndFusedOps.insert_range(tilingResult->tiledOps);

  DenseMap<Value, Value> replacements;
  for (auto [origVal, replacement] : llvm::zip_equal(
           consumer->getResults(), tilingResult->mergeResult.replacements)) {
    replacements[origVal] = replacement;
  }

  // If there are no loops generated, fusion is immaterial.
  auto &loops = tilingResult->loops;
  if (loops.empty()) {
    return scf::SCFTileAndFuseResult{fusedProducers, tiledAndFusedOps, loops,
                                     replacements};
  }

  // Since the loop gets potentially replaced during fusion, we need to track
  // the mutation of replacement values. To do this, we attach a listener to
  // update the replacements as they happen.
  OpBuilder::Listener *previousListener = rewriter.getListener();
  auto resetListener =
      llvm::make_scope_exit([&]() { rewriter.setListener(previousListener); });
  ReplacementListener replaceListener(replacements, previousListener);
  rewriter.setListener(&replaceListener);

  // 2. Typically, the operands of the tiled operation are slices of the
  //    operands of the untiled operation. These are expressed in IR using
  //    `tensor.extract_slice` operations with source being the operands of
  //    the untiled operation. Create a worklist of these
  //    `tensor.extract_slice` operations. If the producers of the source of
  //    the `tensor.extract_slice` can be tiled such that the tiled value is
  //    generated in-place, that effectively tiles + fuses the operations.
  struct WorklistItem {
    tensor::ExtractSliceOp candidateSlice;
    SCFTileAndFuseOptions::ControlFnResult controlFnResult;
  };

  SliceTrackingListener sliceTracker =
      SliceTrackingListener(options.cleanupPatterns);

  if (failed(
          sliceTracker.insertAndApplyPatterns(tilingResult->generatedSlices))) {
    return rewriter.notifyMatchFailure(consumer, "cleanup patterns failed");
  }
  OpBuilder::InsertionGuard g(rewriter);
  while (!sliceTracker.worklist.empty()) {
    auto candidateSlice = sliceTracker.worklist.front();
    sliceTracker.worklist.pop_front();

    auto [fusableProducer, destinationInitArg] =
        getUntiledProducerFromSliceSource(&candidateSlice.getSourceMutable(),
                                          loops);
    if (!fusableProducer)
      continue;

    std::optional<SCFTileAndFuseOptions::ControlFnResult> controlFnResult =
        options.fusionControlFn(candidateSlice, fusableProducer,
                                destinationInitArg.has_value());
    if (!controlFnResult)
      continue;

    WorklistItem worklistItem = {candidateSlice, controlFnResult.value()};

    // The operands of the fused producer might themselved be slices of
    // values produced by operations that implement the `TilingInterface`.
    // Add these operations to the worklist.
    std::optional<scf::SCFFuseProducerOfSliceResult> fusedResult =
        tileAndFuseProducerOfSlice(rewriter, worklistItem.candidateSlice,
                                   loops);
    if (!fusedResult)
      continue;

    SmallVector<Operation *> worklistCandidates = fusedResult->generatedSlices;

    if (worklistItem.controlFnResult.yieldProducerReplacement) {
      // Reconstruct and yield all opResult of fusableProducerOp by default.
      // The caller can specific which one to yield by designating optional
      // argument named `yieldResultNumber` of
      // `yieldReplacementForFusedProducer`.
      Operation *fusableProducerOp = fusedResult->origProducer.getOwner();
      FailureOr<SmallVector<Operation *>> newSlices =
          yieldReplacementForFusedProducer(rewriter,
                                           worklistItem.candidateSlice,
                                           fusedResult.value(), loops);
      if (failed(newSlices)) {
        return rewriter.notifyMatchFailure(
            fusableProducerOp, "failed to replacement value for this "
                               "operation from within the tiled loop");
      }
      worklistCandidates.append(newSlices.value());
      for (auto [index, result] :
           llvm::enumerate(fusableProducerOp->getResults())) {
        replacements[result] = loops.front()->getResult(
            loops.front()->getNumResults() -
            fusableProducerOp->getNumResults() + index);
      }
    }
    if (Operation *tiledAndFusedOp =
            fusedResult->tiledAndFusedProducer.getDefiningOp()) {
      fusedProducers.insert(fusedResult->origProducer.getDefiningOp());
      tiledAndFusedOps.insert(tiledAndFusedOp);
    }

    if (failed(sliceTracker.insertAndApplyPatterns(worklistCandidates))) {
      return rewriter.notifyMatchFailure(consumer, "cleanup patterns failed");
    }
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

/// An utility to get the first user of the given loopOp. If any of user stay
/// in different block of loopOp, return failure.
static FailureOr<Operation *> getFirstUserOfLoop(Operation *loopOp) {
  if (!isa<LoopLikeOpInterface>(loopOp))
    return failure();
  Operation *firstUserOfLoop = nullptr;
  for (Operation *userOp : loopOp->getUsers()) {
    // `ParallelInsertSlice` located inside `InParallelOp` has no same parent
    // block with any other types of operation. Thus, just redirecting to its
    // parent `InParallelOp`. E.g.
    //
    // ```
    // %1 = scf.for {
    //   ...
    // }
    // %2 = consumerOp ins(%1, ...)
    // scf.forall.in_parallel {
    //    tensor.parallel_insert_slice %1
    // }
    // ```
    // where `InParallelOp` but not `ParallelInsertSlice` stays in the same
    // same block with `consumerOp`.
    if (isa<tensor::ParallelInsertSliceOp>(userOp))
      userOp = userOp->getParentOfType<scf::InParallelOp>();

    if (loopOp->getBlock() != userOp->getBlock())
      return failure();

    if (!firstUserOfLoop || userOp->isBeforeInBlock(firstUserOfLoop))
      firstUserOfLoop = userOp;
  }
  return firstUserOfLoop;
}

/// This utility currently checks whether the first userOp of loop is NOT
/// before the last defineOp of consumer operand. Because that we need to move
/// the whole loop structure right before the `firstUserOfLoop`. This utility
/// thus helps ensuring that no invalid IR is formed, i.e. no backward slice
/// of consumerOp is dominated by the `firstUserOfLoop`. Saying that:
///
/// ```
/// %0 = scf.for() {
///   ...
/// }
/// ...
/// %1 = firstUserOfLoop(%0)
/// ...
/// %2 = lastDefOfConsumerOperand
/// ...
/// %3 = consumerOp(%2)
/// ```
///
/// If the `firstUserOfLoop` is before `lastDefOfConsumerOperand`, then it
/// would be invalid to move the `loopOp` right before the `firstUserOfLoop`,
/// a.k.a. use-def chain violation:
///
/// ```
/// %0:2 = scf.for() {
///    // use before define error
///    %3 = tiledConsumerOp(%2)
/// }
/// %1 = firstUserOfLoop(%0)
/// ...
/// %2 = lastDefOfConsumerOperand
/// ```
///
/// @param loopOp: loop operation
/// @param consumerOp: consumer operation
/// @param reorderOperations: the flag controls whether to reorder the
/// backward slice w.r.t. the defineOp of `consumerOp` operands.
/// @return: computed backward slice of consumerOp, but excluding those
/// already dominates `firstUserOfLoop`.
static FailureOr<llvm::SetVector<Operation *>>
checkAssumptionForLoop(Operation *loopOp, Operation *consumerOp,
                       bool reorderOperations) {
  FailureOr<Operation *> firstUserOfLoop = getFirstUserOfLoop(loopOp);
  if (failed(firstUserOfLoop))
    return failure();

  BackwardSliceOptions options;
  DominanceInfo dominanceInfo;
  options.inclusive = true;
  options.omitBlockArguments = true;
  bool includeLoopOp = false;
  options.filter = [&](Operation *op) {
    if (op == loopOp) {
      includeLoopOp = true;
      return false;
    }
    // Cut off the slice to not include any operation that already dominates
    // firstUserOfLoop.
    return !dominanceInfo.properlyDominates(op, *firstUserOfLoop);
  };
  llvm::SetVector<Operation *> slice;
  for (auto operand : consumerOp->getOperands()) {
    getBackwardSlice(operand, &slice, options);
  }

  if (!slice.empty()) {
    // If consumerOp has one producer, which is also the user of loopOp.
    // E.g.
    // ```
    //  %0 = %loopOp
    //  %1 = consumerOp1 ins(%0)
    //  %2 = consumerOp2 ins(%0, %1)
    // ```
    // We can not fuse consumerOp2 into loopOp due to UD chain, unless
    // consumerOp1 has already been fused into loopOp before.
    if (includeLoopOp || !reorderOperations)
      return failure();
  }

  return slice;
}

/// Fetches the OpOperand of the first valid user (and use) of the value `val`
/// which implements `TilingInterface` and `DestinationStyleOpInterface`.
/// Returns failure otherwise.
static FailureOr<OpOperand *> getConsumerFromLoopUses(RewriterBase &rewriter,
                                                      Operation *loopOp,
                                                      unsigned resultNumber) {
  if (!isa<LoopLikeOpInterface>(loopOp))
    return failure();
  Value val = loopOp->getResult(resultNumber);
  Block *loopBlock = loopOp->getBlock();
  for (OpOperand &opOperand : val.getUses()) {
    Operation *consumerOp = opOperand.getOwner();
    // Step 1. Check if the user is tilable.
    if (!isa<TilingInterface>(consumerOp) ||
        !isa<DestinationStyleOpInterface>(consumerOp)) {
      // TODO: We have to init result of consumer before scf.for, use
      // DestinationStyleOpInterface to get result shape from init for now.
      // Add support for other op such as op has InferTypeOpInterface.
      continue;
    }
    // Step 2. Check if user stay in the same block.
    if (loopBlock != consumerOp->getBlock())
      continue;
    // Step 3. Check if user has succeeding user. Otherwise, it usually
    // represents already tiled.
    if (consumerOp->use_empty())
      continue;
    // Step 4. Check assumption for loop with `reorderOperations` enabled.
    FailureOr<llvm::SetVector<Operation *>> slice =
        checkAssumptionForLoop(loopOp, consumerOp, true);
    if (failed(slice))
      continue;
    // Step 5. If backward sice is not empty, move them before
    // firstUserOfLoop.
    if (!slice->empty()) {
      mlir::topologicalSort(*slice);
      FailureOr<Operation *> firstUserOfLoop = getFirstUserOfLoop(loopOp);
      assert(succeeded(firstUserOfLoop) && "First user of loop is not found");
      for (auto op : *slice) {
        rewriter.moveOpBefore(op, *firstUserOfLoop);
      }
    }
    return &opOperand;
  }
  return failure();
}

/// Check that the loop is perfectly nested.
/// The loops are expected to be ordered from outer most to inner most.
/// For example:
/// ```
///  %0 = scf.for()
///    %1 = scf.for()
///      %2 = scf.for()
///         %3 = ...
///         yield %3
///      yield %2
///    yield %1
/// ```
/// Here loops should be [%0, %1].
static bool
isPerfectlyNestedForLoops(MutableArrayRef<LoopLikeOpInterface> loops) {
  assert(!loops.empty() && "unexpected empty loop nest");
  if (loops.size() == 1) {
    return isa_and_nonnull<scf::ForOp>(loops.front().getOperation());
  }
  for (auto [outerLoop, innerLoop] :
       llvm::zip_equal(loops.drop_back(), loops.drop_front())) {
    auto outerFor = dyn_cast_or_null<scf::ForOp>(outerLoop.getOperation());
    auto innerFor = dyn_cast_or_null<scf::ForOp>(innerLoop.getOperation());
    if (!outerFor || !innerFor) {
      return false;
    }
    auto outerBBArgs = outerFor.getRegionIterArgs();
    auto innerIterArgs = innerFor.getInitArgs();
    if (outerBBArgs.size() != innerIterArgs.size()) {
      return false;
    }

    for (auto [outerBBArg, innerIterArg] :
         llvm::zip_equal(outerBBArgs, innerIterArgs)) {
      if (!llvm::hasSingleElement(outerBBArg.getUses()) ||
          innerIterArg != outerBBArg) {
        return false;
      }
    }

    ValueRange outerYields =
        cast<scf::YieldOp>(outerFor.getBody()->getTerminator())->getOperands();
    ValueRange innerResults = innerFor.getResults();
    if (outerYields.size() != innerResults.size()) {
      return false;
    }
    for (auto [outerYield, innerResult] :
         llvm::zip_equal(outerYields, innerResults)) {
      if (!llvm::hasSingleElement(innerResult.getUses()) ||
          outerYield != innerResult) {
        return false;
      }
    }
  }
  return true;
}

/// Fetch the untiled consumer of the outermost scf.for's result which is
/// yielded by a tensor.insert_slice from the innermost scf.for. This function
/// makes the following assumptions :
/// 1. tensor.insert_slice has scf.yield as its only user.
/// 2. scf.for's corresponding result has only one use.
/// 3. The `loops` passed in are perfectly nested `scf.for` operations.
static FailureOr<OpOperand *>
getUntiledConsumerFromSlice(RewriterBase &rewriter,
                            tensor::InsertSliceOp candidateSliceOp,
                            MutableArrayRef<LoopLikeOpInterface> loops) {
  assert(!loops.empty() && "unexpected loops to be empty");
  // 1. Expect slice to be part of the body of the inner most loop.
  Operation *containingOp = candidateSliceOp->getParentOp();
  if (containingOp != loops.back()) {
    return rewriter.notifyMatchFailure(
        candidateSliceOp,
        "expected slice to be within body of inner-most loop");
  }

  // 2. Check that the loop is perfectly nested.
  if (!isPerfectlyNestedForLoops(loops)) {
    return rewriter.notifyMatchFailure(
        candidateSliceOp, "expected passed loops to be perfectly nested.");
  }

  if (failed(checkAssumptionForFusingConsumer(candidateSliceOp)))
    return failure();
  Value sliceResult = candidateSliceOp.getResult();

  // 3. Fetch the corresponding output.
  OpOperand &yieldOpOperand = (*sliceResult.getUses().begin());
  unsigned resultNumber = yieldOpOperand.getOperandNumber();

  scf::ForOp topLevelForOp = cast<scf::ForOp>(loops.front().getOperation());

  return getConsumerFromLoopUses(rewriter, topLevelForOp, resultNumber);
}

/// Fetch the first untiled consumer of a scf.forall's result which is yielded
/// by a tensor.parallel_insert_slice.
static FailureOr<OpOperand *>
getUntiledConsumerFromSlice(RewriterBase &rewriter,
                            tensor::ParallelInsertSliceOp candidateSliceOp,
                            MutableArrayRef<LoopLikeOpInterface> loops) {
  assert(!loops.empty() && "unexpected loops to be empty");
  // 1. Check that the surrounding loop is a single scf.forall loop.
  if (loops.size() != 1) {
    return rewriter.notifyMatchFailure(
        candidateSliceOp, "expected single surrounding scf.forall");
  }
  auto forallOp = dyn_cast<scf::ForallOp>(loops.front().getOperation());
  if (!forallOp) {
    return rewriter.notifyMatchFailure(
        candidateSliceOp, "expected single surrounding scf.forall");
  }

  // 2. Fetch the corresponding output
  Value sliceDest = candidateSliceOp.getDest();
  auto iterArg = dyn_cast<BlockArgument>(sliceDest);
  if (!iterArg)
    return failure();
  if (iterArg.getOwner()->getParentOp() != forallOp)
    return failure();

  unsigned resultNumber =
      forallOp.getTiedOpResult(forallOp.getTiedOpOperand(iterArg))
          .getResultNumber();

  return getConsumerFromLoopUses(rewriter, forallOp, resultNumber);
}

/// A utility to fetch an untiled consumer of
/// tensor.insert_slice/tensor.parallel_insert_slice.
static FailureOr<OpOperand *>
getUntiledConsumerFromSlice(RewriterBase &rewriter, Operation *sliceOp,
                            MutableArrayRef<LoopLikeOpInterface> loops) {
  assert(!loops.empty() && "unexpected empty loops");
  if (auto insertSlice = dyn_cast<tensor::InsertSliceOp>(sliceOp)) {
    return getUntiledConsumerFromSlice(rewriter, insertSlice, loops);
  } else if (auto parallelInsertSlice =
                 dyn_cast<tensor::ParallelInsertSliceOp>(sliceOp)) {
    return getUntiledConsumerFromSlice(rewriter, parallelInsertSlice, loops);
  } else {
    return failure();
  }
}

/// Implementation of fusing consumer of a single slice by computing the
/// slice of the consumer in-place for scf loop.
FailureOr<scf::SCFFuseConsumerOfSliceResult>
mlir::scf::tileAndFuseConsumerOfSlice(
    RewriterBase &rewriter, Operation *candidateSliceOp,
    MutableArrayRef<LoopLikeOpInterface> loops) {
  // Return if `loops` is empty, return an error for now. Caller is expected
  // to handle this case.
  if (loops.empty()) {
    return candidateSliceOp->emitOpError(
        "cannot call tile and fuse consumer with an empty loop nest");
  }
  if (!isa<tensor::InsertSliceOp, tensor::ParallelInsertSliceOp>(
          candidateSliceOp))
    return failure();

  // 1. Get the consumer of scf.for for the result yielded by
  // tensor.insert_slice/parallel_insert_slice.
  FailureOr<OpOperand *> maybeConsumerOpOperand =
      getUntiledConsumerFromSlice(rewriter, candidateSliceOp, loops);
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

  LoopLikeOpInterface outerMostLoop = loops.front();
  LoopLikeOpInterface innerMostLoop = loops.back();

  // Check assumption for loop with `reorderOperations` disabled.
  if (failed(checkAssumptionForLoop(outerMostLoop, consumerOp, false))) {
    return rewriter.notifyMatchFailure(
        outerMostLoop, "the first user of loop should not dominate any define "
                       "of consumer operand(s)");
  }

  OpBuilder::InsertionGuard g(rewriter);

  // 2. Check consumer is not using scf loop's output as init.
  auto dstOp = dyn_cast<DestinationStyleOpInterface>(consumerOp);
  if (!dstOp)
    return rewriter.notifyMatchFailure(consumerOp,
                                       "consumer op is not DPS operation");
  SmallVector<Value> dpsInits =
      llvm::map_to_vector(dstOp.getDpsInits(), [](Value v) { return v; });
  if (llvm::is_contained(dpsInits, outerMostLoop->getResult(resultNumber))) {
    return rewriter.notifyMatchFailure(
        consumerOp,
        "consumer op taking the result of scf.for as init is not supported");
  }
  SmallVector<Value> newInits = dpsInits;

  Location loc = outerMostLoop->getLoc();

  // 3. Move the whole loop structure right before firstUserOfLoop, the
  // dominance should be already ensured by `checkAssumptionForLoop`.
  FailureOr<Operation *> firstUserOfLoop = getFirstUserOfLoop(outerMostLoop);
  if (failed(firstUserOfLoop)) {
    return rewriter.notifyMatchFailure(
        outerMostLoop, "could not find the first user of outer most loop");
  }
  rewriter.moveOpBefore(outerMostLoop, *firstUserOfLoop);

  // 4. Set insertion point before terminator op of the loop and create a new
  // tensor.insert_slice. In the scf.for case this is a clone of the
  // candidateSliceOp whereas in the scf.forall case this is created from the
  // operands of tensor.parallel_insert_slice.
  tensor::InsertSliceOp clonedInsertSliceOp;
  if (auto sliceOp =
          dyn_cast<tensor::ParallelInsertSliceOp>(candidateSliceOp)) {
    auto newForallOp = cast<scf::ForallOp>(innerMostLoop.getOperation());
    rewriter.setInsertionPoint(newForallOp.getTerminator());
    clonedInsertSliceOp = rewriter.create<tensor::InsertSliceOp>(
        loc, sliceOp.getSource(), sliceOp.getDest(), sliceOp.getMixedOffsets(),
        sliceOp.getMixedSizes(), sliceOp.getMixedStrides());
  } else {
    rewriter.setInsertionPoint(candidateSliceOp);
    clonedInsertSliceOp =
        cast<tensor::InsertSliceOp>(rewriter.clone(*candidateSliceOp));
  }

  // 5.a. Clone consumer op.
  auto clonedConsumerOp = cast<TilingInterface>(rewriter.clone(*consumerOp));

  // 5.b. Replace all uses of the loop result with the result of the cloned
  // tensor.insert_slice.
  OpOperand &operandToReplace = clonedConsumerOp->getOpOperand(operandNumber);
  rewriter.modifyOpInPlace(clonedConsumerOp, [&]() {
    operandToReplace.set(clonedInsertSliceOp.getResult());
  });

  // 6. Perform tiling of the cloned consumer and replace the operand at
  // `operandNumber` with the source of the cloned tensor.insert_slice op.
  auto ossSliceOp =
      cast<OffsetSizeAndStrideOpInterface>(clonedInsertSliceOp.getOperation());
  FailureOr<TilingResult> tileAndFuseResult =
      tensor::replaceInsertSliceWithTiledConsumer(
          rewriter, ossSliceOp, clonedConsumerOp->getOpOperand(operandNumber));
  if (failed(tileAndFuseResult)) {
    return failure();
  }
  auto tiledConsumerOp = cast<TilingInterface>(tileAndFuseResult->tiledOps[0]);
  rewriter.replaceAllUsesWith(tiledConsumerOp->getOperand(operandNumber),
                              clonedInsertSliceOp.getSource());

  // 7. Reconstruct [nested] loop with new inits.
  YieldTiledValuesFn newYieldValuesFn =
      [&](RewriterBase &innerRewriter, Location loc, ValueRange /*ivs*/,
          ValueRange newRegionIterArgs, SmallVector<Value> &tiledResult,
          SmallVector<SmallVector<OpFoldResult>> &tiledOffset,
          SmallVector<SmallVector<OpFoldResult>> &tiledSizes) -> LogicalResult {
    OpBuilder::InsertionGuard g(innerRewriter);
    // 8. Set inner insertPoint right before tiled consumer op.
    innerRewriter.setInsertionPoint(tiledConsumerOp);

    SmallVector<OpFoldResult> offsets = ossSliceOp.getMixedOffsets();
    SmallVector<OpFoldResult> sizes = ossSliceOp.getMixedSizes();
    SmallVector<OpFoldResult> strides = ossSliceOp.getMixedStrides();

    // 9. Check all insert stride is 1.
    if (!llvm::all_of(strides, isOneInteger)) {
      return rewriter.notifyMatchFailure(
          candidateSliceOp, "containingOp's result yield with stride");
    }

    // 10. Try to get iter domain position from input position. Use
    // clonedConsumerOp instead of tiledConsumerOp, because the iteration
    // domain may require index computation based on the result size. The
    // sizes and offsets should be the same either way, but using
    // tiledConsumerOp could lead to some chained unnecessary extra index
    // computation.
    SmallVector<OpFoldResult> iterDomainOffsets, iterDomainSizes;
    if (failed(clonedConsumerOp.getIterationDomainTileFromOperandTile(
            rewriter, operandNumber, offsets, sizes, iterDomainOffsets,
            iterDomainSizes))) {
      return rewriter.notifyMatchFailure(
          clonedConsumerOp,
          "can't get iter domain position from input position");
    }

    // 11. Try to fetch the offset and size for all results of the cloned
    // consumer. This would then be used to form the corresponding
    // tensor.insert_slice/parallel_insert_slice later.
    unsigned totalNumResultsOfConsumer = tiledConsumerOp->getNumResults();
    SmallVector<SmallVector<OpFoldResult>> resultOffsets(
        totalNumResultsOfConsumer);
    SmallVector<SmallVector<OpFoldResult>> resultSizes(
        totalNumResultsOfConsumer);
    for (auto [idx, v] : llvm::enumerate(tiledConsumerOp->getResults())) {
      if (failed(tiledConsumerOp.getResultTilePosition(
              rewriter, idx, iterDomainOffsets, iterDomainSizes,
              resultOffsets[idx], resultSizes[idx]))) {
        return rewriter.notifyMatchFailure(
            tiledConsumerOp,
            "can't get result domain position from iter domain position");
      }
    }

    // 12. Create `extract_slice` for `iter_args` for DPS operation if
    // necessary.
    if (auto tiledDestStyleOp = dyn_cast<DestinationStyleOpInterface>(
            tiledConsumerOp.getOperation())) {
      rewriter.setInsertionPoint(tiledDestStyleOp);
      for (const auto &&[index, newRegionArg] :
           llvm::enumerate(newRegionIterArgs)) {
        auto destSlice = rewriter.create<tensor::ExtractSliceOp>(
            loc, newRegionArg, resultOffsets[index], resultSizes[index],
            SmallVector<OpFoldResult>(resultOffsets[index].size(),
                                      rewriter.getIndexAttr(1)));
        // Make a copy of index to avoid a capturing structured binding, which
        // is a C++20 extension.
        auto dstNumber = index;
        rewriter.modifyOpInPlace(tiledDestStyleOp, [&]() {
          tiledDestStyleOp.getDpsInitsMutable()[dstNumber].set(destSlice);
        });
      }
    }

    // 13. Prepare tiled offset and sizes for later `insert_slice` creation by
    // caller.
    Block *block = rewriter.getInsertionPoint()->getBlock();
    rewriter.setInsertionPoint(block->getTerminator());
    for (const auto &&[index, result] :
         llvm::enumerate(tiledConsumerOp->getResults())) {
      tiledResult.push_back(result);
      tiledOffset.emplace_back(resultOffsets[index]);
      tiledSizes.emplace_back(resultSizes[index]);
    }
    return success();
  };
  // 14. Add new inits to [nested] loops.
  if (failed(addInitOperandsToLoopNest(rewriter, loops, newInits,
                                       newYieldValuesFn))) {
    return rewriter.notifyMatchFailure(tiledConsumerOp,
                                       "unable to add new inits to nest loop");
  }

  // 15. Replace the result of scf loop and consumer op with new loop's
  // results.

  for (auto &&[oldResult, newResult] :
       llvm::zip(consumerOp->getResults(),
                 loops.front()->getResults().take_back(newInits.size()))) {
    rewriter.replaceAllUsesWith(oldResult, newResult);
  }

  // 16. Need to erase the old scf loop and the cloned consumer op.
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
