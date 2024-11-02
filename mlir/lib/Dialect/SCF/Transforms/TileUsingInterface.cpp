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
      if (isConstantIntValue(nt, 0))
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
      if (isConstantIntValue(nt, 0)) {
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
      if (!isConstantIntValue(residualTileSize, 0)) {
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
      if (isConstantIntValue(tileSize, 0)) {
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
    if (isConstantIntValue(tileSize, 0))
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
  SmallVector<OpFoldResult> offsets(loopRanges.size()),
      sizes(loopRanges.size());

  std::optional<ArrayAttr> mappingAttr;
  if (!mappingVector.empty())
    mappingAttr = rewriter.getArrayAttr(mappingVector);

  scf::ForallOp forallOp;
  bool useNumThreads = !numThreads.empty();

  if (useNumThreads) {
    // Prune the zero numthreads.
    SmallVector<OpFoldResult> nonZeroNumThreads;
    for (auto nt : numThreads) {
      if (isConstantIntValue(nt, 0))
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
  if (llvm::all_of(tileSizes, isZeroIndex)) {
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

    // 5b. Early return cloned op if tiling is not happening. We can not return
    // the original op because it could lead to
    // `rewriter.replaceOp(op, op->getResults())` and users would get crash.
    if (llvm::all_of(tileSizes, isZeroIndex)) {
      tiledResults.append(clonedOp->result_begin(), clonedOp->result_end());
      tilingResult =
          TilingResult{/*tiledOps=*/{clonedOp}, clonedOp->getResults(),
                       /*generatedSlices=*/{}};
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
                              tileSizes, numThreads, destinationTensors,
                              innerYieldTiledValuesFn, loops)))
    return op.emitOpError("failed to generate tiling loops");
  assert(succeeded(tilingResult) &&
         "expected tiling result to be computed after loop generation");

  // If loops are empty, the tiled op is used as the replacement for the untiled
  // op.
  if (loops.empty()) {
    return scf::SCFTilingResult{tilingResult->tiledOps, loops,
                                tilingResult->tiledValues,
                                tilingResult->generatedSlices};
  }

  SmallVector<Value> replacements = llvm::map_to_vector(
      loops.front()->getResults(), [](OpResult r) -> Value { return r; });
  return scf::SCFTilingResult{tilingResult->tiledOps, loops, replacements,
                              tilingResult->generatedSlices};
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
                              /*numThreads=*/ArrayRef<OpFoldResult>{},
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
  struct WorklistItem {
    tensor::ExtractSliceOp candidateSlice;
    SCFTileAndFuseOptions::ControlFnResult controlFnResult;
  };
  std::deque<WorklistItem> worklist;
  auto addCandidateSlices = [&worklist, &options,
                             &loops](ArrayRef<Operation *> candidates) {
    for (auto candidate : candidates) {
      auto sliceOp = dyn_cast<tensor::ExtractSliceOp>(candidate);
      if (!sliceOp || sliceOp.use_empty())
        continue;

      auto [fusableProducer, destinationInitArg] =
          getUntiledProducerFromSliceSource(&sliceOp.getSourceMutable(), loops);
      if (!fusableProducer)
        continue;
      std::optional<SCFTileAndFuseOptions::ControlFnResult> controlFnResult =
          options.fusionControlFn(sliceOp, fusableProducer,
                                  destinationInitArg.has_value());
      if (!controlFnResult)
        continue;
      worklist.emplace_back(WorklistItem{sliceOp, controlFnResult.value()});
    }
  };

  addCandidateSlices(tilingResult->generatedSlices);
  OpBuilder::InsertionGuard g(rewriter);
  while (!worklist.empty()) {
    // Traverse the slices in BFS fashion.
    WorklistItem worklistItem = worklist.front();
    worklist.pop_front();

    // The operands of the fused producer might themselved be slices of
    // values produced by operations that implement the `TilingInterface`.
    // Add these operations to the worklist.
    std::optional<scf::SCFFuseProducerOfSliceResult> fusedResult =
        tileAndFuseProducerOfSlice(rewriter, worklistItem.candidateSlice,
                                   loops);
    if (!fusedResult)
      continue;

    if (worklistItem.controlFnResult.yieldProducerReplacement) {
      // Reconstruct and yield all opResult of fusableProducerOp by default. The
      // caller can specific which one to yield by designating optional argument
      // named `yieldResultNumber` of `yieldReplacementForFusedProducer`.
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
      addCandidateSlices(newSlices.value());
      for (auto [index, result] :
           llvm::enumerate(fusableProducerOp->getResults())) {
        origValToResultNumber[result] = loops.front()->getNumResults() -
                                        fusableProducerOp->getNumResults() +
                                        index;
      }
    }
    addCandidateSlices(fusedResult->generatedSlices);
    if (Operation *tiledAndFusedOp =
            fusedResult->tiledAndFusedProducer.getDefiningOp()) {
      fusedProducers.insert(fusedResult->origProducer.getDefiningOp());
      tiledAndFusedOps.insert(tiledAndFusedOp);
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

/// Find the perfectly nested loops outside of given loop(included) sorted from
/// outer to inner.
///
/// E.g.
///
/// ```
///  %0 = scf.for()
///    %1 = scf.for()
///      %2 = scf.for()
///         %3 = ...
///         yield %3
///      yield %2
///    yield %1
/// ```
///
/// This function will return three perfectly nested loops: %0 + %1 + %2, when
/// target inner loop is %2.
static SmallVector<scf::ForOp>
getPerfectlyNestedLoopsOutsideOf(scf::ForOp loop) {
  SmallVector<scf::ForOp> nestLoops = {loop};
  auto outerLoop = dyn_cast<scf::ForOp>(loop->getParentOp());

  // Check if it is the ForOp that yield the result of inner loop.
  auto isForOpYieldResultOfInnerLoop =
      [](scf::ForOp outerLoop) -> LogicalResult {
    Block *body = outerLoop.getBody();
    if (!llvm::hasSingleElement(body->without_terminator()))
      return failure();
    auto yieldOp = cast<scf::YieldOp>(body->getTerminator());
    auto innerForOp = dyn_cast<scf::ForOp>(body->front());
    if (!innerForOp)
      return failure();
    // All of innerForOp results should be yielded.
    return success(innerForOp->getNumResults() == yieldOp->getNumOperands());
  };

  while (outerLoop && succeeded(isForOpYieldResultOfInnerLoop(outerLoop))) {
    nestLoops.push_back(outerLoop);
    outerLoop = dyn_cast<scf::ForOp>(outerLoop->getParentOp());
  }
  // sorted from outer to inner
  return {nestLoops.rbegin(), nestLoops.rend()};
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
  scf::ForOp topLevelForOp = getPerfectlyNestedLoopsOutsideOf(forOp).front();
  Value resultingValue = topLevelForOp->getResult(resultNumber);

  return getConsumerFromUses(resultingValue, topLevelForOp->getBlock());
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

  // There are two possible cases regarding `oldLoopOp` here:
  // 1. single `scf.forall` or `scf.for`.
  // 2. inner-most `scf.for` insider nest `scf.loop` structure, where the
  // top-level loop is the outer-most one of these nested loops.
  LoopLikeOpInterface innerMostLoop =
      candidateSliceOp->getParentOfType<LoopLikeOpInterface>();
  SmallVector<LoopLikeOpInterface> nestedLoops;
  if (isInsertSliceOp) {
    nestedLoops = llvm::map_to_vector(
        getPerfectlyNestedLoopsOutsideOf(
            cast<scf::ForOp>(innerMostLoop.getOperation())),
        [](scf::ForOp forOp) {
          return cast<LoopLikeOpInterface>(forOp.getOperation());
        });
  } else {
    nestedLoops = {innerMostLoop};
  }

  LoopLikeOpInterface outerMostLoop = nestedLoops.front();

  if (failed(checkAssumptionForLoop(outerMostLoop, consumerOp))) {
    return rewriter.notifyMatchFailure(
        outerMostLoop,
        "containing loop op should either yield just one value or "
        "have the consumer op as its first user");
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

  // 3. Move the whole loop structure right before consumer Op, the dominance
  // should be already ensured by `checkAssumptionForLoop`.
  rewriter.moveOpBefore(outerMostLoop, consumerOp);

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
    if (llvm::any_of(strides, [](OpFoldResult stride) {
          return !isConstantIntValue(stride, 1);
        })) {
      return rewriter.notifyMatchFailure(
          candidateSliceOp, "containingOp's result yield with stride");
    }

    // 10. Try to get iter domain position from input position.
    SmallVector<OpFoldResult> iterDomainOffsets, iterDomainSizes;
    if (failed(tiledConsumerOp.getIterationDomainTileFromOperandTile(
            rewriter, operandNumber, offsets, sizes, iterDomainOffsets,
            iterDomainSizes))) {
      return rewriter.notifyMatchFailure(
          tiledConsumerOp,
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
  if (failed(addInitOperandsToLoopNest(rewriter, nestedLoops, newInits,
                                       newYieldValuesFn))) {
    return rewriter.notifyMatchFailure(tiledConsumerOp,
                                       "unable to add new inits to nest loop");
  }

  // 15. Replace the result of scf loop and consumer op with new loop's results.

  for (auto &&[oldResult, newResult] : llvm::zip(
           consumerOp->getResults(),
           nestedLoops.front()->getResults().take_back(newInits.size()))) {
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
