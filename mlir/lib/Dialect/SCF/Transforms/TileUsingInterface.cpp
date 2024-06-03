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
#include "mlir/Dialect/Affine/Utils.h"
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
                              tileSizes, numThreads, destinationTensors,
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

/// Traverse and collect all outer loops of given sliceOp, sorted by
/// outer-to-inner. If `untilLoop` found, stop walk through in advance.
static SmallVector<LoopLikeOpInterface> getOuterLoopsOfSliceOp(
    Operation *sliceOp,
    std::optional<LoopLikeOpInterface> untilLoop = std::nullopt) {
  assert(isa<OffsetSizeAndStrideOpInterface>(sliceOp));
  SmallVector<LoopLikeOpInterface> outerLoops;
  auto forOp = sliceOp->getParentOfType<LoopLikeOpInterface>();
  while (forOp) {
    outerLoops.push_back(forOp);
    if (untilLoop.has_value() && *untilLoop == forOp)
      break;
    forOp = forOp->getParentOfType<LoopLikeOpInterface>();
  }
  return {outerLoops.rbegin(), outerLoops.rend()};
}

/// Get the result of top-level loop which yields the target InsertSliceOp. E.g
/// ```
/// %1 = scf.for
///  %2 = scf.for
///   %3 = scf.for
///      ...
///      %4 = insert
///      yield %4
///   %5 = insert %3
///   yield %5
///  yield %2
/// ```
/// @param targetSliceOp: %4 = insert
/// @return first: resultValue: %1
///         second: Collected insertSliceOp List during walk including
///                   targetSliceOp: %4 = insert and %5 = insert %3
static FailureOr<std::pair<Value, SmallVector<OffsetSizeAndStrideOpInterface>>>
getResultOfTopLevelLoopYieldInsertSliceOp(Operation *targetSliceOp,
                                          int curDepth = 0, int maxDepth = 5) {
  assert(isa<OffsetSizeAndStrideOpInterface>(targetSliceOp));
  // Control recursive time in avoid of stack overflow
  if (curDepth > maxDepth)
    return failure();

  SmallVector<OffsetSizeAndStrideOpInterface> candidateSliceOpList;
  candidateSliceOpList.push_back(
      cast<OffsetSizeAndStrideOpInterface>(targetSliceOp));
  Value resultOfLoop;
  if (auto sliceOp = dyn_cast<tensor::ParallelInsertSliceOp>(targetSliceOp)) {
    Value destValue = sliceOp.getDest();
    auto iterArg = cast<BlockArgument>(destValue);
    auto forallOp = dyn_cast<scf::ForallOp>(iterArg.getOwner()->getParentOp());
    if (!forallOp)
      return failure();
    resultOfLoop = forallOp.getTiedOpResult(forallOp.getTiedOpOperand(iterArg));
  } else if (auto sliceOp = dyn_cast<tensor::InsertSliceOp>(targetSliceOp)) {
    Value resultValue = sliceOp.getResult();
    for (auto &useOperand : resultValue.getUses()) {
      if (auto yieldOp = dyn_cast<scf::YieldOp>(useOperand.getOwner())) {
        if (llvm::detail::isPresent(resultOfLoop))
          return failure();
        auto forOp = dyn_cast<LoopLikeOpInterface>(yieldOp->getParentOp());
        if (!forOp)
          return failure();
        resultOfLoop = forOp->getResult(useOperand.getOperandNumber());
      }
    }
  }

  if (!llvm::detail::isPresent(resultOfLoop))
    return failure();

  while (true) {
    bool walkThroughOuterLoop = false;
    for (OpOperand &useOperand : resultOfLoop.getUses()) {
      if (auto sliceOp =
              dyn_cast<OffsetSizeAndStrideOpInterface>(useOperand.getOwner())) {
        FailureOr<std::pair<Value, SmallVector<OffsetSizeAndStrideOpInterface>>>
            resultAndSliceOpsPair = getResultOfTopLevelLoopYieldInsertSliceOp(
                sliceOp, curDepth + 1);
        if (failed(resultAndSliceOpsPair))
          return failure();
        candidateSliceOpList.append((*resultAndSliceOpsPair).second.begin(),
                                    (*resultAndSliceOpsPair).second.end());
        return std::make_pair((*resultAndSliceOpsPair).first,
                              candidateSliceOpList);
      } else if (auto yieldOp = dyn_cast<scf::YieldOp>(useOperand.getOwner())) {
        // walk through outer loop
        auto forOp = dyn_cast<LoopLikeOpInterface>(yieldOp->getParentOp());
        if (!forOp)
          return failure();
        resultOfLoop = forOp->getResult(useOperand.getOperandNumber());
        walkThroughOuterLoop = true;
        break;
      }
    }
    if (!walkThroughOuterLoop)
      break;
  }
  return std::make_pair(resultOfLoop, candidateSliceOpList);
}

/// After fusing consumer into scf.for we want to modify the scf.yield operation
/// to reflect the same by returning the values yielded by the tiled consumer.
static void
fixTerminatorSCFYield(RewriterBase &rewriter, scf::ForOp newForOp,
                      ResultRange tilingResult,
                      ArrayRef<SmallVector<OpFoldResult>> resultOffsets,
                      ArrayRef<SmallVector<OpFoldResult>> resultSizes,
                      ArrayRef<BlockArgument> bbArgs) {
  scf::YieldOp oldTerminatorOp =
      cast<scf::YieldOp>(newForOp.getBody()->getTerminator());
  unsigned totalOldResults = oldTerminatorOp->getNumResults();
  unsigned totalTiledResults = tilingResult.size();
  SmallVector<Value> newYieldOperands;
  newYieldOperands.reserve(totalOldResults + totalTiledResults);
  for (auto oldResult : oldTerminatorOp.getResults()) {
    newYieldOperands.push_back(oldResult);
  }
  rewriter.setInsertionPointAfter(oldTerminatorOp);
  Location loc = newForOp.getLoc();
  for (auto [tiledResult, bbArg, resultOffset, resultSize] :
       llvm::zip_equal(tilingResult, bbArgs, resultOffsets, resultSizes)) {
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
                           ResultRange tilingResult,
                           ArrayRef<SmallVector<OpFoldResult>> resultOffsets,
                           ArrayRef<SmallVector<OpFoldResult>> resultSizes,
                           ArrayRef<BlockArgument> bbArgs) {
  scf::InParallelOp newTerminatorOp = newForallOp.getTerminator();
  rewriter.setInsertionPointToStart(newTerminatorOp.getBody());
  Location firstYieldOpLoc =
      (*(newTerminatorOp.getYieldingOps().begin())).getLoc();
  for (auto [tiledResult, bbArg, resultOffset, resultSize] :
       llvm::zip_equal(tilingResult, bbArgs, resultOffsets, resultSizes)) {
    SmallVector<OpFoldResult> strides(resultOffset.size(),
                                      rewriter.getIndexAttr(1));
    rewriter.create<tensor::ParallelInsertSliceOp>(
        firstYieldOpLoc, tiledResult, bbArg, resultOffset, resultSize, strides);
  }
}

/// If the top level loop of nested loop structure is scf.forall, need to create
/// additional tensor.extract_slice for its new appended `shared_outs` in order
/// to pass correct local memory for inner loops. E.g.
///
/// scf.forall shared_outs(%o1=..., %o2=...) {
///     %local_o1 = extract_slice %o1
///     // fix new appended `shared_out` %o2
///     %local_o2 = extract_slice %o2
///     scf.for init_args(%init1=%local_o1, %init2=%local_o2) {
///        ...
///     }
///     ...
/// }
static SmallVector<tensor::ExtractSliceOp> fixLoopInitFromSharedOutSCFForall(
    RewriterBase &rewriter, Operation *loop, ArrayRef<Value> newSharedOuts,
    ArrayRef<SmallVector<OpFoldResult>> resultOffsets,
    ArrayRef<SmallVector<OpFoldResult>> resultSizes) {
  rewriter.setInsertionPoint(loop);
  Location loc = loop->getLoc();
  // create new ExtractOps for newInits from scf.forall
  SmallVector<tensor::ExtractSliceOp> newExtractOps;
  newExtractOps.reserve(resultOffsets.size());
  for (auto [bbArg, offset, sizes] :
       llvm::zip_equal(newSharedOuts, resultOffsets, resultSizes)) {
    SmallVector<OpFoldResult> strides(offset.size(), rewriter.getIndexAttr(1));
    auto newExtractOp = rewriter.create<tensor::ExtractSliceOp>(
        loc, bbArg, offset, sizes, strides);
    newExtractOps.push_back(newExtractOp);
  }
  return newExtractOps;
}

/// If outerMost loop of nested loop structure is `scf.forall`, need to deal
/// with DpsInit of tiled consumer
static void
fixDpsInitsOfTiledConsumer(RewriterBase &rewriter, Operation *tiledConsumer,
                           ArrayRef<BlockArgument> bbArgs,
                           ArrayRef<SmallVector<OpFoldResult>> resultOffsets,
                           ArrayRef<SmallVector<OpFoldResult>> resultSizes) {
  rewriter.setInsertionPoint(tiledConsumer);
  Location loc = tiledConsumer->getLoc();
  for (auto &&[bbArg, offset, sizes, dpsInit] :
       llvm::zip_equal(bbArgs, resultOffsets, resultSizes,
                       cast<DestinationStyleOpInterface>(tiledConsumer)
                           .getDpsInitsMutable())) {
    SmallVector<OpFoldResult> strides(offset.size(), rewriter.getIndexAttr(1));
    auto newExtractOp = rewriter.create<tensor::ExtractSliceOp>(
        loc, bbArg, offset, sizes, strides);
    dpsInit.set(newExtractOp.getResult());
  }
}

/// Compute all results tile by given SliceOp along operand
static LogicalResult computeAllResultTileForOpGivenOperandSliceOp(
    RewriterBase &rewriter, TilingInterface tilableOp, unsigned operandNumber,
    OffsetSizeAndStrideOpInterface ossSliceOp,
    SmallVector<SmallVector<OpFoldResult>> &allResultOffsets,
    SmallVector<SmallVector<OpFoldResult>> &allResultSizes) {
  // 1. Check all stride all 1
  if (llvm::any_of(ossSliceOp.getMixedStrides(), [](OpFoldResult stride) {
        return !isConstantIntValue(stride, 1);
      })) {
    return rewriter.notifyMatchFailure(ossSliceOp, "ossSliceOp has stride");
  }
  // 2. Compute iteration domain tile from the input position
  SmallVector<OpFoldResult> iterDomainOffsets, iterDomainSizes;
  if (failed(tilableOp.getIterationDomainTileFromOperandTile(
          rewriter, operandNumber, ossSliceOp.getMixedOffsets(),
          ossSliceOp.getMixedSizes(), iterDomainOffsets, iterDomainSizes))) {
    return rewriter.notifyMatchFailure(
        tilableOp, "can't get iter domain position from input position");
  }
  unsigned totalNumResultsOfConsumer = tilableOp->getNumResults();
  SmallVector<SmallVector<OpFoldResult>> resultOffsets(
      totalNumResultsOfConsumer);
  SmallVector<SmallVector<OpFoldResult>> resultSizes(totalNumResultsOfConsumer);
  // 3. Compute result Tile by resultNumber
  for (auto [idx, v] : llvm::enumerate(tilableOp->getResults())) {
    if (failed(tilableOp.getResultTilePosition(
            rewriter, idx, iterDomainOffsets, iterDomainSizes,
            resultOffsets[idx], resultSizes[idx]))) {
      return rewriter.notifyMatchFailure(
          tilableOp,
          "can't get result domain position from iter domain position");
    }
  }
  allResultOffsets = resultOffsets;
  allResultSizes = resultSizes;
  return success();
}

/// Considering multi-level tensor.*SliceOp maybe based on different
/// coordination, this utility computes the real OFFSET coordinated on ROOT
/// SliceOp. E.g
///             %0 = insert_slice %1 into %2[OFFSET1] [SIZE1]
///         %3 = insert_slice %4 into %5[OFFSET2] [SIZE2]
///
/// where the coordination can be illustrated as follow:
///
///  %3 ----------------------------------
///  |         |         |
///  | OFFSET2 | OFFSET1 |
///  | ------ %0         |
///  |                   |
///  |                   |
///  |------------------ %1 ------ |
///  |                   |  SIZE1  |
///  |                   |         |
///  |                   |         |
///  |                   | ------- |
///  |
///
/// The real OFFSET of %1 coordinated on %3 is actually `OFFSET1` + `OFFSET2`
static FailureOr<SmallVector<OpFoldResult>>
computeRealOffsetsCoordinatedRootSliceOp(
    RewriterBase &rewriter, Location loc,
    OffsetSizeAndStrideOpInterface candidateSliceOp,
    MutableArrayRef<OffsetSizeAndStrideOpInterface> candidateSliceOpList) {
  if (llvm::any_of(candidateSliceOp.getMixedStrides(), [](OpFoldResult stride) {
        return !isConstantIntValue(stride, 1);
      })) {
    return rewriter.notifyMatchFailure(candidateSliceOp,
                                       "candidateSliceOp has stride");
  }
  SmallVector<OpFoldResult> realOffsets = candidateSliceOp.getMixedOffsets();
  // Real offsets equals to accumulative offsets of outer candidates
  for (auto iter = candidateSliceOpList.rbegin(); *iter != candidateSliceOp;
       iter++) {
    // assert each outer candidate slice has no stride
    if (llvm::any_of(iter->getMixedStrides(), [](OpFoldResult stride) {
          return !isConstantIntValue(stride, 1);
        })) {
      return failure();
    }
    for (auto &&[ofr1, ofr2] :
         llvm::zip_equal(realOffsets, iter->getMixedOffsets())) {
      using AVE = affine::AffineValueExpr;
      affine::AffineBuilder ab(rewriter, loc);
      AffineExpr dim0, dim1, sym;
      bindDims(rewriter.getContext(), dim0, dim1);
      bindSymbols(rewriter.getContext(), sym);
      auto aveOffset1 = AVE(dim0).bind(ofr1), aveOffset2 = AVE(dim1).bind(ofr2);
      ofr1 = ab.add(aveOffset1, aveOffset2);
    }
  }
  return realOffsets;
}

/// Get the first tilable user of given Value and check its domination at the
/// same time
static FailureOr<OpOperand *>
getTilableConsumerOperandFirstUseVal(Value val, Operation *loopOp) {
  for (auto &useOfval : val.getUses()) {
    Operation *consumerOp = useOfval.getOwner();
    // 1. Check whether consumerOp is tilable
    if (!isa<TilingInterface>(consumerOp) ||
        !isa<DestinationStyleOpInterface>(consumerOp))
      continue;
    // 2. Check stay in same block with loopOp
    if (loopOp->getBlock() != consumerOp->getBlock())
      continue;
    // 3. Check no other user before it
    if (failed(checkAssumptionForLoop(loopOp, consumerOp))) {
      continue;
    }
    return &useOfval;
  }
  return failure();
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

  // 1.a. Get the real consumer of candidate
  // tensor.insert_slice/parallel_insert_slice by walking through
  // scf.for/scf.forall and collect all [Parallel]insertSliceOp(s) along the
  // way.
  FailureOr<std::pair<Value, SmallVector<OffsetSizeAndStrideOpInterface>>>
      resultAndSliceOpsPair =
          getResultOfTopLevelLoopYieldInsertSliceOp(candidateSliceOp);
  if (failed(resultAndSliceOpsPair)) {
    return rewriter.notifyMatchFailure(candidateSliceOp,
                                       "could not fetch consumer to fuse");
  }

  // 1.b. Get all outer loops of candidateSliceOp.
  SmallVector<LoopLikeOpInterface> outerLoops = getOuterLoopsOfSliceOp(
      candidateSliceOp, dyn_cast<OpResult>((*resultAndSliceOpsPair).first)
                            .getDefiningOp<LoopLikeOpInterface>());
  LoopLikeOpInterface outerMostLoop = outerLoops.front();

  // 2. Get first tilable consumer op
  FailureOr<OpOperand *> maybeConsumerOpOperand =
      getTilableConsumerOperandFirstUseVal((*resultAndSliceOpsPair).first,
                                           outerMostLoop);
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

  // 3. Check consumer is not using outerMostLoop's output as init.
  auto dstOp = cast<DestinationStyleOpInterface>(consumerOp);
  SmallVector<Value> dpsInits =
      llvm::map_to_vector(dstOp.getDpsInits(), [](Value v) { return v; });
  if (llvm::is_contained(dpsInits, outerMostLoop->getResult(resultNumber))) {
    return rewriter.notifyMatchFailure(
        consumerOp,
        "consumer op taking the result of scf.for as init is not supported");
  }

  // 4.a. Reconstruct nested loop from outer to inner.
  SmallVector<OffsetSizeAndStrideOpInterface> candidateSliceOpList =
      (*resultAndSliceOpsPair).second;
  SmallVector<LoopLikeOpInterface> newOuterLoops;
  SmallVector<SmallVector<OpFoldResult>> allResultOffsets, allResultSizes;
  SmallVector<tensor::ExtractSliceOp> newExtractOps;

  Block *oldLoopBody = nullptr;
  Block *newLoopBody = nullptr;
  SmallVector<Value> newInitAppend = dpsInits, newOuts;

  OpBuilder::InsertionGuard g(rewriter);
  // 4.b. Set insertPoint right before consumerOp
  rewriter.setInsertionPoint(consumerOp);

  for (auto [index, loop] :
       llvm::enumerate(MutableArrayRef(outerLoops).drop_back())) {
    if (index > 0) {
      rewriter.setInsertionPoint(loop);
      // 4.c. Create `extractSliceOp` for newInits if they comes from sharedOut
      // of previous `scf.forall` loop.
      if (auto prevOuterLoop =
              dyn_cast<scf::ForallOp>(newOuterLoops.back().getOperation())) {
        if (index != 1) {
          return rewriter.notifyMatchFailure(
              prevOuterLoop, "Currently only outerMostLoop assumed forallOp");
        }
        OffsetSizeAndStrideOpInterface outerMostCandidate =
            candidateSliceOpList.back();
        if (failed(computeAllResultTileForOpGivenOperandSliceOp(
                rewriter, cast<TilingInterface>(consumerOp), operandNumber,
                outerMostCandidate, allResultOffsets, allResultSizes))) {
          return failure();
        }
        newExtractOps = fixLoopInitFromSharedOutSCFForall(
            rewriter, loop, newInitAppend, allResultOffsets, allResultSizes);
        newInitAppend = llvm::map_to_vector(
            newExtractOps,
            [](tensor::ExtractSliceOp op) -> Value { return op.getResult(); });
      }
    }
    LoopLikeOpInterface newLoopOp;
    // 4.d. Create a new loop with the new init values for this loop.
    if (auto forOp = dyn_cast<scf::ForOp>(loop.getOperation())) {
      newOuts = llvm::to_vector(forOp.getInits());
      newOuts.append(newInitAppend.begin(), newInitAppend.end());
      auto newLoop = rewriter.create<scf::ForOp>(
          forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
          forOp.getStep(), newOuts);
      newLoopOp = newLoop;
      oldLoopBody = forOp.getBody();
      newLoopBody = newLoop.getBody();
    } else if (auto forallOp = dyn_cast<scf::ForallOp>(loop.getOperation())) {
      newOuts = llvm::to_vector(forallOp.getOutputs());
      newOuts.append(newInitAppend.begin(), newInitAppend.end());
      auto newLoop = rewriter.create<scf::ForallOp>(
          forallOp.getLoc(), forallOp.getMixedLowerBound(),
          forallOp.getMixedUpperBound(), forallOp.getMixedStep(), newOuts,
          forallOp.getMapping());
      rewriter.eraseOp(newLoop.getTerminator());
      newLoopOp = newLoop;
      oldLoopBody = forallOp.getBody();
      newLoopBody = newLoop.getBody();
    }
    newInitAppend = llvm::map_to_vector(
        newLoopBody->getArguments().take_back(newInitAppend.size()),
        [](BlockArgument bArg) -> Value { return bArg; });
    rewriter.mergeBlocks(
        oldLoopBody, newLoopBody,
        newLoopBody->getArguments().take_front(oldLoopBody->getNumArguments()));
    rewriter.replaceOp(
        loop, newLoopOp->getResults().take_front(loop->getNumResults()));
    newOuterLoops.push_back(newLoopOp);
  }

  // 5.a. Reconstruct inner-most loop.
  LoopLikeOpInterface oldInnerMostLoop = outerLoops.back(), newInnerMostLoop;
  Location loc = oldInnerMostLoop->getLoc();
  if (outerLoops.size() > 1)
    rewriter.setInsertionPoint(oldInnerMostLoop);

  if (isInsertSliceOp) {
    auto forOp = cast<scf::ForOp>(oldInnerMostLoop.getOperation());
    newOuts = llvm::to_vector(forOp.getInits());
    newOuts.append(newInitAppend.begin(), newInitAppend.end());
    oldLoopBody = forOp.getBody();
    auto newForOp = rewriter.create<scf::ForOp>(loc, forOp.getLowerBound(),
                                                forOp.getUpperBound(),
                                                forOp.getStep(), newOuts);
    newInnerMostLoop = newForOp;
    newLoopBody = newForOp.getBody();
  } else {
    auto forallOp = cast<scf::ForallOp>(oldInnerMostLoop.getOperation());
    newOuts = llvm::to_vector(forallOp.getOutputs());
    newOuts.append(newInitAppend.begin(), newInitAppend.end());
    oldLoopBody = forallOp.getBody();
    auto newForallOp = rewriter.create<scf::ForallOp>(
        loc, forallOp.getMixedLowerBound(), forallOp.getMixedUpperBound(),
        forallOp.getMixedStep(), newOuts, forallOp.getMapping());
    newInnerMostLoop = newForallOp;
    rewriter.eraseOp(newForallOp.getTerminator());
    newLoopBody = newForallOp.getBody();
  }

  // 5.b. Move the loop body to the new op.
  unsigned oldNumArguments = oldLoopBody->getNumArguments();
  rewriter.mergeBlocks(oldLoopBody, newLoopBody,
                       newLoopBody->getArguments().take_front(oldNumArguments));
  // 5.c. Replace the result of old oldInnerMostLoop with newInnerMostLoop's
  // results.
  rewriter.replaceOp(oldInnerMostLoop,
                     newInnerMostLoop->getResults().take_front(
                         oldInnerMostLoop->getNumResults()));

  // 6. Set insertion point before terminator op of the loop and create a new
  // tensor.insert_slice. In the scf.for case this is a clone of the
  // candidateSliceOp whereas in the scf.forall case this is created from the
  // operands of tensor.parallel_insert_slice.
  tensor::InsertSliceOp clonedInsertSliceOp;
  if (auto sliceOp =
          dyn_cast<tensor::ParallelInsertSliceOp>(candidateSliceOp)) {
    auto newForallOp = cast<scf::ForallOp>(newInnerMostLoop);
    rewriter.setInsertionPoint(newForallOp.getTerminator());
  } else {
    rewriter.setInsertionPoint(candidateSliceOp);
  }

  // 7.a. Due to current assumption of `getTiledImplementation` that all
  // `Operands` are untiled with original tensor size, create dummy
  // `insertSliceOp` to align with that requirement.
  auto ossSliceOp = cast<OffsetSizeAndStrideOpInterface>(candidateSliceOp);
  FailureOr<SmallVector<OpFoldResult>> realOffsets =
      computeRealOffsetsCoordinatedRootSliceOp(rewriter, loc, ossSliceOp,
                                               candidateSliceOpList);
  if (failed(realOffsets))
    return failure();
  clonedInsertSliceOp = rewriter.create<tensor::InsertSliceOp>(
      loc, candidateSliceOp->getOperand(0),
      candidateSliceOpList.back()->getOperand(1), *realOffsets,
      ossSliceOp.getMixedSizes(), ossSliceOp.getMixedStrides());

  SmallVector<Value> newDpsInitsForConsumerDest = llvm::map_to_vector(
      newLoopBody->getArguments().drop_front(oldNumArguments),
      [](BlockArgument bArg) -> Value { return bArg; });

  // 7.b. If the outerMostLoop is scf.forall, then the `newExtractOps` has been
  // additionally created at `step 4.e` for `dpsInits`. As the counterpart, the
  // `insertSliceOp` is also needed for the same purpose with `step 7.a`.
  if (!newExtractOps.empty()) {
    for (auto &&[extractOp, newDpsInit] :
         llvm::zip_equal(newExtractOps, newDpsInitsForConsumerDest)) {
      auto alignDpsInsertSliceOp = rewriter.create<tensor::InsertSliceOp>(
          loc, newDpsInit, extractOp.getSource(), extractOp.getMixedOffsets(),
          extractOp.getMixedSizes(), extractOp.getMixedStrides());
      newDpsInit = alignDpsInsertSliceOp.getResult();
    }
  }

  // 8.a. Clone consumer op.
  auto clonedConsumerOp = cast<TilingInterface>(cloneOpAndUpdateDestinationArgs(
      rewriter, consumerOp, newDpsInitsForConsumerDest));

  // 8.b. Replace all uses of the loop result with the result of the cloned
  // tensor.insert_slice.
  OpOperand &operandToReplace = clonedConsumerOp->getOpOperand(operandNumber);
  rewriter.modifyOpInPlace(clonedConsumerOp, [&]() {
    operandToReplace.set(clonedInsertSliceOp.getResult());
  });

  // 9. Perform tiling of the cloned consumer and replace the operand at
  // `operandNumber` with the source of the cloned tensor.insert_slice op.
  FailureOr<TilingResult> tileAndFuseResult =
      tensor::replaceInsertSliceWithTiledConsumer(
          rewriter,
          cast<OffsetSizeAndStrideOpInterface>(
              clonedInsertSliceOp.getOperation()),
          clonedConsumerOp->getOpOperand(operandNumber));
  if (failed(tileAndFuseResult)) {
    return failure();
  }
  rewriter.replaceAllUsesWith(
      tileAndFuseResult->tiledOps[0]->getOperand(operandNumber),
      clonedInsertSliceOp.getSource());

  // 10. Try to fetch the offset and size for all results of the cloned
  // consumer. This would then be used to form the corresponding
  // tensor.insert_slice/parallel_insert_slice later.
  if (failed(computeAllResultTileForOpGivenOperandSliceOp(
          rewriter, clonedConsumerOp, operandNumber, ossSliceOp,
          allResultOffsets, allResultSizes))) {
    return failure();
  }

  if (!newExtractOps.empty()) {
    fixDpsInitsOfTiledConsumer(
        rewriter, tileAndFuseResult->tiledOps[0],
        newLoopBody->getArguments().drop_front(oldNumArguments),
        allResultOffsets, allResultSizes);
  }

  if (isInsertSliceOp) {
    auto newForOp = cast<scf::ForOp>(newInnerMostLoop);
    fixTerminatorSCFYield(
        rewriter, newForOp, tileAndFuseResult->tiledOps[0]->getResults(),
        allResultOffsets, allResultSizes,
        newForOp.getBody()->getArguments().take_back(newInitAppend.size()));
  } else {
    auto newForallOp = cast<scf::ForallOp>(newInnerMostLoop);
    fixTerminatorSCFInParallel(
        rewriter, newForallOp, tileAndFuseResult->tiledOps[0]->getResults(),
        allResultOffsets, allResultSizes,
        newForallOp.getBody()->getArguments().take_back(newInitAppend.size()));
  }

  newOuterLoops.push_back(cast<LoopLikeOpInterface>(newInnerMostLoop));

  // 11.a. Reconstruct terminator of outer loop by inner loop.
  auto outerCandidateIter = candidateSliceOpList.rbegin();
  for (auto [outerLoop, innerLoop] :
       llvm::zip_equal(MutableArrayRef(newOuterLoops).drop_back(),
                       MutableArrayRef(newOuterLoops).drop_front())) {
    // 11.b. Create insertSliceOp according outer candidateSliceOp
    if (outerCandidateIter != candidateSliceOpList.rend() &&
        outerCandidateIter->getOperation()
                ->getParentOfType<LoopLikeOpInterface>() == outerLoop) {
      if (auto forallOp = dyn_cast<scf::ForallOp>(outerLoop.getOperation())) {
        rewriter.setInsertionPoint(forallOp.getTerminator());
      } else {
        rewriter.setInsertionPointAfter(*outerCandidateIter);
      }

      if (failed(computeAllResultTileForOpGivenOperandSliceOp(
              rewriter, clonedConsumerOp, operandNumber, *outerCandidateIter,
              allResultOffsets, allResultSizes))) {
        return failure();
      }

      if (auto forOp = dyn_cast<scf::ForOp>(outerLoop.getOperation())) {
        fixTerminatorSCFYield(
            rewriter, forOp,
            innerLoop->getResults().take_back(newInitAppend.size()),
            allResultOffsets, allResultSizes,
            forOp.getBody()->getArguments().take_back(newInitAppend.size()));
      } else if (auto forallOp =
                     dyn_cast<scf::ForallOp>(outerLoop.getOperation())) {
        fixTerminatorSCFInParallel(
            rewriter, forallOp,
            innerLoop->getResults().take_back(newInitAppend.size()),
            allResultOffsets, allResultSizes,
            forallOp.getBody()->getArguments().take_back(newInitAppend.size()));
      }
      outerCandidateIter++;
    } else {
      // 11.c. Yield additional new appended results of innerLoop
      assert(isa<scf::ForOp>(outerLoop));
      auto forOp = cast<scf::ForOp>(outerLoop);
      auto outerLoopYield =
          cast<scf::YieldOp>(forOp.getBody()->getTerminator());
      SmallVector<Value> newYields =
          llvm::to_vector(outerLoopYield.getOperands());
      ValueRange additionalYields =
          innerLoop->getResults().take_back(newInitAppend.size());
      newYields.append(additionalYields.begin(), additionalYields.end());
      rewriter.setInsertionPoint(outerLoopYield);
      rewriter.replaceOpWithNewOp<scf::YieldOp>(outerLoopYield, newYields);
    }
  }

  // 12. Replace the result of consumer op with new outerMost loop's
  // results.
  for (auto &&[oldResult, newResult] :
       llvm::zip(consumerOp->getResults(),
                 newOuterLoops.front()->getResults().take_back(
                     newInitAppend.size()))) {
    rewriter.replaceAllUsesWith(oldResult, newResult);
  }

  // 13. Need to erase the cloned consumer op.
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
