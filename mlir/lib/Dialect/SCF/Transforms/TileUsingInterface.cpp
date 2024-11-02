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
  SmallVector<Value> ivs;

  for (auto [loopRange, tileSize] : llvm::zip_equal(loopRanges, tileSizes)) {
    // No loops if tile size is zero. Set offset and size to the loop
    // offset and size.
    if (isConstantIntValue(tileSize, 0))
      continue;

    Value lb = getValueOrCreateConstantIndexOp(rewriter, loc, loopRange.offset);
    Value ub = getValueOrCreateConstantIndexOp(rewriter, loc, loopRange.size);
    Value step = getValueOrCreateConstantIndexOp(rewriter, loc, tileSize);
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
    ArrayRef<OpFoldResult> tileSizes, ArrayRef<Attribute> mappingVector,
    ValueRange destinationTensors, YieldTiledValuesFn tiledBodyFn,
    SmallVector<LoopLikeOpInterface> &loops) {
  SmallVector<OpFoldResult> lbs, ubs, steps;
  assert(!loopRanges.empty() && "unexpected empty loop ranges");
  assert(loopRanges.size() == tileSizes.size() &&
         "expected as many tile sizes as loop ranges");
  OpBuilder::InsertionGuard guard(rewriter);
  SmallVector<OpFoldResult> offsets(loopRanges.size()),
      sizes(loopRanges.size());

  for (auto [tileSize, loopRange] : llvm::zip_equal(tileSizes, loopRanges)) {
    if (isConstantIntValue(tileSize, 0))
      continue;
    lbs.push_back(loopRange.offset);
    ubs.push_back(loopRange.size);
    steps.push_back(tileSize);
  }
  assert(!lbs.empty() && "Expected at least one loop range");

  std::optional<ArrayAttr> mappingAttr;
  if (!mappingVector.empty())
    mappingAttr = rewriter.getArrayAttr(mappingVector);

  auto forallOp = rewriter.create<scf::ForallOp>(
      loc, lbs, ubs, steps, destinationTensors, mappingAttr);
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
static LogicalResult generateLoopNest(RewriterBase &rewriter, Location loc,
                                      const scf::SCFTilingOptions &options,
                                      ArrayRef<Range> loopRanges,
                                      ArrayRef<OpFoldResult> tileSizes,
                                      ValueRange destinationTensors,
                                      YieldTiledValuesFn tiledBodyFn,
                                      SmallVector<LoopLikeOpInterface> &loops) {
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
        rewriter, loc, loopRanges, tileSizes, options.mappingVector,
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
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfter(op);

  if (!options.tileSizeComputationFunction) {
    return rewriter.notifyMatchFailure(
        op, "missing tile size computation function");
  }

  // 1. Get the range of the loops that are represented by the operation.
  SmallVector<Range> iterationDomain = op.getIterationDomain(rewriter);
  size_t numLoops = iterationDomain.size();

  // 2. Materialize the tile sizes. Enforce the convention that "tiling by zero"
  // skips tiling a particular dimension. This convention is significantly
  // simpler to handle instead of adjusting affine maps to account for missing
  // dimensions.
  SmallVector<OpFoldResult> tileSizes =
      options.tileSizeComputationFunction(rewriter, op);
  if (tileSizes.size() < iterationDomain.size()) {
    auto zero = rewriter.getIndexAttr(0);
    tileSizes.append(numLoops - tileSizes.size(), zero);
  }

  // 3. If there is an interchange specified, permute the iteration domain and
  // the tile sizes.
  SmallVector<int64_t> interchangeVector;
  if (!options.interchangeVector.empty()) {
    interchangeVector = fillInterchangeVector(options.interchangeVector,
                                              iterationDomain.size());
  }
  if (!interchangeVector.empty()) {
    if (!isPermutationVector(interchangeVector)) {
      return rewriter.notifyMatchFailure(
          op, "invalid intechange vector, not a permutation of the entire "
              "iteration space");
    }

    applyPermutationToVector(iterationDomain, interchangeVector);
    applyPermutationToVector(tileSizes, interchangeVector);
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
    {
      int materializedLoopNum = 0;
      for (auto [tileSize, loopRange] :
           llvm::zip_equal(tileSizes, iterationDomain)) {
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
                              tileSizes, destinationTensors,
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
  if (op->getNumResults() != 1)
    return b.notifyMatchFailure(
        op, "don't support ops with multiple results for now");
  SmallVector<utils::IteratorType> iterators =
      tilingInterfaceOp.getLoopIteratorTypes();

  SmallVector<int> reductionDims;
  for (auto [idx, iteratorType] :
       llvm::enumerate(tilingInterfaceOp.getLoopIteratorTypes())) {
    if (iteratorType == utils::IteratorType::reduction)
      reductionDims.push_back(idx);
  }

  // 2. create the inital tensor value.
  FailureOr<Operation *> identityTensor =
      op.generateInitialTensorForPartialReduction(b, loc, tileSizesVector,
                                                  reductionDims);
  if (failed(identityTensor))
    return b.notifyMatchFailure(op,
                                "cannot create a tensor of identity value.");

  // 3. Define the callback to use for generating the inner most tile loop body.
  Operation *parallelOp = nullptr;
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
    auto clonedOp = cast<PartialReductionOpInterface>(
        cloneOpAndUpdateDestinationArgs(b, op, regionIterArgs));

    // 4b. Tile the cloned operation.
    parallelOp = clonedOp.tileToPartialReduction(b, loc, regionIterArgs,
                                                 offsets, sizes, reductionDims);
    // 4c. Delete the cloned operation.
    b.eraseOp(clonedOp);

    tiledResult.append(parallelOp->result_begin(), parallelOp->result_end());
    // 4d. Compute the offsets and sizes needed to insert the result of the
    // tiled value back into destination before yielding the destination.
    SmallVector<OpFoldResult> outOffsets(offsets.size(), b.getIndexAttr(0));
    resultOffsets.emplace_back(std::move(outOffsets));

    SmallVector<OpFoldResult> outSizes;
    for (size_t i = 0; i < offsets.size(); i++) {
      outSizes.push_back(
          tensor::getMixedSize(b, loc, parallelOp->getResult(0), i));
    }
    resultSizes.emplace_back(std::move(outSizes));
    return success();
  };

  // 5. Generate the tiled implementation using the destination tensors.
  SmallVector<Value> destinationTensors =
      llvm::map_to_vector(identityTensor.value()->getResults(),
                          [](OpResult res) -> Value { return res; });

  SmallVector<LoopLikeOpInterface> loops;
  scf::SCFTilingOptions options;
  options.setLoopType(scf::SCFTilingOptions::LoopType::ForOp);
  if (failed(generateLoopNest(b, loc, options, iterationDomain, tileSizesVector,
                              destinationTensors, innerYieldTiledValuesFn,
                              loops)))
    return b.notifyMatchFailure(op, "failed to tile for parallel reduction");

  SmallVector<Value> replacements = llvm::map_to_vector(
      loops.front()->getResults(), [](OpResult r) -> Value { return r; });

  // 5. Apply the merge reduction to combine all the partial values.
  b.setInsertionPointAfter(*loops.begin());
  Operation *mergeOp = op.mergeReductions(b, loc, replacements, reductionDims);
  b.replaceOp(op, mergeOp->getResults());

  SCFReductionTilingResult results;
  results.initialOp = *identityTensor;
  results.loops = loops;
  results.parallelTiledOp = parallelOp;
  results.mergeOp = mergeOp;
  return results;
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
    MutableArrayRef<LoopLikeOpInterface> loops) {
  if (loops.empty())
    return success();

  OpResult fusableProducer = fusedProducerInfo.origProducer;
  Value tiledAndFusedProducer = fusedProducerInfo.tiledAndFusedProducer;
  FailureOr<Value> initValue = tensor::getOrCreateDestination(
      rewriter, fusableProducer.getOwner()->getLoc(), fusableProducer);
  if (succeeded(initValue)) {

    YieldTiledValuesFn newYieldValuesFn =
        [&](RewriterBase &innerRewriter, Location loc, ValueRange /*ivs*/,
            ValueRange newRegionIterArgs, SmallVector<Value> &tiledResult,
            SmallVector<SmallVector<OpFoldResult>> &tiledOffset,
            SmallVector<SmallVector<OpFoldResult>> &tiledSizes)
        -> LogicalResult {
      OpBuilder::InsertionGuard g(innerRewriter);
      if (auto tiledDestStyleOp =
              tiledAndFusedProducer
                  .getDefiningOp<DestinationStyleOpInterface>()) {
        rewriter.setInsertionPoint(tiledDestStyleOp);
        Value newRegionArg = newRegionIterArgs.back();
        auto destSlice = rewriter.create<tensor::ExtractSliceOp>(
            sliceOp.getLoc(), newRegionArg, sliceOp.getMixedOffsets(),
            sliceOp.getMixedSizes(), sliceOp.getMixedStrides());
        unsigned resultNumber = fusableProducer.getResultNumber();
        rewriter.modifyOpInPlace(tiledDestStyleOp, [&]() {
          tiledDestStyleOp.getDpsInitsMutable()[resultNumber].set(destSlice);
        });
      }
      Block *block = rewriter.getInsertionPoint()->getBlock();
      rewriter.setInsertionPoint(block->getTerminator());
      tiledResult.push_back(fusedProducerInfo.tiledAndFusedProducer);
      tiledOffset.emplace_back(sliceOp.getMixedOffsets());
      tiledSizes.emplace_back(sliceOp.getMixedSizes());
      return success();
    };

    return addInitOperandsToLoopNest(rewriter, loops,
                                     SmallVector<Value>{initValue.value()},
                                     newYieldValuesFn);
  }
  return success();
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
      if (failed(yieldReplacementForFusedProducer(
              rewriter, candidateSliceOp, fusedResult.value(), loops))) {
        return rewriter.notifyMatchFailure(
            fusableProducer.getOwner(), "failed to replacement value for this "
                                        "oepration from within the tiled loop");
      }
      origValToResultNumber[fusableProducer] =
          loops.front()->getNumResults() - 1;
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
