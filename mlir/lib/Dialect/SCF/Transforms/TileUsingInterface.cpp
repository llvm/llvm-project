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
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tile-using-interface"

using namespace mlir;

scf::SCFTilingOptions &
scf::SCFTilingOptions::setTileSizes(ArrayRef<int64_t> ts) {
  assert(!tileSizeComputationFunction && "tile sizes already set");
  SmallVector<int64_t> tileSizes(ts.begin(), ts.end());
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

/// Helper method to apply permutation to a vector
template <typename T>
static SmallVector<T> applyPermutationToVector(const SmallVector<T> &vector,
                                               ArrayRef<int64_t> interchange) {
  assert(interchange.size() == vector.size());
  return llvm::to_vector(
      llvm::map_range(interchange, [&](int64_t val) { return vector[val]; }));
}
/// Helper method to apply to invert a permutation.
static SmallVector<int64_t>
invertPermutationVector(ArrayRef<int64_t> interchange) {
  SmallVector<int64_t> inversion(interchange.size());
  for (const auto &pos : llvm::enumerate(interchange)) {
    inversion[pos.value()] = pos.index();
  }
  return inversion;
}
/// Method to check if an interchange vector is a permutation.
static bool isPermutation(ArrayRef<int64_t> interchange) {
  llvm::SmallDenseSet<int64_t, 4> seenVals;
  for (auto val : interchange) {
    if (seenVals.count(val))
      return false;
    seenVals.insert(val);
  }
  return seenVals.size() == interchange.size();
}

//===----------------------------------------------------------------------===//
// tileUsingSCFForOp implementation.
//===----------------------------------------------------------------------===//

// Check if `stride` evenly divides the trip count `size - offset`.
static bool tileDividesIterationDomain(Range loopRange) {
  Optional<int64_t> offsetAsInt = getConstantIntValue(loopRange.offset);
  if (!offsetAsInt)
    return false;
  Optional<int64_t> sizeAsInt = getConstantIntValue(loopRange.size);
  if (!sizeAsInt)
    return false;
  Optional<int64_t> strideAsInt = getConstantIntValue(loopRange.stride);
  if (!strideAsInt)
    return false;
  return ((sizeAsInt.value() - offsetAsInt.value()) % strideAsInt.value() == 0);
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
    Value offset =
        getValueOrCreateConstantIndexOp(builder, loc, loopRange.value().offset);
    Value size =
        getValueOrCreateConstantIndexOp(builder, loc, loopRange.value().size);
    // No loops if tile size is zero. Set offset and size to the loop
    // offset and size.
    if (matchPattern(tileSizeVals[loopRange.index()], m_Zero())) {
      offsets[loopRange.index()] = offset;
      sizes[loopRange.index()] = size;
      continue;
    }

    auto loop = builder.create<scf::ForOp>(
        loc, offset, size, tileSizeVals[loopRange.index()], ValueRange{},
        [&](OpBuilder &bodyBuilder, Location bodyLoc, Value iv,
            ValueRange /*iterArgs*/) {
          bool canAvoidMap = tileDividesIterationDomain(
              Range{loopRange.value().offset, loopRange.value().size,
                    tileSizeVals[loopRange.index()]});
          Value boundedTileSize =
              (canAvoidMap)
                  ? tileSizeVals[loopRange.index()]
                  : builder.create<AffineMinOp>(
                        bodyLoc, minMap,
                        ValueRange{iv, tileSizeVals[loopRange.index()], size});
          sizes[loopRange.index()] = boundedTileSize;
          builder.create<scf::YieldOp>(loc);
        });
    offsets[loopRange.index()] = loop.getInductionVar();
    loops.push_back(loop);
    builder.setInsertionPoint(loop.getBody()->getTerminator());
  }
  return loops;
}

/// For a value to be yielded (`yieldedValue`) from within a loop nest `loops`,
/// construct the destructive update pattern that inserts the yielded
/// value into a destination tensor provided by `initValue` at offset
/// `tileOffsets` and size `tileSizes`. For example,
///
/// ```mlir
/// scf.for %iv0 = ... {
///   %0 = tiled_op
/// }
/// ```
///
/// is transformed to
///
/// ```mlir
/// scf.for %iv0 = ... iter_args(%arg = %0) {
///   %1 = tensor.extract_slice %arg
///   %2 = tiled_op
///   %3 = tensor.insert_slice %2 into %arg
///   scf.yield %3
/// }
/// ```
/// TODO: This API can be cleaned up by using `SubsetExtractOpInterface`.
static FailureOr<SmallVector<Value>>
yieldTiledValues(RewriterBase &rewriter, ValueRange initValues,
                 ValueRange yieldedValues,
                 ArrayRef<SmallVector<OpFoldResult>> tileOffsetsList,
                 ArrayRef<SmallVector<OpFoldResult>> tileSizesList,
                 MutableArrayRef<scf::ForOp> loops) {
  NewYieldValueFn yieldValueFn =
      [&](OpBuilder &b, Location loc,
          ArrayRef<BlockArgument> newBBArgs) -> SmallVector<Value> {
    SmallVector<Value> inserts;
    for (const auto &yieldedValue : llvm::enumerate(yieldedValues)) {
      ArrayRef<OpFoldResult> tileOffsets =
          tileOffsetsList[yieldedValue.index()];
      ArrayRef<OpFoldResult> tileSizes = tileSizesList[yieldedValue.index()];
      SmallVector<OpFoldResult> tileStrides(tileOffsets.size(),
                                            b.getIndexAttr(1));
      Value insert = b.create<tensor::InsertSliceOp>(
          loc, yieldedValue.value(), newBBArgs[yieldedValue.index()],
          tileOffsets, tileSizes, tileStrides);
      inserts.push_back(insert);
    }
    return inserts;
  };

  SmallVector<scf::ForOp> newLoops =
      replaceLoopNestWithNewYields(rewriter, loops, initValues, yieldValueFn,
                                   /*replaceIterOperandsUsesInLoop =*/false);
  for (const auto &loop : llvm::enumerate(loops)) {
    rewriter.eraseOp(loop.value());
    loops[loop.index()] = newLoops[loop.index()];
  }
  return llvm::to_vector(llvm::map_range(
      loops.front().getResults().take_back(yieldedValues.size()),
      [](OpResult r) -> Value { return r; }));
}

/// If the tiled operation is destination passing style, update the
/// slice of the destination used (which refers to the untiled destination)
/// to use the corresponding region argument of the innermost loop.
///
/// ```mlir
/// %0 =
/// scf.for %iv0 = ... iter_args(%arg = %0) {
///   %1 = tensor.extract_slice %0
///   %2 = tiled_op
///   %3 = tensor.insert_slice %2 into %arg
///   scf.yield %3
/// }
/// ```
///
/// is transformed to
///
/// ```mlir
/// scf.for %iv0 = ... iter_args(%arg = %0) {
///   %1 = tensor.extract_slice %arg
///   %2 = tiled_op
///   %3 = tensor.insert_slice %2 into %arg
///   scf.yield %3
/// }
/// ```
static void
updateDestinationOperandsForTiledOp(OpBuilder &builder,
                                    ValueRange tiledOpDestinationValues,
                                    ValueRange bbArgsList) {
  for (const auto &destValue : llvm::enumerate(tiledOpDestinationValues)) {
    auto sliceOp = destValue.value().getDefiningOp<tensor::ExtractSliceOp>();
    if (!sliceOp)
      continue;
    sliceOp.setOperand(0, bbArgsList[destValue.index()]);
  }
}

/// Implementation of tiling transformation of `op` that implements the
/// `TilingInterface` using `scf.for` to iterate over the tiles.
FailureOr<scf::SCFTilingResult>
mlir::scf::tileUsingSCFForOp(RewriterBase &rewriter, TilingInterface op,
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
  if (numLoops == 0) {
    return rewriter.notifyMatchFailure(
        op, "unable to tile op with no iteration domain");
  }

  // 2. Materialize the tile sizes. Enforce the convention that "tiling by zero"
  // skips tiling a particular dimension. This convention is significantly
  // simpler to handle instead of adjusting affine maps to account for missing
  // dimensions.
  SmallVector<Value> tileSizeVector =
      options.tileSizeComputationFunction(rewriter, op);
  if (tileSizeVector.size() < iterationDomain.size()) {
    auto zero = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
    tileSizeVector.append(numLoops - tileSizeVector.size(), zero);
  }

  scf::SCFTilingResult tilingResult;
  SmallVector<OpFoldResult> offsets, sizes;
  {
    // If there is an interchange specified, permute the iteration domain and
    // the tile sizes.
    SmallVector<int64_t> interchangeVector;
    if (!options.interchangeVector.empty()) {
      interchangeVector = fillInterchangeVector(options.interchangeVector,
                                                iterationDomain.size());
    }
    if (!interchangeVector.empty()) {
      if (!isPermutation(interchangeVector)) {
        return rewriter.notifyMatchFailure(
            op, "invalid intechange vector, not a permutation of the entire "
                "iteration space");
      }

      iterationDomain =
          applyPermutationToVector(iterationDomain, interchangeVector);
      tileSizeVector =
          applyPermutationToVector(tileSizeVector, interchangeVector);
    }

    // 3. Materialize an empty loop nest that iterates over the tiles. These
    // loops for now do not return any values even if the original operation has
    // results.
    tilingResult.loops = generateTileLoopNest(
        rewriter, op.getLoc(), iterationDomain, tileSizeVector, offsets, sizes);

    if (!interchangeVector.empty()) {
      auto inversePermutation = invertPermutationVector(interchangeVector);
      offsets = applyPermutationToVector(offsets, inversePermutation);
      sizes = applyPermutationToVector(sizes, inversePermutation);
    }
  }

  LLVM_DEBUG({
    if (!tilingResult.loops.empty()) {
      llvm::dbgs() << "LoopNest shell :\n";
      tilingResult.loops.front().dump();
      llvm::dbgs() << "\n";
    }
  });

  // 4. Generate the tiled implementation within the inner most loop.
  if (!tilingResult.loops.empty())
    rewriter.setInsertionPoint(
        tilingResult.loops.back().getBody()->getTerminator());
  SmallVector<Operation *> tiledImplementation =
      op.getTiledImplementation(rewriter, offsets, sizes);
  if (tiledImplementation.size() != 1) {
    return rewriter.notifyMatchFailure(
        op, "expected tiled implementation to return a single op");
  }
  tilingResult.tiledOp = tiledImplementation[0];
  if (op->getNumResults() == 0) {
    // nothing more to do.
    return tilingResult;
  }

  // If loops are empty, the tiled op is used as the replacement for the untiled
  // op.
  if (tilingResult.loops.empty()) {
    tilingResult.replacements = llvm::to_vector(
        llvm::map_range(tiledImplementation[0]->getResults(),
                        [](OpResult result) -> Value { return result; }));
    return tilingResult;
  }

  // 5. Yield all the results of the tiled operation. The surrounding loop
  //    nest is modified to insert a destructive update pattern to yield
  //    from the loop nest values to replace the untiled op with.
  int64_t numResults = op->getNumResults();
  SmallVector<SmallVector<OpFoldResult>> resultOffsetsList(numResults),
      resultSizesList(numResults);
  for (const auto &result : llvm::enumerate(op->getResults())) {
    if (failed(op.getResultTilePosition(rewriter, result.index(), offsets,
                                        sizes,
                                        resultOffsetsList[result.index()],
                                        resultSizesList[result.index()]))) {
      return rewriter.notifyMatchFailure(
          op, "failed to get slice of result produced");
    }
  }

  FailureOr<SmallVector<Value>> replacementOr =
      yieldTiledValues(rewriter, op.getDestinationOperands(rewriter),
                       tilingResult.tiledOp->getResults(), resultOffsetsList,
                       resultSizesList, tilingResult.loops);
  if (failed(replacementOr))
    return rewriter.notifyMatchFailure(op, "failed to yield replacement");
  if (auto tiledInterfaceOp = dyn_cast<TilingInterface>(tilingResult.tiledOp)) {
    auto innerMostLoop = tilingResult.loops.back();
    updateDestinationOperandsForTiledOp(
        rewriter, tiledInterfaceOp.getDestinationOperands(rewriter),
        innerMostLoop.getRegionIterArgs());
  }

  tilingResult.replacements = replacementOr.value();

  LLVM_DEBUG({
    if (!tilingResult.loops.empty()) {
      llvm::dbgs() << "After tiled implementation :\n";
      tilingResult.loops.front().dump();
      llvm::dbgs() << "\n";
    }
  });
  return tilingResult;
}

//===----------------------------------------------------------------------===//
// tileConsumerAndFuseProducerGreedilyUsingSCFForOp implementation.
//===----------------------------------------------------------------------===//

/// Return the untiled producer whose slice is used in a tiled consumer. The
/// method traverses the tile loop nest (`loops`) if needed, and returns the
/// `iter_args` of the outer most that is encountered. Traversing the iter_args
/// indicates that this is a destination operand of the consumer. If there was
/// no loop traversal needed, the second value of the returned tuple is empty.
static std::tuple<OpResult, Optional<OpOperand *>>
getUntiledProducerFromSliceSource(OpOperand *source,
                                  ArrayRef<scf::ForOp> loops) {
  Optional<OpOperand *> destinationIterArg;
  auto loopIt = loops.rbegin();
  while (auto iterArg = source->get().dyn_cast<BlockArgument>()) {
    scf::ForOp loop = *loopIt;
    if (iterArg.getOwner()->getParentOp() != loop)
      break;
    source = &loop.getOpOperandForRegionIterArg(iterArg);
    loopIt++;
  }
  if (loopIt == loops.rend())
    destinationIterArg = source;
  return {source->get().dyn_cast<OpResult>(), destinationIterArg};
}

/// Implementation of tile consumer and fuse producer greedily.
FailureOr<scf::SCFTileAndFuseResult>
mlir::scf::tileConsumerAndFuseProducerGreedilyUsingSCFForOp(
    RewriterBase &rewriter, TilingInterface consumer,
    const scf::SCFTileAndFuseOptions &options) {
  // This transformation is only valid for ops that return values (i.e. not
  // valid to use with operations that have memref operands).
  if (!consumer->getNumResults()) {
    return rewriter.notifyMatchFailure(
        consumer, "invalid pattern for op with no results");
  }

  // 1. First tile the consumer.
  scf::SCFTileAndFuseResult tileAndFuseResult;
  llvm::SmallDenseMap<Value, int64_t> yieldedValueToResultNumber;
  {
    FailureOr<scf::SCFTilingResult> tilingResult =
        tileUsingSCFForOp(rewriter, consumer, options.tilingOptions);
    if (failed(tilingResult))
      return rewriter.notifyMatchFailure(consumer, "failed to tile consumer");
    tileAndFuseResult.tiledAndFusedOps.insert(tilingResult->tiledOp);
    tileAndFuseResult.loops = std::move(tilingResult->loops);
    for (const auto &result : llvm::enumerate(
             llvm::zip(consumer->getResults(), tilingResult->replacements))) {
      tileAndFuseResult.replacements[std::get<0>(result.value())] =
          std::get<1>(result.value());
      yieldedValueToResultNumber[tilingResult->tiledOp->getResult(
          result.index())] = result.index();
    }
  }

  // If there are no loops generated, fusion is immaterial.
  if (tileAndFuseResult.loops.empty())
    return tileAndFuseResult;

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
  addCandidateSlices(tileAndFuseResult.tiledAndFusedOps.back(), candidates);
  OpBuilder::InsertionGuard g(rewriter);
  while (!candidates.empty()) {
    // 2a. Traverse the slices in BFS fashion.
    tensor::ExtractSliceOp candidateSliceOp = candidates.front();
    candidates.pop_front();

    // 2b. Get the producer of the source (potentially walking through
    // `iter_args` of nested `scf.for`)
    auto [fusableProducer, destinationIterArg] =
        getUntiledProducerFromSliceSource(&candidateSliceOp->getOpOperand(0),
                                          tileAndFuseResult.loops);
    if (!fusableProducer)
      continue;

    // 2c. Generate the tiled implementation of the producer of the source
    rewriter.setInsertionPoint(candidateSliceOp);
    FailureOr<Value> fusedProducerValue =
        tensor::replaceExtractSliceWithTiledProducer(rewriter, candidateSliceOp,
                                                     fusableProducer);
    if (failed(fusedProducerValue))
      continue;
    rewriter.replaceOp(candidateSliceOp, fusedProducerValue.value());

    // 2d. The operands of the fused producer might themselved be slices of
    //     values produced by operations that implement the `TilingInterface`.
    //     Add these operations to the worklist.
    Operation *fusedProducer = fusedProducerValue->getDefiningOp();
    tileAndFuseResult.tiledAndFusedOps.insert(fusedProducer);
    addCandidateSlices(fusedProducer, candidates);

    // 2e. If the slice is for a destination operand, for example,
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
    //     %4 = tensor.extract_slice %0 /*incorrect value */ [..]
    //     %5 = linalg.fill .. outs(%4 : )
    //     .. = linalg.matmul .. outs(%5 : )
    //   }
    // }
    // ```
    //
    // The untiled `linalg.fill` is still used as the `init_value` since it
    // was originally a destination operand of the untiled `linalg.matmul`.
    // When fusing an operand that is a destination operand.
    //   - Update the iter_arg of the outer most loop to use the destination
    //     of the untiled producer.
    //   - Update the destination of the slice of the tiled producer generated
    //     to use the same basic block argument as the slice that was used to
    //     generate inplace the tiled implementation of the producer.
    // With this the IR will be.
    //
    // ```
    // %0 = linalg.init
    // %1 = scf.for .. iter_args(%arg0 = %0 /* corrected value */ ) {
    //   %2 = scf.for .. iter_args(%arg1 = %arg0) {
    //     %3 = tensor.extract_slice %arg1 /* corrected value */ [..]
    //     %4 = linalg.fill .. outs(%3 : )
    //     .. = linalg.matmul .. outs(%4 : )
    //   }
    // }
    // ```
    // TODO: This can be modeled better if the `DestinationStyleOpInterface`.
    // Update to use that when it does become available.
    scf::ForOp outerMostLoop = tileAndFuseResult.loops.front();
    Optional<unsigned> iterArgNumber;
    if (destinationIterArg) {
      iterArgNumber = outerMostLoop.getIterArgNumberForOpOperand(
          *destinationIterArg.value());
    }
    if (iterArgNumber) {
      int64_t resultNumber = fusableProducer.getResultNumber();
      if (auto producerOp =
              dyn_cast<TilingInterface>(fusableProducer.getOwner())) {
        SmallVector<Value> destination =
            producerOp.getDestinationOperands(rewriter);
        outerMostLoop.setIterArg(iterArgNumber.value(),
                                 destination[resultNumber]);
      }
      if (auto tiledAndFusedInterfaceOp =
              fusedProducerValue.value().getDefiningOp<TilingInterface>()) {
        scf::ForOp innerMostLoop = tileAndFuseResult.loops.back();
        SmallVector<Value> destination =
            tiledAndFusedInterfaceOp.getDestinationOperands(rewriter);
        updateDestinationOperandsForTiledOp(
            rewriter, destination[resultNumber],
            innerMostLoop.getRegionIterArgs()[iterArgNumber.value()]);
      }
    }
  }
  return tileAndFuseResult;
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
