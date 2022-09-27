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
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Arithmetic/Utils/Utils.h"
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
static SmallVector<unsigned>
fillInterchangeVector(ArrayRef<unsigned> interchangeVector,
                      size_t iterationDomainSize) {
  SmallVector<unsigned> filledVector = llvm::to_vector(interchangeVector);
  if (filledVector.size() < iterationDomainSize) {
    auto range = llvm::seq<unsigned>(filledVector.size(), iterationDomainSize);
    filledVector.append(range.begin(), range.end());
  }
  if (filledVector.size() > iterationDomainSize)
    filledVector.resize(iterationDomainSize);
  return filledVector;
}

/// Helper method to apply permutation to a vector
template <typename T>
static SmallVector<T> applyPermutationToVector(const SmallVector<T> &vector,
                                               ArrayRef<unsigned> interchange) {
  assert(interchange.size() == vector.size());
  return llvm::to_vector(
      llvm::map_range(interchange, [&](unsigned val) { return vector[val]; }));
}
/// Helper method to apply to invert a permutation.
static SmallVector<unsigned>
invertPermutationVector(ArrayRef<unsigned> interchange) {
  SmallVector<unsigned> inversion(interchange.size());
  for (const auto &pos : llvm::enumerate(interchange)) {
    inversion[pos.value()] = pos.index();
  }
  return inversion;
}
/// Method to check if an interchange vector is a permutation.
static bool isPermutation(ArrayRef<unsigned> interchange) {
  llvm::SmallDenseSet<unsigned, 4> seenVals;
  for (auto val : interchange) {
    if (seenVals.count(val))
      return false;
    seenVals.insert(val);
  }
  return seenVals.size() == interchange.size();
}

//===----------------------------------------------------------------------===//
// TileUsingSCFForOp pattern implementation.
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

/// If the tiled operation is in destination passing style, update the
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
/// TODO: This can be made much cleaner when `DestinationStyleOp` interface is
/// available generally.
static void
updateDestinationOperandsForTiledOp(OpBuilder &builder,
                                    ValueRange tiledOpDestinationValues,
                                    ValueRange bbArgsList) {
  for (auto destValue : llvm::enumerate(tiledOpDestinationValues)) {
    auto sliceOp = destValue.value().getDefiningOp<tensor::ExtractSliceOp>();
    if (!sliceOp)
      continue;
    sliceOp.setOperand(0, bbArgsList[destValue.index()]);
  }
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
    SmallVector<unsigned> interchangeVector;
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
    SmallVector<Operation *> tiledImplementation =
        op.getTiledImplementation(rewriter, offsets, sizes);
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
  if (tilingResult.loops.empty()) {
    // 5a. If there were no loops, the tiled implementation results are the
    // replacements.
    rewriter.replaceOp(op, tilingResult.tiledOp->getResults());
    return tilingResult;
  }

  // 6. Yield the results of the tiled operation from the loop nest as
  //    replacements for the original untiled ops.
  if (tilingResult.tiledOp->getNumResults() != op->getNumResults()) {
    return rewriter.notifyMatchFailure(
        tilingResult.tiledOp,
        "expected tiled op to have as many results as the untiled operation");
  }

  // `scf.for` with tensor semantics requires the loop nest to yield the
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
      yieldValueFn, /*replaceIterOperandsUsesInLoops =*/false);
  for (const auto &loop : llvm::enumerate(tilingResult.loops)) {
    rewriter.eraseOp(loop.value());
    tilingResult.loops[loop.index()] = newLoops[loop.index()];
  }
  rewriter.replaceOp(op, tilingResult.loops.front().getResults());
  return tilingResult;
}

//===----------------------------------------------------------------------===//
// TileConsumerAndFuseProducersUsingSCFForOp pattern implementation.
//===----------------------------------------------------------------------===//

scf::TileConsumerAndFuseProducersUsingSCFForOp::
    TileConsumerAndFuseProducersUsingSCFForOp(MLIRContext *context,
                                              scf::SCFTilingOptions options,
                                              PatternBenefit benefit)
    : OpInterfaceRewritePattern<TilingInterface>(context, benefit),
      tilingPattern(context, std::move(options)) {}

scf::TileConsumerAndFuseProducersUsingSCFForOp::
    TileConsumerAndFuseProducersUsingSCFForOp(StringRef opName,
                                              MLIRContext *context,
                                              scf::SCFTilingOptions options,
                                              PatternBenefit benefit)
    : OpInterfaceRewritePattern<TilingInterface>(context, benefit),
      tilingPattern(context, std::move(options)) {}

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

FailureOr<scf::SCFTileAndFuseResult>
scf::TileConsumerAndFuseProducersUsingSCFForOp::returningMatchAndRewrite(
    TilingInterface op, PatternRewriter &rewriter) const {
  // This transformation is only valid for ops that return values (i.e. not
  // valid to use with operations that have memref operands).
  if (!op->getNumResults()) {
    return rewriter.notifyMatchFailure(
        op, "invalid pattern for op with no results");
  }

  // 1. First tile the consumer.
  SCFTileAndFuseResult tileAndFuseResult;
  {
    FailureOr<SCFTilingResult> tilingResult =
        tilingPattern.returningMatchAndRewrite(op, rewriter);
    if (failed(tilingResult)) {
      return failure();
    }
    tileAndFuseResult.tiledAndFusedOps.push_back(tilingResult->tiledOp);
    tileAndFuseResult.loops = std::move(tilingResult->loops);
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
    tileAndFuseResult.tiledAndFusedOps.push_back(fusedProducer);
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
      unsigned resultNumber = fusableProducer.getResultNumber();
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
// LowerToLoopsUsingSCFForOp
//===----------------------------------------------------------------------===//

FailureOr<SmallVector<scf::ForOp>>
scf::LowerToLoopsUsingSCFForOp::returningMatchAndRewrite(
    TilingInterface op, PatternRewriter &rewriter) const {
  SmallVector<Range> domain = op.getIterationDomain(rewriter);

  // TODO: Handle cases where the op has results if needed.
  if (op->getNumResults() > 0) {
    return rewriter.notifyMatchFailure(
        op, "unable to lower to loops operations with return values");
  }

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
  rewriter.eraseOp(op);
  return loops;
}
