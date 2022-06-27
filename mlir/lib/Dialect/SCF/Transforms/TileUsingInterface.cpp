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

//===----------------------------------------------------------------------===//
// TileUsingSCFForOp pattern implementation.
//===----------------------------------------------------------------------===//

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

/// Return the `Value` that is defined by an operation that implements
/// the `TilingInterface`. Looks through `iter_args` of scf.for nest
/// if required.
static Optional<OpResult> getFusableProducer(Value v) {
  while (auto blockArg = v.dyn_cast<BlockArgument>()) {
    auto loopOp = dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp());
    if (!loopOp)
      return llvm::None;
    v = loopOp.getOpOperandForRegionIterArg(blockArg).get();
  }
  if (!isa_and_nonnull<TilingInterface>(v.getDefiningOp()))
    return llvm::None;
  return v.cast<OpResult>();
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
    Optional<OpResult> fusableProducer =
        getFusableProducer(candidateSliceOp.source());
    if (!fusableProducer)
      continue;

    // 2c. Generate the tiled implementation of the producer of the source
    rewriter.setInsertionPoint(candidateSliceOp);
    FailureOr<Value> fusedProducerValue =
        tensor::replaceExtractSliceWithTiledProducer(
            rewriter, candidateSliceOp, fusableProducer.getValue());
    if (failed(fusedProducerValue))
      continue;
    rewriter.replaceOp(candidateSliceOp, fusedProducerValue.getValue());

    // 2d. The operands of the fused producer might themselved be slices of
    //     values produced by operations that implement the `TilingInterface`.
    //     Add these operations to the worklist.
    Operation *fusedProducer = fusedProducerValue->getDefiningOp();
    tileAndFuseResult.tiledAndFusedOps.push_back(fusedProducer);
    addCandidateSlices(fusedProducer, candidates);

    // 2e. If the operation being fused creates a value that is used as `outs`
    //     in the tiled operation, the result of the unfused operation will be
    //     used in the `iter_args` of the tiled loop generated. When the
    //     operation is fused, this use in `iter_args` needs to be modified to
    //     use the destination of the fused operation. For example, starting
    //     with
    //
    //     ```mlir
    //     %0 = linalg.init_tensor ...
    //     %1 = linalg.fill ... outs(%0:...)...
    //     %2 = linalg.matmul ... outs(%1:...)....
    //     ```
    //
    //     First the `linalg.matmul` gets tiled
    //
    //     ```mlir
    //     %0 = linalg.init_tensor
    //     %1 = linalg.fill
    //     %2 = scf.for .... iter_args(%arg0 = %1)...
    //        ...
    //        ... = linalg.matmul ...
    //
    //     ```
    //
    //     When the `linalg.fill` gets fused, the `iter_args` needs to be
    //     modified
    //
    //     ```mlir
    //     %0 = linalg.init_tensor
    //     %1 = scf.for ... iter_args(%arg0 = %0)...
    //        ...
    //        %2 = linalg.fill ...
    //        %3 = linalg.matmul ... outs(%2: ...)...
    //     ```
    TilingInterface unfusedProducerOp =
        cast<TilingInterface>(fusableProducer->getOwner());
    scf::ForOp outerMostTiledLoop = tileAndFuseResult.loops.front();
    SmallVector<Value> unfusedProducerOpDestValues =
        unfusedProducerOp.getDestinationOperands(rewriter);
    for (OpOperand &uses : unfusedProducerOp->getUses()) {
      if (uses.getOwner() == outerMostTiledLoop.getOperation()) {
        unsigned resultNumber = uses.get().cast<OpResult>().getResultNumber();
        unsigned operandNumber = uses.getOperandNumber();
        outerMostTiledLoop->setOperand(
            operandNumber, unfusedProducerOpDestValues[resultNumber]);
      }
    }
  }
  return tileAndFuseResult;
}
