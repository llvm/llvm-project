//===- Tiling.cpp - Implementation of linalg Tiling -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the linalg dialect Tiling pass.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include <utility>

namespace mlir {
#define GEN_PASS_DEF_LINALGTILINGPASS
#include "mlir/Dialect/Linalg/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::linalg;
using namespace mlir::scf;

#define DEBUG_TYPE "linalg-tiling"

static bool isZero(OpFoldResult v) {
  if (!v)
    return false;
  if (auto attr = v.dyn_cast<Attribute>()) {
    IntegerAttr intAttr = attr.dyn_cast<IntegerAttr>();
    return intAttr && intAttr.getValue().isZero();
  }
  if (auto cst = v.get<Value>().getDefiningOp<arith::ConstantIndexOp>())
    return cst.value() == 0;
  return false;
}

std::tuple<SmallVector<Range, 4>, LoopIndexToRangeIndexMap>
mlir::linalg::makeTiledLoopRanges(RewriterBase &b, Location loc, AffineMap map,
                                  ArrayRef<OpFoldResult> allShapeSizes,
                                  ArrayRef<OpFoldResult> allTileSizes) {
  assert(allTileSizes.size() == map.getNumResults());
  // Apply `map` to get shape sizes in loop order.
  SmallVector<OpFoldResult> shapeSizes =
      makeComposedFoldedMultiResultAffineApply(b, loc, map, allShapeSizes);
  SmallVector<OpFoldResult> tileSizes(allTileSizes.begin(), allTileSizes.end());

  // Traverse the tile sizes, which are in loop order, erase zeros everywhere.
  LoopIndexToRangeIndexMap loopIndexToRangeIndex;
  for (int idx = 0, e = tileSizes.size(), zerosCount = 0; idx < e; ++idx) {
    if (isZero(tileSizes[idx - zerosCount])) {
      shapeSizes.erase(shapeSizes.begin() + idx - zerosCount);
      tileSizes.erase(tileSizes.begin() + idx - zerosCount);
      ++zerosCount;
      continue;
    }
    loopIndexToRangeIndex[idx] = idx - zerosCount;
  }

  // Create a new range with the applied tile sizes.
  SmallVector<Range, 4> res;
  for (unsigned idx = 0, e = tileSizes.size(); idx < e; ++idx)
    res.push_back(Range{b.getIndexAttr(0), shapeSizes[idx], tileSizes[idx]});
  return std::make_tuple(res, loopIndexToRangeIndex);
}

void mlir::linalg::transformIndexOps(
    RewriterBase &b, LinalgOp op, SmallVectorImpl<Value> &ivs,
    const LoopIndexToRangeIndexMap &loopIndexToRangeIndex) {
  SmallVector<Value> allIvs(op.getNumLoops(), nullptr);
  for (auto &en : enumerate(allIvs)) {
    auto rangeIndex = loopIndexToRangeIndex.find(en.index());
    if (rangeIndex == loopIndexToRangeIndex.end())
      continue;
    en.value() = ivs[rangeIndex->second];
  }
  offsetIndices(b, op, getAsOpFoldResult(allIvs));
}

/// Asserts that the given index-typed value is strictly positive. If the value
/// is an attribute, asserts at compile time, otherwise emits an assertion
/// checked at runtime.
static void emitIsPositiveIndexAssertion(ImplicitLocOpBuilder &b,
                                         OpFoldResult value) {
  if (auto attr = value.dyn_cast<Attribute>()) {
    assert(attr.cast<IntegerAttr>().getValue().isStrictlyPositive() &&
           "expected strictly positive tile size and divisor");
    return;
  }

  Value zero = b.create<arith::ConstantIndexOp>(0);
  Value condition = b.create<arith::CmpIOp>(arith::CmpIPredicate::sgt,
                                            value.get<Value>(), zero);
  b.create<cf::AssertOp>(
      condition,
      b.getStringAttr("expected strictly positive tile size and divisor"));
}

FailureOr<StaticMultiSizeSpecification>
mlir::linalg::computeStaticMultiTileSizes(LinalgOp op, unsigned dimension,
                                          int64_t targetSize, int64_t divisor) {
  assert(!op.hasDynamicShape() &&
         "cannot compute static multi-tile sizes for an op with dynamic shape");
  assert(targetSize > 0 && "target size must be non-negative");
  assert(divisor > 0 && "divisor must be non-negative");
  assert(dimension < op.getNumLoops() && "dimension overflow");

  StaticMultiSizeSpecification spec;
  int64_t tripCount = op.getStaticLoopRanges()[dimension];
  int64_t a = tripCount / divisor;
  int64_t t = (targetSize + divisor - 1) / divisor;
  int64_t totalTripCount = (a + t - 1) / t;
  spec.lowTileSize = (a / totalTripCount) * divisor;
  spec.highTileSize = spec.lowTileSize + divisor;
  spec.highTripCount = a % totalTripCount;
  spec.lowTripCount = totalTripCount - spec.highTripCount;
  if (spec.lowTileSize * spec.lowTripCount +
          spec.highTileSize * spec.highTripCount !=
      tripCount) {
    return failure();
  }
  return spec;
}

FailureOr<MultiSizeSpecification>
mlir::linalg::computeMultiTileSizes(OpBuilder &builder, LinalgOp op,
                                    unsigned dimension, OpFoldResult targetSize,
                                    OpFoldResult divisor, bool emitAssertions) {
  // Bail out on dimension overflow.
  if (dimension >= op.getNumLoops())
    return failure();

  // The code below works only on values.
  Location loc = op.getLoc();
  ImplicitLocOpBuilder b(loc, builder);
  if (emitAssertions) {
    emitIsPositiveIndexAssertion(b, targetSize);
    emitIsPositiveIndexAssertion(b, divisor);
  }
  Value targetSizeValue =
      getValueOrCreateConstantIndexOp(builder, loc, targetSize);
  Value divisorValue = getValueOrCreateConstantIndexOp(builder, loc, divisor);

  // Find the trip count of the iteration space dimension for which the tile
  // sizes are computed.
  SmallVector<OpFoldResult> allShapes =
      op.createFlatListOfOperandDims(b, b.getLoc());
  AffineMap shapesToLoops = op.getShapesToLoopsMap();
  SmallVector<OpFoldResult> loopRanges =
      makeComposedFoldedMultiResultAffineApply(b, op.getLoc(), shapesToLoops,
                                               allShapes);
  Value tripCount =
      getValueOrCreateConstantIndexOp(b, op.getLoc(), loopRanges[dimension]);

  // Compute the tile sizes and the respective numbers of tiles.
  AffineExpr s0 = b.getAffineSymbolExpr(0);
  AffineExpr s1 = b.getAffineSymbolExpr(1);
  AffineExpr s2 = b.getAffineSymbolExpr(2);
  auto apply = [&](AffineExpr expr, ValueRange values) -> Value {
    return makeComposedAffineApply(b, b.getLoc(), expr, values);
  };
  Value a = apply(s0.floorDiv(s1), {tripCount, divisorValue});
  Value t = apply((s0 + s1 - 1).floorDiv(s1), {targetSizeValue, divisorValue});
  Value d = apply((s0 + s1 - 1).floorDiv(s1), {a, t});
  Value s = apply(s0.floorDiv(s1) * s2, {a, d, divisorValue});
  Value v = apply(s0 % s1, {a, d});
  Value u = apply(s0 - s1, {d, v});

  MultiSizeSpecification spec;
  spec.lowTileSize = s;
  spec.highTileSize = apply(s0 + s1, {s, divisorValue});
  spec.lowTripCount = u;
  spec.highTripCount = v;

  // If requested, emit the check that the tile sizes are computed correctly.
  // For example, for iteration dimension size of 15 and the target size 8 it is
  // impossible to find two tile sizes both divisible by 8 that fully cover the
  // original space dimension.
  if (emitAssertions) {
    AffineExpr s3 = builder.getAffineSymbolExpr(3);
    Value coveredSize =
        apply(s0 * s1 + s2 * s3, {spec.lowTileSize, spec.lowTripCount,
                                  spec.highTileSize, spec.highTripCount});
    Value equals = b.create<arith::CmpIOp>(arith::CmpIPredicate::eq,
                                           coveredSize, tripCount);
    b.create<cf::AssertOp>(
        equals, builder.getStringAttr(
                    "could not compute dynamic multi-size tile shapes"));
  }

  return spec;
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

/// Build an `affine_max` of all the `vals`.
static OpFoldResult buildMax(OpBuilder &b, Location loc,
                             ArrayRef<OpFoldResult> vals) {
  return makeComposedFoldedAffineMax(
      b, loc, AffineMap::getMultiDimIdentityMap(vals.size(), loc.getContext()),
      vals);
}

/// Build an `affine_min` of all the `vals`.
static OpFoldResult buildMin(OpBuilder &b, Location loc,
                             ArrayRef<OpFoldResult> vals) {
  return makeComposedFoldedAffineMin(
      b, loc, AffineMap::getMultiDimIdentityMap(vals.size(), loc.getContext()),
      vals);
}

/// Fill out the `tiledOffsets` and `tiledSizes` to be used to tile to a given
/// number of threads.
static void calculateTileOffsetsAndSizes(
    RewriterBase &b, Location loc, scf::ForeachThreadOp foreachThreadOp,
    ArrayRef<OpFoldResult> numThreads, SmallVector<Range> loopRanges,
    bool omitTileOffsetBoundsCheck,
    std::optional<ArrayRef<OpFoldResult>> nominalTileSizes,
    SmallVector<OpFoldResult> &tiledOffsets,
    SmallVector<OpFoldResult> &tiledSizes) {
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPointToStart(foreachThreadOp.getBody(0));

  ValueRange threadIds = foreachThreadOp.getThreadIndices();
  SmallVector<OpFoldResult> nonZeroNumThreads =
      llvm::to_vector(llvm::make_filter_range(numThreads, [](OpFoldResult ofr) {
        return !isConstantIntValue(ofr, 0);
      }));
  int64_t nLoops = loopRanges.size();
  tiledOffsets.reserve(nLoops);
  tiledSizes.reserve(nLoops);
  for (unsigned loopIdx = 0, threadIdIdx = 0; loopIdx < nLoops; ++loopIdx) {
    bool overflow = loopIdx >= numThreads.size();
    bool isZero = !overflow && isConstantIntValue(numThreads[loopIdx], 0);
    // Degenerate case: take the whole domain.
    if (overflow || isZero) {
      tiledOffsets.push_back(loopRanges[loopIdx].offset);
      tiledSizes.push_back(loopRanges[loopIdx].size);
      continue;
    }

    // Tiled case: compute the offset and size.
    AffineExpr i, j, m, n, o;
    bindDims(b.getContext(), i, j);
    bindSymbols(b.getContext(), m, n, o);
    OpFoldResult size = loopRanges[loopIdx].size;
    OpFoldResult offset = loopRanges[loopIdx].offset;
    OpFoldResult threadId = threadIds[threadIdIdx];
    // Symbolic fixed max size per thread.
    // TODO: floor + 0/1 depending on case for better load-balancing.
    OpFoldResult tileSizePerThread =
        nominalTileSizes.has_value()
            ? (*nominalTileSizes)[loopIdx]
            : makeComposedFoldedAffineApply(
                  b, loc, m.ceilDiv(n),
                  ArrayRef<OpFoldResult>{size, nonZeroNumThreads[threadIdIdx]});

    // Dynamic offset shifted by threadId * maxSizePerThread.
    OpFoldResult offsetPerThread = makeComposedFoldedAffineApply(
        b, loc, i + j * m, {offset, threadId, tileSizePerThread});
    // Dynamic upper-bound depending on the threadId.
    OpFoldResult residualTileSize = makeComposedFoldedAffineApply(
        b, loc, i + j * m - n,
        {offset, nonZeroNumThreads[threadIdIdx], tileSizePerThread, size});
    if (!isConstantIntValue(residualTileSize, 0)) {
      OpFoldResult sizeMinusOffsetPerThread = makeComposedFoldedAffineApply(
          b, loc, -i + m, {offsetPerThread, size});
      tileSizePerThread =
          buildMin(b, loc, {sizeMinusOffsetPerThread, tileSizePerThread});
    }

    tiledOffsets.push_back(offsetPerThread);
    // TODO: if tileSizePerThread <= 0 early exit.
    if (!omitTileOffsetBoundsCheck &&
        !canOmitTileOffsetInBoundsCheck(tileSizePerThread,
                                        nonZeroNumThreads[threadIdIdx], size))
      tileSizePerThread =
          buildMax(b, loc, {b.getIndexAttr(0), tileSizePerThread});

    tiledSizes.push_back(tileSizePerThread);
    ++threadIdIdx;
  }
}

/// Rewrite a TilingInterface `op` to a tiled `scf.foreach_thread`. The
/// tiling is specified by the number of tiles/threads `numThreads` and the
/// optional nominal tile size `nominalTileSizes`. If `nominalTilSizes` is
/// not specified, then  it is derived from `numThreads` as `ceilDiv(dimSize[i],
/// numThreads[i])`. If non-empty, the `mapping` is added as an
/// attribute to the resulting `scf.foreach_thread`. A zero tile sizes indicate
/// that the dimension is not tiled, and can be thought of as tiling by the full
/// size of data.
/// It is the user's responsibility to ensure that `numThreads` is a valid
/// tiling specification (i.e. that only tiles parallel dimensions, e.g. in the
/// Linalg case). If `omitTileOffsetBoundsCheck` is true, then the function will
/// assume that `tileSize[i] * (numThread[i] -1) <= dimSize[i]` holds.
static FailureOr<ForeachThreadTilingResult> tileToForeachThreadOpImpl(
    RewriterBase &b, TilingInterface op, ArrayRef<OpFoldResult> numThreads,
    std::optional<ArrayRef<OpFoldResult>> nominalTileSizes,
    std::optional<ArrayAttr> mapping, bool omitTileOffsetBoundsCheck) {
  Location loc = op->getLoc();
  OpBuilder::InsertionGuard g(b);

  SmallVector<Range> loopRanges = op.getIterationDomain(b);
  if (loopRanges.empty())
    return op->emitOpError("expected non-empty loop ranges");
  auto hasStrideOne = [](Range r) { return !isConstantIntValue(r.stride, 1); };
  if (llvm::any_of(loopRanges, hasStrideOne))
    return op->emitOpError("only stride-1 supported atm");

  // Gather destination tensors.
  SmallVector<Value> dest;
  if (failed(tensor::getOrCreateDestinations(b, loc, op, dest)))
    return op->emitOpError("failed to get destination tensors");

  SmallVector<OpFoldResult> nonZeroNumThreads =
      llvm::to_vector(llvm::make_filter_range(numThreads, [](OpFoldResult ofr) {
        return !isConstantIntValue(ofr, 0);
      }));
  SmallVector<Value> materializedNonZeroNumThreads =
      llvm::to_vector(llvm::map_range(nonZeroNumThreads, [&](OpFoldResult ofr) {
        return getValueOrCreateConstantIndexOp(b, loc, ofr);
      }));

  Operation *tiledOp = nullptr;

  // 1. Create the ForeachThreadOp. We don't use the lambda body-builder
  // version because we require the use of RewriterBase in the body, so we
  // manually move the insertion point to the body below.
  scf::ForeachThreadOp foreachThreadOp = b.create<scf::ForeachThreadOp>(
      loc, dest, ValueRange(materializedNonZeroNumThreads), mapping);

  // 2. Fill out the ForeachThreadOp body.
  SmallVector<OpFoldResult> tiledOffsets, tiledSizes;
  calculateTileOffsetsAndSizes(b, loc, foreachThreadOp, numThreads, loopRanges,
                               omitTileOffsetBoundsCheck, nominalTileSizes,
                               tiledOffsets, tiledSizes);

  // 3. Clone the tileable op and update its destination operands to use the
  // output bbArgs of the ForeachThreadOp.
  ArrayRef<BlockArgument> destBbArgs =
      foreachThreadOp.getOutputBlockArguments();
  {
    // 3.a. RAII guard, inserting within foreachThreadOp, before terminator.
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPoint(foreachThreadOp.getTerminator());
    Operation *clonedOp = b.clone(*op.getOperation());
    auto destinationStyleOp = dyn_cast<DestinationStyleOpInterface>(clonedOp);
    if (destinationStyleOp) {
      for (OpOperand *outOperand : destinationStyleOp.getDpsInitOperands()) {
        auto *it = llvm::find(dest, outOperand->get());
        if (it == dest.end())
          return op->emitOpError("must have \"tensor semantic\" for tiling");
        unsigned destNum = std::distance(dest.begin(), it);
        outOperand->set(destBbArgs[destNum]);
      }
    }

    // 4. Tile the cloned op and delete the clone.
    SmallVector<Operation *> tiledOps =
        cast<TilingInterface>(clonedOp).getTiledImplementation(b, tiledOffsets,
                                                               tiledSizes);
    b.eraseOp(clonedOp);
    assert(tiledOps.size() == 1 && "expected a single produced tiled op");
    tiledOp = tiledOps.front();
  }

  // 5. Parallel insert back into the result tensor.
  auto tilingInterfaceOp = dyn_cast<TilingInterface>(tiledOp);
  assert(tilingInterfaceOp && "Tiled op does not implement TilingInterface");
  for (auto it : llvm::zip(llvm::seq(unsigned(0), unsigned(dest.size())),
                           tilingInterfaceOp->getResults(), destBbArgs)) {
    // 5.a. Partial subset information is inserted just before the terminator.
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPoint(foreachThreadOp.getTerminator());

    SmallVector<OpFoldResult> resultOffsets, resultSizes;
    if (failed(op.getResultTilePosition(b, std::get<0>(it), tiledOffsets,
                                        tiledSizes, resultOffsets,
                                        resultSizes)))
      return op->emitOpError("output offsets couldn't be calculated");
    SmallVector<OpFoldResult> strides(resultSizes.size(), b.getIndexAttr(1));

    // 5.b. Parallel insertions are inserted at the end of the combining
    // terminator.
    b.setInsertionPointToEnd(foreachThreadOp.getTerminator().getBody());
    b.create<tensor::ParallelInsertSliceOp>(loc, std::get<1>(it),
                                            std::get<2>(it), resultOffsets,
                                            resultSizes, strides);
  }
  return ForeachThreadTilingResult{foreachThreadOp, tiledOp};
}

FailureOr<ForeachThreadTilingResult>
linalg::tileToForeachThreadOp(RewriterBase &b, TilingInterface op,
                              ArrayRef<OpFoldResult> numThreads,
                              std::optional<ArrayAttr> mapping) {
  return tileToForeachThreadOpImpl(b, op, numThreads,
                                   /*nominalTileSizes=*/std::nullopt, mapping,
                                   /*omitTileOffsetBoundsCheck=*/false);
}

FailureOr<ForeachThreadTilingResult>
linalg::tileToForeachThreadOpUsingTileSizes(RewriterBase &b, TilingInterface op,
                                            ArrayRef<OpFoldResult> tileSizes,
                                            std::optional<ArrayAttr> mapping) {
  SmallVector<Range> loopRanges = op.getIterationDomain(b);
  unsigned nLoops = loopRanges.size();
  SmallVector<OpFoldResult> numThreads;
  numThreads.reserve(nLoops);
  AffineExpr s0, s1;
  bindSymbols(b.getContext(), s0, s1);
  AffineExpr divExpr = s0.ceilDiv(s1);
  for (const auto &it : llvm::zip(tileSizes, loopRanges)) {
    OpFoldResult numTiles = std::get<0>(it);
    if (!isConstantIntValue(numTiles, 0))
      numTiles = makeComposedFoldedAffineApply(
          b, op.getLoc(), divExpr, {std::get<1>(it).size, std::get<0>(it)});
    numThreads.push_back(numTiles);
  }
  return tileToForeachThreadOpImpl(b, op, numThreads,
                                   /*nominalTileSizes=*/tileSizes, mapping,
                                   /*omitTileOffsetBoundsCheck=*/true);
}

template <typename LoopTy>
static FailureOr<TiledLinalgOp>
tileLinalgOpImpl(RewriterBase &b, LinalgOp op, ArrayRef<OpFoldResult> tileSizes,
                 const LinalgTilingOptions &options) {
  OpBuilder::InsertionGuard g(b);

  auto nLoops = op.getNumLoops();
  // Initial tile sizes may be too big, only take the first nLoops.
  tileSizes = tileSizes.take_front(nLoops);

  if (llvm::all_of(tileSizes, isZero)) {
    TiledLinalgOp tiledOp;
    tiledOp.op = cast<LinalgOp>(b.clone(*op.getOperation()));
    tiledOp.tensorResults.assign(tiledOp.op->result_begin(),
                                 tiledOp.op->result_end());
    return tiledOp;
  }

  // 1. Build the tiled loop ranges.
  SmallVector<OpFoldResult> allShapeSizes =
      op.createFlatListOfOperandDims(b, op.getLoc());
  AffineMap shapeSizesToLoopsMap = op.getShapesToLoopsMap();
  if (!shapeSizesToLoopsMap)
    return failure();

  auto [loopRanges, loopIndexToRangeIndex] = makeTiledLoopRanges(
      b, op.getLoc(), shapeSizesToLoopsMap, allShapeSizes, tileSizes);

  SmallVector<utils::IteratorType, 4> iteratorTypes;
  for (const auto &attr : enumerate(op.getIteratorTypesArray())) {
    if (loopIndexToRangeIndex.count(attr.index()))
      iteratorTypes.push_back(attr.value());
  }
  // If interchangeVector is empty, use the identity. Build the permutation map
  // otherwise.
  auto invPermutationMap =
      AffineMap::getMultiDimIdentityMap(tileSizes.size(), b.getContext());
  if (!options.interchangeVector.empty()) {
    // Based on the pruned iterations (due to zero tile size), recompute the
    // interchange vector.
    SmallVector<unsigned, 4> interchangeVector;
    interchangeVector.reserve(options.interchangeVector.size());
    for (auto pos : options.interchangeVector) {
      auto it = loopIndexToRangeIndex.find(pos);
      if (it == loopIndexToRangeIndex.end())
        continue;
      interchangeVector.push_back(it->second);
    }
    // Interchange vector is guaranteed to be a permutation,
    // `inversePermutation` must succeed.
    invPermutationMap = inversePermutation(
        AffineMap::getPermutationMap(interchangeVector, b.getContext()));
    assert(invPermutationMap);
    SmallVector<int64_t> permutation(interchangeVector.begin(),
                                     interchangeVector.end());
    applyPermutationToVector(loopRanges, permutation);
    applyPermutationToVector(iteratorTypes, permutation);
  }

  // Handle distribution. Create a vector of the same size of loops that are to
  // be tiled.
  SmallVector<linalg::ProcInfo> procInfo;
  if (options.distribution) {
    procInfo.resize(
        iteratorTypes.size(),
        linalg::ProcInfo{nullptr, nullptr, linalg::DistributionMethod::None});
    // Collect loop ranges of tiled loopss, loops that are parallel.
    SmallVector<Range> parallelLoopRanges;
    for (const auto &iteratorType : llvm::enumerate(iteratorTypes)) {
      if (!isParallelIterator(iteratorType.value()))
        break;
      parallelLoopRanges.push_back(loopRanges[iteratorType.index()]);
    }
    auto returnedProcInfo =
        options.distribution->procInfo(b, op.getLoc(), parallelLoopRanges);
    unsigned procIdIdx = 0;
    // Update the distribution information for the loops.
    for (const auto &iteratorType : llvm::enumerate(iteratorTypes)) {
      if (!isParallelIterator(iteratorType.value()))
        break;
      procInfo[iteratorType.index()] = returnedProcInfo[procIdIdx++];
    }
  }

  // 2. Create the tiled loops.
  LinalgOp res = op;
  SmallVector<Value, 4> ivs, tensorResults;
  auto tiledLoopBodyBuilder =
      [&](OpBuilder &builder, Location loc, ValueRange localIvs,
          ValueRange operandValuesToUse) -> scf::ValueVector {
    ivs.assign(localIvs.begin(), localIvs.end());

    // When an `interchangeVector` is present, it has been applied to the
    // loop ranges and the iterator types. Apply its inverse to the
    // resulting loop `ivs` to match the op definition.
    SmallVector<Value, 4> interchangedIvs;
    if (!options.interchangeVector.empty())
      interchangedIvs = applyMapToValues(b, loc, invPermutationMap, ivs);
    else
      interchangedIvs.assign(ivs.begin(), ivs.end());

    // Tile the `operandValuesToUse` that either match the `op` operands
    // themselves or the tile loop arguments forwarding them.
    assert(operandValuesToUse.size() ==
               static_cast<size_t>(op->getNumOperands()) &&
           "expect the number of operands and inputs and outputs to match");
    SmallVector<Value> valuesToTile = operandValuesToUse;
    SmallVector<OpFoldResult> sizeBounds =
        makeComposedFoldedMultiResultAffineApply(b, loc, shapeSizesToLoopsMap,
                                                 allShapeSizes);
    SmallVector<Value> tiledOperands = makeTiledShapes(
        b, loc, op, valuesToTile, getAsOpFoldResult(interchangedIvs), tileSizes,
        sizeBounds,
        /*omitPartialTileCheck=*/false);

    SmallVector<Type> resultTensorTypes =
        getTensorOutputTypes(op, tiledOperands);
    res = clone(b, op, resultTensorTypes, tiledOperands);
    tensorResults =
        insertSlicesBack(builder, loc, op, tiledOperands, res->getResults());
    return scf::ValueVector(tensorResults.begin(), tensorResults.end());
  };
  GenerateLoopNest<LoopTy>::doit(b, op.getLoc(), loopRanges, op, iteratorTypes,
                                 tiledLoopBodyBuilder, procInfo);

  // 3. Transform IndexOp results w.r.t. the tiling.
  transformIndexOps(b, res, ivs, loopIndexToRangeIndex);

  // 4. Gather the newly created loops and return them with the new op.
  SmallVector<Operation *, 8> loops;
  loops.reserve(ivs.size());
  for (auto iv : ivs) {
    if (iv.isa<BlockArgument>()) {
      loops.push_back(iv.cast<BlockArgument>().getOwner()->getParentOp());
      assert(loops.back() && "no owner found for induction variable!");
    } else {
      // TODO: Instead of doing this, try to recover the ops used instead of the
      // loop.
      loops.push_back(nullptr);
    }
  }

  // 5. Get the tensor results from the outermost loop if available. Otherwise
  // use the previously captured `tensorResults`.
  Operation *outermostLoop = nullptr;
  for (Operation *loop : loops)
    if ((outermostLoop = loop))
      break;

  return TiledLinalgOp{
      res, loops, outermostLoop ? outermostLoop->getResults() : tensorResults};
}

FailureOr<linalg::ForeachThreadReductionTilingResult>
linalg::tileReductionUsingForeachThread(RewriterBase &b,
                                        PartialReductionOpInterface op,
                                        ArrayRef<OpFoldResult> numThreads,
                                        ArrayRef<OpFoldResult> tileSizes,
                                        std::optional<ArrayAttr> mapping) {
  Location loc = op.getLoc();
  OpBuilder::InsertionGuard g(b);

  // Ops implementing PartialReductionOpInterface are expected to implement
  // TilingInterface.
  // TODO: proper core mechanism to tie interfaces together.
  auto tilingInterfaceOp = cast<TilingInterface>(op.getOperation());

  // Ops implementing PartialReductionOpInterface are not necessarily expected
  // to implement TilingInterface.. This cast is unsafe atm.
  // TODO: proper core mechanism to tie interfaces together.
  // TODO: this function requires a pair of interfaces ..
  auto destinationStyleOp =
      dyn_cast<DestinationStyleOpInterface>(op.getOperation());
  if (!destinationStyleOp)
    return b.notifyMatchFailure(op, "not a destination style op");

  // Actually this only work for Linalg ops atm.
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op.getOperation());
  if (!linalgOp)
    return b.notifyMatchFailure(op, "not a linalg op");

  SmallVector<Range> iterationDomain = tilingInterfaceOp.getIterationDomain(b);
  if (op->getNumResults() != 1)
    return b.notifyMatchFailure(
        op, "don't support ops with multiple results for now");

  SmallVector<utils::IteratorType> iterators =
      tilingInterfaceOp.getLoopIteratorTypes();
  SmallVector<unsigned> redDims;
  linalgOp.getReductionDims(redDims);
  if (redDims.size() != 1)
    return b.notifyMatchFailure(
        op, "only support ops with one reduction dimension.");
  if (!tileSizes.empty() && tileSizes.size() != numThreads.size())
    return b.notifyMatchFailure(op, "if tile sizes are present it must have as "
                                    "many elements as number of threads");
  int reductionDim = static_cast<int>(redDims.front());

  if (redDims.front() >= numThreads.size())
    return b.notifyMatchFailure(
        op, "reduction dimension must be mapped to threads");

  // 1. Create the inital tensor value.
  FailureOr<Operation *> identityTensor =
      op.generateInitialTensorForPartialReduction(b, loc, numThreads,
                                                  reductionDim);
  if (failed(identityTensor))
    return b.notifyMatchFailure(op,
                                "cannot create a tensor of identity value.");

  // Gather destination tensors.
  SmallVector<Value> dest;
  if (failed(tensor::getOrCreateDestinations(b, loc, op, dest)))
    return b.notifyMatchFailure(op, "failed to get destination tensors");

  Operation *tiledOp = nullptr;

  SmallVector<OpFoldResult> nonZeroNumThreads =
      llvm::to_vector(llvm::make_filter_range(numThreads, [](OpFoldResult ofr) {
        return !isConstantIntValue(ofr, 0);
      }));
  SmallVector<Value> materializedNonZeroNumThreads =
      getAsValues(b, loc, nonZeroNumThreads);

  // 2. Create the ForeachThreadOp with an empty region.
  scf::ForeachThreadOp foreachThreadOp = b.create<scf::ForeachThreadOp>(
      loc, (*identityTensor)->getResults(),
      ValueRange(materializedNonZeroNumThreads), mapping);

  // 3. Calculate the tile offsets and sizes for the subsequent loop that will
  // be nested under `foreachThreadOp`.
  SmallVector<OpFoldResult> tiledOffsets, tiledSizes;
  calculateTileOffsetsAndSizes(
      b, loc, foreachThreadOp, numThreads, iterationDomain,
      /*omitTileOffsetBoundsCheck =*/false,
      /*nominalTileSizes=*/std::nullopt, tiledOffsets, tiledSizes);

  // 4. Clone the tileable op and update its destination operands to use the
  // output bbArgs of the ForeachThreadOp.
  ValueRange tilingResults;
  ArrayRef<BlockArgument> destBbArgs =
      foreachThreadOp.getOutputBlockArguments();
  {
    // 4.a. RAII guard, inserting within foreachThreadOp, before terminator.
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPoint(foreachThreadOp.getTerminator());

    SmallVector<Value> tiledDpsInitOperands;
    for (OpOperand *initOperand : destinationStyleOp.getDpsInitOperands()) {
      auto *it = llvm::find(dest, initOperand->get());
      assert(it != dest.end() && "dest operand not found in dest");
      unsigned destNum = std::distance(dest.begin(), it);
      SmallVector<OpFoldResult> strides(numThreads.size(), b.getIndexAttr(1));
      SmallVector<OpFoldResult> outOffsets(numThreads.size(),
                                           b.getIndexAttr(0));
      SmallVector<OpFoldResult> sizes = tiledSizes;
      sizes[reductionDim] = b.getIndexAttr(1);
      outOffsets[reductionDim] = foreachThreadOp.getThreadIndices().front();
      // TODO: use SubsetExtractOpInterface once it is available.
      tiledDpsInitOperands.push_back(b.create<tensor::ExtractSliceOp>(
          loc, initOperand->get().getType().cast<RankedTensorType>(),
          destBbArgs[destNum], outOffsets, sizes, strides));
    }

    // 4.b. Clone the op and update init operands.
    // We cannot use a IRMapping here because it can replace
    // different OpOperands with the same value.
    Operation *clonedOp = b.clone(*op.getOperation());
    b.updateRootInPlace(clonedOp, [&]() {
      for (auto [initOperandPtr, tiledInitValue] : llvm::zip_equal(
               cast<DestinationStyleOpInterface>(clonedOp).getDpsInitOperands(),
               tiledDpsInitOperands)) {
        initOperandPtr->set(tiledInitValue);
      }
    });

    // 5. Tile the cloned op and delete the clone.
    if (tileSizes.empty()) {
      SmallVector<Operation *> tiledOps =
          cast<TilingInterface>(clonedOp).getTiledImplementation(
              b, tiledOffsets, tiledSizes);
      assert(tiledOps.size() == 1 && "expected a single produced tiled op");
      tiledOp = tiledOps.front();
      tilingResults = tiledOp->getResults();
    } else {
      LinalgTilingOptions options;
      FailureOr<TiledLinalgOp> maybeTiled = tileLinalgOpImpl<scf::ForOp>(
          b, cast<LinalgOp>(clonedOp), tileSizes, options);
      if (failed(maybeTiled))
        return b.notifyMatchFailure(op, "failed tileLinalgOpImpl");

      SmallVector<Value> ids = foreachThreadOp.getThreadIndices();
      mapLoopToProcessorIds(cast<scf::ForOp>(maybeTiled->loops.back()), ids,
                            materializedNonZeroNumThreads);
      assert(maybeTiled->loops.size() == 1 &&
             "expected a single produced loop");
      tiledOp = maybeTiled->op;
      tilingResults = maybeTiled->loops.front()->getResults();
    }

    b.eraseOp(clonedOp);
  }

  // 6. Insert the partial reductions back into a new tensor.
  for (auto [index, result, bbArg] : llvm::zip(
           llvm::seq<unsigned>(0, dest.size()), tilingResults, destBbArgs)) {
    // 6.a. Partial subset information is inserted just before the terminator.
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPoint(foreachThreadOp.getTerminator());

    SmallVector<OpFoldResult> resultOffsets, resultSizes;
    if (failed(tilingInterfaceOp.getResultTilePosition(
            b, index, tiledOffsets, tiledSizes, resultOffsets, resultSizes)))
      return op->emitOpError("output offsets couldn't be calculated");
    SmallVector<OpFoldResult> resultOffsetsRank, resultSizesRank;
    int64_t offIdx = 0;
    int64_t sizeIdx = 0;
    for (int64_t i = 0, e = numThreads.size(); i < e; ++i) {
      if (i == reductionDim) {
        resultOffsetsRank.push_back(foreachThreadOp.getThreadIndices().front());
        resultSizesRank.push_back(b.getIndexAttr(1));
        continue;
      }
      resultOffsetsRank.push_back(resultOffsets[offIdx++]);
      resultSizesRank.push_back(resultSizes[sizeIdx++]);
    }
    SmallVector<OpFoldResult> strides(resultSizesRank.size(),
                                      b.getIndexAttr(1));

    // 6.b. Parallel insertions are inserted at the end of the combining
    // terminator.
    b.setInsertionPointToEnd(foreachThreadOp.getTerminator().getBody());
    b.create<tensor::ParallelInsertSliceOp>(
        loc, result, bbArg, resultOffsetsRank, resultSizesRank, strides);
  }

  // 7. Merge the partial reductions.
  b.setInsertionPointAfter(foreachThreadOp);
  Operation *mergeOp =
      op.mergeReductions(b, loc, foreachThreadOp->getResults(), reductionDim);
  b.replaceOp(op, mergeOp->getResults());

  // 8. Return.
  ForeachThreadReductionTilingResult results;
  results.initialOp = *identityTensor;
  results.loops = foreachThreadOp;
  results.parallelTiledOp = tiledOp;
  results.mergeOp = mergeOp;
  return results;
}

// Insert a tile `source` into the destination tensor `dest`. The position at
// which the tile is inserted (as well as size of tile) is taken from a given
// ExtractSliceOp `sliceOp`.
static Value insertSliceIntoTensor(OpBuilder &b, Location loc,
                                   tensor::ExtractSliceOp sliceOp, Value source,
                                   Value dest) {
  return b.create<tensor::InsertSliceOp>(
      loc, sliceOp.getSource().getType(), source, dest, sliceOp.getOffsets(),
      sliceOp.getSizes(), sliceOp.getStrides(), sliceOp.getStaticOffsets(),
      sliceOp.getStaticSizes(), sliceOp.getStaticStrides());
}

template <typename LoopTy>
FailureOr<TiledLinalgOp> static tileLinalgOpImpl(
    RewriterBase &b, LinalgOp op, const LinalgTilingOptions &options) {
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(op);

  if (!options.tileSizeComputationFunction)
    return failure();

  // Enforce the convention that "tiling by zero" skips tiling a particular
  // dimension. This convention is significantly simpler to handle instead of
  // adjusting affine maps to account for missing dimensions.
  auto nLoops = op.getNumLoops();
  SmallVector<OpFoldResult> tileSizeVector =
      getAsOpFoldResult(options.tileSizeComputationFunction(b, op));
  if (tileSizeVector.size() < nLoops) {
    tileSizeVector.append(nLoops - tileSizeVector.size(), b.getIndexAttr(0));
  }

  return tileLinalgOpImpl<LoopTy>(b, op, tileSizeVector, options);
}

FailureOr<TiledLinalgOp>
mlir::linalg::tileLinalgOp(RewriterBase &b, LinalgOp op,
                           const LinalgTilingOptions &options) {
  switch (options.loopType) {
  case LinalgTilingLoopType::Loops:
    return tileLinalgOpImpl<scf::ForOp>(b, op, options);
  case LinalgTilingLoopType::ParallelLoops:
    return tileLinalgOpImpl<scf::ParallelOp>(b, op, options);
  default:;
  }
  return failure();
}

/// Generate a loop nest around a given tensor::PadOp (for tiling). `newPadOp`
/// and `loopNest` are output parameters that return the new (tiled)
/// tensor::PadOp and the loop nest.
static LogicalResult tilePadOp(RewriterBase &builder, tensor::PadOp op,
                               tensor::PadOp &newPadOp, LoopNest &loopNest,
                               const LinalgTilingOptions &options) {
  Location loc = op.getLoc();
  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPoint(op);

  // Clone tensor::PadOp so that the existing op can be replaced more easily.
  newPadOp = cast<tensor::PadOp>(builder.clone(*op.getOperation()));
  // Get rank and tile sizes.
  int64_t rank = op.getResultType().getRank();
  SmallVector<OpFoldResult> tileSizes =
      getAsOpFoldResult(options.tileSizeComputationFunction(builder, op));
  // Normalize untiled padding dimensions to 0.
  tileSizes.append(rank - tileSizes.size(), builder.getIndexAttr(0));
  // Compute lower and upper bounds of the loop nest.
  TilingInterface tilingInterface =
      dyn_cast<TilingInterface>(op.getOperation());
  SmallVector<Range> ranges = tilingInterface.getIterationDomain(builder);
  SmallVector<Value> lbs, dims, steps;
  SmallVector<OpFoldResult> allDims;
  for (int64_t i = 0; i < rank; ++i) {
    allDims.push_back(ranges[i].size);
    if (!isZero(tileSizes[i])) {
      lbs.push_back(
          getValueOrCreateConstantIndexOp(builder, loc, ranges[i].offset));
      dims.push_back(
          getValueOrCreateConstantIndexOp(builder, loc, ranges[i].size));
      steps.push_back(
          getValueOrCreateConstantIndexOp(builder, loc, tileSizes[i]));
    }
  }
  SmallVector<Value> destinationTensors;
  if (failed(tensor::getOrCreateDestinations(builder, loc, tilingInterface,
                                             destinationTensors)))
    return failure();

  loopNest = mlir::scf::buildLoopNest(
      builder, loc, lbs, /*ubs=*/dims, steps, ValueRange(destinationTensors),
      [&](OpBuilder &b, Location loc, ValueRange localIvs,
          ValueRange iterArgs) -> scf::ValueVector {
        // Compute offsets and sizes of ExtractSliceOp.
        SmallVector<Value> localIVVector = llvm::to_vector(localIvs);
        SmallVector<OpFoldResult> offsets = computeTileOffsets(
            b, loc, getAsOpFoldResult(localIVVector), tileSizes);
        SmallVector<OpFoldResult> sizes =
            computeTileSizes(b, loc, tileSizes, allDims);
        // Create ExtractSliceOp: Extract a tile from the tensor::PadOp.
        // Note: The tensor::PadOp is located outside of the loop nest. It is
        // later moved inside by ExtractSliceOfPadTensorSwapPattern.
        auto map = AffineMap::getMultiDimIdentityMap(rank, b.getContext());
        Value tiledOutput = makeTiledShape(
            b, loc, newPadOp->getResult(0), tileSizes, map, offsets, allDims,
            sizes, /*omitPartialTileCheck=*/false);
        auto sliceOp = tiledOutput.getDefiningOp<tensor::ExtractSliceOp>();
        assert(sliceOp && "expected ExtractSliceOp");
        // Insert the tile into the output tensor.
        Value yieldValue =
            insertSliceIntoTensor(b, loc, sliceOp, sliceOp, iterArgs[0]);
        return scf::ValueVector({yieldValue});
      });
  return success();
}

namespace {
struct PadOpTilingPattern : public OpRewritePattern<tensor::PadOp> {
  PadOpTilingPattern(MLIRContext *ctx, LinalgTilingOptions opt)
      : OpRewritePattern<tensor::PadOp>(ctx), options(std::move(opt)) {}

  LogicalResult matchAndRewrite(tensor::PadOp op,
                                PatternRewriter &rewriter) const override {
    tensor::PadOp newPadOp;
    LoopNest loopNest;
    if (failed(tilePadOp(rewriter, op, newPadOp, loopNest, options)))
      return failure();
    // Replace all uses of the original tensor::PadOp.
    rewriter.replaceOp(op, loopNest.results.front());
    return success();
  }

  LinalgTilingOptions options;
};
} // namespace

namespace {
/// Helper classes for type list expansion.
template <typename... OpTypes>
class CanonicalizationPatternList;

template <>
class CanonicalizationPatternList<> {
public:
  static void insert(RewritePatternSet &patterns) {}
};

template <typename OpTy, typename... OpTypes>
class CanonicalizationPatternList<OpTy, OpTypes...> {
public:
  static void insert(RewritePatternSet &patterns) {
    OpTy::getCanonicalizationPatterns(patterns, patterns.getContext());
    CanonicalizationPatternList<OpTypes...>::insert(patterns);
  }
};
} // namespace

RewritePatternSet
mlir::linalg::getLinalgTilingCanonicalizationPatterns(MLIRContext *ctx) {
  RewritePatternSet patterns(ctx);
  populateLinalgTilingCanonicalizationPatterns(patterns);
  return patterns;
}

void mlir::linalg::populateLinalgTilingCanonicalizationPatterns(
    RewritePatternSet &patterns) {
  auto *ctx = patterns.getContext();
  AffineApplyOp::getCanonicalizationPatterns(patterns, ctx);
  AffineForOp::getCanonicalizationPatterns(patterns, ctx);
  AffineMinOp::getCanonicalizationPatterns(patterns, ctx);
  AffineMaxOp::getCanonicalizationPatterns(patterns, ctx);
  arith::ConstantIndexOp::getCanonicalizationPatterns(patterns, ctx);

  memref::SubViewOp::getCanonicalizationPatterns(patterns, ctx);
  memref::ViewOp::getCanonicalizationPatterns(patterns, ctx);

  scf::ForOp::getCanonicalizationPatterns(patterns, ctx);
  scf::ParallelOp::getCanonicalizationPatterns(patterns, ctx);

  tensor::CastOp::getCanonicalizationPatterns(patterns, ctx);
  tensor::EmptyOp::getCanonicalizationPatterns(patterns, ctx);
  tensor::ExtractSliceOp::getCanonicalizationPatterns(patterns, ctx);
  tensor::InsertSliceOp::getCanonicalizationPatterns(patterns, ctx);
  tensor::PadOp::getCanonicalizationPatterns(patterns, ctx);
  ctx->getLoadedDialect<LinalgDialect>()->getCanonicalizationPatterns(patterns);

  CanonicalizationPatternList<
#define GET_OP_LIST
#include "mlir/Dialect/Linalg/IR/LinalgStructuredOps.cpp.inc"
      >::insert(patterns);
}

void mlir::linalg::populatePadTensorTilingPatterns(
    RewritePatternSet &patterns, const LinalgTilingOptions &options) {
  auto *ctx = patterns.getContext();
  patterns.add<PadOpTilingPattern>(ctx, options);
}
