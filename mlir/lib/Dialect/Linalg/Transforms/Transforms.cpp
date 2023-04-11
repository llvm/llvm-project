//===- Transforms.cpp - Linalg transformations as patterns ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements logic and helpers to expose Linalg transforms as rewrite
// patterns.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/IR/TensorTilingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <type_traits>
#include <utility>

#define DEBUG_TYPE "linalg-transforms"

using namespace mlir;
using namespace mlir::linalg;

#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")

/// Pad the `opOperand` in the `paddingDimensions` using the padding value and
/// the nofold flag found in `paddingValues` and `packPaddings`, respectively.
/// Exit early and return the `opOperand` value if the shape dimensions that
/// match `paddingDimensions` have a static size and the nofold flag is not set.
/// Otherwise, try to pad the shape dimensions that match the iterator
/// dimensions `paddingDimensions` and return the tensor::PadOp result if
/// padding succeeds or failure otherwise.
static FailureOr<Value> padOperandToSmallestStaticBoundingBox(
    RewriterBase &rewriter, linalg::LinalgOp opToPad, OpOperand *opOperand,
    ArrayRef<int64_t> paddingDimensions, ArrayRef<Attribute> paddingValues,
    ArrayRef<bool> packPaddings) {
  AffineMap indexingMap = opToPad.getMatchingIndexingMap(opOperand);
  ArrayRef<int64_t> shape = opToPad.getShape(opOperand);

  // Collect the shape dimension that are a function of the `paddingDimensions`.
  llvm::SmallDenseSet<int64_t> shapeDimsToPad;
  for (int64_t dim : paddingDimensions)
    for (const auto &en : enumerate(indexingMap.getResults()))
      if (en.value().isFunctionOfDim(dim))
        shapeDimsToPad.insert(en.index());

  // Return the unpadded operand if padding to a static shape is not needed and
  // if the nofold flag is not set.
  bool nofold = opOperand->getOperandNumber() < packPaddings.size()
                    ? packPaddings[opOperand->getOperandNumber()]
                    : false;
  bool hasStaticShape = llvm::none_of(shapeDimsToPad, [&](int64_t dim) {
    return ShapedType::isDynamic(shape[dim]);
  });
  if (!nofold && hasStaticShape)
    return opOperand->get();

  // Fail if `paddingValues` specifies no padding value.
  if (opOperand->getOperandNumber() >= paddingValues.size()) {
    return rewriter.notifyMatchFailure(opToPad, "--no padding value specified");
  }
  Attribute paddingAttr = paddingValues[opOperand->getOperandNumber()];
  Type paddingType = rewriter.getType<NoneType>();
  if (auto typedAttr = paddingAttr.dyn_cast<TypedAttr>())
    paddingType = typedAttr.getType();
  Value paddingValue = rewriter.create<arith::ConstantOp>(
      opToPad.getLoc(), paddingType, paddingAttr);

  // Follow the use-def chain if `currOpOperand` is defined by a LinalgOp.
  OpOperand *currOpOperand = opOperand;
  while (auto linalgOp = currOpOperand->get().getDefiningOp<LinalgOp>()) {
    OpResult result = currOpOperand->get().cast<OpResult>();
    currOpOperand = linalgOp.getDpsInitOperand(result.getResultNumber());
  }

  // Fail if `currOpOperand` is not defined by an ExtractSliceOp or EmptyOp.
  auto sliceOp = currOpOperand->get().getDefiningOp<tensor::ExtractSliceOp>();
  auto emptyOp = currOpOperand->get().getDefiningOp<tensor::EmptyOp>();
  if (!sliceOp && !emptyOp) {
    // TODO: may want to add support for going through loop iter args.
    // This is not strictly necessary as we can pad before hoisting but it would
    // make the system more resilient to minor transformation reordering.
    LLVM_DEBUG(DBGS() << "--not defined by an extractSlice or emptyOp\n");
    return rewriter.notifyMatchFailure(
        opToPad, "not defined by an extractSlice or emptyOp");
  }

  llvm::SmallBitVector droppedDims;
  SmallVector<OpFoldResult> mixedSizes;
  if (sliceOp) {
    // Compute the dropped dimensions if `sliceOp` is rank-reducing.
    droppedDims = sliceOp.getDroppedDims();
    mixedSizes = sliceOp.getMixedSizes();
  }
  if (emptyOp) {
    mixedSizes = emptyOp.getMixedSizes();
    droppedDims.resize(mixedSizes.size());
  }
  LLVM_DEBUG(llvm::interleaveComma(mixedSizes, DBGS() << "--mixedSizes:  ");
             llvm::dbgs() << "\n");

  // Upper bound the sizes to obtain a static bounding box.
  SmallVector<int64_t> paddedShape(shape.begin(), shape.end());
  int64_t shapeIdx = 0;
  for (const auto &en : enumerate(mixedSizes)) {
    LLVM_DEBUG(DBGS() << "----mixedSizes:  " << en.value() << "\n");
    // Skip dropped dimensions.
    if (droppedDims.test(en.index())) {
      LLVM_DEBUG(DBGS() << "------dim is dropped, SKIP\n");
      continue;
    }
    // Skip dimensions that do not require padding.
    if (!shapeDimsToPad.contains(shapeIdx)) {
      shapeIdx++;
      LLVM_DEBUG(DBGS() << "------dim does not require padding, SKIP\n");
      continue;
    }
    // If the size is an attribute add it directly to `paddedShape`.
    if (en.value().is<Attribute>()) {
      paddedShape[shapeIdx++] =
          en.value().get<Attribute>().dyn_cast<IntegerAttr>().getInt();
      LLVM_DEBUG(
          DBGS() << "------dim is an attr, add it to padded shape, SKIP\n");
      continue;
    }
    // Otherwise, try to compute a constant upper bound for the size value.
    FailureOr<int64_t> upperBound =
        ValueBoundsConstraintSet::computeConstantBound(
            presburger::BoundType::UB, en.value().get<Value>(),
            /*dim=*/std::nullopt, /*stopCondition=*/nullptr, /*closedUB=*/true);
    if (failed(upperBound)) {
      LLVM_DEBUG(DBGS() << "--count not compute a bounding box for padding");
      return rewriter.notifyMatchFailure(
          opToPad, "count not compute a bounding box for padding");
    }
    paddedShape[shapeIdx++] = *upperBound;
  }
  assert(shapeIdx == static_cast<int64_t>(shape.size()) &&
         "expect the dynamic and static ranks to match");

  // Pad the operand to the bounding box defined by `paddedShape`.
  auto paddedTensorType = RankedTensorType::get(
      paddedShape, getElementTypeOrSelf(opOperand->get()));
  LLVM_DEBUG(DBGS() << "--SUCCESS, makeComposedPadHighOp with type: "
                    << paddedTensorType);
  return makeComposedPadHighOp(rewriter, opToPad->getLoc(), paddedTensorType,
                               opOperand->get(), paddingValue, nofold);
}

static SmallVector<utils::IteratorType>
getNParallelLoopsAttrs(unsigned nParallelLoops) {
  return SmallVector<utils::IteratorType>(nParallelLoops,
                                          utils::IteratorType::parallel);
}

//===----------------------------------------------------------------------===//
// Transformations exposed as functional-style API calls.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// rewriteAsPaddedOp transformation.
//===----------------------------------------------------------------------===//
FailureOr<SmallVector<Value>>
linalg::rewriteAsPaddedOp(RewriterBase &rewriter, LinalgOp opToPad,
                          ArrayRef<int64_t> paddingDimensions,
                          ArrayRef<Attribute> paddingValues,
                          ArrayRef<bool> packPaddings, LinalgOp &paddedOp) {
  LLVM_DEBUG(DBGS() << "Start rewriteAsPaddedOp : " << opToPad << "\n");
  Location loc = opToPad->getLoc();

  // TODO: there are cases where we may still want to pad to larger sizes.
  if (!opToPad.hasTensorSemantics())
    return rewriter.notifyMatchFailure(opToPad,
                                       "expected operation on tensors");

  OpBuilder::InsertionGuard g(rewriter);
  // Set IP after op because we also take the dims of the original output.
  rewriter.setInsertionPointAfter(opToPad);

  // Make a copy of the shaped operands and update it.
  SmallVector<Value> newOperands;
  newOperands.reserve(opToPad->getNumOperands());
  for (OpOperand &opOperand : opToPad->getOpOperands()) {
    FailureOr<Value> paddedOperand = padOperandToSmallestStaticBoundingBox(
        rewriter, opToPad, &opOperand, paddingDimensions, paddingValues,
        packPaddings);
    // Exit if `paddingDimensions` cannot be bounded statically.
    if (failed(paddedOperand)) {
      LLVM_DEBUG(DBGS() << "--operand cannot be bound statically : "
                        << opOperand.get() << " -> FAIL\n");
      return rewriter.notifyMatchFailure(opToPad,
                                         "operand cannot be bound statically");
    }
    newOperands.push_back(*paddedOperand);
  }

  ReifiedRankedShapedTypeDims reifiedResultShapes;
  if (failed(reifyResultShapes(rewriter, opToPad, reifiedResultShapes))) {
    LLVM_DEBUG(DBGS() << "--failed to reify result shapes -> FAIL\n");
    return rewriter.notifyMatchFailure(opToPad,
                                       "failed to reify result shapes");
  }
  assert(reifiedResultShapes.size() == opToPad->getNumResults() &&
         "expected same number of results");

  // Clone `opToPad` to operate on the statically padded shapes.
  auto resultTensorTypes =
      ValueRange(newOperands).take_back(opToPad.getNumDpsInits()).getTypes();
  // clone **should** properly notify the rewriter.
  paddedOp = clone(rewriter, opToPad, resultTensorTypes, newOperands);
  LLVM_DEBUG(DBGS() << "--cloned padded op: " << paddedOp << "\n");

  // Recover the slice out of the new static results. This keeps the original
  // linalg op around because it uses the dims of the original results.
  SmallVector<Value> paddedSubtensorResults;
  paddedSubtensorResults.reserve(opToPad->getNumResults());
  for (const auto &en : llvm::enumerate(paddedOp->getResults())) {
    Value paddedResult = en.value();
    int64_t resultNumber = en.index();
    int64_t rank = paddedResult.getType().cast<RankedTensorType>().getRank();
    SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> sizes;
    SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));
    paddedSubtensorResults.push_back(rewriter.create<tensor::ExtractSliceOp>(
        loc, paddedResult, offsets, reifiedResultShapes[resultNumber],
        strides));
  }
  return paddedSubtensorResults;
}

//===----------------------------------------------------------------------===//
// peelLoop transformation.
//===----------------------------------------------------------------------===//

/// Try to peel and canonicalize loop `op` and return the new result.
/// Also applies affine_min/max bounds simplification on the fly where relevant.
// TODO: Add support for scf.parallel and affine.for loops.
SmallVector<Value> mlir::linalg::peelLoop(RewriterBase &rewriter,
                                          Operation *op) {
  return llvm::TypeSwitch<Operation *, SmallVector<Value, 4>>(op)
      .Case<scf::ForOp>([&](scf::ForOp forOp) {
        scf::ForOp partialIteration;
        if (succeeded(scf::peelForLoopAndSimplifyBounds(rewriter, forOp,
                                                        partialIteration)))
          return partialIteration->getResults();
        assert(!partialIteration && "expected that loop was not peeled");
        return forOp->getResults();
      })
      .Default([&](Operation *op) { return op->getResults(); });
}

/// Peel 'loops' and applies affine_min/max bounds simplification on the fly
/// where relevant.
void mlir::linalg::peelLoops(RewriterBase &rewriter,
                             ArrayRef<scf::ForOp> loops) {
  for (auto loopOp : loops)
    peelLoop(rewriter, loopOp);
}

//===----------------------------------------------------------------------===//
// pad transformation.
//===----------------------------------------------------------------------===//

FailureOr<LinalgOp>
mlir::linalg::padAndHoistLinalgOp(RewriterBase &rewriter, LinalgOp linalgOp,
                                  LinalgPaddingOptions options) {
  if (!linalgOp.hasTensorSemantics())
    return rewriter.notifyMatchFailure(
        linalgOp, "only applies to Linalg ops with tensor semantics");

  // Pad the operation.
  LinalgOp paddedOp;
  FailureOr<SmallVector<Value>> newResults =
      rewriteAsPaddedOp(rewriter, linalgOp, options.paddingDimensions,
                        options.paddingValues, options.packPaddings, paddedOp);
  if (failed(newResults))
    return rewriter.notifyMatchFailure(linalgOp,
                                       "failed to rewrite as a padded op");

  // Hoist the padding.
  for (const auto &en : enumerate(options.hoistPaddings)) {
    if (static_cast<int64_t>(en.index()) >= paddedOp->getNumOperands())
      break;
    OpOperand &opOperand = paddedOp->getOpOperand(en.index());
    auto padOp = opOperand.get().getDefiningOp<tensor::PadOp>();
    if (!padOp || en.value() == 0) {
      (void)rewriter.notifyMatchFailure(linalgOp, "not a tensor.pad -- skip");
      continue;
    }

    // Fail hoisting if the operand shape is not fully static.
    if (llvm::any_of(paddedOp.getShape(&opOperand), ShapedType::isDynamic)) {
      (void)rewriter.notifyMatchFailure(linalgOp,
                                        "non static padding shape -- skip");
      continue;
    }

    tensor::PadOp hoistedOp;
    SmallVector<GenericOp> transposeOps;
    SmallVector<int64_t> transposeVector =
        en.index() < options.transposePaddings.size()
            ? options.transposePaddings[en.index()]
            : SmallVector<int64_t>{};

    FailureOr<Value> newResult = hoistPaddingOnTensors(
        padOp, en.value(), transposeVector, hoistedOp, transposeOps);
    if (failed(newResult)) {
      (void)rewriter.notifyMatchFailure(linalgOp,
                                        "failed to apply hoistPadding");
      continue;
    }
    rewriter.replaceOp(padOp, *newResult);
  }

  // Replace the original operation to pad.
  rewriter.replaceOp(linalgOp, *newResults);

  return paddedOp;
}

//===----------------------------------------------------------------------===//
// pack transformation.
//===----------------------------------------------------------------------===//

#ifndef NDEBUG
/// Return true if `map` has 0 or 1 result function of AffineDimExpr(dim).
static bool hasAtMostOneResultFunctionOfDim(AffineMap map, int64_t dim) {
  bool found = false;
  for (AffineExpr e : map.getResults()) {
    if (!e.isFunctionOfDim(dim))
      continue;
    if (found)
      return false;
    found = true;
  }
  return true;
}
#endif // NDEBUG

/// Return the index of the first result of `map` that is a function of
/// AffineDimExpr(dim), std::nullopt otherwise.
static std::optional<int64_t> getFirstResultIndexFunctionOf(AffineMap map,
                                                            int64_t dim) {
  for (int64_t i = 0, e = map.getNumResults(); i < e; ++i) {
    AffineExpr expr = map.getResult(i);
    if (!expr.isFunctionOfDim(dim))
      continue;
    return i;
  }
  return std::nullopt;
}

/// Perform one step of packing of a LinalgOp's metadata along `dim` into the
/// `newDim` at `iteratorTypes.size()` by:
///   1. Appending `iteratorTypes[newDim]`, equal to `iteratorTypes[dim]`.
///   2. Appending a `newDim` to the domain of every indexing map.
///   3. For each operand (i.e. for each map in `indexingMaps`), perform packing
///      by potentially adding a `newDim` result to `map`.
/// The preserved invariant is that `iteratorTypes.size()` is always equal to
/// `map.getNumDims()` for every map in `indexingMaps`.
///
/// Update `indexingMaps` and `iteratorTypes` inplace as one step of the update.
/// Return a vector that records the optional packing for each operand.
/// Return failure if the packed indexing cannot be represented with a LinalgOp.
///
/// Further details:
/// ================
/// The current implementation of packing (i.e. data tiling) consists of
/// rewriting a linearized strip-mined form into a higher-dimensional access.
/// e.g. consider an access `A[I][f(j, k, l)]` and packing by 4; we rewrite
/// `I` into `4 * i + ii`, where `0 <= ii < 4`.
/// The access is further rewritten as `A[i][f(j, k, l)][ii]`.
///
/// This rewrite into higher dimensional access is not possible for general
/// AffineExpr in Linalg atm, it is restricted to an AffineDimExpr:
/// e.g. consider an access `A[I + J][f(j, k, l)]` and packing by 4; we
/// rewrite `I + J` into `4 * i + ii + J`, where `0 <= ii < 4`.
/// The rewrite of the access would be a form not representable in Linalg:
///   `A[i + (ii + J) / 4][f(j, k, l)][(ii + J) % 4]`.
/// Note however that as `J` and `ii` iterate, the accesses do not have a
/// particular alignment, so packing does not achieve alignment in this case
///
/// In the future, we may want to consider a mixed-form that allows some
/// alignment in the presence of multiple accesses:
///   `A[I][f(j, k, l)]` and `B[I + J][f(j, k, l)]`
/// And would rewrite accesses as:
///   `A[i][f(j, k, l)][ii]` and `B[4 * i + ii + J][f(j, k, l)]`
static FailureOr<SmallVector<std::optional<int64_t>>>
packLinalgMetadataOnce(SmallVectorImpl<AffineMap> &indexingMaps,
                       SmallVectorImpl<utils::IteratorType> &iteratorTypes,
                       int64_t dim) {
  int64_t newDim = iteratorTypes.size();
  iteratorTypes.push_back(iteratorTypes[dim]);

  SmallVector<std::optional<int64_t>> packedDimPerIndexingMap(
      indexingMaps.size(), std::nullopt);
  SmallVector<AffineMap> newMaps;
  for (int64_t operandIdx = 0, e = indexingMaps.size(); operandIdx < e;
       ++operandIdx) {
    AffineMap map = indexingMaps[operandIdx];

    // Add the `newDim` to map whatever the case.
    assert(map.getNumDims() == newDim && "num dims invariant violation");
    map = map.shiftDims(1, newDim);

    // Get the at-most-1 index of the result that is a function of `dim`.
    // If we can find one, we insert `AffineDimExpr(newDim)` to the map, which
    // logically chunks dimension `dim` into `K * dim + newDim`, where the
    // packing factor `K` is specified separately.
    assert(hasAtMostOneResultFunctionOfDim(map, dim) &&
           "num results invariant violation");
    auto maybeOperandDimensionToPack = getFirstResultIndexFunctionOf(map, dim);
    if (!maybeOperandDimensionToPack.has_value()) {
      newMaps.push_back(map);
      continue;
    }

    // We can only pack AffineDimExpr atm.
    if (!map.getResult(maybeOperandDimensionToPack.value())
             .isa<AffineDimExpr>())
      return failure();

    // Add `newDim` to the results of the map.
    map = map.insertResult(Builder(map.getContext()).getAffineDimExpr(newDim),
                           map.getNumResults());
    newMaps.push_back(map);

    // Record the that `operandIdx` is packed.
    packedDimPerIndexingMap[operandIdx] = maybeOperandDimensionToPack;
  }
  indexingMaps = newMaps;

  return packedDimPerIndexingMap;
}

namespace {

/// Helper struct to encode packing along one dimension of a LinalgOp.
struct PackedOperandsDim {
  OpFoldResult packedSize;
  SmallVector<std::optional<int64_t>> packedDimForEachOperand;
};

/// Helper struct to encode packing along all dimensions of a LinalgOp.
struct PackedOperandsDimList {
  void push_back(PackedOperandsDim &&packedOperandsDims) {
    spec.emplace_back(packedOperandsDims);
  }
  /// Return all the dims that have been packed for operand @ `operandPos`.
  SmallVector<int64_t> extractPackedDimsForOperand(int64_t operandPos);
  /// Return all the pack sizes by which an operand @ `operandPos` is packed.
  SmallVector<OpFoldResult> extractPackSizesForOperand(int64_t operandPos);

private:
  SmallVector<PackedOperandsDim> spec;
};

} // namespace

SmallVector<int64_t>
PackedOperandsDimList::extractPackedDimsForOperand(int64_t operandPos) {
  SmallVector<int64_t> res;
  for (int64_t i = 0, e = spec.size(); i < e; ++i) {
    if (!spec[i].packedDimForEachOperand[operandPos].has_value())
      continue;
    res.push_back(spec[i].packedDimForEachOperand[operandPos].value());
  }
  return res;
}

SmallVector<OpFoldResult>
PackedOperandsDimList::extractPackSizesForOperand(int64_t operandPos) {
  SmallVector<OpFoldResult> res;
  for (int64_t i = 0, e = spec.size(); i < e; ++i) {
    if (!spec[i].packedDimForEachOperand[operandPos].has_value())
      continue;
    res.push_back(spec[i].packedSize);
  }
  return res;
}

/// Implement packing of a single LinalgOp by performing packing by
/// `packedSizes`. There must be one packedSizes entry per `linalgOp` iterator.
/// Return the packed Linalg op on success, failure otherwise.
FailureOr<PackResult> linalg::pack(RewriterBase &rewriter,
                                   linalg::LinalgOp linalgOp,
                                   ArrayRef<OpFoldResult> packedSizes) {
  if (packedSizes.size() != linalgOp.getNumLoops()) {
    return rewriter.notifyMatchFailure(linalgOp,
                                       "incorrect number of pack sizes");
  }

  Location loc = linalgOp->getLoc();
  SmallVector<AffineMap> indexingMaps = linalgOp.getIndexingMapsArray();
  SmallVector<utils::IteratorType> iteratorTypes =
      linalgOp.getIteratorTypesArray();
  LLVM_DEBUG(DBGS() << "Start packing: " << linalgOp << "\n";
             llvm::interleaveComma(indexingMaps, DBGS() << "maps: "); DBGSNL();
             llvm::interleaveComma(iteratorTypes, DBGS() << "iterators: ");
             DBGSNL(););

  SmallVector<tensor::PackOp> packOps;
  SmallVector<tensor::UnPackOp> unPackOps;
  // Step 1. Pack each dim of the LinalgOp metadata by packedSizes[i].
  PackedOperandsDimList listOfPackedOperandsDim;
  for (int64_t i = 0, e = packedSizes.size(); i < e; ++i) {
    std::optional<int64_t> maybeConstant = getConstantIntValue(packedSizes[i]);
    // Skip tile sizes explicitly set to 0.
    if (maybeConstant.has_value() && maybeConstant.value() == 0)
      continue;

    PackedOperandsDim packedOperandsDims;
    packedOperandsDims.packedSize = packedSizes[i];
    FailureOr<SmallVector<std::optional<int64_t>>>
        maybePackedDimForEachOperand =
            packLinalgMetadataOnce(indexingMaps, iteratorTypes, i);
    if (failed(maybePackedDimForEachOperand))
      return failure();
    packedOperandsDims.packedDimForEachOperand = *maybePackedDimForEachOperand;
    listOfPackedOperandsDim.push_back(std::move(packedOperandsDims));

    LLVM_DEBUG(
        DBGS() << "++++ After pack size #" << i << ": " << packedSizes[i]
               << "\n";
        llvm::interleaveComma(indexingMaps, DBGS() << "maps: "); DBGSNL();
        llvm::interleaveComma(iteratorTypes, DBGS() << "iterators: "); DBGSNL();
        llvm::interleaveComma(packedOperandsDims.packedDimForEachOperand,
                              DBGS() << "packedDimForEachOperand: ");
        DBGSNL(););
  }

  // Step 2. Propagate packing to all LinalgOp operands.
  SmallVector<Value> inputsAndInits, results;
  for (const auto &operandsList :
       {linalgOp.getDpsInputOperands(), linalgOp.getDpsInitOperands()}) {
    for (OpOperand *opOperandPtr : operandsList) {
      int64_t pos = opOperandPtr->getOperandNumber();
      Value operand = opOperandPtr->get();
      SmallVector<int64_t> innerPos =
          listOfPackedOperandsDim.extractPackedDimsForOperand(pos);
      SmallVector<OpFoldResult> innerPackSizes =
          listOfPackedOperandsDim.extractPackSizesForOperand(pos);
      LLVM_DEBUG(
          DBGS() << "operand: " << operand << "\n";
          llvm::interleaveComma(innerPos, DBGS() << "innerPos: "); DBGSNL();
          llvm::interleaveComma(innerPackSizes, DBGS() << "innerPackSizes: ");
          DBGSNL(););
      if (innerPackSizes.empty()) {
        inputsAndInits.push_back(operand);
        continue;
      }
      Value dest = tensor::PackOp::createDestinationTensor(
          rewriter, loc, operand, innerPackSizes, innerPos,
          /*outerDimsPerm=*/{});
      // TODO: value of the padding attribute should be determined by consumers.
      Attribute zeroAttr =
          rewriter.getZeroAttr(getElementTypeOrSelf(dest.getType()));
      Value zero = rewriter.create<arith::ConstantOp>(loc, zeroAttr);
      packOps.push_back(rewriter.create<tensor::PackOp>(
          loc, operand, dest, innerPos, innerPackSizes, zero));
      inputsAndInits.push_back(packOps.back());
    }
  }

  // Step 3. Build the packed op, use the type of `inits` as result types.
  ValueRange inputs =
      ValueRange{inputsAndInits}.take_front(linalgOp.getNumDpsInputs());
  ValueRange inits =
      ValueRange{inputsAndInits}.take_back(linalgOp.getNumDpsInits());
  auto packedLinalgOp = rewriter.create<linalg::GenericOp>(
      linalgOp.getLoc(), inits.getTypes(), inputs, inits, indexingMaps,
      iteratorTypes);
  packedLinalgOp.getRegion().takeBody(linalgOp->getRegion(0));

  // Step 4. Propagate packing to all the op results.
  for (OpResult result : packedLinalgOp->getResults()) {
    int64_t resultNum = result.getResultNumber();
    tensor::PackOp maybePackedInit =
        inits[resultNum].getDefiningOp<tensor::PackOp>();
    if (!maybePackedInit) {
      results.push_back(result);
      continue;
    }
    // Build the symmetrical UnPackOp to the existing PackOp.
    unPackOps.push_back(rewriter.create<tensor::UnPackOp>(
        packedLinalgOp->getLoc(), result, maybePackedInit.getSource(),
        maybePackedInit.getInnerDimsPos(), maybePackedInit.getMixedTiles()));
    results.push_back(unPackOps.back());
  }

  // Step 5. Replace `linalgOp`.
  rewriter.replaceOp(linalgOp, results);

  // Return packedLinalgOp.
  return PackResult{packOps,
                    cast<linalg::LinalgOp>(packedLinalgOp.getOperation()),
                    unPackOps};
}

//===----------------------------------------------------------------------===//
// packTranspose transformation.
//===----------------------------------------------------------------------===//

/// Return a copy of `tensorType` after permutation by `permutationVector`.
// Note: Should be a new method in of MemRef/RankedTensor/VectorType::Builder
// but this would introduce a dependence on Dialect in IR.
// TODO: Restructure.
static RankedTensorType permuteShape(RankedTensorType tensorType,
                                     ArrayRef<int64_t> permutationVector) {
  SmallVector<int64_t> shape(tensorType.getShape());
  applyPermutationToVector(shape, permutationVector);
  return RankedTensorType::Builder(tensorType).setShape(shape);
}

/// Return a new GenericOp obtained by transposing opOperand by the permutation
/// vector:
///   - the corresponding indexing map is transposed by `permutation`
///   - the corresponding operand value is replaced by `transposedValue`
/// `linalgOp` is replaced by the return op in the process.
/// Asserts that `transposedValue` is of the proper transposed ShapedType.
static LinalgOp transposeOneLinalgOperandAndReplace(
    RewriterBase &rewriter, LinalgOp linalgOp, OpOperand &opOperand,
    ArrayRef<int64_t> permutation, Value transposedValue) {
  // Sanity check the operand.
  assert(linalgOp == opOperand.getOwner() && "linalg op must own the operand");

  // Sanity check of the expected transposed tensor type.
  auto tensorType = permuteShape(
      opOperand.get().getType().cast<RankedTensorType>(), permutation);
  (void)tensorType;
  assert(tensorType == transposedValue.getType() &&
         "expected tensor type mismatch");

  // Compute the transposed indexing map.
  // Sigh unsigned pollution.
  SmallVector<unsigned> tmpTransposition = llvm::to_vector(
      llvm::map_range(permutation, [](int64_t i) -> unsigned { return i; }));
  AffineMap permutationMap =
      AffineMap::getPermutationMap(tmpTransposition, rewriter.getContext());
  AffineMap transposedMap =
      permutationMap.compose(linalgOp.getMatchingIndexingMap(&opOperand));

  // Set the transposed indexing map in the proper position.
  SmallVector<AffineMap> indexingMaps = linalgOp.getIndexingMapsArray();
  indexingMaps[linalgOp.getIndexingMapIndex(&opOperand)] = transposedMap;
  // Set the transposedValue in the proper operand position.
  SmallVector<Value> operands = linalgOp->getOperands();
  operands[opOperand.getOperandNumber()] = transposedValue;

  ValueRange operandsRef(operands);
  auto transposedGenericOp = rewriter.create<linalg::GenericOp>(
      /*location=*/linalgOp->getLoc(),
      /*resultTensorTypes=*/
      operandsRef.drop_front(linalgOp.getNumDpsInputs()).getTypes(),
      /*inputs=*/operandsRef.take_front(linalgOp.getNumDpsInputs()),
      /*outputs=*/operandsRef.drop_front(linalgOp.getNumDpsInputs()),
      /*indexingMaps=*/indexingMaps,
      /*iteratorTypes=*/linalgOp.getIteratorTypesArray());
  transposedGenericOp.getRegion().takeBody(linalgOp->getRegion(0));
  rewriter.replaceOp(linalgOp, transposedGenericOp->getResults());

  return cast<linalg::LinalgOp>(transposedGenericOp.getOperation());
}

FailureOr<PackTransposeResult>
linalg::packTranspose(RewriterBase &rewriter, tensor::PackOp packOp,
                      linalg::LinalgOp linalgOp, tensor::UnPackOp maybeUnPackOp,
                      ArrayRef<int64_t> outerPerm,
                      ArrayRef<int64_t> innerPerm) {
  Location loc = linalgOp.getLoc();

  // Step 1. Transpose packOp.
  rewriter.setInsertionPoint(packOp);
  tensor::PackOp transposedPackOp =
      packOp.createTransposedClone(rewriter, loc, innerPerm, outerPerm);

  if (!packOp.getResult().hasOneUse())
    return rewriter.notifyMatchFailure(linalgOp, "expect single pack use");

  OpOperand &packUse = *packOp->getUses().begin();
  if (packUse.getOwner() != linalgOp) {
    return rewriter.notifyMatchFailure(
        linalgOp, "not a single use by the LinalgOp target");
  }
  if (maybeUnPackOp &&
      (!linalgOp.isDpsInit(&packUse) ||
       maybeUnPackOp.getSource() != linalgOp.getTiedOpResult(&packUse))) {
    return rewriter.notifyMatchFailure(linalgOp,
                                       "not produced by the LinalgOp target");
  }

  // Step 2. Transpose linalgOp.
  // transposedPackOp.getOuterDimsPerm() may be empty, in which case it is the
  // identity. Don't rely on it.
  int64_t numLeadingDims = packOp.getSourceRank();
  int64_t numTrailingDims = packOp.getInnerDimsPos().size();
  // Step 2.a. Compute the permutation on the whole operand.
  // Leading part just reuse the outerPerm.
  SmallVector<int64_t> permutation(outerPerm);
  if (permutation.empty())
    llvm::append_range(permutation, llvm::seq<int64_t>(0, numLeadingDims));
  // Trailing part needs to reindex positions by `numLeadingDims`.
  if (innerPerm.empty()) {
    llvm::append_range(
        permutation,
        llvm::seq<int64_t>(numLeadingDims, numLeadingDims + numTrailingDims));
  } else {
    llvm::append_range(permutation,
                       llvm::map_range(innerPerm, [&](int64_t pos) {
                         return numLeadingDims + pos;
                       }));
  }
  if (!isPermutationVector(permutation))
    return rewriter.notifyMatchFailure(linalgOp, "invalid permutation");

  // Step 2.b. Save the transposedPackUse operand number in case we need to
  // get the tied OpResult after `linalgOp` has been replaced.
  int64_t packUseOperandNumber = packUse.getOperandNumber();
  // Step 2.c. Actually perform the transposition.
  rewriter.setInsertionPoint(linalgOp);
  linalg::LinalgOp transposedLinalgOp = transposeOneLinalgOperandAndReplace(
      rewriter, linalgOp, packUse, permutation, transposedPackOp.getResult());

  // Step 3. Maybe transpose unPackOp.
  tensor::UnPackOp transposedUnPackOp;
  if (maybeUnPackOp) {
    OpOperand &opOperand =
        transposedLinalgOp->getOpOperand(packUseOperandNumber);
    OpResult transposedResult = transposedLinalgOp.getTiedOpResult(&opOperand);
    rewriter.setInsertionPoint(maybeUnPackOp);
    transposedUnPackOp = maybeUnPackOp.createTransposedClone(
        rewriter, loc, transposedResult, innerPerm, outerPerm);

    rewriter.replaceOp(maybeUnPackOp, transposedUnPackOp->getResults());
  }

  // Step 4. Finally, replace packOp now that we don't need it anymore.
  rewriter.replaceOp(packOp, transposedPackOp->getResults());

  return PackTransposeResult{transposedPackOp, transposedLinalgOp,
                             transposedUnPackOp};
}

//===----------------------------------------------------------------------===//
// Transformations exposed as rewrite patterns.
//===----------------------------------------------------------------------===//

LinalgTilingOptions &
mlir::linalg::LinalgTilingOptions::setTileSizes(ArrayRef<int64_t> ts) {
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

///
/// Padding pattern.
///
mlir::linalg::LinalgPaddingPattern::LinalgPaddingPattern(
    MLIRContext *context, LinalgPaddingOptions options, PatternBenefit benefit)
    : OpInterfaceRewritePattern<LinalgOp>(context, benefit),
      options(std::move(options)) {}

LogicalResult mlir::linalg::LinalgPaddingPattern::matchAndRewrite(
    LinalgOp op, PatternRewriter &rewriter) const {
  return padAndHoistLinalgOp(rewriter, op, options);
}

LogicalResult mlir::linalg::CopyVectorizationPattern::matchAndRewrite(
    memref::CopyOp copyOp, PatternRewriter &rewriter) const {
  return vectorizeCopy(rewriter, copyOp);
}

///
/// Pattern to rewrite a tensor::PadOp into a sequence of EmptyOp, FillOp (to
/// initialize with pad_val) and GenericOp (to copy contents).
///
LogicalResult
PadOpTransformationPattern::matchAndRewrite(tensor::PadOp padOp,
                                            PatternRewriter &rewriter) const {

  auto inputShapedType = padOp.getSource().getType().cast<ShapedType>();
  auto resultShapedType = padOp.getResult().getType().cast<ShapedType>();

  // Bail on non-static shapes.
  if (!inputShapedType.hasStaticShape())
    return failure();
  if (!resultShapedType.hasStaticShape())
    return failure();

  // Only support padding with a constant for now, i.e. either:
  //   1. A BBarg from a different block.
  //   2. A value defined outside of the current block.
  Block &block = padOp.getRegion().front();
  auto yieldOp = cast<tensor::YieldOp>(block.getTerminator());
  Value padValue = yieldOp.getValue();
  Operation *definingOp = padValue.getDefiningOp();
  if (definingOp && definingOp->getBlock() == &block)
    return failure();
  if (!definingOp && padValue.cast<BlockArgument>().getOwner() == &block)
    return failure();

  // Create tensor with the padded shape
  Location loc = padOp.getLoc();
  SmallVector<Value> indices(resultShapedType.getRank(),
                             rewriter.create<arith::ConstantIndexOp>(loc, 0));
  Value emptyTensor = rewriter.create<tensor::EmptyOp>(
      loc, resultShapedType.getShape(), resultShapedType.getElementType());

  // Initialize tensor with the pad value
  Value tmpTensor = rewriter
                        .create<linalg::FillOp>(loc, ValueRange{padValue},
                                                ValueRange{emptyTensor})
                        .result();

  // Copy original contents into new tensor
  // Uses linalg.generic, but could be done with tensor.insert_slice
  SmallVector<AffineExpr, 4> outputExprs;
  for (unsigned i = 0; i < resultShapedType.getRank(); ++i) {
    outputExprs.push_back(getAffineDimExpr(i, rewriter.getContext()) +
                          padOp.getStaticLow()[i]);
  }

  SmallVector<AffineMap, 2> transferMaps = {
      rewriter.getMultiDimIdentityMap(inputShapedType.getRank()),
      AffineMap::get(resultShapedType.getRank(),
                     /*symbolCount=*/0, outputExprs, rewriter.getContext())};

  rewriter.replaceOpWithNewOp<linalg::GenericOp>(
      padOp, resultShapedType, padOp.getSource(), tmpTensor, transferMaps,
      getNParallelLoopsAttrs(resultShapedType.getRank()),
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
        nestedBuilder.create<linalg::YieldOp>(nestedLoc, args[0]);
      });

  return success();
}

/// Filling `dest` using FillOp constant padding value if possible.
/// Otherwise, generate a tensor::GenerateOp.
Value GeneralizePadOpPattern::createFillOrGenerateOp(
    RewriterBase &rewriter, tensor::PadOp padOp, Value dest,
    const SmallVector<Value> &dynSizes) const {
  auto padValue = padOp.getConstantPaddingValue();
  if (padValue)
    return rewriter.create<FillOp>(padOp.getLoc(), padValue, dest).result();

  // Fill could not be optimized: Lower to tensor::GenerateOp with region.
  auto generateOp = rewriter.create<tensor::GenerateOp>(
      padOp.getLoc(), padOp.getResultType(), dynSizes);
  // Copy region to new op.
  IRMapping bvm;
  padOp.getRegion().cloneInto(&generateOp.getRegion(), bvm);
  return generateOp;
}

LogicalResult
GeneralizePadOpPattern::matchAndRewrite(tensor::PadOp padOp,
                                        PatternRewriter &rewriter) const {
  // Given an OpFoldResult, return an index-typed value.
  auto getIdxValue = [&](OpFoldResult ofr) {
    if (auto val = ofr.dyn_cast<Value>())
      return val;
    return rewriter
        .create<arith::ConstantIndexOp>(
            padOp.getLoc(), ofr.get<Attribute>().cast<IntegerAttr>().getInt())
        .getResult();
  };

  auto resultType = padOp.getResultType();
  // Compute size of EmptyOp. Any combination of static/dynamic is supported.
  SmallVector<Value> dynSizes;
  SmallVector<int64_t> staticSizes;
  for (unsigned dim = 0; dim < resultType.getRank(); ++dim) {
    if (resultType.isDynamicDim(dim)) {
      auto srcSize = rewriter.createOrFold<tensor::DimOp>(
          padOp.getLoc(), padOp.getSource(), dim);
      // Add low and high padding value.
      auto plusLow = rewriter.createOrFold<arith::AddIOp>(
          padOp.getLoc(), srcSize, getIdxValue(padOp.getMixedLowPad()[dim]));
      auto plusHigh = rewriter.createOrFold<arith::AddIOp>(
          padOp.getLoc(), plusLow, getIdxValue(padOp.getMixedHighPad()[dim]));
      dynSizes.push_back(plusHigh);
    }
    staticSizes.push_back(resultType.getDimSize(dim));
  }

  // Init tensor and fill it with padding.
  Value emptyTensor = rewriter.create<tensor::EmptyOp>(
      padOp.getLoc(), staticSizes, resultType.getElementType(), dynSizes);
  Value fill = createFillOrGenerateOp(rewriter, padOp, emptyTensor, dynSizes);

  // Try optimize the copy of source.
  if (optimizeCopyFn && optimizeCopyFn(rewriter, padOp, fill).succeeded())
    return success();

  // tensor::PadOps cannot be optimized. Generate a InsertSliceOp instead
  // for copying the PadOp source.
  auto sourceType = padOp.getSourceType();
  // Compute size of source of tensor::PadOp.
  SmallVector<OpFoldResult> srcSizes;
  for (unsigned dim = 0; dim < sourceType.getRank(); ++dim) {
    if (sourceType.isDynamicDim(dim)) {
      srcSizes.push_back(rewriter.createOrFold<tensor::DimOp>(
          padOp.getLoc(), padOp.getSource(), dim));
    } else {
      srcSizes.push_back(rewriter.getIndexAttr(sourceType.getDimSize(dim)));
    }
  }
  // Strides of InsertSliceOp are all 1.
  SmallVector<OpFoldResult> strides(sourceType.getRank(),
                                    rewriter.getIndexAttr(1));
  rewriter.replaceOpWithNewOp<tensor::InsertSliceOp>(
      padOp, padOp.getSource(), fill, padOp.getMixedLowPad(), srcSizes,
      strides);

  return success();
}

LogicalResult ExtractSliceOfPadTensorSwapPattern::matchAndRewrite(
    tensor::ExtractSliceOp sliceOp, PatternRewriter &rewriter) const {
  if (!sliceOp.hasUnitStride())
    return failure();

  auto padOp = sliceOp.getSource().getDefiningOp<tensor::PadOp>();
  if (!padOp)
    return failure();

  bool zeroSliceGuard = true;
  if (controlFn) {
    if (std::optional<bool> control = controlFn(sliceOp))
      zeroSliceGuard = *control;
    else
      return failure();
  }

  FailureOr<TilingResult> tilingResult =
      tensor::bubbleUpPadSlice(rewriter, padOp, sliceOp.getMixedOffsets(),
                               sliceOp.getMixedSizes(), zeroSliceGuard);
  if (failed(tilingResult))
    return failure();
  // All shapes are static and the data source is actually used. Rewrite into
  // pad(extract_slice(x)).
  rewriter.replaceOp(sliceOp, tilingResult->tiledValues);
  return success();
}

/// Returns a tensor.pad op if padding value is set. Otherwise, returns the
/// source directly. The method assumes that the `packOp` has static shapes.
static Value getPackOpSourceOrPaddedSource(OpBuilder &builder,
                                           tensor::PackOp packOp) {
  Value input = packOp.getSource();
  if (!packOp.getPaddingValue()) {
    return input;
  }

  Location loc = packOp.getLoc();
  ShapedType inputType = packOp.getSourceType();
  int64_t inputRank = inputType.getRank();
  assert(llvm::all_of(packOp.getDestType().getShape().take_front(inputRank),
                      [](int64_t val) { return val == 1; }));

  SmallVector<int64_t> paddedShape;
  DenseMap<int64_t, OpFoldResult> tileAndPosMapping =
      packOp.getDimAndTileMapping();
  for (int64_t dim = 0; dim < inputRank; ++dim) {
    int64_t size = inputType.getDimSize(dim);
    if (!tileAndPosMapping.count(dim)) {
      paddedShape.push_back(size);
      continue;
    }

    // The size is less than or equal to tileSize because outer dims are all 1s.
    std::optional<int64_t> tileSize =
        getConstantIntValue(tileAndPosMapping.lookup(dim));
    assert(tileSize.has_value() && "dynamic inner tile size is not supported");
    paddedShape.push_back(tileSize.value());
  }
  auto resultType =
      RankedTensorType::get(paddedShape, inputType.getElementType());
  return tensor::createPadHighOp(resultType, input, packOp.getPaddingValue(),
                                 /*nofold=*/false, loc, builder);
}

static SmallVector<int64_t>
getPackUnpackNormalizedInnerPerm(int rank, ArrayRef<int64_t> innerDimsPos) {
  constexpr int64_t kNonTiledMarker = -1;
  SmallVector<int64_t> vec(rank, kNonTiledMarker);
  for (auto [index, value] : llvm::enumerate(innerDimsPos))
    vec[value] = index;
  SmallVector<int64_t> perm = llvm::to_vector(llvm::make_filter_range(
      vec, [&](int64_t v) { return v != kNonTiledMarker; }));
  return perm;
}

LogicalResult GeneralizeOuterUnitDimsPackOpPattern::matchAndRewrite(
    tensor::PackOp packOp, PatternRewriter &rewriter) const {
  // TODO: support the case that outer dimensions are not all 1s A
  // tensor.expand_shape will be generated in this case.
  int64_t srcRank = packOp.getSourceRank();
  if (llvm::any_of(packOp.getDestType().getShape().take_front(srcRank),
                   [](int64_t val) { return val != 1; })) {
    return rewriter.notifyMatchFailure(
        packOp, "require the outer dimension of the result are all 1s");
  }

  if (llvm::any_of(packOp.getMixedTiles(),
                   [](OpFoldResult tile) { return tile.is<Value>(); })) {
    return rewriter.notifyMatchFailure(packOp,
                                       "require inner tile sizes being static");
  }

  // 1. Use rank-reduced tensor.extract_slice op to extract the tile.
  Location loc = packOp.getLoc();
  Attribute zeroIdxAttr = rewriter.getIndexAttr(0);
  Attribute oneIdxAttr = rewriter.getIndexAttr(1);
  SmallVector<OpFoldResult> readOffsets(srcRank, zeroIdxAttr);
  SmallVector<OpFoldResult> readStrides(srcRank, oneIdxAttr);
  SmallVector<OpFoldResult> readSizes;
  SmallVector<int64_t> readShape;
  DenseMap<int64_t, OpFoldResult> dimAndTileMapping =
      packOp.getDimAndTileMapping();
  for (auto i : llvm::seq<unsigned>(0, srcRank)) {
    if (!dimAndTileMapping.count(i)) {
      readSizes.push_back(oneIdxAttr);
      continue;
    }
    readSizes.push_back(dimAndTileMapping[i]);
    readShape.push_back(getConstantIntValue(dimAndTileMapping[i])
                            .value_or(ShapedType::kDynamic));
  }
  Type elemType = packOp.getSourceType().getElementType();
  auto readType = RankedTensorType::get(readShape, elemType);

  Value input = getPackOpSourceOrPaddedSource(rewriter, packOp);
  Value tile = rewriter.create<tensor::ExtractSliceOp>(
      loc, readType, input, readOffsets, readSizes, readStrides);

  // 2. Transpose the tile to match the inner tile order.
  SmallVector<int64_t> perm =
      getPackUnpackNormalizedInnerPerm(srcRank, packOp.getInnerDimsPos());
  // The permutation is inverted when normalizing so invert back to match the
  // ordering in the pack op.
  perm = invertPermutationVector(perm);

  LLVM_DEBUG(DBGS() << "Pack permutation: " << packOp << "\n";
             llvm::interleaveComma(perm, DBGS() << "perm: "); DBGSNL(););

  SmallVector<int64_t> transpShape = readShape;
  applyPermutationToVector<int64_t>(transpShape, perm);

  Value empty = rewriter.create<tensor::EmptyOp>(loc, transpShape, elemType);
  auto transposedOp =
      rewriter.create<linalg::TransposeOp>(loc, tile, empty, perm);

  // 3. Insert the inner tile to the destination.
  int64_t destRank = packOp.getDestRank();
  SmallVector<OpFoldResult> writeStrides(destRank, oneIdxAttr);
  SmallVector<OpFoldResult> writeOffsets(destRank, zeroIdxAttr);
  SmallVector<OpFoldResult> writeSizes(srcRank, oneIdxAttr);
  for (auto size : transpShape)
    writeSizes.push_back(rewriter.getIndexAttr(size));

  auto insert = rewriter.create<tensor::InsertSliceOp>(
      loc, transposedOp.getResult()[0], packOp.getDest(), writeOffsets,
      writeSizes, writeStrides);
  rewriter.replaceOp(packOp, insert.getResult());

  return success();
}

LogicalResult GeneralizeOuterUnitDimsUnPackOpPattern::matchAndRewrite(
    tensor::UnPackOp unpackOp, PatternRewriter &rewriter) const {
  int64_t srcRank = unpackOp.getSourceRank();
  int64_t destRank = unpackOp.getDestRank();
  ArrayRef<int64_t> srcShape = unpackOp.getSourceType().getShape();
  if (llvm::any_of(srcShape.take_front(destRank),
                   [](int64_t val) { return val != 1; })) {
    return rewriter.notifyMatchFailure(
        unpackOp, "require the outer dimension of the result are all 1s");
  }

  // 1. Use rank-reduced tensor.extract_slice op to extract the tile.
  Location loc = unpackOp.getLoc();
  Attribute zeroIdxAttr = rewriter.getIndexAttr(0);
  Attribute oneIdxAttr = rewriter.getIndexAttr(1);
  SmallVector<OpFoldResult> readOffsets(srcRank, zeroIdxAttr);
  SmallVector<OpFoldResult> readStrides(srcRank, oneIdxAttr);

  auto mixedTiles = unpackOp.getMixedTiles();
  SmallVector<OpFoldResult> readSizes(destRank, oneIdxAttr);
  readSizes.append(mixedTiles.begin(), mixedTiles.end());

  // Explicitly create the type for extract_slice op because the inner tile
  // size could be 1. We want to represent the whole inner tile in this case.
  ArrayRef<int64_t> readShape = srcShape.drop_front(destRank);
  Type elemType = unpackOp.getSourceType().getElementType();
  auto readType = RankedTensorType::get(readShape, elemType);
  Value innerTile = rewriter.create<tensor::ExtractSliceOp>(
      loc, readType, unpackOp.getSource(), readOffsets, readSizes, readStrides);

  // 2. Transpose the tile to match the outer corresponding tile order.
  ArrayRef<int64_t> innerDimsPos = unpackOp.getInnerDimsPos();
  SmallVector<int64_t> perm =
      getPackUnpackNormalizedInnerPerm(srcRank, innerDimsPos);
  SmallVector<int64_t> transpShape(readShape);
  applyPermutationToVector<int64_t>(transpShape, perm);

  Value empty = rewriter.create<tensor::EmptyOp>(loc, transpShape, elemType);
  auto transposedOp =
      rewriter.create<linalg::TransposeOp>(loc, innerTile, empty, perm);

  // 3. Handle in-complete tiles if needed. It truncates trailing data from the
  // transposed tile.
  int numLoops = transpShape.size();
  SmallVector<OpFoldResult> tileStrides(numLoops, oneIdxAttr);
  SmallVector<OpFoldResult> tileOffsets(numLoops, zeroIdxAttr);
  SmallVector<OpFoldResult> tileSizes;
  for (int dim : innerDimsPos)
    tileSizes.push_back(getAsOpFoldResult(
        rewriter.createOrFold<tensor::DimOp>(loc, unpackOp.getDest(), dim)));

  applyPermutationToVector<OpFoldResult>(tileSizes, perm);
  auto partialTile = rewriter.create<tensor::ExtractSliceOp>(
      loc, transposedOp.getResult()[0], tileOffsets, tileSizes, tileStrides);

  // 4. Insert the result to the destination tensor.
  SmallVector<OpFoldResult> writeSizes;
  SmallVector<OpFoldResult> writeStrides(destRank, oneIdxAttr);
  SmallVector<OpFoldResult> writeOffsets(destRank, zeroIdxAttr);
  DenseMap<int64_t, OpFoldResult> dimAndTileMapping =
      unpackOp.getDimAndTileMapping();
  for (int i = 0, idx = 0; i < destRank; ++i) {
    if (dimAndTileMapping.count(i))
      writeSizes.push_back(tileSizes[idx++]);
    else
      writeSizes.push_back(oneIdxAttr);
  }
  auto insert = rewriter.create<tensor::InsertSliceOp>(
      loc, partialTile, unpackOp.getDest(), writeOffsets, writeSizes,
      writeStrides);
  rewriter.replaceOp(unpackOp, insert.getResult());

  return success();
}

// The following are patterns for downscaling convolution ops with size-1
// window dimensions.
//
// Note that we'd eventually want to write such transformations in a generic
// way, e.g., converting to linalg.generic, removing the size-1 dimensions,
// and then turning back to named ops. But for now it's fine to have a few
// patterns matching special ops to get started.

template <typename Conv2DOp, typename Conv1DOp>
FailureOr<Conv1DOp> DownscaleSizeOneWindowed2DConvolution<Conv2DOp, Conv1DOp>::
    returningMatchAndRewrite(Conv2DOp convOp, PatternRewriter &rewriter) const {
  if (convOp.hasBufferSemantics())
    return failure(); // To be implemented.

  Value input = convOp.getInputs().front();
  Value kernel = convOp.getInputs().back();
  Value output = convOp.getOutputs().front();

  auto inputType = input.getType().dyn_cast<RankedTensorType>();
  auto kernelType = kernel.getType().dyn_cast<RankedTensorType>();
  auto outputType = output.getType().dyn_cast<RankedTensorType>();

  auto kernelShape = kernelType.getShape();
  auto outputShape = outputType.getShape();

  // Get domain indices based on conv2D layout.
  auto [khIndex, kwIndex, ohIndex, owIndex] =
      TypeSwitch<Operation *, std::tuple<int64_t, int64_t, int64_t, int64_t>>(
          convOp)
          .Case([&](linalg::Conv2DNhwcHwcfOp op) {
            return std::make_tuple(0, 1, 1, 2);
          })
          .Case([&](linalg::Conv2DNchwFchwOp op) {
            return std::make_tuple(2, 3, 2, 3);
          })
          .Case([&](linalg::PoolingNhwcSumOp op) {
            return std::make_tuple(0, 1, 1, 2);
          })
          .Case([&](linalg::PoolingNchwSumOp op) {
            return std::make_tuple(0, 1, 2, 3);
          })
          .Case([&](linalg::PoolingNhwcMaxOp op) {
            return std::make_tuple(0, 1, 1, 2);
          })
          .Case([&](linalg::PoolingNhwcMaxUnsignedOp op) {
            return std::make_tuple(0, 1, 1, 2);
          })
          .Case([&](linalg::PoolingNhwcMinOp op) {
            return std::make_tuple(0, 1, 1, 2);
          })
          .Case([&](linalg::PoolingNhwcMinUnsignedOp op) {
            return std::make_tuple(0, 1, 1, 2);
          })
          .Case([&](linalg::PoolingNchwMaxOp op) {
            return std::make_tuple(0, 1, 2, 3);
          })
          .Default([&](Operation *op) {
            llvm_unreachable("unexpected conv2d/pool2d operation.");
            return std::make_tuple(0, 0, 0, 0);
          });

  // Only handle the case where at least one of the window dimensions is
  // of size 1. Other cases can rely on tiling to reduce to such cases.
  int64_t khSize = kernelShape[khIndex], kwSize = kernelShape[kwIndex];
  int64_t ohSize = outputShape[ohIndex], owSize = outputShape[owIndex];
  bool removeH = (khSize == 1 && ohSize == 1);
  bool removeW = (kwSize == 1 && owSize == 1);
  if (!removeH && !removeW)
    return failure();

  // Get new shapes and types for all operands by removing the size-1
  // dimension.
  using RTTBuilder = RankedTensorType::Builder;
  RankedTensorType newInputType =
      RTTBuilder(inputType).dropDim((removeH ? ohIndex : owIndex));
  RankedTensorType newKernelType =
      RTTBuilder(kernelType).dropDim((removeH ? khIndex : kwIndex));
  RankedTensorType newOutputType =
      RTTBuilder(outputType).dropDim((removeH ? ohIndex : owIndex));

  // Rank-reduce operands.
  Location loc = convOp.getLoc();
  Value newInput = tensor::createCanonicalRankReducingExtractSliceOp(
      rewriter, loc, input, newInputType);
  Value newKernel = tensor::createCanonicalRankReducingExtractSliceOp(
      rewriter, loc, kernel, newKernelType);
  Value newOutput = tensor::createCanonicalRankReducingExtractSliceOp(
      rewriter, loc, output, newOutputType);

  // Rank-reduce strides and dilations too.
  // TODO: dropDim 1-liner helper.
  auto strides =
      llvm::to_vector<4>(convOp.getStrides().template getValues<int64_t>());
  strides.erase(strides.begin() + (removeH ? 0 : 1));
  auto stridesAttr = rewriter.getI64VectorAttr(strides);

  auto dilations =
      llvm::to_vector<4>(convOp.getDilations().template getValues<int64_t>());
  dilations.erase(dilations.begin() + (removeH ? 0 : 1));
  auto dilationsAttr = rewriter.getI64VectorAttr(dilations);

  auto conv1DOp = rewriter.create<Conv1DOp>(
      loc, newOutputType, ValueRange{newInput, newKernel},
      ValueRange{newOutput}, stridesAttr, dilationsAttr);

  // Insert back.
  Value inserted = tensor::createCanonicalRankReducingInsertSliceOp(
      rewriter, loc, conv1DOp.getResult(0), output);
  rewriter.replaceOp(convOp, inserted);

  return conv1DOp;
}

template struct linalg::DownscaleSizeOneWindowed2DConvolution<Conv2DNhwcHwcfOp,
                                                              Conv1DNwcWcfOp>;
template struct linalg::DownscaleSizeOneWindowed2DConvolution<Conv2DNchwFchwOp,
                                                              Conv1DNcwFcwOp>;
template struct linalg::DownscaleSizeOneWindowed2DConvolution<PoolingNhwcSumOp,
                                                              PoolingNwcSumOp>;
template struct linalg::DownscaleSizeOneWindowed2DConvolution<PoolingNchwSumOp,
                                                              PoolingNcwSumOp>;
template struct linalg::DownscaleSizeOneWindowed2DConvolution<PoolingNhwcMaxOp,
                                                              PoolingNwcMaxOp>;
template struct linalg::DownscaleSizeOneWindowed2DConvolution<
    PoolingNhwcMaxUnsignedOp, PoolingNwcMaxUnsignedOp>;
template struct linalg::DownscaleSizeOneWindowed2DConvolution<PoolingNhwcMinOp,
                                                              PoolingNwcMinOp>;
template struct linalg::DownscaleSizeOneWindowed2DConvolution<
    PoolingNhwcMinUnsignedOp, PoolingNwcMinUnsignedOp>;
template struct linalg::DownscaleSizeOneWindowed2DConvolution<PoolingNchwMaxOp,
                                                              PoolingNcwMaxOp>;

FailureOr<DepthwiseConv1DNwcWcOp>
DownscaleDepthwiseConv2DNhwcHwcOp::returningMatchAndRewrite(
    DepthwiseConv2DNhwcHwcOp convOp, PatternRewriter &rewriter) const {
  if (convOp.hasBufferSemantics())
    return failure(); // To be implemented.

  Value input = convOp.getInputs().front();
  Value kernel = convOp.getInputs().back();
  Value output = convOp.getOutputs().front();

  auto inputType = input.getType().dyn_cast<RankedTensorType>();
  auto kernelType = kernel.getType().dyn_cast<RankedTensorType>();
  auto outputType = output.getType().dyn_cast<RankedTensorType>();

  auto kernelShape = kernelType.getShape();
  auto outputShape = outputType.getShape();

  // Only handle the case where at least one of the window dimensions is
  // of size 1. Other cases can rely on tiling to reduce to such cases.
  int64_t khSize = kernelShape[0], kwSize = kernelShape[1];
  int64_t ohSize = outputShape[1], owSize = outputShape[2];
  bool removeH = (khSize == 1 && ohSize == 1);
  bool removeW = (kwSize == 1 && owSize == 1);
  if (!removeH && !removeW)
    return failure();

  // Get new shapes and types for all operands by removing the size-1
  // dimension.
  using RTTBuilder = RankedTensorType::Builder;
  RankedTensorType newInputType =
      RTTBuilder(inputType).dropDim((removeH ? 1 : 2));
  RankedTensorType newKernelType =
      RTTBuilder(kernelType).dropDim((removeH ? 0 : 1));
  RankedTensorType newOutputType =
      RTTBuilder(outputType).dropDim(removeH ? 1 : 2);

  // Rank-reduce operands.
  Location loc = convOp.getLoc();
  Value newInput = tensor::createCanonicalRankReducingExtractSliceOp(
      rewriter, loc, input, newInputType);
  Value newKernel = tensor::createCanonicalRankReducingExtractSliceOp(
      rewriter, loc, kernel, newKernelType);
  Value newOutput = tensor::createCanonicalRankReducingExtractSliceOp(
      rewriter, loc, output, newOutputType);

  // Rank-reduce strides and dilations too.
  // TODO: dropDim 1-liner helper.
  auto strides = llvm::to_vector<4>(convOp.getStrides().getValues<int64_t>());
  strides.erase(strides.begin() + (removeH ? 0 : 1));
  auto stridesAttr = rewriter.getI64VectorAttr(strides);

  auto dilations =
      llvm::to_vector<4>(convOp.getDilations().getValues<int64_t>());
  dilations.erase(dilations.begin() + (removeH ? 0 : 1));
  auto dilationsAttr = rewriter.getI64VectorAttr(dilations);

  auto conv1DOp = rewriter.create<DepthwiseConv1DNwcWcOp>(
      loc, newOutputType, ValueRange{newInput, newKernel},
      ValueRange{newOutput}, stridesAttr, dilationsAttr);

  // Insert back.
  Value inserted = tensor::createCanonicalRankReducingInsertSliceOp(
      rewriter, loc, conv1DOp.getResult(0), output);
  rewriter.replaceOp(convOp, inserted);

  return conv1DOp;
}

FailureOr<Conv1DOp>
DownscaleConv2DOp::returningMatchAndRewrite(Conv2DOp convOp,
                                            PatternRewriter &rewriter) const {
  if (convOp.hasBufferSemantics())
    return failure(); // To be implemented.

  Value input = convOp.getInputs().front();
  Value kernel = convOp.getInputs().back();
  Value output = convOp.getOutputs().front();

  auto inputType = input.getType().dyn_cast<RankedTensorType>();
  auto kernelType = kernel.getType().dyn_cast<RankedTensorType>();
  auto outputType = output.getType().dyn_cast<RankedTensorType>();

  auto kernelShape = kernelType.getShape();
  auto outputShape = outputType.getShape();

  // Only handle the case where at least one of the window dimensions is
  // of size 1. Other cases can rely on tiling to reduce to such cases.
  int64_t khSize = kernelShape[0], kwSize = kernelShape[1];
  int64_t ohSize = outputShape[0], owSize = outputShape[1];
  bool removeH = (khSize == 1 && ohSize == 1);
  bool removeW = (kwSize == 1 && owSize == 1);
  if (!removeH && !removeW)
    return failure();

  // Get new shapes and types for all operands by removing the size-1
  // dimension.
  using RTTBuilder = RankedTensorType::Builder;
  RankedTensorType newInputType =
      RTTBuilder(inputType).dropDim((removeH ? 0 : 1));
  RankedTensorType newKernelType =
      RTTBuilder(kernelType).dropDim((removeH ? 0 : 1));
  RankedTensorType newOutputType =
      RTTBuilder(outputType).dropDim(removeH ? 0 : 1);

  // Rank-reduce operands.
  Location loc = convOp.getLoc();
  Value newInput = tensor::createCanonicalRankReducingExtractSliceOp(
      rewriter, loc, input, newInputType);
  Value newKernel = tensor::createCanonicalRankReducingExtractSliceOp(
      rewriter, loc, kernel, newKernelType);
  Value newOutput = tensor::createCanonicalRankReducingExtractSliceOp(
      rewriter, loc, output, newOutputType);

  auto conv1DOp = rewriter.create<Conv1DOp>(loc, newOutputType,
                                            ValueRange{newInput, newKernel},
                                            ValueRange{newOutput});

  // Insert back.
  Value inserted = tensor::createCanonicalRankReducingInsertSliceOp(
      rewriter, loc, conv1DOp.getResult(0), output);
  rewriter.replaceOp(convOp, inserted);

  return conv1DOp;
}

void linalg::populateDecomposeConvolutionPatterns(RewritePatternSet &patterns,
                                                  PatternBenefit benefit) {
  patterns.add<DownscaleSizeOneWindowed2DConvolution<linalg::Conv2DNhwcHwcfOp,
                                                     Conv1DNwcWcfOp>,
               DownscaleSizeOneWindowed2DConvolution<linalg::Conv2DNchwFchwOp,
                                                     Conv1DNcwFcwOp>,
               DownscaleDepthwiseConv2DNhwcHwcOp, DownscaleConv2DOp>(
      patterns.getContext(), benefit);
  patterns.add<
      DownscaleSizeOneWindowed2DConvolution<PoolingNhwcSumOp, PoolingNwcSumOp>,
      DownscaleSizeOneWindowed2DConvolution<PoolingNchwSumOp, PoolingNcwSumOp>,
      DownscaleSizeOneWindowed2DConvolution<PoolingNhwcMaxOp, PoolingNwcMaxOp>,
      DownscaleSizeOneWindowed2DConvolution<PoolingNhwcMaxUnsignedOp,
                                            PoolingNwcMaxUnsignedOp>,
      DownscaleSizeOneWindowed2DConvolution<PoolingNhwcMinOp, PoolingNwcMinOp>,
      DownscaleSizeOneWindowed2DConvolution<PoolingNhwcMinUnsignedOp,
                                            PoolingNwcMinUnsignedOp>,
      DownscaleSizeOneWindowed2DConvolution<PoolingNchwMaxOp, PoolingNcwMaxOp>>(
      patterns.getContext(), benefit);
}
