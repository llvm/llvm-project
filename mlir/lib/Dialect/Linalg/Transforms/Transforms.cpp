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

//===----------------------------------------------------------------------===//
// Transformations exposed as functional-style API calls.
//===----------------------------------------------------------------------===//

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
    if (!isa<AffineDimExpr>(map.getResult(maybeOperandDimensionToPack.value())))
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
  void pushBack(PackedOperandsDim &&packedOperandsDims) {
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

FailureOr<LowerPackResult> linalg::lowerPack(RewriterBase &rewriter,
                                             tensor::PackOp packOp) {
  // 1. Filter out NYI cases.
  auto packedTensorType =
      cast<RankedTensorType>(packOp->getResultTypes().front());
  if (llvm::any_of(packOp.getStaticInnerTiles(),
                   [](int64_t size) { return ShapedType::isDynamic(size); })) {
    return rewriter.notifyMatchFailure(
        packOp,
        "non-static shape NYI, needs a more powerful tensor.expand_shape op");
  }

  Location loc = packOp->getLoc();
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(packOp);

  // 2. Compute the permutation vector to shuffle packed shape into the shape
  // before any outer or inner permutations have been applied.
  PackingMetadata packingMetadata = computePackingMetadata(
      packedTensorType.getRank(), packOp.getInnerDimsPos());
  SmallVector<int64_t> packedToStripMinedShapePerm =
      tensor::getPackInverseDestPerm(packOp);

  // 3. Compute the stripMinedShape: this is the packed shape before any outer
  // or inner permutations have been applied.
  SmallVector<int64_t> stripMinedShape(packedTensorType.getShape());
  applyPermutationToVector(stripMinedShape, packedToStripMinedShapePerm);

  // 4. Pad the source of packOp to a shape we can expand into stripMinedShape.
  SmallVector<OpFoldResult> lows(packOp.getSourceRank(),
                                 rewriter.getIndexAttr(0));
  SmallVector<OpFoldResult> highs(packOp.getSourceRank(),
                                  rewriter.getIndexAttr(0));
  for (auto [pos, innerSize] :
       llvm::zip_equal(packOp.getInnerDimsPos(), packOp.getMixedTiles())) {
    int outerPos =
        packedToStripMinedShapePerm[packingMetadata.outerPositions[pos]];
    OpFoldResult origSize =
        tensor::getMixedSize(rewriter, loc, packOp.getSource(), pos);
    OpFoldResult outerSize =
        tensor::getMixedSize(rewriter, loc, packOp.getDest(), outerPos);
    AffineExpr s0, d0, d1;
    bindDims(rewriter.getContext(), d0, d1);
    bindSymbols(rewriter.getContext(), s0);
    auto map = AffineMap::get(/*dimCount=*/2, /*symbolCount=*/1, d0 * s0 - d1);
    highs[pos] = affine::makeComposedFoldedAffineApply(
        rewriter, loc, map, {outerSize, origSize, innerSize});
  }
  RankedTensorType collapsed = tensor::CollapseShapeOp::inferCollapsedType(
      RankedTensorType::Builder(packedTensorType).setShape(stripMinedShape),
      packingMetadata.reassociations);
  Value paddingValue = packOp.getPaddingValue();
  if (!paddingValue) {
    paddingValue = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(getElementTypeOrSelf(collapsed)));
  }
  auto padOp =
      rewriter.create<tensor::PadOp>(loc, collapsed, packOp.getSource(), lows,
                                     highs, paddingValue, /*nofold=*/false);

  LLVM_DEBUG(
      DBGSNL(); DBGSNL(); llvm::interleaveComma(packingMetadata.insertPositions,
                                                DBGS() << "insertPositions: ");
      DBGSNL(); llvm::interleaveComma(packingMetadata.outerPositions,
                                      DBGS() << "outerPositions: ");
      DBGSNL(); llvm::interleaveComma(packedTensorType.getShape(),
                                      DBGS() << "packedShape: ");
      DBGSNL();
      llvm::interleaveComma(packedToStripMinedShapePerm,
                            DBGS() << "packedToStripMinedShapePerm: ");
      DBGSNL(); llvm::interleaveComma(
          packingMetadata.reassociations, DBGS() << "reassociations: ",
          [&](ReassociationIndices ri) {
            llvm::interleaveComma(ri, llvm::dbgs() << "|");
          });
      DBGSNL();
      llvm::interleaveComma(stripMinedShape, DBGS() << "stripMinedShape: ");
      DBGSNL(); DBGS() << "collapsed type: " << collapsed; DBGSNL(););

  if (packOp.isLikePad()) {
    // Pack ops which operate as simple pads may not produce legal
    // tensor.insert_slice operations when the packed type does not rank reduce
    // to the padded type.
    SliceVerificationResult rankReduces =
        isRankReducedType(packedTensorType, padOp.getResultType());

    if (rankReduces == SliceVerificationResult::Success) {
      // This pack is just a plain pad.
      // Just insert the pad in the higher ranked tensor.
      auto emptyOp =
          rewriter.create<tensor::EmptyOp>(loc, packedTensorType, ValueRange{});
      // Offsets.
      SmallVector<OpFoldResult> zeros(packOp.getDestRank(),
                                      rewriter.getIndexAttr(0));
      // Strides.
      SmallVector<OpFoldResult> ones(packOp.getDestRank(),
                                     rewriter.getIndexAttr(1));
      SmallVector<OpFoldResult> sizes =
          tensor::getMixedSizes(rewriter, loc, packOp.getDest());

      auto insertSliceOp = rewriter.create<tensor::InsertSliceOp>(
          loc, /*source=*/padOp, /*dest=*/emptyOp,
          /*offsets=*/zeros, sizes,
          /*strides=*/ones);

      LLVM_DEBUG(DBGS() << "insert_slice op: " << insertSliceOp; DBGSNL(););

      rewriter.replaceOp(packOp, insertSliceOp->getResults());

      return LowerPackResult{padOp, /*reshapeOp=*/nullptr,
                             /*transposeOp=*/nullptr};
    }
  }
  // 5. Expand from the padded result to the stripMinedShape.
  auto reshapeOp = rewriter.create<tensor::ExpandShapeOp>(
      loc,
      RankedTensorType::Builder(packedTensorType).setShape(stripMinedShape),
      padOp.getResult(), packingMetadata.reassociations);

  // 6. Transpose stripMinedShape to packedShape.
  SmallVector<int64_t> transpPerm =
      invertPermutationVector(packedToStripMinedShapePerm);
  auto transposeOp = rewriter.create<linalg::TransposeOp>(
      loc, reshapeOp.getResult(), packOp.getDest(), transpPerm);

  LLVM_DEBUG(DBGSNL(); DBGSNL(); DBGSNL();
             DBGS() << "reshape op: " << reshapeOp; DBGSNL();
             llvm::interleaveComma(transpPerm, DBGS() << "transpPerm: ");
             DBGSNL(); DBGS() << "transpose op: " << transposeOp; DBGSNL(););

  // 7. Replace packOp by transposeOp.
  rewriter.replaceOp(packOp, transposeOp->getResults());

  return LowerPackResult{padOp, reshapeOp, transposeOp};
}

FailureOr<LowerUnPackOpResult> linalg::lowerUnPack(RewriterBase &rewriter,
                                                   tensor::UnPackOp unPackOp) {
  // 1. Filter out NYI cases.
  if (!unPackOp.getOuterDimsPerm().empty() &&
      !isIdentityPermutation(unPackOp.getOuterDimsPerm())) {
    return rewriter.notifyMatchFailure(unPackOp,
                                       "non-identity outer dims perm NYI");
  }

  Location loc = unPackOp->getLoc();
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(unPackOp);

  RankedTensorType packedTensorType = unPackOp.getSourceType();
  int64_t packedRank = packedTensorType.getRank();

  OpFoldResult zero = rewriter.getIndexAttr(0), one = rewriter.getIndexAttr(1);
  auto destTensorType = cast<RankedTensorType>(unPackOp.getDest().getType());
  if (unPackOp.isLikeUnPad()) {
    // This unpack is just a plain unpad.
    // Just extract the slice from the higher ranked tensor.
    ArrayRef<int64_t> destShape = destTensorType.getShape();
    // The inner dimensions stay the same as the destination tensor, but the
    // outer ones are additional 1s.
    SmallVector<OpFoldResult> sizes(packedRank - destShape.size(), one);
    sizes.append(tensor::getMixedSizes(rewriter, loc, unPackOp.getDest()));

    auto extractSliceOp = rewriter.create<tensor::ExtractSliceOp>(
        loc, destTensorType, unPackOp.getSource(),
        SmallVector<OpFoldResult>(packedRank, zero), sizes,
        SmallVector<OpFoldResult>(packedRank, one));

    rewriter.replaceOp(unPackOp, extractSliceOp->getResults());

    return LowerUnPackOpResult{/*emptyOp=*/nullptr, /*transposeOp=*/nullptr,
                               /*reshapeOp=*/nullptr, extractSliceOp};
  }
  // 2. Compute the permutation vector to move the last `numPackedDims` into
  // the `innerPosDims` of a shape of rank `packedRank`.
  int64_t numPackedDims = unPackOp.getInnerDimsPos().size();
  auto lastDims = llvm::to_vector(
      llvm::seq<int64_t>(packedRank - numPackedDims, packedRank));
  PackingMetadata packingMetadata =
      computePackingMetadata(packedRank, unPackOp.getInnerDimsPos());
  SmallVector<int64_t> lastDimsToInsertPositionsPerm = computePermutationVector(
      packedRank, lastDims, packingMetadata.insertPositions);

  // 3. Compute the stripMinedShape: this is the packed shape without outer and
  // inner permutations.
  SmallVector<int64_t> stripMinedShape(packedTensorType.getShape());
  applyPermutationToVector(stripMinedShape, lastDimsToInsertPositionsPerm);

  // 4. Transpose packedShape to stripMinedShape.
  RankedTensorType stripMinedTensorType =
      RankedTensorType::Builder(packedTensorType).setShape(stripMinedShape);
  RankedTensorType collapsedType = tensor::CollapseShapeOp::inferCollapsedType(
      stripMinedTensorType, packingMetadata.reassociations);

  // Get dynamic dims from input tensor based on lastDimsToInsertPositionsPerm
  // permutation.
  SmallVector<OpFoldResult, 4> dims =
      tensor::getMixedSizes(rewriter, loc, unPackOp.getSource());
  applyPermutationToVector(dims, lastDimsToInsertPositionsPerm);
  auto emptyOp = rewriter.create<tensor::EmptyOp>(
      loc, dims, stripMinedTensorType.getElementType());
  auto transposeOp = rewriter.create<linalg::TransposeOp>(
      loc, unPackOp.getSource(), emptyOp, lastDimsToInsertPositionsPerm);

  LLVM_DEBUG(
      DBGSNL(); DBGSNL(); llvm::interleaveComma(packingMetadata.insertPositions,
                                                DBGS() << "insertPositions: ");
      DBGSNL(); llvm::interleaveComma(packedTensorType.getShape(),
                                      DBGS() << "packedShape: ");
      DBGSNL();
      llvm::interleaveComma(lastDimsToInsertPositionsPerm,
                            DBGS() << "lastDimsToInsertPositionsPerm: ");
      DBGSNL(); llvm::interleaveComma(
          packingMetadata.reassociations, DBGS() << "reassociations: ",
          [&](ReassociationIndices ri) {
            llvm::interleaveComma(ri, llvm::dbgs() << "|");
          });
      DBGSNL();
      llvm::interleaveComma(stripMinedShape, DBGS() << "stripMinedShape: ");
      DBGSNL(); DBGS() << "collapsed type: " << collapsedType; DBGSNL(););

  // 5. Collapse from the stripMinedShape to the padded result.
  auto reshapeOp = rewriter.create<tensor::CollapseShapeOp>(
      loc, collapsedType, transposeOp->getResult(0),
      packingMetadata.reassociations);

  // 6. ExtractSlice.
  int64_t destRank = destTensorType.getRank();
  auto extractSliceOp = rewriter.create<tensor::ExtractSliceOp>(
      loc, destTensorType, reshapeOp->getResult(0),
      SmallVector<OpFoldResult>(destRank, zero),
      tensor::getMixedSizes(rewriter, loc, unPackOp.getDest()),
      SmallVector<OpFoldResult>(destRank, one));

  // 7. Inject a copy to preserve DPS.
  auto copyOp = rewriter.create<linalg::CopyOp>(
      loc, extractSliceOp->getResult(0), unPackOp.getDest());

  // 8. Replace unPackOp by extractSliceOp.
  rewriter.replaceOp(unPackOp, copyOp->getResults());

  return LowerUnPackOpResult{emptyOp, transposeOp, reshapeOp, extractSliceOp};
}

SmallVector<int64_t>
PackedOperandsDimList::extractPackedDimsForOperand(int64_t operandPos) {
  SmallVector<int64_t> res;
  for (auto &i : spec) {
    if (!i.packedDimForEachOperand[operandPos].has_value())
      continue;
    res.push_back(i.packedDimForEachOperand[operandPos].value());
  }
  return res;
}

SmallVector<OpFoldResult>
PackedOperandsDimList::extractPackSizesForOperand(int64_t operandPos) {
  SmallVector<OpFoldResult> res;
  for (auto &i : spec) {
    if (!i.packedDimForEachOperand[operandPos].has_value())
      continue;
    res.push_back(i.packedSize);
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
    listOfPackedOperandsDim.pushBack(std::move(packedOperandsDims));

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
  SmallVector<OpOperand *> initOperands = llvm::to_vector(llvm::map_range(
      linalgOp.getDpsInitsMutable(), [](OpOperand &o) { return &o; }));
  SmallVector<OpOperand *> inputOperands = linalgOp.getDpsInputOperands();
  for (const auto &operandsList : {inputOperands, initOperands}) {
    for (OpOperand *opOperand : operandsList) {
      int64_t pos = opOperand->getOperandNumber();
      Value operand = opOperand->get();
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
      ShapedType operandType = operand.getType().cast<ShapedType>();
      bool areConstantTiles =
          llvm::all_of(innerPackSizes, [](OpFoldResult tile) {
            return getConstantIntValue(tile).has_value();
          });
      if (areConstantTiles && operandType.hasStaticShape() &&
          !tensor::PackOp::requirePaddingValue(
              operandType.getShape(), innerPos,
              dest.getType().cast<ShapedType>().getShape(), {},
              innerPackSizes)) {
        packOps.push_back(rewriter.create<tensor::PackOp>(
            loc, operand, dest, innerPos, innerPackSizes));
      } else {
        // TODO: value of the padding attribute should be determined by
        // consumers.
        auto zeroAttr =
            rewriter.getZeroAttr(getElementTypeOrSelf(dest.getType()));
        Value zero = rewriter.create<arith::ConstantOp>(loc, zeroAttr);
        packOps.push_back(rewriter.create<tensor::PackOp>(
            loc, operand, dest, innerPos, innerPackSizes, zero));
      }
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
      cast<RankedTensorType>(opOperand.get().getType()), permutation);
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
// packMatmulGreedily transformation.
//===----------------------------------------------------------------------===//

/// Pack a LinalgOp by greedily inferring matmul dimensions (m, n, k) where m
/// and n are proper parallel dimensions and k is a proper reduction
/// dimension. Packing occurs by rewriting the op as a linalg.generic and
/// calling linalg::pack by `mnkPackedSizes`. The order of the packed
/// dimensions is customizable: the `mnkOrder` is a permutation of {0, 1, 2}
/// to reorder {m, n, k} into one of the 8 possible forms. The outer
/// dimensions of the operands are not permuted at this time, this is left for
/// future work.
FailureOr<PackResult>
linalg::packMatmulGreedily(RewriterBase &rewriter, LinalgOp linalgOp,
                           ArrayRef<OpFoldResult> mnkPackedSizes,
                           ArrayRef<int64_t> mnkPaddedSizesNextMultipleOf,
                           ArrayRef<int64_t> mnkOrder) {
  assert(mnkPackedSizes.size() == 3 && "unexpected num of packing sizes");
  assert((mnkPaddedSizesNextMultipleOf.empty() ||
          mnkPaddedSizesNextMultipleOf.size() == 3) &&
         "num of packing sizes next multiple should be empty or of size 3");
  assert(mnkOrder.size() == 3 && "unexpected mnkOrder size");
  assert(isPermutationVector(mnkOrder) && "expected a permutation");

  int64_t numLoops = linalgOp.getNumLoops();
  if (numLoops <= 2) {
    LLVM_DEBUG(DBGS() << "need 3+ loops to find a matmul to pack, got "
                      << numLoops << "\nin: " << linalgOp << "\n");
    return rewriter.notifyMatchFailure(
        linalgOp, "need 3+ loops to find a matmul to pack");
  }

  // Locally adjust the desired iterator position of mnk and packing sizes.
  int64_t numPackedDims = mnkPackedSizes.size();
  SmallVector<int64_t> mmnnkkPos(numPackedDims);
  for (int64_t i = 0, e = numPackedDims; i < e; ++i)
    mmnnkkPos[i] = numLoops - numPackedDims + mnkOrder[i];
  SmallVector<OpFoldResult> packedSizes(numPackedDims);
  for (int64_t i = 0, e = numPackedDims; i < e; ++i)
    packedSizes[mnkOrder[i]] = mnkPackedSizes[i];
  SmallVector<int64_t> paddedSizesNextMultipleOf(numPackedDims);
  for (int64_t i = 0, e = numPackedDims; i < e; ++i) {
    paddedSizesNextMultipleOf[mnkOrder[i]] =
        mnkPaddedSizesNextMultipleOf.empty() ? 0
                                             : mnkPaddedSizesNextMultipleOf[i];
  }

  // 1. Infer dims that are important for matmul.
  FailureOr<ContractionDimensions> maybeDimensions =
      inferContractionDims(linalgOp);
  if (failed(maybeDimensions)) {
    LLVM_DEBUG(DBGS() << "couldn't infer matmul iterators in: " << linalgOp
                      << "\n");
    return rewriter.notifyMatchFailure(linalgOp,
                                       "couldn't infer matmul iterators");
  }

  // 2. Normalize linalgOp to an kmn-matmul-like with [red, par, par] most
  // minor iterators. In cases with multiple options for m, n, k bias towards
  // the most minor embedding.
  // If we wanted a different normalization order, this is where it would have
  // to plug a heuristic.
  int64_t mPos = maybeDimensions->m.back(), nPos = maybeDimensions->n.back(),
          kPos = maybeDimensions->k.back();
  LLVM_DEBUG(DBGSNL(); DBGSNL(); DBGSNL();
             DBGS() << "Start packing generic op greedily with (m@" << mPos
                    << ", n@" << nPos << ", k@" << kPos << "): " << linalgOp
                    << "\n";);

  // 2.a. Rewrite as a generic.
  auto genericOp = dyn_cast<GenericOp>(linalgOp.getOperation());
  if (!genericOp) {
    FailureOr<GenericOp> generalizeResult =
        generalizeNamedOp(rewriter, linalgOp);
    assert(succeeded(generalizeResult) && "unexpected failure generalizing op");
    genericOp = *generalizeResult;
  }

  // 2.b. Interchange to move the dimensions (k, m, n) as most-minor
  // iterators. Note that this only normalized the iteration order and does
  // not change the indexings of any operand.
  SmallVector<int64_t> permutation =
      computePermutationVector(numLoops, {mPos, nPos, kPos}, mmnnkkPos);
  LLVM_DEBUG(llvm::interleaveComma(permutation, DBGS() << "perm: "); DBGSNL(););
  // Sign .. unsigned pollution.
  SmallVector<unsigned> unsignedPerm(permutation.begin(), permutation.end());
  FailureOr<GenericOp> interchangeResult =
      interchangeGenericOp(rewriter, genericOp, unsignedPerm);
  assert(succeeded(interchangeResult) && "unexpected failure interchanging op");
  genericOp = *interchangeResult;
  LLVM_DEBUG(DBGS() << "Generalized Op to pack: " << genericOp << "\n";);

  // At this point, the op iterators are normalized to {leading, k, m, n}.
  // The layouts induced by packing will always be:
  //   - LHS{leading_lhs, kk, mm}
  //   - RHS{leading_rhs, kk, nn}
  //   - RES{leading_res, mm, nn}
  // If we wanted to change the packed order, we would reorder (k, m, n) to
  // something else above.
  //
  // Additional permutations of the outer dims of the operands (i.e.
  // leading_lhs, leading_rhs and leading_res) could follow by computing the
  // desired outerPerm for each operand.
  // This is left for future work.

  // TODO: this creates too much IR, go use reifyResultShapes.
  SmallVector<Range, 4> loopRanges =
      cast<LinalgOp>(genericOp.getOperation())
          .createLoopRanges(rewriter, genericOp.getLoc());

  // Add leading zeros to match numLoops, we only pack the last 3 dimensions
  // post interchange.
  LLVM_DEBUG(llvm::interleaveComma(paddedSizesNextMultipleOf,
                                   DBGS() << "paddedSizesNextMultipleOf: ");
             DBGSNL(););
  LLVM_DEBUG(llvm::interleaveComma(loopRanges, DBGS() << "loopRanges: ",
                                   [](Range r) { llvm::dbgs() << r.size; });
             DBGSNL(););
  SmallVector<OpFoldResult> adjustedPackedSizes(numLoops - packedSizes.size(),
                                                rewriter.getIndexAttr(0));
  for (int64_t i = 0, e = numPackedDims; i < e; ++i) {
    if (paddedSizesNextMultipleOf[i] == 0) {
      adjustedPackedSizes.push_back(packedSizes[i]);
      continue;
    }
    AffineExpr d0, s0;
    bindDims(rewriter.getContext(), d0);
    bindSymbols(rewriter.getContext(), s0);
    adjustedPackedSizes.push_back(affine::makeComposedFoldedAffineApply(
        rewriter, genericOp->getLoc(), d0.ceilDiv(s0) * s0,
        {loopRanges[adjustedPackedSizes.size()].size,
         rewriter.getIndexAttr(paddedSizesNextMultipleOf[i])}));
  }
  LLVM_DEBUG(llvm::interleaveComma(adjustedPackedSizes,
                                   DBGS() << "adjustedPackedSizes: ");
             DBGSNL(););

  // TODO: If we wanted to give the genericOp a name after packing, after
  // calling `pack` would be a good time. One would still need to check that
  // `containsMostMinorMatmul(packingRes->packedLinalgOp)` is true, since we
  // also allow degenerate matmul cases (i.e. matvec, dot).
  return pack(rewriter, genericOp, adjustedPackedSizes);
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

LogicalResult mlir::linalg::CopyVectorizationPattern::matchAndRewrite(
    memref::CopyOp copyOp, PatternRewriter &rewriter) const {
  return vectorizeCopy(rewriter, copyOp);
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
    if (auto val = llvm::dyn_cast_if_present<Value>(ofr))
      return val;
    return rewriter
        .create<arith::ConstantIndexOp>(
            padOp.getLoc(), cast<IntegerAttr>(ofr.get<Attribute>()).getInt())
        .getResult();
  };

  auto resultType = padOp.getResultType();
  // Compute size of EmptyOp. Any combination of static/dynamic is supported.
  SmallVector<Value> dynSizes;
  SmallVector<int64_t> staticSizes;
  for (unsigned dim = 0; dim < resultType.getRank(); ++dim) {
    if (resultType.isDynamicDim(dim)) {
      auto srcSize = getIdxValue(tensor::getMixedSize(rewriter, padOp.getLoc(),
                                                      padOp.getSource(), dim));
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
  SmallVector<OpFoldResult> srcSizes =
      tensor::getMixedSizes(rewriter, padOp.getLoc(), padOp.getSource());
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

// Normalizes a permutation on a higher rank space to its actual size, e.g.
//   perm = [1, 4, 2]
// becomes
//   norm = [0, 2, 1]
static SmallVector<int64_t>
getPackUnpackNormalizedPerm(int rank, ArrayRef<int64_t> perm) {
  constexpr int64_t kNonTiledMarker = -1;
  SmallVector<int64_t> vec(rank, kNonTiledMarker);
  for (auto [index, value] : llvm::enumerate(perm))
    vec[value] = index;
  SmallVector<int64_t> normalizedPerm = llvm::to_vector(llvm::make_filter_range(
      vec, [&](int64_t v) { return v != kNonTiledMarker; }));
  // This inverts the permutation in addition to normalizing so invert back.
  return invertPermutationVector(normalizedPerm);
}

// Gets the normalized permutation implied by innerDimsPos and outerDimsPerm
// assuming rank reduction of unit outer dims.
static SmallVector<int64_t>
getPackUnpackRankReducedPerm(ArrayRef<int64_t> shape,
                             ArrayRef<int64_t> innerDimsPos,
                             ArrayRef<int64_t> outerDimsPerm) {
  SmallVector<int64_t> rankReducedOuterDimsPerm;
  SmallVector<int64_t> outerDims;
  SmallVector<int64_t> innerDims;
  int64_t dim = 0;
  int64_t unpackedRank = shape.size();
  for (auto i : llvm::seq<unsigned>(0, unpackedRank)) {
    if (llvm::is_contained(innerDimsPos, i)) {
      innerDims.push_back(dim++);
      continue;
    }
    if (shape[i] == 1)
      continue;
    outerDims.push_back(dim++);
    if (!outerDimsPerm.empty())
      rankReducedOuterDimsPerm.push_back(outerDimsPerm[i]);
  }

  // Get the position of the inner dims after permutation.
  SmallVector<int64_t> innerPerm =
      getPackUnpackNormalizedPerm(unpackedRank, innerDimsPos);
  applyPermutationToVector<int64_t>(innerDims, innerPerm);

  // Ditto for the outer dims.
  SmallVector<int64_t> perm = outerDims;

  rankReducedOuterDimsPerm =
      getPackUnpackNormalizedPerm(unpackedRank, rankReducedOuterDimsPerm);
  if (!rankReducedOuterDimsPerm.empty())
    applyPermutationToVector<int64_t>(perm, rankReducedOuterDimsPerm);

  // The tile always ends up as the inner most dims after packing.
  perm.append(innerDims);

  return perm;
}

LogicalResult GeneralizeOuterUnitDimsPackOpPattern::matchAndRewrite(
    tensor::PackOp packOp, PatternRewriter &rewriter) const {
  if (llvm::any_of(packOp.getMixedTiles(),
                   [](OpFoldResult tile) { return tile.is<Value>(); })) {
    return rewriter.notifyMatchFailure(packOp,
                                       "require inner tile sizes being static");
  }

  // TODO: support the case that outer dimensions are not all 1s. A
  // tensor.expand_shape will be generated in this case.
  auto innerDimsPos = packOp.getInnerDimsPos();
  int64_t srcRank = packOp.getSourceRank();
  auto destShape = packOp.getDestType().getShape();
  if (llvm::any_of(innerDimsPos, [destShape](int64_t index) {
        return destShape[index] != 1;
      })) {
    return rewriter.notifyMatchFailure(
        packOp, "require the tiled outer dimensions of the result are all 1s");
  }

  // 1. Use rank-reduced tensor.extract_slice op to extract the tile and untiled
  // outer dims.
  Location loc = packOp.getLoc();
  Value input = getPackOpSourceOrPaddedSource(rewriter, packOp);
  auto inputShape = packOp.getSourceType().getShape();
  DenseMap<int64_t, OpFoldResult> dimAndTileMapping =
      packOp.getDimAndTileMapping();
  Attribute zeroIdxAttr = rewriter.getIndexAttr(0);
  Attribute oneIdxAttr = rewriter.getIndexAttr(1);
  SmallVector<OpFoldResult> readOffsets(srcRank, zeroIdxAttr);
  SmallVector<OpFoldResult> readStrides(srcRank, oneIdxAttr);
  SmallVector<OpFoldResult> readSizes;
  SmallVector<int64_t> readShape;
  for (auto i : llvm::seq<unsigned>(0, srcRank)) {
    if (dimAndTileMapping.count(i)) {
      readShape.push_back(getConstantIntValue(dimAndTileMapping[i])
                              .value_or(ShapedType::kDynamic));
      readSizes.push_back(dimAndTileMapping[i]);
      continue;
    }
    if (ShapedType::isDynamic(inputShape[i])) {
      readSizes.push_back(
          rewriter.create<tensor::DimOp>(loc, input, i).getResult());
    } else {
      readSizes.push_back(rewriter.getIndexAttr(inputShape[i]));
    }
    if (inputShape[i] != 1)
      readShape.push_back(inputShape[i]);
  }

  Type elemType = packOp.getSourceType().getElementType();
  auto readType = RankedTensorType::get(readShape, elemType);

  Value tile = rewriter.create<tensor::ExtractSliceOp>(
      loc, readType, input, readOffsets, readSizes, readStrides);

  // 2. Transpose the tile to match the inner tile order.

  SmallVector<int64_t> perm = getPackUnpackRankReducedPerm(
      inputShape, innerDimsPos, packOp.getOuterDimsPerm());

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
  SmallVector<OpFoldResult> writeSizes =
      tensor::getMixedSizes(rewriter, loc, packOp.getDest());

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
  ArrayRef<int64_t> innerDimsPos = unpackOp.getInnerDimsPos();
  if (llvm::any_of(innerDimsPos, [srcShape](int64_t index) {
        return srcShape[index] != 1;
      })) {
    return rewriter.notifyMatchFailure(
        unpackOp,
        "require the tiled outer dimensions of the result are all 1s");
  }

  // 1. Use rank-reduced tensor.extract_slice op to extract the tile.
  Location loc = unpackOp.getLoc();
  Value source = unpackOp.getSource();
  DenseMap<int64_t, OpFoldResult> dimAndTileMapping =
      unpackOp.getDimAndTileMapping();
  Attribute zeroIdxAttr = rewriter.getIndexAttr(0);
  Attribute oneIdxAttr = rewriter.getIndexAttr(1);
  SmallVector<OpFoldResult> readOffsets(srcRank, zeroIdxAttr);
  SmallVector<OpFoldResult> readStrides(srcRank, oneIdxAttr);
  SmallVector<OpFoldResult> readSizes;
  SmallVector<int64_t> readShape;
  SmallVector<Value> dynamicDims;
  for (auto i : llvm::seq<unsigned>(0, destRank)) {
    if (dimAndTileMapping.count(i)) {
      readSizes.push_back(oneIdxAttr);
      continue;
    }

    if (ShapedType::isDynamic(srcShape[i])) {
      Value dynamicDim =
          rewriter.create<tensor::DimOp>(loc, source, i).getResult();
      readSizes.push_back(dynamicDim);
      dynamicDims.push_back(dynamicDim);
    } else {
      readSizes.push_back(rewriter.getIndexAttr(srcShape[i]));
    }
    if (srcShape[i] != 1)
      readShape.push_back(srcShape[i]);
  }
  auto mixedTiles = unpackOp.getMixedTiles();
  readSizes.append(mixedTiles.begin(), mixedTiles.end());

  // Explicitly create the type for extract_slice op because the inner tile
  // size could be 1. We want to represent the whole inner tile in this case.
  auto tileShape = srcShape.drop_front(destRank);
  // Append the inner tile shape to the permuted and rank-reduced outer shape.
  readShape.append(tileShape.begin(), tileShape.end());
  Type elemType = unpackOp.getSourceType().getElementType();
  auto readType = RankedTensorType::get(readShape, elemType);
  Value innerTile = rewriter.create<tensor::ExtractSliceOp>(
      loc, readType, unpackOp.getSource(), readOffsets, readSizes, readStrides);

  // 2. Transpose the tile to match the outer corresponding tile order.
  SmallVector<int64_t> perm = getPackUnpackRankReducedPerm(
      srcShape.take_front(destRank), innerDimsPos, unpackOp.getOuterDimsPerm());
  // Unpack is a transition out of packed space so we invert the permutation.
  perm = invertPermutationVector(perm);
  SmallVector<int64_t> transpShape(readShape);
  applyPermutationToVector<int64_t>(transpShape, perm);

  Value empty =
      rewriter.create<tensor::EmptyOp>(loc, transpShape, elemType, dynamicDims);
  auto transposedOp =
      rewriter.create<linalg::TransposeOp>(loc, innerTile, empty, perm);

  // 3. Handle in-complete tiles if needed. It truncates trailing data from the
  // transposed tile.
  int numLoops = transpShape.size();
  SmallVector<OpFoldResult> tileStrides(numLoops, oneIdxAttr);
  SmallVector<OpFoldResult> tileOffsets(numLoops, zeroIdxAttr);
  SmallVector<OpFoldResult> tileSizes;
  ArrayRef<int64_t> destShape = unpackOp.getDestType().getShape();
  for (auto i : llvm::seq<unsigned>(0, destRank)) {
    if (dimAndTileMapping.count(i) || destShape[i] != 1)
      tileSizes.push_back(
          tensor::getMixedSize(rewriter, loc, unpackOp.getDest(), i));
  }

  auto partialTile = rewriter.create<tensor::ExtractSliceOp>(
      loc, transposedOp.getResult()[0], tileOffsets, tileSizes, tileStrides);

  // 4. Insert the result to the destination tensor.
  SmallVector<OpFoldResult> writeSizes;
  SmallVector<OpFoldResult> writeStrides(destRank, oneIdxAttr);
  SmallVector<OpFoldResult> writeOffsets(destRank, zeroIdxAttr);
  for (int i = 0, idx = 0; i < destRank; ++i) {
    if (dimAndTileMapping.count(i) || destShape[i] != 1)
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
  if (convOp.hasPureBufferSemantics())
    return failure(); // To be implemented.

  Value input = convOp.getInputs().front();
  Value kernel = convOp.getInputs().back();
  Value output = convOp.getOutputs().front();

  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  auto kernelType = dyn_cast<RankedTensorType>(kernel.getType());
  auto outputType = dyn_cast<RankedTensorType>(output.getType());

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
  if (convOp.hasPureBufferSemantics())
    return failure(); // To be implemented.

  Value input = convOp.getInputs().front();
  Value kernel = convOp.getInputs().back();
  Value output = convOp.getOutputs().front();

  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  auto kernelType = dyn_cast<RankedTensorType>(kernel.getType());
  auto outputType = dyn_cast<RankedTensorType>(output.getType());

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
  if (convOp.hasPureBufferSemantics())
    return failure(); // To be implemented.

  Value input = convOp.getInputs().front();
  Value kernel = convOp.getInputs().back();
  Value output = convOp.getOutputs().front();

  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  auto kernelType = dyn_cast<RankedTensorType>(kernel.getType());
  auto outputType = dyn_cast<RankedTensorType>(output.getType());

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
