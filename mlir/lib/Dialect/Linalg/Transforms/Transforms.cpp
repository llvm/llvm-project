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
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/InterleavedRange.h"
#include "llvm/Support/raw_ostream.h"
#include <utility>

#define DEBUG_TYPE "linalg-transforms"

using namespace mlir;
using namespace mlir::linalg;

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

static std::string stringifyReassocIndices(ReassociationIndicesRef ri) {
  return llvm::interleaved(ri, ", ", /*Prefix=*/"|", /*Suffix=*/"");
}

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
                                             linalg::PackOp packOp,
                                             bool lowerPadLikeWithInsertSlice) {
  // 1. Filter out NYI cases.
  auto packedTensorType =
      cast<RankedTensorType>(packOp->getResultTypes().front());
  if (llvm::any_of(packOp.getStaticInnerTiles(), ShapedType::isDynamic)) {
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
      getPackInverseDestPerm(packOp);

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
    paddingValue = arith::ConstantOp::create(
        rewriter, loc, rewriter.getZeroAttr(getElementTypeOrSelf(collapsed)));
  }
  auto padOp =
      tensor::PadOp::create(rewriter, loc, collapsed, packOp.getSource(), lows,
                            highs, paddingValue, /*nofold=*/false);

  LDBG() << "insertPositions: "
         << llvm::interleaved(packingMetadata.insertPositions);
  LDBG() << "outerPositions: "
         << llvm::interleaved(packingMetadata.outerPositions);
  LDBG() << "packedShape: " << llvm::interleaved(packedTensorType.getShape());
  LDBG() << "packedToStripMinedShapePerm: "
         << llvm::interleaved(packedToStripMinedShapePerm);
  LDBG() << "reassociations: "
         << llvm::interleaved(llvm::map_range(packingMetadata.reassociations,
                                              stringifyReassocIndices));
  LDBG() << "stripMinedShape: " << llvm::interleaved(stripMinedShape);
  LDBG() << "collapsed type: " << collapsed;

  if (lowerPadLikeWithInsertSlice && packOp.isLikePad()) {
    // Pack ops which operate as simple pads may not produce legal
    // tensor.insert_slice operations when the packed type does not rank reduce
    // to the padded type.
    SliceVerificationResult rankReduces =
        isRankReducedType(packedTensorType, padOp.getResultType());

    if (rankReduces == SliceVerificationResult::Success) {
      // This pack is just a plain pad.
      // Just insert the pad in the higher ranked tensor.
      // Offsets.
      SmallVector<OpFoldResult> zeros(packOp.getDestRank(),
                                      rewriter.getIndexAttr(0));
      // Strides.
      SmallVector<OpFoldResult> ones(packOp.getDestRank(),
                                     rewriter.getIndexAttr(1));
      SmallVector<OpFoldResult> sizes =
          tensor::getMixedSizes(rewriter, loc, packOp.getDest());

      auto insertSliceOp = tensor::InsertSliceOp::create(
          rewriter, loc, /*source=*/padOp, /*dest=*/packOp.getDest(),
          /*offsets=*/zeros, sizes, /*strides=*/ones);

      LDBG() << "insert_slice op: " << insertSliceOp;

      rewriter.replaceOp(packOp, insertSliceOp->getResults());

      return LowerPackResult{padOp, /*reshapeOp=*/nullptr,
                             /*transposeOp=*/nullptr};
    }
  }

  // 5. Expand from the padded result to the stripMinedShape.
  auto expandShapeResultType =
      RankedTensorType::Builder(packedTensorType).setShape(stripMinedShape);
  auto reshapeOp = tensor::ExpandShapeOp::create(
      rewriter, loc, expandShapeResultType, padOp.getResult(),
      packingMetadata.reassociations);

  // 6. Transpose stripMinedShape to packedShape.
  SmallVector<int64_t> transpPerm =
      invertPermutationVector(packedToStripMinedShapePerm);
  auto transposeOp = linalg::TransposeOp::create(
      rewriter, loc, reshapeOp.getResult(), packOp.getDest(), transpPerm);

  LDBG() << "reshape op: " << reshapeOp;
  LDBG() << "transpPerm: " << llvm::interleaved(transpPerm);
  LDBG() << "transpose op: " << transposeOp;

  // 7. Replace packOp by transposeOp.
  rewriter.replaceOp(packOp, transposeOp->getResults());

  return LowerPackResult{padOp, reshapeOp, transposeOp};
}

FailureOr<LowerUnPackOpResult>
linalg::lowerUnPack(RewriterBase &rewriter, linalg::UnPackOp unPackOp,
                    bool lowerUnpadLikeWithExtractSlice) {
  Location loc = unPackOp->getLoc();
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(unPackOp);

  RankedTensorType packedTensorType = unPackOp.getSourceType();
  int64_t packedRank = packedTensorType.getRank();

  OpFoldResult zero = rewriter.getIndexAttr(0), one = rewriter.getIndexAttr(1);
  auto destTensorType = cast<RankedTensorType>(unPackOp.getDest().getType());
  if (lowerUnpadLikeWithExtractSlice && unPackOp.isLikeUnPad()) {
    // This unpack is just a plain unpad.
    // Just extract the slice from the higher ranked tensor.
    ArrayRef<int64_t> destShape = destTensorType.getShape();
    // The inner dimensions stay the same as the destination tensor, but the
    // outer ones are additional 1s.
    SmallVector<OpFoldResult> sizes(packedRank - destShape.size(), one);
    sizes.append(tensor::getMixedSizes(rewriter, loc, unPackOp.getDest()));

    auto extractSliceOp = tensor::ExtractSliceOp::create(
        rewriter, loc, destTensorType, unPackOp.getSource(),
        SmallVector<OpFoldResult>(packedRank, zero), sizes,
        SmallVector<OpFoldResult>(packedRank, one));

    rewriter.replaceOp(unPackOp, extractSliceOp->getResults());

    return LowerUnPackOpResult{/*emptyOp=*/nullptr, /*transposeOp=*/nullptr,
                               /*reshapeOp=*/nullptr, extractSliceOp};
  }

  // 1. Compute the permutation vector to shuffle packed shape into the shape
  // before any outer or inner permutations have been applied.
  PackingMetadata packingMetadata;
  SmallVector<int64_t> packedToStripMinedShapePerm =
      getUnPackInverseSrcPerm(unPackOp, packingMetadata);

  // 2. Compute the stripMinedShape: this is the packed shape without outer and
  // inner permutations.
  SmallVector<int64_t> stripMinedShape(packedTensorType.getShape());
  applyPermutationToVector(stripMinedShape, packedToStripMinedShapePerm);

  // 3. Transpose packedShape to stripMinedShape.
  RankedTensorType stripMinedTensorType =
      RankedTensorType::Builder(packedTensorType).setShape(stripMinedShape);
  RankedTensorType collapsedType = tensor::CollapseShapeOp::inferCollapsedType(
      stripMinedTensorType, packingMetadata.reassociations);

  // Get dynamic dims from input tensor based on packedToStripMinedShapePerm
  // permutation.
  SmallVector<OpFoldResult, 4> dims =
      tensor::getMixedSizes(rewriter, loc, unPackOp.getSource());
  applyPermutationToVector(dims, packedToStripMinedShapePerm);
  auto emptyOp = tensor::EmptyOp::create(rewriter, loc, dims,
                                         stripMinedTensorType.getElementType());
  auto transposeOp =
      linalg::TransposeOp::create(rewriter, loc, unPackOp.getSource(), emptyOp,
                                  packedToStripMinedShapePerm);

  LDBG() << "insertPositions: "
         << llvm::interleaved(packingMetadata.insertPositions);
  LDBG() << "packedShape: " << llvm::interleaved(packedTensorType.getShape());
  LDBG() << "packedToStripMinedShapePerm: "
         << llvm::interleaved(packedToStripMinedShapePerm);
  LDBG() << "reassociations: "
         << llvm::interleaved(llvm::map_range(packingMetadata.reassociations,
                                              stringifyReassocIndices));
  LDBG() << "stripMinedShape: " << llvm::interleaved(stripMinedShape);
  LDBG() << "collapsed type: " << collapsedType;

  // 4. Collapse from the stripMinedShape to the padded result.
  auto reshapeOp = tensor::CollapseShapeOp::create(
      rewriter, loc, collapsedType, transposeOp->getResult(0),
      packingMetadata.reassociations);

  // 5. ExtractSlice.
  int64_t destRank = destTensorType.getRank();
  auto extractSliceOp = tensor::ExtractSliceOp::create(
      rewriter, loc, destTensorType, reshapeOp->getResult(0),
      SmallVector<OpFoldResult>(destRank, zero),
      tensor::getMixedSizes(rewriter, loc, unPackOp.getDest()),
      SmallVector<OpFoldResult>(destRank, one));

  // 6. Inject a copy to preserve DPS.
  auto copyOp = linalg::CopyOp::create(
      rewriter, loc, extractSliceOp->getResult(0), unPackOp.getDest());

  // 7. Replace unPackOp by copyOp.
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
  LDBG() << "Start packing: " << linalgOp;
  LDBG() << "maps: " << llvm::interleaved(indexingMaps);
  LDBG() << "iterators: " << llvm::interleaved(iteratorTypes);

  SmallVector<linalg::PackOp> packOps;
  SmallVector<linalg::UnPackOp> unPackOps;
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

    LDBG() << "++++ After pack size #" << i << ": " << packedSizes[i];
    LDBG() << "maps: " << llvm::interleaved(indexingMaps);
    LDBG() << "iterators: " << llvm::interleaved(iteratorTypes);
    LDBG() << "packedDimForEachOperand: "
           << llvm::interleaved(packedOperandsDims.packedDimForEachOperand);
  }

  // Step 2. Propagate packing to all LinalgOp operands.
  SmallVector<Value> inputsAndInits, results;
  SmallVector<OpOperand *> initOperands =
      llvm::to_vector(llvm::make_pointer_range(linalgOp.getDpsInitsMutable()));
  SmallVector<OpOperand *> inputOperands = linalgOp.getDpsInputOperands();
  for (const auto &operandsList : {inputOperands, initOperands}) {
    for (OpOperand *opOperand : operandsList) {
      int64_t pos = opOperand->getOperandNumber();
      Value operand = opOperand->get();
      SmallVector<int64_t> innerPos =
          listOfPackedOperandsDim.extractPackedDimsForOperand(pos);
      SmallVector<OpFoldResult> innerPackSizes =
          listOfPackedOperandsDim.extractPackSizesForOperand(pos);
      LDBG() << "operand: " << operand;
      LDBG() << "innerPos: " << llvm::interleaved(innerPos);
      LDBG() << "innerPackSizes: " << llvm::interleaved(innerPackSizes);
      if (innerPackSizes.empty()) {
        inputsAndInits.push_back(operand);
        continue;
      }
      Value dest = linalg::PackOp::createDestinationTensor(
          rewriter, loc, operand, innerPackSizes, innerPos,
          /*outerDimsPerm=*/{});
      ShapedType operandType = cast<ShapedType>(operand.getType());
      bool areConstantTiles =
          llvm::all_of(innerPackSizes, [](OpFoldResult tile) {
            return getConstantIntValue(tile).has_value();
          });
      if (areConstantTiles && operandType.hasStaticShape() &&
          !linalg::PackOp::requirePaddingValue(
              operandType.getShape(), innerPos,
              cast<ShapedType>(dest.getType()).getShape(), {},
              innerPackSizes)) {
        packOps.push_back(linalg::PackOp::create(rewriter, loc, operand, dest,
                                                 innerPos, innerPackSizes));
      } else {
        // TODO: value of the padding attribute should be determined by
        // consumers.
        auto zeroAttr =
            rewriter.getZeroAttr(getElementTypeOrSelf(dest.getType()));
        Value zero = arith::ConstantOp::create(rewriter, loc, zeroAttr);
        packOps.push_back(linalg::PackOp::create(
            rewriter, loc, operand, dest, innerPos, innerPackSizes, zero));
      }
      inputsAndInits.push_back(packOps.back());
    }
  }

  // Step 3. Build the packed op, use the type of `inits` as result types.
  ValueRange inputs =
      ValueRange{inputsAndInits}.take_front(linalgOp.getNumDpsInputs());
  ValueRange inits =
      ValueRange{inputsAndInits}.take_back(linalgOp.getNumDpsInits());
  auto packedLinalgOp =
      linalg::GenericOp::create(rewriter, linalgOp.getLoc(), inits.getTypes(),
                                inputs, inits, indexingMaps, iteratorTypes);
  packedLinalgOp.getRegion().takeBody(linalgOp->getRegion(0));

  // Step 4. Propagate packing to all the op results.
  for (OpResult result : packedLinalgOp->getResults()) {
    int64_t resultNum = result.getResultNumber();
    linalg::PackOp maybePackedInit =
        inits[resultNum].getDefiningOp<linalg::PackOp>();
    if (!maybePackedInit) {
      results.push_back(result);
      continue;
    }
    // Build the symmetrical UnPackOp to the existing PackOp.
    unPackOps.push_back(linalg::UnPackOp::create(
        rewriter, packedLinalgOp->getLoc(), result, maybePackedInit.getSource(),
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
  auto transposedGenericOp = linalg::GenericOp::create(
      rewriter,
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
linalg::packTranspose(RewriterBase &rewriter, linalg::PackOp packOp,
                      linalg::LinalgOp linalgOp, linalg::UnPackOp maybeUnPackOp,
                      ArrayRef<int64_t> outerPerm,
                      ArrayRef<int64_t> innerPerm) {
  Location loc = linalgOp.getLoc();

  // Step 1. Transpose packOp.
  rewriter.setInsertionPoint(packOp);
  linalg::PackOp transposedPackOp =
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
  linalg::UnPackOp transposedUnPackOp;
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
    LDBG() << "need 3+ loops to find a matmul to pack, got " << numLoops
           << " in: " << linalgOp;
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
    LDBG() << "couldn't infer matmul iterators in: " << linalgOp;
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
  LDBG() << "Start packing generic op greedily with (m@" << mPos << ", n@"
         << nPos << ", k@" << kPos << "): " << linalgOp;

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
  LDBG() << "perm: " << llvm::interleaved(permutation);
  // Sign .. unsigned pollution.
  SmallVector<unsigned> unsignedPerm(permutation.begin(), permutation.end());
  FailureOr<GenericOp> interchangeResult =
      interchangeGenericOp(rewriter, genericOp, unsignedPerm);
  assert(succeeded(interchangeResult) && "unexpected failure interchanging op");
  genericOp = *interchangeResult;
  LDBG() << "Generalized Op to pack: " << genericOp;

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
  LDBG() << "paddedSizesNextMultipleOf: "
         << llvm::interleaved(paddedSizesNextMultipleOf);
  LDBG() << "loopRanges: "
         << llvm::interleaved(
                llvm::map_range(loopRanges, [](Range r) { return r.size; }));
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
  LDBG() << "adjustedPackedSizes: " << llvm::interleaved(adjustedPackedSizes);

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
  SmallVector<int64_t, 4> tileSizes(ts);
  tileSizeComputationFunction = [tileSizes](OpBuilder &b, Operation *op) {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(
        &op->getParentOfType<func::FuncOp>().getBody().front());
    return llvm::to_vector<4>(map_range(tileSizes, [&](int64_t s) {
      Value v = arith::ConstantIndexOp::create(b, op->getLoc(), s);
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
Value DecomposePadOpPattern::createFillOrGenerateOp(
    RewriterBase &rewriter, tensor::PadOp padOp, Value dest,
    const SmallVector<Value> &dynSizes) const {
  auto padValue = padOp.getConstantPaddingValue();
  if (padValue) {
    // Move the padding value defined inside the PadOp block to outside.
    if (padValue.getParentBlock() == &padOp.getRegion().front())
      rewriter.moveOpBefore(padValue.getDefiningOp(), padOp);
    return FillOp::create(rewriter, padOp.getLoc(), padValue, dest).result();
  }

  // Fill could not be optimized: Lower to tensor::GenerateOp with region.
  auto generateOp = tensor::GenerateOp::create(rewriter, padOp.getLoc(),
                                               padOp.getResultType(), dynSizes);
  // Copy region to new op.
  IRMapping bvm;
  padOp.getRegion().cloneInto(&generateOp.getRegion(), bvm);
  return generateOp;
}

LogicalResult
DecomposePadOpPattern::matchAndRewrite(tensor::PadOp padOp,
                                       PatternRewriter &rewriter) const {
  // Given an OpFoldResult, return an index-typed value.
  auto getIdxValue = [&](OpFoldResult ofr) {
    if (auto val = llvm::dyn_cast_if_present<Value>(ofr))
      return val;
    return arith::ConstantIndexOp::create(
               rewriter, padOp.getLoc(),
               cast<IntegerAttr>(cast<Attribute>(ofr)).getInt())
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
  Value emptyTensor =
      tensor::EmptyOp::create(rewriter, padOp.getLoc(), staticSizes,
                              resultType.getElementType(), dynSizes);
  Value fill = createFillOrGenerateOp(rewriter, padOp, emptyTensor, dynSizes);

  // Generate a InsertSliceOp for copying the PadOp source.
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

  RankedTensorType sourceType = sliceOp.getSourceType();
  RankedTensorType resultType = sliceOp.getResultType();

  // If the extract_slice is not rank-reduced, all shapes are static and the
  // data source is actually used. Rewrite into pad(extract_slice(x)).
  if (sourceType.getRank() == resultType.getRank()) {
    rewriter.replaceOp(sliceOp, tilingResult->tiledValues);
    return success();
  }

  // Handle rank-reduced slice by creating another extract_slice op.
  Value rankReduced = tensor::createCanonicalRankReducingExtractSliceOp(
      rewriter, sliceOp.getLoc(), tilingResult->tiledValues[0], resultType);

  rewriter.replaceOp(sliceOp, rankReduced);
  return success();
}

/// If padding value is set, returns a tensor.pad Op for the source tensor,
/// with the output shape matching the output of `packOp`. Otherwise, returns
/// the source directly.
///
/// This method assumes that all outer dims for this pack Op are 1.
static Value getPackOpSourceOrPaddedSource(OpBuilder &builder,
                                           linalg::PackOp packOp) {
  Value input = packOp.getSource();
  if (!packOp.getPaddingValue()) {
    return input;
  }

  assert(llvm::all_of(packOp.getAllOuterDims(),
                      [](int64_t val) { return val == 1; }) &&
         "some outer dims are != 1");

  Location loc = packOp.getLoc();
  ShapedType inputType = packOp.getSourceType();
  int64_t inputRank = inputType.getRank();

  DenseMap<int64_t, OpFoldResult> tileAndPosMapping =
      packOp.getDimAndTileMapping();

  // The sizes of dynamic tiles
  SmallVector<Value> dynamicTileSizes;

  // Collect dims for the padded shape.
  SmallVector<int64_t> paddedShape;
  for (int64_t dimIdx = 0; dimIdx < inputRank; ++dimIdx) {
    // 1. Non-tiled outer dims.
    // These dims should be 1 and we simply preserve them.
    if (!tileAndPosMapping.count(dimIdx)) {
      int64_t inputDimSize = inputType.getDimSize(dimIdx);
      assert(inputDimSize == 1 &&
             "with all outer dims == 1, this non-tiled input dim should be 1!");
      paddedShape.push_back(inputDimSize);
      continue;
    }

    // 2. Tiled outer dims
    // As all outer dims == 1, it is safe to use the tile size for the padded
    // shape.
    OpFoldResult tileSizeForDim = tileAndPosMapping.lookup(dimIdx);

    // 2.1 Static tile sizes
    std::optional<int64_t> cstTileSize = getConstantIntValue(tileSizeForDim);
    if (cstTileSize.has_value()) {
      paddedShape.push_back(cstTileSize.value());
      continue;
    }

    // 2.2 Dynamic tile sizes
    paddedShape.push_back(ShapedType::kDynamic);

    // Get the value that holds the dynamic size.
    dynamicTileSizes.push_back(llvm::dyn_cast<Value>(tileSizeForDim));
  }
  auto resultType =
      RankedTensorType::get(paddedShape, inputType.getElementType());
  return tensor::createPadHighOp(resultType, input, packOp.getPaddingValue(),
                                 /*nofold=*/false, loc, builder,
                                 dynamicTileSizes);
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
  SmallVector<int64_t> normalizedPerm = llvm::filter_to_vector(
      vec, [&](int64_t v) { return v != kNonTiledMarker; });
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

LogicalResult DecomposeOuterUnitDimsPackOpPattern::matchAndRewrite(
    linalg::PackOp packOp, PatternRewriter &rewriter) const {
  // TODO: support the case that outer dimensions are not all 1s. A
  // tensor.expand_shape will be generated in this case.
  if (llvm::any_of(packOp.getAllOuterDims(),
                   [](int64_t dim) { return dim != 1; })) {
    return rewriter.notifyMatchFailure(
        packOp, "not all outer dimensions of the result are 1s");
  }

  Attribute zeroIdxAttr = rewriter.getIndexAttr(0);
  Attribute oneIdxAttr = rewriter.getIndexAttr(1);
  Location loc = packOp.getLoc();

  Value input = getPackOpSourceOrPaddedSource(rewriter, packOp);
  DenseMap<int64_t, OpFoldResult> dimAndTileMapping =
      packOp.getDimAndTileMapping();
  int64_t srcRank = packOp.getSourceRank();
  int64_t destRank = packOp.getDestRank();
  int64_t numTiles = destRank - srcRank;

  // 1. Extract the inner tile sizes.
  // Where possible, values are replaced with constant attributes (to match the
  // behaviour of `getPackOpSourceOrPaddedSource`).
  SmallVector<OpFoldResult> tileSizes;
  for (auto i : llvm::seq<unsigned>(0, srcRank)) {
    if (dimAndTileMapping.count(i)) {
      // Rather than taking the tile size as is, extact the actual constant
      // value Attribute where possible, e.g.:
      //    [Value: %tile_size = arith.constant 8 : index] --> [Attribute: 8]
      auto [_, tileSize] =
          getSimplifiedOfrAndStaticSizePair(dimAndTileMapping[i], rewriter);
      tileSizes.push_back(tileSize);
    }
  }

  // 2. Transpose the input to match the inner tile order:
  //    %init = tensor.empty()
  //    %transposed_tile = linalg.transpose ins(%source_or_padded_source),
  //                                        outs(%init)
  // Assumptions made:
  //  1. All outer dims are 1 - the corresponding transposition order doesn't
  //     matter, but requires all dim indices to be present.
  SmallVector<int64_t> srcPermForTranspose;
  ArrayRef<int64_t> innerDimPos(packOp.getInnerDimsPos());
  for (int64_t i = 0; i < srcRank; i++) {
    // We assume the `k` dimensions of the inner dim position, where `k` is the
    // rank of the inner tiling, correspond to the last `k` indices of the
    // transpose permutation. This is done by adding the indices not contained
    // in the inner dimension position in order from 0 to `n`. Where n is the
    // rank of the source tensor. For example if we have a source tensor with
    // indices [0, 1, 2, 3] and inner dim position of [3, 0], the remaining
    // indices are [1, 2]. and the transpose will be [1, 2, 3, 0].
    if (llvm::is_contained(innerDimPos, i))
      continue;
    srcPermForTranspose.push_back(i);
  }
  srcPermForTranspose.append(innerDimPos.begin(), innerDimPos.end());

  LDBG() << "Pack permutation: " << packOp;
  LDBG() << "perm: " << llvm::interleaved(srcPermForTranspose);

  // 2.1 Create tensor.empty (init value for TransposeOp)
  SmallVector<OpFoldResult> transShapeForEmptyOp(srcRank - numTiles,
                                                 oneIdxAttr);
  transShapeForEmptyOp.append(tileSizes);

  applyPermutationToVector<OpFoldResult>(transShapeForEmptyOp,
                                         srcPermForTranspose);
  Value empty =
      tensor::EmptyOp::create(rewriter, loc, transShapeForEmptyOp,
                              packOp.getSourceType().getElementType());

  // 2.2 Create linalg.transpose
  auto transposedOp = linalg::TransposeOp::create(rewriter, loc, input, empty,
                                                  srcPermForTranspose);

  // 3. Insert the inner tile to the destination:
  //  %inserted_tile = tensor.insert_slice(%transposed_tile)
  SmallVector<OpFoldResult> writeStrides(destRank, oneIdxAttr);
  SmallVector<OpFoldResult> writeOffsets(destRank, zeroIdxAttr);
  // Outer dims are all 1s!
  SmallVector<OpFoldResult> writeSizes(destRank - dimAndTileMapping.size(),
                                       oneIdxAttr);
  SmallVector<int64_t> writeShape;

  for (auto tileSize : packOp.getMixedTiles()) {
    auto [tileSizeStatic, tileSizeOfr] =
        getSimplifiedOfrAndStaticSizePair(tileSize, rewriter);
    writeSizes.push_back(tileSizeOfr);
    writeShape.push_back(tileSizeStatic);
  }

  // 4. Replace tensor.packOp with tensor.insert_slice created above
  auto insert = tensor::InsertSliceOp::create(
      rewriter, loc, transposedOp.getResult()[0], packOp.getDest(),
      writeOffsets, writeSizes, writeStrides);
  rewriter.replaceOp(packOp, insert.getResult());

  return success();
}

LogicalResult DecomposeOuterUnitDimsUnPackOpPattern::matchAndRewrite(
    linalg::UnPackOp unpackOp, PatternRewriter &rewriter) const {
  int64_t srcRank = unpackOp.getSourceRank();
  int64_t destRank = unpackOp.getDestRank();
  ArrayRef<int64_t> srcShape = unpackOp.getSourceType().getShape();
  ArrayRef<int64_t> innerDimsPos = unpackOp.getInnerDimsPos();
  if (llvm::any_of(unpackOp.getTiledOuterDims(),
                   [](int64_t dim) { return dim != 1; })) {
    return rewriter.notifyMatchFailure(
        unpackOp,
        "require the tiled outer dimensions of the result are all 1s");
  }

  // 1. Use rank-reduced tensor.extract_slice op to extract the tile:
  //    %extracted_tile = tensor.extract_slice(%unpack_op_input)
  Location loc = unpackOp.getLoc();
  Value source = unpackOp.getSource();
  DenseMap<int64_t, OpFoldResult> dimAndTileMapping =
      unpackOp.getDimAndTileMapping();
  Attribute zeroIdxAttr = rewriter.getIndexAttr(0);
  Attribute oneIdxAttr = rewriter.getIndexAttr(1);

  // The shape for ExtractSliceOp. Note that this will consist of 3 blocks of
  // dims:
  //    [ outer-untiled-dims, outer-tiled-dims, tile-sizes ]
  SmallVector<int64_t> readShapeForExtractSlice;
  // The sizes attribute for ExtractSliceOp. Due to rank-reducing (and
  // outer-tiled-dims being all 1), this will be
  //    [ outer-untiled-dims, tile-sizes ]
  SmallVector<OpFoldResult> extractSliceSizes;
  // The offset and strides attributes for ExtractSliceOp.
  SmallVector<OpFoldResult> extractSliceOffsets(srcRank, zeroIdxAttr);
  SmallVector<OpFoldResult> extractSliceStrides(srcRank, oneIdxAttr);

  // Shape for EmptyOp that's used as the init value for TransposeOp below.
  // This should be:
  //    [ outer-untiled-dims, tile-sizes ]
  // However, skip unit dims - TransposeOp (below) applies rank-reduced
  // permutation.
  SmallVector<OpFoldResult> shapeForEmptyOp;

  for (auto i : llvm::seq<unsigned>(0, destRank)) {
    // Compute sizes attribute for ExtractSliceOp - outer-tiled-dims.
    //
    // As all outer tiled dims are 1, so the corresponding
    // slice size to read will also 1. As this will be rank-reducing "extract
    // slice" (i.e. the unit dims will be "collapsed"), there's no need to
    // update:
    //  * the output shape for ExtractSliceOp, nor
    //  * the shape for EmptyOp.
    if (dimAndTileMapping.count(i)) {
      extractSliceSizes.push_back(oneIdxAttr);
      continue;
    }

    // Compute sizes attribute for ExtractSliceOp + EmptyOp -
    // outer-untiled-dims
    if (ShapedType::isDynamic(srcShape[i])) {
      OpFoldResult dynamicDim =
          tensor::DimOp::create(rewriter, loc, source, i).getResult();
      extractSliceSizes.push_back(dynamicDim);
      shapeForEmptyOp.push_back(dynamicDim);
    } else {
      extractSliceSizes.push_back(rewriter.getIndexAttr(srcShape[i]));
      if (srcShape[i] != 1)
        shapeForEmptyOp.push_back(rewriter.getIndexAttr(srcShape[i]));
    }
    // Compute the output shape for ExtractSliceOp  - outer-untiled-dims (take
    // into account rank-reducing)
    if (srcShape[i] != 1) {
      readShapeForExtractSlice.push_back(srcShape[i]);
    }
  }
  // Append the tile sizes to "sizes attribute" for ExtractSliceOp and the
  // shape for EmptyOp.
  auto mixedTiles = unpackOp.getMixedTiles();
  extractSliceSizes.append(mixedTiles.begin(), mixedTiles.end());
  shapeForEmptyOp.append(mixedTiles.begin(), mixedTiles.end());

  // Explicitly create the type for extract_slice op because the inner tile
  // size could be 1. We want to represent the whole inner tile in this case.
  auto tileShape = srcShape.drop_front(destRank);
  // Append the inner tile shape to the permuted and rank-reduced outer shape.
  readShapeForExtractSlice.append(tileShape.begin(), tileShape.end());
  Type elemType = unpackOp.getSourceType().getElementType();
  auto readType = RankedTensorType::get(readShapeForExtractSlice, elemType);
  Value innerTile = tensor::ExtractSliceOp::create(
      rewriter, loc, readType, unpackOp.getSource(), extractSliceOffsets,
      extractSliceSizes, extractSliceStrides);

  // 2. Transpose the tile to match the outer corresponding tile order.
  SmallVector<int64_t> perm = getPackUnpackRankReducedPerm(
      srcShape.take_front(destRank), innerDimsPos, unpackOp.getOuterDimsPerm());
  // Unpack is a transition out of packed space so we invert the permutation.
  perm = invertPermutationVector(perm);
  applyPermutationToVector<OpFoldResult>(shapeForEmptyOp, perm);

  Value empty =
      tensor::EmptyOp::create(rewriter, loc, shapeForEmptyOp, elemType);
  auto transposedOp =
      linalg::TransposeOp::create(rewriter, loc, innerTile, empty, perm);

  // 3. Handle in-complete tiles if needed. It truncates trailing data from the
  // transposed tile.
  int numLoops = shapeForEmptyOp.size();
  SmallVector<OpFoldResult> tileStrides(numLoops, oneIdxAttr);
  SmallVector<OpFoldResult> tileOffsets(numLoops, zeroIdxAttr);
  SmallVector<OpFoldResult> tileSizes;
  ArrayRef<int64_t> destShape = unpackOp.getDestType().getShape();
  for (auto i : llvm::seq<unsigned>(0, destRank)) {
    if (dimAndTileMapping.count(i) || destShape[i] != 1)
      tileSizes.push_back(
          tensor::getMixedSize(rewriter, loc, unpackOp.getDest(), i));
  }

  auto partialTile =
      tensor::ExtractSliceOp::create(rewriter, loc, transposedOp.getResult()[0],
                                     tileOffsets, tileSizes, tileStrides);

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
  auto insert = tensor::InsertSliceOp::create(rewriter, loc, partialTile,
                                              unpackOp.getDest(), writeOffsets,
                                              writeSizes, writeStrides);
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
          .DefaultUnreachable("unexpected conv2d/pool2d operation.");

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

  auto conv1DOp = Conv1DOp::create(
      rewriter, loc, newOutputType, ValueRange{newInput, newKernel},
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

  auto conv1DOp = DepthwiseConv1DNwcWcOp::create(
      rewriter, loc, newOutputType, ValueRange{newInput, newKernel},
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

  auto conv1DOp =
      Conv1DOp::create(rewriter, loc, newOutputType,
                       ValueRange{newInput, newKernel}, ValueRange{newOutput});

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

void linalg::populateDecomposePackUnpackPatterns(RewritePatternSet &patterns) {
  patterns.add<DecomposeOuterUnitDimsPackOpPattern>(patterns.getContext());
  patterns.add<DecomposeOuterUnitDimsUnPackOpPattern>(patterns.getContext());
}

void linalg::populateDecomposePadPatterns(RewritePatternSet &patterns) {
  patterns.add<DecomposePadOpPattern>(patterns.getContext());
}
