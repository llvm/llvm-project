//===- ExpandStridedMetadata.cpp - Simplify this operation -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// The pass expands memref operations that modify the metadata of a memref
/// (sizes, offset, strides) into a sequence of easier to analyze constructs.
/// In particular, this pass transforms operations into explicit sequence of
/// operations that model the effect of this operation on the different
/// metadata. This pass uses affine constructs to materialize these effects.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include <optional>

namespace mlir {
namespace memref {
#define GEN_PASS_DEF_EXPANDSTRIDEDMETADATA
#include "mlir/Dialect/MemRef/Transforms/Passes.h.inc"
} // namespace memref
} // namespace mlir
using namespace mlir;

namespace {
/// Replace `dst = subview(memref, subOffset, subSizes, subStrides))`
/// With
///
/// \verbatim
/// baseBuffer, baseOffset, baseSizes, baseStrides =
///     extract_strided_metadata(memref)
/// strides#i = baseStrides#i * subSizes#i
/// offset = baseOffset + sum(subOffset#i * baseStrides#i)
/// sizes = subSizes
/// dst = reinterpret_cast baseBuffer, offset, sizes, strides
/// \endverbatim
///
/// In other words, get rid of the subview in that expression and canonicalize
/// on its effects on the offset, the sizes, and the strides using affine.apply.
struct SubviewFolder : public OpRewritePattern<memref::SubViewOp> {
public:
  using OpRewritePattern<memref::SubViewOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::SubViewOp subview,
                                PatternRewriter &rewriter) const override {
    // Build a plain extract_strided_metadata(memref) from subview(memref).
    Location origLoc = subview.getLoc();
    Value source = subview.getSource();
    auto sourceType = source.getType().cast<MemRefType>();
    unsigned sourceRank = sourceType.getRank();

    auto newExtractStridedMetadata =
        rewriter.create<memref::ExtractStridedMetadataOp>(origLoc, source);

    auto [sourceStrides, sourceOffset] = getStridesAndOffset(sourceType);

    // Compute the new strides and offset from the base strides and offset:
    // newStride#i = baseStride#i * subStride#i
    // offset = baseOffset + sum(subOffsets#i * newStrides#i)
    SmallVector<OpFoldResult> strides;
    SmallVector<OpFoldResult> subStrides = subview.getMixedStrides();
    auto origStrides = newExtractStridedMetadata.getStrides();

    // Hold the affine symbols and values for the computation of the offset.
    SmallVector<OpFoldResult> values(2 * sourceRank + 1);
    SmallVector<AffineExpr> symbols(2 * sourceRank + 1);

    detail::bindSymbolsList(rewriter.getContext(), symbols);
    AffineExpr expr = symbols.front();
    values[0] = ShapedType::isDynamic(sourceOffset)
                    ? getAsOpFoldResult(newExtractStridedMetadata.getOffset())
                    : rewriter.getIndexAttr(sourceOffset);
    SmallVector<OpFoldResult> subOffsets = subview.getMixedOffsets();

    AffineExpr s0 = rewriter.getAffineSymbolExpr(0);
    AffineExpr s1 = rewriter.getAffineSymbolExpr(1);
    for (unsigned i = 0; i < sourceRank; ++i) {
      // Compute the stride.
      OpFoldResult origStride =
          ShapedType::isDynamic(sourceStrides[i])
              ? origStrides[i]
              : OpFoldResult(rewriter.getIndexAttr(sourceStrides[i]));
      strides.push_back(makeComposedFoldedAffineApply(
          rewriter, origLoc, s0 * s1, {subStrides[i], origStride}));

      // Build up the computation of the offset.
      unsigned baseIdxForDim = 1 + 2 * i;
      unsigned subOffsetForDim = baseIdxForDim;
      unsigned origStrideForDim = baseIdxForDim + 1;
      expr = expr + symbols[subOffsetForDim] * symbols[origStrideForDim];
      values[subOffsetForDim] = subOffsets[i];
      values[origStrideForDim] = origStride;
    }

    // Compute the offset.
    OpFoldResult finalOffset =
        makeComposedFoldedAffineApply(rewriter, origLoc, expr, values);

    // The final result is  <baseBuffer, offset, sizes, strides>.
    // Thus we need 1 + 1 + subview.getRank() + subview.getRank(), to hold all
    // the values.
    auto subType = subview.getType().cast<MemRefType>();
    unsigned subRank = subType.getRank();

    // The sizes of the final type are defined directly by the input sizes of
    // the subview.
    // Moreover subviews can drop some dimensions, some strides and sizes may
    // not end up in the final <base, offset, sizes, strides> value that we are
    // replacing.
    // Do the filtering here.
    SmallVector<OpFoldResult> subSizes = subview.getMixedSizes();
    llvm::SmallBitVector droppedDims = subview.getDroppedDims();

    SmallVector<OpFoldResult> finalSizes;
    finalSizes.reserve(subRank);

    SmallVector<OpFoldResult> finalStrides;
    finalStrides.reserve(subRank);

    for (unsigned i = 0; i < sourceRank; ++i) {
      if (droppedDims.test(i))
        continue;

      finalSizes.push_back(subSizes[i]);
      finalStrides.push_back(strides[i]);
    }
    assert(finalSizes.size() == subRank &&
           "Should have populated all the values at this point");

    auto memrefDesc = rewriter.create<memref::ReinterpretCastOp>(
        origLoc, subType, newExtractStridedMetadata.getBaseBuffer(),
        finalOffset,
        /*sizes=*/finalSizes,
        /*strides=*/finalStrides);
    rewriter.replaceOp(subview, memrefDesc.getResult());
    return success();
  }
};

/// Compute the expanded sizes of the given \p expandShape for the
/// \p groupId-th reassociation group.
/// \p origSizes hold the sizes of the source shape as values.
/// This is used to compute the new sizes in cases of dynamic shapes.
///
/// sizes#i =
///     baseSizes#groupId / product(expandShapeSizes#j,
///                                  for j in group excluding reassIdx#i)
/// Where reassIdx#i is the reassociation index at index i in \p groupId.
///
/// \post result.size() == expandShape.getReassociationIndices()[groupId].size()
///
/// TODO: Move this utility function directly within ExpandShapeOp. For now,
/// this is not possible because this function uses the Affine dialect and the
/// MemRef dialect cannot depend on the Affine dialect.
static SmallVector<OpFoldResult>
getExpandedSizes(memref::ExpandShapeOp expandShape, OpBuilder &builder,
                 ArrayRef<OpFoldResult> origSizes, unsigned groupId) {
  SmallVector<int64_t, 2> reassocGroup =
      expandShape.getReassociationIndices()[groupId];
  assert(!reassocGroup.empty() &&
         "Reassociation group should have at least one dimension");

  unsigned groupSize = reassocGroup.size();
  SmallVector<OpFoldResult> expandedSizes(groupSize);

  uint64_t productOfAllStaticSizes = 1;
  Optional<unsigned> dynSizeIdx;
  MemRefType expandShapeType = expandShape.getResultType();

  // Fill up all the statically known sizes.
  for (unsigned i = 0; i < groupSize; ++i) {
    uint64_t dimSize = expandShapeType.getDimSize(reassocGroup[i]);
    if (ShapedType::isDynamic(dimSize)) {
      assert(!dynSizeIdx && "There must be at most one dynamic size per group");
      dynSizeIdx = i;
      continue;
    }
    productOfAllStaticSizes *= dimSize;
    expandedSizes[i] = builder.getIndexAttr(dimSize);
  }

  // Compute the dynamic size using the original size and all the other known
  // static sizes:
  // expandSize = origSize / productOfAllStaticSizes.
  if (dynSizeIdx) {
    AffineExpr s0 = builder.getAffineSymbolExpr(0);
    expandedSizes[*dynSizeIdx] = makeComposedFoldedAffineApply(
        builder, expandShape.getLoc(), s0.floorDiv(productOfAllStaticSizes),
        origSizes[groupId]);
  }

  return expandedSizes;
}

/// Compute the expanded strides of the given \p expandShape for the
/// \p groupId-th reassociation group.
/// \p origStrides and \p origSizes hold respectively the strides and sizes
/// of the source shape as values.
/// This is used to compute the strides in cases of dynamic shapes and/or
/// dynamic stride for this reassociation group.
///
/// strides#i =
///     origStrides#reassDim * product(expandShapeSizes#j, for j in
///                                    reassIdx#i+1..reassIdx#i+group.size-1)
///
/// Where reassIdx#i is the reassociation index for at index i in \p groupId
/// and expandShapeSizes#j is either:
/// - The constant size at dimension j, derived directly from the result type of
///   the expand_shape op, or
/// - An affine expression: baseSizes#reassDim / product of all constant sizes
///   in expandShapeSizes. (Remember expandShapeSizes has at most one dynamic
///   element.)
///
/// \post result.size() == expandShape.getReassociationIndices()[groupId].size()
///
/// TODO: Move this utility function directly within ExpandShapeOp. For now,
/// this is not possible because this function uses the Affine dialect and the
/// MemRef dialect cannot depend on the Affine dialect.
SmallVector<OpFoldResult> getExpandedStrides(memref::ExpandShapeOp expandShape,
                                             OpBuilder &builder,
                                             ArrayRef<OpFoldResult> origSizes,
                                             ArrayRef<OpFoldResult> origStrides,
                                             unsigned groupId) {
  SmallVector<int64_t, 2> reassocGroup =
      expandShape.getReassociationIndices()[groupId];
  assert(!reassocGroup.empty() &&
         "Reassociation group should have at least one dimension");

  unsigned groupSize = reassocGroup.size();
  MemRefType expandShapeType = expandShape.getResultType();

  Optional<int64_t> dynSizeIdx;

  // Fill up the expanded strides, with the information we can deduce from the
  // resulting shape.
  uint64_t currentStride = 1;
  SmallVector<OpFoldResult> expandedStrides(groupSize);
  for (int i = groupSize - 1; i >= 0; --i) {
    expandedStrides[i] = builder.getIndexAttr(currentStride);
    uint64_t dimSize = expandShapeType.getDimSize(reassocGroup[i]);
    if (ShapedType::isDynamic(dimSize)) {
      assert(!dynSizeIdx && "There must be at most one dynamic size per group");
      dynSizeIdx = i;
      continue;
    }

    currentStride *= dimSize;
  }

  // Collect the statically known information about the original stride.
  Value source = expandShape.getSrc();
  auto sourceType = source.getType().cast<MemRefType>();
  auto [strides, offset] = getStridesAndOffset(sourceType);

  OpFoldResult origStride =
      ShapedType::isDynamic(strides[groupId])
          ? origStrides[groupId]
          : builder.getIndexAttr(strides[groupId]);

  // Apply the original stride to all the strides.
  int64_t doneStrideIdx = 0;
  // If we saw a dynamic dimension, we need to fix-up all the strides up to
  // that dimension with the dynamic size.
  if (dynSizeIdx) {
    int64_t productOfAllStaticSizes = currentStride;
    assert(ShapedType::isDynamic(sourceType.getDimSize(groupId)) &&
           "We shouldn't be able to change dynamicity");
    OpFoldResult origSize = origSizes[groupId];

    AffineExpr s0 = builder.getAffineSymbolExpr(0);
    AffineExpr s1 = builder.getAffineSymbolExpr(1);
    for (; doneStrideIdx < *dynSizeIdx; ++doneStrideIdx) {
      int64_t baseExpandedStride = expandedStrides[doneStrideIdx]
                                       .get<Attribute>()
                                       .cast<IntegerAttr>()
                                       .getInt();
      expandedStrides[doneStrideIdx] = makeComposedFoldedAffineApply(
          builder, expandShape.getLoc(),
          (s0 * baseExpandedStride).floorDiv(productOfAllStaticSizes) * s1,
          {origSize, origStride});
    }
  }

  // Now apply the origStride to the remaining dimensions.
  AffineExpr s0 = builder.getAffineSymbolExpr(0);
  for (; doneStrideIdx < groupSize; ++doneStrideIdx) {
    int64_t baseExpandedStride = expandedStrides[doneStrideIdx]
                                     .get<Attribute>()
                                     .cast<IntegerAttr>()
                                     .getInt();
    expandedStrides[doneStrideIdx] = makeComposedFoldedAffineApply(
        builder, expandShape.getLoc(), s0 * baseExpandedStride, {origStride});
  }

  return expandedStrides;
}

/// Produce an OpFoldResult object with \p builder at \p loc representing
/// `prod(valueOrConstant#i, for i in {indices})`,
/// where valueOrConstant#i is maybeConstant[i] when \p isDymamic is false,
/// values[i] otherwise.
///
/// \pre for all index in indices: index < values.size()
/// \pre for all index in indices: index < maybeConstants.size()
static OpFoldResult
getProductOfValues(ArrayRef<int64_t> indices, OpBuilder &builder, Location loc,
                   ArrayRef<int64_t> maybeConstants,
                   ArrayRef<OpFoldResult> values,
                   llvm::function_ref<bool(int64_t)> isDynamic) {
  AffineExpr productOfValues = builder.getAffineConstantExpr(1);
  SmallVector<OpFoldResult> inputValues;
  unsigned numberOfSymbols = 0;
  unsigned groupSize = indices.size();
  for (unsigned i = 0; i < groupSize; ++i) {
    productOfValues =
        productOfValues * builder.getAffineSymbolExpr(numberOfSymbols++);
    unsigned srcIdx = indices[i];
    int64_t maybeConstant = maybeConstants[srcIdx];

    inputValues.push_back(isDynamic(maybeConstant)
                              ? values[srcIdx]
                              : builder.getIndexAttr(maybeConstant));
  }

  return makeComposedFoldedAffineApply(builder, loc, productOfValues,
                                       inputValues);
}

/// Compute the collapsed size of the given \p collpaseShape for the
/// \p groupId-th reassociation group.
/// \p origSizes hold the sizes of the source shape as values.
/// This is used to compute the new sizes in cases of dynamic shapes.
///
/// Conceptually this helper function computes:
/// `prod(origSizes#i, for i in {ressociationGroup[groupId]})`.
///
/// \post result.size() == 1, in other words, each group collapse to one
/// dimension.
///
/// TODO: Move this utility function directly within CollapseShapeOp. For now,
/// this is not possible because this function uses the Affine dialect and the
/// MemRef dialect cannot depend on the Affine dialect.
static SmallVector<OpFoldResult>
getCollapsedSize(memref::CollapseShapeOp collapseShape, OpBuilder &builder,
                 ArrayRef<OpFoldResult> origSizes, unsigned groupId) {
  SmallVector<OpFoldResult> collapsedSize;

  MemRefType collapseShapeType = collapseShape.getResultType();

  uint64_t size = collapseShapeType.getDimSize(groupId);
  if (!ShapedType::isDynamic(size)) {
    collapsedSize.push_back(builder.getIndexAttr(size));
    return collapsedSize;
  }

  // We are dealing with a dynamic size.
  // Build the affine expr of the product of the original sizes involved in that
  // group.
  Value source = collapseShape.getSrc();
  auto sourceType = source.getType().cast<MemRefType>();

  SmallVector<int64_t, 2> reassocGroup =
      collapseShape.getReassociationIndices()[groupId];

  collapsedSize.push_back(getProductOfValues(
      reassocGroup, builder, collapseShape.getLoc(), sourceType.getShape(),
      origSizes, ShapedType::isDynamic));

  return collapsedSize;
}

/// Compute the collapsed stride of the given \p collpaseShape for the
/// \p groupId-th reassociation group.
/// \p origStrides and \p origSizes hold respectively the strides and sizes
/// of the source shape as values.
/// This is used to compute the strides in cases of dynamic shapes and/or
/// dynamic stride for this reassociation group.
///
/// Conceptually this helper function returns the stride of the inner most
/// dimension of that group in the original shape.
///
/// \post result.size() == 1, in other words, each group collapse to one
/// dimension.
static SmallVector<OpFoldResult>
getCollapsedStride(memref::CollapseShapeOp collapseShape, OpBuilder &builder,
                   ArrayRef<OpFoldResult> origSizes,
                   ArrayRef<OpFoldResult> origStrides, unsigned groupId) {
  SmallVector<int64_t, 2> reassocGroup =
      collapseShape.getReassociationIndices()[groupId];
  assert(!reassocGroup.empty() &&
         "Reassociation group should have at least one dimension");

  Value source = collapseShape.getSrc();
  auto sourceType = source.getType().cast<MemRefType>();

  auto [strides, offset] = getStridesAndOffset(sourceType);

  SmallVector<OpFoldResult> groupStrides;
  ArrayRef<int64_t> srcShape = sourceType.getShape();
  for (int64_t currentDim : reassocGroup) {
    // Skip size-of-1 dimensions, since right now their strides may be
    // meaningless.
    // FIXME: size-of-1 dimensions shouldn't be used in collapse shape, unless
    // they are truly contiguous. When they are truly contiguous, we shouldn't
    // need to skip them.
    if (srcShape[currentDim] == 1)
      continue;

    int64_t currentStride = strides[currentDim];
    groupStrides.push_back(ShapedType::isDynamic(currentStride)
                               ? origStrides[currentDim]
                               : builder.getIndexAttr(currentStride));
  }
  if (groupStrides.empty()) {
    // We're dealing with a 1x1x...x1 shape. The stride is meaningless,
    // but we still have to make the type system happy.
    MemRefType collapsedType = collapseShape.getResultType();
    auto [collapsedStrides, collapsedOffset] =
        getStridesAndOffset(collapsedType);
    int64_t finalStride = collapsedStrides[groupId];
    if (ShapedType::isDynamic(finalStride)) {
      // Look for a dynamic stride. At this point we don't know which one is
      // desired, but they are all equally good/bad.
      for (int64_t currentDim : reassocGroup) {
        assert(srcShape[currentDim] == 1 &&
               "We should be dealing with 1x1x...x1");

        if (ShapedType::isDynamic(strides[currentDim]))
          return {origStrides[currentDim]};
      }
      llvm_unreachable("We should have found a dynamic stride");
    }
    return {builder.getIndexAttr(finalStride)};
  }

  // For the general case, we just want the minimum stride
  // since the collapsed dimensions are contiguous.
  auto minMap = AffineMap::getMultiDimIdentityMap(groupStrides.size(),
                                                  builder.getContext());
  return {makeComposedFoldedAffineMin(builder, collapseShape.getLoc(), minMap,
                                      groupStrides)};
}
/// Replace `baseBuffer, offset, sizes, strides =
///              extract_strided_metadata(reshapeLike(memref))`
/// With
///
/// \verbatim
/// baseBuffer, offset, baseSizes, baseStrides =
///     extract_strided_metadata(memref)
/// sizes = getReshapedSizes(reshapeLike)
/// strides = getReshapedStrides(reshapeLike)
/// \endverbatim
///
///
/// Notice that `baseBuffer` and `offset` are unchanged.
///
/// In other words, get rid of the expand_shape in that expression and
/// materialize its effects on the sizes and the strides using affine apply.
template <typename ReassociativeReshapeLikeOp,
          SmallVector<OpFoldResult> (*getReshapedSizes)(
              ReassociativeReshapeLikeOp, OpBuilder &,
              ArrayRef<OpFoldResult> /*origSizes*/, unsigned /*groupId*/),
          SmallVector<OpFoldResult> (*getReshapedStrides)(
              ReassociativeReshapeLikeOp, OpBuilder &,
              ArrayRef<OpFoldResult> /*origSizes*/,
              ArrayRef<OpFoldResult> /*origStrides*/, unsigned /*groupId*/)>
struct ReshapeFolder : public OpRewritePattern<ReassociativeReshapeLikeOp> {
public:
  using OpRewritePattern<ReassociativeReshapeLikeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReassociativeReshapeLikeOp reshape,
                                PatternRewriter &rewriter) const override {
    // Build a plain extract_strided_metadata(memref) from
    // extract_strided_metadata(reassociative_reshape_like(memref)).
    Location origLoc = reshape.getLoc();
    Value source = reshape.getSrc();
    auto sourceType = source.getType().cast<MemRefType>();
    unsigned sourceRank = sourceType.getRank();

    auto newExtractStridedMetadata =
        rewriter.create<memref::ExtractStridedMetadataOp>(origLoc, source);

    // Collect statically known information.
    auto [strides, offset] = getStridesAndOffset(sourceType);
    MemRefType reshapeType = reshape.getResultType();
    unsigned reshapeRank = reshapeType.getRank();

    OpFoldResult offsetOfr =
        ShapedType::isDynamic(offset)
            ? getAsOpFoldResult(newExtractStridedMetadata.getOffset())
            : rewriter.getIndexAttr(offset);

    // Get the special case of 0-D out of the way.
    if (sourceRank == 0) {
      SmallVector<OpFoldResult> ones(reshapeRank, rewriter.getIndexAttr(1));
      auto memrefDesc = rewriter.create<memref::ReinterpretCastOp>(
          origLoc, reshapeType, newExtractStridedMetadata.getBaseBuffer(),
          offsetOfr, /*sizes=*/ones, /*strides=*/ones);
      rewriter.replaceOp(reshape, memrefDesc.getResult());
      return success();
    }

    SmallVector<OpFoldResult> finalSizes;
    finalSizes.reserve(reshapeRank);
    SmallVector<OpFoldResult> finalStrides;
    finalStrides.reserve(reshapeRank);

    // Compute the reshaped strides and sizes from the base strides and sizes.
    SmallVector<OpFoldResult> origSizes =
        getAsOpFoldResult(newExtractStridedMetadata.getSizes());
    SmallVector<OpFoldResult> origStrides =
        getAsOpFoldResult(newExtractStridedMetadata.getStrides());
    unsigned idx = 0, endIdx = reshape.getReassociationIndices().size();
    for (; idx != endIdx; ++idx) {
      SmallVector<OpFoldResult> reshapedSizes =
          getReshapedSizes(reshape, rewriter, origSizes, /*groupId=*/idx);
      SmallVector<OpFoldResult> reshapedStrides = getReshapedStrides(
          reshape, rewriter, origSizes, origStrides, /*groupId=*/idx);

      unsigned groupSize = reshapedSizes.size();
      for (unsigned i = 0; i < groupSize; ++i) {
        finalSizes.push_back(reshapedSizes[i]);
        finalStrides.push_back(reshapedStrides[i]);
      }
    }
    assert(((isa<memref::ExpandShapeOp>(reshape) && idx == sourceRank) ||
            (isa<memref::CollapseShapeOp>(reshape) && idx == reshapeRank)) &&
           "We should have visited all the input dimensions");
    assert(finalSizes.size() == reshapeRank &&
           "We should have populated all the values");
    auto memrefDesc = rewriter.create<memref::ReinterpretCastOp>(
        origLoc, reshapeType, newExtractStridedMetadata.getBaseBuffer(),
        offsetOfr, finalSizes, finalStrides);
    rewriter.replaceOp(reshape, memrefDesc.getResult());
    return success();
  }
};

/// Replace `base, offset, sizes, strides =
///              extract_strided_metadata(allocLikeOp)`
///
/// With
///
/// ```
/// base = reinterpret_cast allocLikeOp(allocSizes) to a flat memref<eltTy>
/// offset = 0
/// sizes = allocSizes
/// strides#i = prod(allocSizes#j, for j in {i+1..rank-1})
/// ```
///
/// The transformation only applies if the allocLikeOp has been normalized.
/// In other words, the affine_map must be an identity.
template <typename AllocLikeOp>
struct ExtractStridedMetadataOpAllocFolder
    : public OpRewritePattern<memref::ExtractStridedMetadataOp> {
public:
  using OpRewritePattern<memref::ExtractStridedMetadataOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::ExtractStridedMetadataOp op,
                                PatternRewriter &rewriter) const override {
    auto allocLikeOp = op.getSource().getDefiningOp<AllocLikeOp>();
    if (!allocLikeOp)
      return failure();

    auto memRefType =
        allocLikeOp.getResult().getType().template cast<MemRefType>();
    if (!memRefType.getLayout().isIdentity())
      return rewriter.notifyMatchFailure(
          allocLikeOp, "alloc-like operations should have been normalized");

    Location loc = op.getLoc();
    int rank = memRefType.getRank();

    // Collect the sizes.
    ValueRange dynamic = allocLikeOp.getDynamicSizes();
    SmallVector<OpFoldResult> sizes;
    sizes.reserve(rank);
    unsigned dynamicPos = 0;
    for (int64_t size : memRefType.getShape()) {
      if (ShapedType::isDynamic(size))
        sizes.push_back(dynamic[dynamicPos++]);
      else
        sizes.push_back(rewriter.getIndexAttr(size));
    }

    // Strides (just creates identity strides).
    SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));
    AffineExpr expr = rewriter.getAffineConstantExpr(1);
    unsigned symbolNumber = 0;
    for (int i = rank - 2; i >= 0; --i) {
      expr = expr * rewriter.getAffineSymbolExpr(symbolNumber++);
      assert(i + 1 + symbolNumber == sizes.size() &&
             "The ArrayRef should encompass the last #symbolNumber sizes");
      ArrayRef<OpFoldResult> sizesInvolvedInStride(&sizes[i + 1], symbolNumber);
      strides[i] = makeComposedFoldedAffineApply(rewriter, loc, expr,
                                                 sizesInvolvedInStride);
    }

    // Put all the values together to replace the results.
    SmallVector<Value> results;
    results.reserve(rank * 2 + 2);

    auto baseBufferType = op.getBaseBuffer().getType().cast<MemRefType>();
    int64_t offset = 0;
    if (allocLikeOp.getType() == baseBufferType)
      results.push_back(allocLikeOp);
    else
      results.push_back(rewriter.create<memref::ReinterpretCastOp>(
          loc, baseBufferType, allocLikeOp, offset,
          /*sizes=*/ArrayRef<int64_t>(),
          /*strides=*/ArrayRef<int64_t>()));

    // Offset.
    results.push_back(rewriter.create<arith::ConstantIndexOp>(loc, offset));

    for (OpFoldResult size : sizes)
      results.push_back(getValueOrCreateConstantIndexOp(rewriter, loc, size));

    for (OpFoldResult stride : strides)
      results.push_back(getValueOrCreateConstantIndexOp(rewriter, loc, stride));

    rewriter.replaceOp(op, results);
    return success();
  }
};

/// Rewrite memref.extract_aligned_pointer_as_index of a ViewLikeOp to the
/// source of the ViewLikeOp.
class RewriteExtractAlignedPointerAsIndexOfViewLikeOp
    : public OpRewritePattern<memref::ExtractAlignedPointerAsIndexOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult
  matchAndRewrite(memref::ExtractAlignedPointerAsIndexOp extractOp,
                  PatternRewriter &rewriter) const override {
    auto viewLikeOp =
        extractOp.getSource().getDefiningOp<ViewLikeOpInterface>();
    if (!viewLikeOp)
      return rewriter.notifyMatchFailure(extractOp, "not a ViewLike source");
    rewriter.updateRootInPlace(extractOp, [&]() {
      extractOp.getSourceMutable().assign(viewLikeOp.getViewSource());
    });
    return success();
  }
};

/// Replace `base, offset, sizes, strides =
///              extract_strided_metadata(
///                 reinterpret_cast(src, srcOffset, srcSizes, srcStrides))`
/// With
/// ```
/// base, ... = extract_strided_metadata(src)
/// offset = srcOffset
/// sizes = srcSizes
/// strides = srcStrides
/// ```
///
/// In other words, consume the `reinterpret_cast` and apply its effects
/// on the offset, sizes, and strides.
class ExtractStridedMetadataOpReinterpretCastFolder
    : public OpRewritePattern<memref::ExtractStridedMetadataOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult
  matchAndRewrite(memref::ExtractStridedMetadataOp extractStridedMetadataOp,
                  PatternRewriter &rewriter) const override {
    auto reinterpretCastOp = extractStridedMetadataOp.getSource()
                                 .getDefiningOp<memref::ReinterpretCastOp>();
    if (!reinterpretCastOp)
      return failure();

    Location loc = extractStridedMetadataOp.getLoc();
    // Check if the source is suitable for extract_strided_metadata.
    SmallVector<Type> inferredReturnTypes;
    if (failed(extractStridedMetadataOp.inferReturnTypes(
            rewriter.getContext(), loc, {reinterpretCastOp.getSource()},
            /*attributes=*/{}, /*regions=*/{}, inferredReturnTypes)))
      return rewriter.notifyMatchFailure(
          reinterpretCastOp, "reinterpret_cast source's type is incompatible");

    auto memrefType =
        reinterpretCastOp.getResult().getType().cast<MemRefType>();
    unsigned rank = memrefType.getRank();
    SmallVector<OpFoldResult> results;
    results.resize_for_overwrite(rank * 2 + 2);

    auto newExtractStridedMetadata =
        rewriter.create<memref::ExtractStridedMetadataOp>(
            loc, reinterpretCastOp.getSource());

    // Register the base_buffer.
    results[0] = newExtractStridedMetadata.getBaseBuffer();

    // Register the new offset.
    results[1] = getValueOrCreateConstantIndexOp(
        rewriter, loc, reinterpretCastOp.getMixedOffsets()[0]);

    const unsigned sizeStartIdx = 2;
    const unsigned strideStartIdx = sizeStartIdx + rank;

    SmallVector<OpFoldResult> sizes = reinterpretCastOp.getMixedSizes();
    SmallVector<OpFoldResult> strides = reinterpretCastOp.getMixedStrides();
    for (unsigned i = 0; i < rank; ++i) {
      results[sizeStartIdx + i] = sizes[i];
      results[strideStartIdx + i] = strides[i];
    }
    rewriter.replaceOp(extractStridedMetadataOp,
                       getValueOrCreateConstantIndexOp(rewriter, loc, results));
    return success();
  }
};

/// Replace `base, offset =
///            extract_strided_metadata(extract_strided_metadata(src)#0)`
/// With
/// ```
/// base, ... = extract_strided_metadata(src)
/// offset = 0
/// ```
class ExtractStridedMetadataOpExtractStridedMetadataFolder
    : public OpRewritePattern<memref::ExtractStridedMetadataOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult
  matchAndRewrite(memref::ExtractStridedMetadataOp extractStridedMetadataOp,
                  PatternRewriter &rewriter) const override {
    auto sourceExtractStridedMetadataOp =
        extractStridedMetadataOp.getSource()
            .getDefiningOp<memref::ExtractStridedMetadataOp>();
    if (!sourceExtractStridedMetadataOp)
      return failure();
    Location loc = extractStridedMetadataOp.getLoc();
    rewriter.replaceOp(extractStridedMetadataOp,
                       {sourceExtractStridedMetadataOp.getBaseBuffer(),
                        getValueOrCreateConstantIndexOp(
                            rewriter, loc, rewriter.getIndexAttr(0))});
    return success();
  }
};
} // namespace

void memref::populateExpandStridedMetadataPatterns(
    RewritePatternSet &patterns) {
  patterns.add<SubviewFolder,
               ReshapeFolder<memref::ExpandShapeOp, getExpandedSizes,
                             getExpandedStrides>,
               ReshapeFolder<memref::CollapseShapeOp, getCollapsedSize,
                             getCollapsedStride>,
               ExtractStridedMetadataOpAllocFolder<memref::AllocOp>,
               ExtractStridedMetadataOpAllocFolder<memref::AllocaOp>,
               RewriteExtractAlignedPointerAsIndexOfViewLikeOp,
               ExtractStridedMetadataOpReinterpretCastFolder,
               ExtractStridedMetadataOpExtractStridedMetadataFolder>(
      patterns.getContext());
}

//===----------------------------------------------------------------------===//
// Pass registration
//===----------------------------------------------------------------------===//

namespace {

struct ExpandStridedMetadataPass final
    : public memref::impl::ExpandStridedMetadataBase<
          ExpandStridedMetadataPass> {
  void runOnOperation() override;
};

} // namespace

void ExpandStridedMetadataPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  memref::populateExpandStridedMetadataPatterns(patterns);
  (void)applyPatternsAndFoldGreedily(getOperation()->getRegions(),
                                     std::move(patterns));
}

std::unique_ptr<Pass> memref::createExpandStridedMetadataPass() {
  return std::make_unique<ExpandStridedMetadataPass>();
}
