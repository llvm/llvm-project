//===- SimplifyExtractStridedMetadata.cpp - Simplify this operation -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// This pass simplifies extract_strided_metadata(other_op(memref) to
/// extract_strided_metadata(memref) when it is possible to express the effect
// of other_op using affine apply on the results of
// extract_strided_metadata(memref).
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"

namespace mlir {
namespace memref {
#define GEN_PASS_DEF_SIMPLIFYEXTRACTSTRIDEDMETADATA
#include "mlir/Dialect/MemRef/Transforms/Passes.h.inc"
} // namespace memref
} // namespace mlir
using namespace mlir;

namespace {
/// Replace `baseBuffer, offset, sizes, strides =
///              extract_strided_metadata(subview(memref, subOffset,
///                                               subSizes, subStrides))`
/// With
///
/// \verbatim
/// baseBuffer, baseOffset, baseSizes, baseStrides =
///     extract_strided_metadata(memref)
/// strides#i = baseStrides#i * subSizes#i
/// offset = baseOffset + sum(subOffset#i * strides#i)
/// sizes = subSizes
/// \endverbatim
///
/// In other words, get rid of the subview in that expression and canonicalize
/// on its effects on the offset, the sizes, and the strides using affine.apply.
struct ExtractStridedMetadataOpSubviewFolder
    : public OpRewritePattern<memref::ExtractStridedMetadataOp> {
public:
  using OpRewritePattern<memref::ExtractStridedMetadataOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::ExtractStridedMetadataOp op,
                                PatternRewriter &rewriter) const override {
    auto subview = op.getSource().getDefiningOp<memref::SubViewOp>();
    if (!subview)
      return failure();

    // Build a plain extract_strided_metadata(memref) from
    // extract_strided_metadata(subview(memref)).
    Location origLoc = op.getLoc();
    IndexType indexType = rewriter.getIndexType();
    Value source = subview.getSource();
    auto sourceType = source.getType().cast<MemRefType>();
    unsigned sourceRank = sourceType.getRank();
    SmallVector<Type> sizeStrideTypes(sourceRank, indexType);

    auto newExtractStridedMetadata =
        rewriter.create<memref::ExtractStridedMetadataOp>(
            origLoc, op.getBaseBuffer().getType(), indexType, sizeStrideTypes,
            sizeStrideTypes, source);

    SmallVector<int64_t> sourceStrides;
    int64_t sourceOffset;

    bool hasKnownStridesAndOffset =
        succeeded(getStridesAndOffset(sourceType, sourceStrides, sourceOffset));
    (void)hasKnownStridesAndOffset;
    assert(hasKnownStridesAndOffset &&
           "getStridesAndOffset must work on valid subviews");

    // Compute the new strides and offset from the base strides and offset:
    // newStride#i = baseStride#i * subStride#i
    // offset = baseOffset + sum(subOffsets#i * newStrides#i)
    SmallVector<OpFoldResult> strides;
    SmallVector<OpFoldResult> subStrides = subview.getMixedStrides();
    auto origStrides = newExtractStridedMetadata.getStrides();

    // Hold the affine symbols and values for the computation of the offset.
    SmallVector<OpFoldResult> values(3 * sourceRank + 1);
    SmallVector<AffineExpr> symbols(3 * sourceRank + 1);

    detail::bindSymbolsList(rewriter.getContext(), symbols);
    AffineExpr expr = symbols.front();
    values[0] = ShapedType::isDynamicStrideOrOffset(sourceOffset)
                    ? getAsOpFoldResult(newExtractStridedMetadata.getOffset())
                    : rewriter.getIndexAttr(sourceOffset);
    SmallVector<OpFoldResult> subOffsets = subview.getMixedOffsets();

    AffineExpr s0 = rewriter.getAffineSymbolExpr(0);
    AffineExpr s1 = rewriter.getAffineSymbolExpr(1);
    for (unsigned i = 0; i < sourceRank; ++i) {
      // Compute the stride.
      OpFoldResult origStride =
          ShapedType::isDynamicStrideOrOffset(sourceStrides[i])
              ? origStrides[i]
              : OpFoldResult(rewriter.getIndexAttr(sourceStrides[i]));
      strides.push_back(makeComposedFoldedAffineApply(
          rewriter, origLoc, s0 * s1, {subStrides[i], origStride}));

      // Build up the computation of the offset.
      unsigned baseIdxForDim = 1 + 3 * i;
      unsigned subOffsetForDim = baseIdxForDim;
      unsigned subStrideForDim = baseIdxForDim + 1;
      unsigned origStrideForDim = baseIdxForDim + 2;
      expr = expr + symbols[subOffsetForDim] * symbols[subStrideForDim] *
                        symbols[origStrideForDim];
      values[subOffsetForDim] = subOffsets[i];
      values[subStrideForDim] = subStrides[i];
      values[origStrideForDim] = origStride;
    }

    // Compute the offset.
    OpFoldResult finalOffset =
        makeComposedFoldedAffineApply(rewriter, origLoc, expr, values);

    SmallVector<Value> results;
    // The final result is  <baseBuffer, offset, sizes, strides>.
    // Thus we need 1 + 1 + subview.getRank() + subview.getRank(), to hold all
    // the values.
    auto subType = subview.getType().cast<MemRefType>();
    unsigned subRank = subType.getRank();
    // Properly size the array so that we can do random insertions
    // at the right indices.
    // We do that to populate the non-dropped sizes and strides in one go.
    results.resize_for_overwrite(subRank * 2 + 2);

    results[0] = newExtractStridedMetadata.getBaseBuffer();
    results[1] =
        getValueOrCreateConstantIndexOp(rewriter, origLoc, finalOffset);

    // The sizes of the final type are defined directly by the input sizes of
    // the subview.
    // Moreover subviews can drop some dimensions, some strides and sizes may
    // not end up in the final <base, offset, sizes, strides> value that we are
    // replacing.
    // Do the filtering here.
    SmallVector<OpFoldResult> subSizes = subview.getMixedSizes();
    const unsigned sizeStartIdx = 2;
    const unsigned strideStartIdx = sizeStartIdx + subRank;
    unsigned insertedDims = 0;
    llvm::SmallBitVector droppedDims = subview.getDroppedDims();
    for (unsigned i = 0; i < sourceRank; ++i) {
      if (droppedDims.test(i))
        continue;

      results[sizeStartIdx + insertedDims] =
          getValueOrCreateConstantIndexOp(rewriter, origLoc, subSizes[i]);
      results[strideStartIdx + insertedDims] =
          getValueOrCreateConstantIndexOp(rewriter, origLoc, strides[i]);
      ++insertedDims;
    }
    assert(insertedDims == subRank &&
           "Should have populated all the values at this point");

    rewriter.replaceOp(op, results);
    return success();
  }
};

/// Compute the expanded sizes of the given \p expandShape for the
/// \p groupId-th reassociation group.
/// \p origSizes hold the sizes of the source shape as values.
/// This is used to compute the new sizes in cases of dynamic shapes.
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
  SmallVector<int64_t> strides;
  int64_t offset;
  bool hasKnownStridesAndOffset =
      succeeded(getStridesAndOffset(sourceType, strides, offset));
  (void)hasKnownStridesAndOffset;
  assert(hasKnownStridesAndOffset &&
         "getStridesAndOffset must work on valid expand_shape");

  OpFoldResult origStride =
      ShapedType::isDynamicStrideOrOffset(strides[groupId])
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

/// Replace `baseBuffer, offset, sizes, strides =
///              extract_strided_metadata(expand_shape(memref))`
/// With
///
/// \verbatim
/// baseBuffer, offset, baseSizes, baseStrides =
///     extract_strided_metadata(memref)
/// sizes#reassIdx =
///     baseSizes#reassDim / product(expandShapeSizes#j,
///                                  for j in group excluding reassIdx)
/// strides#reassIdx =
///     baseStrides#reassDim * product(expandShapeSizes#j, for j in
///                                    reassIdx+1..reassIdx+group.size-1)
/// \endverbatim
///
/// Where reassIdx is a reassociation index for the group at reassDim
/// and expandShapeSizes#j is either:
/// - The constant size at dimension j, derived directly from the result type of
///   the expand_shape op, or
/// - An affine expression: baseSizes#reassDim / product of all constant sizes
///   in expandShapeSizes. (Remember expandShapeSizes has at most one dynamic
///   element.)
///
/// Notice that `baseBuffer` and `offset` are unchanged.
///
/// In other words, get rid of the expand_shape in that expression and
/// materialize its effects on the sizes and the strides using affine apply.
struct ExtractStridedMetadataOpExpandShapeFolder
    : public OpRewritePattern<memref::ExtractStridedMetadataOp> {
public:
  using OpRewritePattern<memref::ExtractStridedMetadataOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::ExtractStridedMetadataOp op,
                                PatternRewriter &rewriter) const override {
    auto expandShape = op.getSource().getDefiningOp<memref::ExpandShapeOp>();
    if (!expandShape)
      return failure();

    // Build a plain extract_strided_metadata(memref) from
    // extract_strided_metadata(expand_shape(memref)).
    Location origLoc = op.getLoc();
    IndexType indexType = rewriter.getIndexType();
    Value source = expandShape.getSrc();
    auto sourceType = source.getType().cast<MemRefType>();
    unsigned sourceRank = sourceType.getRank();
    SmallVector<Type> sizeStrideTypes(sourceRank, indexType);

    auto newExtractStridedMetadata =
        rewriter.create<memref::ExtractStridedMetadataOp>(
            origLoc, op.getBaseBuffer().getType(), indexType, sizeStrideTypes,
            sizeStrideTypes, source);

    // Collect statically known information.
    SmallVector<int64_t> strides;
    int64_t offset;
    bool hasKnownStridesAndOffset =
        succeeded(getStridesAndOffset(sourceType, strides, offset));
    (void)hasKnownStridesAndOffset;
    assert(hasKnownStridesAndOffset &&
           "getStridesAndOffset must work on valid expand_shape");
    MemRefType expandShapeType = expandShape.getResultType();
    unsigned expandShapeRank = expandShapeType.getRank();

    // The result value will start with the base_buffer and offset.
    unsigned baseIdxInResult = 2;
    SmallVector<OpFoldResult> results(baseIdxInResult + expandShapeRank * 2);
    results[0] = newExtractStridedMetadata.getBaseBuffer();
    results[1] = ShapedType::isDynamicStrideOrOffset(offset)
                     ? getAsOpFoldResult(newExtractStridedMetadata.getOffset())
                     : rewriter.getIndexAttr(offset);

    // Get the special case of 0-D out of the way.
    if (sourceRank == 0) {
      Value constantOne = getValueOrCreateConstantIndexOp(
          rewriter, origLoc, rewriter.getIndexAttr(1));
      SmallVector<Value> resultValues(baseIdxInResult + expandShapeRank * 2,
                                      constantOne);
      for (unsigned i = 0; i < baseIdxInResult; ++i)
        resultValues[i] =
            getValueOrCreateConstantIndexOp(rewriter, origLoc, results[i]);
      rewriter.replaceOp(op, resultValues);
      return success();
    }

    // Compute the expanded strides and sizes from the base strides and sizes.
    SmallVector<OpFoldResult> origSizes =
        getAsOpFoldResult(newExtractStridedMetadata.getSizes());
    SmallVector<OpFoldResult> origStrides =
        getAsOpFoldResult(newExtractStridedMetadata.getStrides());
    unsigned idx = 0, endIdx = expandShape.getReassociationIndices().size();
    for (; idx != endIdx; ++idx) {
      SmallVector<OpFoldResult> expandedSizes =
          getExpandedSizes(expandShape, rewriter, origSizes, /*groupId=*/idx);
      SmallVector<OpFoldResult> expandedStrides = getExpandedStrides(
          expandShape, rewriter, origSizes, origStrides, /*groupId=*/idx);

      unsigned groupSize = expandShape.getReassociationIndices()[idx].size();
      const unsigned sizeStartIdx = baseIdxInResult;
      const unsigned strideStartIdx = sizeStartIdx + expandShapeRank;
      for (unsigned i = 0; i < groupSize; ++i) {
        results[sizeStartIdx + i] = expandedSizes[i];
        results[strideStartIdx + i] = expandedStrides[i];
      }
      baseIdxInResult += groupSize;
    }
    assert(idx == sourceRank &&
           "We should have visited all the input dimensions");
    assert(baseIdxInResult == expandShapeRank + 2 &&
           "We should have populated all the values");
    rewriter.replaceOp(
        op, getValueOrCreateConstantIndexOp(rewriter, origLoc, results));
    return success();
  }
};

/// Helper function to perform the replacement of all constant uses of `values`
/// by a materialized constant extracted from `maybeConstants`.
/// `values` and `maybeConstants` are expected to have the same size.
template <typename Container>
bool replaceConstantUsesOf(PatternRewriter &rewriter, Location loc,
                           Container values, ArrayRef<int64_t> maybeConstants,
                           llvm::function_ref<bool(int64_t)> isDynamic) {
  assert(values.size() == maybeConstants.size() &&
         " expected values and maybeConstants of the same size");
  bool atLeastOneReplacement = false;
  for (auto [maybeConstant, result] : llvm::zip(maybeConstants, values)) {
    // Don't materialize a constant if there are no uses: this would indice
    // infinite loops in the driver.
    if (isDynamic(maybeConstant) || result.use_empty())
      continue;
    Value constantVal =
        rewriter.create<arith::ConstantIndexOp>(loc, maybeConstant);
    for (Operation *op : llvm::make_early_inc_range(result.getUsers())) {
      rewriter.startRootUpdate(op);
      // updateRootInplace: lambda cannot capture structured bindings in C++17
      // yet.
      op->replaceUsesOfWith(result, constantVal);
      rewriter.finalizeRootUpdate(op);
      atLeastOneReplacement = true;
    }
  }
  return atLeastOneReplacement;
}

// Forward propagate all constants information from an ExtractStridedMetadataOp.
struct ForwardStaticMetadata
    : public OpRewritePattern<memref::ExtractStridedMetadataOp> {
  using OpRewritePattern<memref::ExtractStridedMetadataOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::ExtractStridedMetadataOp metadataOp,
                                PatternRewriter &rewriter) const override {
    auto memrefType = metadataOp.getSource().getType().cast<MemRefType>();
    SmallVector<int64_t> strides;
    int64_t offset;
    LogicalResult res = getStridesAndOffset(memrefType, strides, offset);
    (void)res;
    assert(succeeded(res) && "must be a strided memref type");

    bool atLeastOneReplacement = replaceConstantUsesOf(
        rewriter, metadataOp.getLoc(),
        ArrayRef<TypedValue<IndexType>>(metadataOp.getOffset()),
        ArrayRef<int64_t>(offset), ShapedType::isDynamicStrideOrOffset);
    atLeastOneReplacement |= replaceConstantUsesOf(
        rewriter, metadataOp.getLoc(), metadataOp.getSizes(),
        memrefType.getShape(), ShapedType::isDynamic);
    atLeastOneReplacement |= replaceConstantUsesOf(
        rewriter, metadataOp.getLoc(), metadataOp.getStrides(), strides,
        ShapedType::isDynamicStrideOrOffset);

    return success(atLeastOneReplacement);
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
      extractOp.sourceMutable().assign(viewLikeOp.getViewSource());
    });
    return success();
  }
};
} // namespace

void memref::populateSimplifyExtractStridedMetadataOpPatterns(
    RewritePatternSet &patterns) {
  patterns.add<ExtractStridedMetadataOpSubviewFolder,
               ExtractStridedMetadataOpExpandShapeFolder, ForwardStaticMetadata,
               ExtractStridedMetadataOpAllocFolder<memref::AllocOp>,
               ExtractStridedMetadataOpAllocFolder<memref::AllocaOp>,
               RewriteExtractAlignedPointerAsIndexOfViewLikeOp>(
      patterns.getContext());
}

//===----------------------------------------------------------------------===//
// Pass registration
//===----------------------------------------------------------------------===//

namespace {

struct SimplifyExtractStridedMetadataPass final
    : public memref::impl::SimplifyExtractStridedMetadataBase<
          SimplifyExtractStridedMetadataPass> {
  void runOnOperation() override;
};

} // namespace

void SimplifyExtractStridedMetadataPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  memref::populateSimplifyExtractStridedMetadataOpPatterns(patterns);
  (void)applyPatternsAndFoldGreedily(getOperation()->getRegions(),
                                     std::move(patterns));
}

std::unique_ptr<Pass> memref::createSimplifyExtractStridedMetadataPass() {
  return std::make_unique<SimplifyExtractStridedMetadataPass>();
}
