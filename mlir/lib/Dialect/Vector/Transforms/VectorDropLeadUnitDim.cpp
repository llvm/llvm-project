//===- VectorDropLeadUnitDim.cpp - Conversion within the Vector dialect ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <numeric>
#include <utility>

#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/Repeated.h"
#include "llvm/ADT/STLExtras.h"

#define DEBUG_TYPE "vector-drop-unit-dim"

using namespace mlir;
using namespace mlir::vector;

// Trims leading unit dimensions from `oldType` and returns the result type.
static VectorType trimLeadingUnitDims(VectorType oldType,
                                      bool zeroDimsAllowed) {
  ArrayRef<int64_t> oldShape = oldType.getShape();
  ArrayRef<int64_t> newShape = oldShape;

  ArrayRef<bool> oldScalableDims = oldType.getScalableDims();
  ArrayRef<bool> newScalableDims = oldScalableDims;

  while (!newShape.empty() && newShape.front() == 1 &&
         !newScalableDims.front()) {
    newShape = newShape.drop_front(1);
    newScalableDims = newScalableDims.drop_front(1);
  }

  // Some vector ops forbid 0-D vectors.
  if (!zeroDimsAllowed && newShape.empty()) {
    newShape = oldShape.take_back();
    newScalableDims = oldType.getScalableDims().take_back();
  }
  return VectorType::get(newShape, oldType.getElementType(), newScalableDims);
}

static bool isNonScalableUnitDim(VectorType type, int64_t dim) {
  assert(dim >= 0 && dim < type.getRank() &&
         "expected a valid vector dimension");
  return type.getShape()[dim] == 1 && !type.getScalableDims()[dim];
}

/// Returns true if the first `k` dimensions of `type` are non-scalable unit
/// dimensions.
static bool areLeadingDimsUnit(VectorType type, int64_t k) {
  assert(k >= 0 && k <= type.getRank() &&
         "expected a valid leading dimension count");
  return llvm::all_of(llvm::seq<int64_t>(0, k), [&](int64_t dim) {
    return isNonScalableUnitDim(type, dim);
  });
}

static bool areLeadingDimsUnitAfterPermutation(VectorType type,
                                               ArrayRef<int64_t> permutation,
                                               int64_t k) {
  assert(k >= 0 && k <= static_cast<int64_t>(permutation.size()) &&
         "expected a valid leading dimension count");
  return llvm::all_of(permutation.take_front(k), [&](int64_t dim) {
    return isNonScalableUnitDim(type, dim);
  });
}

/// Shape-casts `operand` to the vector type obtained by dropping dimension
/// `dim`, which must be non-scalable and unit-sized.
static Value dropUnitDim(OpBuilder &b, Location loc, Value operand,
                         int64_t dimToDrop, bool zeroDimsAllowed) {
  auto oldType = cast<VectorType>(operand.getType());
  assert(isNonScalableUnitDim(oldType, dimToDrop) &&
         "expected a non-scalable unit dim to drop");
  int64_t rank = oldType.getRank();
  assert((zeroDimsAllowed || rank > 1) &&
         "target op does not allow 0-D vectors");

  SmallVector<int64_t> newShape;
  SmallVector<bool> newScalableDims;
  newShape.reserve(rank - 1);
  newScalableDims.reserve(rank - 1);
  for (auto [i, size, scalable] :
       llvm::enumerate(oldType.getShape(), oldType.getScalableDims())) {
    if (static_cast<int64_t>(i) == dimToDrop)
      continue;
    newShape.push_back(size);
    newScalableDims.push_back(scalable);
  }

  return b.createOrFold<vector::ShapeCastOp>(
      loc, VectorType::get(newShape, oldType.getElementType(), newScalableDims),
      operand);
}

/// Shape-casts `operand` to the vector type obtained by dropping the first
/// `k` non-scalable unit dimensions.
static Value dropLeadingUnitDims(OpBuilder &b, Location loc, Value operand,
                                 int64_t k, bool zeroDimsAllowed) {
  auto oldType = cast<VectorType>(operand.getType());
  assert(areLeadingDimsUnit(oldType, k) &&
         "expected non-scalable leading unit dims to drop");
  assert((zeroDimsAllowed || k < oldType.getRank()) &&
         "target op does not allow 0-D vectors");
  VectorType newType = VectorType::get(oldType.getShape().drop_front(k),
                                       oldType.getElementType(),
                                       oldType.getScalableDims().drop_front(k));
  return b.createOrFold<vector::ShapeCastOp>(loc, newType, operand);
}

/// Returns the vector type obtained by applying `permutation` to `type`.
static VectorType permuteVectorType(VectorType type,
                                    ArrayRef<int64_t> permutation) {
  assert(static_cast<int64_t>(permutation.size()) == type.getRank() &&
         "expected a permutation matching the operand rank");
  SmallVector<int64_t> permutedShape =
      applyPermutation(type.getShape(), permutation);
  SmallVector<bool> permutedScalableDims =
      applyPermutation(type.getScalableDims(), permutation);
  return VectorType::get(permutedShape, type.getElementType(),
                         permutedScalableDims);
}

/// Like `dropLeadingUnitDims` except that if all dimensions would be dropped,
/// the single element inside that vector is extracted and returned.
static Value dropLeadingUnitDims0DIsScalar(OpBuilder &b, Location loc,
                                           Value operand, int64_t k) {
  auto oldType = cast<VectorType>(operand.getType());
  assert(areLeadingDimsUnit(oldType, k) &&
         "expected non-scalable leading unit dims to drop");

  if (k == oldType.getRank()) {
    SmallVector<int64_t> zeros(k, static_cast<int64_t>(0));
    return vector::ExtractOp::create(b, loc, operand, zeros);
  }

  return dropLeadingUnitDims(b, loc, operand, k,
                             /*zeroDimsAllowed=*/true);
}

namespace {

// Casts away leading one dimensions in vector.extract_strided_slice's vector
// input by inserting vector.shape_cast.
struct CastAwayExtractStridedSliceLeadingOneDim
    : public OpRewritePattern<vector::ExtractStridedSliceOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(vector::ExtractStridedSliceOp extractOp,
                                PatternRewriter &rewriter) const override {
    // vector.extract_strided_slice requires the input and output vector to have
    // the same rank. Here we drop leading one dimensions from the input vector
    // type to make sure we don't cause mismatch.
    VectorType oldSrcType = extractOp.getSourceVectorType();
    VectorType newSrcType =
        trimLeadingUnitDims(oldSrcType, /*zeroDimsAllowed=*/false);

    if (newSrcType.getRank() == oldSrcType.getRank())
      return failure();

    int64_t dropCount = oldSrcType.getRank() - newSrcType.getRank();

    VectorType oldDstType = extractOp.getType();
    VectorType newDstType =
        VectorType::get(oldDstType.getShape().drop_front(dropCount),
                        oldDstType.getElementType(),
                        oldDstType.getScalableDims().drop_front(dropCount));

    Location loc = extractOp.getLoc();

    Value newSrcVector = rewriter.createOrFold<vector::ShapeCastOp>(
        loc, newSrcType, extractOp.getSource());

    // The offsets/sizes/strides attribute can have a less number of elements
    // than the input vector's rank: it is meant for the leading dimensions.
    auto newOffsets = rewriter.getArrayAttr(
        extractOp.getOffsets().getValue().drop_front(dropCount));
    auto newSizes = rewriter.getArrayAttr(
        extractOp.getSizes().getValue().drop_front(dropCount));
    auto newStrides = rewriter.getArrayAttr(
        extractOp.getStrides().getValue().drop_front(dropCount));

    auto newExtractOp = vector::ExtractStridedSliceOp::create(
        rewriter, loc, newDstType, newSrcVector, newOffsets, newSizes,
        newStrides);

    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(extractOp, oldDstType,
                                                     newExtractOp);

    return success();
  }
};

// Casts away leading one dimensions in vector.insert_strided_slice's vector
// inputs by inserting vector.shape_cast.
struct CastAwayInsertStridedSliceLeadingOneDim
    : public OpRewritePattern<vector::InsertStridedSliceOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(vector::InsertStridedSliceOp insertOp,
                                PatternRewriter &rewriter) const override {
    VectorType oldSrcType = insertOp.getSourceVectorType();
    VectorType newSrcType =
        trimLeadingUnitDims(oldSrcType, /*zeroDimsAllowed=*/false);
    VectorType oldDstType = insertOp.getDestVectorType();
    VectorType newDstType =
        trimLeadingUnitDims(oldDstType, /*zeroDimsAllowed=*/false);

    int64_t srcDropCount = oldSrcType.getRank() - newSrcType.getRank();
    int64_t dstDropCount = oldDstType.getRank() - newDstType.getRank();
    if (srcDropCount == 0 && dstDropCount == 0)
      return failure();

    // Trim leading one dimensions from both operands.
    Location loc = insertOp.getLoc();

    Value newSrcVector = rewriter.createOrFold<vector::ShapeCastOp>(
        loc, newSrcType, insertOp.getValueToStore());
    Value newDstVector = rewriter.createOrFold<vector::ShapeCastOp>(
        loc, newDstType, insertOp.getDest());

    auto newOffsets = rewriter.getArrayAttr(
        insertOp.getOffsets().getValue().take_back(newDstType.getRank()));
    auto newStrides = rewriter.getArrayAttr(
        insertOp.getStrides().getValue().take_back(newSrcType.getRank()));

    auto newInsertOp = vector::InsertStridedSliceOp::create(
        rewriter, loc, newDstType, newSrcVector, newDstVector, newOffsets,
        newStrides);

    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(insertOp, oldDstType,
                                                     newInsertOp);

    return success();
  }
};

// Casts away leading one dimensions in vector.insert's vector inputs by
// inserting vector.shape_cast.
struct CastAwayInsertLeadingOneDim : public OpRewritePattern<vector::InsertOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(vector::InsertOp insertOp,
                                PatternRewriter &rewriter) const override {
    Type oldSrcType = insertOp.getValueToStoreType();
    Type newSrcType = oldSrcType;
    int64_t oldSrcRank = 0, newSrcRank = 0;
    if (auto type = dyn_cast<VectorType>(oldSrcType)) {
      newSrcType = trimLeadingUnitDims(type, /*zeroDimsAllowed=*/false);
      oldSrcRank = type.getRank();
      newSrcRank = cast<VectorType>(newSrcType).getRank();
    }

    VectorType oldDstType = insertOp.getDestVectorType();
    VectorType newDstType =
        trimLeadingUnitDims(oldDstType, /*zeroDimsAllowed=*/oldSrcRank == 0);

    int64_t srcDropCount = oldSrcRank - newSrcRank;
    int64_t dstDropCount = oldDstType.getRank() - newDstType.getRank();
    if (srcDropCount == 0 && dstDropCount == 0)
      return failure();

    // Trim leading one dimensions from both operands.
    Location loc = insertOp.getLoc();

    Value newSrcVector = insertOp.getValueToStore();
    if (oldSrcRank != 0)
      newSrcVector = rewriter.createOrFold<vector::ShapeCastOp>(
          loc, cast<VectorType>(newSrcType), insertOp.getValueToStore());
    Value newDstVector = rewriter.createOrFold<vector::ShapeCastOp>(
        loc, newDstType, insertOp.getDest());

    // New position rank needs to be computed in two steps: (1) if destination
    // type has leading unit dims, we also trim the position array accordingly,
    // then (2) if source type also has leading unit dims, we need to append
    // zeroes to the position array accordingly.
    unsigned oldPosRank = insertOp.getNumIndices();
    unsigned newPosRank = std::max<int64_t>(0, oldPosRank - dstDropCount);
    SmallVector<OpFoldResult> oldPosition = insertOp.getMixedPosition();
    SmallVector<OpFoldResult> newPosition =
        llvm::to_vector(ArrayRef(oldPosition).take_back(newPosRank));
    newPosition.resize(newDstType.getRank() - newSrcRank,
                       rewriter.getI64IntegerAttr(0));

    auto newInsertOp = vector::InsertOp::create(rewriter, loc, newSrcVector,
                                                newDstVector, newPosition);

    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(insertOp, oldDstType,
                                                     newInsertOp);

    return success();
  }
};

static Value dropUnitDimsFromMask(OpBuilder &b, Location loc, Value mask,
                                  VectorType newType, AffineMap newMap) {
  // Infer the type of the new mask from the new map.
  VectorType newMaskType = inferTransferOpMaskType(newType, newMap);
  return b.createOrFold<vector::ShapeCastOp>(loc, newMaskType, mask);
}

// Turns vector.transfer_read on vector with leading 1 dimensions into
// vector.shape_cast followed by vector.transfer_read on vector without leading
// 1 dimensions.
struct CastAwayTransferReadLeadingOneDim
    : public OpRewritePattern<vector::TransferReadOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(vector::TransferReadOp read,
                                PatternRewriter &rewriter) const override {
    // TODO(#78787): Not supported masked op yet.
    if (cast<MaskableOpInterface>(read.getOperation()).isMasked())
      return failure();
    // Nothing to trim when the transfer itself has rank zero.
    if (read.getTransferRank() == 0)
      return failure();

    auto shapedType = cast<ShapedType>(read.getBase().getType());
    if (shapedType.getElementType() != read.getVectorType().getElementType())
      return failure();

    VectorType oldType = read.getVectorType();
    VectorType newType = trimLeadingUnitDims(oldType, /*zeroDimsAllowed=*/true);

    if (newType == oldType)
      return failure();

    AffineMap oldMap = read.getPermutationMap();
    ArrayRef<AffineExpr> newResults =
        oldMap.getResults().take_back(newType.getRank());
    AffineMap newMap =
        AffineMap::get(oldMap.getNumDims(), oldMap.getNumSymbols(), newResults,
                       rewriter.getContext());

    ArrayAttr inBoundsAttr;
    if (read.getInBounds())
      inBoundsAttr = rewriter.getArrayAttr(
          read.getInBoundsAttr().getValue().take_back(newType.getRank()));

    Value mask = Value();
    if (read.getMask())
      mask = dropUnitDimsFromMask(rewriter, read.getLoc(), read.getMask(),
                                  newType, newMap);

    auto newRead = vector::TransferReadOp::create(
        rewriter, read.getLoc(), newType, read.getBase(), read.getIndices(),
        AffineMapAttr::get(newMap), read.getPadding(), mask, inBoundsAttr);
    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(read, oldType, newRead);

    return success();
  }
};

// Turns vector.transfer_write on vector with leading 1 dimensions into
// vector.shape_cast followed by vector.transfer_write on vector without leading
// 1 dimensions.
struct CastAwayTransferWriteLeadingOneDim
    : public OpRewritePattern<vector::TransferWriteOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(vector::TransferWriteOp write,
                                PatternRewriter &rewriter) const override {
    // TODO(#78787): Not supported masked op yet.
    if (cast<MaskableOpInterface>(write.getOperation()).isMasked())
      return failure();
    // Nothing to trim when the transfer itself has rank zero.
    if (write.getTransferRank() == 0)
      return failure();

    auto shapedType = dyn_cast<ShapedType>(write.getBase().getType());
    if (shapedType.getElementType() != write.getVectorType().getElementType())
      return failure();

    VectorType oldType = write.getVectorType();
    VectorType newType = trimLeadingUnitDims(oldType, /*zeroDimsAllowed=*/true);
    if (newType == oldType)
      return failure();
    AffineMap oldMap = write.getPermutationMap();
    ArrayRef<AffineExpr> newResults =
        oldMap.getResults().take_back(newType.getRank());
    AffineMap newMap =
        AffineMap::get(oldMap.getNumDims(), oldMap.getNumSymbols(), newResults,
                       rewriter.getContext());

    ArrayAttr inBoundsAttr;
    if (write.getInBounds())
      inBoundsAttr = rewriter.getArrayAttr(
          write.getInBoundsAttr().getValue().take_back(newType.getRank()));

    auto newVector = rewriter.createOrFold<vector::ShapeCastOp>(
        write.getLoc(), newType, write.getVector());

    if (write.getMask()) {
      Value newMask = dropUnitDimsFromMask(rewriter, write.getLoc(),
                                           write.getMask(), newType, newMap);
      rewriter.replaceOpWithNewOp<vector::TransferWriteOp>(
          write, newVector, write.getBase(), write.getIndices(),
          AffineMapAttr::get(newMap), newMask, inBoundsAttr);
      return success();
    }

    rewriter.replaceOpWithNewOp<vector::TransferWriteOp>(
        write, newVector, write.getBase(), write.getIndices(),
        AffineMapAttr::get(newMap), inBoundsAttr);
    return success();
  }
};

} // namespace

namespace {
struct VectorContractOperandCastPlan {
  AffineMap map;
  SmallVector<int64_t> permutation;
  bool dropLeadingUnitDim = false;
  bool permuteOperand = false;
};
} // namespace

FailureOr<Value>
mlir::vector::castAwayContractionLeadingOneDim(vector::ContractionOp contractOp,
                                               MaskingOpInterface maskingOp,
                                               RewriterBase &rewriter) {
  VectorType oldAccType = dyn_cast<VectorType>(contractOp.getAccType());
  if (oldAccType == nullptr)
    return failure();
  if (oldAccType.getRank() < 1)
    return failure();
  if (!isNonScalableUnitDim(oldAccType, 0))
    return failure();
  // currently we support only dropping one dim but the pattern can be applied
  // greedily to drop more.
  int64_t dropDim = 1;

  auto oldIndexingMaps = contractOp.getIndexingMapsArray();
  SmallVector<AffineMap> newIndexingMaps;

  auto oldIteratorTypes = contractOp.getIteratorTypes();
  SmallVector<Attribute> newIteratorTypes;

  int64_t dimToDrop = oldIndexingMaps[2].getDimPosition(0);

  if (!isParallelIterator(oldIteratorTypes[dimToDrop]))
    // only parallel type iterators can be dropped.
    return failure();

  for (const auto &it : llvm::enumerate(oldIteratorTypes)) {
    int64_t currDim = it.index();
    if (currDim == dimToDrop)
      continue;
    newIteratorTypes.push_back(it.value());
  }

  SmallVector<Value> operands = {contractOp.getLhs(), contractOp.getRhs(),
                                 contractOp.getAcc()};
  SmallVector<VectorContractOperandCastPlan> operandCastPlans;
  SmallVector<Value> newOperands;
  auto loc = contractOp.getLoc();

  if (maskingOp) {
    auto oldMaskType = cast<VectorType>(maskingOp.getMask().getType());
    if (oldMaskType.getRank() <= 1 || dimToDrop >= oldMaskType.getRank() ||
        !isNonScalableUnitDim(oldMaskType, dimToDrop))
      return failure();
  }

  for (const auto &it : llvm::enumerate(oldIndexingMaps)) {
    // Check if the dim to be dropped exists as a leading dim in the operand
    // if it does then we use vector.shape_cast to drop it.
    VectorContractOperandCastPlan plan;
    SmallVector<AffineExpr> results;
    plan.map = it.value();
    int64_t originalZeroDim = plan.map.getDimPosition(0);
    if (originalZeroDim != dimToDrop) {
      // There are two reasons to be in this path, 1. We need to
      // permute the operand type to make the dim to be dropped
      // leading. 2. The dim to be dropped does not exist and in
      // that case we dont want to add a unit permutation but we must
      // check all the indices to make sure this is the case.
      SmallVector<AffineExpr> permutedResults;

      for (int64_t i = 0, e = plan.map.getNumResults(); i < e; ++i) {
        int64_t currDim = plan.map.getDimPosition(i);
        if (currDim == dimToDrop) {
          plan.permuteOperand = true;
          plan.permutation.insert(plan.permutation.begin(), i);
          auto targetExpr = rewriter.getAffineDimExpr(currDim);
          permutedResults.insert(permutedResults.begin(), targetExpr);
        } else {
          plan.permutation.push_back(i);
          auto targetExpr = rewriter.getAffineDimExpr(currDim);
          permutedResults.push_back(targetExpr);
        }
      }

      // Update the map now so that the later shape_cast drops the correct dim.
      if (plan.permuteOperand) {
        plan.map = AffineMap::get(plan.map.getNumDims(), 0, permutedResults,
                                  contractOp.getContext());
        if (plan.map.getDimPosition(0) == dimToDrop) {
          auto operandType = cast<VectorType>(operands[it.index()].getType());
          if (!areLeadingDimsUnitAfterPermutation(operandType, plan.permutation,
                                                  dropDim))
            return failure();
        }
      }
    }
    // We have taken care to have the dim to be dropped be
    // the leading dim. If its still not leading that means it
    // does not exist in this operand and hence we do not need a shape_cast.
    if (plan.map.getDimPosition(0) == dimToDrop)
      plan.dropLeadingUnitDim = true;
    if (plan.dropLeadingUnitDim && originalZeroDim == dimToDrop &&
        !areLeadingDimsUnit(cast<VectorType>(operands[it.index()].getType()),
                            dropDim))
      return failure();

    for (int64_t i = 0, e = plan.map.getNumResults(); i < e; ++i) {
      int64_t currDim = plan.map.getDimPosition(i);
      if (currDim == dimToDrop)
        // This is the dim we are dropping.
        continue;
      auto targetExpr = rewriter.getAffineDimExpr(
          currDim < dimToDrop ? currDim : currDim - 1);
      results.push_back(targetExpr);
    }
    newIndexingMaps.push_back(AffineMap::get(plan.map.getNumDims() - 1, 0,
                                             results, contractOp.getContext()));
    operandCastPlans.push_back(std::move(plan));
  }

  for (auto [plan, operand] : llvm::zip_equal(operandCastPlans, operands)) {
    Value newOperand = operand;
    if (plan.permuteOperand)
      newOperand = rewriter.createOrFold<vector::ShapeCastOp>(
          loc,
          permuteVectorType(cast<VectorType>(newOperand.getType()),
                            plan.permutation),
          newOperand);
    if (plan.dropLeadingUnitDim)
      newOperand =
          dropLeadingUnitDims0DIsScalar(rewriter, loc, newOperand, dropDim);
    newOperands.push_back(newOperand);
  }

  // Depending on whether this vector.contract is masked, the replacing Op
  // should either be a new vector.contract Op or vector.mask Op.
  Operation *newOp = vector::ContractionOp::create(
      rewriter, loc, newOperands[0], newOperands[1], newOperands[2],
      rewriter.getAffineMapArrayAttr(newIndexingMaps),
      rewriter.getArrayAttr(newIteratorTypes), contractOp.getKind());

  if (maskingOp) {
    Value newMask = dropUnitDim(rewriter, loc, maskingOp.getMask(), dimToDrop,
                                /*zeroDimsAllowed=*/false);

    newOp = mlir::vector::maskOperation(rewriter, newOp, newMask);
  }

  if (!isa<VectorType>(newOp->getResults()[0].getType()))
    return vector::BroadcastOp::create(rewriter, loc,
                                       contractOp->getResultTypes()[0],
                                       newOp->getResults()[0])
        .getResult();

  return vector::ShapeCastOp::create(rewriter, loc,
                                     contractOp->getResultTypes()[0],
                                     newOp->getResults()[0])
      .getResult();
}

namespace {

/// Turns vector.contract on vector with leading 1 dimensions into
/// vector.shape_cast followed by vector.contract on vector without leading
/// 1 dimensions. Non-leading unit dimensions are dropped via direct
/// shape_casts.
struct CastAwayContractionLeadingOneDim
    : public MaskableOpRewritePattern<vector::ContractionOp> {
  using MaskableOpRewritePattern::MaskableOpRewritePattern;

  FailureOr<Value>
  matchAndRewriteMaskableOp(vector::ContractionOp contractOp,
                            MaskingOpInterface maskingOp,
                            PatternRewriter &rewriter) const override {
    return castAwayContractionLeadingOneDim(contractOp, maskingOp, rewriter);
  }
};

/// Looks at elementwise operations on vectors with at least one leading
/// dimension equal 1, e.g. vector<1x[4]x1xf32> (but not vector<2x[4]x1xf32>),
/// and casts away the leading one dimensions (_plural_) with shape_cast.
///
/// Example before:
///     %1 = arith.mulf %arg0, %arg1 : vector<1x4x1xf32>
/// Example after:
///    %2 = vector.shape_cast %arg0 : vector<1x4x1xf32> to vector<4x1xf32>
///    %3 = vector.shape_cast %arg1 : vector<1x4x1xf32> to vector<4x1xf32>
///    %4 = arith.mulf %2, %3 : vector<4x1xf32>
///    %5 = vector.shape_cast %4 : vector<4x1xf32> to vector<1x4x1xf32>
///
/// Does support scalable vectors.
class CastAwayElementwiseLeadingOneDim : public RewritePattern {
public:
  CastAwayElementwiseLeadingOneDim(MLIRContext *context,
                                   PatternBenefit benefit = 1)
      : RewritePattern(MatchAnyOpTypeTag(), benefit, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!OpTrait::hasElementwiseMappableTraits(op) || op->getNumResults() != 1)
      return failure();
    auto vecType = dyn_cast<VectorType>(op->getResultTypes()[0]);
    if (!vecType)
      return failure();
    VectorType newVecType =
        trimLeadingUnitDims(vecType, /*zeroDimsAllowed=*/true);
    if (newVecType == vecType)
      return failure();
    SmallVector<Value, 4> newOperands;
    for (Value operand : op->getOperands()) {
      if (auto opVecType = dyn_cast<VectorType>(operand.getType()))
        newOperands.push_back(rewriter.createOrFold<vector::ShapeCastOp>(
            op->getLoc(),
            trimLeadingUnitDims(opVecType, /*zeroDimsAllowed=*/true), operand));
      else
        newOperands.push_back(operand);
    }
    Operation *newOp =
        rewriter.create(op->getLoc(), op->getName().getIdentifier(),
                        newOperands, newVecType, op->getAttrs());
    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(op, vecType,
                                                     newOp->getResult(0));
    return success();
  }
};
} // namespace

namespace {

// Drops leading unit dimensions from load-like memory operations by
// shape_casting each vector operand and shape_casting the result back to the
// original type.
template <typename OpTy>
struct CastAwayLoadLikeLeadingOneDim : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    VectorType oldResultType = op.getVectorType();
    constexpr bool zeroDimsAllowed =
        llvm::is_one_of<OpTy, vector::LoadOp>::value;
    VectorType newResultType =
        trimLeadingUnitDims(oldResultType, zeroDimsAllowed);
    if (newResultType == oldResultType)
      return failure();
    int64_t nDropped = oldResultType.getRank() - newResultType.getRank();

    Location loc = op.getLoc();
    SmallVector<Value> newOperands;
    newOperands.reserve(op->getNumOperands());
    for (Value operand : op->getOperands()) {
      if (isa<VectorType>(operand.getType())) {
        newOperands.push_back(dropLeadingUnitDims(rewriter, loc, operand,
                                                  nDropped, zeroDimsAllowed));
      } else {
        newOperands.push_back(operand);
      }
    }

    Operation *newOp =
        rewriter.create(loc, op->getName().getIdentifier(), newOperands,
                        TypeRange{newResultType}, op->getAttrs());
    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(op, oldResultType,
                                                     newOp->getResult(0));
    return success();
  }
};

// Drops leading unit dimensions from store-like memory operations by
// shape_casting each vector operand and leaving any scalar operands alone.
template <typename OpTy>
struct CastAwayStoreLikeLeadingOneDim : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    VectorType oldVecType = op.getVectorType();
    constexpr bool zeroDimsAllowed =
        llvm::is_one_of<OpTy, vector::StoreOp>::value;
    VectorType newVecType = trimLeadingUnitDims(oldVecType, zeroDimsAllowed);
    if (newVecType == oldVecType)
      return failure();
    int64_t nDropped = oldVecType.getRank() - newVecType.getRank();

    Location loc = op.getLoc();
    SmallVector<Value> newOperands;
    newOperands.reserve(op->getNumOperands());
    for (Value operand : op->getOperands()) {
      if (isa<VectorType>(operand.getType())) {
        newOperands.push_back(dropLeadingUnitDims(rewriter, loc, operand,
                                                  nDropped, zeroDimsAllowed));
      } else {
        newOperands.push_back(operand);
      }
    }

    Operation *newOp =
        rewriter.create(loc, op->getName().getIdentifier(), newOperands,
                        op->getResultTypes(), op->getAttrs());
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

// Drops leading 1 dimensions from vector.constant_mask and shape_casts back to
// the original shape.
struct CastAwayConstantMaskLeadingOneDim
    : public OpRewritePattern<vector::ConstantMaskOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(vector::ConstantMaskOp mask,
                                PatternRewriter &rewriter) const override {
    VectorType oldType = mask.getType();
    VectorType newType = trimLeadingUnitDims(oldType,
                                             /*zeroDimsAllowed=*/true);

    if (newType == oldType)
      return failure();

    int64_t dropDim = oldType.getRank() - newType.getRank();
    ArrayRef<int64_t> dimSizes = mask.getMaskDimSizes();

    // If any of the folded unit dims has a size of `0`, the entire leading
    // mask region is zero. Otherwise the folded unit dims have no effect on
    // the mask.
    SmallVector<int64_t> newDimSizes;
    if (newType.getRank() == 0) {
      newDimSizes.push_back(llvm::product_of(dimSizes));
    } else {
      int64_t flatLeadingSize =
          llvm::product_of(dimSizes.take_front(dropDim + 1));
      newDimSizes.push_back(flatLeadingSize);
      newDimSizes.append(dimSizes.begin() + dropDim + 1, dimSizes.end());
    }

    auto newMask = vector::ConstantMaskOp::create(rewriter, mask.getLoc(),
                                                  newType, newDimSizes);
    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(mask, oldType, newMask);
    return success();
  }
};

} // namespace

void mlir::vector::populateCastAwayVectorLeadingOneDimPatterns(
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns
      .add<CastAwayExtractStridedSliceLeadingOneDim,
           CastAwayInsertStridedSliceLeadingOneDim, CastAwayInsertLeadingOneDim,
           CastAwayConstantMaskLeadingOneDim, CastAwayTransferReadLeadingOneDim,
           CastAwayTransferWriteLeadingOneDim, CastAwayElementwiseLeadingOneDim,
           CastAwayContractionLeadingOneDim,
           CastAwayLoadLikeLeadingOneDim<vector::LoadOp>,
           CastAwayLoadLikeLeadingOneDim<vector::MaskedLoadOp>,
           CastAwayLoadLikeLeadingOneDim<vector::ExpandLoadOp>,
           CastAwayLoadLikeLeadingOneDim<vector::GatherOp>,
           CastAwayStoreLikeLeadingOneDim<vector::StoreOp>,
           CastAwayStoreLikeLeadingOneDim<vector::MaskedStoreOp>,
           CastAwayStoreLikeLeadingOneDim<vector::CompressStoreOp>,
           CastAwayStoreLikeLeadingOneDim<vector::ScatterOp>>(
          patterns.getContext(), benefit);
}
