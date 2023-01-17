//===- VectorTransferPermutationMapRewritePatterns.cpp - Xfer map rewrite -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements rewrite patterns for the permutation_map attribute of
// vector.transfer operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Interfaces/VectorInterfaces.h"

using namespace mlir;
using namespace mlir::vector;

/// Transpose a vector transfer op's `in_bounds` attribute by applying reverse
/// permutation based on the given indices.
static ArrayAttr
inverseTransposeInBoundsAttr(OpBuilder &builder, ArrayAttr attr,
                             const SmallVector<unsigned> &permutation) {
  SmallVector<bool> newInBoundsValues(permutation.size());
  size_t index = 0;
  for (unsigned pos : permutation)
    newInBoundsValues[pos] =
        attr.getValue()[index++].cast<BoolAttr>().getValue();
  return builder.getBoolArrayAttr(newInBoundsValues);
}

/// Extend the rank of a vector Value by `addedRanks` by adding outer unit
/// dimensions.
static Value extendVectorRank(OpBuilder &builder, Location loc, Value vec,
                              int64_t addedRank) {
  auto originalVecType = vec.getType().cast<VectorType>();
  SmallVector<int64_t> newShape(addedRank, 1);
  newShape.append(originalVecType.getShape().begin(),
                  originalVecType.getShape().end());
  VectorType newVecType =
      VectorType::get(newShape, originalVecType.getElementType());
  return builder.create<vector::BroadcastOp>(loc, newVecType, vec);
}

/// Lower transfer_read op with permutation into a transfer_read with a
/// permutation map composed of leading zeros followed by a minor identiy +
/// vector.transpose op.
/// Ex:
///     vector.transfer_read ...
///         permutation_map: (d0, d1, d2) -> (0, d1)
/// into:
///     %v = vector.transfer_read ...
///         permutation_map: (d0, d1, d2) -> (d1, 0)
///     vector.transpose %v, [1, 0]
///
///     vector.transfer_read ...
///         permutation_map: (d0, d1, d2, d3) -> (0, 0, 0, d1, d3)
/// into:
///     %v = vector.transfer_read ...
///         permutation_map: (d0, d1, d2, d3) -> (0, 0, d1, 0, d3)
///     vector.transpose %v, [0, 1, 3, 2, 4]
/// Note that an alternative is to transform it to linalg.transpose +
/// vector.transfer_read to do the transpose in memory instead.
struct TransferReadPermutationLowering
    : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp op,
                                PatternRewriter &rewriter) const override {
    // TODO: support 0-d corner case.
    if (op.getTransferRank() == 0)
      return failure();

    SmallVector<unsigned> permutation;
    AffineMap map = op.getPermutationMap();
    if (map.getNumResults() == 0)
      return failure();
    if (!map.isPermutationOfMinorIdentityWithBroadcasting(permutation))
      return failure();
    AffineMap permutationMap =
        map.getPermutationMap(permutation, op.getContext());
    if (permutationMap.isIdentity())
      return failure();

    permutationMap = map.getPermutationMap(permutation, op.getContext());
    // Caluclate the map of the new read by applying the inverse permutation.
    permutationMap = inversePermutation(permutationMap);
    AffineMap newMap = permutationMap.compose(map);
    // Apply the reverse transpose to deduce the type of the transfer_read.
    ArrayRef<int64_t> originalShape = op.getVectorType().getShape();
    SmallVector<int64_t> newVectorShape(originalShape.size());
    for (const auto &pos : llvm::enumerate(permutation)) {
      newVectorShape[pos.value()] = originalShape[pos.index()];
    }

    // Transpose in_bounds attribute.
    ArrayAttr newInBoundsAttr =
        op.getInBounds() ? inverseTransposeInBoundsAttr(
                               rewriter, op.getInBounds().value(), permutation)
                         : ArrayAttr();

    // Generate new transfer_read operation.
    VectorType newReadType =
        VectorType::get(newVectorShape, op.getVectorType().getElementType());
    Value newRead = rewriter.create<vector::TransferReadOp>(
        op.getLoc(), newReadType, op.getSource(), op.getIndices(),
        AffineMapAttr::get(newMap), op.getPadding(), op.getMask(),
        newInBoundsAttr);

    // Transpose result of transfer_read.
    SmallVector<int64_t> transposePerm(permutation.begin(), permutation.end());
    rewriter.replaceOpWithNewOp<vector::TransposeOp>(op, newRead,
                                                     transposePerm);
    return success();
  }
};

/// Lower transfer_write op with permutation into a transfer_write with a
/// minor identity permutation map. (transfer_write ops cannot have broadcasts.)
/// Ex:
///     vector.transfer_write %v ...
///         permutation_map: (d0, d1, d2) -> (d2, d0, d1)
/// into:
///     %tmp = vector.transpose %v, [2, 0, 1]
///     vector.transfer_write %tmp ...
///         permutation_map: (d0, d1, d2) -> (d0, d1, d2)
///
///     vector.transfer_write %v ...
///         permutation_map: (d0, d1, d2, d3) -> (d3, d2)
/// into:
///     %tmp = vector.transpose %v, [1, 0]
///     %v = vector.transfer_write %tmp ...
///         permutation_map: (d0, d1, d2, d3) -> (d2, d3)
struct TransferWritePermutationLowering
    : public OpRewritePattern<vector::TransferWriteOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferWriteOp op,
                                PatternRewriter &rewriter) const override {
    // TODO: support 0-d corner case.
    if (op.getTransferRank() == 0)
      return failure();

    SmallVector<unsigned> permutation;
    AffineMap map = op.getPermutationMap();
    if (map.isMinorIdentity())
      return failure();
    if (!map.isPermutationOfMinorIdentityWithBroadcasting(permutation))
      return failure();

    // Remove unused dims from the permutation map. E.g.:
    // E.g.:  (d0, d1, d2, d3, d4, d5) -> (d5, d3, d4)
    // comp = (d0, d1, d2) -> (d2, d0, d1)
    auto comp = compressUnusedDims(map);
    AffineMap permutationMap = inversePermutation(comp);
    // Get positions of remaining result dims.
    SmallVector<int64_t> indices;
    llvm::transform(permutationMap.getResults(), std::back_inserter(indices),
                    [](AffineExpr expr) {
                      return expr.dyn_cast<AffineDimExpr>().getPosition();
                    });

    // Transpose in_bounds attribute.
    ArrayAttr newInBoundsAttr =
        op.getInBounds() ? inverseTransposeInBoundsAttr(
                               rewriter, op.getInBounds().value(), permutation)
                         : ArrayAttr();

    // Generate new transfer_write operation.
    Value newVec = rewriter.create<vector::TransposeOp>(
        op.getLoc(), op.getVector(), indices);
    auto newMap = AffineMap::getMinorIdentityMap(
        map.getNumDims(), map.getNumResults(), rewriter.getContext());
    rewriter.replaceOpWithNewOp<vector::TransferWriteOp>(
        op, newVec, op.getSource(), op.getIndices(), AffineMapAttr::get(newMap),
        op.getMask(), newInBoundsAttr);

    return success();
  }
};

/// Convert a transfer.write op with a map which isn't the permutation of a
/// minor identity into a vector.broadcast + transfer_write with permutation of
/// minor identity map by adding unit dim on inner dimension. Ex:
/// ```
///   vector.transfer_write %v
///     {permutation_map = affine_map<(d0, d1, d2, d3) -> (d1, d2)>} :
///     vector<8x16xf32>
/// ```
/// into:
/// ```
///   %v1 = vector.broadcast %v : vector<8x16xf32> to vector<1x8x16xf32>
///   vector.transfer_write %v1
///     {permutation_map = affine_map<(d0, d1, d2, d3) -> (d3, d1, d2)>} :
///     vector<1x8x16xf32>
/// ```
struct TransferWriteNonPermutationLowering
    : public OpRewritePattern<vector::TransferWriteOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferWriteOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getTransferRank() == 0)
      return failure();
    SmallVector<unsigned> permutation;
    AffineMap map = op.getPermutationMap();
    if (map.isPermutationOfMinorIdentityWithBroadcasting(permutation))
      return failure();

    // Missing outer dimensions are allowed, find the most outer existing
    // dimension then deduce the missing inner dimensions.
    SmallVector<bool> foundDim(map.getNumDims(), false);
    for (AffineExpr exp : map.getResults()) {
      foundDim[exp.cast<AffineDimExpr>().getPosition()] = true;
    }
    SmallVector<AffineExpr> exprs;
    bool foundFirstDim = false;
    SmallVector<int64_t> missingInnerDim;
    for (size_t i = 0; i < foundDim.size(); i++) {
      if (foundDim[i]) {
        foundFirstDim = true;
        continue;
      }
      if (!foundFirstDim)
        continue;
      // Once we found one outer dimension existing in the map keep track of all
      // the missing dimensions after that.
      missingInnerDim.push_back(i);
      exprs.push_back(rewriter.getAffineDimExpr(i));
    }
    // Add unit dims at the beginning of the shape.
    Value newVec = extendVectorRank(rewriter, op.getLoc(), op.getVector(),
                                    missingInnerDim.size());
    exprs.append(map.getResults().begin(), map.getResults().end());
    AffineMap newMap =
        AffineMap::get(map.getNumDims(), 0, exprs, op.getContext());
    ArrayAttr newInBoundsAttr;
    if (op.getInBounds()) {
      // All the new dimensions added are inbound.
      SmallVector<bool> newInBoundsValues(missingInnerDim.size(), true);
      for (Attribute attr : op.getInBounds().value().getValue()) {
        newInBoundsValues.push_back(attr.cast<BoolAttr>().getValue());
      }
      newInBoundsAttr = rewriter.getBoolArrayAttr(newInBoundsValues);
    }
    rewriter.replaceOpWithNewOp<vector::TransferWriteOp>(
        op, newVec, op.getSource(), op.getIndices(), AffineMapAttr::get(newMap),
        op.getMask(), newInBoundsAttr);
    return success();
  }
};

/// Lower transfer_read op with broadcast in the leading dimensions into
/// transfer_read of lower rank + vector.broadcast.
/// Ex: vector.transfer_read ...
///         permutation_map: (d0, d1, d2, d3) -> (0, d1, 0, d3)
/// into:
///     %v = vector.transfer_read ...
///         permutation_map: (d0, d1, d2, d3) -> (d1, 0, d3)
///     vector.broadcast %v
struct TransferOpReduceRank : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp op,
                                PatternRewriter &rewriter) const override {
    // TODO: support 0-d corner case.
    if (op.getTransferRank() == 0)
      return failure();

    AffineMap map = op.getPermutationMap();
    unsigned numLeadingBroadcast = 0;
    for (auto expr : map.getResults()) {
      auto dimExpr = expr.dyn_cast<AffineConstantExpr>();
      if (!dimExpr || dimExpr.getValue() != 0)
        break;
      numLeadingBroadcast++;
    }
    // If there are no leading zeros in the map there is nothing to do.
    if (numLeadingBroadcast == 0)
      return failure();
    VectorType originalVecType = op.getVectorType();
    unsigned reducedShapeRank = originalVecType.getRank() - numLeadingBroadcast;
    // Calculate new map, vector type and masks without the leading zeros.
    AffineMap newMap = AffineMap::get(
        map.getNumDims(), 0, map.getResults().take_back(reducedShapeRank),
        op.getContext());
    // Only remove the leading zeros if the rest of the map is a minor identity
    // with broadasting. Otherwise we first want to permute the map.
    if (!newMap.isMinorIdentityWithBroadcasting())
      return failure();

    // TODO: support zero-dimension vectors natively.  See:
    // https://llvm.discourse.group/t/should-we-have-0-d-vectors/3097.
    // In the meantime, lower these to a scalar load when they pop up.
    if (reducedShapeRank == 0) {
      Value newRead;
      if (op.getShapedType().isa<TensorType>()) {
        newRead = rewriter.create<tensor::ExtractOp>(
            op.getLoc(), op.getSource(), op.getIndices());
      } else {
        newRead = rewriter.create<memref::LoadOp>(
            op.getLoc(), originalVecType.getElementType(), op.getSource(),
            op.getIndices());
      }
      rewriter.replaceOpWithNewOp<vector::BroadcastOp>(op, originalVecType,
                                                       newRead);
      return success();
    }
    SmallVector<int64_t> newShape = llvm::to_vector<4>(
        originalVecType.getShape().take_back(reducedShapeRank));
    // Vector rank cannot be zero. Handled by TransferReadToVectorLoadLowering.
    if (newShape.empty())
      return failure();
    VectorType newReadType =
        VectorType::get(newShape, originalVecType.getElementType());
    ArrayAttr newInBoundsAttr =
        op.getInBounds()
            ? rewriter.getArrayAttr(
                  op.getInBoundsAttr().getValue().take_back(reducedShapeRank))
            : ArrayAttr();
    Value newRead = rewriter.create<vector::TransferReadOp>(
        op.getLoc(), newReadType, op.getSource(), op.getIndices(),
        AffineMapAttr::get(newMap), op.getPadding(), op.getMask(),
        newInBoundsAttr);
    rewriter.replaceOpWithNewOp<vector::BroadcastOp>(op, originalVecType,
                                                     newRead);
    return success();
  }
};

void mlir::vector::populateVectorTransferPermutationMapLoweringPatterns(
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns
      .add<TransferReadPermutationLowering, TransferWritePermutationLowering,
           TransferOpReduceRank, TransferWriteNonPermutationLowering>(
          patterns.getContext(), benefit);
}
