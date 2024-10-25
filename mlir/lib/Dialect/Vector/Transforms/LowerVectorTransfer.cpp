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
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
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
        cast<BoolAttr>(attr.getValue()[index++]).getValue();
  return builder.getBoolArrayAttr(newInBoundsValues);
}

/// Extend the rank of a vector Value by `addedRanks` by adding outer unit
/// dimensions.
static Value extendVectorRank(OpBuilder &builder, Location loc, Value vec,
                              int64_t addedRank) {
  auto originalVecType = cast<VectorType>(vec.getType());
  SmallVector<int64_t> newShape(addedRank, 1);
  newShape.append(originalVecType.getShape().begin(),
                  originalVecType.getShape().end());

  SmallVector<bool> newScalableDims(addedRank, false);
  newScalableDims.append(originalVecType.getScalableDims().begin(),
                         originalVecType.getScalableDims().end());
  VectorType newVecType = VectorType::get(
      newShape, originalVecType.getElementType(), newScalableDims);
  return builder.create<vector::BroadcastOp>(loc, newVecType, vec);
}

/// Extend the rank of a vector Value by `addedRanks` by adding inner unit
/// dimensions.
static Value extendMaskRank(OpBuilder &builder, Location loc, Value vec,
                            int64_t addedRank) {
  Value broadcasted = extendVectorRank(builder, loc, vec, addedRank);
  SmallVector<int64_t> permutation;
  for (int64_t i = addedRank,
               e = cast<VectorType>(broadcasted.getType()).getRank();
       i < e; ++i)
    permutation.push_back(i);
  for (int64_t i = 0; i < addedRank; ++i)
    permutation.push_back(i);
  return builder.create<vector::TransposeOp>(loc, broadcasted, permutation);
}

//===----------------------------------------------------------------------===//
// populateVectorTransferPermutationMapLoweringPatterns
//===----------------------------------------------------------------------===//

namespace {
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
    : public MaskableOpRewritePattern<vector::TransferReadOp> {
  using MaskableOpRewritePattern::MaskableOpRewritePattern;

  FailureOr<mlir::Value>
  matchAndRewriteMaskableOp(vector::TransferReadOp op,
                            MaskingOpInterface maskOp,
                            PatternRewriter &rewriter) const override {
    // TODO: support 0-d corner case.
    if (op.getTransferRank() == 0)
      return rewriter.notifyMatchFailure(op, "0-d corner case not supported");
    // TODO: Support transfer_read inside MaskOp case.
    if (maskOp)
      return rewriter.notifyMatchFailure(op, "Masked case not supported");

    SmallVector<unsigned> permutation;
    AffineMap map = op.getPermutationMap();
    if (map.getNumResults() == 0)
      return rewriter.notifyMatchFailure(op, "0 result permutation map");
    if (!map.isPermutationOfMinorIdentityWithBroadcasting(permutation)) {
      return rewriter.notifyMatchFailure(
          op, "map is not permutable to minor identity, apply another pattern");
    }
    AffineMap permutationMap =
        map.getPermutationMap(permutation, op.getContext());
    if (permutationMap.isIdentity())
      return rewriter.notifyMatchFailure(op, "map is not identity");

    permutationMap = map.getPermutationMap(permutation, op.getContext());
    // Caluclate the map of the new read by applying the inverse permutation.
    permutationMap = inversePermutation(permutationMap);
    AffineMap newMap = permutationMap.compose(map);
    // Apply the reverse transpose to deduce the type of the transfer_read.
    ArrayRef<int64_t> originalShape = op.getVectorType().getShape();
    SmallVector<int64_t> newVectorShape(originalShape.size());
    ArrayRef<bool> originalScalableDims = op.getVectorType().getScalableDims();
    SmallVector<bool> newScalableDims(originalShape.size());
    for (const auto &pos : llvm::enumerate(permutation)) {
      newVectorShape[pos.value()] = originalShape[pos.index()];
      newScalableDims[pos.value()] = originalScalableDims[pos.index()];
    }

    // Transpose in_bounds attribute.
    ArrayAttr newInBoundsAttr =
        inverseTransposeInBoundsAttr(rewriter, op.getInBounds(), permutation);

    // Generate new transfer_read operation.
    VectorType newReadType = VectorType::get(
        newVectorShape, op.getVectorType().getElementType(), newScalableDims);
    Value newRead = rewriter.create<vector::TransferReadOp>(
        op.getLoc(), newReadType, op.getSource(), op.getIndices(),
        AffineMapAttr::get(newMap), op.getPadding(), op.getMask(),
        newInBoundsAttr);

    // Transpose result of transfer_read.
    SmallVector<int64_t> transposePerm(permutation.begin(), permutation.end());
    return rewriter
        .create<vector::TransposeOp>(op.getLoc(), newRead, transposePerm)
        .getResult();
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
    : public MaskableOpRewritePattern<vector::TransferWriteOp> {
  using MaskableOpRewritePattern::MaskableOpRewritePattern;

  FailureOr<mlir::Value>
  matchAndRewriteMaskableOp(vector::TransferWriteOp op,
                            MaskingOpInterface maskOp,
                            PatternRewriter &rewriter) const override {
    // TODO: support 0-d corner case.
    if (op.getTransferRank() == 0)
      return rewriter.notifyMatchFailure(op, "0-d corner case not supported");
    // TODO: Support transfer_write inside MaskOp case.
    if (maskOp)
      return rewriter.notifyMatchFailure(op, "Masked case not supported");

    SmallVector<unsigned> permutation;
    AffineMap map = op.getPermutationMap();
    if (map.isMinorIdentity())
      return rewriter.notifyMatchFailure(op, "map is already minor identity");

    if (!map.isPermutationOfMinorIdentityWithBroadcasting(permutation)) {
      return rewriter.notifyMatchFailure(
          op, "map is not permutable to minor identity, apply another pattern");
    }

    // Remove unused dims from the permutation map. E.g.:
    // E.g.:  (d0, d1, d2, d3, d4, d5) -> (d5, d3, d4)
    // comp = (d0, d1, d2) -> (d2, d0, d1)
    auto comp = compressUnusedDims(map);
    AffineMap permutationMap = inversePermutation(comp);
    // Get positions of remaining result dims.
    SmallVector<int64_t> indices;
    llvm::transform(permutationMap.getResults(), std::back_inserter(indices),
                    [](AffineExpr expr) {
                      return dyn_cast<AffineDimExpr>(expr).getPosition();
                    });

    // Transpose in_bounds attribute.
    ArrayAttr newInBoundsAttr =
        inverseTransposeInBoundsAttr(rewriter, op.getInBounds(), permutation);

    // Generate new transfer_write operation.
    Value newVec = rewriter.create<vector::TransposeOp>(
        op.getLoc(), op.getVector(), indices);
    auto newMap = AffineMap::getMinorIdentityMap(
        map.getNumDims(), map.getNumResults(), rewriter.getContext());
    auto newWrite = rewriter.create<vector::TransferWriteOp>(
        op.getLoc(), newVec, op.getSource(), op.getIndices(),
        AffineMapAttr::get(newMap), op.getMask(), newInBoundsAttr);
    if (newWrite.hasPureTensorSemantics())
      return newWrite.getResult();
    // In the memref case there's no return value. Use empty value to signal
    // success.
    return Value();
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
    : public MaskableOpRewritePattern<vector::TransferWriteOp> {
  using MaskableOpRewritePattern::MaskableOpRewritePattern;

  FailureOr<mlir::Value>
  matchAndRewriteMaskableOp(vector::TransferWriteOp op,
                            MaskingOpInterface maskOp,
                            PatternRewriter &rewriter) const override {
    // TODO: support 0-d corner case.
    if (op.getTransferRank() == 0)
      return rewriter.notifyMatchFailure(op, "0-d corner case not supported");
    // TODO: Support transfer_write inside MaskOp case.
    if (maskOp)
      return rewriter.notifyMatchFailure(op, "Masked case not supported");

    SmallVector<unsigned> permutation;
    AffineMap map = op.getPermutationMap();
    if (map.isPermutationOfMinorIdentityWithBroadcasting(permutation)) {
      return rewriter.notifyMatchFailure(
          op,
          "map is already permutable to minor identity, apply another pattern");
    }

    // Missing outer dimensions are allowed, find the most outer existing
    // dimension then deduce the missing inner dimensions.
    SmallVector<bool> foundDim(map.getNumDims(), false);
    for (AffineExpr exp : map.getResults())
      foundDim[cast<AffineDimExpr>(exp).getPosition()] = true;
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
    // Vector: add unit dims at the beginning of the shape.
    Value newVec = extendVectorRank(rewriter, op.getLoc(), op.getVector(),
                                    missingInnerDim.size());
    // Mask: add unit dims at the end of the shape.
    Value newMask;
    if (op.getMask())
      newMask = extendMaskRank(rewriter, op.getLoc(), op.getMask(),
                               missingInnerDim.size());
    exprs.append(map.getResults().begin(), map.getResults().end());
    AffineMap newMap =
        AffineMap::get(map.getNumDims(), 0, exprs, op.getContext());
    // All the new dimensions added are inbound.
    SmallVector<bool> newInBoundsValues(missingInnerDim.size(), true);
    for (int64_t i = 0, e = op.getVectorType().getRank(); i < e; ++i) {
      newInBoundsValues.push_back(op.isDimInBounds(i));
    }
    ArrayAttr newInBoundsAttr = rewriter.getBoolArrayAttr(newInBoundsValues);
    auto newWrite = rewriter.create<vector::TransferWriteOp>(
        op.getLoc(), newVec, op.getSource(), op.getIndices(),
        AffineMapAttr::get(newMap), newMask, newInBoundsAttr);
    if (newWrite.hasPureTensorSemantics())
      return newWrite.getResult();
    // In the memref case there's no return value. Use empty value to signal
    // success.
    return Value();
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
struct TransferOpReduceRank
    : public MaskableOpRewritePattern<vector::TransferReadOp> {
  using MaskableOpRewritePattern::MaskableOpRewritePattern;

  FailureOr<mlir::Value>
  matchAndRewriteMaskableOp(vector::TransferReadOp op,
                            MaskingOpInterface maskOp,
                            PatternRewriter &rewriter) const override {
    // TODO: support 0-d corner case.
    if (op.getTransferRank() == 0)
      return rewriter.notifyMatchFailure(op, "0-d corner case not supported");
    // TODO: support masked case.
    if (maskOp)
      return rewriter.notifyMatchFailure(op, "Masked case not supported");

    AffineMap map = op.getPermutationMap();
    unsigned numLeadingBroadcast = 0;
    for (auto expr : map.getResults()) {
      auto dimExpr = dyn_cast<AffineConstantExpr>(expr);
      if (!dimExpr || dimExpr.getValue() != 0)
        break;
      numLeadingBroadcast++;
    }
    // If there are no leading zeros in the map there is nothing to do.
    if (numLeadingBroadcast == 0)
      return rewriter.notifyMatchFailure(op, "no leading broadcasts in map");

    VectorType originalVecType = op.getVectorType();
    unsigned reducedShapeRank = originalVecType.getRank() - numLeadingBroadcast;
    // Calculate new map, vector type and masks without the leading zeros.
    AffineMap newMap = AffineMap::get(
        map.getNumDims(), 0, map.getResults().take_back(reducedShapeRank),
        op.getContext());
    // Only remove the leading zeros if the rest of the map is a minor identity
    // with broadasting. Otherwise we first want to permute the map.
    if (!newMap.isMinorIdentityWithBroadcasting()) {
      return rewriter.notifyMatchFailure(
          op, "map is not a minor identity with broadcasting");
    }

    SmallVector<int64_t> newShape(
        originalVecType.getShape().take_back(reducedShapeRank));
    SmallVector<bool> newScalableDims(
        originalVecType.getScalableDims().take_back(reducedShapeRank));

    VectorType newReadType = VectorType::get(
        newShape, originalVecType.getElementType(), newScalableDims);
    ArrayAttr newInBoundsAttr =
        op.getInBounds()
            ? rewriter.getArrayAttr(
                  op.getInBoundsAttr().getValue().take_back(reducedShapeRank))
            : ArrayAttr();
    Value newRead = rewriter.create<vector::TransferReadOp>(
        op.getLoc(), newReadType, op.getSource(), op.getIndices(),
        AffineMapAttr::get(newMap), op.getPadding(), op.getMask(),
        newInBoundsAttr);
    return rewriter
        .create<vector::BroadcastOp>(op.getLoc(), originalVecType, newRead)
        .getVector();
  }
};

} // namespace

void mlir::vector::populateVectorTransferPermutationMapLoweringPatterns(
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns
      .add<TransferReadPermutationLowering, TransferWritePermutationLowering,
           TransferOpReduceRank, TransferWriteNonPermutationLowering>(
          patterns.getContext(), benefit);
}

//===----------------------------------------------------------------------===//
// populateVectorTransferLoweringPatterns
//===----------------------------------------------------------------------===//

namespace {
/// Progressive lowering of transfer_read. This pattern supports lowering of
/// `vector.transfer_read` to a combination of `vector.load` and
/// `vector.broadcast` if all of the following hold:
/// - Stride of most minor memref dimension must be 1.
/// - Out-of-bounds masking is not required.
/// - If the memref's element type is a vector type then it coincides with the
///   result type.
/// - The permutation map doesn't perform permutation (broadcasting is allowed).
struct TransferReadToVectorLoadLowering
    : public MaskableOpRewritePattern<vector::TransferReadOp> {
  TransferReadToVectorLoadLowering(MLIRContext *context,
                                   std::optional<unsigned> maxRank,
                                   PatternBenefit benefit = 1)
      : MaskableOpRewritePattern<vector::TransferReadOp>(context, benefit),
        maxTransferRank(maxRank) {}

  FailureOr<mlir::Value>
  matchAndRewriteMaskableOp(vector::TransferReadOp read,
                            MaskingOpInterface maskOp,
                            PatternRewriter &rewriter) const override {
    if (maxTransferRank && read.getVectorType().getRank() > *maxTransferRank) {
      return rewriter.notifyMatchFailure(
          read, "vector type is greater than max transfer rank");
    }

    if (maskOp)
      return rewriter.notifyMatchFailure(read, "Masked case not supported");
    SmallVector<unsigned> broadcastedDims;
    // Permutations are handled by VectorToSCF or
    // populateVectorTransferPermutationMapLoweringPatterns.
    // We let the 0-d corner case pass-through as it is supported.
    if (!read.getPermutationMap().isMinorIdentityWithBroadcasting(
            &broadcastedDims))
      return rewriter.notifyMatchFailure(read, "not minor identity + bcast");

    auto memRefType = dyn_cast<MemRefType>(read.getShapedType());
    if (!memRefType)
      return rewriter.notifyMatchFailure(read, "not a memref source");

    // Non-unit strides are handled by VectorToSCF.
    if (!isLastMemrefDimUnitStride(memRefType))
      return rewriter.notifyMatchFailure(read, "!= 1 stride needs VectorToSCF");

    // If there is broadcasting involved then we first load the unbroadcasted
    // vector, and then broadcast it with `vector.broadcast`.
    ArrayRef<int64_t> vectorShape = read.getVectorType().getShape();
    SmallVector<int64_t> unbroadcastedVectorShape(vectorShape);
    for (unsigned i : broadcastedDims)
      unbroadcastedVectorShape[i] = 1;
    VectorType unbroadcastedVectorType = read.getVectorType().cloneWith(
        unbroadcastedVectorShape, read.getVectorType().getElementType());

    // `vector.load` supports vector types as memref's elements only when the
    // resulting vector type is the same as the element type.
    auto memrefElTy = memRefType.getElementType();
    if (isa<VectorType>(memrefElTy) && memrefElTy != unbroadcastedVectorType)
      return rewriter.notifyMatchFailure(read, "incompatible element type");

    // Otherwise, element types of the memref and the vector must match.
    if (!isa<VectorType>(memrefElTy) &&
        memrefElTy != read.getVectorType().getElementType())
      return rewriter.notifyMatchFailure(read, "non-matching element type");

    // Out-of-bounds dims are handled by MaterializeTransferMask.
    if (read.hasOutOfBoundsDim())
      return rewriter.notifyMatchFailure(read, "out-of-bounds needs mask");

    // Create vector load op.
    Operation *res;
    if (read.getMask()) {
      if (read.getVectorType().getRank() != 1)
        // vector.maskedload operates on 1-D vectors.
        return rewriter.notifyMatchFailure(
            read, "vector type is not rank 1, can't create masked load, needs "
                  "VectorToSCF");

      Value fill = rewriter.create<vector::SplatOp>(
          read.getLoc(), unbroadcastedVectorType, read.getPadding());
      res = rewriter.create<vector::MaskedLoadOp>(
          read.getLoc(), unbroadcastedVectorType, read.getSource(),
          read.getIndices(), read.getMask(), fill);
    } else {
      res = rewriter.create<vector::LoadOp>(
          read.getLoc(), unbroadcastedVectorType, read.getSource(),
          read.getIndices());
    }

    // Insert a broadcasting op if required.
    if (!broadcastedDims.empty())
      res = rewriter.create<vector::BroadcastOp>(
          read.getLoc(), read.getVectorType(), res->getResult(0));
    return res->getResult(0);
  }

  std::optional<unsigned> maxTransferRank;
};

/// Replace a 0-d vector.load with a memref.load + vector.broadcast.
// TODO: we shouldn't cross the vector/scalar domains just for this
// but atm we lack the infra to avoid it. Possible solutions include:
// - go directly to LLVM + bitcast
// - introduce a bitcast op and likely a new pointer dialect
// - let memref.load/store additionally support the 0-d vector case
// There are still deeper data layout issues lingering even in this
// trivial case (for architectures for which this matters).
struct VectorLoadToMemrefLoadLowering
    : public OpRewritePattern<vector::LoadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::LoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    auto vecType = loadOp.getVectorType();
    if (vecType.getNumElements() != 1)
      return rewriter.notifyMatchFailure(loadOp, "not a single element vector");

    auto memrefLoad = rewriter.create<memref::LoadOp>(
        loadOp.getLoc(), loadOp.getBase(), loadOp.getIndices());
    rewriter.replaceOpWithNewOp<vector::BroadcastOp>(loadOp, vecType,
                                                     memrefLoad);
    return success();
  }
};

/// Replace a 0-d vector.store with a vector.extractelement + memref.store.
struct VectorStoreToMemrefStoreLowering
    : public OpRewritePattern<vector::StoreOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::StoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    auto vecType = storeOp.getVectorType();
    if (vecType.getNumElements() != 1)
      return rewriter.notifyMatchFailure(storeOp, "not single element vector");

    Value extracted;
    if (vecType.getRank() == 0) {
      // TODO: Unifiy once ExtractOp supports 0-d vectors.
      extracted = rewriter.create<vector::ExtractElementOp>(
          storeOp.getLoc(), storeOp.getValueToStore());
    } else {
      SmallVector<int64_t> indices(vecType.getRank(), 0);
      extracted = rewriter.create<vector::ExtractOp>(
          storeOp.getLoc(), storeOp.getValueToStore(), indices);
    }

    rewriter.replaceOpWithNewOp<memref::StoreOp>(
        storeOp, extracted, storeOp.getBase(), storeOp.getIndices());
    return success();
  }
};

/// Progressive lowering of transfer_write. This pattern supports lowering of
/// `vector.transfer_write` to `vector.store` if all of the following hold:
/// - Stride of most minor memref dimension must be 1.
/// - Out-of-bounds masking is not required.
/// - If the memref's element type is a vector type then it coincides with the
///   type of the written value.
/// - The permutation map is the minor identity map (neither permutation nor
///   broadcasting is allowed).
struct TransferWriteToVectorStoreLowering
    : public MaskableOpRewritePattern<vector::TransferWriteOp> {
  TransferWriteToVectorStoreLowering(MLIRContext *context,
                                     std::optional<unsigned> maxRank,
                                     PatternBenefit benefit = 1)
      : MaskableOpRewritePattern<vector::TransferWriteOp>(context, benefit),
        maxTransferRank(maxRank) {}

  FailureOr<mlir::Value>
  matchAndRewriteMaskableOp(vector::TransferWriteOp write,
                            MaskingOpInterface maskOp,
                            PatternRewriter &rewriter) const override {
    if (maxTransferRank && write.getVectorType().getRank() > *maxTransferRank) {
      return rewriter.notifyMatchFailure(
          write, "vector type is greater than max transfer rank");
    }
    if (maskOp)
      return rewriter.notifyMatchFailure(write, "Masked case not supported");

    // Permutations are handled by VectorToSCF or
    // populateVectorTransferPermutationMapLoweringPatterns.
    if ( // pass-through for the 0-d corner case.
        !write.getPermutationMap().isMinorIdentity())
      return rewriter.notifyMatchFailure(write.getLoc(), [=](Diagnostic &diag) {
        diag << "permutation map is not minor identity: " << write;
      });

    auto memRefType = dyn_cast<MemRefType>(write.getShapedType());
    if (!memRefType)
      return rewriter.notifyMatchFailure(write.getLoc(), [=](Diagnostic &diag) {
        diag << "not a memref type: " << write;
      });

    // Non-unit strides are handled by VectorToSCF.
    if (!isLastMemrefDimUnitStride(memRefType))
      return rewriter.notifyMatchFailure(write.getLoc(), [=](Diagnostic &diag) {
        diag << "most minor stride is not 1: " << write;
      });

    // `vector.store` supports vector types as memref's elements only when the
    // type of the vector value being written is the same as the element type.
    auto memrefElTy = memRefType.getElementType();
    if (isa<VectorType>(memrefElTy) && memrefElTy != write.getVectorType())
      return rewriter.notifyMatchFailure(write.getLoc(), [=](Diagnostic &diag) {
        diag << "elemental type mismatch: " << write;
      });

    // Otherwise, element types of the memref and the vector must match.
    if (!isa<VectorType>(memrefElTy) &&
        memrefElTy != write.getVectorType().getElementType())
      return rewriter.notifyMatchFailure(write.getLoc(), [=](Diagnostic &diag) {
        diag << "elemental type mismatch: " << write;
      });

    // Out-of-bounds dims are handled by MaterializeTransferMask.
    if (write.hasOutOfBoundsDim())
      return rewriter.notifyMatchFailure(write.getLoc(), [=](Diagnostic &diag) {
        diag << "out of bounds dim: " << write;
      });
    if (write.getMask()) {
      if (write.getVectorType().getRank() != 1)
        // vector.maskedstore operates on 1-D vectors.
        return rewriter.notifyMatchFailure(
            write.getLoc(), [=](Diagnostic &diag) {
              diag << "vector type is not rank 1, can't create masked store, "
                      "needs VectorToSCF: "
                   << write;
            });

      rewriter.create<vector::MaskedStoreOp>(
          write.getLoc(), write.getSource(), write.getIndices(),
          write.getMask(), write.getVector());
    } else {
      rewriter.create<vector::StoreOp>(write.getLoc(), write.getVector(),
                                       write.getSource(), write.getIndices());
    }
    // There's no return value for StoreOps. Use Value() to signal success to
    // matchAndRewrite.
    return Value();
  }

  std::optional<unsigned> maxTransferRank;
};
} // namespace

void mlir::vector::populateVectorTransferLoweringPatterns(
    RewritePatternSet &patterns, std::optional<unsigned> maxTransferRank,
    PatternBenefit benefit) {
  patterns.add<TransferReadToVectorLoadLowering,
               TransferWriteToVectorStoreLowering>(patterns.getContext(),
                                                   maxTransferRank, benefit);
  patterns
      .add<VectorLoadToMemrefLoadLowering, VectorStoreToMemrefStoreLowering>(
          patterns.getContext(), benefit);
}
