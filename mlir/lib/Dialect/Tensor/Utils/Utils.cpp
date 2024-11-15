//===- Utils.cpp - Utilities to support the Tensor dialect ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements utilities for the Tensor dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tensor/Utils/Utils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"

using namespace mlir;
using namespace mlir::tensor;

PadOp mlir::tensor::createPadHighOp(RankedTensorType resType, Value source,
                                    Value pad, bool nofold, Location loc,
                                    OpBuilder &b,
                                    SmallVector<Value> dynOutDims) {

  assert((resType.getNumDynamicDims() == dynOutDims.size()) ||
         dynOutDims.empty() &&
             "Either none or all output dynamic dims must be specified!");

  // Init "low" and "high" padding values ("low" is kept as is, "high" is
  // computed below).
  SmallVector<OpFoldResult> low(resType.getRank(), b.getIndexAttr(0));
  SmallVector<OpFoldResult> high(resType.getRank(), b.getIndexAttr(0));

  size_t outDimIdx = 0;

  for (const auto [idx, val] : enumerate(resType.getShape())) {
    bool isDimDynamic = ShapedType::isDynamic(val);
    bool updatePadHigh = !isDimDynamic || !dynOutDims.empty();

    // Keep the default padding width (i.e. "0") when the output dim is dynamic
    // and no actual output sizes have been provided.
    if (!updatePadHigh)
      continue;

    // Compute the padding width: resDim - sourceDim.
    AffineExpr d0, d1;
    bindDims(b.getContext(), d0, d1);
    OpFoldResult sourceDim = tensor::getMixedSize(b, loc, source, idx);
    OpFoldResult outDim = isDimDynamic ? OpFoldResult(dynOutDims[outDimIdx++])
                                       : OpFoldResult(b.getIndexAttr(val));

    high[idx] = affine::makeComposedFoldedAffineApply(b, loc, d0 - d1,
                                                      {outDim, sourceDim});
  }
  return b.create<PadOp>(loc, resType, source, low, high, pad, nofold);
}

SmallVector<Value> mlir::tensor::createDynamicDimValues(OpBuilder &b,
                                                        Location loc,
                                                        Value rankedTensor) {
  auto tensorTy = cast<RankedTensorType>(rankedTensor.getType());
  SmallVector<Value> dynamicDims;
  for (const auto &en : llvm::enumerate(tensorTy.getShape())) {
    if (en.value() == ShapedType::kDynamic)
      dynamicDims.push_back(
          b.create<tensor::DimOp>(loc, rankedTensor, en.index()));
  }
  return dynamicDims;
}

FailureOr<RankedTensorType>
mlir::tensor::computeTransposedType(RankedTensorType rankedTensorType,
                                    ArrayRef<int64_t> transposeVector) {
  if (transposeVector.empty())
    return rankedTensorType;

  if (!isPermutationVector(transposeVector) ||
      transposeVector.size() != static_cast<size_t>(rankedTensorType.getRank()))
    return failure();

  SmallVector<int64_t> transposedShape(rankedTensorType.getShape());
  applyPermutationToVector(transposedShape, transposeVector);

  using RTTBuilder = RankedTensorType::Builder;
  RankedTensorType transposedTensorType =
      RTTBuilder(rankedTensorType).setShape(transposedShape);
  return transposedTensorType;
}

/// The permutation can be obtained from two permutations:
///   a) Compute the permutation vector to move the last `numPackedDims` into
///      the `innerPosDims` of a shape of rank `rank`.
///   b) Compute the permutation vector to move outer dims if the
///      `outerPerm` parameter is not empty.
/// Apply (b) permutation on (a) permutation to get the final permutation.
static SmallVector<int64_t>
computePackUnPackPerm(int64_t rank, ArrayRef<int64_t> &innerDimsPos,
                      ArrayRef<int64_t> &outerPerm,
                      PackingMetadata &packingMetadata) {
  int64_t numPackedDims = innerDimsPos.size();
  auto lastDims =
      llvm::to_vector(llvm::seq<int64_t>(rank - numPackedDims, rank));
  packingMetadata = computePackingMetadata(rank, innerDimsPos);
  SmallVector<int64_t> innerPositionsPerm =
      computePermutationVector(rank, lastDims, packingMetadata.insertPositions);

  SmallVector<int64_t> outerPos = packingMetadata.outerPositions;
  if (!outerPerm.empty())
    applyPermutationToVector(outerPos, outerPerm);
  SmallVector<int64_t> outerPositionPerm =
      computePermutationVector(rank, packingMetadata.outerPositions, outerPos);

  SmallVector<int64_t> packInverseDestPermutation = innerPositionsPerm;
  applyPermutationToVector(packInverseDestPermutation, outerPositionPerm);
  return packInverseDestPermutation;
}

SmallVector<int64_t> mlir::tensor::getPackInverseDestPerm(PackOp packOp) {

  PackingMetadata pMetadata;
  int64_t packedRank = packOp.getDestType().getRank();
  ArrayRef<int64_t> innerDimPos = packOp.getInnerDimsPos();
  ArrayRef<int64_t> outerPerm = packOp.getOuterDimsPerm();
  SmallVector<int64_t> packInvDestPerm =
      computePackUnPackPerm(packedRank, innerDimPos, outerPerm, pMetadata);
  return packInvDestPerm;
}

SmallVector<int64_t> mlir::tensor::getUnPackInverseSrcPerm(UnPackOp unpackOp) {
  PackingMetadata metadata;
  return mlir::tensor::getUnPackInverseSrcPerm(unpackOp, metadata);
}

SmallVector<int64_t>
mlir::tensor::getUnPackInverseSrcPerm(UnPackOp unpackOp,
                                      PackingMetadata &metadata) {
  int64_t unpackRank = unpackOp.getSourceType().getRank();
  ArrayRef<int64_t> innerDimPos = unpackOp.getInnerDimsPos();
  ArrayRef<int64_t> outerPerm = unpackOp.getOuterDimsPerm();
  SmallVector<int64_t> unpackInvSrcPerm =
      computePackUnPackPerm(unpackRank, innerDimPos, outerPerm, metadata);
  return unpackInvSrcPerm;
}

bool mlir::tensor::isCastLikeInsertSliceOp(InsertSliceOp op) {
  llvm::SmallBitVector droppedDims = op.getDroppedDims();
  int64_t srcDim = 0;
  RankedTensorType resultType = op.getDestType();
  // Source dims and destination dims (apart from dropped dims) must have the
  // same size.
  for (int64_t resultDim = 0; resultDim < resultType.getRank(); ++resultDim) {
    if (droppedDims.test(resultDim)) {
      // InsertSlice may expand unit dimensions that result from inserting a
      // size-1 slice into a non-size-1 result dimension.
      if (resultType.getDimSize(resultDim) != 1)
        return false;
      continue;
    }
    FailureOr<bool> equalDimSize = ValueBoundsConstraintSet::areEqual(
        {op.getSource(), srcDim}, {op.getResult(), resultDim});
    if (failed(equalDimSize) || !*equalDimSize)
      return false;
    ++srcDim;
  }

  return true;
}

bool mlir::tensor::isCastLikeExtractSliceOp(ExtractSliceOp op) {
  llvm::SmallBitVector droppedDims = op.getDroppedDims();
  int64_t resultDim = 0;
  // Source dims and result dims (apart from dropped dims) must have the same
  // size.
  RankedTensorType sourceType = op.getSourceType();
  for (int64_t dim = 0, e = sourceType.getRank(); dim < e; ++dim) {
    if (droppedDims.test(dim)) {
      // ExtractSlice may drop unit dimensions that result from taking a size-1
      // slice from a non-size-1 source dimension.
      if (sourceType.getDimSize(dim) != 1)
        return false;
      continue;
    }
    FailureOr<bool> equalDimSize = ValueBoundsConstraintSet::areEqual(
        {op.getSource(), dim}, {op.getResult(), resultDim});
    if (failed(equalDimSize) || !*equalDimSize)
      return false;
    ++resultDim;
  }

  return true;
}
