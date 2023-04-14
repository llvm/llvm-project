//===- ViewLikeInterfaceUtils.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/ViewLikeInterfaceUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;

LogicalResult mlir::mergeOffsetsSizesAndStrides(
    OpBuilder &builder, Location loc, ArrayRef<OpFoldResult> producerOffsets,
    ArrayRef<OpFoldResult> producerSizes,
    ArrayRef<OpFoldResult> producerStrides,
    const llvm::SmallBitVector &droppedProducerDims,
    ArrayRef<OpFoldResult> consumerOffsets,
    ArrayRef<OpFoldResult> consumerSizes,
    ArrayRef<OpFoldResult> consumerStrides,
    SmallVector<OpFoldResult> &combinedOffsets,
    SmallVector<OpFoldResult> &combinedSizes,
    SmallVector<OpFoldResult> &combinedStrides) {
  combinedOffsets.resize(producerOffsets.size());
  combinedSizes.resize(producerOffsets.size());
  combinedStrides.resize(producerOffsets.size());

  AffineExpr s0, s1, s2;
  bindSymbols(builder.getContext(), s0, s1, s2);

  unsigned consumerPos = 0;
  for (auto i : llvm::seq<unsigned>(0, producerOffsets.size())) {
    if (droppedProducerDims.test(i)) {
      // For dropped dims, get the values from the producer.
      combinedOffsets[i] = producerOffsets[i];
      combinedSizes[i] = producerSizes[i];
      combinedStrides[i] = producerStrides[i];
      continue;
    }
    SmallVector<OpFoldResult> offsetSymbols, strideSymbols;
    // The combined offset is computed as
    //    producer_offset + consumer_offset * producer_strides.
    combinedOffsets[i] = makeComposedFoldedAffineApply(
        builder, loc, s0 * s1 + s2,
        {consumerOffsets[consumerPos], producerStrides[i], producerOffsets[i]});
    combinedSizes[i] = consumerSizes[consumerPos];
    // The combined stride is computed as
    //    consumer_stride * producer_stride.
    combinedStrides[i] = makeComposedFoldedAffineApply(
        builder, loc, s0 * s1,
        {consumerStrides[consumerPos], producerStrides[i]});

    consumerPos++;
  }
  return success();
}

LogicalResult mlir::mergeOffsetsSizesAndStrides(
    OpBuilder &builder, Location loc, OffsetSizeAndStrideOpInterface producer,
    OffsetSizeAndStrideOpInterface consumer,
    const llvm::SmallBitVector &droppedProducerDims,
    SmallVector<OpFoldResult> &combinedOffsets,
    SmallVector<OpFoldResult> &combinedSizes,
    SmallVector<OpFoldResult> &combinedStrides) {
  SmallVector<OpFoldResult> consumerOffsets = consumer.getMixedOffsets();
  SmallVector<OpFoldResult> consumerSizes = consumer.getMixedSizes();
  SmallVector<OpFoldResult> consumerStrides = consumer.getMixedStrides();
  SmallVector<OpFoldResult> producerOffsets = producer.getMixedOffsets();
  SmallVector<OpFoldResult> producerSizes = producer.getMixedSizes();
  SmallVector<OpFoldResult> producerStrides = producer.getMixedStrides();
  return mergeOffsetsSizesAndStrides(
      builder, loc, producerOffsets, producerSizes, producerStrides,
      droppedProducerDims, consumerOffsets, consumerSizes, consumerStrides,
      combinedOffsets, combinedSizes, combinedStrides);
}

void mlir::resolveIndicesIntoOpWithOffsetsAndStrides(
    RewriterBase &rewriter, Location loc,
    ArrayRef<OpFoldResult> mixedSourceOffsets,
    ArrayRef<OpFoldResult> mixedSourceStrides,
    const llvm::SmallBitVector &rankReducedDims,
    ArrayRef<OpFoldResult> consumerIndices,
    SmallVectorImpl<Value> &resolvedIndices) {
  OpFoldResult zero = rewriter.getIndexAttr(0);

  // For each dimension that is rank-reduced, add a zero to the indices.
  int64_t indicesDim = 0;
  SmallVector<OpFoldResult> indices;
  for (auto dim : llvm::seq<int64_t>(0, mixedSourceOffsets.size())) {
    OpFoldResult ofr =
        (rankReducedDims.test(dim)) ? zero : consumerIndices[indicesDim++];
    indices.push_back(ofr);
  }

  resolvedIndices.resize(indices.size());
  resolvedIndices.clear();
  for (auto [offset, index, stride] :
       llvm::zip_equal(mixedSourceOffsets, indices, mixedSourceStrides)) {
    AffineExpr off, idx, str;
    bindSymbols(rewriter.getContext(), off, idx, str);
    OpFoldResult ofr = makeComposedFoldedAffineApply(
        rewriter, loc, AffineMap::get(0, 3, off + idx * str),
        {offset, index, stride});
    resolvedIndices.push_back(
        getValueOrCreateConstantIndexOp(rewriter, loc, ofr));
  }
}

void mlir::resolveSizesIntoOpWithSizes(
    ArrayRef<OpFoldResult> sourceSizes, ArrayRef<OpFoldResult> destSizes,
    const llvm::SmallBitVector &rankReducedSourceDims,
    SmallVectorImpl<OpFoldResult> &resolvedSizes) {
  int64_t dim = 0;
  int64_t srcRank = sourceSizes.size();
  for (int64_t srcDim = 0; srcDim < srcRank; ++srcDim) {
    if (rankReducedSourceDims[srcDim]) {
      resolvedSizes.push_back(sourceSizes[srcDim]);
      continue;
    }
    resolvedSizes.push_back(destSizes[dim++]);
  }
}
