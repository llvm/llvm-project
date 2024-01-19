//===- Split.cpp - Structured op splitting --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/TilingInterface.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::linalg;

/// Creates a part of the given `op` split along the iteration space `dimension`
/// with the given `size` and an optional `offset` (default 0). Makes slices
/// of operands, using the input operands of the original op and the output
/// operands provided as `resultOperands`. Expects `offsets` and `sizes` to
/// define the shape of the iteration space of the original op. Returns the
/// split-out op as well as the output operand values updated with the partial
/// results produced by this op through `results`.
static TilingInterface
createSplitPart(RewriterBase &b, Location loc, TilingInterface op,
                ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,
                ValueRange resultOperands, unsigned dimension,
                OpFoldResult size, OpFoldResult offset,
                SmallVectorImpl<Value> &results) {
  // Iteration space of the current part.
  SmallVector<OpFoldResult> sizesCopy = llvm::to_vector(sizes);
  SmallVector<OpFoldResult> offsetsCopy = llvm::to_vector(offsets);
  sizesCopy[dimension] = size;
  offsetsCopy[dimension] = offset;

  // Create the part as it it were a single tile.
  FailureOr<TilingResult> tilingResult =
      op.getTiledImplementation(b, offsetsCopy, sizesCopy);

  // Insert the results back and populate the `results` list.
  for (auto [index, result] : llvm::enumerate(tilingResult->tiledValues)) {
    SmallVector<OpFoldResult> resultOffsets, resultSizes;
    if (failed(op.getResultTilePosition(b, index, offsetsCopy, sizesCopy,
                                        resultOffsets, resultSizes)))
      return nullptr;
    SmallVector<OpFoldResult> resultStrides(resultOffsets.size(),
                                            b.getIndexAttr(1));
    Value inserted = b.create<tensor::InsertSliceOp>(
        loc, result, resultOperands[index], resultOffsets, resultSizes,
        resultStrides);
    results.push_back(inserted);
  }
  // TODO: this part can be generalized maybe to not expect a single op.
  assert(tilingResult->tiledOps.size() == 1 &&
         "expected split part to return a single tiled operation");
  return cast<TilingInterface>(tilingResult->tiledOps[0]);
}

std::pair<TilingInterface, TilingInterface>
linalg::splitOp(RewriterBase &rewriter, TilingInterface op, unsigned dimension,
                OpFoldResult splitPoint) {
  // Compute the iteration space.
  SmallVector<Range> iterationSpace = op.getIterationDomain(rewriter);

  // Bail out on dimension overflow.
  if (dimension >= iterationSpace.size())
    return std::make_pair(op, TilingInterface());

  SmallVector<OpFoldResult> offsets = llvm::to_vector(llvm::map_range(
      iterationSpace, [](const Range &range) { return range.offset; }));
  SmallVector<OpFoldResult> sizes = llvm::to_vector(llvm::map_range(
      iterationSpace, [](const Range &range) { return range.size; }));

  // Adjust the split point so that it doesn't overflow the size.
  AffineExpr d0, d1, d2;
  bindDims(rewriter.getContext(), d0, d1, d2);
  OpFoldResult minSplitPoint = affine::makeComposedFoldedAffineMin(
      rewriter, op.getLoc(),
      AffineMap::inferFromExprList(ArrayRef<AffineExpr>{d0, d1 + d2}).front(),
      {splitPoint, offsets[dimension], sizes[dimension]});

  // Compute the size of the second part. Return early if the second part would
  // have an empty iteration space.
  OpFoldResult remainingSize = affine::makeComposedFoldedAffineApply(
      rewriter, op.getLoc(), d0 + d1 - d2,
      {iterationSpace[dimension].offset, iterationSpace[dimension].size,
       minSplitPoint});
  if (auto attr = llvm::dyn_cast_if_present<Attribute>(remainingSize)) {
    if (cast<IntegerAttr>(attr).getValue().isZero())
      return {op, TilingInterface()};
  }

  // Compute destination tensors.
  SmallVector<Value> destinationTensors;
  LogicalResult destStatus = tensor::getOrCreateDestinations(
      rewriter, op.getLoc(), op, destinationTensors);
  (void)destStatus;
  assert(succeeded(destStatus) && "failed to get destination tensors");

  // Create the first part.
  SmallVector<Value> firstResults;
  TilingInterface firstPart = createSplitPart(
      rewriter, op.getLoc(), op, offsets, sizes, destinationTensors, dimension,
      minSplitPoint, iterationSpace[dimension].offset, firstResults);

  // Need to pretend that the original op now takes as operands firstResults,
  // otherwise tiling interface implementation will take the wrong value to
  // produce data tiles.
  rewriter.modifyOpInPlace(op, [&]() {
    unsigned numTotalOperands = op->getNumOperands();
    unsigned numOutputOperands = firstResults.size();
    op->setOperands(numTotalOperands - numOutputOperands, numOutputOperands,
                    firstResults);
  });

  // Create the second part.
  OpFoldResult totalOffset = affine::makeComposedFoldedAffineApply(
      rewriter, op.getLoc(), d0 + d1, {offsets[dimension], minSplitPoint});
  SmallVector<Value> secondResults;
  TilingInterface secondPart =
      createSplitPart(rewriter, op.getLoc(), op, offsets, sizes, firstResults,
                      dimension, remainingSize, totalOffset, secondResults);

  // Propagate any errors in part creation.
  if (!firstPart || !secondPart)
    return {TilingInterface(), TilingInterface()};

  // Replace the original op with the results of the two newly created ops.
  rewriter.replaceOp(op, secondResults);
  return {firstPart, secondPart};
}
