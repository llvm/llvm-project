//===- Split.cpp - Structured op splitting --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"

#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::linalg;

/// Extract the slices of `operands` supplied to the given operation `op` such
/// that they are sufficient to execute the op for the subset of its iteration
/// space defined by `splitIterationSpace`. The subset is a part of the original
/// iteration space split at the given `dimension`. If `offset` is provided, it
/// indicates the iterator value at which the dimension has been split and
/// requires the "high" part starting at the given offset of the operands to be
/// generated; otherwise, the "low" part with no offset is generated. Note that
/// `operands` are not necessarily the actual operands of `op`.
static SmallVector<Value>
getOperandSlices(RewriterBase &b, Location loc, LinalgOp op,
                 ValueRange splitIterationSpace, ValueRange operands,
                 unsigned dimension, Value offset = nullptr) {
  SmallVector<Value> slices;
  slices.reserve(op.getNumInputsAndOutputs());
  for (OpOperand *opOperand : op.getInputAndOutputOperands()) {
    auto type = opOperand->get().getType().dyn_cast<ShapedType>();
    AffineMap indexing = op.getTiedIndexingMap(opOperand);

    // If the type is not sliceable, or the slice is requested along the
    // dimension that is not used in indexing this type, just use the entire
    // operand.
    if (!type || dimension >= indexing.getNumDims() ||
        !indexing.isFunctionOfDim(dimension)) {
      slices.push_back(opOperand->get());
      continue;
    }

    SmallVector<OpFoldResult> sizes;
    sizes.reserve(indexing.getNumResults());
    for (AffineExpr dimIndexing : indexing.getResults()) {
      sizes.push_back(makeComposedFoldedAffineApply(
          b, loc, dimIndexing,
          getAsOpFoldResult(llvm::to_vector(splitIterationSpace))));
    }
    SmallVector<OpFoldResult> offsets(type.getRank(), b.getIndexAttr(0));
    SmallVector<OpFoldResult> strides(type.getRank(), b.getIndexAttr(1));

    if (offset) {
      offsets[dimension] = offset;
      offsets = applyMapToValues(b, loc, indexing, offsets);
    }

    slices.push_back(createSlice(b, loc,
                                 operands[opOperand->getOperandNumber()],
                                 offsets, sizes, strides));
  }

  return slices;
}

/// Creates a part of the given `op` split along the iteration space `dimension`
/// with the given `size` and an optional `offset` (default 0). Makes slices
/// of operands, using the input operands of the original op and the output
/// operands provided as `resultOperands`. Expects `splitIterationSpace` to be
/// a list of values representing the shape of the iteration space of the
/// original op and updates it to be the iteration space of the curent part.
/// Returns the split-out op as well as the output operand values updated with
/// the partial results produced by this op through `results`.
static LinalgOp
createSplitPart(RewriterBase &b, Location loc, LinalgOp op,
                ValueRange resultOperands,
                llvm::MutableArrayRef<Value> splitIterationSpace,
                unsigned dimension, OpFoldResult size,
                SmallVectorImpl<Value> &results, Value offset = nullptr) {
  ImplicitLocOpBuilder implicit(op.getLoc(), b);
  splitIterationSpace[dimension] = materializeOpFoldResult(implicit, size);
  SmallVector<Value> operands = llvm::to_vector(
      llvm::map_range(op.getInputOperands(),
                      [](OpOperand *opOperand) { return opOperand->get(); }));
  llvm::append_range(operands, resultOperands);
  operands = getOperandSlices(b, loc, op, splitIterationSpace, operands,
                              dimension, offset);
  Operation *part =
      op.clone(b, loc, getTensorOutputTypes(op, operands), operands);
  results = insertSlicesBack(b, loc, op, operands, part->getResults());
  return cast<LinalgOp>(part);
}

std::pair<LinalgOp, LinalgOp> linalg::splitOp(RewriterBase &rewriter,
                                              LinalgOp op, unsigned dimension,
                                              OpFoldResult splitPoint) {
  // Bail out on dimension overflow.
  if (dimension >= op.getNumLoops())
    return std::make_pair(op, LinalgOp());

  // Compute the iteration space size as values.
  SmallVector<Value, 4> allShapes =
      op.createFlatListOfOperandDims(rewriter, op.getLoc());
  AffineMap shapesToLoops = op.getShapesToLoopsMap();
  SmallVector<Value, 4> iterationSpaceShapes =
      applyMapToValues(rewriter, op.getLoc(), shapesToLoops, allShapes);

  // Update the iteration space to have `splitPoint` as the size of `dimension`
  // and use it to slice operands and results for a new, smaller instance of the
  // `op`. Adjust the size if necessary to prevent overflows. Insert the partial
  // results back.
  OpFoldResult dimSize = getAsOpFoldResult(iterationSpaceShapes[dimension]);
  OpFoldResult minSplitPoint = makeComposedFoldedAffineMin(
      rewriter, op->getLoc(),
      AffineMap::getMultiDimIdentityMap(/*numDims=*/2, rewriter.getContext()),
      {splitPoint, dimSize});
  SmallVector<Value> splitIterationSpace =
      llvm::to_vector(iterationSpaceShapes);
  SmallVector<Value> originalResults = llvm::to_vector(
      llvm::map_range(op.getOutputOperands(),
                      [](OpOperand *opOperand) { return opOperand->get(); }));
  SmallVector<Value> firstResults;
  LinalgOp first = createSplitPart(rewriter, op.getLoc(), op, originalResults,
                                   splitIterationSpace, dimension,
                                   minSplitPoint, firstResults);

  // Update the iteration space to cover the remaining part of the original
  // space, then create another instance of the `op` in that space. The size of
  // the remaining part may become zero, but is never negative because of the
  // adjustment above.
  AffineExpr d0 = rewriter.getAffineDimExpr(0);
  AffineExpr d1 = rewriter.getAffineDimExpr(1);
  OpFoldResult remainingSize = makeComposedFoldedAffineApply(
      rewriter, op.getLoc(), d0 - d1, {dimSize, minSplitPoint});
  SmallVector<Value> secondResults;
  ImplicitLocOpBuilder implicit(op.getLoc(), rewriter);
  Value splitPointValue = materializeOpFoldResult(implicit, minSplitPoint);
  LinalgOp second = createSplitPart(
      rewriter, op.getLoc(), op, firstResults, splitIterationSpace, dimension,
      remainingSize, secondResults, splitPointValue);

  // Fixup the linalg.index results in the second part.
  SmallVector<Value> ivAdditions;
  ivAdditions.resize(splitIterationSpace.size());
  ivAdditions[dimension] = splitPointValue;
  linalg::offsetIndices(rewriter, cast<LinalgOp>(second), ivAdditions);

  // Replace the original op with the results of the two newly created ops.
  rewriter.replaceOp(op, secondResults);
  return std::make_pair(first, second);
}
