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

#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::linalg;

/// Turns an OpFoldResult into a value, creating an index-typed constant if
/// necessary.
static Value materializeOpFoldResult(ImplicitLocOpBuilder &builder,
                                     OpFoldResult opFoldResult) {
  if (opFoldResult.is<Value>())
    return opFoldResult.get<Value>();
  auto attr = opFoldResult.get<Attribute>().cast<IntegerAttr>();
  return builder.create<arith::ConstantIndexOp>(attr.getValue().getSExtValue());
}

/// Extract the slices of `operands` supplied to the given operation `op` such
/// that they are sufficient to execute the op for the subset of its iteration
/// space defined by `splitIterationSpace`. The subset is a part of the original
/// iteration space split at the given `dimension`. If `offset` is provided, it
/// indicates the iterator value at which the dimension has been split and
/// requires the "high" part starting at the given offset of the operands to be
/// generated; otherwise, the "low" part with no offset is generated. Note that
/// `operands` are not necessarily the actual operands of `op`.
static SmallVector<Value>
getOperandSlices(ImplicitLocOpBuilder &builder, LinalgOp op,
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

    SmallVector<Value, 4> sizes =
        applyMapToValues(builder, op.getLoc(), indexing, splitIterationSpace);
    SmallVector<OpFoldResult> offsets(type.getRank(), builder.getIndexAttr(0));
    SmallVector<OpFoldResult> strides(type.getRank(), builder.getIndexAttr(1));

    if (offset) {
      offsets[dimension] = offset;
      IRRewriter rewriter(builder);
      offsets = applyMapToValues(rewriter, builder.getLoc(), indexing, offsets);
    }

    slices.push_back(createSlice(builder, op.getLoc(),
                                 operands[opOperand->getOperandNumber()],
                                 offsets, getAsOpFoldResult(sizes), strides));
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
static LinalgOp createSplitPart(
    ImplicitLocOpBuilder &builder, LinalgOp op, ValueRange resultOperands,
    llvm::MutableArrayRef<Value> splitIterationSpace, unsigned dimension,
    Value size, SmallVectorImpl<Value> &results, Value offset = nullptr) {
  splitIterationSpace[dimension] = size;
  SmallVector<Value> operands = llvm::to_vector(
      llvm::map_range(op.getInputOperands(),
                      [](OpOperand *opOperand) { return opOperand->get(); }));
  llvm::append_range(operands, resultOperands);
  operands = getOperandSlices(builder, op, splitIterationSpace, operands,
                              dimension, offset);
  Operation *part = op.clone(builder, op.getLoc(),
                             getTensorOutputTypes(op, operands), operands);
  results = insertSlicesBack(builder, builder.getLoc(), op, operands,
                             part->getResults());
  return cast<LinalgOp>(part);
}

std::pair<LinalgOp, LinalgOp> linalg::splitOp(RewriterBase &rewriter,
                                              LinalgOp op, unsigned dimension,
                                              OpFoldResult splitPoint) {
  // Bail out on dimension overflow.
  if (dimension >= op.getNumLoops())
    return std::make_pair(op, LinalgOp());

  // Compute the iteration space size as values.
  ImplicitLocOpBuilder builder(op.getLoc(), rewriter);
  SmallVector<Value, 4> allShapes =
      op.createFlatListOfOperandDims(builder, op.getLoc());
  AffineMap shapesToLoops = op.getShapesToLoopsMap();
  SmallVector<Value, 4> iterationSpaceShapes =
      applyMapToValues(builder, op.getLoc(), shapesToLoops, allShapes);

  // Update the iteration space to have `splitPoint` as the size of `dimension`
  // and use it to slice operands and results for a new, smaller instance of the
  // `op`. Adjust the size if necessary to prevent overflows. Insert the partial
  // results back.
  Value splitPointValue = materializeOpFoldResult(builder, splitPoint);
  splitPointValue = builder.createOrFold<AffineMinOp>(
      builder.getIndexType(),
      AffineMap::getMultiDimIdentityMap(/*numDims=*/2, builder.getContext()),
      ValueRange({splitPointValue, iterationSpaceShapes[dimension]}));
  SmallVector<Value> splitIterationSpace =
      llvm::to_vector(iterationSpaceShapes);
  SmallVector<Value> originalResults = llvm::to_vector(
      llvm::map_range(op.getOutputOperands(),
                      [](OpOperand *opOperand) { return opOperand->get(); }));
  SmallVector<Value> firstResults;
  LinalgOp first =
      createSplitPart(builder, op, originalResults, splitIterationSpace,
                      dimension, splitPointValue, firstResults);

  // Update the iteration space to cover the remaining part of the original
  // space, then create another instance of the `op` in that space. The size of
  // the remaining part may become zero, but is never negative because of the
  // adjustment above.
  AffineExpr d0 = builder.getAffineDimExpr(0);
  AffineExpr d1 = builder.getAffineDimExpr(1);
  SmallVector<Value, 4> remainingSizes = applyMapToValues(
      builder, op.getLoc(), AffineMap::inferFromExprList({d0 - d1}).front(),
      {iterationSpaceShapes[dimension], splitPointValue});
  SmallVector<Value> secondResults;
  LinalgOp second =
      createSplitPart(builder, op, firstResults, splitIterationSpace, dimension,
                      remainingSizes.front(), secondResults, splitPointValue);

  // Fixup the linalg.index results in the second part.
  SmallVector<Value> ivAdditions;
  ivAdditions.resize(splitIterationSpace.size());
  ivAdditions[dimension] = splitPointValue;
  linalg::addTileLoopIvsToIndexOpResults(builder, cast<LinalgOp>(second),
                                         ivAdditions);

  // Replace the original op with the results of the two newly created ops.
  rewriter.replaceOp(op, secondResults);
  return std::make_pair(first, second);
}
