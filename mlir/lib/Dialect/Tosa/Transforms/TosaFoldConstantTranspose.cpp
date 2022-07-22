//===- TosaFoldConstantTranspose.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Fold TOSA Transpose operation on constant data
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::tosa;

namespace {

struct TosaFoldConstantTranspose : public OpRewritePattern<tosa::TransposeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    auto outputType = op.getType().cast<ShapedType>();
    // TOSA supports quantized types.
    if (!outputType.getElementType().isIntOrIndexOrFloat())
      return failure();

    DenseElementsAttr inputValues;
    if (!matchPattern(op.getInput1(), m_Constant(&inputValues)))
      return failure();
    // Make sure the input is a constant that has a single user.
    if (!llvm::hasSingleElement(op.getInput1().getDefiningOp()->getUsers()))
      return failure();

    DenseIntElementsAttr permAttr;
    if (!matchPattern(op.getPerms(), m_Constant(&permAttr)))
      return failure();
    auto permValues = llvm::to_vector<6>(llvm::map_range(
        // TOSA allows both 32- and 64-bit integer tensors here.
        permAttr.getValues<APInt>(),
        [](const APInt &val) { return val.getZExtValue(); }));

    auto inputType = op.getInput1().getType().cast<ShapedType>();
    ArrayRef<int64_t> inputShape = inputType.getShape();
    int64_t numElements = inputType.getNumElements();

    SmallVector<Attribute, 4> outputValues;
    outputValues.resize(numElements);

    // Transpose the input constant. Because we don't know its rank in advance,
    // we need to loop over the range [0, element count) and delinearize the
    // index.
    auto attrValues = inputValues.getValues<Attribute>();
    ArrayRef<int64_t> outputShape = outputType.getShape();
    for (int srcLinearIndex = 0; srcLinearIndex < numElements;
         ++srcLinearIndex) {
      SmallVector<uint64_t, 6> srcIndices(inputType.getRank(), 0);
      int totalCount = srcLinearIndex;
      for (int dim = inputType.getRank() - 1; dim >= 0; --dim) {
        srcIndices[dim] = totalCount % inputShape[dim];
        totalCount /= inputShape[dim];
      }

      SmallVector<uint64_t, 6> dstIndices(outputType.getRank(), 0);
      for (int dim = outputType.getRank() - 1; dim >= 0; --dim)
        dstIndices[dim] = srcIndices[permValues[dim]];

      uint64_t dstLinearIndex = dstIndices.front();
      for (int dim = 1; dim < outputType.getRank(); ++dim)
        dstLinearIndex = dstLinearIndex * outputShape[dim] + dstIndices[dim];

      outputValues[dstLinearIndex] = attrValues[srcIndices];
    }

    rewriter.replaceOpWithNewOp<tosa::ConstOp>(
        op, outputType, DenseElementsAttr::get(outputType, outputValues));
    return success();
  }
};

} // namespace

void mlir::tosa::populateTosaFoldConstantTransposePatterns(
    MLIRContext *ctx, RewritePatternSet &patterns) {
  patterns.add<TosaFoldConstantTranspose>(ctx);
}
