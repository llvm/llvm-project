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
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::tosa;

namespace {

template <typename BaseType>
DenseElementsAttr transposeType(ElementsAttr attr, ShapedType inputType,
                                ShapedType outputType,
                                llvm::ArrayRef<int64_t> permValues) {
  if (inputType.getNumElements() == 0)
    return DenseElementsAttr::get(outputType, llvm::ArrayRef<BaseType>{});

  auto attrValues = attr.getValues<BaseType>();
  auto inputShape = inputType.getShape();

  // The inverted permutation map and strides of the output are used to compute
  // the contribution of a given dimension to the destination linear index in
  // an order-independent way.
  auto outputStrides = computeStrides(outputType.getShape());
  auto invertedPermValues = invertPermutationVector(permValues);

  auto initialValue = *std::begin(attrValues);
  SmallVector<BaseType> outputValues(inputType.getNumElements(), initialValue);

  for (const auto &it : llvm::enumerate(attrValues)) {
    auto srcLinearIndex = it.index();

    uint64_t dstLinearIndex = 0;
    for (int64_t dim = inputShape.size() - 1; dim >= 0; --dim) {
      // Compute the index into the current dimension of the source vector.
      auto sourceIndexForDim = srcLinearIndex % inputShape[dim];
      srcLinearIndex /= inputShape[dim];

      // Add the contribution of the current dimension to the output using the
      // permutation map.
      dstLinearIndex +=
          outputStrides[invertedPermValues[dim]] * sourceIndexForDim;
    }

    outputValues[dstLinearIndex] = it.value();
  }

  return DenseElementsAttr::get(outputType,
                                llvm::ArrayRef<BaseType>(outputValues));
}

// A type specialized transposition of an ElementsAttr.
// This implementation tries to operate on the underlying data in its raw
// representation when possible to avoid allocating a large number of Attribute
// objects.
DenseElementsAttr transpose(ElementsAttr attr, ShapedType inputType,
                            ShapedType outputType,
                            llvm::ArrayRef<int64_t> permValues) {
  auto baseType = inputType.getElementType();

  // Handle possible integer types
  if (auto intType = dyn_cast<IntegerType>(baseType)) {
    switch (intType.getWidth()) {
    case 1:
      return transposeType<bool>(attr, inputType, outputType, permValues);
    case 8:
      return transposeType<int8_t>(attr, inputType, outputType, permValues);
    case 16:
      return transposeType<int16_t>(attr, inputType, outputType, permValues);
    case 32:
      return transposeType<int32_t>(attr, inputType, outputType, permValues);
    case 64:
      return transposeType<int64_t>(attr, inputType, outputType, permValues);
    default:
      return transposeType<APInt>(attr, inputType, outputType, permValues);
    }
  }

  // Handle possible float types
  if (baseType.isF32()) {
    return transposeType<float>(attr, inputType, outputType, permValues);
  }

  return transposeType<APFloat>(attr, inputType, outputType, permValues);
}

struct TosaFoldConstantTranspose : public OpRewritePattern<tosa::TransposeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    auto outputType = cast<ShapedType>(op.getType());
    // TOSA supports quantized types.
    if (!outputType.getElementType().isIntOrIndexOrFloat())
      return failure();

    ElementsAttr inputValues;
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
        [](const APInt &val) { return val.getSExtValue(); }));

    auto inputType = cast<ShapedType>(op.getInput1().getType());

    auto resultAttr = transpose(inputValues, inputType, outputType, permValues);
    rewriter.replaceOpWithNewOp<tosa::ConstOp>(op, outputType, resultAttr);
    return success();
  }
};

} // namespace

void mlir::tosa::populateTosaFoldConstantTransposePatterns(
    MLIRContext *ctx, RewritePatternSet &patterns) {
  patterns.add<TosaFoldConstantTranspose>(ctx);
}
