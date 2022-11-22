//===- TosaDecomposeConv2D.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Decompose TOSA Conv2D operation to a series of TOSA Ops specifically
// (1) Convert a 1x1 Convolution to a Reshape->FC->Reshape
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::tosa;

namespace {

SmallVector<int64_t> convertFromMlirShape(ArrayRef<int64_t> shape) {
  return to_vector(llvm::map_range(shape, [](int64_t dim) {
    return ShapedType::isDynamic(dim) ? -1 : dim;
  }));
}

struct Conv2DIsFullyConnected : public OpRewritePattern<tosa::Conv2DOp> {
  explicit Conv2DIsFullyConnected(MLIRContext *context)
      : OpRewritePattern(context) {}

  LogicalResult matchAndRewrite(tosa::Conv2DOp op,
                                PatternRewriter &rewriter) const override {
    Value input = op.getInput();
    Value weight = op.getWeight();
    ShapedType inputType = input.getType().cast<ShapedType>();
    ShapedType weightType = weight.getType().cast<ShapedType>();
    ShapedType resultType = op.getType().cast<ShapedType>();

    auto numDynamic =
        llvm::count_if(inputType.getShape(), ShapedType::isDynamic);
    if (numDynamic > 1)
      return rewriter.notifyMatchFailure(
          op, "at most one dim in input may be dynamic");
    if (!weightType.hasRank())
      return rewriter.notifyMatchFailure(op, "unranked weight input");

    // Stride must be 1 for this optimization.
    for (APInt stride : op.getStride().getAsValueRange<IntegerAttr>()) {
      if (!stride.isOne())
        return failure();
    }

    // Only works for a 1x1 kernel.
    ArrayRef<int64_t> weightShape = weightType.getShape();
    if (weightShape[1] != 1 || weightShape[2] != 1)
      return failure();

    // Reshape input to [N,IH,IW,IC] -> [N * IH * IW, IC].
    ArrayRef<int64_t> inputShape = inputType.getShape();
    int64_t combined = ShapedType::kDynamicSize;
    if (numDynamic == 0)
      combined = inputShape[0] * inputShape[1] * inputShape[2];
    llvm::SmallVector<int64_t, 2> revisedInputShape{combined, inputShape[3]};
    auto revisedInputShapeType =
        RankedTensorType::get(revisedInputShape, inputType.getElementType());
    auto reshapedInput = rewriter
                             .create<tosa::ReshapeOp>(
                                 op.getLoc(), revisedInputShapeType, input,
                                 rewriter.getI64ArrayAttr(
                                     convertFromMlirShape(revisedInputShape)))
                             .getResult();

    // Reshape kernel to [OC,KH,KW,IC] -> [OC, IC].
    llvm::SmallVector<int64_t, 2> revisedWeightShape{weightShape[0],
                                                     weightShape[3]};
    auto revisedWeightShapeType = RankedTensorType::get(
        revisedWeightShape,
        weight.getType().dyn_cast<RankedTensorType>().getElementType());
    auto reshapedWeight = rewriter
                              .create<tosa::ReshapeOp>(
                                  op.getLoc(), revisedWeightShapeType, weight,
                                  rewriter.getI64ArrayAttr(
                                      convertFromMlirShape(revisedWeightShape)))
                              .getResult();

    // Perform a fully connected network over the reshaped input and weight.
    llvm::SmallVector<int64_t, 2> fullyConnectedShape{combined, weightShape[0]};
    auto fullyConnectedShapeType =
        RankedTensorType::get(fullyConnectedShape, resultType.getElementType());

    Value fullyConnectedValue;
    if (op.getQuantizationInfo()) {
      fullyConnectedValue =
          rewriter
              .create<tosa::FullyConnectedOp>(
                  op.getLoc(), fullyConnectedShapeType, reshapedInput,
                  reshapedWeight, op.getBias(), *op.getQuantizationInfo())
              .getResult();
    } else {
      fullyConnectedValue = rewriter
                                .create<tosa::FullyConnectedOp>(
                                    op.getLoc(), fullyConnectedShapeType,
                                    reshapedInput, reshapedWeight, op.getBias())
                                .getResult();
    }

    // Reshape output to [N, IH, IW, OC].
    llvm::SmallVector<int64_t, 4> outputShape{inputShape[0], inputShape[1],
                                              inputShape[2], weightShape[0]};
    rewriter.replaceOpWithNewOp<tosa::ReshapeOp>(
        op, resultType, fullyConnectedValue,
        rewriter.getI64ArrayAttr(convertFromMlirShape(outputShape)));
    return success();
  }
};

} // namespace

void mlir::tosa::populateTosaDecomposeConv2D(MLIRContext *ctx,
                                             RewritePatternSet &patterns) {
  patterns.add<Conv2DIsFullyConnected>(ctx);
}
