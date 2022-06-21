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
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::tosa;

namespace {

struct Conv2DIsFullyConnected : public OpRewritePattern<tosa::Conv2DOp> {
  explicit Conv2DIsFullyConnected(MLIRContext *context)
      : OpRewritePattern(context) {}

  LogicalResult matchAndRewrite(tosa::Conv2DOp op,
                                PatternRewriter &rewriter) const override {
    Value input = op.input();
    Value weight = op.weight();
    ShapedType inputType = input.getType().cast<ShapedType>();
    ShapedType weightType = weight.getType().cast<ShapedType>();
    ShapedType resultType = op.getType().cast<ShapedType>();

    if (!inputType.hasStaticShape() || !weightType.hasRank()) {
      return failure();
    }

    // Stride must be 1 for this optimization.
    for (Attribute stride : op.stride().getValue()) {
      if (!stride.cast<IntegerAttr>().getValue().isOne()) {
        return failure();
      }
    }

    // Only works for a 1x1 kernel.
    ArrayRef<int64_t> weightShape = weightType.getShape();
    if (weightShape[1] != 1 || weightShape[2] != 1) {
      return failure();
    }

    // Reshape input to [N,IH,IW,IC] -> [N * IH * IW, IC].
    ArrayRef<int64_t> inputShape = inputType.getShape();
    llvm::SmallVector<int64_t, 2> revisedInputShape{
        inputShape[0] * inputShape[1] * inputShape[2], inputShape[3]};
    auto revisedInputShapeType = RankedTensorType::get(
        revisedInputShape,
        input.getType().dyn_cast<RankedTensorType>().getElementType());
    auto reshapedInput = rewriter
                             .create<tosa::ReshapeOp>(
                                 op.getLoc(), revisedInputShapeType, input,
                                 rewriter.getI64ArrayAttr(revisedInputShape))
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
                                  rewriter.getI64ArrayAttr(revisedWeightShape))
                              .getResult();

    // Perform a fully connected network over the reshaped input and weight.
    llvm::SmallVector<int64_t, 2> fullyConnectedShape{
        inputShape[0] * inputShape[1] * inputShape[2], weightShape[0]};
    auto fullyConnectedShapeType = RankedTensorType::get(
        fullyConnectedShape,
        resultType.dyn_cast<ShapedType>().getElementType());

    Value fullyConnectedValue;
    if (op.quantization_info()) {
      fullyConnectedValue =
          rewriter
              .create<tosa::FullyConnectedOp>(
                  op.getLoc(), fullyConnectedShapeType, reshapedInput,
                  reshapedWeight, op.bias(), *op.quantization_info())
              .getResult();
    } else {
      fullyConnectedValue = rewriter
                                .create<tosa::FullyConnectedOp>(
                                    op.getLoc(), fullyConnectedShapeType,
                                    reshapedInput, reshapedWeight, op.bias())
                                .getResult();
    }

    // Reshape output to [N, IH, IW, OC].
    llvm::SmallVector<int64_t, 4> outputShape{inputShape[0], inputShape[1],
                                              inputShape[2], weightShape[0]};
    rewriter.replaceOpWithNewOp<tosa::ReshapeOp>(
        op, resultType, fullyConnectedValue,
        rewriter.getI64ArrayAttr(outputShape));
    return success();
  }
};

} // namespace

void mlir::tosa::populateTosaDecomposeConv2D(MLIRContext *ctx,
                                             RewritePatternSet &patterns) {
  patterns.add<Conv2DIsFullyConnected>(ctx);
}
