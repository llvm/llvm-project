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
#include "mlir/Dialect/Tosa/Utils/ConversionUtils.h"

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
    ShapedType inputType = cast<ShapedType>(input.getType());
    ShapedType weightType = cast<ShapedType>(weight.getType());
    ShapedType resultType = cast<ShapedType>(op.getType());

    auto numDynamic =
        llvm::count_if(inputType.getShape(), ShapedType::isDynamic);
    if (numDynamic > 1)
      return rewriter.notifyMatchFailure(
          op, "at most one dim in input may be dynamic");
    if (!weightType.hasRank())
      return rewriter.notifyMatchFailure(op, "unranked weight input");

    if (!llvm::all_of(op.getStride(), [](int64_t v) { return v == 1; }))
      return failure();

    // Only works for a 1x1 kernel.
    ArrayRef<int64_t> weightShape = weightType.getShape();
    if (weightShape[1] != 1 || weightShape[2] != 1)
      return failure();

    llvm::ArrayRef<int64_t> padAttr = op.getPad();
    llvm::SmallVector<int64_t> pad(8, 0);
    for (const auto &it : llvm::enumerate(padAttr))
      pad[it.index() + 2] = it.value();

    Type inputETy = inputType.getElementType();
    if (llvm::any_of(pad, [](int64_t p) { return p != 0; })) {
      auto failureOrMaybeZps = extractConvZpPair(op, rewriter);
      if (failed(failureOrMaybeZps))
        return failure();

      auto maybeZps = failureOrMaybeZps.value();

      Attribute zeroAttr =
          maybeZps ? rewriter.getIntegerAttr(inputETy, maybeZps->inputZp)
                   : rewriter.getZeroAttr(inputETy);

      llvm::SmallVector<int64_t> newShape(inputType.getShape());

      for (int i = 0, s = newShape.size(); i < s; ++i) {
        if (newShape[i] != ShapedType::kDynamic) {
          newShape[i] += pad[i * 2] + pad[i * 2 + 1];
        }
      }

      Value padSizeVal = getTosaConstShape(rewriter, op->getLoc(), pad);

      auto padTy = RankedTensorType::get({}, inputETy);
      auto padAttr = DenseElementsAttr::get(padTy, zeroAttr);
      Value padVal =
          rewriter.create<tosa::ConstOp>(op->getLoc(), padTy, padAttr);
      inputType = RankedTensorType::get(newShape, inputETy);
      input = rewriter.create<tosa::PadOp>(op->getLoc(), inputType, input,
                                           padSizeVal, padVal);
    }

    // Reshape input to [N,IH,IW,IC] -> [N * IH * IW, IC].
    ArrayRef<int64_t> inputShape = inputType.getShape();
    int64_t combined = ShapedType::kDynamic;
    if (numDynamic == 0)
      combined = inputShape[0] * inputShape[1] * inputShape[2];
    llvm::SmallVector<int64_t, 2> revisedInputShape{combined, inputShape[3]};
    auto revisedInputShapeType =
        RankedTensorType::get(revisedInputShape, inputType.getElementType());
    auto reshapedInput = rewriter
                             .create<tosa::ReshapeOp>(
                                 op.getLoc(), revisedInputShapeType, input,
                                 rewriter.getDenseI64ArrayAttr(
                                     convertFromMlirShape(revisedInputShape)))
                             .getResult();

    // Reshape kernel to [OC,KH,KW,IC] -> [OC, IC].
    llvm::SmallVector<int64_t, 2> revisedWeightShape{weightShape[0],
                                                     weightShape[3]};
    auto revisedWeightShapeType = RankedTensorType::get(
        revisedWeightShape,
        dyn_cast<RankedTensorType>(weight.getType()).getElementType());
    auto reshapedWeight = rewriter
                              .create<tosa::ReshapeOp>(
                                  op.getLoc(), revisedWeightShapeType, weight,
                                  rewriter.getDenseI64ArrayAttr(
                                      convertFromMlirShape(revisedWeightShape)))
                              .getResult();

    // Perform a fully connected network over the reshaped input and weight.
    llvm::SmallVector<int64_t, 2> fullyConnectedShape{combined, weightShape[0]};
    auto fullyConnectedShapeType =
        RankedTensorType::get(fullyConnectedShape, resultType.getElementType());

    auto failureOrMaybeZps = extractConvZpPair(op, rewriter);
    if (failed(failureOrMaybeZps))
      return failure();

    auto maybeZps = failureOrMaybeZps.value();
    Value fullyConnectedValue;
    if (maybeZps) {
      fullyConnectedValue =
          rewriter
              .create<tosa::FullyConnectedOp>(
                  op.getLoc(), fullyConnectedShapeType, reshapedInput,
                  reshapedWeight, op.getBias(),
                  rewriter.getI32IntegerAttr(maybeZps->inputZp),
                  rewriter.getI32IntegerAttr(maybeZps->weightZp))
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
        rewriter.getDenseI64ArrayAttr(convertFromMlirShape(outputShape)));
    return success();
  }
};

} // namespace

void mlir::tosa::populateTosaDecomposeConv2D(MLIRContext *ctx,
                                             RewritePatternSet &patterns) {
  patterns.add<Conv2DIsFullyConnected>(ctx);
}
