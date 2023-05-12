//===- TosaDecomposeDepthwise.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Decompose TOSA Depthwise operation to a series of TOSA Ops specifically
// (1) Convert a 1x1 Depthwise to Reshape -> Mul -> Reshape -> Add
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::tosa;

namespace {

struct DepthwiseConv2DIsMul : public OpRewritePattern<tosa::DepthwiseConv2DOp> {
  explicit DepthwiseConv2DIsMul(MLIRContext *context)
      : OpRewritePattern(context) {}

  LogicalResult matchAndRewrite(tosa::DepthwiseConv2DOp op,
                                PatternRewriter &rewriter) const override {
    Value input = op.getInput();
    Value weight = op.getWeight();
    ShapedType inputType = cast<ShapedType>(input.getType());
    ShapedType weightType = cast<ShapedType>(weight.getType());
    ShapedType resultType = cast<ShapedType>(op.getOutput().getType());

    if (!(inputType.hasStaticShape() && weightType.hasStaticShape() &&
          resultType.hasStaticShape())) {
      return failure();
    }

    if (!llvm::all_of(op.getStride(), [](int64_t v) { return v == 1; }))
      return failure();

    // Only works for a 1x1 kernel.
    ArrayRef<int64_t> weightShape = weightType.getShape();
    if (weightShape[0] != 1 || weightShape[1] != 1) {
      return failure();
    }

    // Reshape input to [N, H, W, C] -> [N, H, W, C, 1].
    ArrayRef<int64_t> inputShape = inputType.getShape();
    llvm::SmallVector<int64_t, 2> revisedInputShape{
        inputShape[0], inputShape[1], inputShape[2], inputShape[3], 1};
    inputType = RankedTensorType::get(
        revisedInputShape,
        dyn_cast<RankedTensorType>(input.getType()).getElementType());
    input = rewriter
                .create<tosa::ReshapeOp>(
                    op.getLoc(), inputType, input,
                    rewriter.getDenseI64ArrayAttr(revisedInputShape))
                .getResult();

    if (inputType.getElementType() != resultType.getElementType()) {
      inputType = inputType.clone(resultType.getElementType());
      input = rewriter.create<tosa::CastOp>(op.getLoc(), inputType, input);
    }

    if (weightType.getElementType() != resultType.getElementType()) {
      weightType = weightType.clone(resultType.getElementType());
      weight = rewriter.create<tosa::CastOp>(op.getLoc(), weightType, weight);
    }

    if (auto quantizationInfo = op.getQuantizationInfo()) {
      auto iZp = quantizationInfo->getInputZp();
      auto wZp = quantizationInfo->getWeightZp();

      auto applyZp = [&](Value val, int64_t zp) -> Value {
        if (zp == 0)
          return val;
        auto ety = cast<ShapedType>(val.getType()).getElementType();
        auto zpTy = RankedTensorType::get({}, ety);
        auto zpAttr =
            DenseElementsAttr::get(zpTy, rewriter.getIntegerAttr(ety, zp));
        auto zpVal = rewriter.create<tosa::ConstOp>(op.getLoc(), zpTy, zpAttr);
        return rewriter.create<tosa::SubOp>(op.getLoc(), val.getType(), val,
                                            zpVal);
      };

      input = applyZp(input, iZp);
      weight = applyZp(weight, wZp);
    }

    ArrayRef<int64_t> padAttr = op.getPad();
    llvm::SmallVector<int64_t> pad(10, 0);
    for (const auto &it : llvm::enumerate(padAttr))
      pad[it.index() + 2] = it.value();

    if (llvm::any_of(pad, [](int64_t p) { return p != 0; })) {
      Type inputETy = inputType.getElementType();
      Attribute zeroAttr = rewriter.getZeroAttr(inputETy);

      llvm::SmallVector<int64_t> newShape(inputType.getShape());
      for (int i = 0, s = pad.size(); i < s; ++i) {
        if (newShape[i / 2] != ShapedType::kDynamic) {
          newShape[i / 2] += pad[i];
        }
      }

      auto padSizeTy = RankedTensorType::get({5, 2}, rewriter.getI64Type());
      auto padSize =
          DenseIntElementsAttr::get(padSizeTy, ArrayRef<int64_t>(pad));
      Value padSizeVal =
          rewriter.create<tosa::ConstOp>(op->getLoc(), padSizeTy, padSize);

      auto padTy = RankedTensorType::get({}, inputETy);
      auto padAttr = DenseElementsAttr::get(padTy, zeroAttr);
      Value padVal =
          rewriter.create<tosa::ConstOp>(op->getLoc(), padTy, padAttr);
      inputType = RankedTensorType::get(newShape, inputETy);
      input = rewriter.create<tosa::PadOp>(op->getLoc(), inputType, input,
                                           padSizeVal, padVal);
    }

    // Perform an elementwise mul over the reshaped input and weight.
    llvm::SmallVector<int64_t, 2> mulShape{
        inputType.getDimSize(0), inputType.getDimSize(1),
        inputType.getDimSize(2), inputType.getDimSize(3), weightShape[3]};
    auto mulShapeType = RankedTensorType::get(
        mulShape,
        dyn_cast<RankedTensorType>(weight.getType()).getElementType());
    Value mulValue = rewriter
                         .create<tosa::MulOp>(op.getLoc(), mulShapeType, input,
                                              weight, /*shift=*/0)
                         .getResult();

    // Reshape output to [N, H, W, C * M].
    auto outputShape = cast<ShapedType>(op.getOutput().getType()).getShape();
    auto outputShapeType = RankedTensorType::get(
        outputShape,
        dyn_cast<RankedTensorType>(input.getType()).getElementType());
    auto outputValue = rewriter.create<tosa::ReshapeOp>(
        op.getLoc(), outputShapeType, mulValue,
        rewriter.getDenseI64ArrayAttr(outputShape));

    // Add in the bias.
    rewriter
        .replaceOpWithNewOp<tosa::AddOp>(op, outputShapeType, outputValue,
                                         op.getBias())
        .getResult();
    return success();
  }
};

} // namespace

void mlir::tosa::populateTosaDecomposeDepthwise(MLIRContext *ctx,
                                                RewritePatternSet &patterns) {
  patterns.add<DepthwiseConv2DIsMul>(ctx);
}
