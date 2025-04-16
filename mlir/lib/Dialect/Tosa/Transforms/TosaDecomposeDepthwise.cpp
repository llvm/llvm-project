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
#include "mlir/Dialect/Tosa/Utils/ConversionUtils.h"
#include "mlir/IR/BuiltinTypes.h"
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

    Type inputETy = inputType.getElementType();
    Type weightETy = weightType.getElementType();
    Type resultETy = resultType.getElementType();

    if (inputETy != resultETy) {
      inputType = inputType.clone(resultETy);
      input = rewriter.create<tosa::CastOp>(op.getLoc(), inputType, input);
    }

    if (weightETy != resultETy) {
      weightType = weightType.clone(resultETy);
      weight = rewriter.create<tosa::CastOp>(op.getLoc(), weightType, weight);
    }

    auto failureOrMaybeZps = extractConvZpPair(op, rewriter);
    if (failed(failureOrMaybeZps))
      return failure();

    auto maybeZps = failureOrMaybeZps.value();
    if (maybeZps) {
      auto applyZp = [&](Value val, int64_t zp) -> Value {
        if (zp == 0)
          return val;
        auto ety = cast<ShapedType>(val.getType()).getElementType();
        std::vector<int64_t> shape(cast<ShapedType>(val.getType()).getRank(),
                                   1);
        auto zpTy = RankedTensorType::get(shape, ety);
        auto zpAttr =
            DenseElementsAttr::get(zpTy, rewriter.getIntegerAttr(ety, zp));
        auto zpVal = rewriter.create<tosa::ConstOp>(op.getLoc(), zpTy, zpAttr);
        return rewriter.create<tosa::SubOp>(op.getLoc(), val.getType(), val,
                                            zpVal);
      };

      input = applyZp(input, maybeZps->inputZp);
      weight = applyZp(weight, maybeZps->weightZp);
    }

    ArrayRef<int64_t> padAttr = op.getPad();
    llvm::SmallVector<int64_t> pad(10, 0);
    for (const auto &it : llvm::enumerate(padAttr))
      pad[it.index() + 2] = it.value();

    if (llvm::any_of(pad, [](int64_t p) { return p != 0; })) {
      Attribute zeroAttr = rewriter.getZeroAttr(inputETy);

      llvm::SmallVector<int64_t> newShape(inputType.getShape());
      for (int i = 0, s = pad.size(); i < s; ++i) {
        if (newShape[i / 2] != ShapedType::kDynamic) {
          newShape[i / 2] += pad[i];
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

    // Perform an elementwise mul over the reshaped input and weight.
    llvm::SmallVector<int64_t, 2> mulShape{
        inputType.getDimSize(0), inputType.getDimSize(1),
        inputType.getDimSize(2), inputType.getDimSize(3), weightShape[3]};
    auto mulShapeType = RankedTensorType::get(
        mulShape,
        dyn_cast<RankedTensorType>(weight.getType()).getElementType());

    if (EqualizeRanks(rewriter, op.getLoc(), input, weight).failed()) {
      return failure();
    }

    auto shiftElementType = IntegerType::get(rewriter.getContext(), 8);
    auto shiftType = RankedTensorType::get({1}, shiftElementType);
    auto shiftZeroAttr = DenseElementsAttr::get(
        shiftType, rewriter.getIntegerAttr(shiftElementType, 0));
    Value constZero =
        rewriter.create<tosa::ConstOp>(op.getLoc(), shiftType, shiftZeroAttr);
    Value mulValue = rewriter
                         .create<tosa::MulOp>(op.getLoc(), mulShapeType, input,
                                              weight, constZero)
                         .getResult();

    // Reshape output to [N, H, W, C * M].
    auto outputShape = cast<ShapedType>(op.getOutput().getType()).getShape();
    auto outputShapeType = RankedTensorType::get(
        outputShape,
        dyn_cast<RankedTensorType>(input.getType()).getElementType());
    Value outputValue = rewriter.create<tosa::ReshapeOp>(
        op.getLoc(), outputShapeType, mulValue,
        rewriter.getDenseI64ArrayAttr(outputShape));

    Value bias = op.getBias();
    if (EqualizeRanks(rewriter, op.getLoc(), outputValue, bias).failed()) {
      return failure();
    }

    // Add in the bias.
    rewriter
        .replaceOpWithNewOp<tosa::AddOp>(op, outputShapeType, outputValue, bias)
        .getResult();
    return success();
  }
};

} // namespace

void mlir::tosa::populateTosaDecomposeDepthwise(MLIRContext *ctx,
                                                RewritePatternSet &patterns) {
  patterns.add<DepthwiseConv2DIsMul>(ctx);
}
