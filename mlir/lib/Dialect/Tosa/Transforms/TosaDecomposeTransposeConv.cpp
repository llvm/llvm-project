//===- TosaDecomposeTransposeConv.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Decompose TOSA TransposeConv operation to a series of TOSA Ops specifically
// (1) Convert a Dilated TransposeConv2D to Conv2D including reversing/reshaping
// etc.. of the weights (2) Convert a Strided TransposeConv2D to Conv2D
// including transposing/reversing/reshaping etc..
//     of the weights and input/output tenors and reversing/reshaping etc .. of
//     the weights
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Dialect/Tosa/Utils/ConversionUtils.h"
#include "mlir/Dialect/Tosa/Utils/ShapeUtils.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::tosa;

namespace {

class TransposeConvNonStridedConverter
    : public OpRewritePattern<tosa::TransposeConv2DOp> {
public:
  using OpRewritePattern<tosa::TransposeConv2DOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tosa::TransposeConv2DOp op,
                                PatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    Value input = op->getOperand(0);
    Value weight = op->getOperand(1);
    Value bias = op->getOperand(2);

    ShapedType inputTy = cast<ShapedType>(input.getType());
    ShapedType weightTy = cast<ShapedType>(weight.getType());
    ShapedType biasTy = cast<ShapedType>(bias.getType());
    ShapedType resultTy = cast<ShapedType>(op->getResult(0).getType());

    llvm::ArrayRef<int64_t> stride = op.getStride();
    llvm::ArrayRef<int64_t> pad = op.getOutPad();

    // If striding is all 1 we can modify padding and reverse the kernel along
    // the x/y direction to make it a regular convolution. This is much simpler
    // then handling striding....
    if (llvm::any_of(stride, [](int64_t v) { return v != 1; }))
      return failure();

    if (!inputTy.hasStaticShape() || !weightTy.hasStaticShape() ||
        !biasTy.hasStaticShape() || !resultTy.hasStaticShape())
      return failure();

    int64_t kernelHeight = weightTy.getDimSize(1);
    int64_t kernelWidth = weightTy.getDimSize(2);

    llvm::SmallVector<int64_t> convPad(4, 0);
    convPad[0] = kernelHeight - 1 + pad[0];
    convPad[1] = kernelHeight - 1 + pad[1];
    convPad[2] = kernelWidth - 1 + pad[2];
    convPad[3] = kernelWidth - 1 + pad[3];

    auto reverse1 = rewriter.create<tosa::ReverseOp>(
        loc, weightTy, weight, /* axis = */ rewriter.getI32IntegerAttr(1));
    auto reverse2 = rewriter.create<tosa::ReverseOp>(
        loc, weightTy, reverse1, /* axis = */ rewriter.getI32IntegerAttr(2));

    Value conv2d = rewriter.create<tosa::Conv2DOp>(
        loc, resultTy, input, reverse2, bias, op.getInputZp(), op.getWeightZp(),
        rewriter.getDenseI64ArrayAttr(convPad),
        rewriter.getDenseI64ArrayAttr(stride),
        rewriter.getDenseI64ArrayAttr({1, 1}),
        /* acc_type = */ op.getAccType());

    rewriter.replaceOp(op, conv2d);
    return success();
  }
};

class TransposeConvStridedConverter
    : public OpRewritePattern<tosa::TransposeConv2DOp> {
public:
  using OpRewritePattern<tosa::TransposeConv2DOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tosa::TransposeConv2DOp op,
                                PatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    Value input = op->getOperand(0);
    Value weight = op->getOperand(1);
    Value bias = op->getOperand(2);

    ShapedType inputTy = cast<ShapedType>(input.getType());
    ShapedType weightTy = cast<ShapedType>(weight.getType());
    ShapedType biasTy = cast<ShapedType>(bias.getType());
    ShapedType resultTy = cast<ShapedType>(op->getResult(0).getType());

    Type inputETy = inputTy.getElementType();
    Type weightETy = weightTy.getElementType();
    Type biasETy = biasTy.getElementType();
    Type resultETy = resultTy.getElementType();

    llvm::ArrayRef<int64_t> pad = op.getOutPad();
    llvm::ArrayRef<int64_t> stride = op.getStride();

    // If striding is all 1 we can modify padding and reverse the kernel along
    // the x/y direction to make it a regular convolution. This is much simpler
    // then handling striding....

    // If strides are all 1 we dont need to use this one.
    if (llvm::all_of(stride, [](int64_t v) { return v == 1; }))
      return rewriter.notifyMatchFailure(op, "non-one stride found.");

    if (!inputTy.hasStaticShape() || !weightTy.hasStaticShape() ||
        !biasTy.hasStaticShape() || !resultTy.hasStaticShape())
      return failure();

    int64_t batch = inputTy.getDimSize(0);

    int64_t outputChannels = weightTy.getDimSize(0);
    int64_t weightHeight = weightTy.getDimSize(1);
    int64_t weightWidth = weightTy.getDimSize(2);
    int64_t inputChannels = weightTy.getDimSize(3);

    // Pad the weight so that it is modulo of the striding.
    llvm::SmallVector<int64_t, 8> weightPadding = {0, 0, 0, 0, 0, 0, 0, 0};
    weightPadding[3] =
        (weightHeight % stride[0]) ? (stride[0] - weightHeight % stride[0]) : 0;
    weightPadding[5] =
        weightWidth % stride[1] ? stride[1] - weightWidth % stride[1] : 0;

    Value weightPaddingVal =
        getTosaConstShape(rewriter, op->getLoc(), weightPadding);

    auto failureOrMaybeZps = extractConvZpPair(op, rewriter);
    if (failed(failureOrMaybeZps))
      return failure();

    auto maybeZps = failureOrMaybeZps.value();
    if (maybeZps) {
      weight = CreateOpAndInferShape<tosa::PadOp>(
          rewriter, loc, UnrankedTensorType::get(weightETy), weight,
          weightPaddingVal, nullptr,
          rewriter.getAttr<PadOpQuantizationAttr>(maybeZps->weightZp));

    } else {
      weight = CreateOpAndInferShape<tosa::PadOp>(
          rewriter, loc, UnrankedTensorType::get(weightETy), weight,
          weightPaddingVal);
    }

    weightTy = cast<ShapedType>(weight.getType());
    weightHeight = weightTy.getDimSize(1);
    weightWidth = weightTy.getDimSize(2);

    // Split out the width / height by the stride dimensions.
    llvm::SmallVector<int64_t, 6> weightReshapeDims0 = {
        outputChannels, weightHeight / stride[0],
        stride[0],      weightWidth / stride[1],
        stride[1],      inputChannels};
    weight = CreateOpAndInferShape<tosa::ReshapeOp>(
        rewriter, loc, UnrankedTensorType::get(weightETy), weight,
        rewriter.getDenseI64ArrayAttr(weightReshapeDims0));

    // Transpose the factored-out stride to the output channels.
    Value transposeWeightVal = rewriter.create<tosa::ConstOp>(
        loc, RankedTensorType::get({6}, rewriter.getI32Type()),
        rewriter.getI32TensorAttr({2, 4, 0, 1, 3, 5}));

    weight = CreateOpAndInferShape<tosa::TransposeOp>(
        rewriter, loc, UnrankedTensorType::get(weightETy), weight,
        transposeWeightVal);

    // Collapse the strides and output channels into a single dimension.
    llvm::SmallVector<int64_t, 6> weightReshapeDims1 = {
        outputChannels * stride[0] * stride[1], weightHeight / stride[0],
        weightWidth / stride[1], inputChannels};
    weight = CreateOpAndInferShape<tosa::ReshapeOp>(
        rewriter, loc, UnrankedTensorType::get(weightETy), weight,
        rewriter.getDenseI64ArrayAttr(weightReshapeDims1));
    ShapedType restridedWeightTy = cast<ShapedType>(weight.getType());

    weight = CreateOpAndInferShape<tosa::ReverseOp>(
        rewriter, loc, UnrankedTensorType::get(weightETy), weight,
        /* axis = */ rewriter.getI32IntegerAttr(1));
    weight = CreateOpAndInferShape<tosa::ReverseOp>(
        rewriter, loc, UnrankedTensorType::get(weightETy), weight,
        /* axis = */ rewriter.getI32IntegerAttr(2));

    // We need to pad the input far enough that we can pull all values.
    llvm::SmallVector<int64_t, 8> inputPadding = {0, 0, 0, 0, 0, 0, 0, 0};
    inputPadding[2] += restridedWeightTy.getDimSize(1) - 1;
    inputPadding[3] += restridedWeightTy.getDimSize(1) - 1;
    inputPadding[4] += restridedWeightTy.getDimSize(2) - 1;
    inputPadding[5] += restridedWeightTy.getDimSize(2) - 1;

    Value inputPaddingVal =
        getTosaConstShape(rewriter, op->getLoc(), inputPadding);

    if (maybeZps) {
      input = CreateOpAndInferShape<tosa::PadOp>(
          rewriter, loc, UnrankedTensorType::get(inputETy), input,
          inputPaddingVal, nullptr,
          rewriter.getAttr<PadOpQuantizationAttr>(maybeZps->inputZp));
    } else {
      input = CreateOpAndInferShape<tosa::PadOp>(
          rewriter, loc, UnrankedTensorType::get(inputETy), input,
          inputPaddingVal);
    }

    // We use a zero bias as we need to broadcast the bias.
    auto zeroBias = rewriter.create<tosa::ConstOp>(
        loc,
        RankedTensorType::get({outputChannels * stride[0] * stride[1]},
                              biasETy),
        DenseElementsAttr::get(
            RankedTensorType::get({outputChannels * stride[0] * stride[1]},
                                  biasETy),
            rewriter.getZeroAttr(biasETy)));

    Value inputZp, weightZp;
    if (maybeZps) {
      auto maybeInputZp = createZeroPointTensor(
          rewriter, loc, getElementTypeOrSelf(input.getType()),
          maybeZps->inputZp);
      auto maybeWeightZp = createZeroPointTensor(
          rewriter, loc, getElementTypeOrSelf(weight.getType()),
          maybeZps->weightZp);

      if (!maybeInputZp.has_value() || !maybeWeightZp.has_value()) {
        return rewriter.notifyMatchFailure(
            op, "fail to create a const zero point tensor");
      }

      inputZp = *maybeInputZp;
      weightZp = *maybeWeightZp;
    }

    // Perform the convolution using the zero bias.
    Value conv2d = CreateOpAndInferShape<tosa::Conv2DOp>(
                       rewriter, loc, UnrankedTensorType::get(resultETy), input,
                       weight, zeroBias, inputZp, weightZp,
                       /*pad=*/rewriter.getDenseI64ArrayAttr({0, 0, 0, 0}),
                       /*stride=*/rewriter.getDenseI64ArrayAttr({1, 1}),
                       /*dilation=*/rewriter.getDenseI64ArrayAttr({1, 1}),
                       /* acc_type = */ op.getAccType())
                       .getResult();

    // Factor the resulting width / height.
    ShapedType convTy = cast<ShapedType>(conv2d.getType());
    Type convETy = convTy.getElementType();

    int64_t convHeight = convTy.getDimSize(1);
    int64_t convWidth = convTy.getDimSize(2);

    // Factor striding out of the convolution result.
    llvm::SmallVector<int64_t, 6> convReshapeDims0 = {
        batch, convHeight, convWidth, stride[0], stride[1], outputChannels};
    conv2d = CreateOpAndInferShape<tosa::ReshapeOp>(
        rewriter, loc, UnrankedTensorType::get(resultETy), conv2d,
        rewriter.getDenseI64ArrayAttr(convReshapeDims0));

    // Transpose the factored-out stride to the output channels.
    Value transposeConvVal = rewriter.create<tosa::ConstOp>(
        loc, RankedTensorType::get({6}, rewriter.getI32Type()),
        rewriter.getI32TensorAttr({0, 1, 3, 2, 4, 5}));

    conv2d = CreateOpAndInferShape<tosa::TransposeOp>(
        rewriter, loc, UnrankedTensorType::get(convETy), conv2d,
        transposeConvVal);

    // Fuse striding behavior back into width / height.
    llvm::SmallVector<int64_t, 6> convReshapeDims1 = {
        batch, convHeight * stride[0], convWidth * stride[1], outputChannels};
    conv2d = CreateOpAndInferShape<tosa::ReshapeOp>(
        rewriter, loc, UnrankedTensorType::get(resultETy), conv2d,
        rewriter.getDenseI64ArrayAttr(convReshapeDims1));

    // Determine the amount to slice / pad from the result start.
    int64_t resultSliceTop = std::max<int64_t>(0, -pad[0]);
    int64_t resultSliceLeft = std::max<int64_t>(0, -pad[2]);
    int64_t resultPadTop = std::max<int64_t>(0, pad[0]);
    int64_t resultPadLeft = std::max<int64_t>(0, pad[2]);

    // Try to slice the targetted result size, cap to the convolutions width.
    int64_t resultSliceHeight =
        std::min<int64_t>(convReshapeDims1[1] - resultSliceTop,
                          resultTy.getDimSize(1) - resultPadTop);
    int64_t resultSliceWidth =
        std::min<int64_t>(convReshapeDims1[2] - resultSliceLeft,
                          resultTy.getDimSize(2) - resultPadLeft);

    llvm::SmallVector<int64_t, 4> sliceBegin = {0, resultSliceTop,
                                                resultSliceLeft, 0};
    llvm::SmallVector<int64_t, 4> sliceSize(convReshapeDims1.begin(),
                                            convReshapeDims1.end());
    sliceSize[1] = resultSliceHeight;
    sliceSize[2] = resultSliceWidth;

    auto slice = CreateOpAndInferShape<tosa::SliceOp>(
                     rewriter, loc, UnrankedTensorType::get(resultETy), conv2d,
                     getTosaConstShape(rewriter, loc, sliceBegin),
                     getTosaConstShape(rewriter, loc, sliceSize))
                     .getResult();

    llvm::SmallVector<int64_t, 8> resultPadding = {0, 0, 0, 0, 0, 0, 0, 0};
    resultPadding[2] = resultPadTop;
    resultPadding[3] = resultTy.getDimSize(1) - resultPadTop - sliceSize[1];
    resultPadding[4] = resultPadLeft;
    resultPadding[5] = resultTy.getDimSize(2) - resultPadLeft - sliceSize[2];

    Value resultPaddingVal =
        getTosaConstShape(rewriter, op->getLoc(), resultPadding);

    Value resultPad = CreateOpAndInferShape<tosa::PadOp>(
        rewriter, loc, UnrankedTensorType::get(resultETy), slice,
        resultPaddingVal);

    if (EqualizeRanks(rewriter, op.getLoc(), resultPad, bias).failed()) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<tosa::AddOp>(op, op.getType(), resultPad, bias);
    return success();
  }
};

} // namespace

void mlir::tosa::populateTosaDecomposeTransposeConv(
    MLIRContext *ctx, RewritePatternSet &patterns) {
  patterns.add<TransposeConvNonStridedConverter>(ctx);
  patterns.add<TransposeConvStridedConverter>(ctx);
}
