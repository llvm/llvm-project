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
#include "mlir/Dialect/Tosa/Utils/ShapeUtils.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::tosa;

namespace {

template <typename TosaOp, typename... Args>
TosaOp createOpAndInfer(PatternRewriter &rewriter, Location loc, Type resultTy,
                        Args &&...args) {
  auto op = rewriter.create<TosaOp>(loc, resultTy, args...);

  InferShapedTypeOpInterface shapeInterface =
      dyn_cast<InferShapedTypeOpInterface>(op.getOperation());
  if (!shapeInterface)
    return op;

  SmallVector<ShapedTypeComponents> returnedShapes;
  if (shapeInterface
          .inferReturnTypeComponents(op.getContext(), op.getLoc(),
                                     op->getOperands(), op->getAttrDictionary(),
                                     op->getRegions(), returnedShapes)
          .failed())
    return op;

  // We need to use the element type of the existing result type to generate
  // the new result shaped type. This is because rescale can include a cast to
  // different bit-width types and does not have a TypeAttr to define the
  // target type.
  auto result = op->getResult(0);
  auto predictedShape = returnedShapes[0];
  auto currentKnowledge =
      mlir::tosa::ValueKnowledge::getKnowledgeFromType(resultTy);

  // Compute the knowledge based on the inferred type.
  auto inferredKnowledge =
      mlir::tosa::ValueKnowledge::getPessimisticValueState();
  inferredKnowledge.dtype = resultTy.cast<ShapedType>().getElementType();
  inferredKnowledge.hasRank = predictedShape.hasRank();
  if (predictedShape.hasRank()) {
    for (auto dim : predictedShape.getDims()) {
      inferredKnowledge.sizes.push_back(dim);
    }
  }

  // Compute the new type based on the joined version.
  auto newKnowledge =
      mlir::tosa::ValueKnowledge::join(currentKnowledge, inferredKnowledge);
  auto newTy = newKnowledge.getType();
  result.setType(newTy);
  return op;
}

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

    ShapedType inputTy = input.getType().cast<ShapedType>();
    ShapedType weightTy = weight.getType().cast<ShapedType>();
    ShapedType biasTy = bias.getType().cast<ShapedType>();
    ShapedType resultTy = op->getResult(0).getType().cast<ShapedType>();

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
        loc, weightTy, weight, rewriter.getI64IntegerAttr(1));
    auto reverse2 = rewriter.create<tosa::ReverseOp>(
        loc, weightTy, reverse1, rewriter.getI64IntegerAttr(2));

    Value conv2d;
    if (op.getQuantizationInfo()) {
      conv2d = rewriter.create<tosa::Conv2DOp>(
          loc, resultTy, input, reverse2, bias,
          rewriter.getDenseI64ArrayAttr(convPad),
          rewriter.getDenseI64ArrayAttr(stride),
          rewriter.getDenseI64ArrayAttr({1, 1}), *op.getQuantizationInfo());
    } else {
      conv2d = rewriter.create<tosa::Conv2DOp>(
          loc, resultTy, input, reverse2, bias,
          rewriter.getDenseI64ArrayAttr(convPad),
          rewriter.getDenseI64ArrayAttr(stride),
          rewriter.getDenseI64ArrayAttr({1, 1}));
    }

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

    ShapedType inputTy = input.getType().cast<ShapedType>();
    ShapedType weightTy = weight.getType().cast<ShapedType>();
    ShapedType biasTy = bias.getType().cast<ShapedType>();
    ShapedType resultTy = op->getResult(0).getType().cast<ShapedType>();

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
    llvm::SmallVector<int32_t, 8> weightPadding = {0, 0, 0, 0, 0, 0, 0, 0};
    weightPadding[3] =
        weightHeight % stride[0] ? stride[0] - weightHeight % stride[0] : 0;
    weightPadding[5] =
        weightWidth % stride[1] ? stride[1] - weightWidth % stride[1] : 0;
    DenseElementsAttr weightPaddingAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({4, 2}, rewriter.getI32Type()), weightPadding);
    Value weightPaddingVal = createOpAndInfer<tosa::ConstOp>(
        rewriter, loc, weightPaddingAttr.getType(), weightPaddingAttr);

    if (op.getQuantizationInfo().has_value()) {
      auto quantInfo = op.getQuantizationInfo().value();
      weight = createOpAndInfer<tosa::PadOp>(
          rewriter, loc, UnrankedTensorType::get(weightETy), weight,
          weightPaddingVal, nullptr,
          rewriter.getAttr<PadOpQuantizationAttr>(quantInfo.getWeightZp()));

    } else {
      weight = createOpAndInfer<tosa::PadOp>(rewriter, loc,
                                             UnrankedTensorType::get(weightETy),
                                             weight, weightPaddingVal);
    }

    weightTy = weight.getType().cast<ShapedType>();
    weightHeight = weightTy.getDimSize(1);
    weightWidth = weightTy.getDimSize(2);

    // Split out the width / height by the stride dimensions.
    llvm::SmallVector<int64_t, 6> weightReshapeDims0 = {
        outputChannels, weightHeight / stride[0],
        stride[0],      weightWidth / stride[1],
        stride[1],      inputChannels};
    weight = createOpAndInfer<tosa::ReshapeOp>(
        rewriter, loc, UnrankedTensorType::get(weightETy), weight,
        rewriter.getDenseI64ArrayAttr(weightReshapeDims0));

    // Transpose the factored-out stride to the output channels.
    Value transposeWeightVal = rewriter.create<tosa::ConstOp>(
        loc, RankedTensorType::get({6}, rewriter.getI32Type()),
        rewriter.getI32TensorAttr({2, 4, 0, 1, 3, 5}));

    weight = createOpAndInfer<tosa::TransposeOp>(
        rewriter, loc, UnrankedTensorType::get(weightETy), weight,
        transposeWeightVal);

    // Collapse the strides and output channels into a single dimension.
    llvm::SmallVector<int64_t, 6> weightReshapeDims1 = {
        outputChannels * stride[0] * stride[1], weightHeight / stride[0],
        weightWidth / stride[1], inputChannels};
    weight = createOpAndInfer<tosa::ReshapeOp>(
        rewriter, loc, UnrankedTensorType::get(weightETy), weight,
        rewriter.getDenseI64ArrayAttr(weightReshapeDims1));
    ShapedType restridedWeightTy = weight.getType().cast<ShapedType>();

    weight = createOpAndInfer<tosa::ReverseOp>(
        rewriter, loc, UnrankedTensorType::get(weightETy), weight,
        rewriter.getI64IntegerAttr(1));
    weight = createOpAndInfer<tosa::ReverseOp>(
        rewriter, loc, UnrankedTensorType::get(weightETy), weight,
        rewriter.getI64IntegerAttr(2));

    // We need to pad the input far enough that we can pull all values.
    llvm::SmallVector<int32_t, 8> inputPadding = {0, 0, 0, 0, 0, 0, 0, 0};
    inputPadding[2] += restridedWeightTy.getDimSize(1) - 1;
    inputPadding[3] += restridedWeightTy.getDimSize(1) - 1;
    inputPadding[4] += restridedWeightTy.getDimSize(2) - 1;
    inputPadding[5] += restridedWeightTy.getDimSize(2) - 1;

    DenseElementsAttr inputPaddingAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({4, 2}, rewriter.getI32Type()), inputPadding);

    Value inputPaddingVal = createOpAndInfer<tosa::ConstOp>(
        rewriter, loc, inputPaddingAttr.getType(), inputPaddingAttr);

    if (op.getQuantizationInfo().has_value()) {
      auto quantInfo = op.getQuantizationInfo().value();
      input = createOpAndInfer<tosa::PadOp>(
          rewriter, loc, UnrankedTensorType::get(inputETy), input,
          inputPaddingVal, nullptr,
          rewriter.getAttr<PadOpQuantizationAttr>(quantInfo.getInputZp()));
    } else {
      input = createOpAndInfer<tosa::PadOp>(rewriter, loc,
                                            UnrankedTensorType::get(inputETy),
                                            input, inputPaddingVal);
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

    // Perform the convolution using the zero bias.
    Value conv2d;
    if (op.getQuantizationInfo()) {
      conv2d = createOpAndInfer<tosa::Conv2DOp>(
                   rewriter, loc, UnrankedTensorType::get(resultETy), input,
                   weight, zeroBias,
                   /*pad=*/rewriter.getDenseI64ArrayAttr({0, 0, 0, 0}),
                   /*stride=*/rewriter.getDenseI64ArrayAttr({1, 1}),
                   /*dilation=*/rewriter.getDenseI64ArrayAttr({1, 1}),
                   *op.getQuantizationInfo())
                   .getResult();
    } else {
      conv2d = createOpAndInfer<tosa::Conv2DOp>(
                   rewriter, loc, UnrankedTensorType::get(resultETy), input,
                   weight, zeroBias,
                   /*pad=*/rewriter.getDenseI64ArrayAttr({0, 0, 0, 0}),
                   /*stride=*/rewriter.getDenseI64ArrayAttr({1, 1}),
                   /*dilation=*/rewriter.getDenseI64ArrayAttr({1, 1}))
                   .getResult();
    }

    // Factor the resulting width / height.
    ShapedType convTy = conv2d.getType().cast<ShapedType>();
    Type convETy = convTy.getElementType();

    int64_t convHeight = convTy.getDimSize(1);
    int64_t convWidth = convTy.getDimSize(2);

    // Factor striding out of the convolution result.
    llvm::SmallVector<int64_t, 6> convReshapeDims0 = {
        batch, convHeight, convWidth, stride[0], stride[1], outputChannels};
    conv2d = createOpAndInfer<tosa::ReshapeOp>(
        rewriter, loc, UnrankedTensorType::get(resultETy), conv2d,
        rewriter.getDenseI64ArrayAttr(convReshapeDims0));

    // Transpose the factored-out stride to the output channels.
    Value transposeConvVal = rewriter.create<tosa::ConstOp>(
        loc, RankedTensorType::get({6}, rewriter.getI32Type()),
        rewriter.getI32TensorAttr({0, 1, 3, 2, 4, 5}));

    conv2d = createOpAndInfer<tosa::TransposeOp>(
        rewriter, loc, UnrankedTensorType::get(convETy), conv2d,
        transposeConvVal);

    // Fuse striding behavior back into width / height.
    llvm::SmallVector<int64_t, 6> convReshapeDims1 = {
        batch, convHeight * stride[0], convWidth * stride[1], outputChannels};
    conv2d = createOpAndInfer<tosa::ReshapeOp>(
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

    auto slice = createOpAndInfer<tosa::SliceOp>(
                     rewriter, loc, UnrankedTensorType::get(resultETy), conv2d,
                     rewriter.getDenseI64ArrayAttr(sliceBegin),
                     rewriter.getDenseI64ArrayAttr(sliceSize))
                     .getResult();

    llvm::SmallVector<int32_t, 8> resultPadding = {0, 0, 0, 0, 0, 0, 0, 0};
    resultPadding[2] = resultPadTop;
    resultPadding[3] = resultTy.getDimSize(1) - resultPadTop - sliceSize[1];
    resultPadding[4] = resultPadLeft;
    resultPadding[5] = resultTy.getDimSize(2) - resultPadLeft - sliceSize[2];

    DenseElementsAttr resultPaddingAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({4, 2}, rewriter.getI32Type()), resultPadding);

    Value resultPaddingVal = createOpAndInfer<tosa::ConstOp>(
        rewriter, loc, resultPaddingAttr.getType(), resultPaddingAttr);

    auto resultPad = createOpAndInfer<tosa::PadOp>(
        rewriter, loc, UnrankedTensorType::get(resultETy), slice,
        resultPaddingVal);

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
