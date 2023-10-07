//===- TosaToLinalgNamed.cpp - Lowering Tosa to Linalg Named Ops ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// These rewriters lower from the Tosa to the Linalg named ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/TosaToLinalg/TosaToLinalg.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Utils/ConversionUtils.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <numeric>

using namespace mlir;
using namespace mlir::tosa;

static mlir::Value applyPad(Location loc, Value input, ArrayRef<int64_t> pad,
                            TypedAttr padAttr, OpBuilder &rewriter) {
  // Input should be padded if necessary.
  if (llvm::all_of(pad, [](int64_t p) { return p == 0; }))
    return input;

  ShapedType inputTy = cast<ShapedType>(input.getType());
  Type inputETy = inputTy.getElementType();
  auto inputShape = inputTy.getShape();

  assert((inputShape.size() * 2) == pad.size());

  SmallVector<int64_t, 4> paddedShape;
  SmallVector<OpFoldResult, 8> lowIndices;
  SmallVector<OpFoldResult, 8> highIndices;
  for (int i = 0, s = inputShape.size(); i < s; i++) {
    auto lowPad = pad[i * 2];
    auto highPad = pad[i * 2 + 1];
    if (ShapedType::isDynamic(inputShape[i]))
      paddedShape.push_back(inputShape[i]);
    else
      paddedShape.push_back(inputShape[i] + highPad + lowPad);
    lowIndices.push_back(rewriter.getIndexAttr(lowPad));
    highIndices.push_back(rewriter.getIndexAttr(highPad));
  }

  Value padValue = rewriter.create<arith::ConstantOp>(loc, padAttr);

  return rewriter.create<tensor::PadOp>(
      loc, RankedTensorType::get(paddedShape, inputETy), input, lowIndices,
      highIndices, padValue);
}

static mlir::Value
linalgIntBroadcastExtSIAdd(PatternRewriter &rewriter, Location loc, Value bias,
                           Value conv, Value result,
                           ArrayRef<AffineMap> indexingMaps) {
  ShapedType resultTy = cast<ShapedType>(conv.getType());
  return rewriter
      .create<linalg::GenericOp>(
          loc, resultTy, ValueRange({bias, conv}), result, indexingMaps,
          getNParallelLoopsAttrs(resultTy.getRank()),
          [](OpBuilder &builder, Location loc, ValueRange args) {
            Value biasVal = args[0];
            Type resType = args[1].getType();
            if (resType != biasVal.getType()) {
              biasVal = builder.create<arith::ExtSIOp>(loc, resType, biasVal);
            }
            Value added = builder.create<arith::AddIOp>(loc, biasVal, args[1]);
            builder.create<linalg::YieldOp>(loc, added);
          })
      .getResult(0);
}

static mlir::Value reifyConstantDim(int64_t attr,
                                    ImplicitLocOpBuilder &builder) {
  return builder.createOrFold<arith::IndexCastOp>(
      builder.getIndexType(),
      builder.create<arith::ConstantOp>(builder.getI64IntegerAttr(attr)));
}

// Calculating the output width/height using the formula:
// H = ((IH+pad_top+pad_bottom-(dilation_y*(KH-1)+1))/stride_y)+1
// W = ((IW+pad_left+pad_right-(dilation_x*(KW-1)+1))/stride_x)+1

static mlir::Value getConvOutputDim(Location loc, Value inputDim,
                                    int64_t padBeforeAttr, int64_t padAfterAttr,
                                    Value kernelDim, int64_t strideAttr,
                                    int64_t dilationAttr, Type inputETy,
                                    OpBuilder &rewriter) {
  ImplicitLocOpBuilder builder(loc, rewriter);
  auto one = rewriter.create<arith::ConstantOp>(
      loc, IntegerAttr::get(inputDim.getType(), 1));
  Value padBefore = reifyConstantDim(padBeforeAttr, builder);
  Value paddedBefore = builder.create<arith::AddIOp>(inputDim, padBefore);
  Value padAfter = reifyConstantDim(padAfterAttr, builder);
  Value paddedAfter = builder.create<arith::AddIOp>(paddedBefore, padAfter);

  Value subOne = builder.create<arith::SubIOp>(kernelDim, one);
  Value dilation = reifyConstantDim(dilationAttr, builder);
  Value dilated = builder.create<arith::MulIOp>(dilation, subOne);
  Value addOne = builder.create<arith::AddIOp>(dilated, one);

  Value subtract = builder.create<arith::SubIOp>(paddedAfter, addOne);
  Value stride = reifyConstantDim(strideAttr, builder);
  Value divide = builder.create<arith::DivUIOp>(subtract, stride);
  return builder.create<arith::AddIOp>(divide, one);
}

// Creates a vector of the dynamic output dims for Conv2D and Depthwise_Conv2D
static SmallVector<Value> inferDynamicDimsForConv(
    Location loc, Value input, Value weight, ShapedType resultTy,
    ArrayRef<int64_t> padAttr, ArrayRef<int64_t> strideAttr,
    ArrayRef<int64_t> dilationAttr, ArrayRef<int64_t> inputSizeDims,
    ArrayRef<int64_t> kernelSizeDims, OpBuilder &rewriter) {
  ShapedType inputTy = cast<ShapedType>(input.getType());
  Type inputETy = inputTy.getElementType();
  int64_t inputRank = inputTy.getRank();

  SmallVector<Value> dynDims;
  dynDims.resize(resultTy.getRank());

  for (uint32_t i = 0, s = inputSizeDims.size(); i < s; ++i) {
    int64_t inputDim = inputSizeDims[i];
    int64_t kernelDim = kernelSizeDims[i];
    if (inputTy.isDynamicDim(inputDim)) {
      auto padTop = padAttr[i * 2];
      auto padBottom = padAttr[i * 2 + 1];
      auto stride = strideAttr[i];
      auto dilation = dilationAttr[i];
      Value initDynDim = rewriter.create<tensor::DimOp>(loc, input, inputDim);
      Value kernelDynDim =
          rewriter.create<tensor::DimOp>(loc, weight, kernelDim);
      // H = F(IH, pad_top, pad_bottom, dilation_y, KH, stride_y)
      dynDims[inputDim] =
          getConvOutputDim(loc, initDynDim, padTop, padBottom, kernelDynDim,
                           stride, dilation, inputETy, rewriter);
    }
  }

  // Get the batch/channels dimensions.
  for (int i = 0; i < inputRank; i++) {
    if (inputTy.isDynamicDim(i) && !dynDims[i])
      dynDims[i] = rewriter.create<tensor::DimOp>(loc, input, i);
  }

  SmallVector<Value> filteredDims = condenseValues(dynDims);
  return filteredDims;
}

// Creates a map to collapse the last dimension of the Depthwise convolution op
// due to a shape mismatch
static void createDepthwiseConvCollapseMap(
    int64_t outputRank, SmallVector<ReassociationExprs, 4> &reassociationMap,
    OpBuilder &rewriter) {
  reassociationMap.resize(outputRank);
  for (int i = 0; i < outputRank; i++) {
    reassociationMap[i].push_back(rewriter.getAffineDimExpr(i));
  }
  reassociationMap[outputRank - 1].push_back(
      rewriter.getAffineDimExpr(outputRank));
}

namespace {

template <typename TosaConvOp, typename LinalgConvOp, typename LinalgConvQOp>
class ConvConverter : public OpConversionPattern<TosaConvOp> {
public:
  using OpConversionPattern<TosaConvOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(TosaConvOp op, typename TosaConvOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    Value input = op->getOperand(0);
    Value weight = op->getOperand(1);
    Value bias = op->getOperand(2);

    ShapedType inputTy = cast<ShapedType>(input.getType());
    ShapedType weightTy = cast<ShapedType>(weight.getType());
    ShapedType biasTy = cast<ShapedType>(bias.getType());
    ShapedType resultTy = cast<ShapedType>(op->getResult(0).getType());

    Type inputETy = inputTy.getElementType();
    Type resultETy = resultTy.getElementType();

    DenseI64ArrayAttr padAttr = op.getPadAttr();
    DenseI64ArrayAttr strideTosaAttr = op.getStrideAttr();
    DenseI64ArrayAttr dilationTosaAttr = op.getDilationAttr();
    bool isQuantized = op.getQuantizationInfo().has_value();

    if (!weightTy.hasStaticShape() || !biasTy.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "tosa.conv ops require static shapes for weight and bias");

    if (inputETy.isUnsignedInteger())
      return rewriter.notifyMatchFailure(
          op, "tosa.conv ops does not support unsigned integer input");

    llvm::SmallVector<int64_t> inputSizeDims;
    llvm::SmallVector<int64_t> kernelSizeDims;
    for (int i = 1; i < resultTy.getRank() - 1; i++) {
      inputSizeDims.push_back(i);
      kernelSizeDims.push_back(i);
    }

    SmallVector<Value> filteredDims = inferDynamicDimsForConv(
        loc, input, weight, resultTy, padAttr.asArrayRef(),
        strideTosaAttr.asArrayRef(), dilationTosaAttr.asArrayRef(),
        inputSizeDims, kernelSizeDims, rewriter);

    auto weightShape = weightTy.getShape();

    // Apply padding as necessary.
    TypedAttr zeroAttr = rewriter.getZeroAttr(inputETy);
    if (isQuantized) {
      auto quantizationInfo = *op.getQuantizationInfo();
      int64_t iZp = quantizationInfo.getInputZp();

      int64_t intMin =
          APInt::getSignedMinValue(inputETy.getIntOrFloatBitWidth())
              .getSExtValue();
      int64_t intMax =
          APInt::getSignedMaxValue(inputETy.getIntOrFloatBitWidth())
              .getSExtValue();

      if (iZp < intMin || iZp > intMax)
        return rewriter.notifyMatchFailure(
            op, "tosa.conv op quantization has zp outside of input range");

      zeroAttr = rewriter.getIntegerAttr(inputETy, iZp);
    }

    llvm::SmallVector<int64_t> pad;
    pad.resize(2, 0);
    llvm::append_range(pad, padAttr.asArrayRef());
    pad.resize(pad.size() + 2, 0);
    input = applyPad(loc, input, pad, zeroAttr, rewriter);

    // For Conv3D transpose the kernel to match dimension ordering of the linalg
    // convolution operation. Conv2D has a 1-1 mapping in linalg so better to
    // map directly and then transpose later if desired.
    if (5 == inputTy.getRank()) {
      // TODO(suderman): See if this can be efficiently folded - check whether
      // the input is used anywhere else, if not fold the constant.
      SmallVector<int64_t> weightPerm;
      for (int i = 1; i < resultTy.getRank(); i++)
        weightPerm.push_back(i);
      weightPerm.push_back(0);

      SmallVector<int64_t> newWeightShape;
      for (auto dim : weightPerm)
        newWeightShape.push_back(weightShape[dim]);
      auto weightPermAttr = rewriter.getI64TensorAttr(weightPerm);
      Value weightPermValue =
          rewriter.create<arith::ConstantOp>(loc, weightPermAttr);
      Type newWeightTy =
          RankedTensorType::get(newWeightShape, weightTy.getElementType());
      weight = rewriter.create<tosa::TransposeOp>(loc, newWeightTy, weight,
                                                  weightPermValue);
    }

    auto resultZeroAttr = rewriter.getZeroAttr(resultETy);
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultTy.getShape(), resultETy, filteredDims);
    Value zero = rewriter.create<arith::ConstantOp>(loc, resultZeroAttr);
    Value zeroTensor = rewriter
                           .create<linalg::FillOp>(loc, ValueRange{zero},
                                                   ValueRange{emptyTensor})
                           .result();

    // Extract the attributes for convolution.
    ArrayRef<int64_t> stride = strideTosaAttr;
    ArrayRef<int64_t> dilation = dilationTosaAttr;

    // Create the convolution op.
    auto strideAttr = rewriter.getI64TensorAttr(stride);
    auto dilationAttr = rewriter.getI64TensorAttr(dilation);

    // Create maps for the bias broadcasting
    SmallVector<AffineMap, 4> indexingMaps;
    indexingMaps.push_back(AffineMap::get(
        /*dimCount=*/resultTy.getRank(), /*symbolCount=*/0,
        {rewriter.getAffineDimExpr(resultTy.getRank() - 1)},
        rewriter.getContext()));
    indexingMaps.push_back(rewriter.getMultiDimIdentityMap(resultTy.getRank()));
    indexingMaps.push_back(rewriter.getMultiDimIdentityMap(resultTy.getRank()));

    Value biasEmptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultTy.getShape(), resultETy, filteredDims);

    if (isQuantized) {
      auto quantizationInfo = *op.getQuantizationInfo();
      auto iZp = rewriter.getI32IntegerAttr(quantizationInfo.getInputZp());
      auto kZp = rewriter.getI32IntegerAttr(quantizationInfo.getWeightZp());

      auto iZpVal = rewriter.create<arith::ConstantOp>(loc, iZp);
      auto kZpVal = rewriter.create<arith::ConstantOp>(loc, kZp);
      Value conv =
          rewriter
              .create<LinalgConvQOp>(
                  loc, resultTy, ValueRange{input, weight, iZpVal, kZpVal},
                  ValueRange{zeroTensor}, strideAttr, dilationAttr)
              ->getResult(0);
      Value result = linalgIntBroadcastExtSIAdd(rewriter, loc, bias, conv,
                                                biasEmptyTensor, indexingMaps);
      rewriter.replaceOp(op, result);
      return success();
    }

    Value conv = rewriter
                     .create<LinalgConvOp>(
                         loc, resultTy, ValueRange{input, weight},
                         ValueRange{zeroTensor}, strideAttr, dilationAttr)
                     ->getResult(0);

    Value result =
        rewriter
            .create<linalg::GenericOp>(
                loc, resultTy, ValueRange({bias, conv}), biasEmptyTensor,
                indexingMaps, getNParallelLoopsAttrs(resultTy.getRank()),
                [&](OpBuilder &nestedBuilder, Location nestedLoc,
                    ValueRange args) {
                  Value added = nestedBuilder.create<arith::AddFOp>(
                      loc, args[0], args[1]);
                  nestedBuilder.create<linalg::YieldOp>(nestedLoc, added);
                })
            .getResult(0);

    rewriter.replaceOp(op, result);
    return success();
  }
};

class DepthwiseConvConverter
    : public OpConversionPattern<tosa::DepthwiseConv2DOp> {
public:
  using OpConversionPattern<tosa::DepthwiseConv2DOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tosa::DepthwiseConv2DOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    Value input = op->getOperand(0);
    Value weight = op->getOperand(1);
    Value bias = op->getOperand(2);

    ShapedType inputTy = cast<ShapedType>(input.getType());
    ShapedType weightTy = cast<ShapedType>(weight.getType());
    ShapedType biasTy = cast<ShapedType>(bias.getType());
    ShapedType resultTy = cast<ShapedType>(op->getResult(0).getType());
    int64_t resultRank = resultTy.getRank();

    Type inputETy = inputTy.getElementType();
    Type resultETy = resultTy.getElementType();

    auto padAttr = cast<DenseI64ArrayAttr>(op->getAttr("pad"));
    auto strideTosaAttr = cast<DenseI64ArrayAttr>(op->getAttr("stride"));
    auto dilationTosaAttr = cast<DenseI64ArrayAttr>(op->getAttr("dilation"));

    if (!weightTy.hasStaticShape() || !biasTy.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "tosa.depthwise_conv ops require static shapes");

    // Compute output dynamic dims
    SmallVector<Value> filteredDims = inferDynamicDimsForConv(
        loc, input, weight, resultTy, padAttr.asArrayRef(),
        strideTosaAttr.asArrayRef(), dilationTosaAttr.asArrayRef(),
        /*inputSizeDims=*/{1, 2},
        /*kernelSizeDims=*/{0, 1}, rewriter);

    bool isQuantized = op->hasAttr("quantization_info");
    IntegerAttr iZp;
    IntegerAttr kZp;
    if (isQuantized) {
      auto quantizationInfo =
          cast<tosa::ConvOpQuantizationAttr>(op->getAttr("quantization_info"));
      iZp = rewriter.getI32IntegerAttr(quantizationInfo.getInputZp());
      kZp = rewriter.getI32IntegerAttr(quantizationInfo.getWeightZp());
    }

    auto weightShape = weightTy.getShape();
    auto resultShape = resultTy.getShape();

    // Apply padding as necessary.
    TypedAttr zeroAttr = rewriter.getZeroAttr(inputETy);
    if (isQuantized) {
      auto quantizationInfo =
          cast<tosa::ConvOpQuantizationAttr>(op->getAttr("quantization_info"));
      int64_t iZp = quantizationInfo.getInputZp();

      int64_t intMin =
          APInt::getSignedMinValue(inputETy.getIntOrFloatBitWidth())
              .getSExtValue();
      int64_t intMax =
          APInt::getSignedMaxValue(inputETy.getIntOrFloatBitWidth())
              .getSExtValue();

      if (iZp < intMin || iZp > intMax)
        return rewriter.notifyMatchFailure(
            op, "tosa.depthwise_conv op quantization has zp outside of input "
                "range");

      zeroAttr = rewriter.getIntegerAttr(inputETy, iZp);
    }

    llvm::SmallVector<int64_t> pad;
    pad.resize(2, 0);
    llvm::append_range(pad, padAttr.asArrayRef());
    pad.resize(pad.size() + 2, 0);

    input = applyPad(loc, input, pad, zeroAttr, rewriter);

    // Extract the attributes for convolution.
    ArrayRef<int64_t> stride = strideTosaAttr;
    ArrayRef<int64_t> dilation = dilationTosaAttr;

    // Create the convolution op.
    auto strideAttr = rewriter.getI64TensorAttr(stride);
    auto dilationAttr = rewriter.getI64TensorAttr(dilation);
    ShapedType linalgConvTy =
        RankedTensorType::get({resultShape[0], resultShape[1], resultShape[2],
                               weightShape[2], weightShape[3]},
                              resultETy);

    // Broadcast the initial value to the output tensor before convolving.
    SmallVector<AffineMap, 4> indexingMaps;
    indexingMaps.push_back(AffineMap::get(
        /*dimCount=*/resultRank, /*symbolCount=*/0,
        {rewriter.getAffineDimExpr(3)}, rewriter.getContext()));
    indexingMaps.push_back(rewriter.getMultiDimIdentityMap(resultRank));
    indexingMaps.push_back(rewriter.getMultiDimIdentityMap(resultRank));

    auto resultZeroAttr = rewriter.getZeroAttr(resultETy);
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, linalgConvTy.getShape(), resultETy, filteredDims);
    Value zero = rewriter.create<arith::ConstantOp>(loc, resultZeroAttr);
    Value zeroTensor = rewriter
                           .create<linalg::FillOp>(loc, ValueRange{zero},
                                                   ValueRange{emptyTensor})
                           .result();

    Value biasEmptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultTy.getShape(), resultETy, filteredDims);
    if (!isQuantized) {
      Value conv = rewriter
                       .create<linalg::DepthwiseConv2DNhwcHwcmOp>(
                           loc, linalgConvTy, ValueRange{input, weight},
                           ValueRange{zeroTensor}, strideAttr, dilationAttr)
                       .getResult(0);

      SmallVector<ReassociationExprs, 4> reassociationMap;
      createDepthwiseConvCollapseMap(resultRank, reassociationMap, rewriter);
      Value convReshape = rewriter.create<tensor::CollapseShapeOp>(
          loc, resultTy, conv, reassociationMap);

      Value result =
          rewriter
              .create<linalg::GenericOp>(
                  loc, resultTy, ValueRange({bias, convReshape}),
                  biasEmptyTensor, indexingMaps,
                  getNParallelLoopsAttrs(resultRank),
                  [&](OpBuilder &nestedBuilder, Location nestedLoc,
                      ValueRange args) {
                    Value added = nestedBuilder.create<arith::AddFOp>(
                        loc, args[0], args[1]);
                    nestedBuilder.create<linalg::YieldOp>(nestedLoc, added);
                  })
              .getResult(0);
      rewriter.replaceOp(op, result);
    } else {
      auto iZpVal = rewriter.create<arith::ConstantOp>(loc, iZp);
      auto kZpVal = rewriter.create<arith::ConstantOp>(loc, kZp);
      Value conv =
          rewriter
              .create<linalg::DepthwiseConv2DNhwcHwcmQOp>(
                  loc, linalgConvTy, ValueRange{input, weight, iZpVal, kZpVal},
                  ValueRange{zeroTensor}, strideAttr, dilationAttr)
              .getResult(0);
      SmallVector<ReassociationExprs, 4> reassociationMap;
      createDepthwiseConvCollapseMap(resultRank, reassociationMap, rewriter);
      Value convReshape = rewriter.create<tensor::CollapseShapeOp>(
          loc, resultTy, conv, reassociationMap);
      Value result = linalgIntBroadcastExtSIAdd(
          rewriter, loc, bias, convReshape, biasEmptyTensor, indexingMaps);
      rewriter.replaceOp(op, result);
    }
    return success();
  }
};

class MatMulConverter : public OpConversionPattern<tosa::MatMulOp> {
public:
  using OpConversionPattern<tosa::MatMulOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tosa::MatMulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Location loc = op.getLoc();

    auto outputTy = cast<ShapedType>(op.getType());
    auto outputElementTy = outputTy.getElementType();

    auto firstOperandTy = cast<ShapedType>(op->getOperand(0).getType());
    auto secondOperandTy = cast<ShapedType>(op->getOperand(1).getType());

    SmallVector<Value> dynDims;
    dynDims.resize(cast<ShapedType>(op->getResult(0).getType()).getRank());

    if (!firstOperandTy.hasRank() || firstOperandTy.isDynamicDim(0)) {
      dynDims[0] = rewriter.create<tensor::DimOp>(loc, op->getOperand(0), 0);
    }

    if (!firstOperandTy.hasRank() || firstOperandTy.isDynamicDim(1)) {
      dynDims[1] = rewriter.create<tensor::DimOp>(loc, op->getOperand(0), 1);
    }

    if (!secondOperandTy.hasRank() || secondOperandTy.isDynamicDim(2)) {
      dynDims[2] = rewriter.create<tensor::DimOp>(loc, op->getOperand(1), 2);
    }

    SmallVector<Value> filteredDims = condenseValues(dynDims);

    auto zeroAttr = rewriter.getZeroAttr(outputElementTy);
    Value zero = rewriter.create<arith::ConstantOp>(loc, zeroAttr);
    auto emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, outputTy.getShape(), outputTy.getElementType(), filteredDims);
    Value zeroTensor = rewriter
                           .create<linalg::FillOp>(loc, ValueRange{zero},
                                                   ValueRange{emptyTensor})
                           .result();
    if (!op.getQuantizationInfo()) {
      rewriter.replaceOpWithNewOp<linalg::BatchMatmulOp>(
          op, TypeRange{op.getType()},
          ValueRange{adaptor.getA(), adaptor.getB()}, ValueRange{zeroTensor});
      return success();
    }

    auto quantizationInfo = *op.getQuantizationInfo();
    auto aZp = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI32IntegerAttr(quantizationInfo.getAZp()));
    auto bZp = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI32IntegerAttr(quantizationInfo.getBZp()));
    rewriter.replaceOpWithNewOp<linalg::QuantizedBatchMatmulOp>(
        op, TypeRange{op.getType()},
        ValueRange{adaptor.getA(), adaptor.getB(), aZp, bZp}, zeroTensor);

    return success();
  }
};

class FullyConnectedConverter
    : public OpConversionPattern<tosa::FullyConnectedOp> {
public:
  using OpConversionPattern<tosa::FullyConnectedOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tosa::FullyConnectedOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    auto outputTy = cast<ShapedType>(op.getType());
    auto input = op.getInput();
    auto inputTy = cast<ShapedType>(input.getType());

    auto bias = op.getBias();

    auto weight = op.getWeight();
    auto weightTy = cast<ShapedType>(weight.getType());
    auto weightShape = weightTy.getShape();

    auto outputETy = outputTy.getElementType();

    SmallVector<Value> dynDims;
    dynDims.resize(cast<ShapedType>(op->getResult(0).getType()).getRank());

    if (!inputTy.hasRank() || inputTy.isDynamicDim(0)) {
      dynDims[0] = rewriter.create<tensor::DimOp>(loc, input, 0);
    }

    if (!weightTy.hasRank() || weightTy.isDynamicDim(0)) {
      dynDims[1] = rewriter.create<tensor::DimOp>(loc, weight, 0);
    }

    SmallVector<Value> filteredDims = condenseValues(dynDims);

    // Creating maps for the output of MatMul and the bias
    SmallVector<AffineMap, 4> indexingMaps;

    // Broadcast the bias.
    indexingMaps.push_back(AffineMap::get(/*dimCount=*/2, /*symbolCount=*/0,
                                          {rewriter.getAffineDimExpr(1)},
                                          rewriter.getContext()));

    indexingMaps.push_back(rewriter.getMultiDimIdentityMap(outputTy.getRank()));
    indexingMaps.push_back(rewriter.getMultiDimIdentityMap(outputTy.getRank()));

    auto emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, outputTy.getShape(), outputTy.getElementType(), filteredDims);

    // When quantized, the input elemeny type is not the same as the output
    auto resultZeroAttr = rewriter.getZeroAttr(outputETy);
    Value zero = rewriter.create<arith::ConstantOp>(loc, resultZeroAttr);
    Value zeroTensor = rewriter
                           .create<linalg::FillOp>(loc, ValueRange{zero},
                                                   ValueRange{emptyTensor})
                           .result();

    SmallVector<int64_t> permutation{1, 0};
    auto permutationAttr = rewriter.getI64TensorAttr(permutation);
    Value permutationValue =
        rewriter.create<arith::ConstantOp>(loc, permutationAttr);

    SmallVector<int64_t> newWeightShape{weightShape[1], weightShape[0]};
    Type newWeightTy =
        RankedTensorType::get(newWeightShape, weightTy.getElementType());

    Value transposedWeight = rewriter.create<tosa::TransposeOp>(
        loc, newWeightTy, weight, permutationValue);

    Value biasEmptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, outputTy.getShape(), outputETy, filteredDims);

    if (!op.getQuantizationInfo()) {
      Value matmul = rewriter
                         .create<linalg::MatmulOp>(
                             loc, TypeRange{op.getType()},
                             ValueRange{input, transposedWeight}, zeroTensor)
                         ->getResult(0);

      Value result =
          rewriter
              .create<linalg::GenericOp>(
                  loc, outputTy, ValueRange({bias, matmul}), biasEmptyTensor,
                  indexingMaps, getNParallelLoopsAttrs(outputTy.getRank()),
                  [&](OpBuilder &nestedBuilder, Location nestedLoc,
                      ValueRange args) {
                    Value added = nestedBuilder.create<arith::AddFOp>(
                        loc, args[0], args[1]);
                    nestedBuilder.create<linalg::YieldOp>(nestedLoc, added);
                  })
              .getResult(0);
      rewriter.replaceOp(op, result);
      return success();
    }

    auto quantizationInfo = *op.getQuantizationInfo();
    auto inputZp = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI32IntegerAttr(quantizationInfo.getInputZp()));
    auto outputZp = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI32IntegerAttr(quantizationInfo.getWeightZp()));
    Value matmul =
        rewriter
            .create<linalg::QuantizedMatmulOp>(
                loc, TypeRange{op.getType()},
                ValueRange{input, transposedWeight, inputZp, outputZp},
                zeroTensor)
            ->getResult(0);
    Value result = linalgIntBroadcastExtSIAdd(rewriter, loc, bias, matmul,
                                              biasEmptyTensor, indexingMaps);
    rewriter.replaceOp(op, result);
    return success();
  }
};

class MaxPool2dConverter : public OpRewritePattern<tosa::MaxPool2dOp> {
public:
  using OpRewritePattern<tosa::MaxPool2dOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::MaxPool2dOp op,
                                PatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    Value input = op.getInput();
    ShapedType inputTy = cast<ShapedType>(input.getType());

    ShapedType resultTy = cast<ShapedType>(op.getType());
    Type resultETy = inputTy.getElementType();

    auto dynamicDimsOr =
        checkHasDynamicBatchDims(rewriter, op, {input, op.getOutput()});
    if (!dynamicDimsOr.has_value())
      return failure();
    SmallVector<Value> dynamicDims = *dynamicDimsOr;

    // Determine what the initial value needs to be for the max pool op.
    TypedAttr initialAttr;
    if (resultETy.isF32())
      initialAttr = rewriter.getFloatAttr(
          resultETy, APFloat::getLargest(
                         cast<FloatType>(resultETy).getFloatSemantics(), true));

    if (isa<IntegerType>(resultETy))
      initialAttr = rewriter.getIntegerAttr(
          resultETy,
          APInt::getSignedMinValue(resultETy.getIntOrFloatBitWidth()));

    if (!initialAttr)
      return rewriter.notifyMatchFailure(
          op, "Unsupported initial value for tosa.maxpool_2d op");

    // Apply padding as necessary.
    llvm::SmallVector<int64_t> pad;
    pad.resize(2, 0);
    llvm::append_range(pad, op.getPad());
    pad.resize(pad.size() + 2, 0);
    Value paddedInput = applyPad(loc, input, pad, initialAttr, rewriter);

    Value initialValue = rewriter.create<arith::ConstantOp>(loc, initialAttr);

    ArrayRef<int64_t> kernel = op.getKernel();
    ArrayRef<int64_t> stride = op.getStride();

    Attribute strideAttr = rewriter.getI64VectorAttr(stride);
    Attribute dilationAttr = rewriter.getI64VectorAttr({1, 1});

    // Create the linalg op that performs pooling.
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultTy.getShape(), resultTy.getElementType(), dynamicDims);

    Value filledEmptyTensor =
        rewriter
            .create<linalg::FillOp>(loc, ValueRange{initialValue},
                                    ValueRange{emptyTensor})
            .result();

    Value fakeWindowDims =
        rewriter.create<tensor::EmptyOp>(loc, kernel, resultETy);

    rewriter.replaceOpWithNewOp<linalg::PoolingNhwcMaxOp>(
        op, ArrayRef<Type>{resultTy}, ValueRange{paddedInput, fakeWindowDims},
        filledEmptyTensor, strideAttr, dilationAttr);
    return success();
  }
};

class AvgPool2dConverter : public OpRewritePattern<tosa::AvgPool2dOp> {
public:
  using OpRewritePattern<tosa::AvgPool2dOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::AvgPool2dOp op,
                                PatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    Value input = op.getInput();
    ShapedType inputTy = cast<ShapedType>(input.getType());
    Type inElementTy = inputTy.getElementType();

    ShapedType resultTy = cast<ShapedType>(op.getType());
    Type resultETy = cast<ShapedType>(op.getType()).getElementType();

    Type accETy = op.getAccType();
    ShapedType accTy = resultTy.clone(accETy);

    auto dynamicDimsOr =
        checkHasDynamicBatchDims(rewriter, op, {input, op.getOutput()});
    if (!dynamicDimsOr.has_value())
      return failure();
    SmallVector<Value> dynamicDims = *dynamicDimsOr;

    // Apply padding as necessary.
    llvm::SmallVector<int64_t> pad;
    pad.resize(2, 0);
    llvm::append_range(pad, op.getPad());
    pad.resize(pad.size() + 2, 0);
    TypedAttr padAttr = rewriter.getZeroAttr(inElementTy);
    // Unsupported element type
    if (!padAttr)
      return failure();
    Value paddedInput = applyPad(loc, input, pad, padAttr, rewriter);

    auto initialAttr = rewriter.getZeroAttr(accETy);
    Value initialValue = rewriter.create<arith::ConstantOp>(loc, initialAttr);

    ArrayRef<int64_t> kernel = op.getKernel();
    ArrayRef<int64_t> stride = op.getStride();

    Attribute strideAttr = rewriter.getI64VectorAttr(stride);
    Attribute dilationAttr = rewriter.getI64VectorAttr({1, 1});

    // Create the linalg op that performs pooling.
    Value poolEmptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, accTy.getShape(), accETy, dynamicDims);

    Value filledEmptyTensor =
        rewriter
            .create<linalg::FillOp>(loc, ValueRange{initialValue},
                                    ValueRange{poolEmptyTensor})
            .result();

    Value fakeWindowDims =
        rewriter.create<tensor::EmptyOp>(loc, kernel, accETy);

    // Sum across the pooled region.
    Value poolingOp = rewriter
                          .create<linalg::PoolingNhwcSumOp>(
                              loc, ArrayRef<Type>{accTy},
                              ValueRange{paddedInput, fakeWindowDims},
                              filledEmptyTensor, strideAttr, dilationAttr)
                          .getResult(0);

    // Normalize the summed value by the number of elements grouped in each
    // pool.
    Value iH = rewriter.create<tensor::DimOp>(loc, poolingOp, 1);
    Value iW = rewriter.create<tensor::DimOp>(loc, poolingOp, 2);

    auto one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    iH = rewriter.create<arith::SubIOp>(loc, iH, one);
    iW = rewriter.create<arith::SubIOp>(loc, iW, one);

    Value genericEmptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultTy.getShape(), resultETy, dynamicDims);

    auto affineMap = rewriter.getMultiDimIdentityMap(resultTy.getRank());
    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, ArrayRef<Type>({resultTy}), ValueRange{poolingOp},
        ValueRange{genericEmptyTensor},
        ArrayRef<AffineMap>({affineMap, affineMap}),
        getNParallelLoopsAttrs(resultTy.getRank()),
        [&](OpBuilder &b, Location loc, ValueRange args) {
          auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);

          // Determines what the portion of valid input is covered by the
          // kernel.
          auto padFn = [&](Value valid, Value pos, int64_t pad) -> Value {
            if (pad == 0)
              return valid;

            auto padVal = rewriter.create<arith::ConstantIndexOp>(loc, pad);
            Value dpos = rewriter.create<arith::SubIOp>(loc, pos, padVal);

            Value cmp = rewriter.create<arith::CmpIOp>(
                loc, arith::CmpIPredicate::slt, dpos, zero);
            Value offset =
                rewriter.create<arith::SelectOp>(loc, cmp, dpos, zero);
            return rewriter.create<arith::AddIOp>(loc, valid, offset)
                ->getResult(0);
          };

          auto coverageFn = [&](int64_t i, Value isize) -> Value {
            Value strideVal =
                rewriter.create<arith::ConstantIndexOp>(loc, stride[i - 1]);
            Value val =
                rewriter.create<arith::ConstantIndexOp>(loc, kernel[i - 1]);

            // Find the position relative to the input tensor's ends.
            Value left = rewriter.create<linalg::IndexOp>(loc, i);
            Value right = rewriter.create<arith::SubIOp>(loc, isize, left);
            left = rewriter.create<arith::MulIOp>(loc, left, strideVal);
            right = rewriter.create<arith::MulIOp>(loc, right, strideVal);

            // Determine how much padding was included.
            val = padFn(val, left, pad[i * 2]);
            val = padFn(val, right, pad[i * 2 + 1]);
            Value cmp = rewriter.create<arith::CmpIOp>(
                loc, arith::CmpIPredicate::slt, val, one);
            return rewriter.create<arith::SelectOp>(loc, cmp, one, val);
          };

          // Compute the indices from either end.
          Value kH3 = coverageFn(1, iH);
          Value kW3 = coverageFn(2, iW);

          // Compute the total number of elements and normalize.
          auto count = rewriter.create<arith::IndexCastOp>(
              loc, rewriter.getI32Type(),
              rewriter.create<arith::MulIOp>(loc, kH3, kW3));

          // Divide by the number of summed values. For floats this is just
          // a div however for quantized values input normalization had
          // to be applied.
          Value poolVal = args[0];
          if (isa<FloatType>(accETy)) {
            auto countF = rewriter.create<arith::SIToFPOp>(loc, accETy, count);
            poolVal = rewriter.create<arith::DivFOp>(loc, poolVal, countF)
                          ->getResult(0);
          } else {

            // If we have quantization information we need to apply an offset
            // for the input zp value.
            if (op.getQuantizationInfo()) {
              auto quantizationInfo = *op.getQuantizationInfo();
              auto inputZp = rewriter.create<arith::ConstantOp>(
                  loc, b.getIntegerAttr(accETy, quantizationInfo.getInputZp()));
              Value offset =
                  rewriter.create<arith::MulIOp>(loc, accETy, count, inputZp);
              poolVal =
                  rewriter.create<arith::SubIOp>(loc, accETy, poolVal, offset);
            }

            // Compute: k = 32 - count_leading_zeros(value - 1)
            Value one32 = rewriter.create<arith::ConstantOp>(
                loc, rewriter.getI32IntegerAttr(1));
            Value thirtyTwo32 = rewriter.create<arith::ConstantOp>(
                loc, rewriter.getI32IntegerAttr(32));

            Value countSubOne =
                rewriter.create<arith::SubIOp>(loc, count, one32);
            Value leadingZeros =
                rewriter.create<math::CountLeadingZerosOp>(loc, countSubOne);
            Value k =
                rewriter.create<arith::SubIOp>(loc, thirtyTwo32, leadingZeros);

            // Compute: numerator = ((1 << 30) + 1) << k
            Value k64 =
                rewriter.create<arith::ExtUIOp>(loc, rewriter.getI64Type(), k);
            Value thirtyShiftPlusOne = rewriter.create<arith::ConstantOp>(
                loc, rewriter.getI64IntegerAttr((1 << 30) + 1));
            Value numerator =
                rewriter.create<arith::ShLIOp>(loc, thirtyShiftPlusOne, k64);

            // Compute: scale.multiplier = numerator / value;
            Value count64 = rewriter.create<arith::ExtUIOp>(
                loc, rewriter.getI64Type(), count);
            Value multiplier =
                rewriter.create<arith::DivUIOp>(loc, numerator, count64);
            multiplier = rewriter.create<arith::TruncIOp>(
                loc, rewriter.getI32Type(), multiplier);

            // Compute: scale.shift = 30 + k
            Value k8 =
                rewriter.create<arith::TruncIOp>(loc, rewriter.getI8Type(), k);
            Value thirty8 = rewriter.create<arith::ConstantOp>(
                loc, rewriter.getI8IntegerAttr(30));
            Value shift = rewriter.create<arith::AddIOp>(loc, k8, thirty8);

            auto scaled =
                rewriter
                    .create<tosa::ApplyScaleOp>(loc, rewriter.getI32Type(),
                                                poolVal, multiplier, shift,
                                                rewriter.getBoolAttr(false))
                    .getResult();

            // If we have quantization information we need to apply output
            // zeropoint.
            if (op.getQuantizationInfo()) {
              auto quantizationInfo = *op.getQuantizationInfo();
              auto outputZp = rewriter.create<arith::ConstantOp>(
                  loc, b.getIntegerAttr(scaled.getType(),
                                        quantizationInfo.getOutputZp()));
              scaled = rewriter.create<arith::AddIOp>(loc, scaled, outputZp)
                           .getResult();
            }

            // Apply Clip.
            int64_t outBitwidth = resultETy.getIntOrFloatBitWidth();

            auto min = rewriter.create<arith::ConstantIntOp>(
                loc, APInt::getSignedMinValue(outBitwidth).getSExtValue(),
                accETy);
            auto max = rewriter.create<arith::ConstantIntOp>(
                loc, APInt::getSignedMaxValue(outBitwidth).getSExtValue(),
                accETy);
            auto clamp = clampIntHelper(loc, scaled, min, max, rewriter);

            poolVal = clamp;
            // Convert type.
            if (resultETy != clamp.getType()) {
              poolVal =
                  rewriter.create<arith::TruncIOp>(loc, resultETy, poolVal);
            }
          }

          rewriter.create<linalg::YieldOp>(loc, poolVal);
        });

    rewriter.replaceOp(op, genericOp.getResult(0));
    return success();
  }
};

} // namespace

void mlir::tosa::populateTosaToLinalgNamedConversionPatterns(
    RewritePatternSet *patterns) {
  patterns->add<
      // clang-format off
      ConvConverter<tosa::Conv2DOp, linalg::Conv2DNhwcFhwcOp, linalg::Conv2DNhwcFhwcQOp>,
      ConvConverter<tosa::Conv3DOp, linalg::Conv3DNdhwcDhwcfOp, linalg::Conv3DNdhwcDhwcfQOp>,
      DepthwiseConvConverter,
      MatMulConverter,
      MaxPool2dConverter,
      AvgPool2dConverter,
      FullyConnectedConverter>(patterns->getContext());
  // clang-format on
}
