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
#include "mlir/Dialect/Tosa/Utils/CoversionUtils.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <numeric>

using namespace mlir;
using namespace mlir::tosa;

static mlir::Value applyPad(Location loc, Value input, ArrayRef<int64_t> pad,
                            Attribute padAttr, OpBuilder &rewriter) {
  // Input should be padded if necessary.
  if (llvm::all_of(pad, [](int64_t p) { return p == 0; }))
    return input;

  ShapedType inputTy = input.getType().cast<ShapedType>();
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

  return tensor::createPadScalarOp(RankedTensorType::get(paddedShape, inputETy),
                                   input, padValue, lowIndices, highIndices,
                                   /*nofold=*/false, loc, rewriter)
      .getResult();
}

static mlir::Value reifyConstantDim(Attribute attr,
                                    ImplicitLocOpBuilder &builder) {
  return builder.createOrFold<arith::IndexCastOp>(
      builder.getIndexType(), builder.create<arith::ConstantOp>(attr));
}

// Calculating the output width/height using the formula:
// H = ((IH+pad_top+pad_bottom-(dilation_y*(KH-1)+1))/stride_y)+1
// W = ((IW+pad_left+pad_right-(dilation_x*(KW-1)+1))/stride_x)+1
static mlir::Value
getConvOutputDim(Location loc, Value inputDim, Attribute padBeforeAttr,
                 Attribute padAfterAttr, Value kernelDim, Attribute strideAttr,
                 Attribute dilationAttr, Type inputETy, OpBuilder &rewriter) {
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
    ArrayAttr padAttr, ArrayAttr strideAttr, ArrayAttr dilationAttr,
    int64_t weightHDim, int64_t weightWDim, OpBuilder &rewriter) {
  ShapedType inputTy = input.getType().cast<ShapedType>();
  Type inputETy = inputTy.getElementType();
  int64_t inputRank = inputTy.getRank();
  int64_t heightDim = 1;
  int64_t weightDim = 2;

  SmallVector<Value> dynDims;
  dynDims.resize(resultTy.getRank());
  for (int i = 0; i < inputRank; i++) {
    if (inputTy.isDynamicDim(i) && i != heightDim && i != weightDim)
      dynDims[i] = rewriter.create<tensor::DimOp>(loc, input, i);
  }

  // Dynamic input height
  if (inputTy.isDynamicDim(heightDim)) {
    Value initHDim =
        rewriter.create<tensor::DimOp>(loc, input, heightDim).getResult();
    Value kernelHDim =
        rewriter.create<tensor::DimOp>(loc, weight, weightHDim).getResult();
    // H = F(IH, pad_top, pad_bottom, dilation_y, KH, stride_y)
    dynDims[heightDim] = getConvOutputDim(
        loc, initHDim, padAttr.getValue()[0], padAttr.getValue()[1], kernelHDim,
        strideAttr.getValue()[0], dilationAttr.getValue()[0], inputETy,
        rewriter);
  }

  // Dynamic input weight
  if (inputTy.isDynamicDim(weightDim)) {
    Value initWDim =
        rewriter.create<tensor::DimOp>(loc, input, weightDim).getResult();
    Value kernelWDim =
        rewriter.create<tensor::DimOp>(loc, weight, weightWDim).getResult();
    // W = F(IW, pad_left, pad_right, dilation_x, KW, stride_x)
    dynDims[weightDim] = getConvOutputDim(
        loc, initWDim, padAttr.getValue()[2], padAttr.getValue()[3], kernelWDim,
        strideAttr.getValue()[1], dilationAttr.getValue()[1], inputETy,
        rewriter);
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

class ConvConverter : public OpConversionPattern<tosa::Conv2DOp> {
public:
  using OpConversionPattern<tosa::Conv2DOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tosa::Conv2DOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    Value input = op->getOperand(0);
    Value weight = op->getOperand(1);
    Value bias = op->getOperand(2);

    ShapedType inputTy = input.getType().cast<ShapedType>();
    ShapedType weightTy = weight.getType().cast<ShapedType>();
    ShapedType biasTy = bias.getType().cast<ShapedType>();
    ShapedType resultTy = op->getResult(0).getType().cast<ShapedType>();

    Type inputETy = inputTy.getElementType();
    Type resultETy = resultTy.getElementType();

    auto padAttr = op->getAttr("pad").cast<ArrayAttr>();
    auto strideTosaAttr = op->getAttr("stride").cast<ArrayAttr>();
    auto dilationTosaAttr = op->getAttr("dilation").cast<ArrayAttr>();
    bool isQuantized = op->hasAttr("quantization_info");

    if (!weightTy.hasStaticShape() || !biasTy.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "tosa.conv ops require static shapes for weight and bias");

    if (inputETy.isUnsignedInteger())
      return rewriter.notifyMatchFailure(
          op, "tosa.conv ops does not support unsigned integer input");

    SmallVector<Value> filteredDims = inferDynamicDimsForConv(
        loc, input, weight, resultTy, padAttr, strideTosaAttr, dilationTosaAttr,
        /*weightHDim=*/1, /*weightWDim=*/2, rewriter);

    auto weightShape = weightTy.getShape();

    // Apply padding as necessary.
    Attribute zeroAttr = rewriter.getZeroAttr(inputETy);
    if (isQuantized) {
      auto quantizationInfo =
          op->getAttr("quantization_info").cast<tosa::ConvOpQuantizationAttr>();
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
    getValuesFromIntArrayAttribute(padAttr, pad);
    pad.resize(pad.size() + 2, 0);
    input = applyPad(loc, input, pad, zeroAttr, rewriter);

    // Transpose the kernel to match dimension ordering of the linalg
    // convolution operation.
    // TODO(suderman): See if this can be efficiently folded - check whether
    // the input is used anywhere else, if not fold the constant.
    SmallVector<int64_t> weightPerm{1, 2, 3, 0};
    SmallVector<int64_t> newWeightShape{weightShape[1], weightShape[2],
                                        weightShape[3], weightShape[0]};
    auto weightPermAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({4}, rewriter.getI64Type()), weightPerm);
    Value weightPermValue =
        rewriter.create<arith::ConstantOp>(loc, weightPermAttr);
    Type newWeightTy =
        RankedTensorType::get(newWeightShape, weightTy.getElementType());
    weight = rewriter.create<tosa::TransposeOp>(loc, newWeightTy, weight,
                                                weightPermValue);

    Attribute resultZeroAttr = rewriter.getZeroAttr(resultETy);
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultTy.getShape(), resultETy, filteredDims);
    Value zero = rewriter.create<arith::ConstantOp>(loc, resultZeroAttr);
    Value zeroTensor = rewriter
                           .create<linalg::FillOp>(loc, ValueRange{zero},
                                                   ValueRange{emptyTensor})
                           .result();

    // Extract the attributes for convolution.
    llvm::SmallVector<int64_t> stride, dilation;
    getValuesFromIntArrayAttribute(strideTosaAttr, stride);
    getValuesFromIntArrayAttribute(dilationTosaAttr, dilation);

    // Create the convolution op.
    auto strideAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()), stride);
    auto dilationAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()), dilation);

    // Create maps for the bias broadcasting
    SmallVector<AffineMap, 4> indexingMaps;
    indexingMaps.push_back(AffineMap::get(
        /*dimCount=*/resultTy.getRank(), /*symbolCount=*/0,
        {rewriter.getAffineDimExpr(3)}, rewriter.getContext()));
    indexingMaps.push_back(rewriter.getMultiDimIdentityMap(resultTy.getRank()));
    indexingMaps.push_back(rewriter.getMultiDimIdentityMap(resultTy.getRank()));

    Value biasEmptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultTy.getShape(), resultETy, filteredDims);

    if (isQuantized) {
      auto quantizationInfo =
          op->getAttr("quantization_info").cast<tosa::ConvOpQuantizationAttr>();
      auto iZp = rewriter.getI32IntegerAttr(quantizationInfo.getInputZp());
      auto kZp = rewriter.getI32IntegerAttr(quantizationInfo.getWeightZp());

      auto iZpVal = rewriter.create<arith::ConstantOp>(loc, iZp);
      auto kZpVal = rewriter.create<arith::ConstantOp>(loc, kZp);
      Value conv =
          rewriter
              .create<linalg::Conv2DNhwcHwcfQOp>(
                  loc, resultTy, ValueRange{input, weight, iZpVal, kZpVal},
                  ValueRange{zeroTensor}, strideAttr, dilationAttr)
              ->getResult(0);

      Value result =
          rewriter
              .create<linalg::GenericOp>(
                  loc, resultTy, ValueRange({bias, conv}), biasEmptyTensor,
                  indexingMaps, getNParallelLoopsAttrs(resultTy.getRank()),
                  [&](OpBuilder &nestedBuilder, Location nestedLoc,
                      ValueRange args) {
                    Value added = nestedBuilder.create<arith::AddIOp>(
                        loc, args[0], args[1]);
                    nestedBuilder.create<linalg::YieldOp>(nestedLoc, added);
                  })
              .getResult(0);
      rewriter.replaceOp(op, result);
      return success();
    }

    Value conv = rewriter
                     .create<linalg::Conv2DNhwcHwcfOp>(
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

    ShapedType inputTy = input.getType().cast<ShapedType>();
    ShapedType weightTy = weight.getType().cast<ShapedType>();
    ShapedType biasTy = bias.getType().cast<ShapedType>();
    ShapedType resultTy = op->getResult(0).getType().cast<ShapedType>();
    int64_t resultRank = resultTy.getRank();

    Type inputETy = inputTy.getElementType();
    Type resultETy = resultTy.getElementType();

    auto padAttr = op->getAttr("pad").cast<ArrayAttr>();
    auto strideTosaAttr = op->getAttr("stride").cast<ArrayAttr>();
    auto dilationTosaAttr = op->getAttr("dilation").cast<ArrayAttr>();

    if (!weightTy.hasStaticShape() || !biasTy.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "tosa.depthwise_conv ops require static shapes");

    // Compute output dynamic dims
    SmallVector<Value> filteredDims = inferDynamicDimsForConv(
        loc, input, weight, resultTy, padAttr, strideTosaAttr, dilationTosaAttr,
        0, 1, rewriter);

    bool isQuantized = op->hasAttr("quantization_info");
    IntegerAttr iZp;
    IntegerAttr kZp;
    if (isQuantized) {
      auto quantizationInfo =
          op->getAttr("quantization_info").cast<tosa::ConvOpQuantizationAttr>();
      iZp = rewriter.getI32IntegerAttr(quantizationInfo.getInputZp());
      kZp = rewriter.getI32IntegerAttr(quantizationInfo.getWeightZp());
    }

    auto weightShape = weightTy.getShape();
    auto resultShape = resultTy.getShape();

    // Apply padding as necessary.
    Attribute zeroAttr = rewriter.getZeroAttr(inputETy);
    if (isQuantized) {
      auto quantizationInfo =
          op->getAttr("quantization_info").cast<tosa::ConvOpQuantizationAttr>();
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
    getValuesFromIntArrayAttribute(padAttr, pad);
    pad.resize(pad.size() + 2, 0);

    input = applyPad(loc, input, pad, zeroAttr, rewriter);

    // Extract the attributes for convolution.
    llvm::SmallVector<int64_t> stride, dilation;
    getValuesFromIntArrayAttribute(strideTosaAttr, stride);
    getValuesFromIntArrayAttribute(dilationTosaAttr, dilation);

    // Create the convolution op.
    auto strideAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()), stride);
    auto dilationAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()), dilation);
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

    Attribute resultZeroAttr = rewriter.getZeroAttr(resultETy);
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
      Value result =
          rewriter
              .create<linalg::GenericOp>(
                  loc, resultTy, ValueRange({bias, convReshape}),
                  biasEmptyTensor, indexingMaps,
                  getNParallelLoopsAttrs(resultRank),
                  [&](OpBuilder &nestedBuilder, Location nestedLoc,
                      ValueRange args) {
                    Value added = nestedBuilder.create<arith::AddIOp>(
                        loc, args[0], args[1]);
                    nestedBuilder.create<linalg::YieldOp>(nestedLoc, added);
                  })
              .getResult(0);
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

    auto outputTy = op.getType().cast<ShapedType>();
    auto outputElementTy = outputTy.getElementType();

    auto firstOperandTy = op->getOperand(0).getType().cast<ShapedType>();
    auto secondOperandTy = op->getOperand(1).getType().cast<ShapedType>();

    SmallVector<Value> dynDims;
    dynDims.resize(op->getResult(0).getType().cast<ShapedType>().getRank());

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
    auto outputTy = op.getType().cast<ShapedType>();
    auto input = op.getInput();
    auto inputTy = input.getType().cast<ShapedType>();

    auto bias = op.getBias();

    auto weight = op.getWeight();
    auto weightTy = weight.getType().cast<ShapedType>();
    auto weightShape = weightTy.getShape();

    auto outputETy = outputTy.getElementType();

    SmallVector<Value> dynDims;
    dynDims.resize(op->getResult(0).getType().cast<ShapedType>().getRank());

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
    Attribute resultZeroAttr = rewriter.getZeroAttr(outputETy);
    Value zero = rewriter.create<arith::ConstantOp>(loc, resultZeroAttr);
    Value zeroTensor = rewriter
                           .create<linalg::FillOp>(loc, ValueRange{zero},
                                                   ValueRange{emptyTensor})
                           .result();

    SmallVector<int64_t> permutation{1, 0};
    auto permutationAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()), permutation);
    Value permutationValue =
        rewriter.create<arith::ConstantOp>(loc, permutationAttr);

    SmallVector<int64_t> newWeightShape{weightShape[1], weightShape[0]};
    Type newWeightTy =
        RankedTensorType::get(newWeightShape, weightTy.getElementType());

    Value transposedWeight = rewriter.create<tosa::TransposeOp>(
        loc, newWeightTy, weight, permutationValue);

    auto biasEmptyTensor =
        rewriter
            .create<tensor::EmptyOp>(loc, outputTy.getShape(), outputETy,
                                     filteredDims)
            ->getResults();

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
    Value result =
        rewriter
            .create<linalg::GenericOp>(
                loc, outputTy, ValueRange({bias, matmul}), biasEmptyTensor,
                indexingMaps, getNParallelLoopsAttrs(outputTy.getRank()),
                [&](OpBuilder &nestedBuilder, Location nestedLoc,
                    ValueRange args) {
                  Value added = nestedBuilder.create<arith::AddIOp>(
                      loc, args[0], args[1]);
                  nestedBuilder.create<linalg::YieldOp>(nestedLoc, added);
                })
            .getResult(0);
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
    ShapedType inputTy = input.getType().cast<ShapedType>();

    ShapedType resultTy = op.getType().template cast<ShapedType>();
    Type resultETy = inputTy.getElementType();

    auto dynamicDimsOr =
        checkHasDynamicBatchDims(rewriter, op, {input, op.getOutput()});
    if (!dynamicDimsOr.has_value())
      return failure();
    SmallVector<Value> dynamicDims = dynamicDimsOr.value();

    // Determine what the initial value needs to be for the max pool op.
    Attribute initialAttr;
    if (resultETy.isF32())
      initialAttr = rewriter.getFloatAttr(
          resultETy,
          APFloat::getLargest(resultETy.cast<FloatType>().getFloatSemantics(),
                              true));

    if (resultETy.isa<IntegerType>())
      initialAttr = rewriter.getIntegerAttr(
          resultETy,
          APInt::getSignedMinValue(resultETy.getIntOrFloatBitWidth()));

    if (!initialAttr)
      return rewriter.notifyMatchFailure(
          op, "Unsupported initial value for tosa.maxpool_2d op");

    // Apply padding as necessary.
    llvm::SmallVector<int64_t> pad;
    pad.resize(2, 0);
    getValuesFromIntArrayAttribute(op.getPad(), pad);
    pad.resize(pad.size() + 2, 0);
    Value paddedInput = applyPad(loc, input, pad, initialAttr, rewriter);

    Value initialValue = rewriter.create<arith::ConstantOp>(loc, initialAttr);

    SmallVector<int64_t> kernel, stride;
    getValuesFromIntArrayAttribute(op.getKernel(), kernel);
    getValuesFromIntArrayAttribute(op.getStride(), stride);

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
    ShapedType inputTy = input.getType().cast<ShapedType>();
    Type inElementTy = inputTy.getElementType();

    ShapedType resultTy = op.getType().template cast<ShapedType>();
    Type resultETy = op.getType().cast<ShapedType>().getElementType();

    Type accETy =
        inElementTy.isa<IntegerType>() ? rewriter.getI32Type() : inElementTy;
    ShapedType accTy = resultTy.clone(accETy);

    auto dynamicDimsOr =
        checkHasDynamicBatchDims(rewriter, op, {input, op.getOutput()});
    if (!dynamicDimsOr.has_value())
      return failure();
    SmallVector<Value> dynamicDims = dynamicDimsOr.value();

    // Apply padding as necessary.
    llvm::SmallVector<int64_t> pad;
    pad.resize(2, 0);
    getValuesFromIntArrayAttribute(op.getPad(), pad);
    pad.resize(pad.size() + 2, 0);
    Attribute padAttr = rewriter.getZeroAttr(inElementTy);
    Value paddedInput = applyPad(loc, input, pad, padAttr, rewriter);

    Attribute initialAttr = rewriter.getZeroAttr(accETy);
    Value initialValue = rewriter.create<arith::ConstantOp>(loc, initialAttr);

    SmallVector<int64_t> kernel, stride;
    getValuesFromIntArrayAttribute(op.getKernel(), kernel);
    getValuesFromIntArrayAttribute(op.getStride(), stride);

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
    auto poolingOpTy = poolingOp.getType().cast<ShapedType>();
    auto affineMap = rewriter.getMultiDimIdentityMap(resultTy.getRank());

    Value genericEmptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultTy.getShape(), resultETy, dynamicDims);

    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, ArrayRef<Type>({resultTy}), ValueRange{poolingOp},
        ValueRange{genericEmptyTensor},
        ArrayRef<AffineMap>({affineMap, affineMap}),
        getNParallelLoopsAttrs(resultTy.getRank()),
        [&](OpBuilder &b, Location loc, ValueRange args) {
          auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
          auto one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
          auto iH = rewriter.create<arith::ConstantIndexOp>(
              loc, poolingOpTy.getDimSize(1) - 1);
          auto iW = rewriter.create<arith::ConstantIndexOp>(
              loc, poolingOpTy.getDimSize(2) - 1);

          // Compute the indices from either end.
          auto y0 = rewriter.create<linalg::IndexOp>(loc, 1);
          auto x0 = rewriter.create<linalg::IndexOp>(loc, 2);
          auto y1 = rewriter.create<arith::SubIOp>(loc, iH, y0);
          auto x1 = rewriter.create<arith::SubIOp>(loc, iW, x0);

          // Determines what the portion of valid input is covered by the
          // kernel.
          auto padFn = [&](Value v, Value x, int64_t pad) -> Value {
            if (pad == 0)
              return v;

            auto padVal = rewriter.create<arith::ConstantIndexOp>(loc, pad);
            Value dx = rewriter.create<arith::SubIOp>(loc, x, padVal);

            Value cmp = rewriter.create<arith::CmpIOp>(
                loc, arith::CmpIPredicate::slt, dx, zero);
            Value offset = rewriter.create<arith::SelectOp>(loc, cmp, dx, zero);
            return rewriter.create<arith::AddIOp>(loc, v, offset)->getResult(0);
          };

          // Compute the vertical component of coverage.
          auto kH0 = rewriter.create<arith::ConstantIndexOp>(loc, kernel[0]);
          auto kH1 = padFn(kH0, y0, pad[2]);
          auto kH2 = padFn(kH1, y1, pad[3]);
          auto kHCmp = rewriter.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::slt, kH2, one);
          auto kH3 = rewriter.create<arith::SelectOp>(loc, kHCmp, one, kH2);

          // compute the horizontal component of coverage.
          auto kW0 = rewriter.create<arith::ConstantIndexOp>(loc, kernel[1]);
          auto kW1 = padFn(kW0, x0, pad[4]);
          auto kW2 = padFn(kW1, x1, pad[5]);
          auto kWCmp = rewriter.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::slt, kW2, one);
          auto kW3 = rewriter.create<arith::SelectOp>(loc, kWCmp, one, kW2);

          // Compute the total number of elements and normalize.
          Value count = rewriter.create<arith::MulIOp>(loc, kH3, kW3);
          auto countI = rewriter.create<arith::IndexCastOp>(
              loc, rewriter.getI32Type(), count);

          // Divide by the number of summed values. For floats this is just
          // a div however for quantized values input normalization had
          // to be applied.
          Value poolVal = args[0];
          if (accETy.isa<FloatType>()) {
            auto countF = rewriter.create<arith::SIToFPOp>(loc, accETy, countI);
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
                  rewriter.create<arith::MulIOp>(loc, accETy, countI, inputZp);
              poolVal =
                  rewriter.create<arith::SubIOp>(loc, accETy, poolVal, offset);
            }

            // Compute the multiplier and shift values for the quantization
            // normalization. Preferably we would want to compute more bits
            // however 32-bits should be enough for compute. Honestly we
            // should probably straight divide.
            int64_t numerator = ((1 << 30) + 1);
            int64_t shift = 30;

            Value numeratorVal = rewriter.create<arith::ConstantOp>(
                loc, rewriter.getI32IntegerAttr(numerator));
            Value multiplierVal =
                rewriter
                    .create<arith::DivUIOp>(loc, rewriter.getI32Type(),
                                            numeratorVal, countI)
                    .getResult();
            Value shiftVal = rewriter.create<arith::ConstantOp>(
                loc, rewriter.getI8IntegerAttr(shift));

            auto scaled =
                rewriter
                    .create<tosa::ApplyScaleOp>(
                        loc, rewriter.getI32Type(), poolVal, multiplierVal,
                        shiftVal, rewriter.getBoolAttr(false))
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
      ConvConverter,
      DepthwiseConvConverter,
      MatMulConverter,
      MaxPool2dConverter,
      AvgPool2dConverter,
      FullyConnectedConverter>(patterns->getContext());
  // clang-format on
}
