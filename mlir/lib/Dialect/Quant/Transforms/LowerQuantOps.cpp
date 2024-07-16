//===- LowerQuantOps.cpp - Lower 'quant' dialect ops ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Transforms `quant.dcast` and `quant.qcast` into lower-level ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/Dialect/Quant/Transforms/Passes.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace quant {

#define GEN_PASS_DEF_LOWERQUANTOPS
#include "mlir/Dialect/Quant/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// DequantizeCastOp
//===----------------------------------------------------------------------===//

class DequantizeCastOpConversion : public OpConversionPattern<quant::DequantizeCastOp> {
public:
  using OpConversionPattern<quant::DequantizeCastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(quant::DequantizeCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return success();
  }
};


//===----------------------------------------------------------------------===//
// QuantizeCastOp
//===----------------------------------------------------------------------===//

// If 'inputType' is a tensor, return its element type. If it is a scalar,
// return it as is.
Type getScalarType(Type inputType) {
  if (auto tensorType = dyn_cast<TensorType>(inputType))
    return tensorType.getElementType();
  return inputType;
}

// Return the shape of an input value as a list of attributes (static dimensions)
// and values (dynamic dimensions). If 'input' is a scalar, an empty list is
// returned. If 'input' is a tensor, its shape is returned.
SmallVector<OpFoldResult>
getScalarOrTensorShape(OpBuilder &builder, Location loc, Value input) {
  if (isa<TensorType>(input.getType()))
    return tensor::getMixedSizes(builder, loc, input);
  return {};
}

// If 'referenceType' is a scalar, return 'elementType' as is. If
// 'referenceType' is a tensor, return another tensor with the same shape and
// elements of type 'elementType'.
Type getScalarOrTensorType(Type elementType, Type referenceType) {
  if (auto tensorType = dyn_cast<TensorType>(referenceType))
    return tensorType.clone(elementType);
  return elementType;
}

// Return a constant with the given value. If 'referenceType' is a tensor, a
// tensor splat of shape 'referenceShape' is returned. If 'referenceType' is a
// scalar, 'referenceShape' is ignored and a scalar constant is returned.
Value getScalarOrTensorConstant(OpBuilder &builder, Location loc, Value scalar,
                                Type referenceType,
                                ArrayRef<OpFoldResult> referenceShape) {
  // If the result type is a scalar, return the unmodified scalar constant.
  auto tensorType = dyn_cast<TensorType>(referenceType);
  if (!tensorType) {
    assert(referenceShape.empty());
    return scalar;
  }

  // Create tensor splat
  auto tensorConstant =
      builder.create<tensor::SplatOp>(loc, scalar, referenceShape);
  return tensorConstant;
}

std::pair<Value, Value> flattenUnrankedTensor(OpBuilder &builder, Location loc,
                                              Value input) {
  // Get unranked input shape and total size
  auto *context = builder.getContext();
  auto shapeType = shape::getExtentTensorType(context);
  auto inputShape = builder.create<shape::ShapeOfOp>(loc, shapeType, input);
  Value inputSize = builder.create<shape::NumElementsOp>(
      loc, builder.getIndexType(), inputShape);

  // Turn input size into 1D tensor
  auto flatShapeType = shape::getExtentTensorType(context, 1);
  auto flatInputShape = builder.create<tensor::FromElementsOp>(
      loc, flatShapeType, inputSize);

  // Reshape input tensor into 1D
  auto inputType = cast<UnrankedTensorType>(input.getType());
  auto elementType = inputType.getElementType();
  auto flatInputType =
      RankedTensorType::get({ShapedType::kDynamic}, elementType);
  auto flatInput = builder.create<tensor::ReshapeOp>(
      loc, flatInputType, input, flatInputShape);
  return std::make_pair(flatInput, inputShape);
}

std::pair<Value, Value> flattenUnrankedTensorAroundAxis(OpBuilder &builder,
                                                        Location loc,
                                                        Value input,
                                                        int64_t axis,
                                                        int64_t axisSize) {
  // Get full tensor shape
  auto *context = builder.getContext();
  auto indexType = builder.getIndexType();
  auto shapeType = shape::getExtentTensorType(context);
  auto inputShape = builder.create<shape::ShapeOfOp>(loc, shapeType, input);

  // Get shape and sizes on left and right of axis
  auto axisValue = builder.create<arith::ConstantIndexOp>(loc, axis);
  auto axisNextValue = builder.create<arith::ConstantIndexOp>(loc, axis + 1);
  auto shapeLeft = builder.create<shape::SplitAtOp>(
      loc, TypeRange{shapeType, shapeType}, inputShape, axisValue)
      .getResult(0);
  auto sizeLeft = builder.create<shape::NumElementsOp>(
      loc, indexType, shapeLeft);
  auto shapeRight = builder.create<shape::SplitAtOp>(
      loc, TypeRange{shapeType, shapeType}, inputShape, axisNextValue)
      .getResult(1);
  auto sizeRight = builder.create<shape::NumElementsOp>(
      loc, indexType, shapeRight);

  // Compute flat input shape as a 3-element 1D tensor
  auto axisSizeValue = builder.create<arith::ConstantIndexOp>(loc, axisSize);
  auto flatShapeType = shape::getExtentTensorType(context, 3);
  auto flatInputShape = builder.create<tensor::FromElementsOp>(
      loc, flatShapeType, ValueRange{sizeLeft, axisSizeValue, sizeRight});

  // Reshape input to 3D tensor
  auto inputType = cast<UnrankedTensorType>(input.getType());
  auto elementType = inputType.getElementType();
  auto flatInputType = RankedTensorType::get(
      {ShapedType::kDynamic, axisSize, ShapedType::kDynamic}, elementType);
  auto flatInput = builder.create<tensor::ReshapeOp>(
      loc, flatInputType, input, flatInputShape);

  return std::make_pair(flatInput, inputShape);
}

Value restoreUnrankedTensorShape(OpBuilder &builder, Location loc, Value input,
                                 Value inputShape) {
  auto inputType = cast<RankedTensorType>(input.getType());
  auto elementType = inputType.getElementType();
  auto unrankedType = UnrankedTensorType::get(elementType);
  return builder.create<tensor::ReshapeOp>(loc, unrankedType, input, inputShape);
}

Value materializePerChannelScales(OpBuilder &builder, Location loc,
                                  UniformQuantizedPerAxisType quantizedType) {
  auto scales = quantizedType.getScales();
  auto expressedType = quantizedType.getExpressedType();
  auto scaleAttrs = llvm::map_to_vector(scales, [&](double scale) -> Attribute {
    return builder.getFloatAttr(expressedType, scale);
  });
  auto tensorType = RankedTensorType::get({(int64_t) scales.size()}, expressedType);
  auto scalesAttr = DenseElementsAttr::get(tensorType, scaleAttrs);
  return builder.create<arith::ConstantOp>(loc, tensorType, scalesAttr);
}

Value materializePerChannelZeroPoints(OpBuilder &builder, Location loc,
                            UniformQuantizedPerAxisType quantizedType) {
  auto zeroPoints = quantizedType.getZeroPoints();
  auto storageType = quantizedType.getStorageType();
  auto zeroPointAttrs = llvm::map_to_vector(
      zeroPoints,
      [&](int64_t zeroPoint) -> Attribute {
        return builder.getIntegerAttr(storageType, zeroPoint);
      });
  auto tensorType =
      RankedTensorType::get({(int64_t)zeroPoints.size()}, storageType);
  auto zeroPointsAttr = DenseElementsAttr::get(tensorType, zeroPointAttrs);
  return builder.create<arith::ConstantOp>(loc, tensorType, zeroPointsAttr);
}

Value clampScalarOrTensor(OpBuilder &builder, Location loc, Value input,
                          ArrayRef<OpFoldResult> inputShape,
                          QuantizedType quantizedType) {
  // If quantized type does not narrow down the storage type range, there is
  // nothing to do.
  if (!quantizedType.hasStorageTypeBounds())
    return input;

  // Materialize bounds
  auto inputType = input.getType();
  auto storageType = quantizedType.getStorageType();
  auto storageMinScalar = builder.create<arith::ConstantIntOp>(
      loc, quantizedType.getStorageTypeMin(), storageType);
  auto storageMaxScalar = builder.create<arith::ConstantIntOp>(
      loc, quantizedType.getStorageTypeMax(), storageType);
  auto storageMin = getScalarOrTensorConstant(builder, loc, storageMinScalar,
                                              inputType, inputShape);
  auto storageMax = getScalarOrTensorConstant(builder, loc, storageMaxScalar,
                                              inputType, inputShape);

  // Clamp
  if (quantizedType.isSigned()) {
    input = builder.create<arith::MaxSIOp>(loc, input, storageMin);
    input = builder.create<arith::MinSIOp>(loc, input, storageMax);
  } else {
    input = builder.create<arith::MaxUIOp>(loc, input, storageMin);
    input = builder.create<arith::MinUIOp>(loc, input, storageMax);
  }
  return input;
}

Value convertFloatToInteger(OpBuilder &builder, Location loc, Value input,
                            Type resultType, bool isSigned) {
  if (isSigned)
    return builder.create<arith::FPToSIOp>(loc, resultType, input);
  return builder.create<arith::FPToUIOp>(loc, resultType, input);
}

Value convertIntegerToFloat(OpBuilder &builder, Location loc, Value input,
                            Type resultType, bool isSigned) {
  if (isSigned)
    return builder.create<arith::SIToFPOp>(loc, resultType, input);
  return builder.create<arith::UIToFPOp>(loc, resultType, input);
}

// Quantize a floating-point input using the given scale, input shape, and
// storage type bounds in the given quantized type.
Value quantizeScalarOrTensor(OpBuilder &builder, Location loc, Value input,
                             ArrayRef<OpFoldResult> inputShape, Value scale,
                             Value zeroPoint, QuantizedType quantizedType) {
  // Convert scale and zero point to tensors if necessary
  auto inputType = input.getType();
  scale = getScalarOrTensorConstant(
      builder, loc, scale, inputType, inputShape);
  zeroPoint = getScalarOrTensorConstant(
      builder, loc, zeroPoint, inputType, inputShape);

  // Convert zero point from storage to expressed type
  auto expressedScalarOrTensorType =
      getScalarOrTensorType(quantizedType.getExpressedType(), inputType);
  zeroPoint = convertIntegerToFloat(builder, loc, zeroPoint,
                                    expressedScalarOrTensorType,
                                    quantizedType.isSigned());

  // Scale input and add zero point
  auto scaledValue = builder.create<arith::DivFOp>(loc, input, scale);
  auto storedValueAsExpressedType =
      builder.create<arith::AddFOp>(loc, scaledValue, zeroPoint);

  // Convert to storage type
  auto storageScalarOrTensorType =
      getScalarOrTensorType(quantizedType.getStorageType(), inputType);
  auto storedValue = convertFloatToInteger(
      builder, loc, storedValueAsExpressedType, storageScalarOrTensorType,
      quantizedType.isSigned());

  // Clamp stored value it if the storage type is bound
  storedValue =
      clampScalarOrTensor(builder, loc, storedValue, inputShape, quantizedType);
  return storedValue;
}

class QuantizeCastOpConversion : public OpConversionPattern<quant::QuantizeCastOp> {

  Value convertPerLayerRanked(OpBuilder &builder, Location loc, Value input,
                              UniformQuantizedType quantizedType) const {

    // Create scale and zero point constants
    auto expressedType = quantizedType.getExpressedType();
    auto storageType = quantizedType.getStorageType();
    auto scaleAttr =
        builder.getFloatAttr(expressedType, quantizedType.getScale());
    auto scale =
        builder.create<arith::ConstantOp>(loc, expressedType, scaleAttr);
    auto zeroPointAttr =
        builder.getIntegerAttr(storageType, quantizedType.getZeroPoint());
    auto zeroPoint =
        builder.create<arith::ConstantOp>(loc, storageType, zeroPointAttr);

    auto inputShape = getScalarOrTensorShape(builder, loc, input);
    return quantizeScalarOrTensor(builder, loc, input, inputShape, scale,
                                  zeroPoint, quantizedType);
  }

  Value convertPerLayer(OpBuilder &builder, Location loc, Value input,
                        UniformQuantizedType quantizedType) const {
    // Flatten input if unranked
    bool isUnranked = isa<UnrankedTensorType>(input.getType());
    Value inputShape;
    if (isUnranked)
      std::tie(input, inputShape) = flattenUnrankedTensor(builder, loc, input);

    // Process ranked tensor
    auto result = convertPerLayerRanked(builder, loc, input, quantizedType);

    // Restore original shape if unranked
    if (isUnranked)
      result = restoreUnrankedTensorShape(builder, loc, result, inputShape);

    return result;
  }

  Value convertPerChannelRanked(OpBuilder &builder, Location loc, Value input,
                                UniformQuantizedPerAxisType quantizedType,
                                int64_t channelAxis) const {
    auto *context = builder.getContext();

    auto inputType = cast<RankedTensorType>(input.getType());
    auto inputRank = inputType.getRank();

    auto scales = materializePerChannelScales(builder, loc, quantizedType);
    auto zeroPoints =
        materializePerChannelZeroPoints(builder, loc, quantizedType);

    auto storageType = quantizedType.getStorageType();
    auto initShape = tensor::getMixedSizes(builder, loc, input);
    Value init = builder.create<tensor::EmptyOp>(loc, initShape, storageType);

    SmallVector<utils::IteratorType> iteratorTypes(
        inputRank, utils::IteratorType::parallel);
    auto channelAxisAffineMap = AffineMap::get(
        inputRank, 0, builder.getAffineDimExpr(channelAxis), context);
    SmallVector<AffineMap> indexingMaps{
      builder.getMultiDimIdentityMap(inputRank),
      channelAxisAffineMap,
      channelAxisAffineMap,
      builder.getMultiDimIdentityMap(inputRank)
    };
    auto storedValue = builder.create<linalg::GenericOp>(
        loc,
        init.getType(),  // resultType
        ValueRange{input, scales, zeroPoints},  // inputs
        ValueRange{init},  // outputs
        indexingMaps,
        iteratorTypes,
        [&](OpBuilder& builder, Location loc, ValueRange args) {
          assert(args.size() == 4);
          auto expressedValue = args[0];
          auto scale = args[1];
          auto zeroPoint = args[2];

          auto result = quantizeScalarOrTensor(builder, loc, expressedValue, {},
                                               scale, zeroPoint, quantizedType);

          builder.create<linalg::YieldOp>(loc, result);
        })
        .getResult(0);

    return storedValue;
  }

  Value convertPerChannel(OpBuilder &builder, Location loc, Value input,
                          UniformQuantizedPerAxisType quantizedType) const {
    // Flatten unranked tensor into a 3D ranked tensor if necessary
    bool isUnranked = isa<UnrankedTensorType>(input.getType());
    int64_t channelAxis = quantizedType.getQuantizedDimension();
    int64_t channelAxisSize = (int64_t) quantizedType.getScales().size();
    Value inputShape;
    if (isUnranked) {
      std::tie(input, inputShape) = flattenUnrankedTensorAroundAxis(
          builder, loc, input, channelAxis, channelAxisSize);
      channelAxis = 1;
    }

    // Work on a ranked tensor
    auto result = convertPerChannelRanked(builder, loc, input, quantizedType,
                                          channelAxis);

    // Restore original tensor shape if unranked
    if (isUnranked)
      result = restoreUnrankedTensorShape(builder, loc, result, inputShape);

    return result;
  }

public:
  using OpConversionPattern<quant::QuantizeCastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(quant::QuantizeCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto input = op.getInput();
    auto resultScalarType = getScalarType(op.getResult().getType());

    // Flatten unranked tensor input
    Value result;
    if (auto quantizedType = dyn_cast<UniformQuantizedType>(resultScalarType)) {
      result = convertPerLayer(rewriter, loc, input, quantizedType);
    } else if (auto quantizedType =
                   dyn_cast<UniformQuantizedPerAxisType>(resultScalarType)) {
      result = convertPerChannel(rewriter, loc, input, quantizedType);
    } else {
      llvm_unreachable("unexpected uniform quantized type");
    }

    // Cast stored value to result quantized value
    rewriter.replaceOpWithNewOp<quant::StorageCastOp>(
        op, op.getResult().getType(), result);
    return success();
  }
};

struct LowerQuantOps : public impl::LowerQuantOpsBase<LowerQuantOps> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateLowerQuantOpsPatterns(patterns);

    ConversionTarget target(getContext());
    target.addLegalOp<quant::StorageCastOp>();
    target.addIllegalDialect<quant::QuantDialect>();
    target.addLegalDialect<
      arith::ArithDialect,
      linalg::LinalgDialect,
      shape::ShapeDialect,
      tensor::TensorDialect
    >();

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

void populateLowerQuantOpsPatterns(RewritePatternSet &patterns) {
  patterns.add<
    DequantizeCastOpConversion,
    QuantizeCastOpConversion
  >(patterns.getContext());
}

} // namespace quant
} // namespace mlir
