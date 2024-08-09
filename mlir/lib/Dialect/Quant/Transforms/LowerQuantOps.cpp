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
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace quant {

#define GEN_PASS_DEF_LOWERQUANTOPS
#include "mlir/Dialect/Quant/Transforms/Passes.h.inc"

namespace {

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

// Reshape an unranked tensor into a 1D ranked tensor.
//
// - input
//   Unranked tensor.
//
// Return values:
//
// - flatInput
//   1D ranked, dynamically shaped tensor.
//
// - inputShape
//   1D extent tensor containing the shape of the original unranked input.
//
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

// Reshape an unranked tensor into a 3D ranked tensor where the central
// dimension of the result tensor corresponds to dimension 'axis' of the input
// tensor.
//
// - input
//   Unranked tensor.
//
// - axis
//   Index of the input dimension around which other input dimiensions will be
//   collapsed.
//
// - axisSize
//   Size of input dimension 'axis'.
//
// Return values:
//
// - flatInput
//   3D ranked tensor of shape [?, axisSize, ?].
//
// - inputShape
//   1D extent tensor containing the shape of the original unranked input.
//
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

// Reshape an input tensor into its original unranked shape.
//
// - input
//   Ranked tensor.
//
// - inputShape
//   1D extent tensor.
//
Value restoreUnrankedTensorShape(OpBuilder &builder, Location loc, Value input,
                                 Value inputShape) {
  auto inputType = cast<RankedTensorType>(input.getType());
  auto elementType = inputType.getElementType();
  auto unrankedType = UnrankedTensorType::get(elementType);
  return builder.create<tensor::ReshapeOp>(loc, unrankedType, input, inputShape);
}

// Create a tensor constant containing all scales in a per-channel quantized
// type. Example:
//
//   !quant.uniform<i8:f32:1, {2.0:10, 3.0:20}>
//
// produces
//
//   %cst = arith.constant dense<[2.0, 3.0]> : tensor<2xf32>
//
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

// Create a tensor constant containing all zero points in a per-channel
// quantized type. Example:
//
//   !quant.uniform<i8:f32:1, {2.0:10, 3.0:20}>
//
// produces
//
//   %cst = arith.constant dense<[10, 20]> : tensor<2xi8>
//
Value materializePerChannelZeroPoints(
    OpBuilder &builder, Location loc,
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

// Clamp the given scalar or tensor input using the storage bounds encoded in
// the given quantized type, if present.
//
// - input
//   Scalar or ranked tensor input. The element type must match the storage type
//   of 'quantizedType'.
//
// - inputShape
//   If 'input' is a tensor, combination of attributes/values representing its
//   static/dynamic dimensions. If 'input' is a scalar, empty list.
//
// - quantizedType
//   Per-axis or per-channel quantized type.
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

// Emit op 'arith.fptosi' or 'arith.fptoui'.
Value convertFloatToInteger(OpBuilder &builder, Location loc, Value input,
                            Type resultType, bool isSigned) {
  if (isSigned)
    return builder.create<arith::FPToSIOp>(loc, resultType, input);
  return builder.create<arith::FPToUIOp>(loc, resultType, input);
}

// Emit op 'arith.sitofp' or 'arith.uitofp'.
Value convertIntegerToFloat(OpBuilder &builder, Location loc, Value input,
                            Type resultType, bool isSigned) {
  if (isSigned)
    return builder.create<arith::SIToFPOp>(loc, resultType, input);
  return builder.create<arith::UIToFPOp>(loc, resultType, input);
}

// Quantize a scalar or ranked tensor value. The stored value is clamped using 
// the storage bounds encoded in the given quantized type.
//
// See function 'convertRanked()' below for a description of the arguments.
Value quantizeValue(OpBuilder &builder, Location loc, Value input,
                    ArrayRef<OpFoldResult> inputShape, Value scale,
                    Value zeroPoint, QuantizedType quantizedType) {
  // Convert scale to tensor if necessary
  auto inputType = input.getType();
  scale = getScalarOrTensorConstant(
      builder, loc, scale, inputType, inputShape);

  // Scale input
  auto scaledValue = builder.create<arith::DivFOp>(loc, input, scale);

  // Skip unnecessary computations if no zero point is given
  Value storedValueFloat = scaledValue;
  if (!matchPattern(zeroPoint, m_Zero())) {
    // Convert zero point to tensor if necessary
    zeroPoint = getScalarOrTensorConstant(builder, loc, zeroPoint, inputType,
                                          inputShape);

    // Convert zero point from storage to expressed type
    zeroPoint = convertIntegerToFloat(builder, loc, zeroPoint,
                                      scale.getType(),
                                      quantizedType.isSigned());

    // Add zero point to stored value
    storedValueFloat =
        builder.create<arith::AddFOp>(loc, scaledValue, zeroPoint);
  }

  // Convert stored value to storage type
  auto storageScalarOrTensorType =
      getScalarOrTensorType(quantizedType.getStorageType(), inputType);
  auto storedValueInt = convertFloatToInteger(
      builder, loc, storedValueFloat, storageScalarOrTensorType,
      quantizedType.isSigned());

  // Clamp stored value it if the storage type is bound
  auto storedValueClamped = clampScalarOrTensor(builder, loc, storedValueInt,
                                                inputShape, quantizedType);
  return storedValueClamped;
}

// Dequantize a scalar or ranked tensor input.
//
// See function 'convertRanked()' below for a description of the arguments.
Value dequantizeValue(OpBuilder &builder, Location loc, Value input,
                      ArrayRef<OpFoldResult> inputShape, Value scale,
                      Value zeroPoint, QuantizedType quantizedType) {
  // Convert scale to tensor if necessary
  auto inputType = input.getType();
  scale = getScalarOrTensorConstant(
      builder, loc, scale, inputType, inputShape);

  // Convert stored value to float
  auto result = convertIntegerToFloat(
      builder, loc, input, scale.getType(), quantizedType.isSigned());

  // Skip unnecessary computations if no zero point is given
  if (!matchPattern(zeroPoint, m_Zero())) {
    // Convert zero point to tensor if necessary
    zeroPoint = getScalarOrTensorConstant(builder, loc, zeroPoint, inputType,
                                          inputShape);

    // Convert zero point from storage to expressed type
    zeroPoint = convertIntegerToFloat(builder, loc, zeroPoint,
                                      scale.getType(),
                                      quantizedType.isSigned());

    // Subtract zero point to stored value
    result = builder.create<arith::SubFOp>(loc, result, zeroPoint);
  }

  // Multiply by scale
  result = builder.create<arith::MulFOp>(loc, result, scale);
  return result;
}

// Convert a scalar or ranked tensor input with the given scale and zero point
// values.
//
// - input
//   Scalar or ranked tensor value.
//
// - inputShape
//   If 'input' is a tensor, combination or attributes/values representing its
//   static/dynamic dimensions. If 'input' is a scalar, empty list.
//
// - scale
//   Scale as a floating-point scalar value.
//
// - zeroPoint
//   Zero point as an integer scalar value.
//
// - quantizedType
//   Scalar quantized type of the result ('quant.qcast') or of the input
//   ('quant.dcast').
//
Value convertRanked(OpBuilder &builder, Location loc, Operation *op,
                    Value input, ArrayRef<OpFoldResult> inputShape, Value scale,
                    Value zeroPoint, QuantizedType quantizedType) {
  if (isa<QuantizeCastOp>(op))
    return quantizeValue(builder, loc, input, inputShape, scale, zeroPoint,
                         quantizedType);
  if (isa<DequantizeCastOp>(op))
    return dequantizeValue(builder, loc, input, inputShape, scale, zeroPoint,
                           quantizedType);
  llvm_unreachable("unexpected quant op");
}

// Convert an operation using per-layer quantization with a scalar or ranked
// tensor input.
//
// - op
//   'quant.dcast' or 'quant.qcast' op.
//
// - input
//   Scalar or ranked tensor.
//
// - quantizedType
//   Per-layer quantized type.
//
Value convertPerLayerRanked(OpBuilder &builder, Location loc, Operation *op,
                            Value input, UniformQuantizedType quantizedType) {
  // Create scale and zero point constants
  auto expressedType = quantizedType.getExpressedType();
  auto storageType = quantizedType.getStorageType();
  auto scaleAttr =
      builder.getFloatAttr(expressedType, quantizedType.getScale());
  auto scale = builder.create<arith::ConstantOp>(loc, expressedType, scaleAttr);
  auto zeroPointAttr =
      builder.getIntegerAttr(storageType, quantizedType.getZeroPoint());
  auto zeroPoint =
      builder.create<arith::ConstantOp>(loc, storageType, zeroPointAttr);

  auto inputShape = getScalarOrTensorShape(builder, loc, input);
  return convertRanked(builder, loc, op, input, inputShape, scale, zeroPoint,
                       quantizedType);
}

// Convert an operation using per-layer quantization.
//
// - op
//   'quant.dcast' or 'quant.qcast' op.
//
// - input
//   Scalar, ranked tensor, or unranked tensor.
//
// - quantizedType
//   Per-layer quantized type.
//
Value convertPerLayer(OpBuilder &builder, Location loc, Operation *op,
                      Value input, UniformQuantizedType quantizedType) {
  // Flatten input if unranked
  bool isUnranked = isa<UnrankedTensorType>(input.getType());
  Value inputShape;
  if (isUnranked)
    std::tie(input, inputShape) = flattenUnrankedTensor(builder, loc, input);

  // Process ranked tensor
  auto result = convertPerLayerRanked(builder, loc, op, input, quantizedType);

  // Restore original shape if unranked
  if (isUnranked)
    result = restoreUnrankedTensorShape(builder, loc, result, inputShape);

  return result;
}

// Convert an operation using per-channel quantization and a scalar or ranked
// tensor as an input.
//
// - op
//   'quant.dcast' or 'quant.qcast' op.
//
// - input
//   Scalar or ranked tensor.
//
// - quantizedType
//   Per-channel quantized type.
//
Value convertPerChannelRanked(OpBuilder &builder, Location loc, Operation *op,
                              Value input,
                              UniformQuantizedPerAxisType quantizedType,
                              int64_t channelAxis) {
  auto *context = builder.getContext();

  auto inputType = cast<RankedTensorType>(input.getType());
  auto inputRank = inputType.getRank();

  auto scales = materializePerChannelScales(builder, loc, quantizedType);
  auto zeroPoints =
      materializePerChannelZeroPoints(builder, loc, quantizedType);

  auto elementType = isa<FloatType>(inputType.getElementType())
                         ? quantizedType.getStorageType()
                         : quantizedType.getExpressedType();
  auto initShape = tensor::getMixedSizes(builder, loc, input);
  Value init = builder.create<tensor::EmptyOp>(loc, initShape, elementType);

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
  auto result = builder.create<linalg::GenericOp>(
      loc,
      init.getType(),  // resultType
      ValueRange{input, scales, zeroPoints},  // inputs
      ValueRange{init},  // outputs
      indexingMaps,
      iteratorTypes,
      [&](OpBuilder& builder, Location loc, ValueRange args) {
        assert(args.size() == 4);
        auto input = args[0];
        auto scale = args[1];
        auto zeroPoint = args[2];

        auto result = convertRanked(builder, loc, op, input, {}, scale,
                                    zeroPoint, quantizedType);

        builder.create<linalg::YieldOp>(loc, result);
      })
      .getResult(0);

  return result;
}

// Convert an operation using per-channel quantization.
//
// - op
//   'quant.dcast' or 'quant.qcast' op.
//
// - input
//   Scalar, ranked tensor, or unranked tensor.
//
// - quantizedType
//   Per-channel quantized type.
//
Value convertPerChannel(OpBuilder &builder, Location loc, Operation *op,
                        Value input,
                        UniformQuantizedPerAxisType quantizedType) {
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
  auto result = convertPerChannelRanked(builder, loc, op, input, quantizedType,
                                        channelAxis);

  // Restore original tensor shape if unranked
  if (isUnranked)
    result = restoreUnrankedTensorShape(builder, loc, result, inputShape);

  return result;
}

// Convert a quantization operation.
//
// - op
//   'quant.dcast' or 'quant.qcast' op.
//
// - input
//   Scalar, ranked tensor, or unranked tensor. The element type matches
//   the storage type (quant.dcast) or expressed type (quant.qcast) of
//   'quantizedType'.
//
// - quantizedType
//   Per-layer or per-channel quantized type.
//
Value convertQuantized(OpBuilder &builder, Location loc, Operation *op,
                       Value input, Type quantizedType) {
  if (auto uniformQuantizedType = dyn_cast<UniformQuantizedType>(quantizedType))
    return convertPerLayer(builder, loc, op, input, uniformQuantizedType);

  if (auto uniformQuantizedPerAxisType =
          dyn_cast<UniformQuantizedPerAxisType>(quantizedType))
    return convertPerChannel(builder, loc, op, input,
                             uniformQuantizedPerAxisType);

  llvm_unreachable("unexpected quantized type");
}

// Lowering pattern for 'quant.dcast'
struct DequantizeCastOpConversion : public OpConversionPattern<quant::DequantizeCastOp> {
  using OpConversionPattern<quant::DequantizeCastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(quant::DequantizeCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto input = op.getInput();
    auto quantizedType =
        cast<QuantizedType>(getScalarType(op.getInput().getType()));

    // Convert quantized input to storage type
    auto storageScalarOrTensorType =
        getScalarOrTensorType(quantizedType.getStorageType(), input.getType());
    input = rewriter.create<quant::StorageCastOp>(
        loc, storageScalarOrTensorType, input);

    auto result = convertQuantized(rewriter, loc, op, input, quantizedType);

    rewriter.replaceOp(op, result);
    return success();
  }
};

// Lowering pattern for 'quant.qcast'
struct QuantizeCastOpConversion : public OpConversionPattern<quant::QuantizeCastOp> {
  using OpConversionPattern<quant::QuantizeCastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(quant::QuantizeCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto input = op.getInput();
    auto quantizedType = getScalarType(op.getResult().getType());

    // Flatten unranked tensor input
    auto result = convertQuantized(rewriter, loc, op, input, quantizedType);

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
