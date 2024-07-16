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

// If 'containerType' is a tensor, return its element type. If it is a scalar,
// return it as is.
Type getScalarType(Type containerType) {
  if (auto tensorType = dyn_cast<TensorType>(containerType))
    return tensorType.getElementType();
  return containerType;
}

// Return the shape of a container as a combination of attributes (static
// dimensions) and values (dynamic dimensions). If 'container' is a scalar,
// an empty list is returned. If 'container' is a tensor, its shape is returned.
SmallVector<OpFoldResult> getContainerShape(OpBuilder &builder, Location loc,
                                            Value container) {
  if (isa<TensorType>(container.getType()))
    return tensor::getMixedSizes(builder, loc, container);
  return {};
}

// Clone the given 'containerType' with the new given 'elementType'. If
// 'containerType' is a scalar type, there is nothing to clone, and
// 'elementType' itself is returned. If 'constainerType' is a tensor, its
// shape is cloned but the new element type is used.
Type cloneContainerType(Type containerType, Type elementType) {
  if (auto tensorType = dyn_cast<TensorType>(containerType))
    return tensorType.clone(elementType);
  return elementType;
}

// Get a scalar or tensor constant containing the value given in 'attr'.
// If 'containerType' is a scalar, a scalar constant is returned. If
// 'containerType' is a tensor, a tensor splat of shape 'containerShape' is
// returned.
Value getContainerConstant(OpBuilder &builder, Location loc, TypedAttr attr,
                           Type containerType,
                           ArrayRef<OpFoldResult> containerShape) {
  // A statically shaped tensor can be created with 'arith.constant'
  auto tensorType = dyn_cast<TensorType>(containerType);
  if (tensorType && tensorType.hasStaticShape()) {
    auto denseElementsAttr = DenseElementsAttr::get(tensorType, attr);
    return builder.create<arith::ConstantOp>(loc, tensorType, denseElementsAttr);
  }

  // Scalar and dynamically shaped tensor containers need the scalar constant
  // to be first materialized.
  Value containerConstant =
      builder.create<arith::ConstantOp>(loc, attr.getType(), attr);

  // Create tensor splat if necessary
  if (tensorType) {
    containerConstant =
        builder.create<tensor::SplatOp>(loc, containerConstant, containerShape);
  }
  return containerConstant;
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
                                                        int64_t axis) {
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
  Value axisSize = builder.create<tensor::DimOp>(loc, input, axisValue);

  // Compute flat input shape as a 3-element 1D tensor
  auto flatShapeType = shape::getExtentTensorType(context, 3);
  auto flatInputShape = builder.create<tensor::FromElementsOp>(
      loc, flatShapeType, ValueRange{sizeLeft, axisSize, sizeRight});

  // Reshape input to 3D tensor
  auto inputType = cast<UnrankedTensorType>(input.getType());
  auto elementType = inputType.getElementType();
  SmallVector<int64_t> flatInputDims(3, ShapedType::kDynamic);
  auto flatInputType = RankedTensorType::get(flatInputDims, elementType);
  auto flatInput = builder.create<tensor::ReshapeOp>(
      loc, flatInputType, input, flatInputShape);

  return std::make_pair(flatInput, inputShape);
}

Value restoreUnrankedTensor(OpBuilder &builder, Location loc, Value input,
                            Value shape) {
  auto inputType = cast<TensorType>(input.getType());
  auto elementType = inputType.getElementType();
  auto unrankedType = UnrankedTensorType::get(elementType);
  return builder.create<tensor::ReshapeOp>(loc, unrankedType, input, shape);
}

Value materializeScales(OpBuilder &builder, Location loc,
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

Value materializeZeroPoints(OpBuilder &builder, Location loc,
                            UniformQuantizedPerAxisType quantizedType) {
  auto zeroPoints = quantizedType.getZeroPoints();
  auto expressedType = quantizedType.getExpressedType();
  auto zeroPointAttrs = llvm::map_to_vector(zeroPoints, [&](int64_t zeroPoint) -> Attribute {
    return builder.getFloatAttr(expressedType, static_cast<double>(zeroPoint));
  });
  auto tensorType = RankedTensorType::get({(int64_t) zeroPoints.size()}, expressedType);
  auto zeroPointsAttr = DenseElementsAttr::get(tensorType, zeroPointAttrs);
  return builder.create<arith::ConstantOp>(loc, tensorType, zeroPointsAttr);
}

Value quantizeValue(OpBuilder &builder, Location loc, Value input,
                    ArrayRef<OpFoldResult> inputShape, Value scale,
                    Value zeroPoint, QuantizedType quantizedType) {
  auto inputType = input.getType();
  auto storageType = cast<IntegerType>(quantizedType.getStorageType());
  auto storageContainerType = cloneContainerType(inputType, storageType);

  auto scaledValue = builder.create<arith::DivFOp>(loc, input, scale);
  auto storedValueAsExpressedType = builder.create<arith::AddFOp>(loc, scaledValue, zeroPoint);

  Value storedValue;
  if (quantizedType.isSigned()) {
    storedValue = builder.create<arith::FPToSIOp>(
        loc, storageContainerType, storedValueAsExpressedType);
  } else {
    storedValue = builder.create<arith::FPToUIOp>(
        loc, storageContainerType, storedValueAsExpressedType);
  }

  // Clamp stored value if needed
  if (quantizedType.hasStorageTypeBounds()) {
    auto storageMinAttr = builder.getIntegerAttr(storageType, quantizedType.getStorageTypeMin());
    auto storageMaxAttr = builder.getIntegerAttr(storageType, quantizedType.getStorageTypeMax());
    auto storageMin = getContainerConstant(builder, loc, storageMinAttr, inputType, inputShape);
    auto storageMax = getContainerConstant(builder, loc, storageMaxAttr, inputType, inputShape);
    if (quantizedType.isSigned()) {
      storedValue = builder.create<arith::MaxSIOp>(loc, storedValue, storageMin);
      storedValue = builder.create<arith::MinSIOp>(loc, storedValue, storageMax);
    } else {
      storedValue = builder.create<arith::MaxUIOp>(loc, storedValue, storageMin);
      storedValue = builder.create<arith::MinUIOp>(loc, storedValue, storageMax);
    }
  }

  return storedValue;
}

class QuantizeCastOpConversion : public OpConversionPattern<quant::QuantizeCastOp> {

  Value convertPerLayerRanked(OpBuilder &builder, Location loc, Value input,
                              UniformQuantizedType quantizedType) const {

    auto inputType = input.getType();
    auto expressedType = cast<FloatType>(quantizedType.getExpressedType());

    // Create scale and zero point constants
    auto inputShape = getContainerShape(builder, loc, input);
    auto scaleAttr = builder.getFloatAttr(expressedType, quantizedType.getScale());
    auto scale = getContainerConstant(builder, loc, scaleAttr, inputType, inputShape);
    auto zeroPointAttr = builder.getFloatAttr(expressedType, quantizedType.getZeroPoint());
    auto zeroPoint = getContainerConstant(builder, loc, zeroPointAttr, inputType, inputShape);

    return quantizeValue(builder, loc, input, inputShape, scale, zeroPoint,
                         quantizedType);
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
      result = restoreUnrankedTensor(builder, loc, result, inputShape);

    return result;
  }

  Value convertPerChannelRanked(OpBuilder &builder, Location loc, Value input,
                                UniformQuantizedPerAxisType quantizedType,
                                int64_t channelAxis) const {
    auto *context = builder.getContext();

    auto inputType = cast<RankedTensorType>(input.getType());
    auto inputRank = inputType.getRank();

    auto scales = materializeScales(builder, loc, quantizedType);
    auto zeroPoints = materializeZeroPoints(builder, loc, quantizedType);

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

          auto storedValue = quantizeValue(builder, loc, expressedValue, {},
                                           scale, zeroPoint, quantizedType);

          builder.create<linalg::YieldOp>(loc, storedValue);
        })
        .getResult(0);

    return storedValue;
  }

  Value convertPerChannel(OpBuilder &builder, Location loc, Value input,
                          UniformQuantizedPerAxisType quantizedType) const {
    // Flatten unranked tensor if necessary
    bool isUnranked = isa<UnrankedTensorType>(input.getType());
    int64_t channelAxis = quantizedType.getQuantizedDimension();
    Value inputShape;
    if (isUnranked) {
      std::tie(input, inputShape) =
          flattenUnrankedTensorAroundAxis(builder, loc, input, channelAxis);
      channelAxis = 1;
    }

    // Work on a ranked tensor
    auto result = convertPerChannelRanked(builder, loc, input, quantizedType,
                                          channelAxis);

    // Restore original tensor shape if unranked
    if (isUnranked)
      result = restoreUnrankedTensor(builder, loc, result, inputShape);

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
    Value storedValue;
    if (auto quantizedType = dyn_cast<UniformQuantizedType>(resultScalarType)) {
      storedValue = convertPerLayer(rewriter, loc, input, quantizedType);
    } else if (auto quantizedType = dyn_cast<UniformQuantizedPerAxisType>(resultScalarType)) {
      storedValue = convertPerChannel(rewriter, loc, input, quantizedType);
    } else {
      llvm_unreachable("unexpected quantized type");
    }

    // Cast stored value to result quantized value
    rewriter.replaceOpWithNewOp<quant::StorageCastOp>(
        op, op.getResult().getType(), storedValue);
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
