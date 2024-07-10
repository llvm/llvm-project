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
#include "mlir/Dialect/SCF/IR/SCF.h"
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

// Calculate the size of an unranked tensor starting at dimension 'fromDim' up
// to, but not including, dimension 'toDim'.
Value getUnrankedTensorSizeRange(OpBuilder &builder, Location loc, Value input,
                                 Value fromDim, Value toDim, Value one) {
  auto loop = builder.create<scf::ForOp>(
      loc,
      fromDim,  // lowerBound
      toDim,  // upperBound
      one,  // step
      one,  // iterArgs
      [&](OpBuilder &builder, Location loc, Value index, ValueRange args) {
        Value size = builder.create<tensor::DimOp>(loc, input, index);
        Value totalSize = builder.create<arith::MulIOp>(loc, args.front(), size);
        builder.create<scf::YieldOp>(loc, totalSize);
      });
  return loop.getResult(0);
}

// Obtain the shape of an unranked tensor. This function returns a 1D tensor of
// size 'rank' and element type 'index'.
Value getUnrankedTensorShape(OpBuilder &builder, Location loc, Value input,
                             Value rank) {
  auto shapeType =
      RankedTensorType::get({ShapedType::kDynamic}, builder.getIndexType());
  auto shape = builder.create<tensor::GenerateOp>(
      loc,
      shapeType,
      rank,
      [&](OpBuilder &builder, Location loc, ValueRange args) {
        Value size = builder.create<tensor::DimOp>(loc, input, args.front());
        builder.create<tensor::YieldOp>(loc, size);
      });
  return shape;
}

class QuantizeCastOpConversion : public OpConversionPattern<quant::QuantizeCastOp> {

  Value convertPerLayerScalarOrRanked(
      OpBuilder &builder, Location loc, Value input,
      UniformQuantizedType quantizedType) const {

    auto inputType = input.getType();
    auto expressedType = cast<FloatType>(quantizedType.getExpressedType());
    auto storageType = cast<IntegerType>(quantizedType.getStorageType());
    auto storageContainerType = cloneContainerType(inputType, storageType);

    auto inputShape = getContainerShape(builder, loc, input);

    // Scale and zero point scalars
    auto scaleAttr = builder.getFloatAttr(expressedType, quantizedType.getScale());
    auto scale = getContainerConstant(builder, loc, scaleAttr, inputType, inputShape);
    auto zeroPointAttr = builder.getFloatAttr(expressedType, quantizedType.getZeroPoint());
    auto zeroPoint = getContainerConstant(builder, loc, zeroPointAttr, inputType, inputShape);

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
  
  Value convertPerLayerUnranked(
      OpBuilder &builder, Location loc, Value input,
      UniformQuantizedType quantizedType) const {
    auto rank = builder.create<tensor::RankOp>(loc, input);
    auto inputShape = getUnrankedTensorShape(builder, loc, input, rank);
    auto inputType = cast<UnrankedTensorType>(input.getType());

    auto zero = builder.create<arith::ConstantIndexOp>(loc, 0);
    auto one = builder.create<arith::ConstantIndexOp>(loc, 1);
    auto inputSize = getUnrankedTensorSizeRange(builder, loc, input, zero, rank, one);

    // Compute collapsed input shape as a 1D 1-sized index tensor
    auto collapsedInputShapeType = RankedTensorType::get({1}, builder.getIndexType());
    auto collapsedInputShape = builder.create<tensor::FromElementsOp>(
        loc, collapsedInputShapeType, inputSize);

    // Reshape input tensor into 1D
    auto collapsedInputType = RankedTensorType::get({ShapedType::kDynamic},
                                                    inputType.getElementType());
    auto collapsedInput = builder.create<tensor::ReshapeOp>(
        loc, collapsedInputType, input, collapsedInputShape);

    // Now we know how to convert a ranked tensor
    auto collapsedStoredValue = convertPerLayerScalarOrRanked(
        builder, loc, collapsedInput, quantizedType);

    // Expand stored value back to the original shape
    auto expandedStoredValueType =
        UnrankedTensorType::get(quantizedType.getStorageType());
    auto expandedStoredValue = builder.create<tensor::ReshapeOp>(
        loc, expandedStoredValueType, collapsedStoredValue, inputShape);
    return expandedStoredValue;
  }

public:
  using OpConversionPattern<quant::QuantizeCastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(quant::QuantizeCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto input = op.getInput();
    auto resultScalarType = getScalarType(op.getResult().getType());

    // Per-layer vs per-channel quantization
    Value storedValue;
    if (auto quantizedType = dyn_cast<UniformQuantizedType>(resultScalarType)) {
      storedValue = isa<UnrankedTensorType>(input.getType()) ?
          convertPerLayerUnranked(rewriter, loc, input, quantizedType) :
          convertPerLayerScalarOrRanked(rewriter, loc, input, quantizedType);
    } else if (auto quantizedType = dyn_cast<UniformQuantizedPerAxisType>(resultScalarType)) {
      // FIXM
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
      scf::SCFDialect,
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
