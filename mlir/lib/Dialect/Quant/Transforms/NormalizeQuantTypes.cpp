//===- NormalizeQuantTypes.cpp - Normalize quantized types
//----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Normalize generic quantized types to specific quantized types
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/Dialect/Quant/Transforms/Passes.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace quant {

#define GEN_PASS_DEF_NORMALIZEQUANTTYPES
#include "mlir/Dialect/Quant/Transforms/Passes.h.inc"

namespace {

/// Returns true if the given sub-channel quantized type is convertible to a
/// per-tensor quantized type. This is true if the sub-channel type has only
/// one scale and one zero point.
///
/// Assumes that `tensorType` is a tensor with element type
/// `quant::UniformQuantizedSubChannelType`.
static bool isConvertibleToPerTensor(TensorType tensorType) {
  return cast<UniformQuantizedSubChannelType>(tensorType.getElementType())
             .getScales()
             .getType()
             .getNumElements() == 1;
}

/// Returns true if the given sub-channel quantized type is convertible to a
/// per-axis quantized type. This is true if the shape of the scales tensor has
/// all but one non-one value.
///
/// Assumes that `tensorType` is a tensor with element type
/// `quant::UniformQuantizedSubChannelType`.
static bool isConvertibleToPerAxis(TensorType tensorType) {
  auto shape = cast<UniformQuantizedSubChannelType>(tensorType.getElementType())
                   .getScales()
                   .getType()
                   .getShape();
  return llvm::count_if(shape, [](int64_t dim) { return dim != 1; }) == 1;
}

/// This class defines a type converter that converts sub-channel quantized
/// types to per-tensor or per-axis quantized types whenever possible.
class NormalizedQuantTypesConverter : public TypeConverter {

  static Type convertType(Type type) {
    auto tensorType = dyn_cast<TensorType>(type);
    if (!tensorType) {
      return type;
    }

    auto subChannelType =
        dyn_cast<UniformQuantizedSubChannelType>(tensorType.getElementType());
    if (!subChannelType) {
      return type;
    }

    if (isConvertibleToPerTensor(tensorType)) {
      double scale =
          subChannelType.getScales().getValues<APFloat>()[0].convertToDouble();
      int64_t zeroPoint =
          subChannelType.getZeroPoints().getValues<APInt>()[0].getSExtValue();
      auto perTensorType = UniformQuantizedType::get(
          subChannelType.getFlags(), subChannelType.getStorageType(),
          subChannelType.getExpressedType(), scale, zeroPoint,
          subChannelType.getStorageTypeMin(),
          subChannelType.getStorageTypeMax());
      return tensorType.clone(perTensorType);
    }

    if (isConvertibleToPerAxis(tensorType)) {
      auto shape = subChannelType.getScales().getType().getShape();
      auto quantizedDimItr =
          llvm::find_if(shape, [](int64_t dim) { return dim != 1; });
      auto scales = llvm::to_vector(llvm::map_range(
          subChannelType.getScales().getValues<APFloat>(),
          [](const APFloat &scale) { return scale.convertToDouble(); }));
      auto zeroPoints = llvm::to_vector(llvm::map_range(
          subChannelType.getZeroPoints().getValues<APInt>(),
          [](const APInt &zeroPoint) { return zeroPoint.getSExtValue(); }));
      auto perAxisType = UniformQuantizedPerAxisType::get(
          subChannelType.getFlags(), subChannelType.getStorageType(),
          subChannelType.getExpressedType(), scales, zeroPoints,
          quantizedDimItr - shape.begin(), subChannelType.getStorageTypeMin(),
          subChannelType.getStorageTypeMax());
      return tensorType.clone(perAxisType);
    }
    return type;
  }

public:
  explicit NormalizedQuantTypesConverter() { addConversion(convertType); }
};

/// This class implements a conversion pattern that converts any generic
/// operation with sub-channel quantized types to an equivalent operation with
/// per-tensor or per-axis quantized types.
class ConvertGenericOpwithSubChannelType : public ConversionPattern {
public:
  ConvertGenericOpwithSubChannelType(TypeConverter &typeConverter,
                                     MLIRContext *context)
      : ConversionPattern(typeConverter, MatchAnyOpTypeTag{}, 0, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    SmallVector<Type> resultTypes;
    if (failed(typeConverter->convertTypes(op->getResultTypes(), resultTypes)))
      return failure();

    auto *newOp = Operation::create(
        op->getLoc(), op->getName(), resultTypes, operands, op->getAttrs(),
        op->getPropertiesStorage(), op->getSuccessors(), op->getNumRegions());
    for (auto regions : llvm::zip(op->getRegions(), newOp->getRegions())) {
      Region &before = std::get<0>(regions);
      Region &parent = std::get<1>(regions);
      rewriter.inlineRegionBefore(before, parent, parent.end());
      if (failed(rewriter.convertRegionTypes(&parent, *typeConverter)))
        return failure();
    }
    rewriter.insert(newOp);
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

// Conversion pass
class NormalizeQuantTypes
    : public impl::NormalizeQuantTypesBase<NormalizeQuantTypes> {
public:
  void runOnOperation() override {

    auto *context = &getContext();

    NormalizedQuantTypesConverter typeConverter;
    ConversionTarget target(*context);

    // Determine legal operations.
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
             typeConverter.isLegal(&op.getBody());
    });
    target.markUnknownOpDynamicallyLegal([&](Operation *op) {
      return typeConverter.isLegal(op->getOperandTypes()) &&
             typeConverter.isLegal(op->getResultTypes());
    });

    // Register conversion patterns
    RewritePatternSet patterns(context);
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    patterns.add<ConvertGenericOpwithSubChannelType>(typeConverter, context);

    // Apply conversion
    if (failed(
            applyFullConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

} // namespace quant
} // namespace mlir
