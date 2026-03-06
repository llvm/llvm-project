//===- TosaNarrowTypes.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the TOSA narrowing passes that rewrite tensor element
// types to narrower equivalents (i64 -> i32, f64 -> f32, ...).
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/Transforms/Passes.h"

#include "llvm/ADT/APFloat.h"

#include <algorithm>
#include <limits>
#include <type_traits>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Utils/ConversionUtils.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace tosa {
#define GEN_PASS_DEF_TOSANARROWI64TOI32PASS
#define GEN_PASS_DEF_TOSANARROWF64TOF32PASS
#include "mlir/Dialect/Tosa/Transforms/Passes.h.inc"
} // namespace tosa
} // namespace mlir

using namespace mlir;
using namespace mlir::tosa;

namespace {

// Narrowing mode for this pass.
enum class TosaNarrowKind { Int64ToInt32, Float64ToFloat32 };

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

template <TosaNarrowKind Kind>
bool isSourceInteger(IntegerType type) {
  if constexpr (Kind == TosaNarrowKind::Int64ToInt32)
    return type.isInteger(64);
  return false;
}

template <TosaNarrowKind Kind>
bool isSourceFloat(FloatType type) {
  if constexpr (Kind == TosaNarrowKind::Float64ToFloat32)
    return type.isF64();
  return false;
}

template <TosaNarrowKind Kind>
Type convertInteger(IntegerType type) {
  if (!isSourceInteger<Kind>(type))
    return type;
  if constexpr (Kind == TosaNarrowKind::Int64ToInt32)
    return IntegerType::get(type.getContext(), 32);
  return type;
}

template <TosaNarrowKind Kind>
Type convertFloat(FloatType type) {
  if (!isSourceFloat<Kind>(type))
    return type;
  if constexpr (Kind == TosaNarrowKind::Float64ToFloat32)
    return Float32Type::get(type.getContext());
  return type;
}

template <TosaNarrowKind Kind>
bool isSourceElement(Type type) {
  if (auto intTy = dyn_cast<IntegerType>(type))
    return isSourceInteger<Kind>(intTy);
  if (auto floatTy = dyn_cast<FloatType>(type))
    return isSourceFloat<Kind>(floatTy);
  return false;
}

template <TosaNarrowKind Kind>
Type convertElement(Type type) {
  if (auto intTy = dyn_cast<IntegerType>(type))
    return convertInteger<Kind>(intTy);
  if (auto floatTy = dyn_cast<FloatType>(type))
    return convertFloat<Kind>(floatTy);
  return type;
}

template <TosaNarrowKind Kind>
bool typeNeedsConversion(Type type) {
  if (auto shaped = dyn_cast<ShapedType>(type))
    return isSourceElement<Kind>(shaped.getElementType());
  return isSourceElement<Kind>(type);
}

FailureOr<APInt> convertIntegerConstant(IntegerType targetType,
                                        const APInt &value,
                                        bool allowLossyConversion) {
  const unsigned targetWidth = targetType.getWidth();
  if (!allowLossyConversion && !value.isSignedIntN(targetWidth))
    return failure();

  if (allowLossyConversion)
    return value.truncSSat(targetWidth);
  return value.sextOrTrunc(targetWidth);
}

FailureOr<APFloat> convertFloatConstant(FloatType targetType,
                                        const APFloat &value,
                                        bool allowLossyConversion) {
  APFloat converted(value);
  bool losesInfo = false;
  converted.convert(targetType.getFloatSemantics(),
                    APFloat::rmNearestTiesToEven, &losesInfo);
  if (!allowLossyConversion && losesInfo)
    return failure();
  return converted;
}

// Narrows scalar constant attributes so they keep matching the converted
// element types.
template <TosaNarrowKind Kind>
FailureOr<Attribute> tryConvertScalarAttribute(Attribute attribute,
                                               bool allowLossyConversion) {
  if constexpr (Kind == TosaNarrowKind::Int64ToInt32) {
    if (const auto intAttr = dyn_cast<IntegerAttr>(attribute)) {
      if (const auto intType = dyn_cast<IntegerType>(intAttr.getType());
          intType && isSourceInteger<Kind>(intType)) {
        const auto convertedType =
            cast<IntegerType>(convertInteger<Kind>(intType));
        FailureOr<APInt> convertedValue = convertIntegerConstant(
            convertedType, intAttr.getValue(), allowLossyConversion);
        if (failed(convertedValue))
          return failure();
        return IntegerAttr::get(convertedType, convertedValue.value());
      }
    }
  } else if constexpr (Kind == TosaNarrowKind::Float64ToFloat32) {
    if (const auto floatAttr = dyn_cast<FloatAttr>(attribute)) {
      if (const auto floatType = dyn_cast<FloatType>(floatAttr.getType());
          floatType && isSourceFloat<Kind>(floatType)) {
        const auto convertedType =
            cast<FloatType>(convertFloat<Kind>(floatType));
        FailureOr<APFloat> convertedValue = convertFloatConstant(
            convertedType, floatAttr.getValue(), allowLossyConversion);
        if (failed(convertedValue))
          return failure();
        return FloatAttr::get(convertedType, convertedValue.value());
      }
    }
  }

  return attribute;
}

template <TosaNarrowKind Kind>
FailureOr<Attribute>
convertDenseIntElementsAttr(ShapedType type, DenseIntElementsAttr attr,
                            const TypeConverter &typeConverter,
                            bool allowLossyConversion) {
  if constexpr (Kind != TosaNarrowKind::Int64ToInt32)
    return attr;

  const auto oldElementType = dyn_cast<IntegerType>(type.getElementType());
  if (!oldElementType || !isSourceInteger<Kind>(oldElementType))
    return attr;

  const auto newType =
      dyn_cast_or_null<ShapedType>(typeConverter.convertType(type));
  if (!newType)
    return failure();

  const auto newElementType = dyn_cast<IntegerType>(newType.getElementType());
  if (!newElementType)
    return failure();

  if (!allowLossyConversion) {
    for (APInt value : attr.getValues<APInt>())
      if (failed(convertIntegerConstant(newElementType, value,
                                        /*allowLossyConversion=*/false)))
        return failure();
  }

  Attribute convertedAttr =
      attr.mapValues(newElementType, [&](const APInt &value) -> APInt {
        return convertIntegerConstant(newElementType, value,
                                      /*allowLossyConversion=*/true)
            .value();
      });
  return convertedAttr;
}

template <TosaNarrowKind Kind>
FailureOr<Attribute>
convertDenseFPElementsAttr(ShapedType type, DenseFPElementsAttr attr,
                           const TypeConverter &typeConverter,
                           bool allowLossyConversion) {
  if constexpr (Kind != TosaNarrowKind::Float64ToFloat32)
    return attr;

  const auto oldElementType = dyn_cast<FloatType>(type.getElementType());
  if (!oldElementType || !isSourceFloat<Kind>(oldElementType))
    return attr;

  const auto newType =
      dyn_cast_or_null<ShapedType>(typeConverter.convertType(type));
  if (!newType)
    return failure();

  const auto newElementType = dyn_cast<FloatType>(newType.getElementType());
  if (!newElementType)
    return failure();

  if (!allowLossyConversion) {
    for (APFloat value : attr.getValues<APFloat>())
      if (failed(convertFloatConstant(newElementType, value,
                                      /*allowLossyConversion=*/false)))
        return failure();
  }

  Attribute convertedAttr =
      attr.mapValues(newElementType, [&](const APFloat &value) -> APInt {
        APFloat converted = convertFloatConstant(newElementType, value,
                                                 /*allowLossyConversion=*/true)
                                .value();
        // DenseFPElementsAttr stores each float as raw bits, so emit the APInt
        // representation that MLIR expects in the underlying buffer.
        return converted.bitcastToAPInt();
      });
  return convertedAttr;
}

template <TosaNarrowKind Kind>
FailureOr<Attribute> convertDenseResourceElementsAttr(
    ShapedType type, DenseResourceElementsAttr attr,
    const TypeConverter &typeConverter, bool allowLossyConversion) {
  static_assert(Kind == TosaNarrowKind::Int64ToInt32 ||
                Kind == TosaNarrowKind::Float64ToFloat32);
  using From =
      std::conditional_t<Kind == TosaNarrowKind::Int64ToInt32, int64_t, double>;
  using To =
      std::conditional_t<Kind == TosaNarrowKind::Int64ToInt32, int32_t, float>;

  if (Kind == TosaNarrowKind::Int64ToInt32 &&
      !isa<DenseI64ResourceElementsAttr>(attr)) {
    return attr;
  }

  if (Kind == TosaNarrowKind::Float64ToFloat32 &&
      !isa<DenseF64ResourceElementsAttr>(attr)) {
    return attr;
  }

  auto narrow = [](From value) {
    if constexpr (Kind == TosaNarrowKind::Int64ToInt32) {
      value = std::clamp<From>(value, std::numeric_limits<To>::min(),
                               std::numeric_limits<To>::max());
    }

    return static_cast<To>(value);
  };

  const auto newType =
      dyn_cast_or_null<ShapedType>(typeConverter.convertType(type));
  if (!newType) {
    return failure();
  }

  const std::optional<ArrayRef<From>> values =
      tryGetDenseResourceValues<From>(attr);
  if (!values) {
    return failure();
  }

  SmallVector<To> newValues;
  newValues.reserve(values->size());
  for (From value : *values) {
    const To convertedValue = narrow(value);
    if (!allowLossyConversion && convertedValue != value) {
      return failure();
    }

    newValues.push_back(convertedValue);
  }

  AsmResourceBlob blob = HeapAsmResourceBlob::allocateAndCopyInferAlign(
      ArrayRef<To>(newValues.data(), newValues.size()));

  auto resourceManager =
      DenseResourceElementsHandle::getManagerInterface(attr.getContext());
  resourceManager.getBlobManager().update(attr.getRawHandle().getKey(),
                                          std::move(blob));

  return DenseResourceElementsAttr::get(newType, attr.getRawHandle());
}

template <TosaNarrowKind Kind, typename AttrT>
FailureOr<Attribute>
convertAttributeWithTypeConverter(AttrT attr, Type type,
                                  const TypeConverter *typeConverter) {
  if (!typeNeedsConversion<Kind>(type))
    return attr;

  const std::optional<Attribute> convertedAttribute =
      typeConverter->convertTypeAttribute(type, attr);
  if (!convertedAttribute)
    return failure();

  return convertedAttribute.value();
}

// Rejects cast rewrites that would lose precision (unless aggressive mode is
// enabled).
template <TosaNarrowKind Kind>
LogicalResult
verifyCastDoesNotLosePrecision(Operation *op, ShapedType inputType,
                               ShapedType resultType,
                               ConversionPatternRewriter &rewriter) {
  if constexpr (Kind == TosaNarrowKind::Int64ToInt32) {
    const auto elementInputIntType =
        dyn_cast<IntegerType>(inputType.getElementType());
    const auto elementResultIntType =
        dyn_cast<IntegerType>(resultType.getElementType());
    if (elementInputIntType && elementResultIntType &&
        elementInputIntType.getWidth() > elementResultIntType.getWidth())
      return rewriter.notifyMatchFailure(
          op, "Narrowing cast may lead to data loss.");
  } else if constexpr (Kind == TosaNarrowKind::Float64ToFloat32) {
    const auto elementInputFloatType =
        dyn_cast<FloatType>(inputType.getElementType());
    const auto elementResultFloatType =
        dyn_cast<FloatType>(resultType.getElementType());
    if (elementInputFloatType && elementResultFloatType &&
        elementInputFloatType.getIntOrFloatBitWidth() >
            elementResultFloatType.getIntOrFloatBitWidth())
      return rewriter.notifyMatchFailure(
          op, "Narrowing cast may lead to data loss.");
  }

  return success();
}

// ---------------------------------------------------------------------------
// Conversion patterns
// ---------------------------------------------------------------------------

// Applies the narrowing TypeConverter to a single TOSA op, including its
// attributes and nested regions.
template <TosaNarrowKind Kind>
LogicalResult convertGenericOp(Operation *op, ValueRange operands,
                               ConversionPatternRewriter &rewriter,
                               const TypeConverter *typeConverter,
                               bool allowLossyConversion) {
  SmallVector<Type, 4> newResults;
  if (failed(typeConverter->convertTypes(op->getResultTypes(), newResults)))
    return failure();

  OperationState state(op->getLoc(), op->getName().getStringRef(), operands,
                       newResults, {}, op->getSuccessors());

  // Keep attribute payloads consistent with the converted element types.
  for (const NamedAttribute &namedAttribute : op->getAttrs()) {
    const Attribute attribute = namedAttribute.getValue();

    if (isa<IntegerAttr>(attribute) || isa<FloatAttr>(attribute)) {
      FailureOr<Attribute> convertedAttr =
          tryConvertScalarAttribute<Kind>(attribute, allowLossyConversion);
      if (failed(convertedAttr))
        return rewriter.notifyMatchFailure(
            op, "Scalar attribute narrowing would lose precision; enable "
                "aggressive rewrite to override.");
      state.addAttribute(namedAttribute.getName(), convertedAttr.value());
      continue;
    }

    if (const auto typeAttr = dyn_cast<TypeAttr>(attribute)) {
      FailureOr<Attribute> convertedAttr =
          convertAttributeWithTypeConverter<Kind>(typeAttr, typeAttr.getValue(),
                                                  typeConverter);
      if (failed(convertedAttr))
        return rewriter.notifyMatchFailure(op,
                                           "Failed to convert type attribute.");
      state.addAttribute(namedAttribute.getName(), convertedAttr.value());
      continue;
    }

    if (const auto denseElementsAttr = dyn_cast<DenseElementsAttr>(attribute)) {
      FailureOr<Attribute> convertedAttr =
          convertAttributeWithTypeConverter<Kind>(
              denseElementsAttr, denseElementsAttr.getType(), typeConverter);
      if (failed(convertedAttr))
        return rewriter.notifyMatchFailure(
            op, "Failed to convert dense elements attribute without precision "
                "loss; enable aggressive rewrite to override.");
      state.addAttribute(namedAttribute.getName(), convertedAttr.value());
      continue;
    }

    if (const auto denseResourceElementsAttr =
            dyn_cast<DenseResourceElementsAttr>(attribute)) {
      FailureOr<Attribute> convertedAttr =
          convertAttributeWithTypeConverter<Kind>(
              denseResourceElementsAttr, denseResourceElementsAttr.getType(),
              typeConverter);
      if (failed(convertedAttr))
        return rewriter.notifyMatchFailure(
            op, "Failed to convert dense resource elements attribute without "
                "precision loss; enable aggressive rewrite to override.");
      state.addAttribute(namedAttribute.getName(), convertedAttr.value());
      continue;
    }

    state.addAttribute(namedAttribute.getName(), attribute);
  }

  for (Region &region : op->getRegions()) {
    Region *newRegion = state.addRegion();
    rewriter.inlineRegionBefore(region, *newRegion, newRegion->begin());
    if (failed(rewriter.convertRegionTypes(newRegion, *typeConverter)))
      return failure();
  }

  Operation *newOp = rewriter.create(state);
  rewriter.replaceOp(op, newOp->getResults());
  return success();
}

template <TosaNarrowKind Kind>
class ConvertGenericOp : public ConversionPattern {
public:
  ConvertGenericOp(TypeConverter &typeConverter, MLIRContext *context,
                   bool allowLossyConversion)
      : ConversionPattern(typeConverter, MatchAnyOpTypeTag{}, 0, context),
        allowLossyConversion(allowLossyConversion) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    if (!isa<tosa::TosaOp>(op))
      return rewriter.notifyMatchFailure(
          op,
          "Support for operations other than TOSA has not been implemented.");

    return convertGenericOp<Kind>(op, operands, rewriter, typeConverter,
                                  allowLossyConversion);
  }

private:
  const bool allowLossyConversion;
};

template <typename OpTy, TosaNarrowKind Kind>
class ConvertTypedOp : public OpConversionPattern<OpTy> {
public:
  ConvertTypedOp(TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<OpTy>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    return convertGenericOp<Kind>(op, adaptor.getOperands(), rewriter,
                                  this->getTypeConverter(),
                                  /*allowLossyConversion=*/false);
  }
};

// ---------------------------------------------------------------------------
// Kind-specific helpers and patterns
// ---------------------------------------------------------------------------

// Casts get extra checking so we only narrow when it is probably safe.
template <TosaNarrowKind Kind>
class ConvertCastOpWithBoundsChecking
    : public OpConversionPattern<tosa::CastOp> {
  using OpConversionPattern<tosa::CastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tosa::CastOp op, typename tosa::CastOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    const auto inputType = dyn_cast<ShapedType>(adaptor.getInput().getType());
    const auto resultType = dyn_cast<ShapedType>(op.getResult().getType());
    if (!inputType || !resultType)
      return failure();

    const TypeConverter *typeConverter = this->getTypeConverter();
    if (failed(verifyCastDoesNotLosePrecision<Kind>(op, inputType, resultType,
                                                    rewriter)))
      return failure();

    rewriter.replaceOpWithNewOp<tosa::CastOp>(
        op, typeConverter->convertType(resultType), adaptor.getInput());
    return success();
  }
};

// ArgMax indices must fit the axis dimension, so we guard the integer rewrite.
class ConvertArgMaxOpWithBoundsChecking
    : public OpConversionPattern<tosa::ArgMaxOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tosa::ArgMaxOp op, typename tosa::ArgMaxOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    const int32_t axis = op.getAxis();
    const auto inputType = dyn_cast<ShapedType>(adaptor.getInput().getType());
    if (!inputType || !inputType.isStaticDim(axis))
      return rewriter.notifyMatchFailure(
          op, "Requires a static axis dimension for bounds checking.");
    const int64_t axisDim = inputType.getDimSize(axis);
    if (axisDim >= std::numeric_limits<int32_t>::max())
      return rewriter.notifyMatchFailure(
          op, "Axis dimension is too large to narrow safely.");

    const Type resultType = op.getOutput().getType();
    const Type newResultType =
        this->getTypeConverter()->convertType(resultType);
    rewriter.replaceOpWithNewOp<tosa::ArgMaxOp>(op, newResultType,
                                                adaptor.getInput(), axis);
    return success();
  }
};

template <TosaNarrowKind Kind>
class ConvertClampOpWithBoundsChecking
    : public OpConversionPattern<tosa::ClampOp> {
  static_assert(Kind == TosaNarrowKind::Int64ToInt32,
                "Clamp bounds checking only supported for integer narrowing");
  using OpConversionPattern<tosa::ClampOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tosa::ClampOp op, typename tosa::ClampOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto minAttr = dyn_cast<IntegerAttr>(op.getMinValAttr());
    auto maxAttr = dyn_cast<IntegerAttr>(op.getMaxValAttr());
    if (!minAttr || !maxAttr)
      return rewriter.notifyMatchFailure(
          op, "Clamp attributes must be integer constants.");

    const int64_t min = minAttr.getInt();
    const int64_t max = maxAttr.getInt();
    if (min < std::numeric_limits<int32_t>::min() ||
        max > std::numeric_limits<int32_t>::max())
      return rewriter.notifyMatchFailure(
          op, "Clamp bounds exceed int32 range. Narrowing may lose data.");

    const Type resultType = op.getOutput().getType();
    const Type newResultType =
        this->getTypeConverter()->convertType(resultType);
    const auto newResultShaped = dyn_cast<ShapedType>(newResultType);
    if (!newResultShaped)
      return failure();
    const auto newElementType =
        dyn_cast<IntegerType>(newResultShaped.getElementType());
    if (!newElementType)
      return failure();

    const IntegerAttr newMinAttr = IntegerAttr::get(newElementType, min);
    const IntegerAttr newMaxAttr = IntegerAttr::get(newElementType, max);

    rewriter.replaceOpWithNewOp<tosa::ClampOp>(op, newResultType,
                                               adaptor.getInput(), newMinAttr,
                                               newMaxAttr, op.getNanModeAttr());
    return success();
  }
};

// Shared implementation for both narrowing passes; the mode decides which
// element types and attribute payloads participate.
template <TosaNarrowKind Kind>
LogicalResult runTosaNarrowing(Operation *op, bool aggressiveRewrite,
                               bool convertFunctionBoundaries) {
  MLIRContext *context = op->getContext();
  const bool allowLossyConversion = aggressiveRewrite;

  TypeConverter typeConverter;
  typeConverter.addConversion([](Type type) -> Type { return type; });

  typeConverter.addConversion(
      [](IntegerType type) -> Type { return convertInteger<Kind>(type); });
  typeConverter.addConversion(
      [](FloatType type) -> Type { return convertFloat<Kind>(type); });
  typeConverter.addConversion([&typeConverter](RankedTensorType type) -> Type {
    Type elementType = type.getElementType();
    if (!isSourceElement<Kind>(elementType))
      return type;
    Type converted = typeConverter.convertType(elementType);
    if (!converted || converted == elementType)
      return type;
    return RankedTensorType::get(type.getShape(), converted,
                                 type.getEncoding());
  });
  typeConverter.addConversion(
      [&typeConverter](UnrankedTensorType type) -> Type {
        Type elementType = type.getElementType();
        if (!isSourceElement<Kind>(elementType))
          return type;
        Type converted = typeConverter.convertType(elementType);
        if (!converted || converted == elementType)
          return type;
        return UnrankedTensorType::get(converted);
      });

  const auto materializeCast = [](OpBuilder &builder, Type resultType,
                                  ValueRange inputs, Location loc) -> Value {
    if (inputs.size() != 1)
      return Value();
    return tosa::CastOp::create(builder, loc, resultType, inputs.front());
  };
  typeConverter.addSourceMaterialization(materializeCast);
  typeConverter.addTargetMaterialization(materializeCast);

  typeConverter.addTypeAttributeConversion(
      [&typeConverter, allowLossyConversion](ShapedType type,
                                             DenseResourceElementsAttr attr)
          -> TypeConverter::AttributeConversionResult {
        FailureOr<Attribute> converted = convertDenseResourceElementsAttr<Kind>(
            type, attr, typeConverter, allowLossyConversion);
        if (failed(converted))
          return TypeConverter::AttributeConversionResult::abort();
        return TypeConverter::AttributeConversionResult::result(
            converted.value());
      });

  if constexpr (Kind == TosaNarrowKind::Int64ToInt32) {
    typeConverter.addTypeAttributeConversion(
        [allowLossyConversion](IntegerType /*type*/, IntegerAttr attribute)
            -> TypeConverter::AttributeConversionResult {
          FailureOr<Attribute> converted =
              tryConvertScalarAttribute<Kind>(attribute, allowLossyConversion);
          if (failed(converted))
            return TypeConverter::AttributeConversionResult::abort();
          return TypeConverter::AttributeConversionResult::result(
              converted.value());
        });
    typeConverter.addTypeAttributeConversion(
        [&typeConverter, allowLossyConversion](ShapedType type,
                                               DenseIntElementsAttr attr)
            -> TypeConverter::AttributeConversionResult {
          FailureOr<Attribute> converted = convertDenseIntElementsAttr<Kind>(
              type, attr, typeConverter, allowLossyConversion);
          if (failed(converted))
            return TypeConverter::AttributeConversionResult::abort();
          return TypeConverter::AttributeConversionResult::result(
              converted.value());
        });
  } else if constexpr (Kind == TosaNarrowKind::Float64ToFloat32) {
    typeConverter.addTypeAttributeConversion(
        [allowLossyConversion](FloatType /*type*/, FloatAttr attribute)
            -> TypeConverter::AttributeConversionResult {
          FailureOr<Attribute> converted =
              tryConvertScalarAttribute<Kind>(attribute, allowLossyConversion);
          if (failed(converted))
            return TypeConverter::AttributeConversionResult::abort();
          return TypeConverter::AttributeConversionResult::result(
              converted.value());
        });
    typeConverter.addTypeAttributeConversion(
        [&typeConverter, allowLossyConversion](ShapedType type,
                                               DenseFPElementsAttr attr)
            -> TypeConverter::AttributeConversionResult {
          FailureOr<Attribute> converted = convertDenseFPElementsAttr<Kind>(
              type, attr, typeConverter, allowLossyConversion);
          if (failed(converted))
            return TypeConverter::AttributeConversionResult::abort();
          return TypeConverter::AttributeConversionResult::result(
              converted.value());
        });
  }

  ConversionTarget target(*context);
  target.addDynamicallyLegalDialect<tosa::TosaDialect>(
      [&typeConverter](Operation *op) {
        return typeConverter.isLegal(op->getResultTypes()) &&
               typeConverter.isLegal(op->getOperandTypes());
      });
  if (convertFunctionBoundaries) {
    target.addDynamicallyLegalOp<func::FuncOp>(
        [&typeConverter](func::FuncOp op) {
          return typeConverter.isSignatureLegal(op.getFunctionType()) &&
                 typeConverter.isLegal(&op.getBody());
        });
    target.addDynamicallyLegalOp<func::ReturnOp>([](func::ReturnOp op) {
      const FunctionType funcType =
          op->getParentOfType<func::FuncOp>().getFunctionType();
      return llvm::equal(op.getOperandTypes(), funcType.getResults());
    });
  } else {
    target.addDynamicallyLegalOp<func::FuncOp>(
        [](func::FuncOp) { return true; });
    target.addDynamicallyLegalOp<func::ReturnOp>(
        [](func::ReturnOp) { return true; });
  }

  RewritePatternSet patterns(context);
  if (convertFunctionBoundaries) {
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    populateReturnOpTypeConversionPattern(patterns, typeConverter);
  }
  if (aggressiveRewrite) {
    patterns.add<ConvertGenericOp<Kind>>(typeConverter, context,
                                         allowLossyConversion);
  } else {
    if constexpr (Kind == TosaNarrowKind::Int64ToInt32) {
      patterns.add<ConvertArgMaxOpWithBoundsChecking>(typeConverter, context);
      patterns.add<ConvertClampOpWithBoundsChecking<Kind>>(typeConverter,
                                                           context);
    }
    patterns.add<ConvertTypedOp<tosa::ConstOp, Kind>>(typeConverter, context);
    patterns.add<ConvertTypedOp<tosa::ConcatOp, Kind>>(typeConverter, context);
    patterns.add<ConvertTypedOp<tosa::PadOp, Kind>>(typeConverter, context);
    patterns.add<ConvertTypedOp<tosa::ReshapeOp, Kind>>(typeConverter, context);
    patterns.add<ConvertTypedOp<tosa::ReverseOp, Kind>>(typeConverter, context);
    patterns.add<ConvertTypedOp<tosa::SliceOp, Kind>>(typeConverter, context);
    patterns.add<ConvertTypedOp<tosa::TileOp, Kind>>(typeConverter, context);
    patterns.add<ConvertTypedOp<tosa::TransposeOp, Kind>>(typeConverter,
                                                          context);
    patterns.add<ConvertTypedOp<tosa::IdentityOp, Kind>>(typeConverter,
                                                         context);
    patterns.add<ConvertCastOpWithBoundsChecking<Kind>>(typeConverter, context);
    patterns.add<ConvertTypedOp<tosa::IfOp, Kind>>(typeConverter, context);
    patterns.add<ConvertTypedOp<tosa::WhileOp, Kind>>(typeConverter, context);
    patterns.add<ConvertTypedOp<tosa::YieldOp, Kind>>(typeConverter, context);
  }

  if (failed(applyFullConversion(op, target, std::move(patterns))))
    return failure();
  return success();
}

// ---------------------------------------------------------------------------
// Pass adapters that forward to the shared implementation
// ---------------------------------------------------------------------------

struct TosaNarrowI64ToI32
    : public tosa::impl::TosaNarrowI64ToI32PassBase<TosaNarrowI64ToI32> {
  using Base = tosa::impl::TosaNarrowI64ToI32PassBase<TosaNarrowI64ToI32>;

  TosaNarrowI64ToI32() = default;

  explicit TosaNarrowI64ToI32(const TosaNarrowI64ToI32PassOptions &options) {
    this->aggressiveRewrite = options.aggressiveRewrite;
    this->convertFunctionBoundaries = options.convertFunctionBoundaries;
  }

  void runOnOperation() override {
    if (failed(runTosaNarrowing<TosaNarrowKind::Int64ToInt32>(
            getOperation(), this->aggressiveRewrite,
            this->convertFunctionBoundaries)))
      signalPassFailure();
  }
};

struct TosaNarrowF64ToF32
    : public tosa::impl::TosaNarrowF64ToF32PassBase<TosaNarrowF64ToF32> {
  using Base = tosa::impl::TosaNarrowF64ToF32PassBase<TosaNarrowF64ToF32>;

  TosaNarrowF64ToF32() = default;

  explicit TosaNarrowF64ToF32(const TosaNarrowF64ToF32PassOptions &options) {
    this->aggressiveRewrite = options.aggressiveRewrite;
    this->convertFunctionBoundaries = options.convertFunctionBoundaries;
  }

  void runOnOperation() override {
    if (failed(runTosaNarrowing<TosaNarrowKind::Float64ToFloat32>(
            getOperation(), this->aggressiveRewrite,
            this->convertFunctionBoundaries)))
      signalPassFailure();
  }
};

} // namespace
