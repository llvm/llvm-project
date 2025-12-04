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

#include <limits>
#include <type_traits>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
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
static bool isSourceInteger(IntegerType type) {
  if constexpr (Kind == TosaNarrowKind::Int64ToInt32)
    return type.isInteger(64);
  return false;
}

template <TosaNarrowKind Kind>
static bool isSourceFloat(FloatType type) {
  if constexpr (Kind == TosaNarrowKind::Float64ToFloat32)
    return type.isF64();
  return false;
}

template <TosaNarrowKind Kind>
static Type convertInteger(IntegerType type) {
  if (!isSourceInteger<Kind>(type))
    return type;
  if constexpr (Kind == TosaNarrowKind::Int64ToInt32)
    return IntegerType::get(type.getContext(), 32);
  return type;
}

template <TosaNarrowKind Kind>
static Type convertFloat(FloatType type) {
  if (!isSourceFloat<Kind>(type))
    return type;
  if constexpr (Kind == TosaNarrowKind::Float64ToFloat32)
    return Float32Type::get(type.getContext());
  return type;
}

template <TosaNarrowKind Kind>
static bool isSourceElement(Type type) {
  if (auto intTy = dyn_cast<IntegerType>(type))
    return isSourceInteger<Kind>(intTy);
  if (auto floatTy = dyn_cast<FloatType>(type))
    return isSourceFloat<Kind>(floatTy);
  return false;
}

template <TosaNarrowKind Kind>
static Type convertElement(Type type) {
  if (auto intTy = dyn_cast<IntegerType>(type))
    return convertInteger<Kind>(intTy);
  if (auto floatTy = dyn_cast<FloatType>(type))
    return convertFloat<Kind>(floatTy);
  return type;
}

template <TosaNarrowKind Kind>
static bool typeNeedsConversion(Type type) {
  if (auto shaped = dyn_cast<ShapedType>(type))
    return isSourceElement<Kind>(shaped.getElementType());
  return isSourceElement<Kind>(type);
}

// Narrows scalar constant attributes so they keep matching the converted
// element types.
template <TosaNarrowKind Kind>
static bool tryConvertScalarAttribute(Attribute attribute,
                                      Attribute &resultAttr) {
  if constexpr (Kind == TosaNarrowKind::Int64ToInt32) {
    if (const auto intAttr = dyn_cast<IntegerAttr>(attribute)) {
      if (const auto intType = dyn_cast<IntegerType>(intAttr.getType());
          intType && isSourceInteger<Kind>(intType)) {
        const auto convertedType =
            cast<IntegerType>(convertInteger<Kind>(intType));
        const APInt truncated =
            intAttr.getValue().truncSSat(convertedType.getWidth());
        resultAttr = IntegerAttr::get(convertedType, truncated);
        return true;
      }
    }
  }

  if constexpr (Kind == TosaNarrowKind::Float64ToFloat32) {
    if (const auto floatAttr = dyn_cast<FloatAttr>(attribute)) {
      if (const auto floatType = dyn_cast<FloatType>(floatAttr.getType());
          floatType && isSourceFloat<Kind>(floatType)) {
        const auto convertedType =
            cast<FloatType>(convertFloat<Kind>(floatType));
        APFloat value = floatAttr.getValue();
        bool losesInfo = false;
        value.convert(convertedType.getFloatSemantics(),
                      APFloat::rmNearestTiesToEven, &losesInfo);
        resultAttr = FloatAttr::get(convertedType, value);
        return true;
      }
    }
  }

  return false;
}

template <TosaNarrowKind Kind, typename AttrT>
static LogicalResult
convertAttributeWithTypeConverter(AttrT attr, Type type,
                                  const TypeConverter *typeConverter,
                                  Attribute &resultAttr) {
  if (!typeNeedsConversion<Kind>(type)) {
    resultAttr = attr;
    return success();
  }

  const std::optional<Attribute> convertedAttribute =
      typeConverter->convertTypeAttribute(type, attr);
  if (!convertedAttribute)
    return failure();

  resultAttr = convertedAttribute.value();
  return success();
}

// Rejects cast rewrites that would lose precision (unless aggressive mode is
// enabled).
template <TosaNarrowKind Kind>
static LogicalResult
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
  }

  if constexpr (Kind == TosaNarrowKind::Float64ToFloat32) {
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

template <TosaNarrowKind Kind, typename DenseAttrT, typename ElementTypeT,
          typename MapValueFn>
static Attribute convertDenseElementsAttr(ShapedType type, DenseAttrT attr,
                                          const TypeConverter &typeConverter,
                                          MapValueFn &&mapValueFn) {
  const auto oldElementType = dyn_cast<ElementTypeT>(type.getElementType());
  if (!oldElementType)
    return attr;

  if constexpr (std::is_same_v<ElementTypeT, IntegerType>) {
    if (!isSourceInteger<Kind>(oldElementType))
      return attr;
  } else {
    if (!isSourceFloat<Kind>(oldElementType))
      return attr;
  }

  const auto newType =
      dyn_cast_or_null<ShapedType>(typeConverter.convertType(type));
  if (!newType)
    return attr;

  const auto newElementType = dyn_cast<ElementTypeT>(newType.getElementType());
  if (!newElementType)
    return attr;

  return attr.mapValues(newElementType, [&](const auto &value) {
    return mapValueFn(newElementType, value);
  });
}

// ---------------------------------------------------------------------------
// Conversion patterns
// ---------------------------------------------------------------------------

// Applies the narrowing TypeConverter to a single TOSA op, including its
// attributes and nested regions.
template <TosaNarrowKind Kind>
LogicalResult convertGenericOp(Operation *op, ValueRange operands,
                               ConversionPatternRewriter &rewriter,
                               const TypeConverter *typeConverter) {
  SmallVector<Type, 4> newResults;
  if (failed(typeConverter->convertTypes(op->getResultTypes(), newResults)))
    return failure();

  OperationState state(op->getLoc(), op->getName().getStringRef(), operands,
                       newResults, {}, op->getSuccessors());

  // Keep attribute payloads consistent with the converted element types.
  for (const NamedAttribute &namedAttribute : op->getAttrs()) {
    const Attribute attribute = namedAttribute.getValue();

    Attribute convertedAttr;
    if (tryConvertScalarAttribute<Kind>(attribute, convertedAttr)) {
      state.addAttribute(namedAttribute.getName(), convertedAttr);
      continue;
    }

    if (const auto typeAttr = dyn_cast<TypeAttr>(attribute)) {
      if (failed(convertAttributeWithTypeConverter<Kind>(
              typeAttr, typeAttr.getValue(), typeConverter, convertedAttr)))
        return rewriter.notifyMatchFailure(op,
                                           "Failed to convert type attribute.");
      state.addAttribute(namedAttribute.getName(), convertedAttr);
      continue;
    }

    if (const auto denseElementsAttr = dyn_cast<DenseElementsAttr>(attribute)) {
      if (failed(convertAttributeWithTypeConverter<Kind>(
              denseElementsAttr, denseElementsAttr.getType(), typeConverter,
              convertedAttr)))
        return rewriter.notifyMatchFailure(
            op, "Failed to convert dense elements attribute.");
      state.addAttribute(namedAttribute.getName(), convertedAttr);
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
  ConvertGenericOp(TypeConverter &typeConverter, MLIRContext *context)
      : ConversionPattern(typeConverter, MatchAnyOpTypeTag{}, 0, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    if (!isa<tosa::TosaOp>(op))
      return rewriter.notifyMatchFailure(
          op,
          "Support for operations other than TOSA has not been implemented.");

    return convertGenericOp<Kind>(op, operands, rewriter, typeConverter);
  }
};

template <typename OpTy, TosaNarrowKind Kind>
class ConvertTypedOp : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    return convertGenericOp<Kind>(op, adaptor.getOperands(), rewriter,
                                  this->getTypeConverter());
  }
};

// ---------------------------------------------------------------------------
// Kind-specific helpers and patterns
// ---------------------------------------------------------------------------

// Casts get extra checking so we only narrow when it is provably safe.
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

// Shared implementation for both narrowing passes; the mode decides which
// element types and attribute payloads participate.
template <TosaNarrowKind Kind>
LogicalResult runTosaNarrowing(Operation *op, bool aggressiveRewrite,
                               bool convertFunctionBoundaries) {
  MLIRContext *context = op->getContext();

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

  if constexpr (Kind == TosaNarrowKind::Int64ToInt32) {
    typeConverter.addTypeAttributeConversion(
        [](IntegerType type, IntegerAttr attribute) -> Attribute {
          Attribute convertedAttr;
          if (tryConvertScalarAttribute<Kind>(attribute, convertedAttr))
            return convertedAttr;
          return attribute;
        });
    typeConverter.addTypeAttributeConversion([&typeConverter](
                                                 ShapedType type,
                                                 DenseIntElementsAttr attr)
                                                 -> Attribute {
      return convertDenseElementsAttr<Kind, DenseIntElementsAttr, IntegerType>(
          type, attr, typeConverter,
          [](IntegerType newElementType, const APInt &value) {
            return value.truncSSat(newElementType.getWidth());
          });
    });
  }

  if constexpr (Kind == TosaNarrowKind::Float64ToFloat32) {
    typeConverter.addTypeAttributeConversion(
        [](FloatType type, FloatAttr attribute) -> Attribute {
          Attribute convertedAttr;
          if (tryConvertScalarAttribute<Kind>(attribute, convertedAttr))
            return convertedAttr;
          return attribute;
        });
    typeConverter.addTypeAttributeConversion(
        [&typeConverter](ShapedType type,
                         DenseFPElementsAttr attr) -> Attribute {
          return convertDenseElementsAttr<Kind, DenseFPElementsAttr, FloatType>(
              type, attr, typeConverter,
              [](FloatType newElementType, const APFloat &value) {
                APFloat converted(value);
                bool losesInfo = false;
                converted.convert(newElementType.getFloatSemantics(),
                                  APFloat::rmNearestTiesToEven, &losesInfo);
                return converted.bitcastToAPInt();
              });
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
    patterns.add<ConvertGenericOp<Kind>>(typeConverter, context);
  } else {
    if constexpr (Kind == TosaNarrowKind::Int64ToInt32)
      patterns.add<ConvertArgMaxOpWithBoundsChecking>(typeConverter, context);
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
  }
  patterns.add<ConvertTypedOp<tosa::YieldOp, Kind>>(typeConverter, context);

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
