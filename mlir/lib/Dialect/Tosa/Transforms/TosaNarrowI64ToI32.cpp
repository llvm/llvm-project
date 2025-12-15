//===- TosaNarrowI64ToI32.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass narrows TOSA operations with 64-bit integer tensor types to
// 32-bit integer tensor types. This can be useful for backends that do not
// support the EXT-INT64 extension of TOSA. The pass has two options:
//
// - aggressive-rewrite - If enabled, all TOSA operations are rewritten,
//     regardless or whether the narrowing is safe. This option may lead to
//     data loss if not used carefully.
// - convert-function-boundaries - If enabled, the pass will convert function
//     I/O types as well. Otherwise casts will be inserted at the I/O
//     boundaries.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace tosa {
#define GEN_PASS_DEF_TOSANARROWI64TOI32PASS
#include "mlir/Dialect/Tosa/Transforms/Passes.h.inc"
} // namespace tosa
} // namespace mlir

using namespace mlir;
using namespace mlir::tosa;

namespace {

LogicalResult convertGenericOp(Operation *op, ValueRange operands,
                               ConversionPatternRewriter &rewriter,
                               const TypeConverter *typeConverter) {
  // Convert types of results
  SmallVector<Type, 4> newResults;
  if (failed(typeConverter->convertTypes(op->getResultTypes(), newResults)))
    return failure();

  // Create a new operation state
  OperationState state(op->getLoc(), op->getName().getStringRef(), operands,
                       newResults, {}, op->getSuccessors());

  for (const NamedAttribute &namedAttribute : op->getAttrs()) {
    const Attribute attribute = namedAttribute.getValue();

    // Convert integer attribute type
    if (const auto intAttr = dyn_cast<IntegerAttr>(attribute)) {
      const std::optional<Attribute> convertedAttribute =
          typeConverter->convertTypeAttribute(intAttr.getType(), attribute);
      state.addAttribute(namedAttribute.getName(), convertedAttribute.value());
      continue;
    }

    if (const auto typeAttr = dyn_cast<TypeAttr>(attribute)) {
      Type type = typeAttr.getValue();
      const std::optional<Attribute> convertedAttribute =
          typeConverter->convertTypeAttribute(type, attribute);
      if (!convertedAttribute)
        return rewriter.notifyMatchFailure(op,
                                           "Failed to convert type attribute.");
      state.addAttribute(namedAttribute.getName(), convertedAttribute.value());
      continue;
    }

    if (const auto denseElementsAttr = dyn_cast<DenseElementsAttr>(attribute)) {
      const Type type = denseElementsAttr.getType();
      const std::optional<Attribute> convertedAttribute =
          typeConverter->convertTypeAttribute(type, denseElementsAttr);
      if (!convertedAttribute)
        return rewriter.notifyMatchFailure(
            op, "Failed to convert dense elements attribute.");
      state.addAttribute(namedAttribute.getName(), convertedAttribute.value());
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

// ===========================
// Aggressive rewrite patterns
// ===========================

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

    return convertGenericOp(op, operands, rewriter, typeConverter);
  }
};

// ===============================
// Bounds checked rewrite patterns
// ===============================

class ConvertArgMaxOpWithBoundsChecking
    : public OpConversionPattern<tosa::ArgMaxOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tosa::ArgMaxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    // Output type can be narrowed based on the size of the axis dimension
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
    const Type newResultType = typeConverter->convertType(resultType);
    rewriter.replaceOpWithNewOp<tosa::ArgMaxOp>(op, newResultType,
                                                adaptor.getInput(), axis);
    return success();
  }
};

class ConvertCastOpWithBoundsChecking
    : public OpConversionPattern<tosa::CastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tosa::CastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    const auto inputType = dyn_cast<ShapedType>(adaptor.getInput().getType());
    const auto resultType = dyn_cast<ShapedType>(op.getResult().getType());
    if (!inputType || !resultType)
      return failure();

    const auto elementInputIntType =
        dyn_cast<IntegerType>(inputType.getElementType());
    const auto elementResultIntType =
        dyn_cast<IntegerType>(resultType.getElementType());
    if (elementInputIntType && elementResultIntType &&
        elementInputIntType.getWidth() > elementResultIntType.getWidth())
      return rewriter.notifyMatchFailure(
          op, "Narrowing cast may lead to data loss.");

    rewriter.replaceOpWithNewOp<tosa::CastOp>(
        op, typeConverter->convertType(resultType), adaptor.getInput());
    return success();
  }
};

template <typename OpTy>
class ConvertTypedOp : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    return convertGenericOp(op, adaptor.getOperands(), rewriter,
                            this->getTypeConverter());
  }
};

struct TosaNarrowI64ToI32
    : public tosa::impl::TosaNarrowI64ToI32PassBase<TosaNarrowI64ToI32> {
public:
  explicit TosaNarrowI64ToI32() = default;
  explicit TosaNarrowI64ToI32(const TosaNarrowI64ToI32PassOptions &options)
      : TosaNarrowI64ToI32() {
    this->aggressiveRewrite = options.aggressiveRewrite;
    this->convertFunctionBoundaries = options.convertFunctionBoundaries;
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) -> Type { return type; });
    typeConverter.addConversion([](IntegerType type) -> Type {
      if (!type.isInteger(64))
        return type;
      return IntegerType::get(type.getContext(), 32);
    });
    typeConverter.addConversion(
        [&typeConverter](RankedTensorType type) -> Type {
          const Type elementType = type.getElementType();
          if (!elementType.isInteger(64))
            return type;
          return RankedTensorType::get(type.getShape(),
                                       typeConverter.convertType(elementType));
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
        [](IntegerType type, IntegerAttr attribute) -> Attribute {
          const APInt value = attribute.getValue().truncSSat(32);
          return IntegerAttr::get(IntegerType::get(type.getContext(), 32),
                                  value);
        });
    typeConverter.addTypeAttributeConversion(
        [&typeConverter](ShapedType type,
                         DenseIntElementsAttr attr) -> Attribute {
          const ShapedType newType =
              cast<ShapedType>(typeConverter.convertType(type));
          const auto oldElementType = cast<IntegerType>(type.getElementType());
          const auto newElementType =
              cast<IntegerType>(newType.getElementType());
          if (oldElementType.getWidth() == newElementType.getWidth())
            return attr;

          DenseElementsAttr mapped =
              attr.mapValues(newElementType, [&](const APInt &v) {
                return v.truncSSat(newElementType.getWidth());
              });
          return mapped;
        });

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
          [](func::FuncOp op) { return true; });
      target.addDynamicallyLegalOp<func::ReturnOp>(
          [](func::ReturnOp op) { return true; });
    }

    RewritePatternSet patterns(context);
    if (convertFunctionBoundaries) {
      populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
          patterns, typeConverter);
      populateReturnOpTypeConversionPattern(patterns, typeConverter);
    }
    if (aggressiveRewrite) {
      patterns.add<ConvertGenericOp>(typeConverter, context);
    } else {
      // Tensor
      patterns.add<ConvertArgMaxOpWithBoundsChecking>(typeConverter, context);
      // Data layout
      patterns.add<ConvertTypedOp<tosa::ConcatOp>>(typeConverter, context);
      patterns.add<ConvertTypedOp<tosa::PadOp>>(typeConverter, context);
      patterns.add<ConvertTypedOp<tosa::ReshapeOp>>(typeConverter, context);
      patterns.add<ConvertTypedOp<tosa::ReverseOp>>(typeConverter, context);
      patterns.add<ConvertTypedOp<tosa::SliceOp>>(typeConverter, context);
      patterns.add<ConvertTypedOp<tosa::TileOp>>(typeConverter, context);
      patterns.add<ConvertTypedOp<tosa::TransposeOp>>(typeConverter, context);
      patterns.add<ConvertTypedOp<tosa::IdentityOp>>(typeConverter, context);
      // Type conversion
      patterns.add<ConvertCastOpWithBoundsChecking>(typeConverter, context);
      // Controlflow
      patterns.add<ConvertTypedOp<tosa::IfOp>>(typeConverter, context);
      patterns.add<ConvertTypedOp<tosa::WhileOp>>(typeConverter, context);
    }

    if (failed(
            applyFullConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
