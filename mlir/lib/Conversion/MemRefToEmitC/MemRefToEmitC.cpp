//===- MemRefToEmitC.cpp - MemRef to EmitC conversion ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements patterns to convert memref ops into emitc ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/MemRefToEmitC/MemRefToEmitC.h"

#include "mlir/Conversion/ConvertToEmitC/ToEmitCInterface.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"
#include <cstdint>

using namespace mlir;

namespace {
/// Implement the interface to convert MemRef to EmitC.
struct MemRefToEmitCDialectInterface : public ConvertToEmitCPatternInterface {
  using ConvertToEmitCPatternInterface::ConvertToEmitCPatternInterface;

  /// Hook for derived dialect interface to provide conversion patterns
  /// and mark dialect legal for the conversion target.
  void populateConvertToEmitCConversionPatterns(
      ConversionTarget &target, TypeConverter &typeConverter,
      RewritePatternSet &patterns) const final {
    populateMemRefToEmitCTypeConversion(typeConverter);
    populateMemRefToEmitCConversionPatterns(patterns, typeConverter);
  }
};
} // namespace

void mlir::registerConvertMemRefToEmitCInterface(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, memref::MemRefDialect *dialect) {
    dialect->addInterfaces<MemRefToEmitCDialectInterface>();
  });
}

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//

namespace {
struct ConvertAlloca final : public OpConversionPattern<memref::AllocaOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::AllocaOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {

    if (!op.getType().hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          op.getLoc(), "cannot transform alloca with dynamic shape");
    }

    if (op.getAlignment().value_or(1) > 1) {
      // TODO: Allow alignment if it is not more than the natural alignment
      // of the C array.
      return rewriter.notifyMatchFailure(
          op.getLoc(), "cannot transform alloca with alignment requirement");
    }

    auto resultTy = getTypeConverter()->convertType(op.getType());
    if (!resultTy) {
      return rewriter.notifyMatchFailure(op.getLoc(), "cannot convert type");
    }
    auto noInit = emitc::OpaqueAttr::get(getContext(), "");
    rewriter.replaceOpWithNewOp<emitc::VariableOp>(op, resultTy, noInit);
    return success();
  }
};

Type convertMemRefType(MemRefType opTy, const TypeConverter *typeConverter) {
  Type resultTy;
  if (opTy.getRank() == 0) {
    resultTy = typeConverter->convertType(mlir::getElementTypeOrSelf(opTy));
  } else {
    resultTy = typeConverter->convertType(opTy);
  }
  return resultTy;
}

struct ConvertAlloc final : public OpConversionPattern<memref::AllocOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(memref::AllocOp allocOp, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = allocOp.getLoc();
    MemRefType memrefType = allocOp.getType();
    if (!memrefType.hasStaticShape()) {
      // TODO: Handle Dynamic shapes in the future. If the size
      // of the allocation is the result of some function, we could
      // potentially evaluate the function and use the result in the call to
      // allocate.
      return rewriter.notifyMatchFailure(
          loc, "cannot transform alloc with dynamic shape");
    }

    mlir::Type sizeTType = mlir::emitc::SizeTType::get(rewriter.getContext());
    Type elementType = memrefType.getElementType();
    mlir::emitc::CallOpaqueOp sizeofElementOp =
        rewriter.create<mlir::emitc::CallOpaqueOp>(
            loc, sizeTType, rewriter.getStringAttr("sizeof"),
            mlir::ValueRange{},
            mlir::ArrayAttr::get(rewriter.getContext(),
                                 {mlir::TypeAttr::get(elementType)}));
    mlir::Value sizeofElement = sizeofElementOp.getResult(0);

    unsigned int elementWidth = elementType.getIntOrFloatBitWidth();
    IntegerAttr indexAttr = rewriter.getIndexAttr(elementWidth);

    mlir::Value numElements;
    numElements = rewriter.create<emitc::ConstantOp>(
        loc, rewriter.getIndexType(), indexAttr);
    mlir::Value totalSizeBytes = rewriter.create<emitc::MulOp>(
        loc, sizeTType, sizeofElement, numElements);

    int64_t alignment = alignedAllocationGetAlignment(allocOp, elementWidth);
    mlir::Value alignmentValue = rewriter.create<emitc::ConstantOp>(
        loc, sizeTType,
        rewriter.getIntegerAttr(rewriter.getIndexType(), alignment));

    emitc::CallOpaqueOp alignedAllocCall = rewriter.create<emitc::CallOpaqueOp>(
        loc,
        emitc::PointerType::get(
            mlir::emitc::OpaqueType::get(rewriter.getContext(), "void")),
        rewriter.getStringAttr("aligned_alloc"),
        mlir::ValueRange{alignmentValue, totalSizeBytes});
    emitc::PointerType targetPointerType = emitc::PointerType::get(elementType);
    emitc::CastOp castOp = rewriter.create<emitc::CastOp>(
        loc, targetPointerType, alignedAllocCall.getResult(0));

    rewriter.replaceOp(allocOp, castOp);
    return success();
  }

  /// The minimum alignment to use with aligned_alloc (has to be a power of 2).
  static constexpr uint64_t kMinAlignedAllocAlignment = 16UL;

  /// Computes the alignment for aligned_alloc used to allocate the buffer for
  /// the memory allocation op.
  ///
  /// Aligned_alloc requires the allocation size to be a power of two, and the
  /// allocation size to be a multiple of the alignment.
  int64_t alignedAllocationGetAlignment(memref::AllocOp op,
                                        unsigned int elementWidth) const {
    if (std::optional<uint64_t> alignment = op.getAlignment())
      return *alignment;

    // Whenever we don't have alignment set, we will use an alignment
    // consistent with the element type; since the allocation size has to be a
    // power of two, we will bump to the next power of two if it isn't.
    return std::max(kMinAlignedAllocAlignment,
                    llvm::PowerOf2Ceil(elementWidth));
  }
};

struct ConvertGlobal final : public OpConversionPattern<memref::GlobalOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::GlobalOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {
    MemRefType opTy = op.getType();
    if (!op.getType().hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          op.getLoc(), "cannot transform global with dynamic shape");
    }

    if (op.getAlignment().value_or(1) > 1) {
      // TODO: Extend GlobalOp to specify alignment via the `alignas` specifier.
      return rewriter.notifyMatchFailure(
          op.getLoc(), "global variable with alignment requirement is "
                       "currently not supported");
    }

    Type resultTy = convertMemRefType(opTy, getTypeConverter());

    if (!resultTy) {
      return rewriter.notifyMatchFailure(op.getLoc(),
                                         "cannot convert result type");
    }

    SymbolTable::Visibility visibility = SymbolTable::getSymbolVisibility(op);
    if (visibility != SymbolTable::Visibility::Public &&
        visibility != SymbolTable::Visibility::Private) {
      return rewriter.notifyMatchFailure(
          op.getLoc(),
          "only public and private visibility is currently supported");
    }
    // We are explicit in specifing the linkage because the default linkage
    // for constants is different in C and C++.
    bool staticSpecifier = visibility == SymbolTable::Visibility::Private;
    bool externSpecifier = !staticSpecifier;

    Attribute initialValue = operands.getInitialValueAttr();
    if (opTy.getRank() == 0) {
      auto elementsAttr = llvm::cast<ElementsAttr>(*op.getInitialValue());
      initialValue = elementsAttr.getSplatValue<Attribute>();
    }
    if (isa_and_present<UnitAttr>(initialValue))
      initialValue = {};

    rewriter.replaceOpWithNewOp<emitc::GlobalOp>(
        op, operands.getSymName(), resultTy, initialValue, externSpecifier,
        staticSpecifier, operands.getConstant());
    return success();
  }
};

struct ConvertGetGlobal final
    : public OpConversionPattern<memref::GetGlobalOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::GetGlobalOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {

    MemRefType opTy = op.getType();
    Type resultTy = convertMemRefType(opTy, getTypeConverter());

    if (!resultTy) {
      return rewriter.notifyMatchFailure(op.getLoc(),
                                         "cannot convert result type");
    }

    if (opTy.getRank() == 0) {
      emitc::LValueType lvalueType = emitc::LValueType::get(resultTy);
      emitc::GetGlobalOp globalLValue = emitc::GetGlobalOp::create(
          rewriter, op.getLoc(), lvalueType, operands.getNameAttr());
      emitc::PointerType pointerType = emitc::PointerType::get(resultTy);
      rewriter.replaceOpWithNewOp<emitc::ApplyOp>(
          op, pointerType, rewriter.getStringAttr("&"), globalLValue);
      return success();
    }
    rewriter.replaceOpWithNewOp<emitc::GetGlobalOp>(op, resultTy,
                                                    operands.getNameAttr());
    return success();
  }
};

struct ConvertLoad final : public OpConversionPattern<memref::LoadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::LoadOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {

    auto resultTy = getTypeConverter()->convertType(op.getType());
    if (!resultTy) {
      return rewriter.notifyMatchFailure(op.getLoc(), "cannot convert type");
    }

    auto arrayValue =
        dyn_cast<TypedValue<emitc::ArrayType>>(operands.getMemref());
    if (!arrayValue) {
      return rewriter.notifyMatchFailure(op.getLoc(), "expected array type");
    }

    auto subscript = emitc::SubscriptOp::create(
        rewriter, op.getLoc(), arrayValue, operands.getIndices());

    rewriter.replaceOpWithNewOp<emitc::LoadOp>(op, resultTy, subscript);
    return success();
  }
};

struct ConvertStore final : public OpConversionPattern<memref::StoreOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::StoreOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto arrayValue =
        dyn_cast<TypedValue<emitc::ArrayType>>(operands.getMemref());
    if (!arrayValue) {
      return rewriter.notifyMatchFailure(op.getLoc(), "expected array type");
    }

    auto subscript = emitc::SubscriptOp::create(
        rewriter, op.getLoc(), arrayValue, operands.getIndices());
    rewriter.replaceOpWithNewOp<emitc::AssignOp>(op, subscript,
                                                 operands.getValue());
    return success();
  }
};
} // namespace

void mlir::populateMemRefToEmitCTypeConversion(TypeConverter &typeConverter) {
  typeConverter.addConversion(
      [&](MemRefType memRefType) -> std::optional<Type> {
        if (!memRefType.hasStaticShape() ||
            !memRefType.getLayout().isIdentity() || memRefType.getRank() == 0 ||
            llvm::is_contained(memRefType.getShape(), 0)) {
          return {};
        }
        Type convertedElementType =
            typeConverter.convertType(memRefType.getElementType());
        if (!convertedElementType)
          return {};
        return emitc::ArrayType::get(memRefType.getShape(),
                                     convertedElementType);
      });

  auto materializeAsUnrealizedCast = [](OpBuilder &builder, Type resultType,
                                        ValueRange inputs,
                                        Location loc) -> Value {
    if (inputs.size() != 1)
      return Value();

    return UnrealizedConversionCastOp::create(builder, loc, resultType, inputs)
        .getResult(0);
  };

  typeConverter.addSourceMaterialization(materializeAsUnrealizedCast);
  typeConverter.addTargetMaterialization(materializeAsUnrealizedCast);
}

void mlir::populateMemRefToEmitCConversionPatterns(
    RewritePatternSet &patterns, const TypeConverter &converter) {
  patterns.add<ConvertAlloca, ConvertAlloc, ConvertGlobal, ConvertGetGlobal,
               ConvertLoad, ConvertStore>(converter, patterns.getContext());
}
