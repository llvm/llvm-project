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
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"
#include <cstdint>
#include <numeric>

using namespace mlir;

static bool isMemRefTypeLegalForEmitC(MemRefType memRefType) {
  return memRefType.hasStaticShape() &&

         !llvm::is_contained(memRefType.getShape(), 0);
}

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

static Value calculateMemrefTotalSizeBytes(Location loc, MemRefType memrefType,
                                           OpBuilder &builder) {
  assert(isMemRefTypeLegalForEmitC(memrefType) &&
         "incompatible memref type for EmitC conversion");

  emitc::CallOpaqueOp elementSize = emitc::CallOpaqueOp::create(
      builder, loc, emitc::SizeTType::get(builder.getContext()),
      builder.getStringAttr("sizeof"), ValueRange{},
      ArrayAttr::get(builder.getContext(),
                     {TypeAttr::get(memrefType.getElementType())}));

  IndexType indexType = builder.getIndexType();
  int64_t numElements = std::accumulate(memrefType.getShape().begin(),
                                        memrefType.getShape().end(), int64_t{1},
                                        std::multiplies<int64_t>());
  emitc::ConstantOp numElementsValue = builder.create<emitc::ConstantOp>(
      loc, indexType, builder.getIndexAttr(numElements));

  Type sizeTType = emitc::SizeTType::get(builder.getContext());
  emitc::MulOp totalSizeBytes = builder.create<emitc::MulOp>(
      loc, sizeTType, elementSize.getResult(0), numElementsValue);

  return totalSizeBytes.getResult();
}

static emitc::ApplyOp
createPointerFromEmitcArray(Location loc, OpBuilder &builder,
                            TypedValue<emitc::ArrayType> arrayValue) {

  emitc::ConstantOp zeroIndex = emitc::ConstantOp::create(
      builder, loc, builder.getIndexType(), builder.getIndexAttr(0));

  int64_t rank = arrayValue.getType().getRank();
  llvm::SmallVector<mlir::Value> indices(rank, zeroIndex);

  emitc::SubscriptOp subPtr =
      emitc::SubscriptOp::create(builder, loc, arrayValue, ValueRange(indices));
  emitc::ApplyOp ptr = emitc::ApplyOp::create(
      builder, loc,
      emitc::PointerType::get(arrayValue.getType().getElementType()),
      builder.getStringAttr("&"), subPtr);

  return ptr;
}

static emitc::ApplyOp
createPointerFromEmitcArray(Location loc, OpBuilder &builder,
                            TypedValue<emitc::ArrayType> arrayValue,
                            int64_t linearOffset) {

  emitc::ConstantOp zeroIndex = emitc::ConstantOp::create(
      builder, loc, builder.getIndexType(), builder.getIndexAttr(linearOffset));

  int64_t rank = arrayValue.getType().getRank();
  llvm::SmallVector<mlir::Value> indices(rank, zeroIndex);

  emitc::SubscriptOp subPtr =
      emitc::SubscriptOp::create(builder, loc, arrayValue, ValueRange(indices));
  emitc::ApplyOp ptr = emitc::ApplyOp::create(
      builder, loc,
      emitc::PointerType::get(arrayValue.getType().getElementType()),
      builder.getStringAttr("&"), subPtr);

  return ptr;
}

struct ConvertAlloc final : public OpConversionPattern<memref::AllocOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(memref::AllocOp allocOp, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = allocOp.getLoc();
    MemRefType memrefType = allocOp.getType();
    if (!isMemRefTypeLegalForEmitC(memrefType)) {
      return rewriter.notifyMatchFailure(
          loc, "incompatible memref type for EmitC conversion");
    }

    Type sizeTType = emitc::SizeTType::get(rewriter.getContext());
    Type elementType = memrefType.getElementType();
    IndexType indexType = rewriter.getIndexType();
    emitc::CallOpaqueOp sizeofElementOp = emitc::CallOpaqueOp::create(
        rewriter, loc, sizeTType, rewriter.getStringAttr("sizeof"),
        ValueRange{},
        ArrayAttr::get(rewriter.getContext(), {TypeAttr::get(elementType)}));

    int64_t numElements = 1;
    for (int64_t dimSize : memrefType.getShape()) {
      numElements *= dimSize;
    }
    Value numElementsValue = emitc::ConstantOp::create(
        rewriter, loc, indexType, rewriter.getIndexAttr(numElements));

    Value totalSizeBytes =
        emitc::MulOp::create(rewriter, loc, sizeTType,
                             sizeofElementOp.getResult(0), numElementsValue);

    emitc::CallOpaqueOp allocCall;
    StringAttr allocFunctionName;
    Value alignmentValue;
    SmallVector<Value, 2> argsVec;
    if (allocOp.getAlignment()) {
      allocFunctionName = rewriter.getStringAttr(alignedAllocFunctionName);
      alignmentValue = emitc::ConstantOp::create(
          rewriter, loc, sizeTType,
          rewriter.getIntegerAttr(indexType,
                                  allocOp.getAlignment().value_or(0)));
      argsVec.push_back(alignmentValue);
    } else {
      allocFunctionName = rewriter.getStringAttr(mallocFunctionName);
    }

    argsVec.push_back(totalSizeBytes);
    ValueRange args(argsVec);

    allocCall = emitc::CallOpaqueOp::create(
        rewriter, loc,
        emitc::PointerType::get(
            emitc::OpaqueType::get(rewriter.getContext(), "void")),
        allocFunctionName, args);

    emitc::PointerType targetPointerType = emitc::PointerType::get(elementType);
    emitc::CastOp castOp = emitc::CastOp::create(
        rewriter, loc, targetPointerType, allocCall.getResult(0));

    rewriter.replaceOp(allocOp, castOp);
    return success();
  }
};

struct ConvertCopy final : public OpConversionPattern<memref::CopyOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::CopyOp copyOp, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = copyOp.getLoc();
    MemRefType srcMemrefType = cast<MemRefType>(copyOp.getSource().getType());
    MemRefType targetMemrefType =
        cast<MemRefType>(copyOp.getTarget().getType());

    if (!isMemRefTypeLegalForEmitC(srcMemrefType))
      return rewriter.notifyMatchFailure(
          loc, "incompatible source memref type for EmitC conversion");

    if (!isMemRefTypeLegalForEmitC(targetMemrefType))
      return rewriter.notifyMatchFailure(
          loc, "incompatible target memref type for EmitC conversion");

    auto srcArrayValue =
        cast<TypedValue<emitc::ArrayType>>(operands.getSource());
    emitc::ApplyOp srcPtr =
        createPointerFromEmitcArray(loc, rewriter, srcArrayValue);

    auto targetArrayValue =
        cast<TypedValue<emitc::ArrayType>>(operands.getTarget());
    emitc::ApplyOp targetPtr =
        createPointerFromEmitcArray(loc, rewriter, targetArrayValue);

    emitc::CallOpaqueOp memCpyCall = rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "memcpy",
        ValueRange{
            targetPtr.getResult(), srcPtr.getResult(),
            calculateMemrefTotalSizeBytes(loc, srcMemrefType, rewriter)});

    rewriter.replaceOp(copyOp, memCpyCall.getResults());

    return success();
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

struct ConvertReinterpretCast final
    : public OpConversionPattern<memref::ReinterpretCastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::ReinterpretCastOp castOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemRefType srcType = cast<MemRefType>(castOp.getSource().getType());
    MemRefType targetMemRefType =
        cast<MemRefType>(castOp.getResult().getType());

    Location loc = castOp.getLoc();
    Value srcPtr;
    Type targetInEmitC;

    if (srcType.getRank() == 0) {
      // Handle single element memref<i32>
      srcPtr = adaptor.getSource();
      targetInEmitC =
          typeConverter->convertType(mlir::getElementTypeOrSelf(srcType));
    } else {
      if (targetMemRefType.isStrided()) {
        auto [strides, offset] = targetMemRefType.getStridesAndOffset();
        if (strides.empty()) {
          return rewriter.notifyMatchFailure(castOp.getLoc(),
                                             "failed to get strides");
        }
        int64_t linearOffset = offset;
        // Handle memref<n x i32> with strides
        auto srcInEmitC = convertMemRefType(srcType, getTypeConverter());
        if (!srcInEmitC) {
          return rewriter.notifyMatchFailure(castOp.getLoc(),
                                             "cannot convert memref type");
        }
        auto srcArrayValue =
            cast<TypedValue<emitc::ArrayType>>(adaptor.getSource());
        srcPtr = createPointerFromEmitcArray(loc, rewriter, srcArrayValue,
                                             linearOffset)
                     .getResult();
        targetInEmitC = typeConverter->convertType(
            mlir::getElementTypeOrSelf(targetMemRefType));
        if (!targetInEmitC) {
          return rewriter.notifyMatchFailure(castOp.getLoc(),
                                             "cannot convert element type");
        }
      } else {
        // Handle memref<n x i32> without strides
        auto srcInEmitC = convertMemRefType(srcType, getTypeConverter());
        targetInEmitC = convertMemRefType(targetMemRefType, getTypeConverter());
        if (!srcInEmitC || !targetInEmitC) {
          return rewriter.notifyMatchFailure(castOp.getLoc(),
                                             "cannot convert memref type");
        }
        auto srcArrayValue =
            cast<TypedValue<emitc::ArrayType>>(adaptor.getSource());
        srcPtr = createPointerFromEmitcArray(loc, rewriter, srcArrayValue)
                     .getResult();
      }
    }

    auto castCall = emitc::CastOp::create(
        rewriter, loc, emitc::PointerType::get(targetInEmitC), srcPtr);

    rewriter.replaceOp(castOp, castCall);
    return success();
  }
};

struct ConvertExtractStridedMetadata final
    : public OpConversionPattern<memref::ExtractStridedMetadataOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::ExtractStridedMetadataOp extractStridedMetadataOp,
                  OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = extractStridedMetadataOp.getLoc();
    Value source = extractStridedMetadataOp.getSource();

    MemRefType memrefType = cast<MemRefType>(source.getType());
    if (!isMemRefTypeLegalForEmitC(memrefType))
      return rewriter.notifyMatchFailure(
          loc, "incompatible memref type for EmitC conversion");

    TypedValue<emitc::ArrayType> srcArrayValue =
        cast<TypedValue<emitc::ArrayType>>(operands.getSource());

    emitc::ApplyOp srcPtr =
        createPointerFromEmitcArray(loc, rewriter, srcArrayValue);
    auto [strides, offset] = memrefType.getStridesAndOffset();
    Value offsetValue = emitc::ConstantOp::create(
        rewriter, loc, rewriter.getIndexType(), rewriter.getIndexAttr(offset));

    SmallVector<Value> results;

    results.push_back(srcPtr);
    results.push_back(offsetValue);

    for (unsigned i = 0, e = memrefType.getRank(); i < e; ++i) {
      Value sizeValue = emitc::ConstantOp::create(
          rewriter, loc, rewriter.getIndexType(),
          rewriter.getIndexAttr(memrefType.getDimSize(i)));
      results.push_back(sizeValue);

      Value strideValue =
          emitc::ConstantOp::create(rewriter, loc, rewriter.getIndexType(),
                                    rewriter.getIndexAttr(strides[i]));
      results.push_back(strideValue);
    }

    rewriter.replaceOp(extractStridedMetadataOp, results);
    return success();
  }
};
} // namespace

void mlir::populateMemRefToEmitCTypeConversion(TypeConverter &typeConverter) {
  typeConverter.addConversion(
      [&](MemRefType memRefType) -> std::optional<Type> {
        if (!isMemRefTypeLegalForEmitC(memRefType)) {
          return {};
        }
        if (memRefType.getRank() == 0) {
          return typeConverter.convertType(
              mlir::getElementTypeOrSelf(memRefType));
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
  patterns.add<ConvertAlloca, ConvertAlloc, ConvertCopy,
               ConvertExtractStridedMetadata, ConvertReinterpretCast,
               ConvertGlobal, ConvertGetGlobal, ConvertLoad, ConvertStore>(
      converter, patterns.getContext());
}
