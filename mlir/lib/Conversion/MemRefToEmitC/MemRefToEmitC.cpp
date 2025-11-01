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
#include "llvm/ADT/STLExtras.h"
#include <cstdint>
#include <numeric>

using namespace mlir;

static bool isMemRefTypeLegalForEmitC(MemRefType memRefType) {
  return memRefType.hasStaticShape() && memRefType.getLayout().isIdentity() &&
         memRefType.getRank() != 0 &&
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
    auto memRefType = op.getType();
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

    if (op.getType().getRank() == 0 ||
        llvm::is_contained(memRefType.getShape(), 0)) {
      return rewriter.notifyMatchFailure(
          op.getLoc(), "cannot transform alloca with rank 0 or zero-sized dim");
    }

    auto convertedTy = getTypeConverter()->convertType(memRefType);
    if (!convertedTy) {
      return rewriter.notifyMatchFailure(op.getLoc(),
                                         "cannot convert memref type");
    }

    auto arrayTy = emitc::ArrayType::get(memRefType.getShape(),
                                         memRefType.getElementType());
    auto elemTy = memRefType.getElementType();

    auto noInit = emitc::OpaqueAttr::get(getContext(), "");
    auto arrayVar =
        rewriter.create<emitc::VariableOp>(op.getLoc(), arrayTy, noInit);

    // Build zero indices for the base subscript.
    SmallVector<Value> indices;
    for (unsigned i = 0; i < memRefType.getRank(); ++i) {
      auto zero = rewriter.create<emitc::ConstantOp>(
          op.getLoc(), rewriter.getIndexType(), rewriter.getIndexAttr(0));
      indices.push_back(zero);
    }

    auto current = rewriter.create<emitc::SubscriptOp>(
        op.getLoc(), emitc::LValueType::get(elemTy), arrayVar.getResult(),
        indices);

    auto ptrElemTy = emitc::PointerType::get(elemTy);
    auto addrOf = rewriter.create<emitc::ApplyOp>(op.getLoc(), ptrElemTy,
                                                  rewriter.getStringAttr("&"),
                                                  current.getResult());

    auto ptrArrayTy = emitc::PointerType::get(arrayTy);
    auto casted = rewriter.create<emitc::CastOp>(op.getLoc(), ptrArrayTy,
                                                 addrOf.getResult());

    rewriter.replaceOp(op, casted.getResult());
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
  int64_t numElements = llvm::product_of(memrefType.getShape());
  emitc::ConstantOp numElementsValue = emitc::ConstantOp::create(
      builder, loc, indexType, builder.getIndexAttr(numElements));

  Type sizeTType = emitc::SizeTType::get(builder.getContext());
  emitc::MulOp totalSizeBytes = emitc::MulOp::create(
      builder, loc, sizeTType, elementSize.getResult(0), numElementsValue);

  return totalSizeBytes.getResult();
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
    emitc::ArrayType arrayType =
        emitc::ArrayType::get(memrefType.getShape(), elementType);
    emitc::PointerType targetPointerType = emitc::PointerType::get(arrayType);
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

    emitc::CallOpaqueOp memCpyCall = emitc::CallOpaqueOp::create(
        rewriter, loc, TypeRange{}, "memcpy",
        ValueRange{
            operands.getTarget(), operands.getSource(),
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

    Type elemTy = getTypeConverter()->convertType(opTy.getElementType());
    Type globalType;
    if (opTy.getRank() == 0) {
      globalType = elemTy;
    } else {
      SmallVector<int64_t> shape(opTy.getShape().begin(),
                                 opTy.getShape().end());
      globalType = emitc::ArrayType::get(shape, elemTy);
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
        op, operands.getSymName(), globalType, initialValue, externSpecifier,
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
    Location loc = op.getLoc();

    Type elemTy = getTypeConverter()->convertType(opTy.getElementType());
    if (!elemTy)
      return rewriter.notifyMatchFailure(loc, "cannot convert element type");

    Type resultTy = convertMemRefType(opTy, getTypeConverter());

    if (!resultTy) {
      return rewriter.notifyMatchFailure(op.getLoc(),
                                         "cannot convert result type");
    }
      Type globalType;
      if (opTy.getRank() == 0) {
        globalType = elemTy;
      } else {
        SmallVector<int64_t> shape(opTy.getShape().begin(),
                                   opTy.getShape().end());
        globalType = emitc::ArrayType::get(shape, elemTy);
      }

    if (opTy.getRank() == 0) {
      emitc::LValueType lvalueType = emitc::LValueType::get(globalType);
      emitc::GetGlobalOp globalLValue = emitc::GetGlobalOp::create(
          rewriter, op.getLoc(), lvalueType, operands.getNameAttr());
      emitc::PointerType pointerType = emitc::PointerType::get(globalType);
      auto addrOf = rewriter.create<emitc::ApplyOp>(
          loc, pointerType, rewriter.getStringAttr("&"), globalLValue.getResult());

      auto arrayTy = emitc::ArrayType::get({1}, globalType);
      auto ptrArrayTy = emitc::PointerType::get(arrayTy);
      auto casted =
          rewriter.create<emitc::CastOp>(loc, ptrArrayTy, addrOf.getResult());
      rewriter.replaceOp(op, casted.getResult());
      return success();
    }

    auto getGlobal = rewriter.create<emitc::GetGlobalOp>(
        loc, globalType, operands.getNameAttr());

    SmallVector<Value> indices;
    for (unsigned i = 0; i < opTy.getRank(); ++i) {
      auto zero = rewriter.create<emitc::ConstantOp>(
          loc, rewriter.getIndexType(), rewriter.getIndexAttr(0));
      indices.push_back(zero);
    }

    auto current = rewriter.create<emitc::SubscriptOp>(
        loc, emitc::LValueType::get(elemTy), getGlobal.getResult(), indices);

    auto ptrElemTy = emitc::PointerType::get(opTy.getElementType());
    auto addrOf = rewriter.create<emitc::ApplyOp>(
        loc, ptrElemTy, rewriter.getStringAttr("&"), current.getResult());

    auto casted =
        rewriter.create<emitc::CastOp>(loc, resultTy, addrOf.getResult());

    rewriter.replaceOp(op, casted.getResult());
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
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    Value memrefVal = operands.getMemref();
    Value deref;
    if (auto ptrVal = dyn_cast<TypedValue<emitc::PointerType>>(memrefVal)) {
      auto arrayTy = dyn_cast<emitc::ArrayType>(ptrVal.getType().getPointee());
      if (!arrayTy)
        return failure();
      deref = emitc::ApplyOp::create(b, arrayTy, b.getStringAttr("*"), ptrVal);
    }

    auto arrayValue = dyn_cast<TypedValue<emitc::ArrayType>>(deref);
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
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    Value memrefVal = operands.getMemref();
    Value deref;
    if (auto ptrVal = dyn_cast<TypedValue<emitc::PointerType>>(memrefVal)) {
      auto arrayTy = dyn_cast<emitc::ArrayType>(ptrVal.getType().getPointee());
      if (!arrayTy)
        return failure();
      deref = emitc::ApplyOp::create(b, arrayTy, b.getStringAttr("*"), ptrVal);
    }
    auto arrayValue = dyn_cast<TypedValue<emitc::ArrayType>>(deref);
    auto subscript = emitc::SubscriptOp::create(
        rewriter, op.getLoc(), arrayValue, operands.getIndices());
    Value valueToStore = operands.getOperands()[0];

    rewriter.replaceOpWithNewOp<emitc::AssignOp>(op, subscript, valueToStore);
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
        Type convertedElementType =
            typeConverter.convertType(memRefType.getElementType());
        if (!convertedElementType)
          return {};
        Type innerArrayType =
            emitc::ArrayType::get(memRefType.getShape(), convertedElementType);
        return emitc::PointerType::get(innerArrayType);

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
  patterns.add<ConvertAlloca, ConvertAlloc, ConvertCopy, ConvertGlobal,
               ConvertGetGlobal, ConvertLoad, ConvertStore>(
      converter, patterns.getContext());
}
