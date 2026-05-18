//===- FuncToEmitC.cpp - Func to EmitC Patterns -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements patterns to convert the Func dialect to the EmitC
// dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/FuncToEmitC/FuncToEmitC.h"
#include "mlir/Conversion/ConvertToEmitC/ToEmitCInterface.h"
#include "mlir/Conversion/MemRefToEmitC/MemRefToEmitC.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {

/// Implement the interface to convert Func to EmitC.
struct FuncToEmitCDialectInterface : public ConvertToEmitCPatternInterface {
  FuncToEmitCDialectInterface(Dialect *dialect)
      : ConvertToEmitCPatternInterface(dialect) {}

  /// Hook for derived dialect interface to provide conversion patterns
  /// and mark dialect legal for the conversion target.
  void populateConvertToEmitCConversionPatterns(
      ConversionTarget &target, TypeConverter &typeConverter,
      RewritePatternSet &patterns) const final {
    populateFuncToEmitCPatterns(typeConverter, patterns);
    populateMemRefToEmitCTypeConversion(typeConverter);
    target.addLegalOp<UnrealizedConversionCastOp>();
  }
};
} // namespace

void mlir::registerConvertFuncToEmitCInterface(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, func::FuncDialect *dialect) {
    dialect->addInterfaces<FuncToEmitCDialectInterface>();
  });
}

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//

namespace {
// Checks whether return type is a MemRef that can be converted to scalar and
// legal for EmitC.
static bool isScalarizableMemRefResult(Type retTy) {
  auto retTyAsMemRef = dyn_cast<MemRefType>(retTy);
  return retTyAsMemRef && retTyAsMemRef.hasStaticShape() &&
         retTyAsMemRef.getLayout().isIdentity() &&
         retTyAsMemRef.getNumElements() == 1;
}

static Value materializeCastIfNeeded(OpBuilder &builder, Location loc,
                                     Type type, Value value) {
  if (value.getType() == type)
    return value;
  return UnrealizedConversionCastOp::create(builder, loc, type, value)
      .getResult(0);
}

static Value loadScalarFromArray(OpBuilder &builder, Location loc,
                                 Type scalarType,
                                 TypedValue<emitc::ArrayType> arrayValue) {
  emitc::ConstantOp zeroIndex = emitc::ConstantOp::create(
      builder, loc, builder.getIndexType(), builder.getIndexAttr(0));
  SmallVector<Value> indices(arrayValue.getType().getRank(),
                             zeroIndex.getResult());
  auto subscript =
      emitc::SubscriptOp::create(builder, loc, arrayValue, indices);
  return emitc::LoadOp::create(builder, loc, scalarType, subscript.getResult())
      .getResult();
}

static Value loadScalarFromPointer(OpBuilder &builder, Location loc,
                                   Type scalarType,
                                   TypedValue<emitc::PointerType> ptr) {
  Value zeroIndex = emitc::ConstantOp::create(
      builder, loc, builder.getIndexType(), builder.getIndexAttr(0));
  auto subscript = emitc::SubscriptOp::create(builder, loc, ptr, zeroIndex);
  return emitc::LoadOp::create(builder, loc, scalarType, subscript.getResult())
      .getResult();
}

static bool shouldReadMemRefReturnAsPointer(Value value,
                                            MemRefType memrefType) {
  return memrefType.getRank() == 0 || value.getDefiningOp<memref::AllocOp>();
}

static void scalarizeSingleElementMemRefReturns(
    emitc::FuncOp funcOp, ConversionPatternRewriter &rewriter,
    const TypeConverter &typeConverter, MemRefType memrefType) {
  // Due to branching the function may have multiple return statements.
  SmallVector<func::ReturnOp> returnOps;
  funcOp.walk([&](func::ReturnOp returnOp) {
    assert(returnOp.getNumOperands() == 1 &&
           "Expected scalarized function to have one return operand");
    returnOps.push_back(returnOp);
  });

  for (func::ReturnOp returnOp : returnOps) {
    OpBuilder::InsertionGuard guard(rewriter);
    Location loc = returnOp.getLoc();
    rewriter.setInsertionPoint(returnOp);

    Type scalarType = typeConverter.convertType(memrefType.getElementType());
    assert(scalarType && "MemRef element type must be EmitC convertible.");

    Value returnedValue = returnOp.getOperand(0);
    if (auto arrayValue =
            dyn_cast<TypedValue<emitc::ArrayType>>(returnedValue)) {
      Value scalar = loadScalarFromArray(rewriter, loc, scalarType, arrayValue);
      rewriter.replaceOpWithNewOp<emitc::ReturnOp>(returnOp, scalar);
      continue;
    }

    if (auto ptr = dyn_cast<TypedValue<emitc::PointerType>>(returnedValue)) {
      Value scalar = loadScalarFromPointer(rewriter, loc, scalarType, ptr);
      rewriter.replaceOpWithNewOp<emitc::ReturnOp>(returnOp, scalar);
      continue;
    }

    auto returnedMemRefType = dyn_cast<MemRefType>(returnedValue.getType());
    assert(returnedMemRefType &&
           "Expected scalarized return operand to be a memref, emitc.array, or "
           "emitc.ptr.");

    if (shouldReadMemRefReturnAsPointer(returnedValue, returnedMemRefType)) {
      Type ptrTy = emitc::PointerType::get(scalarType);
      Value ptrValue =
          materializeCastIfNeeded(rewriter, loc, ptrTy, returnedValue);
      Value scalar =
          loadScalarFromPointer(rewriter, loc, scalarType,
                                cast<TypedValue<emitc::PointerType>>(ptrValue));
      rewriter.replaceOpWithNewOp<emitc::ReturnOp>(returnOp, scalar);
      continue;
    }

    Type convertedArrayType = typeConverter.convertType(returnedMemRefType);
    assert((convertedArrayType && isa<emitc::ArrayType>(convertedArrayType)) &&
           "MemRef type must convert to emitc.array.");
    Value arrayValue = materializeCastIfNeeded(
        rewriter, loc, convertedArrayType, returnedValue);
    Value scalar =
        loadScalarFromArray(rewriter, loc, scalarType,
                            cast<TypedValue<emitc::ArrayType>>(arrayValue));
    rewriter.replaceOpWithNewOp<emitc::ReturnOp>(returnOp, scalar);
  }
}

class CallOpConversion final : public OpConversionPattern<func::CallOp> {
public:
  using OpConversionPattern<func::CallOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::CallOp callOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Multiple results func cannot be converted to `emitc.func`.
    if (callOp.getNumResults() > 1)
      return rewriter.notifyMatchFailure(
          callOp, "only functions with zero or one result can be converted");

    rewriter.replaceOpWithNewOp<emitc::CallOp>(callOp, callOp.getResultTypes(),
                                               adaptor.getOperands(),
                                               callOp->getAttrs());

    return success();
  }
};

class FuncOpConversion final : public OpConversionPattern<func::FuncOp> {
public:
  using OpConversionPattern<func::FuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FunctionType fnType = funcOp.getFunctionType();

    if (fnType.getNumResults() > 1)
      return rewriter.notifyMatchFailure(
          funcOp, "only functions with zero or one result can be converted");

    TypeConverter::SignatureConversion signatureConverter(
        fnType.getNumInputs());
    for (const auto &argType : enumerate(fnType.getInputs())) {
      auto convertedType = getTypeConverter()->convertType(argType.value());
      if (!convertedType)
        return rewriter.notifyMatchFailure(funcOp,
                                           "argument type conversion failed");
      signatureConverter.addInputs(argType.index(), convertedType);
    }

    Type resultType;
    bool scalarizeMemRefResult = false;
    MemRefType scalarizedMemRefType;
    if (fnType.getNumResults() == 1) {
      Type originalResultType = fnType.getResult(0);
      scalarizeMemRefResult = isScalarizableMemRefResult(originalResultType) &&
                              isPrivateAndUnused(funcOp);
      if (scalarizeMemRefResult) {
        scalarizedMemRefType = cast<MemRefType>(originalResultType);
      }
      Type typeToConvert = scalarizeMemRefResult
                               ? getElementTypeOrSelf(scalarizedMemRefType)
                               : originalResultType;
      resultType = getTypeConverter()->convertType(typeToConvert);
      if (!resultType)
        return rewriter.notifyMatchFailure(funcOp,
                                           "result type conversion failed");
    }

    // Create the converted `emitc.func` op.
    emitc::FuncOp newFuncOp = emitc::FuncOp::create(
        rewriter, funcOp.getLoc(), funcOp.getName(),
        FunctionType::get(rewriter.getContext(),
                          signatureConverter.getConvertedTypes(),
                          resultType ? TypeRange(resultType) : TypeRange()));

    // Copy over all attributes other than the function name and type.
    for (const auto &namedAttr : funcOp->getAttrs()) {
      if (namedAttr.getName() != funcOp.getFunctionTypeAttrName() &&
          namedAttr.getName() != SymbolTable::getSymbolAttrName())
        newFuncOp->setAttr(namedAttr.getName(), namedAttr.getValue());
    }

    // Add `extern` to specifiers if `func.func` is declaration only.
    if (funcOp.isDeclaration()) {
      ArrayAttr specifiers = rewriter.getStrArrayAttr({"extern"});
      newFuncOp.setSpecifiersAttr(specifiers);
    }

    // Add `static` to specifiers if `func.func` is private but not a
    // declaration.
    if (funcOp.isPrivate() && !funcOp.isDeclaration()) {
      ArrayAttr specifiers = rewriter.getStrArrayAttr({"static"});
      newFuncOp.setSpecifiersAttr(specifiers);
    }

    if (!funcOp.isDeclaration()) {
      rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                                  newFuncOp.end());
      if (failed(rewriter.convertRegionTypes(
              &newFuncOp.getBody(), *getTypeConverter(), &signatureConverter)))
        return failure();
      if (scalarizeMemRefResult)
        scalarizeSingleElementMemRefReturns(
            newFuncOp, rewriter, *getTypeConverter(), scalarizedMemRefType);
    }
    rewriter.eraseOp(funcOp);

    return success();
  }

private:
  // Updating the signature of public functions or functions with users is
  // unsafe, since we may not have access to update all symbol users, therefore
  // their call sites would still expect the original signature.
  bool isPrivateAndUnused(func::FuncOp funcOp) const {
    if (!funcOp.isPrivate())
      return false;

    ModuleOp module = funcOp->getParentOfType<ModuleOp>();
    if (!module)
      return false;

    return SymbolTable::symbolKnownUseEmpty(funcOp, module);
  }
};

class ReturnOpConversion final : public OpConversionPattern<func::ReturnOp> {
public:
  using OpConversionPattern<func::ReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::ReturnOp returnOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (returnOp.getNumOperands() > 1)
      return rewriter.notifyMatchFailure(
          returnOp, "only zero or one operand is supported");

    if (returnOp.getNumOperands() == 0) {
      rewriter.replaceOpWithNewOp<emitc::ReturnOp>(returnOp, Value());
      return success();
    }

    rewriter.replaceOpWithNewOp<emitc::ReturnOp>(returnOp,
                                                 adaptor.getOperands()[0]);
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pattern population
//===----------------------------------------------------------------------===//

void mlir::populateFuncToEmitCPatterns(const TypeConverter &typeConverter,
                                       RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();

  patterns.add<CallOpConversion, FuncOpConversion, ReturnOpConversion>(
      typeConverter, ctx);
}
