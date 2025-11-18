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
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {

/// Implement the interface to convert Func to EmitC.
struct FuncToEmitCDialectInterface : public ConvertToEmitCPatternInterface {
  using ConvertToEmitCPatternInterface::ConvertToEmitCPatternInterface;

  /// Hook for derived dialect interface to provide conversion patterns
  /// and mark dialect legal for the conversion target.
  void populateConvertToEmitCConversionPatterns(
      ConversionTarget &target, TypeConverter &typeConverter,
      RewritePatternSet &patterns) const final {
    populateFuncToEmitCPatterns(typeConverter, patterns);
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
    if (fnType.getNumResults() == 1) {
      resultType = getTypeConverter()->convertType(fnType.getResult(0));
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
    }
    rewriter.eraseOp(funcOp);

    return success();
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

    rewriter.replaceOpWithNewOp<emitc::ReturnOp>(
        returnOp,
        returnOp.getNumOperands() ? adaptor.getOperands()[0] : nullptr);
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
