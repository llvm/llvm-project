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

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

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
    // Multiple results func was not converted to `emitc.func`.
    if (callOp.getNumResults() > 1)
      return rewriter.notifyMatchFailure(
          callOp, "only functions with zero or one result can be converted");

    rewriter.replaceOpWithNewOp<emitc::CallOp>(
        callOp,
        callOp.getNumResults() ? callOp.getResult(0).getType() : nullptr,
        adaptor.getOperands(), callOp->getAttrs());

    return success();
  }
};

class FuncOpConversion final : public OpConversionPattern<func::FuncOp> {
public:
  using OpConversionPattern<func::FuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (funcOp.getFunctionType().getNumResults() > 1)
      return rewriter.notifyMatchFailure(
          funcOp, "only functions with zero or one result can be converted");

    // Create the converted `emitc.func` op.
    emitc::FuncOp newFuncOp = rewriter.create<emitc::FuncOp>(
        funcOp.getLoc(), funcOp.getName(), funcOp.getFunctionType());

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

    if (!funcOp.isDeclaration())
      rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                                  newFuncOp.end());
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

void mlir::populateFuncToEmitCPatterns(RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();

  patterns.add<CallOpConversion, FuncOpConversion, ReturnOpConversion>(ctx);
}
