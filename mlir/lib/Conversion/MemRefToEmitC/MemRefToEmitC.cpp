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

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

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

struct ConvertLoad final : public OpConversionPattern<memref::LoadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::LoadOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {

    auto resultTy = getTypeConverter()->convertType(op.getType());
    if (!resultTy) {
      return rewriter.notifyMatchFailure(op.getLoc(), "cannot convert type");
    }

    auto subscript = rewriter.create<emitc::SubscriptOp>(
        op.getLoc(), operands.getMemref(), operands.getIndices());

    auto noInit = emitc::OpaqueAttr::get(getContext(), "");
    auto var =
        rewriter.create<emitc::VariableOp>(op.getLoc(), resultTy, noInit);

    rewriter.create<emitc::AssignOp>(op.getLoc(), var, subscript);
    rewriter.replaceOp(op, var);
    return success();
  }
};

struct ConvertStore final : public OpConversionPattern<memref::StoreOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::StoreOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {

    auto subscript = rewriter.create<emitc::SubscriptOp>(
        op.getLoc(), operands.getMemref(), operands.getIndices());
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
            !memRefType.getLayout().isIdentity() || memRefType.getRank() == 0) {
          return {};
        }
        Type convertedElementType =
            typeConverter.convertType(memRefType.getElementType());
        if (!convertedElementType)
          return {};
        return emitc::ArrayType::get(memRefType.getShape(),
                                     convertedElementType);
      });
}

void mlir::populateMemRefToEmitCConversionPatterns(RewritePatternSet &patterns,
                                                   TypeConverter &converter) {
  patterns.add<ConvertAlloca, ConvertLoad, ConvertStore>(converter,
                                                         patterns.getContext());
}
