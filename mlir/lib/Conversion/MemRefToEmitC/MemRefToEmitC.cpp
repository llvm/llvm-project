//===- MemRefToEmitC.cpp - MemRef to EmitC conversion ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert memref ops into emitc ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/MemRefToEmitC/MemRefToEmitC.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTMEMREFTOEMITC
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

/// Disallow all memrefs even though we only have conversions
/// for memrefs with static shape right now to have good diagnostics.
bool isLegal(Type t) { return !isa<BaseMemRefType>(t); }

template <typename RangeT>
bool areLegal(RangeT &&range) {
  return llvm::all_of(range, [](Type type) { return isLegal(type); });
}

bool isLegal(Operation *op) {
  return areLegal(op->getOperandTypes()) && areLegal(op->getResultTypes());
}

bool isSignatureLegal(FunctionType ty) {
  return areLegal(ty.getInputs()) && areLegal(ty.getResults());
}

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

    rewriter.replaceOpWithNewOp<emitc::SubscriptOp>(op, operands.getMemref(),
                                                    operands.getIndices());
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

struct ConvertMemRefToEmitCPass
    : public impl::ConvertMemRefToEmitCBase<ConvertMemRefToEmitCPass> {
  void runOnOperation() override {
    TypeConverter converter;
    // Fallback for other types.
    converter.addConversion([](Type type) { return type; });
    populateMemRefToEmitCTypeConversion(converter);
    converter.addConversion(
        [&converter](FunctionType ty) -> std::optional<Type> {
          SmallVector<Type> inputs;
          if (failed(converter.convertTypes(ty.getInputs(), inputs)))
            return std::nullopt;

          SmallVector<Type> results;
          if (failed(converter.convertTypes(ty.getResults(), results)))
            return std::nullopt;

          return FunctionType::get(ty.getContext(), inputs, results);
        });

    RewritePatternSet patterns(&getContext());
    populateMemRefToEmitCConversionPatterns(patterns, converter);
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                   converter);
    populateCallOpTypeConversionPattern(patterns, converter);
    populateReturnOpTypeConversionPattern(patterns, converter);

    ConversionTarget target(getContext());
    target.addDynamicallyLegalOp<func::FuncOp>(
        [](func::FuncOp op) { return isSignatureLegal(op.getFunctionType()); });
    target.addDynamicallyLegalDialect<func::FuncDialect>(
        [](Operation *op) { return isLegal(op); });
    target.addIllegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<emitc::EmitCDialect>();

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

void mlir::populateMemRefToEmitCTypeConversion(TypeConverter &typeConverter) {
  typeConverter.addConversion([](MemRefType memRefType) -> std::optional<Type> {
    if (memRefType.hasStaticShape()) {
      return emitc::ArrayType::get(memRefType.getShape(),
                                   memRefType.getElementType());
    }
    return {};
  });
}

void mlir::populateMemRefToEmitCConversionPatterns(RewritePatternSet &patterns,
                                                   TypeConverter &converter) {
  patterns.add<ConvertAlloca, ConvertLoad, ConvertStore>(converter,
                                                         patterns.getContext());
}
