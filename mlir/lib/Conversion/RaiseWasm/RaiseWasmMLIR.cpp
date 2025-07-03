//===- RaiseWasmMLIR.cpp - Convert Wasm to less abstract dialects ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements lowering of wasm operations to standard dialects ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/RaiseWasm/RaiseWasmMLIR.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/WasmSSA/IR/WasmSSA.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/LogicalResult.h"

#include <optional>


#define DEBUG_TYPE "wasm-convert"

namespace mlir {
#define GEN_PASS_DEF_RAISEWASMMLIR
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::wasmssa;

namespace {

struct WasmCallOpConversion : OpConversionPattern<FuncCallOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(FuncCallOp funcCallOp, FuncCallOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<func::CallOp>(
        funcCallOp, funcCallOp.getCallee(), funcCallOp.getResults().getTypes(),
        funcCallOp.getOperands());
    return success();
  }
};

struct WasmFuncImportOpConversion : OpConversionPattern<FuncImportOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(FuncImportOp funcImportOp, FuncImportOp::Adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto nFunc = rewriter.replaceOpWithNewOp<func::FuncOp>(
        funcImportOp, funcImportOp.getSymName(),
        funcImportOp.getType());
    nFunc.setVisibility(SymbolTable::Visibility::Private);
    return success();
  }
};

struct WasmFuncOpConversion : OpConversionPattern<FuncOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(FuncOp funcOp, FuncOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newFunc = rewriter.create<func::FuncOp>(
        funcOp->getLoc(), funcOp.getSymName(), funcOp.getFunctionType());
    rewriter.cloneRegionBefore(funcOp.getBody(), newFunc.getBody(),
                               newFunc.getBody().end());
    Block *oldEntryBlock = &newFunc.getBody().front();
    auto blockArgTypes = oldEntryBlock->getArgumentTypes();
    TypeConverter::SignatureConversion sC{oldEntryBlock->getNumArguments()};
    auto numArgs = blockArgTypes.size();
    for (size_t i = 0; i < numArgs; ++i) {
      auto argType = dyn_cast<LocalRefType>(blockArgTypes[i]);
      if (!argType)
        return failure();
      sC.addInputs(i, argType.getElementType());
    }

    rewriter.applySignatureConversion(oldEntryBlock, sC, getTypeConverter());
    rewriter.replaceOp(funcOp, newFunc);
    return success();
  }
};

struct WasmLocalGetConversion : OpConversionPattern<LocalGetOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(LocalGetOp localGetOp, LocalGetOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<memref::LoadOp>(localGetOp,
                                                localGetOp.getResult().getType(),
                                                adaptor.getLocalVar(),
                                              ValueRange{});
    return success();
  }
};

struct WasmReturnOpConversion : OpConversionPattern<ReturnOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ReturnOp returnOp, ReturnOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<func::ReturnOp>(returnOp, adaptor.getOperands());
    return success();
  }
};

struct RaiseWasmMLIRPass
    : public impl::RaiseWasmMLIRBase<RaiseWasmMLIRPass> {
  void runOnOperation() override {
    ConversionTarget target{getContext()};
    target.addIllegalDialect<WasmSSADialect>();
    target.addLegalDialect<arith::ArithDialect, BuiltinDialect,
                           cf::ControlFlowDialect, func::FuncDialect,
                           memref::MemRefDialect, math::MathDialect>();
    RewritePatternSet patterns(&getContext());
    TypeConverter tc{};
    tc.addConversion([](Type type) -> std::optional<Type> { return type; });
    tc.addConversion([](LocalRefType type)->std::optional<Type> {
      return MemRefType::get({}, type.getElementType());
    });
    tc.addTargetMaterialization([](OpBuilder& builder, MemRefType destType, ValueRange values, Location loc)->Value{
      if (values.size() != 1 || values.front().getType() != destType.getElementType())
        return {};
      auto localVar = builder.create<memref::AllocaOp>(loc, destType);
      builder.create<memref::StoreOp>(loc, values.front(), localVar.getResult());
      return localVar.getResult();
    });
    populateRaiseWasmMLIRConversionPatterns(tc, patterns);

    llvm::DenseMap<StringAttr, StringAttr> idxSymToImportSym{};
    auto *topOp = getOperation();
    topOp->walk([&idxSymToImportSym, this](ImportOpInterface importOp) {
      auto const qualifiedImportName = importOp.getQualifiedImportName();
      auto qualNameAttr = StringAttr::get(&getContext(), qualifiedImportName);
      idxSymToImportSym.insert(
          std::make_pair(importOp.getSymbolName(), qualNameAttr));
    });

    if (failed(applyFullConversion(topOp, target, std::move(patterns))))
      return signalPassFailure();

    auto symTable = SymbolTable{topOp};
    for (auto &[oldName, newName] : idxSymToImportSym) {
      if (failed(symTable.rename(oldName, newName)))
        return signalPassFailure();
    }
  }
};
} // namespace

void mlir::populateRaiseWasmMLIRConversionPatterns(
    TypeConverter &tc, RewritePatternSet &patternSet) {
  auto *ctx = patternSet.getContext();
  // Disable clang-format in patternSet for readability + small diffs.
  // clang-format off
  patternSet
      .add<
           WasmCallOpConversion,
           WasmFuncImportOpConversion,
           WasmFuncOpConversion,
           WasmLocalGetConversion,
           WasmReturnOpConversion
           >(tc, ctx);
  // clang-format on
}

std::unique_ptr<Pass> mlir::createRaiseWasmMLIRPass() {
  return std::make_unique<RaiseWasmMLIRPass>();
}
