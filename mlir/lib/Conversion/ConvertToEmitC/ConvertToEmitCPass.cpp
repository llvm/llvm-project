//===- ConvertToEmitCPass.cpp - MLIR EmitC Conversion ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ConvertToEmitC/ConvertToEmitCPass.h"
#include "mlir/Conversion/ArithToEmitC/ArithToEmitC.h"
#include "mlir/Conversion/FuncToEmitC/FuncToEmitC.h"
#include "mlir/Conversion/MemRefToEmitC/MemRefToEmitC.h"
#include "mlir/Conversion/SCFToEmitC/SCFToEmitC.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "convert-to-emitc"

namespace mlir {
#define GEN_PASS_DEF_CONVERTTOEMITCPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

/// Populate patterns for each dialect.
void populateConvertToEmitCPatterns(TypeConverter &typeConverter,
                                    RewritePatternSet &patterns) {
  populateArithToEmitCPatterns(typeConverter, patterns);
  populateFuncToEmitCPatterns(patterns);
  populateMemRefToEmitCTypeConversion(typeConverter);
  populateMemRefToEmitCConversionPatterns(patterns, typeConverter);
  populateSCFToEmitCConversionPatterns(patterns);
  populateFunctionOpInterfaceTypeConversionPattern<emitc::FuncOp>(
      patterns, typeConverter);
}

/// A pass to perform the SPIR-V conversion.
struct ConvertToEmitCPass final
    : impl::ConvertToEmitCPassBase<ConvertToEmitCPass> {
  using ConvertToEmitCPassBase::ConvertToEmitCPassBase;

  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });

    ConversionTarget target(*context);
    target.addIllegalDialect<arith::ArithDialect, func::FuncDialect,
                             memref::MemRefDialect, scf::SCFDialect>();
    target.addLegalDialect<emitc::EmitCDialect>();
    target.addDynamicallyLegalOp<emitc::FuncOp>(
        [&typeConverter](emitc::FuncOp op) {
          return typeConverter.isSignatureLegal(op.getFunctionType());
        });

    populateConvertToEmitCPatterns(typeConverter, patterns);
    if (failed(applyPartialConversion(op, target, std::move(patterns))))
      return signalPassFailure();
    return;
  }
};

} // namespace
