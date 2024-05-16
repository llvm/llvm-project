//===- FuncToEmitCPass.cpp - Func to EmitC Pass -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert the Func dialect to the EmitC dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/FuncToEmitC/FuncToEmitCPass.h"

#include "mlir/Conversion/FuncToEmitC/FuncToEmitC.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTFUNCTOEMITC
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
struct ConvertFuncToEmitC
    : public impl::ConvertFuncToEmitCBase<ConvertFuncToEmitC> {
  void runOnOperation() override;
};
} // namespace

void ConvertFuncToEmitC::runOnOperation() {
  ConversionTarget target(getContext());

  target.addLegalDialect<emitc::EmitCDialect>();
  target.addIllegalOp<func::CallOp, func::FuncOp, func::ReturnOp>();

  RewritePatternSet patterns(&getContext());
  populateFuncToEmitCPatterns(patterns);

  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}
