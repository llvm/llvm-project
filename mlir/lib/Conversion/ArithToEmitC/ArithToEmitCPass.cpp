//===- ArithToEmitCPass.cpp - Arith to EmitC Pass ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert the Arith dialect to the EmitC
// dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ArithToEmitC/ArithToEmitCPass.h"

#include "mlir/Conversion/ArithToEmitC/ArithToEmitC.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTARITHTOEMITC
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
struct ConvertArithToEmitC
    : public impl::ConvertArithToEmitCBase<ConvertArithToEmitC> {
  void runOnOperation() override;
};
} // namespace

void ConvertArithToEmitC::runOnOperation() {
  ConversionTarget target(getContext());

  target.addLegalDialect<emitc::EmitCDialect>();
  target.addIllegalDialect<arith::ArithDialect>();

  RewritePatternSet patterns(&getContext());

  TypeConverter typeConverter;
  typeConverter.addConversion([](Type type) { return type; });

  populateArithToEmitCPatterns(typeConverter, patterns);

  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}
