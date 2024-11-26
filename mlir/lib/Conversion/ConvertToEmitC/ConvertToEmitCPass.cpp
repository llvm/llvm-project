//===- ConvertToEmitCPass.cpp - Conversion to EmitC pass --*- C++ -*-=========//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ConvertToEmitC/ConvertToEmitCPass.h"

#include "mlir/Conversion/ArithToEmitC/ArithToEmitC.h"
#include "mlir/Conversion/ConvertToEmitC/ConvertToEmitC.h"
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

namespace mlir {
#define GEN_PASS_DEF_CONVERTTOEMITC
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

/// A pass to perform the EmitC conversion.
struct ConvertToEmitC final : impl::ConvertToEmitCBase<ConvertToEmitC> {
  using ConvertToEmitCBase::ConvertToEmitCBase;

  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    TypeConverter typeConverter;

    ConversionTarget target(*context);
    target.addIllegalDialect<arith::ArithDialect, func::FuncDialect,
                             memref::MemRefDialect, scf::SCFDialect>();
    target.addLegalDialect<emitc::EmitCDialect>();

    populateConvertToEmitCTypeConverter(typeConverter);
    populateConvertToEmitCPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(op, target, std::move(patterns))))
      return signalPassFailure();
    return;
  }
};

} // namespace
