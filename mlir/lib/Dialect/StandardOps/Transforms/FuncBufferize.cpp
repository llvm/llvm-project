//===- Bufferize.cpp - Bufferization for std ops --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements bufferization of std.func's and std.call's.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/StandardOps/Transforms/FuncConversions.h"
#include "mlir/Dialect/StandardOps/Transforms/Passes.h"
#include "mlir/Transforms/Bufferize.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {
struct FuncBufferizePass : public FuncBufferizeBase<FuncBufferizePass> {
  void runOnOperation() override {
    auto module = getOperation();
    auto *context = &getContext();

    BufferizeTypeConverter typeConverter;
    OwningRewritePatternList patterns;
    ConversionTarget target(*context);

    populateFuncOpTypeConversionPattern(patterns, context, typeConverter);
    target.addDynamicallyLegalOp<FuncOp>([&](FuncOp op) {
      return typeConverter.isSignatureLegal(op.getType()) &&
             typeConverter.isLegal(&op.getBody());
    });
    populateCallOpTypeConversionPattern(patterns, context, typeConverter);
    populateEliminateBufferizeMaterializationsPatterns(context, typeConverter,
                                                       patterns);
    target.addIllegalOp<TensorLoadOp, TensorToMemrefOp>();

    // If all result types are legal, and all block arguments are legal (ensured
    // by func conversion above), then all types in the program are legal.
    target.markUnknownOpDynamicallyLegal([&](Operation *op) {
      return typeConverter.isLegal(op->getResultTypes());
    });

    if (failed(applyFullConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createFuncBufferizePass() {
  return std::make_unique<FuncBufferizePass>();
}
