//===- Bufferize.cpp - Bufferization for Arithmetic ops ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/Bufferize.h"
#include "PassDetail.h"
#include "mlir/Dialect/Arithmetic/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

using namespace mlir;

namespace {

/// Bufferize arith.index_cast.
struct BufferizeIndexCastOp : public OpConversionPattern<arith::IndexCastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::IndexCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto tensorType = op.getType().cast<RankedTensorType>();
    rewriter.replaceOpWithNewOp<arith::IndexCastOp>(
        op, adaptor.getIn(),
        MemRefType::get(tensorType.getShape(), tensorType.getElementType()));
    return success();
  }
};

/// Pass to bufferize Arithmetic ops.
struct ArithmeticBufferizePass
    : public ArithmeticBufferizeBase<ArithmeticBufferizePass> {
  void runOnFunction() override {
    BufferizeTypeConverter typeConverter;
    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());

    target.addLegalDialect<arith::ArithmeticDialect, memref::MemRefDialect>();

    arith::populateArithmeticBufferizePatterns(typeConverter, patterns);

    target.addDynamicallyLegalOp<arith::IndexCastOp>(
        [&](arith::IndexCastOp op) {
          return typeConverter.isLegal(op.getType());
        });

    if (failed(
            applyPartialConversion(getFunction(), target, std::move(patterns))))
      signalPassFailure();
  }
};

} // end anonymous namespace

void mlir::arith::populateArithmeticBufferizePatterns(
    BufferizeTypeConverter &typeConverter, RewritePatternSet &patterns) {
  patterns.add<BufferizeIndexCastOp>(typeConverter, patterns.getContext());
}

std::unique_ptr<Pass> mlir::arith::createArithmeticBufferizePass() {
  return std::make_unique<ArithmeticBufferizePass>();
}
