//===- Bufferize.cpp - Bufferization for `tensor` dialect ops -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements bufferization of `tensor` dialect ops
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/Bufferize.h"
#include "PassDetail.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {
class BufferizeCastOp : public OpConversionPattern<tensor::CastOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tensor::CastOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<MemRefCastOp>(op, resultType, operands[0]);
    return success();
  }
};
} // namespace

namespace {
class BufferizeExtractOp : public OpConversionPattern<tensor::ExtractOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tensor::ExtractOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    tensor::ExtractOp::Adaptor adaptor(operands);
    rewriter.replaceOpWithNewOp<LoadOp>(op, adaptor.tensor(),
                                        adaptor.indices());
    return success();
  }
};
} // namespace

void mlir::populateTensorBufferizePatterns(
    MLIRContext *context, BufferizeTypeConverter &typeConverter,
    OwningRewritePatternList &patterns) {
  patterns.insert<BufferizeCastOp, BufferizeExtractOp>(typeConverter, context);
}

namespace {
struct TensorBufferizePass : public TensorBufferizeBase<TensorBufferizePass> {
  void runOnFunction() override {
    auto *context = &getContext();
    BufferizeTypeConverter typeConverter;
    OwningRewritePatternList patterns;
    ConversionTarget target(*context);

    populateTensorBufferizePatterns(context, typeConverter, patterns);
    target.addIllegalOp<tensor::CastOp, tensor::ExtractOp>();
    target.addLegalDialect<StandardOpsDialect>();

    if (failed(
            applyPartialConversion(getFunction(), target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createTensorBufferizePass() {
  return std::make_unique<TensorBufferizePass>();
}
