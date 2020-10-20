//===- Bufferize.cpp - Bufferization for std ops --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements bufferization of std ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/Bufferize.h"
#include "PassDetail.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/StandardOps/Transforms/Passes.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {
class BufferizeExtractElementOp : public OpConversionPattern<ExtractElementOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ExtractElementOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    ExtractElementOp::Adaptor adaptor(operands);
    rewriter.replaceOpWithNewOp<LoadOp>(op, adaptor.aggregate(),
                                        adaptor.indices());
    return success();
  }
};
} // namespace

namespace {
class BufferizeTensorCastOp : public OpConversionPattern<TensorCastOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(TensorCastOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<MemRefCastOp>(op, resultType, operands[0]);
    return success();
  }
};
} // namespace

namespace {
class BufferizeTensorFromElementsOp
    : public OpConversionPattern<TensorFromElementsOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(TensorFromElementsOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    int numberOfElements = op.elements().size();
    auto resultType = MemRefType::get(
        {numberOfElements}, op.getType().cast<TensorType>().getElementType());
    Value result = rewriter.create<AllocOp>(op.getLoc(), resultType);
    for (auto element : llvm::enumerate(op.elements())) {
      Value index =
          rewriter.create<ConstantIndexOp>(op.getLoc(), element.index());
      rewriter.create<StoreOp>(op.getLoc(), element.value(), result, index);
    }
    rewriter.replaceOp(op, {result});
    return success();
  }
};
} // namespace

void mlir::populateStdBufferizePatterns(MLIRContext *context,
                                        BufferizeTypeConverter &typeConverter,
                                        OwningRewritePatternList &patterns) {
  patterns.insert<BufferizeExtractElementOp, BufferizeTensorCastOp,
                  BufferizeTensorFromElementsOp>(typeConverter, context);
}

namespace {
struct StdBufferizePass : public StdBufferizeBase<StdBufferizePass> {
  void runOnFunction() override {
    auto *context = &getContext();
    BufferizeTypeConverter typeConverter;
    OwningRewritePatternList patterns;
    ConversionTarget target(*context);

    target.addLegalDialect<StandardOpsDialect>();

    populateStdBufferizePatterns(context, typeConverter, patterns);
    target.addIllegalOp<ExtractElementOp, TensorCastOp, TensorFromElementsOp>();

    if (failed(applyPartialConversion(getFunction(), target, patterns)))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createStdBufferizePass() {
  return std::make_unique<StdBufferizePass>();
}
