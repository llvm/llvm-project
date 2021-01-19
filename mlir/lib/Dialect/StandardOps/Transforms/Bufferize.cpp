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
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/StandardOps/Transforms/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {
class BufferizeDimOp : public OpConversionPattern<DimOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(DimOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    DimOp::Adaptor adaptor(operands);
    rewriter.replaceOpWithNewOp<DimOp>(op, adaptor.memrefOrTensor(),
                                       adaptor.index());
    return success();
  }
};
} // namespace

namespace {
class BufferizeSelectOp : public OpConversionPattern<SelectOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(SelectOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.condition().getType().isa<IntegerType>())
      return rewriter.notifyMatchFailure(op, "requires scalar condition");

    SelectOp::Adaptor adaptor(operands);
    rewriter.replaceOpWithNewOp<SelectOp>(
        op, adaptor.condition(), adaptor.true_value(), adaptor.false_value());
    return success();
  }
};
} // namespace

void mlir::populateStdBufferizePatterns(MLIRContext *context,
                                        BufferizeTypeConverter &typeConverter,
                                        OwningRewritePatternList &patterns) {
  patterns.insert<BufferizeDimOp, BufferizeSelectOp>(typeConverter, context);
}

namespace {
struct StdBufferizePass : public StdBufferizeBase<StdBufferizePass> {
  void runOnFunction() override {
    auto *context = &getContext();
    BufferizeTypeConverter typeConverter;
    OwningRewritePatternList patterns;
    ConversionTarget target(*context);

    target.addLegalDialect<StandardOpsDialect>();
    target.addLegalDialect<scf::SCFDialect>();

    populateStdBufferizePatterns(context, typeConverter, patterns);
    // We only bufferize the case of tensor selected type and scalar condition,
    // as that boils down to a select over memref descriptors (don't need to
    // touch the data).
    target.addDynamicallyLegalOp<SelectOp>([&](SelectOp op) {
      return typeConverter.isLegal(op.getType()) ||
             !op.condition().getType().isa<IntegerType>();
    });
    target.addDynamicallyLegalOp<DimOp>(
        [&](DimOp op) { return typeConverter.isLegal(op); });
    if (failed(
            applyPartialConversion(getFunction(), target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createStdBufferizePass() {
  return std::make_unique<StdBufferizePass>();
}
