//===- Bufferize.cpp - Bufferization for std ops --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements bufferization of tensor-valued arith.constant ops.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/BufferUtils.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/StandardOps/Transforms/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::bufferization;

namespace {
class BufferizeTensorConstantOp
    : public OpConversionPattern<arith::ConstantOp> {
public:
  BufferizeTensorConstantOp(GlobalCreator &globals,
                            TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<arith::ConstantOp>(typeConverter, context,
                                               /*benefit=*/1),
        globals(globals) {}

  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto type = op.getType().dyn_cast<RankedTensorType>();
    if (!type)
      return failure();

    auto globalMemref = globals.getGlobalFor(op);
    rewriter.replaceOpWithNewOp<memref::GetGlobalOp>(op, globalMemref.type(),
                                                     globalMemref.getName());
    return success();
  }
  GlobalCreator &globals;
};
} // namespace

void mlir::populateTensorConstantBufferizePatterns(
    GlobalCreator &globalCreator,
    bufferization::BufferizeTypeConverter &typeConverter,
    RewritePatternSet &patterns) {
  patterns.add<BufferizeTensorConstantOp>(globalCreator, typeConverter,
                                          patterns.getContext());
}

namespace {
class TensorConstantBufferizePass
    : public TensorConstantBufferizeBase<TensorConstantBufferizePass> {
public:
  explicit TensorConstantBufferizePass(unsigned alignment) {
    if (alignment)
      this->alignment = alignment;
  }

  void runOnOperation() override {
    auto module = getOperation();
    GlobalCreator globals(module, alignment);

    auto *context = &getContext();
    bufferization::BufferizeTypeConverter typeConverter;
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);

    target.addLegalDialect<memref::MemRefDialect>();
    populateTensorConstantBufferizePatterns(globals, typeConverter, patterns);
    target.addDynamicallyLegalOp<arith::ConstantOp>([&](arith::ConstantOp op) {
      return typeConverter.isLegal(op.getType());
    });
    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass>
mlir::createTensorConstantBufferizePass(unsigned alignment) {
  return std::make_unique<TensorConstantBufferizePass>(alignment);
}
