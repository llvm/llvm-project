//===- Bufferize.cpp - Bufferization utilities ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/Bufferize.h"
#include "PassDetail.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Operation.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// BufferizeTypeConverter
//===----------------------------------------------------------------------===//

static Value materializeTensorLoad(OpBuilder &builder, TensorType type,
                                   ValueRange inputs, Location loc) {
  assert(inputs.size() == 1);
  assert(inputs[0].getType().isa<BaseMemRefType>());
  return builder.create<memref::TensorLoadOp>(loc, type, inputs[0]);
}

/// Registers conversions into BufferizeTypeConverter
BufferizeTypeConverter::BufferizeTypeConverter() {
  // Keep all types unchanged.
  addConversion([](Type type) { return type; });
  // Convert RankedTensorType to MemRefType.
  addConversion([](RankedTensorType type) -> Type {
    return MemRefType::get(type.getShape(), type.getElementType());
  });
  // Convert UnrankedTensorType to UnrankedMemRefType.
  addConversion([](UnrankedTensorType type) -> Type {
    return UnrankedMemRefType::get(type.getElementType(), 0);
  });
  addArgumentMaterialization(materializeTensorLoad);
  addSourceMaterialization(materializeTensorLoad);
  addTargetMaterialization([](OpBuilder &builder, BaseMemRefType type,
                              ValueRange inputs, Location loc) -> Value {
    assert(inputs.size() == 1);
    assert(inputs[0].getType().isa<TensorType>());
    return builder.create<memref::BufferCastOp>(loc, type, inputs[0]);
  });
}

void mlir::populateBufferizeMaterializationLegality(ConversionTarget &target) {
  target.addLegalOp<memref::TensorLoadOp, memref::BufferCastOp>();
}

namespace {
// In a finalizing bufferize conversion, we know that all tensors have been
// converted to memrefs, thus, this op becomes an identity.
class BufferizeTensorLoadOp : public OpConversionPattern<memref::TensorLoadOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(memref::TensorLoadOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    memref::TensorLoadOp::Adaptor adaptor(operands);
    rewriter.replaceOp(op, adaptor.memref());
    return success();
  }
};
} // namespace

namespace {
// In a finalizing bufferize conversion, we know that all tensors have been
// converted to memrefs, thus, this op becomes an identity.
class BufferizeCastOp : public OpConversionPattern<memref::BufferCastOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(memref::BufferCastOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    memref::BufferCastOp::Adaptor adaptor(operands);
    rewriter.replaceOp(op, adaptor.tensor());
    return success();
  }
};
} // namespace

void mlir::populateEliminateBufferizeMaterializationsPatterns(
    BufferizeTypeConverter &typeConverter, RewritePatternSet &patterns) {
  patterns.add<BufferizeTensorLoadOp, BufferizeCastOp>(typeConverter,
                                                       patterns.getContext());
}

namespace {
struct FinalizingBufferizePass
    : public FinalizingBufferizeBase<FinalizingBufferizePass> {
  using FinalizingBufferizeBase<
      FinalizingBufferizePass>::FinalizingBufferizeBase;

  void runOnFunction() override {
    auto func = getFunction();
    auto *context = &getContext();

    BufferizeTypeConverter typeConverter;
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);

    populateEliminateBufferizeMaterializationsPatterns(typeConverter, patterns);

    // If all result types are legal, and all block arguments are legal (ensured
    // by func conversion above), then all types in the program are legal.
    //
    // We also check that the operand types are legal to avoid creating invalid
    // IR. For example, this prevents
    // populateEliminateBufferizeMaterializationsPatterns from updating the
    // types of the operands to a return op without updating the enclosing
    // function.
    target.markUnknownOpDynamicallyLegal(
        [&](Operation *op) { return typeConverter.isLegal(op); });

    if (failed(applyFullConversion(func, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<FunctionPass> mlir::createFinalizingBufferizePass() {
  return std::make_unique<FinalizingBufferizePass>();
}
