//===- Bufferize.cpp - Bufferization utilities ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/Bufferize.h"
#include "mlir/IR/Operation.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// BufferizeTypeConverter
//===----------------------------------------------------------------------===//

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
  addSourceMaterialization([](OpBuilder &builder, TensorType type,
                              ValueRange inputs, Location loc) -> Value {
    assert(inputs.size() == 1);
    assert(inputs[0].getType().isa<BaseMemRefType>());
    return builder.create<TensorLoadOp>(loc, type, inputs[0]);
  });
  addTargetMaterialization([](OpBuilder &builder, BaseMemRefType type,
                              ValueRange inputs, Location loc) -> Value {
    assert(inputs.size() == 1);
    assert(inputs[0].getType().isa<TensorType>());
    return builder.create<TensorToMemrefOp>(loc, type, inputs[0]);
  });
}

void mlir::populateBufferizeMaterializationLegality(ConversionTarget &target) {
  target.addLegalOp<TensorLoadOp, TensorToMemrefOp>();
}

namespace {
// In a finalizing bufferize conversion, we know that all tensors have been
// converted to memrefs, thus, this op becomes an identity.
class BufferizeTensorLoadOp : public OpConversionPattern<TensorLoadOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(TensorLoadOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    TensorLoadOp::Adaptor adaptor(operands);
    rewriter.replaceOp(op, adaptor.memref());
    return success();
  }
};
} // namespace

namespace {
// In a finalizing bufferize conversion, we know that all tensors have been
// converted to memrefs, thus, this op becomes an identity.
class BufferizeTensorToMemrefOp : public OpConversionPattern<TensorToMemrefOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(TensorToMemrefOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    TensorToMemrefOp::Adaptor adaptor(operands);
    rewriter.replaceOp(op, adaptor.tensor());
    return success();
  }
};
} // namespace

void mlir::populateEliminateBufferizeMaterializationsPatterns(
    MLIRContext *context, BufferizeTypeConverter &typeConverter,
    OwningRewritePatternList &patterns) {
  patterns.insert<BufferizeTensorLoadOp, BufferizeTensorToMemrefOp>(
      typeConverter, context);
}
