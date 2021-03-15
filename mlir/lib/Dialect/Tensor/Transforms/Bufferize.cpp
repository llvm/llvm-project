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
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
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
    rewriter.replaceOpWithNewOp<memref::CastOp>(op, resultType, operands[0]);
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
    rewriter.replaceOpWithNewOp<memref::LoadOp>(op, adaptor.tensor(),
                                                adaptor.indices());
    return success();
  }
};
} // namespace

namespace {
class BufferizeFromElementsOp
    : public OpConversionPattern<tensor::FromElementsOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tensor::FromElementsOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    int numberOfElements = op.elements().size();
    auto resultType = MemRefType::get(
        {numberOfElements}, op.getType().cast<TensorType>().getElementType());
    Value result = rewriter.create<memref::AllocOp>(op.getLoc(), resultType);
    for (auto element : llvm::enumerate(op.elements())) {
      Value index =
          rewriter.create<ConstantIndexOp>(op.getLoc(), element.index());
      rewriter.create<memref::StoreOp>(op.getLoc(), element.value(), result,
                                       index);
    }
    rewriter.replaceOp(op, {result});
    return success();
  }
};
} // namespace

namespace {
class BufferizeGenerateOp : public OpConversionPattern<tensor::GenerateOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tensor::GenerateOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    // Allocate memory.
    Location loc = op.getLoc();
    tensor::GenerateOp::Adaptor transformed(operands);
    RankedTensorType tensorType = op.getType().cast<RankedTensorType>();
    MemRefType memrefType =
        MemRefType::get(tensorType.getShape(), tensorType.getElementType());
    Value result = rewriter.create<memref::AllocOp>(
        loc, memrefType, transformed.dynamicExtents());

    // Collect loop bounds.
    int64_t rank = tensorType.getRank();
    Value zero = rewriter.create<ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<ConstantIndexOp>(loc, 1);
    SmallVector<Value, 4> lowerBounds(rank, zero);
    SmallVector<Value, 4> steps(rank, one);
    SmallVector<Value, 4> upperBounds;
    int nextDynamicIndex = 0;
    for (int i = 0; i < rank; i++) {
      Value upperBound =
          tensorType.isDynamicDim(i)
              ? transformed.dynamicExtents()[nextDynamicIndex++]
              : rewriter.create<ConstantIndexOp>(loc, memrefType.getDimSize(i));
      upperBounds.push_back(upperBound);
    }

    // Generate tensor elements with a parallel loop that stores into
    // each element of the resulting memref.
    //
    // This is a bit tricky. We cannot simply clone the ops because when an op
    // is cloned, it must be legalized. However, we want to allow arbitrary ops
    // in the body that we don't necessarily have legalization patterns for as
    // part of this dialect conversion invocation.
    //
    // To accomplish this, we use mergeBlockBefore to "move" this op's body
    // into the scf.parallel's body.
    auto parallel =
        rewriter.create<scf::ParallelOp>(loc, lowerBounds, upperBounds, steps);
    Block *parallelBody = parallel.getBody();
    rewriter.mergeBlockBefore(op.getBody(), parallelBody->getTerminator(),
                              parallelBody->getArguments());
    // Replace the inlined yield op with a store op. The scf.parallel's builder
    // already populated an scf.yield at the end, so we don't need to worry
    // about creating that.
    Operation *elementYield = parallelBody->getTerminator()->getPrevNode();
    rewriter.setInsertionPointAfter(elementYield);
    rewriter.replaceOpWithNewOp<memref::StoreOp>(
        elementYield, elementYield->getOperands()[0], result,
        parallelBody->getArguments());

    rewriter.replaceOp(op, {result});
    return success();
  }
};
} // namespace

void mlir::populateTensorBufferizePatterns(
    MLIRContext *context, BufferizeTypeConverter &typeConverter,
    OwningRewritePatternList &patterns) {
  patterns.insert<BufferizeCastOp, BufferizeExtractOp, BufferizeFromElementsOp,
                  BufferizeGenerateOp>(typeConverter, context);
}

namespace {
struct TensorBufferizePass : public TensorBufferizeBase<TensorBufferizePass> {
  void runOnFunction() override {
    auto *context = &getContext();
    BufferizeTypeConverter typeConverter;
    OwningRewritePatternList patterns;
    ConversionTarget target(*context);

    populateBufferizeMaterializationLegality(target);

    populateTensorBufferizePatterns(context, typeConverter, patterns);
    target.addIllegalOp<tensor::CastOp, tensor::ExtractOp,
                        tensor::FromElementsOp, tensor::GenerateOp>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<StandardOpsDialect>();
    target.addLegalDialect<scf::SCFDialect>();

    if (failed(
            applyPartialConversion(getFunction(), target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createTensorBufferizePass() {
  return std::make_unique<TensorBufferizePass>();
}
