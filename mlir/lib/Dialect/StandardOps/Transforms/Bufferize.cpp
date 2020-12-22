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
class BufferizeDynamicTensorFromElementsOp
    : public OpConversionPattern<DynamicTensorFromElementsOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(DynamicTensorFromElementsOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    // Allocate memory.
    Location loc = op.getLoc();
    DynamicTensorFromElementsOp::Adaptor transformed(operands);
    RankedTensorType tensorType = op.getType().cast<RankedTensorType>();
    MemRefType memrefType =
        MemRefType::get(tensorType.getShape(), tensorType.getElementType());
    Value result =
        rewriter.create<AllocOp>(loc, memrefType, transformed.dynamicExtents());

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
    rewriter.replaceOpWithNewOp<StoreOp>(elementYield,
                                         elementYield->getOperands()[0], result,
                                         parallelBody->getArguments());

    rewriter.replaceOp(op, {result});
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
  patterns.insert<
      // clang-format off
      BufferizeDimOp,
      BufferizeDynamicTensorFromElementsOp,
      BufferizeSelectOp,
      BufferizeTensorFromElementsOp
      // clang-format on
      >(typeConverter, context);
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
    target.addIllegalOp<DynamicTensorFromElementsOp, TensorFromElementsOp>();
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
