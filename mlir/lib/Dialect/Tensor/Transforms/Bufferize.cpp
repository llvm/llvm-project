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

#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "PassDetail.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {
struct BufferizeCastOp : public OpConversionPattern<tensor::CastOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tensor::CastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<memref::CastOp>(op, resultType,
                                                adaptor.getOperands()[0]);
    return success();
  }
};

struct BufferizeDimOp : public OpConversionPattern<tensor::DimOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tensor::DimOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<memref::DimOp>(op, adaptor.source(),
                                               adaptor.index());
    return success();
  }
};

struct BufferizeExtractOp : public OpConversionPattern<tensor::ExtractOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tensor::ExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<memref::LoadOp>(op, adaptor.tensor(),
                                                adaptor.indices());
    return success();
  }
};

struct BufferizeFromElementsOp
    : public OpConversionPattern<tensor::FromElementsOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tensor::FromElementsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto tensorType = op.getType().cast<RankedTensorType>();
    auto shape = tensorType.getShape();

    // Allocate a buffer for the result.
    auto resultType =
        MemRefType::get(tensorType.getShape(), tensorType.getElementType());
    Value buffer = rewriter.create<memref::AllocOp>(loc, resultType);

    // Case: tensor<0xelem_type>.
    if (op.elements().empty()) {
      rewriter.replaceOp(op, {buffer});
      return success();
    }

    // Case: tensor<elem_type>.
    if (shape.empty()) {
      rewriter.create<memref::StoreOp>(loc, op.elements().front(), buffer);
      rewriter.replaceOp(op, {buffer});
      return success();
    }

    // Create constants for the range of possible indices [0, max{shape_i}).
    auto maxDim = *std::max_element(shape.begin(), shape.end());
    SmallVector<Value, 2> constants;
    constants.reserve(maxDim);
    for (int i = 0; i < maxDim; ++i)
      constants.push_back(rewriter.create<arith::ConstantIndexOp>(loc, i));

    // Traverse all `elements` and create `memref.store` ops.
    ImplicitLocOpBuilder b(loc, rewriter);
    auto elementIt = adaptor.elements().begin();
    SmallVector<Value, 2> indices(tensorType.getRank(), constants[0]);
    createStores(/*dim=*/0, buffer, shape, constants, elementIt, indices, b);

    rewriter.replaceOp(op, {buffer});
    return success();
  }

private:
  // Implements backtracking to traverse indices of the output buffer while
  // iterating over op.elements().
  void createStores(int dim, Value buffer, ArrayRef<int64_t> shape,
                    ArrayRef<Value> constants, ValueRange::iterator &elementIt,
                    SmallVectorImpl<Value> &indices,
                    ImplicitLocOpBuilder b) const {
    if (dim == static_cast<int>(shape.size()) - 1) {
      for (int i = 0; i < shape.back(); ++i) {
        indices.back() = constants[i];
        b.create<memref::StoreOp>(*elementIt, buffer, indices);
        ++elementIt;
      }
      return;
    }
    for (int i = 0; i < shape[dim]; ++i) {
      indices[dim] = constants[i];
      createStores(dim + 1, buffer, shape, constants, elementIt, indices, b);
    }
  }
};

struct BufferizeGenerateOp : public OpConversionPattern<tensor::GenerateOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tensor::GenerateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    // Allocate memory.
    Location loc = op.getLoc();
    RankedTensorType tensorType = op.getType().cast<RankedTensorType>();
    MemRefType memrefType =
        MemRefType::get(tensorType.getShape(), tensorType.getElementType());
    Value result = rewriter.create<memref::AllocOp>(loc, memrefType,
                                                    adaptor.dynamicExtents());

    // Collect loop bounds.
    int64_t rank = tensorType.getRank();
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    SmallVector<Value, 4> lowerBounds(rank, zero);
    SmallVector<Value, 4> steps(rank, one);
    SmallVector<Value, 4> upperBounds;
    int nextDynamicIndex = 0;
    for (int i = 0; i < rank; i++) {
      Value upperBound = tensorType.isDynamicDim(i)
                             ? adaptor.dynamicExtents()[nextDynamicIndex++]
                             : rewriter.create<arith::ConstantIndexOp>(
                                   loc, memrefType.getDimSize(i));
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

struct BufferizeRankOp : public OpConversionPattern<tensor::RankOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tensor::RankOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<memref::RankOp>(op, op.getType(),
                                                adaptor.tensor());
    return success();
  }
};

struct TensorBufferizePass : public TensorBufferizeBase<TensorBufferizePass> {
  void runOnFunction() override {
    auto *context = &getContext();
    bufferization::BufferizeTypeConverter typeConverter;

    ConversionTarget target(*context);
    target.addLegalDialect<scf::SCFDialect, memref::MemRefDialect>();
    target.addDynamicallyLegalDialect<arith::ArithmeticDialect,
                                      StandardOpsDialect>(
        [&](Operation *op) { return typeConverter.isLegal(op); });
    target.addLegalOp<CallOp, ReturnOp>();
    target.addIllegalOp<tensor::CastOp, tensor::ExtractOp,
                        tensor::FromElementsOp, tensor::GenerateOp>();
    bufferization::populateBufferizeMaterializationLegality(target);

    RewritePatternSet patterns(context);
    populateTensorBufferizePatterns(typeConverter, patterns);
    if (failed(
            applyPartialConversion(getFunction(), target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

void mlir::populateTensorBufferizePatterns(
    bufferization::BufferizeTypeConverter &typeConverter,
    RewritePatternSet &patterns) {
  patterns.add<BufferizeCastOp, BufferizeDimOp, BufferizeExtractOp,
               BufferizeFromElementsOp, BufferizeGenerateOp, BufferizeRankOp>(
      typeConverter, patterns.getContext());
}

std::unique_ptr<Pass> mlir::createTensorBufferizePass() {
  return std::make_unique<TensorBufferizePass>();
}
