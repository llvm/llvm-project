//===- Bufferize.cpp - Bufferization of linalg ops ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Arithmetic/Utils/Utils.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/StandardOps/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

using namespace ::mlir;
using namespace ::mlir::linalg;

static Value cloneMemref(Location loc, Value memref, OpBuilder &b) {
  auto memrefType = memref.getType().cast<MemRefType>();
  auto alloc = b.create<memref::AllocOp>(loc, memrefType,
                                         getDynOperands(loc, memref, b));
  b.create<memref::CopyOp>(loc, memref, alloc);
  return alloc;
}

static LogicalResult
allocateBuffersForResults(Location loc, LinalgOp linalgOp, ValueRange outputs,
                          SmallVectorImpl<Value> &resultBuffers, OpBuilder &b) {
  // Lazily compute loopRanges.
  SmallVector<Range, 4> loopRanges;

  // Allocate a buffer for every tensor result.
  assert(linalgOp.getNumOutputs() == linalgOp->getNumResults());
  for (const auto &en : llvm::enumerate(linalgOp->getResultTypes())) {
    size_t resultIndex = en.index();
    Type resultType = en.value();

    auto tensorType = resultType.dyn_cast<RankedTensorType>();
    if (tensorType == nullptr) {
      linalgOp.emitOpError()
          << "tensor to buffer conversion expects ranked tensor results";
      return failure();
    }
    auto tensorShape = tensorType.getShape();
    auto memrefType = MemRefType::get(tensorShape, tensorType.getElementType());
    Value resultTensor = outputs[resultIndex];

    // Clone output buffers whose value is actually used.
    OpOperand *tiedOpOperand = linalgOp.getOutputOperand(resultIndex);
    if (linalgOp.payloadUsesValueFromOperand(tiedOpOperand)) {
      resultBuffers.push_back(cloneMemref(loc, resultTensor, b));
      continue;
    }

    // Allocate buffers for statically-shaped results.
    if (memrefType.hasStaticShape()) {
      resultBuffers.push_back(b.create<memref::AllocOp>(loc, memrefType));
      continue;
    }

    resultBuffers.push_back(b.create<memref::AllocOp>(
        loc, memrefType, getDynOperands(loc, resultTensor, b)));
  }
  return success();
}

/// Create linalg op on buffers given the original tensor-based operation and
/// the buffers for the outputs.
LinalgOp
mlir::linalg::createLinalgOpOnBuffers(ConversionPatternRewriter &rewriter,
                                      LinalgOp linalgOp, ValueRange inputs,
                                      ValueRange outputs) {
  SmallVector<Value, 8> newOperands = inputs;
  newOperands.append(outputs.begin(), outputs.end());
  auto *newOp = linalgOp.cloneWithoutRegions(rewriter, linalgOp.getLoc(),
                                             /*resultTypes=*/ArrayRef<Type>{},
                                             newOperands);
  for (auto regions : llvm::zip(linalgOp->getRegions(), newOp->getRegions())) {
    auto &oldRegion = std::get<0>(regions);
    auto &newRegion = std::get<1>(regions);
    rewriter.inlineRegionBefore(oldRegion, newRegion, newRegion.begin());
  }
  return newOp;
}

//===----------------------------------------------------------------------===//
// Bufferization patterns.
//===----------------------------------------------------------------------===//

namespace {

/// Conversion pattern that replaces `linalg.init_tensor` with allocation.
class BufferizeInitTensorOp : public OpConversionPattern<InitTensorOp> {
public:
  using OpConversionPattern<InitTensorOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(InitTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<memref::AllocOp>(
        op, getTypeConverter()->convertType(op.getType()).cast<MemRefType>(),
        adaptor.sizes());
    return success();
  }
};

/// Conversion pattern that bufferizes `linalg.fill` operation.
class BufferizeFillOp : public OpConversionPattern<FillOp> {
public:
  using OpConversionPattern<FillOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(FillOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (!op.output().getType().isa<TensorType>())
      return rewriter.notifyMatchFailure(op,
                                         "operand must be of a tensor type");

    rewriter.create<FillOp>(op.getLoc(), adaptor.value(), adaptor.output());
    rewriter.replaceOp(op, adaptor.output());

    return success();
  }
};

/// Generic conversion pattern that matches any LinalgOp. This avoids template
/// instantiating one pattern for each LinalgOp.
class BufferizeAnyLinalgOp : public OpInterfaceConversionPattern<LinalgOp> {
public:
  using OpInterfaceConversionPattern<LinalgOp>::OpInterfaceConversionPattern;

  LogicalResult
  matchAndRewrite(LinalgOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    // GenericOpAdaptor below expects an `operand_segment_sizes` attribute.
    if (!op->hasAttr("operand_segment_sizes"))
      return failure();

    // We abuse the GenericOpAdaptor here.
    // TODO: Manually create an Adaptor that captures inputs and outputs for all
    // linalg::LinalgOp interface ops.
    linalg::GenericOpAdaptor adaptor(operands, op->getAttrDictionary());

    Location loc = op.getLoc();
    SmallVector<Value, 2> newOutputBuffers;

    if (failed(allocateBuffersForResults(loc, op, adaptor.outputs(),
                                         newOutputBuffers, rewriter))) {
      return op.emitOpError()
             << "Failed to allocate buffers for tensor results.";
    }
    createLinalgOpOnBuffers(rewriter, op, adaptor.inputs(), newOutputBuffers);
    // Replace the results of the old op with the new output buffers.
    rewriter.replaceOp(op, newOutputBuffers);
    return success();
  }
};
} // namespace

namespace {
/// Converts Linalg operations that work on tensor-type operands or results to
/// work on buffers.
struct LinalgBufferizePass : public LinalgBufferizeBase<LinalgBufferizePass> {
  void runOnOperation() override {
    MLIRContext &context = getContext();
    ConversionTarget target(context);
    bufferization::BufferizeTypeConverter typeConverter;

    // Mark all Standard operations legal.
    target.addLegalDialect<arith::ArithmeticDialect, AffineDialect,
                           memref::MemRefDialect, StandardOpsDialect,
                           tensor::TensorDialect>();
    target.addIllegalOp<InitTensorOp>();

    // Mark all Linalg operations illegal as long as they work on tensors.
    auto isLegalOperation = [&](Operation *op) {
      return typeConverter.isLegal(op);
    };
    target.addDynamicallyLegalDialect<linalg::LinalgDialect>(isLegalOperation);

    RewritePatternSet patterns(&context);
    populateLinalgBufferizePatterns(typeConverter, patterns);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<FuncOp>> mlir::createLinalgBufferizePass() {
  return std::make_unique<LinalgBufferizePass>();
}

void mlir::linalg::populateLinalgBufferizePatterns(
    bufferization::BufferizeTypeConverter &typeConverter,
    RewritePatternSet &patterns) {
  // TODO: Drop this once tensor constants work in standard.
  // clang-format off
  patterns.add<
      BufferizeAnyLinalgOp,
      BufferizeFillOp,
      BufferizeInitTensorOp
    >(typeConverter, patterns.getContext());
  // clang-format on
}
