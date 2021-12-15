//===- Bufferize.cpp - Bufferization of linalg ops ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/StandardOps/Transforms/Passes.h"
#include "mlir/Dialect/StandardOps/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

using namespace ::mlir;
using namespace ::mlir::linalg;

static Value cloneMemref(Location loc, Value memref, OpBuilder &b) {
  auto memrefType = memref.getType().cast<MemRefType>();
  auto alloc = b.create<memref::AllocOp>(loc, memrefType,
                                         getDynOperands(loc, memref, b));
  b.create<linalg::CopyOp>(loc, memref, alloc);
  return alloc;
}

static LogicalResult
allocateBuffersForResults(Location loc, LinalgOp linalgOp, ValueRange outputs,
                          SmallVectorImpl<Value> &resultBuffers, OpBuilder &b) {
  // Lazily compute loopRanges.
  SmallVector<Range, 4> loopRanges;

  // Allocate a buffer for every tensor result.
  assert(linalgOp.getNumOutputs() == linalgOp->getNumResults());
  for (auto en : llvm::enumerate(linalgOp->getResultTypes())) {
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

/// Conversion pattern that replaces `linalg.tensor_reshape` with
/// `linalg.reshape`.
template <typename TensorReshapeOp,
          typename Adaptor = typename TensorReshapeOp::Adaptor>
class BufferizeTensorReshapeOp : public OpConversionPattern<TensorReshapeOp> {
public:
  using OpConversionPattern<TensorReshapeOp>::OpConversionPattern;
  using ReshapeOp = typename std::conditional_t<
      std::is_same<TensorReshapeOp, tensor::ExpandShapeOp>::value,
      memref::ExpandShapeOp, memref::CollapseShapeOp>;

  LogicalResult
  matchAndRewrite(TensorReshapeOp op, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<ReshapeOp>(op,
                                           this->getTypeConverter()
                                               ->convertType(op.getType())
                                               .template cast<MemRefType>(),
                                           adaptor.src(),
                                           adaptor.reassociation());
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

/// Convert `extract_slice %t [offsets][sizes][strides] -> %st` to an
/// alloc + copy pattern.
/// ```
///   %a = alloc(sizes)
///   %sv = subview %source [offsets][sizes][strides]
///   linalg_copy(%sv, %a)
/// ```
///
/// This pattern is arguable a std pattern once linalg::CopyOp becomes
/// std::CopyOp.
class ExtractSliceOpConverter
    : public OpConversionPattern<tensor::ExtractSliceOp> {
public:
  using OpConversionPattern<tensor::ExtractSliceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tensor::ExtractSliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Value sourceMemref = adaptor.source();
    assert(sourceMemref.getType().isa<MemRefType>());

    MemRefType subviewMemRefType =
        getTypeConverter()->convertType(op.getType()).cast<MemRefType>();
    // op.sizes() capture exactly the dynamic alloc operands matching the
    // subviewMemRefType thanks to subview/slice canonicalization and
    // verification.
    Value alloc = rewriter.create<memref::AllocOp>(
        op.getLoc(), subviewMemRefType, op.sizes());
    Value subView = rewriter.create<memref::SubViewOp>(
        op.getLoc(), sourceMemref, op.getMixedOffsets(), op.getMixedSizes(),
        op.getMixedStrides());
    rewriter.create<linalg::CopyOp>(op.getLoc(), subView, alloc);
    rewriter.replaceOp(op, alloc);
    return success();
  }
};

/// Convert `insert_slice %source into %dest [offsets][sizes][strides] ->
/// %t` to an buffer_cast + subview + copy + tensor_load pattern.
/// buffer_cast and tensor_load are inserted automatically by the
/// conversion infra:
/// ```
///   %sv = subview %dest [offsets][sizes][strides]
///   linalg_copy(%source, %sv)
///   // replace with %dest
/// ```
///
/// This pattern is arguable a std pattern once linalg::CopyOp becomes
/// std::CopyOp.
class InsertSliceOpConverter
    : public OpConversionPattern<tensor::InsertSliceOp> {
public:
  using OpConversionPattern<tensor::InsertSliceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tensor::InsertSliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Value sourceMemRef = adaptor.source();
    assert(sourceMemRef.getType().isa<MemRefType>());

    // For now, be conservative and copy the converted input memref.
    // In general, the converted input memref here could be aliased or could
    // point into constant memory, so mutating it would lead to miscompilations.
    Value destMemRef = cloneMemref(op.getLoc(), adaptor.dest(), rewriter);
    assert(destMemRef.getType().isa<MemRefType>());

    // Take a subview to copy the small memref.
    Value subview = rewriter.create<memref::SubViewOp>(
        op.getLoc(), destMemRef, op.getMixedOffsets(), op.getMixedSizes(),
        op.getMixedStrides());
    // Copy the small memref.
    rewriter.create<linalg::CopyOp>(op.getLoc(), sourceMemRef, subview);
    rewriter.replaceOp(op, destMemRef);
    return success();
  }
};

class VectorTransferReadOpConverter
    : public OpConversionPattern<vector::TransferReadOp> {
public:
  using OpConversionPattern<vector::TransferReadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::TransferReadOp readOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (readOp.getShapedType().isa<MemRefType>())
      return failure();
    rewriter.replaceOpWithNewOp<vector::TransferReadOp>(
        readOp, readOp.getType(), adaptor.source(), adaptor.indices(),
        adaptor.permutation_map(), adaptor.padding(), adaptor.mask(),
        adaptor.in_bounds());
    return success();
  }
};

class VectorTransferWriteOpConverter
    : public OpConversionPattern<vector::TransferWriteOp> {
public:
  using OpConversionPattern<vector::TransferWriteOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::TransferWriteOp writeOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (writeOp.getShapedType().isa<MemRefType>())
      return failure();
    rewriter.create<vector::TransferWriteOp>(
        writeOp.getLoc(), adaptor.vector(), adaptor.source(), adaptor.indices(),
        adaptor.permutation_map(),
        adaptor.in_bounds() ? adaptor.in_bounds() : ArrayAttr());
    rewriter.replaceOp(writeOp, adaptor.source());
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
    target.addIllegalOp<InitTensorOp, PadTensorOp, tensor::CollapseShapeOp,
                        tensor::ExpandShapeOp, tensor::ExtractSliceOp,
                        tensor::InsertSliceOp>();

    // Mark all Linalg operations illegal as long as they work on tensors.
    auto isLegalOperation = [&](Operation *op) {
      return typeConverter.isLegal(op);
    };
    target.addDynamicallyLegalDialect<linalg::LinalgDialect>(isLegalOperation);
    target
        .addDynamicallyLegalOp<vector::TransferReadOp, vector::TransferWriteOp>(
            isLegalOperation);

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
      BufferizeInitTensorOp,
      BufferizeTensorReshapeOp<tensor::ExpandShapeOp>,
      BufferizeTensorReshapeOp<tensor::CollapseShapeOp>,
      ExtractSliceOpConverter,
      InsertSliceOpConverter,
      VectorTransferReadOpConverter,
      VectorTransferWriteOpConverter
    >(typeConverter, patterns.getContext());
  // clang-format on
  patterns.add<GeneralizePadTensorOpPattern>(patterns.getContext());
}
