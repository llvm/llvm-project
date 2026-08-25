//===- BufferizableOpInterfaceImpl.cpp - Impl. of BufferizableOpInterface -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/Transforms/BufferizableOpInterfaceImpl.h"

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"

using namespace mlir;
using namespace mlir::bufferization;

namespace mlir {
namespace bufferization {
namespace {

struct AllocTensorOpInterface
    : public BufferizableOpInterface::ExternalModel<AllocTensorOpInterface,
                                                    AllocTensorOp> {
  bool bufferizesToAllocation(Operation *op, Value value) const { return true; }

  bool resultBufferizesToMemoryWrite(Operation *op, OpResult opResult,
                                     const AnalysisState &state) const {
    // AllocTensorOps do not write unless they have a `copy` value.
    return static_cast<bool>(cast<AllocTensorOp>(op).getCopy());
  }

  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    assert(&opOperand == &cast<AllocTensorOp>(op).getCopyMutable()[0] &&
           "expected copy operand");
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    assert(&opOperand == &cast<AllocTensorOp>(op).getCopyMutable()[0] &&
           "expected copy operand");
    return false;
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    // This is a new allocation. It does not alias with any other buffer.
    return {};
  }

  FailureOr<BufferLikeType>
  getBufferType(Operation *op, Value value, const BufferizationOptions &options,
                const BufferizationState &state,
                SmallVector<Value> &invocationStack) const {
    auto allocTensorOp = cast<AllocTensorOp>(op);
    assert(value == allocTensorOp.getResult() && "invalid value");

    // Compute memory space of this allocation.
    Attribute memorySpace;
    if (allocTensorOp.getMemorySpace().has_value()) {
      memorySpace = *allocTensorOp.getMemorySpace();
    } else if (allocTensorOp.getCopy()) {
      auto copyBufferType =
          bufferization::detail::asMemRefType(bufferization::getBufferType(
              allocTensorOp.getCopy(), options, state, invocationStack));
      if (failed(copyBufferType))
        return failure();
      memorySpace = copyBufferType->getMemorySpace();
    } else if (auto ms = options.defaultMemorySpaceFn(
                   cast<TensorLikeType>(allocTensorOp.getType()))) {
      memorySpace = *ms;
    } else {
      return op->emitError("could not infer memory space");
    }

    return cast<BufferLikeType>(getMemRefTypeWithStaticIdentityLayout(
        allocTensorOp.getType(), memorySpace));
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          BufferizationState &state) const {
    auto allocTensorOp = cast<AllocTensorOp>(op);
    OpBuilder::InsertionGuard g(rewriter);
    Location loc = allocTensorOp.getLoc();

    // Nothing to do for dead AllocTensorOps.
    if (op->getUses().empty()) {
      rewriter.eraseOp(op);
      return success();
    }

    // Get "copy" buffer.
    Value copyBuffer;
    if (allocTensorOp.getCopy()) {
      FailureOr<Value> maybeCopyBuffer = bufferization::getBuffer(
          rewriter, allocTensorOp.getCopy(), options, state);
      if (failed(maybeCopyBuffer))
        return failure();
      copyBuffer = *maybeCopyBuffer;
    }

    // Create memory allocation.
    auto allocType =
        bufferization::getBufferType(allocTensorOp.getResult(), options, state);
    if (failed(allocType))
      return failure();
    SmallVector<Value> dynamicDims = allocTensorOp.getDynamicSizes();
    if (allocTensorOp.getCopy()) {
      assert(dynamicDims.empty() && "expected either `copy` or `dynamicDims`");
      populateDynamicDimSizes(rewriter, loc, copyBuffer, dynamicDims);
    }
    FailureOr<Value> alloc =
        options.allocationFn(rewriter, loc, llvm::cast<MemRefType>(*allocType),
                             dynamicDims, options.bufferAlignment);
    if (failed(alloc))
      return failure();

    // Create memory copy (if any).
    if (allocTensorOp.getCopy()) {
      if (failed(options.memCpyFn(rewriter, loc, copyBuffer, *alloc)))
        return failure();
    }

    // Replace op.
    replaceOpWithBufferizedValues(rewriter, op, *alloc);

    return success();
  }
};

struct DeallocTensorOpInterface
    : public BufferizableOpInterface::ExternalModel<DeallocTensorOpInterface,
                                                    DeallocTensorOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return false;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return false;
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    return {};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          BufferizationState &state) const {
    auto deallocTensorOp = cast<DeallocTensorOp>(op);
    FailureOr<Value> buffer = bufferization::getBuffer(
        rewriter, deallocTensorOp.getTensor(), options, state);
    if (failed(buffer))
      return failure();
    memref::DeallocOp::create(rewriter, deallocTensorOp.getLoc(), *buffer);
    rewriter.eraseOp(op);
    return success();
  }
};

struct MaterializeInDestinationOpInterface
    : public BufferizableOpInterface::ExternalModel<
          MaterializeInDestinationOpInterface, MaterializeInDestinationOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return opOperand == cast<MaterializeInDestinationOp>(op).getSourceMutable();
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    auto materializeOp = cast<MaterializeInDestinationOp>(op);
    if (opOperand == materializeOp.getDestMutable()) {
      assert(isa<TensorType>(materializeOp.getDest().getType()) &&
             "expected tensor type");
      return true;
    }
    return false;
  }

  bool mustBufferizeInPlace(Operation *op, OpOperand &opOperand,
                            const AnalysisState &state) const {
    // The source is only read and not written, so it always bufferizes in-place
    // by default. The destination is written and is forced to bufferize
    // in-place (if it is a tensor).
    return true;
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    auto materializeOp = cast<MaterializeInDestinationOp>(op);
    if (opOperand == materializeOp.getDestMutable()) {
      assert(isa<TensorType>(materializeOp.getDest().getType()) &&
             "expected tensor type");
      return {{op->getResult(0), BufferRelation::Equivalent}};
    }
    return {};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          BufferizationState &state) const {
    auto materializeOp = cast<MaterializeInDestinationOp>(op);
    bool tensorDest = isa<TensorType>(materializeOp.getDest().getType());
    Value buffer;
    if (tensorDest) {
      FailureOr<Value> maybeBuffer = bufferization::getBuffer(
          rewriter, materializeOp.getDest(), options, state);
      if (failed(maybeBuffer))
        return failure();
      buffer = *maybeBuffer;
    } else {
      assert(isa<BaseMemRefType>(materializeOp.getDest().getType()) &&
             "expected memref type");
      buffer = materializeOp.getDest();
    }
    auto srcBuffer = bufferization::getBuffer(
        rewriter, materializeOp.getSource(), options, state);
    if (failed(srcBuffer))
      return failure();
    if (failed(options.memCpyFn(rewriter, materializeOp.getLoc(), *srcBuffer,
                                buffer)))
      return failure();
    replaceOpWithBufferizedValues(
        rewriter, op, tensorDest ? ValueRange(buffer) : ValueRange());
    return success();
  }

  bool bufferizesToElementwiseAccess(Operation *op, const AnalysisState &state,
                                     ArrayRef<OpOperand *> opOperands) const {
    // As elements are copied from the "source" buffer to the "dest" buffer,
    // already copied elements are not read a second time.
    return true;
  }

  bool isWritable(Operation *op, Value value,
                  const AnalysisState &state) const {
    auto materializeOp = cast<MaterializeInDestinationOp>(op);
    return isa<TensorType>(materializeOp.getDest().getType())
               ? true
               : materializeOp.getWritable();
  }
};

// Note: ToBufferOp / ToTensorOp are temporary ops that are inserted at the
// bufferization boundary. When One-Shot bufferization is complete, there should
// be no such ops left over. If `allowUnknownOps` (or after running a partial
// bufferization pass), such ops may be part of the resulting IR, but such IR
// may no longer be analyzable by One-Shot analysis.

struct ToTensorOpInterface
    : public BufferizableOpInterface::ExternalModel<ToTensorOpInterface,
                                                    ToTensorOp> {
  bool isWritable(Operation *op, Value value,
                  const AnalysisState &state) const {
    return cast<ToTensorOp>(op).getWritable();
  }

  FailureOr<BufferLikeType>
  getBufferType(Operation *op, Value value, const BufferizationOptions &options,
                const BufferizationState &state,
                SmallVector<Value> &invocationStack) const {
    return cast<ToTensorOp>(op).getBuffer().getType();
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          BufferizationState &state) const {
    // to_tensor/to_buffer pairs fold away after bufferization.
    return success();
  }
};

struct ToBufferOpInterface
    : public BufferizableOpInterface::ExternalModel<ToBufferOpInterface,
                                                    ToBufferOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    // It is unknown whether the resulting memref will be read or not.
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return !cast<ToBufferOp>(op).getReadOnly();
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    return {};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          BufferizationState &state) const {
    // Fold to_buffer(to_tensor(x)) to x. Insert a cast if necessary.
    (void)foldToBufferToTensorPair(rewriter, cast<ToBufferOp>(op), options);
    // Note: The return value of `bufferize` indicates whether there was an
    // error or not. (And not whether the pattern matched or not.)
    return success();
  }
};

} // namespace
} // namespace bufferization
} // namespace mlir

void mlir::bufferization::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, BufferizationDialect *dialect) {
    AllocTensorOp::attachInterface<AllocTensorOpInterface>(*ctx);
    DeallocTensorOp::attachInterface<DeallocTensorOpInterface>(*ctx);
    MaterializeInDestinationOp::attachInterface<
        MaterializeInDestinationOpInterface>(*ctx);
    ToBufferOp::attachInterface<ToBufferOpInterface>(*ctx);
    ToTensorOp::attachInterface<ToTensorOpInterface>(*ctx);
  });
}
