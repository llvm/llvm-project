//===- BufferizableOpInterfaceImpl.cpp - Impl. of BufferizableOpInterface -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Vector/Transforms/BufferizableOpInterfaceImpl.h"

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"

using namespace mlir;
using namespace mlir::bufferization;
using namespace mlir::vector;

namespace mlir {
namespace vector {
namespace {

/// Bufferization of vector.transfer_read. Replaced with a new
/// vector.transfer_read that operates on a memref.
struct TransferReadOpInterface
    : public BufferizableOpInterface::ExternalModel<TransferReadOpInterface,
                                                    vector::TransferReadOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    assert(opOperand.get().getType().isa<RankedTensorType>() &&
           "only tensor types expected");
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    assert(opOperand.get().getType().isa<RankedTensorType>() &&
           "only tensor types expected");
    return false;
  }

  SmallVector<OpResult> getAliasingOpResult(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) const {
    return {};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto readOp = cast<vector::TransferReadOp>(op);
    assert(readOp.getShapedType().isa<TensorType>() &&
           "only tensor types expected");
    FailureOr<Value> buffer = getBuffer(rewriter, readOp.getSource(), options);
    if (failed(buffer))
      return failure();
    replaceOpWithNewBufferizedOp<vector::TransferReadOp>(
        rewriter, readOp, readOp.getVectorType(), *buffer, readOp.getIndices(),
        readOp.getPermutationMap(), readOp.getPadding(), readOp.getMask(),
        readOp.getInBoundsAttr());
    return success();
  }
};

/// Bufferization of vector.transfer_write. Replace with a new
/// vector.transfer_write that operates on a memref.
struct TransferWriteOpInterface
    : public BufferizableOpInterface::ExternalModel<TransferWriteOpInterface,
                                                    vector::TransferWriteOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    assert(opOperand.get().getType().isa<TensorType>() &&
           "only tensor types expected");
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    assert(opOperand.get().getType().isa<TensorType>() &&
           "only tensor types expected");
    return true;
  }

  SmallVector<OpResult> getAliasingOpResult(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) const {
    assert(opOperand.get().getType().isa<TensorType>() &&
           "only tensor types expected");
    return {op->getOpResult(0)};
  }

  BufferRelation bufferRelation(Operation *op, OpResult opResult,
                                const AnalysisState &state) const {
    return BufferRelation::Equivalent;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto writeOp = cast<vector::TransferWriteOp>(op);
    assert(writeOp.getShapedType().isa<TensorType>() &&
           "only tensor types expected");

    // Create a new transfer_write on buffer that doesn't have a return value.
    FailureOr<Value> resultBuffer =
        getBuffer(rewriter, writeOp.getSource(), options);
    if (failed(resultBuffer))
      return failure();
    rewriter.create<vector::TransferWriteOp>(
        writeOp.getLoc(), writeOp.getVector(), *resultBuffer,
        writeOp.getIndices(), writeOp.getPermutationMapAttr(),
        writeOp.getMask(), writeOp.getInBoundsAttr());
    replaceOpWithBufferizedValues(rewriter, op, *resultBuffer);

    return success();
  }
};

/// Bufferization of vector.gather. Replaced with a new vector.gather that
/// operates on a memref.
struct GatherOpInterface
    : public BufferizableOpInterface::ExternalModel<GatherOpInterface,
                                                    vector::GatherOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    assert(opOperand.get().getType().isa<RankedTensorType>() &&
           "only tensor types expected");
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    assert(opOperand.get().getType().isa<RankedTensorType>() &&
           "only tensor types expected");
    return false;
  }

  SmallVector<OpResult> getAliasingOpResult(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) const {
    return {};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto gatherOp = cast<vector::GatherOp>(op);
    assert(gatherOp.getBaseType().isa<TensorType>() &&
           "only tensor types expected");
    FailureOr<Value> buffer = getBuffer(rewriter, gatherOp.getBase(), options);
    if (failed(buffer))
      return failure();
    replaceOpWithNewBufferizedOp<vector::GatherOp>(
        rewriter, gatherOp, gatherOp.getVectorType(), *buffer,
        gatherOp.getIndices(), gatherOp.getIndexVec(), gatherOp.getMask(),
        gatherOp.getPassThru());
    return success();
  }
};

} // namespace
} // namespace vector
} // namespace mlir

void mlir::vector::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, vector::VectorDialect *dialect) {
    TransferReadOp::attachInterface<TransferReadOpInterface>(*ctx);
    TransferWriteOp::attachInterface<TransferWriteOpInterface>(*ctx);
    GatherOp::attachInterface<GatherOpInterface>(*ctx);
  });
}
