//===- VectorInterfaceImpl.cpp - Vector Impl. of BufferizableOpInterface --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/ComprehensiveBufferize/VectorInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"

using namespace mlir::bufferization;

namespace mlir {
namespace linalg {
namespace comprehensive_bufferize {
namespace vector_ext {

/// Bufferization of vector.transfer_read. Replaced with a new
/// vector.transfer_read that operates on a memref.
struct TransferReadOpInterface
    : public BufferizableOpInterface::ExternalModel<TransferReadOpInterface,
                                                    vector::TransferReadOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const BufferizationState &state) const {
    assert(opOperand.get().getType().isa<RankedTensorType>() &&
           "only tensor types expected");
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const BufferizationState &state) const {
    assert(opOperand.get().getType().isa<RankedTensorType>() &&
           "only tensor types expected");
    return false;
  }

  OpResult getAliasingOpResult(Operation *op, OpOperand &opOperand,
                               const BufferizationState &state) const {
    return OpResult();
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationState &state) const {
    auto readOp = cast<vector::TransferReadOp>(op);
    assert(readOp.getShapedType().isa<TensorType>() &&
           "only tensor types expected");

    // TransferReadOp always reads from the bufferized op.source().
    Value buffer =
        *state.getBuffer(rewriter, readOp->getOpOperand(0) /*source*/);
    replaceOpWithNewBufferizedOp<vector::TransferReadOp>(
        rewriter, readOp, readOp.getVectorType(), buffer, readOp.indices(),
        readOp.permutation_map(), readOp.padding(), readOp.mask(),
        readOp.in_boundsAttr());
    return success();
  }
};

/// Bufferization of vector.transfer_write. Replace with a new
/// vector.transfer_write that operates on a memref.
struct TransferWriteOpInterface
    : public BufferizableOpInterface::ExternalModel<TransferWriteOpInterface,
                                                    vector::TransferWriteOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const BufferizationState &state) const {
    assert(opOperand.get().getType().isa<TensorType>() &&
           "only tensor types expected");
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const BufferizationState &state) const {
    assert(opOperand.get().getType().isa<TensorType>() &&
           "only tensor types expected");
    return true;
  }

  OpResult getAliasingOpResult(Operation *op, OpOperand &opOperand,
                               const BufferizationState &state) const {
    assert(opOperand.get().getType().isa<TensorType>() &&
           "only tensor types expected");
    return op->getOpResult(0);
  }

  BufferRelation bufferRelation(Operation *op, OpResult opResult,
                                const BufferizationState &state) const {
    return BufferRelation::Equivalent;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationState &state) const {
    auto writeOp = cast<vector::TransferWriteOp>(op);
    assert(writeOp.getShapedType().isa<TensorType>() &&
           "only tensor types expected");

    // Create a new transfer_write on buffer that doesn't have a return value.
    // Leave the previous transfer_write to dead code as it still has uses at
    // this point.
    FailureOr<Value> resultBuffer =
        state.getBuffer(rewriter, op->getOpOperand(1) /*source*/);
    if (failed(resultBuffer))
      return failure();
    rewriter.create<vector::TransferWriteOp>(
        writeOp.getLoc(), writeOp.vector(), *resultBuffer, writeOp.indices(),
        writeOp.permutation_mapAttr(), writeOp.in_boundsAttr());
    replaceOpWithBufferizedValues(rewriter, op, *resultBuffer);

    return success();
  }
};

} // namespace vector_ext
} // namespace comprehensive_bufferize
} // namespace linalg
} // namespace mlir

void mlir::linalg::comprehensive_bufferize::vector_ext::
    registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry) {
  registry.addOpInterface<vector::TransferReadOp,
                          vector_ext::TransferReadOpInterface>();
  registry.addOpInterface<vector::TransferWriteOp,
                          vector_ext::TransferWriteOpInterface>();
}
