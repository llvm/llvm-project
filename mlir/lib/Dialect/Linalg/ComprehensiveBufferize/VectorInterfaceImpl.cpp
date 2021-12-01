//===- VectorInterfaceImpl.cpp - Vector Impl. of BufferizableOpInterface --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/ComprehensiveBufferize/VectorInterfaceImpl.h"
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/BufferizableOpInterface.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"

namespace mlir {
namespace linalg {
namespace comprehensive_bufferize {
namespace vector_ext {

struct TransferReadOpInterface
    : public BufferizableOpInterface::ExternalModel<TransferReadOpInterface,
                                                    vector::TransferReadOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand) const {
    assert(opOperand.get().getType().isa<RankedTensorType>() &&
           "only tensor types expected");
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand) const {
    assert(opOperand.get().getType().isa<RankedTensorType>() &&
           "only tensor types expected");
    return false;
  }

  OpResult getAliasingOpResult(Operation *op, OpOperand &opOperand) const {
    return OpResult();
  }

  LogicalResult bufferize(Operation *op, OpBuilder &b,
                          BufferizationState &state) const {
    auto transferReadOp = cast<vector::TransferReadOp>(op);

    // Take a guard before anything else.
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPoint(op);

    // TransferReadOp always reads from the bufferized op.source().
    assert(transferReadOp.getShapedType().isa<TensorType>() &&
           "only tensor types expected");
    Value v = state.lookupBuffer(transferReadOp.source());
    transferReadOp.sourceMutable().assign(v);
    return success();
  }
};

struct TransferWriteOpInterface
    : public BufferizableOpInterface::ExternalModel<TransferWriteOpInterface,
                                                    vector::TransferWriteOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand) const {
    assert(opOperand.get().getType().isa<TensorType>() &&
           "only tensor types expected");
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand) const {
    assert(opOperand.get().getType().isa<TensorType>() &&
           "only tensor types expected");
    return true;
  }

  SmallVector<OpOperand *> getAliasingOpOperand(Operation *op,
                                                OpResult opResult) const {
    return {&op->getOpOperand(1)};
  }

  OpResult getAliasingOpResult(Operation *op, OpOperand &opOperand) const {
    assert(opOperand.get().getType().isa<TensorType>() &&
           "only tensor types expected");
    return op->getOpResult(0);
  }

  BufferRelation bufferRelation(Operation *op, OpOperand &opOperand) const {
    return BufferRelation::Equivalent;
  }

  LogicalResult bufferize(Operation *op, OpBuilder &b,
                          BufferizationState &state) const {
    auto writeOp = cast<vector::TransferWriteOp>(op);

    // Take a guard before anything else.
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPoint(op);

    // Create a new transfer_write on buffer that doesn't have a return value.
    // Leave the previous transfer_write to dead code as it still has uses at
    // this point.
    assert(writeOp.getShapedType().isa<TensorType>() &&
           "only tensor types expected");
    Value resultBuffer = getResultBuffer(b, op->getResult(0), state);
    if (!resultBuffer)
      return failure();
    b.create<vector::TransferWriteOp>(
        writeOp.getLoc(), writeOp.vector(), resultBuffer, writeOp.indices(),
        writeOp.permutation_mapAttr(), writeOp.in_boundsAttr());
    state.mapBuffer(op->getResult(0), resultBuffer);

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
