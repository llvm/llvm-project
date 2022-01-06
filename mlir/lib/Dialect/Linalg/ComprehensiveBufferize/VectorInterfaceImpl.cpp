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
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              BufferizationState &state) const {
    assert(opOperand.get().getType().isa<RankedTensorType>() &&
           "only tensor types expected");
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               BufferizationState &state) const {
    assert(opOperand.get().getType().isa<RankedTensorType>() &&
           "only tensor types expected");
    return false;
  }

  OpResult getAliasingOpResult(Operation *op, OpOperand &opOperand,
                               BufferizationState &state) const {
    return OpResult();
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          BufferizationState &state) const {
    auto readOp = cast<vector::TransferReadOp>(op);
    assert(readOp.getShapedType().isa<TensorType>() &&
           "only tensor types expected");

    // TransferReadOp always reads from the bufferized op.source().
    Value buffer = state.lookupBuffer(rewriter, readOp.source());
    Value read = rewriter.create<vector::TransferReadOp>(
        readOp.getLoc(), readOp.getVectorType(), buffer, readOp.indices(),
        readOp.permutation_map(), readOp.padding(), readOp.mask(),
        readOp.in_boundsAttr());
    state.replaceOp(rewriter, op, read);
    return success();
  }
};

struct TransferWriteOpInterface
    : public BufferizableOpInterface::ExternalModel<TransferWriteOpInterface,
                                                    vector::TransferWriteOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              BufferizationState &state) const {
    assert(opOperand.get().getType().isa<TensorType>() &&
           "only tensor types expected");
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               BufferizationState &state) const {
    assert(opOperand.get().getType().isa<TensorType>() &&
           "only tensor types expected");
    return true;
  }

  OpResult getAliasingOpResult(Operation *op, OpOperand &opOperand,
                               BufferizationState &state) const {
    assert(opOperand.get().getType().isa<TensorType>() &&
           "only tensor types expected");
    return op->getOpResult(0);
  }

  BufferRelation bufferRelation(Operation *op, OpResult opResult,
                                const BufferizationAliasInfo &aliasInfo,
                                BufferizationState &state) const {
    return BufferRelation::Equivalent;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          BufferizationState &state) const {
    auto writeOp = cast<vector::TransferWriteOp>(op);
    assert(writeOp.getShapedType().isa<TensorType>() &&
           "only tensor types expected");

    // Create a new transfer_write on buffer that doesn't have a return value.
    // Leave the previous transfer_write to dead code as it still has uses at
    // this point.
    Value resultBuffer = state.getResultBuffer(rewriter, op->getResult(0));
    if (!resultBuffer)
      return failure();
    rewriter.create<vector::TransferWriteOp>(
        writeOp.getLoc(), writeOp.vector(), resultBuffer, writeOp.indices(),
        writeOp.permutation_mapAttr(), writeOp.in_boundsAttr());
    state.replaceOp(rewriter, op, resultBuffer);

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
