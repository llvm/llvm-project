//===- BufferizableOpInterfaceImpl.cpp - Impl. of BufferizableOpInterface -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/StandardOps/Transforms/BufferizableOpInterfaceImpl.h"

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"

using namespace mlir;
using namespace mlir::bufferization;

namespace mlir {
namespace {

/// Bufferization of std.select. Just replace the operands.
struct SelectOpInterface
    : public BufferizableOpInterface::ExternalModel<SelectOpInterface,
                                                    SelectOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const BufferizationState &state) const {
    return false;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const BufferizationState &state) const {
    return false;
  }

  OpResult getAliasingOpResult(Operation *op, OpOperand &opOperand,
                               const BufferizationState &state) const {
    return op->getOpResult(0) /*result*/;
  }

  SmallVector<OpOperand *>
  getAliasingOpOperand(Operation *op, OpResult opResult,
                       const BufferizationState &state) const {
    return {&op->getOpOperand(1) /*true_value*/,
            &op->getOpOperand(2) /*false_value*/};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationState &state) const {
    auto selectOp = cast<SelectOp>(op);
    // `getBuffer` introduces copies if an OpOperand bufferizes out-of-place.
    // TODO: It would be more efficient to copy the result of the `select` op
    // instead of its OpOperands. In the worst case, 2 copies are inserted at
    // the moment (one for each tensor). When copying the op result, only one
    // copy would be needed.
    Value trueBuffer =
        *state.getBuffer(rewriter, selectOp->getOpOperand(1) /*true_value*/);
    Value falseBuffer =
        *state.getBuffer(rewriter, selectOp->getOpOperand(2) /*false_value*/);
    replaceOpWithNewBufferizedOp<SelectOp>(
        rewriter, op, selectOp.getCondition(), trueBuffer, falseBuffer);
    return success();
  }

  BufferRelation bufferRelation(Operation *op, OpResult opResult,
                                const BufferizationState &state) const {
    return BufferRelation::None;
  }
};

} // namespace
} // namespace mlir

void mlir::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addOpInterface<SelectOp, SelectOpInterface>();
}
