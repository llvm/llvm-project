//===- BufferizableOpInterfaceImpl.cpp - Impl. of BufferizableOpInterface -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arithmetic/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/Transforms/BufferUtils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"

using namespace mlir;
using namespace mlir::bufferization;

namespace {
/// Bufferization of arith.constant. Replace with memref.get_global.
struct ConstantOpInterface
    : public BufferizableOpInterface::ExternalModel<ConstantOpInterface,
                                                    arith::ConstantOp> {
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationState &state) const {
    auto constantOp = cast<arith::ConstantOp>(op);

    // Only ranked tensors are supported.
    if (!constantOp.getType().isa<RankedTensorType>())
      return failure();

    // Only constants inside a module are supported.
    auto moduleOp = constantOp->getParentOfType<ModuleOp>();
    if (!moduleOp)
      return failure();

    // Create global memory segment and replace tensor with memref pointing to
    // that memory segment.
    FailureOr<memref::GlobalOp> globalOp =
        getGlobalFor(constantOp, state.getOptions().bufferAlignment);
    if (failed(globalOp))
      return failure();
    memref::GlobalOp globalMemref = globalOp.getValue();
    replaceOpWithNewBufferizedOp<memref::GetGlobalOp>(
        rewriter, op, globalMemref.type(), globalMemref.getName());

    return success();
  }

  bool isWritable(Operation *op, Value value,
                  const BufferizationState &state) const {
    // Memory locations returned by memref::GetGlobalOp may not be written to.
    assert(value.isa<OpResult>());
    return false;
  }
};

struct IndexCastOpInterface
    : public BufferizableOpInterface::ExternalModel<IndexCastOpInterface,
                                                    arith::IndexCastOp> {
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
    return op->getResult(0);
  }

  BufferRelation bufferRelation(Operation *op, OpResult opResult,
                                const BufferizationState &state) const {
    return BufferRelation::Equivalent;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationState &state) const {
    auto castOp = cast<arith::IndexCastOp>(op);

    Value source = *state.getBuffer(rewriter, op->getOpOperand(0) /*in*/);
    auto sourceType = source.getType().cast<BaseMemRefType>();

    // Result type should have same layout and address space as the source type.
    MemRefLayoutAttrInterface layout = {};
    if (auto rankedMemRefType = sourceType.dyn_cast<MemRefType>())
      layout = rankedMemRefType.getLayout();
    Type resultType =
        getMemRefType(castOp.getType().cast<TensorType>(), state.getOptions(),
                      layout, sourceType.getMemorySpace());

    replaceOpWithNewBufferizedOp<arith::IndexCastOp>(rewriter, op, resultType,
                                                     source);
    return success();
  }
};

/// Bufferization of arith.select. Just replace the operands.
struct SelectOpInterface
    : public BufferizableOpInterface::ExternalModel<SelectOpInterface,
                                                    arith::SelectOp> {
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
    auto selectOp = cast<arith::SelectOp>(op);

    // `getBuffer` introduces copies if an OpOperand bufferizes out-of-place.
    // TODO: It would be more efficient to copy the result of the `select` op
    // instead of its OpOperands. In the worst case, 2 copies are inserted at
    // the moment (one for each tensor). When copying the op result, only one
    // copy would be needed.
    Value trueBuffer =
        *state.getBuffer(rewriter, selectOp->getOpOperand(1) /*true_value*/);
    Value falseBuffer =
        *state.getBuffer(rewriter, selectOp->getOpOperand(2) /*false_value*/);
    replaceOpWithNewBufferizedOp<arith::SelectOp>(
        rewriter, op, selectOp.getCondition(), trueBuffer, falseBuffer);
    return success();
  }

  BufferRelation bufferRelation(Operation *op, OpResult opResult,
                                const BufferizationState &state) const {
    return BufferRelation::None;
  }
};

} // namespace

void mlir::arith::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addOpInterface<ConstantOp, ConstantOpInterface>();
  registry.addOpInterface<IndexCastOp, IndexCastOpInterface>();
  registry.addOpInterface<SelectOp, SelectOpInterface>();
}
