//===- ArithInterfaceImpl.cpp - Arith Impl. of BufferizableOpInterface ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/ComprehensiveBufferize/ArithInterfaceImpl.h"

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/BufferizableOpInterface.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/Transforms/BufferUtils.h"

namespace mlir {
namespace linalg {
namespace comprehensive_bufferize {
namespace arith_ext {

struct ConstantOpInterface
    : public BufferizableOpInterface::ExternalModel<ConstantOpInterface,
                                                    arith::ConstantOp> {
  SmallVector<OpOperand *> getAliasingOpOperand(Operation *op,
                                                OpResult opResult) const {
    return {};
  }

  LogicalResult bufferize(Operation *op, OpBuilder &b,
                          BufferizationState &state) const {
    auto constantOp = cast<arith::ConstantOp>(op);
    if (!constantOp.getResult().getType().isa<TensorType>())
      return success();
    assert(constantOp.getType().dyn_cast<RankedTensorType>() &&
           "not a constant ranked tensor");
    auto moduleOp = constantOp->getParentOfType<ModuleOp>();
    if (!moduleOp) {
      return constantOp.emitError(
          "cannot bufferize constants not within builtin.module op");
    }
    GlobalCreator globalCreator(moduleOp);

    // Take a guard before anything else.
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPoint(constantOp);

    auto globalMemref = globalCreator.getGlobalFor(constantOp);
    Value memref = b.create<memref::GetGlobalOp>(
        constantOp.getLoc(), globalMemref.type(), globalMemref.getName());
    state.aliasInfo.insertNewBufferEquivalence(memref, constantOp.getResult());
    state.mapBuffer(constantOp, memref);

    return success();
  }

  bool isWritable(Operation *op, Value value) const {
    // Memory locations returned by memref::GetGlobalOp may not be written to.
    assert(value.isa<OpResult>());
    return false;
  }
};

} // namespace arith_ext
} // namespace comprehensive_bufferize
} // namespace linalg
} // namespace mlir

void mlir::linalg::comprehensive_bufferize::arith_ext::
    registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry) {
  registry.addOpInterface<arith::ConstantOp, arith_ext::ConstantOpInterface>();
}
