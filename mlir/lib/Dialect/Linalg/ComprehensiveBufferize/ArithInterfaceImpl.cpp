//===- ArithInterfaceImpl.cpp - Arith Impl. of BufferizableOpInterface ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/ComprehensiveBufferize/ArithInterfaceImpl.h"

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/Transforms/BufferUtils.h"

using namespace mlir::bufferization;

namespace mlir {
namespace linalg {
namespace comprehensive_bufferize {
namespace arith_ext {

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
    GlobalCreator globalCreator(moduleOp);
    auto globalMemref = globalCreator.getGlobalFor(constantOp);
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

} // namespace arith_ext
} // namespace comprehensive_bufferize
} // namespace linalg
} // namespace mlir

void mlir::linalg::comprehensive_bufferize::arith_ext::
    registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry) {
  registry.addOpInterface<arith::ConstantOp, arith_ext::ConstantOpInterface>();
}
