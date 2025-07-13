//===- BufferViewFlowOpInterfaceImpl.cpp - Buffer View Flow Analysis ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MemRef/Transforms/BufferViewFlowOpInterfaceImpl.h"

#include "mlir/Dialect/Bufferization/IR/BufferViewFlowOpInterface.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

using namespace mlir;
using namespace mlir::bufferization;

namespace mlir {
namespace memref {
namespace {

struct ReallocOpInterface
    : public BufferViewFlowOpInterface::ExternalModel<ReallocOpInterface,
                                                      ReallocOp> {
  void
  populateDependencies(Operation *op,
                       RegisterDependenciesFn registerDependenciesFn) const {
    auto reallocOp = cast<ReallocOp>(op);
    // memref.realloc may return the source operand.
    registerDependenciesFn(reallocOp.getSource(), reallocOp.getResult());
  }

  bool mayBeTerminalBuffer(Operation *op, Value value) const {
    // The return value of memref.realloc is a terminal buffer because the op
    // may return a newly allocated buffer.
    return true;
  }
};

} // namespace
} // namespace memref
} // namespace mlir

void memref::registerBufferViewFlowOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, memref::MemRefDialect *dialect) {
    ReallocOp::attachInterface<ReallocOpInterface>(*ctx);
  });
}
