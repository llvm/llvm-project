//===- BufferizationInterfaceImpl.cpp - Bufferization Impl. of Interface --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/ComprehensiveBufferize/BufferizationInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/BufferizableOpInterface.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"

using namespace mlir;
using namespace linalg;
using namespace comprehensive_bufferize;

namespace mlir {
namespace linalg {
namespace comprehensive_bufferize {
namespace bufferization_ext {

// TODO: These ops should implement BufferizableOpInterface directly when moved
// to the Bufferization dialect.

// TODO: These implementations are conservative and will likely have to be
// loosened for partial bufferization.

/// ToMemrefOp casts a tensor into a memref. The resulting memref is the memory
/// location of the incoming tensor once it will be bufferized. In the anlysis,
/// the incoming tensor is assumed to bufferize to a memory read and to an
/// inplace memory write, since it is unknown what will happen to the resulting
/// memref.
struct ToMemrefOpInterface
    : public BufferizableOpInterface::ExternalModel<ToMemrefOpInterface,
                                                    bufferization::ToMemrefOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand) const {
    // It is unknown whether the resulting MemRef will be read or not.
    return true;
  }

  OpResult getAliasingOpResult(Operation *op, OpOperand &opOperand) const {
    return OpResult();
  }

  LogicalResult bufferize(Operation *op, OpBuilder &b,
                          BufferizationState &state) const {
    return success();
  }
};

/// ToTensorOp conceptually loads a tensor from a memory location. Such ops do
/// not lower any further, and they should have disappeared by the time the
/// input is fully bufferized.
///
/// The analysis has no information about the memref that is loaded from by the
/// ToTensorOp. We have to assume that the loaded tensor may after bufferization
/// potentially alias with any other bufferized tensor. Since ToTensorOp and
/// ToMemrefOp have no aliasing OpOperand/OpResult pairs, this cannot be encoded
/// directly in the analysis. However, declaring ToTensorOp results as not
/// writable also enforces a buffer copy and has the same effect.
struct ToTensorOpInterface
    : public BufferizableOpInterface::ExternalModel<ToTensorOpInterface,
                                                    bufferization::ToTensorOp> {
  LogicalResult bufferize(Operation *op, OpBuilder &b,
                          BufferizationState &state) const {
    auto tensorLoadOp = cast<bufferization::ToTensorOp>(op);
    state.mapBuffer(tensorLoadOp.result(), tensorLoadOp.memref());
    return success();
  }

  bool isWritable(Operation *op, Value value, BufferizationState &state) const {
    // It is unknown whether the MemRef operand is writable or not.
    return false;
  }
};

} // namespace bufferization_ext
} // namespace comprehensive_bufferize
} // namespace linalg
} // namespace mlir

void mlir::linalg::comprehensive_bufferize::bufferization_ext::
    registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry) {
  registry.addOpInterface<bufferization::ToMemrefOp,
                          bufferization_ext::ToMemrefOpInterface>();
  registry.addOpInterface<bufferization::ToTensorOp,
                          bufferization_ext::ToTensorOpInterface>();
}
