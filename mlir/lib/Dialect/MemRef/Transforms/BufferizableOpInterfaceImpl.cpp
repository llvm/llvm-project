//===- BufferizableOpInterfaceImpl.cpp - Impl. of BufferizableOpInterface -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MemRef/Transforms/BufferizableOpInterfaceImpl.h"

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"

using namespace mlir;
using namespace mlir::bufferization;

namespace {
/// Bufferization of memref.tensor_store. Replace with memref.copy.
struct TensorStoreOpInterface
    : public BufferizableOpInterface::ExternalModel<TensorStoreOpInterface,
                                                    memref::TensorStoreOp> {
  AliasingOpResultList getAliasingOpResults(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) const {
    return {};
  }

  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    assert(opOperand.getOperandNumber() == 0 && "expected src operand");
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    // The memref operand is written but not the tensor operand.
    assert(opOperand.getOperandNumber() == 0 && "expected src operand");
    return false;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto tensorStoreOp = cast<memref::TensorStoreOp>(op);
    auto srcBuffer = getBuffer(rewriter, tensorStoreOp.getTensor(), options);
    if (failed(srcBuffer))
      return failure();
    if (failed(options.createMemCpy(rewriter, op->getLoc(), *srcBuffer,
                                    tensorStoreOp.getMemref())))
      return failure();
    rewriter.eraseOp(tensorStoreOp);
    return success();
  }
};

} // namespace

void mlir::memref::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, MemRefDialect *dialect) {
    TensorStoreOp::attachInterface<TensorStoreOpInterface>(*ctx);
  });
}
