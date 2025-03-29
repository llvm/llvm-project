//===- BufferViewFlowOpInterfaceImpl.cpp - Buffer View Flow Analysis ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/Transforms/BufferViewFlowOpInterfaceImpl.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/BufferViewFlowOpInterface.h"

using namespace mlir;
using namespace mlir::bufferization;

namespace mlir {
namespace arith {
namespace {

struct SelectOpInterface
    : public BufferViewFlowOpInterface::ExternalModel<SelectOpInterface,
                                                      SelectOp> {
  void
  populateDependencies(Operation *op,
                       RegisterDependenciesFn registerDependenciesFn) const {
    auto selectOp = cast<SelectOp>(op);

    // Either one of the true/false value may be selected at runtime.
    registerDependenciesFn(selectOp.getTrueValue(), selectOp.getResult());
    registerDependenciesFn(selectOp.getFalseValue(), selectOp.getResult());
  }
};

} // namespace
} // namespace arith
} // namespace mlir

void arith::registerBufferViewFlowOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, arith::ArithDialect *dialect) {
    SelectOp::attachInterface<SelectOpInterface>(*ctx);
  });
}
