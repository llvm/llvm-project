//===- SubsetInsertionOpInterfaceImpl.cpp - Tensor subsets ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/Transforms/SubsetInsertionOpInterfaceImpl.h"

#include "mlir/Dialect/Bufferization/IR/SubsetInsertionOpInterface.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

using namespace mlir;
using namespace mlir::bufferization;
using namespace mlir::linalg;

namespace {
struct LinalgCopyOpInterface
    : public SubsetInsertionOpInterface::ExternalModel<LinalgCopyOpInterface,
                                                       linalg::CopyOp> {
  OpOperand &getSourceOperand(Operation *op) const {
    auto copyOp = cast<CopyOp>(op);
    assert(copyOp.getInputs().size() == 1 && "expected single input");
    return copyOp.getInputsMutable()[0];
  }

  bool
  isEquivalentSubset(Operation *op, Value candidate,
                     function_ref<bool(Value, Value)> equivalenceFn) const {
    auto copyOp = cast<CopyOp>(op);
    assert(copyOp.getOutputs().size() == 1 && "expected single output");
    return equivalenceFn(candidate, copyOp.getOutputs()[0]);
  }

  Value buildSubsetExtraction(Operation *op, OpBuilder &builder,
                              Location loc) const {
    auto copyOp = cast<CopyOp>(op);
    assert(copyOp.getOutputs().size() == 1 && "expected single output");
    return copyOp.getOutputs()[0];
  }

  SmallVector<Value>
  getValuesNeededToBuildSubsetExtraction(Operation *op) const {
    auto copyOp = cast<CopyOp>(op);
    assert(copyOp.getOutputs().size() == 1 && "expected single output");
    return {copyOp.getOutputs()[0]};
  }
};
} // namespace

void mlir::linalg::registerSubsetInsertionOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, linalg::LinalgDialect *dialect) {
    linalg::CopyOp::attachInterface<LinalgCopyOpInterface>(*ctx);
  });
}
