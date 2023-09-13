//===- SubsetInsertionOpInterfaceImpl.cpp - Tensor subsets ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tensor/Transforms/SubsetInsertionOpInterfaceImpl.h"

#include "mlir/Dialect/Bufferization/IR/SubsetInsertionOpInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

using namespace mlir;
using namespace mlir::bufferization;
using namespace mlir::tensor;

namespace {

/// Return "true" if `insertSliceOp` inserts into a subset that is equivalent
/// to the subset defined by `candidate`. `equivalenceFn` is used to determine
/// equivalence of tensors.
template <typename OpTy>
bool isSubsetEquivalentToInsertSliceLikeOp(
    OpTy insertSliceOp, Value candidate,
    function_ref<bool(Value, Value)> equivalenceFn) {
  // Look for a matching tensor.extract_slice op.
  auto extractSliceOp = candidate.getDefiningOp<tensor::ExtractSliceOp>();
  if (!extractSliceOp)
    return false;
  if (!equivalenceFn(extractSliceOp.getSource(), insertSliceOp.getDest()))
    return false;
  return sameOffsetsSizesAndStrides(extractSliceOp, insertSliceOp,
                                    isEqualConstantIntOrValue);
}

struct InsertSliceOpInterface
    : public SubsetInsertionOpInterface::ExternalModel<InsertSliceOpInterface,
                                                       tensor::InsertSliceOp> {
  OpOperand &getSourceOperand(Operation *op) const {
    return op->getOpOperand(0);
  }

  bool
  isEquivalentSubset(Operation *op, Value candidate,
                     function_ref<bool(Value, Value)> equivalenceFn) const {
    auto insertSliceOp = cast<tensor::InsertSliceOp>(op);
    return isSubsetEquivalentToInsertSliceLikeOp(insertSliceOp, candidate,
                                                 equivalenceFn);
  }
};

struct ParallelInsertSliceOpInterface
    : public SubsetInsertionOpInterface::ExternalModel<
          ParallelInsertSliceOpInterface, tensor::ParallelInsertSliceOp> {
  OpOperand &getSourceOperand(Operation *op) const {
    return op->getOpOperand(0);
  }

  OpOperand &getDestinationOperand(Operation *op) const {
    return op->getOpOperand(1);
  }

  bool
  isEquivalentSubset(Operation *op, Value candidate,
                     function_ref<bool(Value, Value)> equivalenceFn) const {
    auto insertSliceOp = cast<tensor::ParallelInsertSliceOp>(op);
    return isSubsetEquivalentToInsertSliceLikeOp(insertSliceOp, candidate,
                                                 equivalenceFn);
  }
};

} // namespace

void mlir::tensor::registerSubsetInsertionOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, tensor::TensorDialect *dialect) {
    InsertSliceOp::attachInterface<InsertSliceOpInterface>(*ctx);
    ParallelInsertSliceOp::attachInterface<ParallelInsertSliceOpInterface>(
        *ctx);
  });
}
