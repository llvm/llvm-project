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

template <typename OpTy>
struct InsertSliceLikeOpInterface
    : public SubsetInsertionOpInterface::ExternalModel<
          InsertSliceLikeOpInterface<OpTy>, OpTy> {
  OpOperand &getSourceOperand(Operation *op) const {
    return cast<OpTy>(op).getSourceMutable();
  }

  OpOperand &getDestinationOperand(Operation *op) const {
    return cast<OpTy>(op).getDestMutable();
  }

  /// Return "true" if `insertSliceOp` inserts into a subset that is equivalent
  /// to the subset defined by `candidate`. `equivalenceFn` is used to determine
  /// equivalence of tensors.
  bool
  isEquivalentSubset(Operation *op, Value candidate,
                     function_ref<bool(Value, Value)> equivalenceFn) const {
    auto insertSliceOp = cast<OpTy>(op);
    // Look for a matching tensor.extract_slice op.
    auto extractSliceOp = candidate.getDefiningOp<tensor::ExtractSliceOp>();
    if (!extractSliceOp)
      return false;
    if (!equivalenceFn(extractSliceOp.getSource(), insertSliceOp.getDest()))
      return false;
    return sameOffsetsSizesAndStrides(extractSliceOp, insertSliceOp,
                                      isEqualConstantIntOrValue);
  }

  Value buildSubsetExtraction(Operation *op, OpBuilder &builder,
                              Location loc) const {
    auto insertSliceOp = cast<OpTy>(op);
    auto extractOp = builder.create<tensor::ExtractSliceOp>(
        loc, insertSliceOp.getSourceType(), insertSliceOp.getDest(),
        insertSliceOp.getMixedOffsets(), insertSliceOp.getMixedSizes(),
        insertSliceOp.getMixedStrides());
    return extractOp.getResult();
  }

  SmallVector<Value>
  getValuesNeededToBuildSubsetExtraction(Operation *op) const {
    auto insertSliceOp = cast<OpTy>(op);
    SmallVector<Value> neededValues;
    // Collect all values that are needed to construct the replacement op.
    neededValues.append(insertSliceOp.getOffsets().begin(),
                        insertSliceOp.getOffsets().end());
    neededValues.append(insertSliceOp.getSizes().begin(),
                        insertSliceOp.getSizes().end());
    neededValues.append(insertSliceOp.getStrides().begin(),
                        insertSliceOp.getStrides().end());
    neededValues.push_back(insertSliceOp.getDest());
    return neededValues;
  }
};

} // namespace

void mlir::tensor::registerSubsetInsertionOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, tensor::TensorDialect *dialect) {
    InsertSliceOp::attachInterface<InsertSliceLikeOpInterface<InsertSliceOp>>(
        *ctx);
    ParallelInsertSliceOp::attachInterface<
        InsertSliceLikeOpInterface<ParallelInsertSliceOp>>(*ctx);
  });
}
