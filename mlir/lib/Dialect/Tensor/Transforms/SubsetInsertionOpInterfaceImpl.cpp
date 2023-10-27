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

template <typename OpTy>
Value buildSubsetExtractionOfInsertSliceLikeOp(OpBuilder &b, Location loc,
                                               OpTy insertSliceOp) {
  auto extractOp = b.create<tensor::ExtractSliceOp>(
      loc, insertSliceOp.getSourceType(), insertSliceOp.getDest(),
      insertSliceOp.getMixedOffsets(), insertSliceOp.getMixedSizes(),
      insertSliceOp.getMixedStrides());
  return extractOp.getResult();
}

template <typename OpTy>
SmallVector<Value>
getValuesNeededToBuildSubsetExtractionOfInsertSliceLikeOp(OpTy insertSliceOp) {
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

struct InsertSliceOpInterface
    : public SubsetInsertionOpInterface::ExternalModel<InsertSliceOpInterface,
                                                       tensor::InsertSliceOp> {
  OpOperand &getSourceOperand(Operation *op) const {
    return cast<tensor::InsertSliceOp>(op).getSourceMutable();
  }

  bool
  isEquivalentSubset(Operation *op, Value candidate,
                     function_ref<bool(Value, Value)> equivalenceFn) const {
    auto insertSliceOp = cast<tensor::InsertSliceOp>(op);
    return isSubsetEquivalentToInsertSliceLikeOp(insertSliceOp, candidate,
                                                 equivalenceFn);
  }

  Value buildSubsetExtraction(Operation *op, OpBuilder &builder,
                              Location loc) const {
    return buildSubsetExtractionOfInsertSliceLikeOp(
        builder, loc, cast<tensor::InsertSliceOp>(op));
  }

  SmallVector<Value>
  getValuesNeededToBuildSubsetExtraction(Operation *op) const {
    return getValuesNeededToBuildSubsetExtractionOfInsertSliceLikeOp(
        cast<tensor::InsertSliceOp>(op));
  }
};

struct ParallelInsertSliceOpInterface
    : public SubsetInsertionOpInterface::ExternalModel<
          ParallelInsertSliceOpInterface, tensor::ParallelInsertSliceOp> {
  OpOperand &getSourceOperand(Operation *op) const {
    return cast<tensor::ParallelInsertSliceOp>(op).getSourceMutable();
  }

  OpOperand &getDestinationOperand(Operation *op) const {
    return cast<tensor::ParallelInsertSliceOp>(op).getDestMutable();
  }

  bool
  isEquivalentSubset(Operation *op, Value candidate,
                     function_ref<bool(Value, Value)> equivalenceFn) const {
    auto insertSliceOp = cast<tensor::ParallelInsertSliceOp>(op);
    return isSubsetEquivalentToInsertSliceLikeOp(insertSliceOp, candidate,
                                                 equivalenceFn);
  }

  Value buildSubsetExtraction(Operation *op, OpBuilder &builder,
                              Location loc) const {
    return buildSubsetExtractionOfInsertSliceLikeOp(
        builder, loc, cast<tensor::ParallelInsertSliceOp>(op));
  }

  SmallVector<Value>
  getValuesNeededToBuildSubsetExtraction(Operation *op) const {
    return getValuesNeededToBuildSubsetExtractionOfInsertSliceLikeOp(
        cast<tensor::ParallelInsertSliceOp>(op));
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
