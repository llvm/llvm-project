//===- SubsetInsertionOpInterfaceImpl.cpp - Tensor subsets ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tensor/Transforms/SubsetInsertionOpInterfaceImpl.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/SubsetOpInterface.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"

using namespace mlir;
using namespace mlir::tensor;

namespace {

/// Return the tensor that the given subset op operates on.
Value getContainerOperand(SubsetOpInterface op) {
  if (auto extractionOp =
          dyn_cast<SubsetExtractionOpInterface>(op.getOperation()))
    return extractionOp.getSourceOperand().get();
  if (auto insertionOp =
          dyn_cast<SubsetInsertionOpInterface>(op.getOperation()))
    return insertionOp.getDestinationOperand().get();
  llvm_unreachable("expected SubsetExtraction/InsertionOpInterface");
}

/// Return "true" if the two ops operate on an equivalent subset.
/// `equivalenceFn` is used to determine equivalence of tensors. Return "false"
/// if the two ops operate non-equivalent subsets, if equivalence cannot be
/// determined or if `op1` is not a subset op.
template <typename OpTy>
bool operateOnEquivalentSubsets(
    OpTy op1, SubsetOpInterface op2,
    function_ref<bool(Value, Value)> equivalenceFn) {
  auto offsetsSizesAndStrides2 =
      dyn_cast<OffsetSizeAndStrideOpInterface>(op2.getOperation());
  if (!offsetsSizesAndStrides2)
    return false;
  if (!sameOffsetsSizesAndStrides(op1, offsetsSizesAndStrides2,
                                  isEqualConstantIntOrValue))
    return false;
  return equivalenceFn(
      getContainerOperand(cast<SubsetOpInterface>(op1.getOperation())),
      getContainerOperand(op2));
}

/// Return "true" if the two ops operate on a disjoint subsets.
/// `equivalenceFn` is used to determine equivalence of tensors. Return "false"
/// if the two ops operate non-disjoint subsets, if disjointness cannot be
/// determined or if `op1` is not a subset op.
template <typename OpTy>
bool operateOnDisjointSubsets(OpTy op1, SubsetOpInterface op2,
                              function_ref<bool(Value, Value)> equivalenceFn) {
  auto offsetsSizesAndStrides2 =
      dyn_cast<OffsetSizeAndStrideOpInterface>(op2.getOperation());
  if (!offsetsSizesAndStrides2)
    return false;
  FailureOr<bool> overlappingSlices =
      ValueBoundsConstraintSet::areOverlappingSlices(op1,
                                                     offsetsSizesAndStrides2);
  if (failed(overlappingSlices) || *overlappingSlices)
    return false;
  return equivalenceFn(
      getContainerOperand(cast<SubsetOpInterface>(op1.getOperation())),
      getContainerOperand(op2));
}

struct ExtractSliceOpSubsetOpInterface
    : public SubsetOpInterface::ExternalModel<ExtractSliceOpSubsetOpInterface,
                                              tensor::ExtractSliceOp> {
  bool operatesOnEquivalentSubset(
      Operation *op, SubsetOpInterface candidate,
      function_ref<bool(Value, Value)> equivalenceFn) const {
    auto extractSliceOp = cast<tensor::ExtractSliceOp>(op);
    return operateOnEquivalentSubsets(extractSliceOp, candidate, equivalenceFn);
  }

  bool operatesOnDisjointSubset(
      Operation *op, SubsetOpInterface candidate,
      function_ref<bool(Value, Value)> equivalenceFn) const {
    auto extractSliceOp = cast<tensor::ExtractSliceOp>(op);
    return operateOnDisjointSubsets(extractSliceOp, candidate, equivalenceFn);
  }
};

struct ExtractSliceOpSubsetExtractionOpInterface
    : public SubsetExtractionOpInterface::ExternalModel<
          ExtractSliceOpSubsetExtractionOpInterface, tensor::ExtractSliceOp> {
  OpOperand &getSourceOperand(Operation *op) const {
    return cast<tensor::ExtractSliceOp>(op).getSourceMutable();
  }
};

template <typename OpTy>
struct InsertSliceLikeOpSubsetOpInterface
    : public SubsetOpInterface::ExternalModel<
          InsertSliceLikeOpSubsetOpInterface<OpTy>, OpTy> {
  bool operatesOnEquivalentSubset(
      Operation *op, SubsetOpInterface candidate,
      function_ref<bool(Value, Value)> equivalenceFn) const {
    auto insertSliceOp = cast<OpTy>(op);
    return operateOnEquivalentSubsets(insertSliceOp, candidate, equivalenceFn);
  }

  bool operatesOnDisjointSubset(
      Operation *op, SubsetOpInterface candidate,
      function_ref<bool(Value, Value)> equivalenceFn) const {
    auto insertSliceOp = cast<OpTy>(op);
    return operateOnDisjointSubsets(insertSliceOp, candidate, equivalenceFn);
  }
};

template <typename OpTy>
struct InsertSliceLikeOpSubsetInsertionOpInterface
    : public SubsetInsertionOpInterface::ExternalModel<
          InsertSliceLikeOpSubsetInsertionOpInterface<OpTy>, OpTy> {
  OpOperand &getSourceOperand(Operation *op) const {
    return cast<OpTy>(op).getSourceMutable();
  }

  OpOperand &getDestinationOperand(Operation *op) const {
    return cast<OpTy>(op).getDestMutable();
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

void mlir::tensor::registerSubsetOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, tensor::TensorDialect *dialect) {
    // Note: `SubsetExtractionOpInterface` and `SubsetInsertionOpInterface`
    // require `SubsetOpInterface`.
    ExtractSliceOp::attachInterface<ExtractSliceOpSubsetOpInterface>(*ctx);
    ExtractSliceOp::attachInterface<ExtractSliceOpSubsetExtractionOpInterface>(
        *ctx);
    InsertSliceOp::attachInterface<
        InsertSliceLikeOpSubsetOpInterface<InsertSliceOp>>(*ctx);
    InsertSliceOp::attachInterface<
        InsertSliceLikeOpSubsetInsertionOpInterface<InsertSliceOp>>(*ctx);
    ParallelInsertSliceOp::attachInterface<
        InsertSliceLikeOpSubsetOpInterface<ParallelInsertSliceOp>>(*ctx);
    ParallelInsertSliceOp::attachInterface<
        InsertSliceLikeOpSubsetInsertionOpInterface<ParallelInsertSliceOp>>(
        *ctx);
  });
}
