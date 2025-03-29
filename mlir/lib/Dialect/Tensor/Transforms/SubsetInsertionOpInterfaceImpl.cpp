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

struct ExtractSliceOpSubsetOpInterface
    : public SubsetOpInterface::ExternalModel<ExtractSliceOpSubsetOpInterface,
                                              tensor::ExtractSliceOp> {
  FailureOr<HyperrectangularSlice>
  getAccessedHyperrectangularSlice(Operation *op) const {
    return HyperrectangularSlice(cast<OffsetSizeAndStrideOpInterface>(op));
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
  FailureOr<HyperrectangularSlice>
  getAccessedHyperrectangularSlice(Operation *op) const {
    return HyperrectangularSlice(cast<OffsetSizeAndStrideOpInterface>(op));
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
