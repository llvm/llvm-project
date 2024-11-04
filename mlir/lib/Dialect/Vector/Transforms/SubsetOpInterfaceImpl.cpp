//===- SubsetOpInterfaceImpl.cpp - Tensor subsets -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Vector/Transforms/SubsetOpInterfaceImpl.h"

#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Interfaces/SubsetOpInterface.h"

using namespace mlir;
using namespace mlir::vector;

namespace {

template <typename OpTy>
struct XferOpSubsetOpInterface
    : public SubsetOpInterface::ExternalModel<XferOpSubsetOpInterface<OpTy>,
                                              OpTy> {
  FailureOr<HyperrectangularSlice>
  getAccessedHyperrectangularSlice(Operation *op) const {
    auto xferOp = cast<OpTy>(op);
    Builder b(xferOp->getContext());
    SmallVector<OpFoldResult> offsets = llvm::map_to_vector(
        xferOp.getIndices(), [](Value v) -> OpFoldResult { return v; });
    SmallVector<OpFoldResult> sizes = llvm::map_to_vector(
        xferOp.getTransferChunkAccessed(),
        [&](int64_t sz) -> OpFoldResult { return b.getIndexAttr(sz); });
    return HyperrectangularSlice(offsets, sizes);
  }
};

struct TransferReadOpSubsetExtractionOpInterface
    : public SubsetExtractionOpInterface::ExternalModel<
          TransferReadOpSubsetExtractionOpInterface, vector::TransferReadOp> {
  OpOperand &getSourceOperand(Operation *op) const {
    return cast<vector::TransferReadOp>(op).getSourceMutable();
  }
};

struct TransferWriteOpSubsetInsertionOpInterface
    : public SubsetInsertionOpInterface::ExternalModel<
          TransferWriteOpSubsetInsertionOpInterface, vector::TransferWriteOp> {
  OpOperand &getSourceOperand(Operation *op) const {
    return cast<vector::TransferWriteOp>(op).getVectorMutable();
  }

  OpOperand &getDestinationOperand(Operation *op) const {
    return cast<vector::TransferWriteOp>(op).getSourceMutable();
  }

  Value buildSubsetExtraction(Operation *op, OpBuilder &builder,
                              Location loc) const {
    // TODO: Implement when needed.
    return Value();
  }

  SmallVector<Value>
  getValuesNeededToBuildSubsetExtraction(Operation *op) const {
    // TODO: Implement when needed.
    return {};
  }
};

} // namespace

void mlir::vector::registerSubsetOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, vector::VectorDialect *dialect) {
    TransferReadOp::attachInterface<XferOpSubsetOpInterface<TransferReadOp>>(
        *ctx);
    TransferReadOp::attachInterface<TransferReadOpSubsetExtractionOpInterface>(
        *ctx);
    TransferWriteOp::attachInterface<XferOpSubsetOpInterface<TransferWriteOp>>(
        *ctx);
    TransferWriteOp::attachInterface<TransferWriteOpSubsetInsertionOpInterface>(
        *ctx);
  });
}
