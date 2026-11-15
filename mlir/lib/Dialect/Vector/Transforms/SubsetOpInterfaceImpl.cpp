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
// Check both the mask operand and an enclosing vector.mask.
bool isMaskedTransfer(VectorTransferOpInterface transfer) {
  return transfer.getMask() ||
         cast<MaskableOpInterface>(transfer.getOperation()).isMasked();
}

template <typename OpTy>
struct XferOpSubsetOpInterface
    : public SubsetOpInterface::ExternalModel<XferOpSubsetOpInterface<OpTy>,
                                              OpTy> {
  bool operatesOnEquivalentSubset(
      Operation *op, SubsetOpInterface candidate,
      function_ref<bool(Value, Value)> equivalenceFn) const {
    auto transfer = cast<VectorTransferOpInterface>(op);
    // A mask can restrict access to part of the rectangular transfer region.
    if (isMaskedTransfer(transfer))
      return false;
    auto other = dyn_cast<VectorTransferOpInterface>(candidate.getOperation());
    if (other && (isMaskedTransfer(other) ||
                  transfer.getVectorType() != other.getVectorType() ||
                  transfer.getPermutationMap() != other.getPermutationMap()))
      return false;
    return mlir::detail::defaultOperatesOnEquivalentSubset(op, candidate,
                                                           equivalenceFn);
  }

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
    return cast<vector::TransferReadOp>(op).getBaseMutable();
  }
};

struct TransferWriteOpSubsetInsertionOpInterface
    : public SubsetInsertionOpInterface::ExternalModel<
          TransferWriteOpSubsetInsertionOpInterface, vector::TransferWriteOp> {
  OpOperand &getSourceOperand(Operation *op) const {
    return cast<vector::TransferWriteOp>(op).getValueToStoreMutable();
  }

  OpOperand &getDestinationOperand(Operation *op) const {
    return cast<vector::TransferWriteOp>(op).getBaseMutable();
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
