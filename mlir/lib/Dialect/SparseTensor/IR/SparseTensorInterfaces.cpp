//===- SparseTensorInterfaces.cpp - SparseTensor interfaces impl ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SparseTensor/IR/SparseTensorInterfaces.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensorType.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::sparse_tensor;

#include "mlir/Dialect/SparseTensor/IR/SparseTensorInterfaces.cpp.inc"

LogicalResult
sparse_tensor::detail::stageWithSortImpl(StageWithSortSparseOp op,
                                         PatternRewriter &rewriter) {
  // TODO: Implement it as an Interface, this can be reused from other
  // operations too (e.g., concatenate, reshape, etc).
  if (!op.needExtraSort())
    return failure();

  Location loc = op.getLoc();
  Type finalTp = op->getOpResult(0).getType();
  SparseTensorType dstStt(finalTp.cast<RankedTensorType>());

  Type srcCOOTp = getCOOFromTypeWithOrdering(
      dstStt.getRankedTensorType(), dstStt.getDimToLvl(), /*ordered=*/false);

  // Clones the original operation but changing the output to an unordered COO.
  Operation *cloned = rewriter.clone(*op.getOperation());
  rewriter.updateRootInPlace(cloned, [cloned, srcCOOTp]() {
    cloned->getOpResult(0).setType(srcCOOTp);
  });
  Value srcCOO = cloned->getOpResult(0);

  // -> sort
  Type dstCOOTp = getCOOFromTypeWithOrdering(
      dstStt.getRankedTensorType(), dstStt.getDimToLvl(), /*ordered=*/true);
  Value dstCOO = rewriter.create<ReorderCOOOp>(
      loc, dstCOOTp, srcCOO, SparseTensorSortKind::HybridQuickSort);

  // -> dest.
  if (dstCOO.getType() == finalTp) {
    rewriter.replaceOp(op, dstCOO);
  } else {
    // Need an extra conversion if the target type is not COO.
    rewriter.replaceOpWithNewOp<ConvertOp>(op, finalTp, dstCOO);
  }
  // TODO: deallocate extra COOs, we should probably delegate it to buffer
  // deallocation pass.
  return success();
}
