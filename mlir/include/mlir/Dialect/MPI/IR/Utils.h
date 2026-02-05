//===- Utils.h - MPI dialect --------------------------------------*- C++-*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_DIALECT_MPI_IR_UTILS_H_
#define MLIR_DIALECT_MPI_IR_UTILS_H_

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/MPI/IR/MPI.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace mpi {
template <typename OpT>
LogicalResult FoldToDLTIConst(OpT op, const char *key,
                              mlir::PatternRewriter &b) {
  auto comm = op.getComm();
  if (!comm.template getDefiningOp<mlir::mpi::CommWorldOp>())
    return mlir::failure();

  // Try to get DLTI attribute for MPI:comm_world_rank
  // If found, set worldRank to the value of the attribute.
  auto dltiAttr = dlti::query(op, {key}, false);
  if (failed(dltiAttr))
    return mlir::failure();
  if (!isa<IntegerAttr>(dltiAttr.value()))
    return op->emitError() << "Expected an integer attribute for " << key;
  Value res = arith::ConstantOp::create(
      b, op.getLoc(), b.getI32Type(),
      b.getI32IntegerAttr(cast<IntegerAttr>(dltiAttr.value()).getInt()));
  if (Value retVal = op.getRetval())
    b.replaceOp(op, {retVal, res});
  else
    b.replaceOp(op, res);
  return mlir::success();
}
} // namespace mpi
} // namespace mlir

#endif // MLIR_DIALECT_MPI_IR_UTILS_H_
