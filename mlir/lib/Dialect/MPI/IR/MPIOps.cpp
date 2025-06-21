//===- MPIOps.cpp - MPI dialect ops implementation ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/MPI/IR/MPI.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::mpi;

namespace {

// If input memref has dynamic shape and is a cast and if the cast's input has
// static shape, fold the cast's static input into the given operation.
template <typename OpT>
struct FoldCast final : public mlir::OpRewritePattern<OpT> {
  using mlir::OpRewritePattern<OpT>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpT op,
                                mlir::PatternRewriter &b) const override {
    auto mRef = op.getRef();
    if (mRef.getType().hasStaticShape()) {
      return mlir::failure();
    }
    auto defOp = mRef.getDefiningOp();
    if (!defOp || !mlir::isa<mlir::memref::CastOp>(defOp)) {
      return mlir::failure();
    }
    auto src = mlir::cast<mlir::memref::CastOp>(defOp).getSource();
    if (!src.getType().hasStaticShape()) {
      return mlir::failure();
    }
    op.getRefMutable().assign(src);
    return mlir::success();
  }
};

struct FoldRank final : public mlir::OpRewritePattern<mlir::mpi::CommRankOp> {
  using mlir::OpRewritePattern<mlir::mpi::CommRankOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::mpi::CommRankOp op,
                                mlir::PatternRewriter &b) const override {
    auto comm = op.getComm();
    if (!comm.getDefiningOp<mlir::mpi::CommWorldOp>())
      return mlir::failure();

    // Try to get DLTI attribute for MPI:comm_world_rank
    // If found, set worldRank to the value of the attribute.
    auto dltiAttr = dlti::query(op, {"MPI:comm_world_rank"}, false);
    if (failed(dltiAttr))
      return mlir::failure();
    if (!isa<IntegerAttr>(dltiAttr.value()))
      return op->emitError()
             << "Expected an integer attribute for MPI:comm_world_rank";
    Value res = b.create<arith::ConstantIndexOp>(
        op.getLoc(), cast<IntegerAttr>(dltiAttr.value()).getInt());
    if (Value retVal = op.getRetval())
      b.replaceOp(op, {retVal, res});
    else
      b.replaceOp(op, res);
    return mlir::success();
  }
};

} // namespace

void mlir::mpi::SendOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &results, mlir::MLIRContext *context) {
  results.add<FoldCast<mlir::mpi::SendOp>>(context);
}

void mlir::mpi::RecvOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &results, mlir::MLIRContext *context) {
  results.add<FoldCast<mlir::mpi::RecvOp>>(context);
}

void mlir::mpi::ISendOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &results, mlir::MLIRContext *context) {
  results.add<FoldCast<mlir::mpi::ISendOp>>(context);
}

void mlir::mpi::IRecvOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &results, mlir::MLIRContext *context) {
  results.add<FoldCast<mlir::mpi::IRecvOp>>(context);
}

void mlir::mpi::CommRankOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &results, mlir::MLIRContext *context) {
  results.add<FoldRank>(context);
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/MPI/IR/MPIOps.cpp.inc"
