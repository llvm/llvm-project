//===- MPIOps.cpp - MPI dialect ops implementation ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

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
} // namespace

void mlir::mpi::SendOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &results, mlir::MLIRContext *context) {
  results.add<FoldCast<mlir::mpi::SendOp>>(context);
}

void mlir::mpi::RecvOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &results, mlir::MLIRContext *context) {
  results.add<FoldCast<mlir::mpi::RecvOp>>(context);
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/MPI/IR/MPIOps.cpp.inc"
