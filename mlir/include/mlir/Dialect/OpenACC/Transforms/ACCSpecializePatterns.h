//===- ACCSpecializePatterns.h - Common ACC Specialization Patterns ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains common rewrite pattern templates used by both
// ACCSpecializeForHost and ACCSpecializeForDevice passes.
//
// The patterns provide the following transformations:
//
// - ACCOpReplaceWithVarConversion<OpTy>: Replaces a data entry operation
//   with its var operand. Used for ops like acc.copyin, acc.create, etc.
//
// - ACCOpEraseConversion<OpTy>: Simply erases an operation. Used for
//   data exit ops like acc.copyout, acc.delete, and runtime ops.
//
// - ACCRegionUnwrapConversion<OpTy>: Inlines the region of an operation
//   and erases the wrapper. Used for structured data constructs
//   (acc.data, acc.host_data) and compute constructs (acc.parallel, etc.)
//
// - ACCDeclareEnterOpConversion: Erases acc.declare_enter and its
//   associated acc.declare_exit operation.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_OPENACC_TRANSFORMS_ACCSPECIALIZEPATTERNS_H
#define MLIR_DIALECT_OPENACC_TRANSFORMS_ACCSPECIALIZEPATTERNS_H

#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// Generic pattern templates for ACC specialization
//===----------------------------------------------------------------------===//

/// Pattern to replace an ACC op with its var operand.
/// Used for data entry ops like acc.copyin, acc.create, acc.attach, etc.
template <typename OpTy>
class ACCOpReplaceWithVarConversion : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

public:
  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    // Replace this op with its var operand; it's possible the op has no uses
    // if the op that had previously used it was already converted.
    if (op->use_empty())
      rewriter.eraseOp(op);
    else
      rewriter.replaceOp(op, op.getVar());
    return success();
  }
};

/// Pattern to simply erase an ACC op (for ops with no results).
/// Used for data exit ops like acc.copyout, acc.delete, acc.detach, etc.
template <typename OpTy>
class ACCOpEraseConversion : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

public:
  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    assert(op->getNumResults() == 0 && "expected op with no results");
    rewriter.eraseOp(op);
    return success();
  }
};

/// Pattern to unwrap a region from an ACC op and erase the wrapper.
/// Moves the region's contents to the parent block and removes the wrapper op.
/// Used for structured data constructs (acc.data, acc.host_data,
/// acc.kernel_environment, acc.declare) and compute constructs (acc.parallel,
/// acc.serial, acc.kernels).
template <typename OpTy>
class ACCRegionUnwrapConversion : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

public:
  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    assert(op.getRegion().hasOneBlock() && "expected one block");
    Block *block = &op.getRegion().front();
    // Erase the terminator (acc.yield or acc.terminator) before unwrapping
    rewriter.eraseOp(block->getTerminator());
    rewriter.inlineBlockBefore(block, op);
    rewriter.eraseOp(op);
    return success();
  }
};

/// Pattern to erase acc.declare_enter and its associated acc.declare_exit.
/// The declare_enter produces a token that is consumed by declare_exit.
class ACCDeclareEnterOpConversion
    : public OpRewritePattern<acc::DeclareEnterOp> {
  using OpRewritePattern<acc::DeclareEnterOp>::OpRewritePattern;

public:
  LogicalResult matchAndRewrite(acc::DeclareEnterOp op,
                                PatternRewriter &rewriter) const override {
    // If the enter token is used by an exit, erase exit first.
    if (!op->use_empty()) {
      assert(op->hasOneUse() && "expected one use");
      auto exitOp = dyn_cast<acc::DeclareExitOp>(*op->getUsers().begin());
      assert(exitOp && "expected declare exit op");
      rewriter.eraseOp(exitOp);
    }
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace acc
} // namespace mlir

#endif // MLIR_DIALECT_OPENACC_TRANSFORMS_ACCSPECIALIZEPATTERNS_H

