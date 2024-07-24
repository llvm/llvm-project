//===- RotateWhileLoop.cpp - scf.while loop rotation ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Rotates `scf.while` loops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/Transforms/Passes.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/IR/Diagnostics.h"

namespace mlir {
#define GEN_PASS_DEF_SCFROTATEWHILELOOPPASS
#include "mlir/Dialect/SCF/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
struct RotateWhileLoopPattern : OpRewritePattern<scf::WhileOp> {
  using OpRewritePattern<scf::WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::WhileOp whileOp,
                                PatternRewriter &rewriter) const final {
    // Setting this option would lead to infinite recursion on a greedy driver
    // as 'do-while' loops wouldn't be skipped.
    constexpr bool forceCreateCheck = false;
    FailureOr<scf::WhileOp> result =
        scf::wrapWhileLoopInZeroTripCheck(whileOp, rewriter, forceCreateCheck);
    // scf::wrapWhileLoopInZeroTripCheck hasn't yet implemented a failure
    // mechanism. 'do-while' loops are simply returned unmodified. In order to
    // stop recursion, we check input and output operations differ.
    return success(succeeded(result) && *result != whileOp);
  }
};

/// We do not use the above pattern in this pass as we can simply walk over the
/// `scf.while` operations and run the function.
struct SCFRotateWhileLoopPass
    : impl::SCFRotateWhileLoopPassBase<SCFRotateWhileLoopPass> {
  using Base::Base;

  void runOnOperation() final {
    Operation *parentOp = getOperation();

    SmallVector<scf::WhileOp> workList;
    parentOp->walk(
        [&workList](scf::WhileOp whileOp) { workList.push_back(whileOp); });

    MLIRContext *context = &getContext();
    PatternRewriter rewriter(context);
    for (scf::WhileOp whileOp : workList)
      (void)scf::wrapWhileLoopInZeroTripCheck(whileOp, rewriter,
                                              forceCreateCheck);
  }
};
} // namespace

namespace mlir {
namespace scf {
void populateSCFRotateWhileLoopPatterns(RewritePatternSet &patterns) {
  patterns.add<RotateWhileLoopPattern>(patterns.getContext());
}
} // namespace scf
} // namespace mlir
