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
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "scf-rotate-while"

namespace mlir {
#define GEN_PASS_DEF_SCFROTATEWHILELOOPPASS
#include "mlir/Dialect/SCF/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
struct RotateWhileLoopPattern : OpRewritePattern<scf::WhileOp> {
  RotateWhileLoopPattern(bool rotateLoop, MLIRContext *context,
                         PatternBenefit benefit = 1,
                         ArrayRef<StringRef> generatedNames = {})
      : OpRewritePattern<scf::WhileOp>(context, benefit, generatedNames),
        forceCreateCheck(rotateLoop) {}

  LogicalResult matchAndRewrite(scf::WhileOp whileOp,
                                PatternRewriter &rewriter) const final {
    FailureOr<scf::WhileOp> result =
        scf::wrapWhileLoopInZeroTripCheck(whileOp, rewriter, forceCreateCheck);
    if (failed(result) || *result == whileOp) {
      LLVM_DEBUG(whileOp->emitRemark("Failed to rotate loop"));
      return failure();
    };
    return success();
  }

  bool forceCreateCheck;
};

struct SCFRotateWhileLoopPass
    : impl::SCFRotateWhileLoopPassBase<SCFRotateWhileLoopPass> {
  using Base::Base;

  void runOnOperation() final {
    Operation *parentOp = getOperation();
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    SCFRotateWhileLoopPassOptions options{forceCreateCheck};
    scf::populateSCFRotateWhileLoopPatterns(patterns, options);
    // Avoid applying the pattern to a loop more than once.
    GreedyRewriteConfig config;
    config.strictMode = GreedyRewriteStrictness::ExistingOps;
    [[maybe_unused]] LogicalResult success =
        applyPatternsAndFoldGreedily(parentOp, std::move(patterns), config);
  }
};
} // namespace

namespace mlir {
namespace scf {
void populateSCFRotateWhileLoopPatterns(
    RewritePatternSet &patterns, const SCFRotateWhileLoopPassOptions &options) {
  patterns.add<RotateWhileLoopPattern>(options.forceCreateCheck,
                                       patterns.getContext());
}
} // namespace scf
} // namespace mlir
