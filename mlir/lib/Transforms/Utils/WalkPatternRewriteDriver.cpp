//===- WalkPatternRewriteDriver.cpp - A fast walk-based rewriter ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements mlir::walkAndApplyPatterns.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/WalkPatternRewriteDriver.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Rewrite/PatternApplicator.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

#define DEBUG_TYPE "walk-rewriter"

namespace mlir {

namespace {
struct WalkAndApplyPatternsAction final
    : tracing::ActionImpl<WalkAndApplyPatternsAction> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(WalkAndApplyPatternsAction)
  using ActionImpl::ActionImpl;
  static constexpr StringLiteral tag = "walk-and-apply-patterns";
  void print(raw_ostream &os) const override { os << tag; }
};

#if MLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS
// Forwarding listener to guard against unsupported erasures of non-descendant
// ops/blocks. Because we use walk-based pattern application, erasing the
// op/block from the *next* iteration (e.g., a user of the visited op) is not
// valid. Note that this is only used with expensive pattern API checks.
struct ErasedOpsListener final : RewriterBase::ForwardingListener {
  using RewriterBase::ForwardingListener::ForwardingListener;

  void notifyOperationErased(Operation *op) override {
    checkErasure(op);
    ForwardingListener::notifyOperationErased(op);
  }

  void notifyBlockErased(Block *block) override {
    checkErasure(block->getParentOp());
    ForwardingListener::notifyBlockErased(block);
  }

  void checkErasure(Operation *op) const {
    Operation *ancestorOp = op;
    while (ancestorOp && ancestorOp != visitedOp)
      ancestorOp = ancestorOp->getParentOp();

    if (ancestorOp != visitedOp)
      llvm::report_fatal_error(
          "unsupported erasure in WalkPatternRewriter; "
          "erasure is only supported for matched ops and their descendants");
  }

  Operation *visitedOp = nullptr;
};
#endif // MLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS
} // namespace

void walkAndApplyPatterns(Operation *op,
                          const FrozenRewritePatternSet &patterns,
                          RewriterBase::Listener *listener) {
#if MLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS
  if (failed(verify(op)))
    llvm::report_fatal_error("walk pattern rewriter input IR failed to verify");
#endif // MLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS

  MLIRContext *ctx = op->getContext();
  PatternRewriter rewriter(ctx);
#if MLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS
  ErasedOpsListener erasedListener(listener);
  rewriter.setListener(&erasedListener);
#else
  rewriter.setListener(listener);
#endif // MLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS

  PatternApplicator applicator(patterns);
  applicator.applyDefaultCostModel();

  ctx->executeAction<WalkAndApplyPatternsAction>(
      [&] {
        for (Region &region : op->getRegions()) {
          region.walk([&](Operation *visitedOp) {
            LLVM_DEBUG(llvm::dbgs() << "Visiting op: "; visitedOp->print(
                llvm::dbgs(), OpPrintingFlags().skipRegions());
                       llvm::dbgs() << "\n";);
#if MLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS
            erasedListener.visitedOp = visitedOp;
#endif // MLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS
            if (succeeded(applicator.matchAndRewrite(visitedOp, rewriter))) {
              LLVM_DEBUG(llvm::dbgs() << "\tOp matched and rewritten\n";);
            }
          });
        }
      },
      {op});

#if MLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS
  if (failed(verify(op)))
    llvm::report_fatal_error(
        "walk pattern rewriter result IR failed to verify");
#endif // MLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS
}

} // namespace mlir
