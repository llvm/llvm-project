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
#include "llvm/ADT/StringRef.h"
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

// Forwarding listener to guard against unsupported erasures. Because we use
// walk-based pattern application, erasing the op from the *next* iteration
// (e.g., a user of the visited op) is not valid.
// Note that this is only used with expensive pattern API checks.
struct ErasedOpsListener final : RewriterBase::ForwardingListener {
  using RewriterBase::ForwardingListener::ForwardingListener;

  void notifyOperationErased(Operation *op) override {
    if (op != visitedOp)
      llvm::report_fatal_error("unsupported op erased in WalkPatternRewriter; "
                               "erasure is only supported for matched ops");

    ForwardingListener::notifyOperationErased(op);
  }

  Operation *visitedOp = nullptr;
};
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
  ErasedOpsListener erasedListener(listener);
#if MLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS
  rewriter.setListener(&erasedListener);
#else
  (void)erasedListener;
  rewriter.setListener(listener);
#endif

  PatternApplicator applicator(patterns);
  applicator.applyDefaultCostModel();

  ctx->executeAction<WalkAndApplyPatternsAction>(
      [&] {
        for (Region &region : op->getRegions()) {
          region.walk([&](Operation *visitedOp) {
            LLVM_DEBUG(llvm::dbgs() << "Visiting op: "; visitedOp->print(
                llvm::dbgs(), OpPrintingFlags().skipRegions());
                       llvm::dbgs() << "\n";);
            erasedListener.visitedOp = visitedOp;
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
