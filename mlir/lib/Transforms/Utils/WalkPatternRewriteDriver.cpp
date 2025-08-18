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
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Rewrite/PatternApplicator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/DebugLog.h"
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

  // Iterator on all reachable operations in the region.
  // Also keep track if we visited the nested regions of the current op
  // already to drive the post-order traversal.
  struct RegionReachableOpIterator {
    RegionReachableOpIterator(Region *region) : region(region) {
      regionIt = region->begin();
      if (regionIt != region->end())
        blockIt = regionIt->begin();
    }
    // Advance the iterator to the next reachable operation.
    void advance() {
      assert(regionIt != region->end());
      hasVisitedRegions = false;
      if (blockIt == regionIt->end()) {
        ++regionIt;
        if (regionIt != region->end())
          blockIt = regionIt->begin();
        return;
      }
      ++blockIt;
      if (blockIt != regionIt->end()) {
        LDBG() << "Incrementing block iterator, next op: "
               << OpWithFlags(&*blockIt, OpPrintingFlags().skipRegions());
      }
    }
    // The region we're iterating over.
    Region *region;
    // The Block currently being iterated over.
    Region::iterator regionIt;
    // The Operation currently being iterated over.
    Block::iterator blockIt;
    // Whether we've visited the nested regions of the current op already.
    bool hasVisitedRegions = false;
  };

  // Worklist of regions to visit to drive the post-order traversal.
  SmallVector<RegionReachableOpIterator> worklist;

  LDBG() << "Starting walk-based pattern rewrite driver";
  ctx->executeAction<WalkAndApplyPatternsAction>(
      [&] {
        // Perform a post-order traversal of the regions, visiting each
        // reachable operation.
        for (Region &region : op->getRegions()) {
          assert(worklist.empty());
          if (region.empty())
            continue;

          // Prime the worklist with the entry block of this region.
          worklist.push_back({&region});
          while (!worklist.empty()) {
            RegionReachableOpIterator &it = worklist.back();
            if (it.regionIt == it.region->end()) {
              // We're done with this region.
              worklist.pop_back();
              continue;
            }
            if (it.blockIt == it.regionIt->end()) {
              // We're done with this block.
              it.advance();
              continue;
            }
            Operation *op = &*it.blockIt;
            // If we haven't visited the nested regions of this op yet,
            // enqueue them.
            if (!it.hasVisitedRegions) {
              it.hasVisitedRegions = true;
              for (Region &nestedRegion : llvm::reverse(op->getRegions())) {
                if (nestedRegion.empty())
                  continue;
                worklist.push_back({&nestedRegion});
              }
            }
            // If we're not at the back of the worklist, we've enqueued some
            // nested region for processing. We'll come back to this op later
            // (post-order)
            if (&it != &worklist.back())
              continue;

            // Preemptively increment the iterator, in case the current op
            // would be erased.
            it.advance();

            LDBG() << "Visiting op: "
                   << OpWithFlags(op, OpPrintingFlags().skipRegions());
#if MLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS
            erasedListener.visitedOp = op;
#endif // MLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS
            if (succeeded(applicator.matchAndRewrite(op, rewriter)))
              LDBG() << "\tOp matched and rewritten";
          }
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
