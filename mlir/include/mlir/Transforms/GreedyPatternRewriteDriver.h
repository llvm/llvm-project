//===- GreedyPatternRewriteDriver.h - Greedy Pattern Driver -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares methods for applying a set of patterns greedily, choosing
// the patterns with the highest local benefit, until a fixed point is reached.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRANSFORMS_GREEDYPATTERNREWRITEDRIVER_H_
#define MLIR_TRANSFORMS_GREEDYPATTERNREWRITEDRIVER_H_

#include "mlir/Rewrite/FrozenRewritePatternSet.h"

namespace mlir {

/// This enum controls which ops are put on the worklist during a greedy
/// pattern rewrite.
enum class GreedyRewriteStrictness {
  /// No restrictions wrt. which ops are processed.
  AnyOp,
  /// Only pre-existing and newly created ops are processed.
  ExistingAndNewOps,
  /// Only pre-existing ops are processed.
  ExistingOps
};

/// This class allows control over how the GreedyPatternRewriteDriver works.
class GreedyRewriteConfig {
public:
  /// This specifies the order of initial traversal that populates the rewriters
  /// worklist.  When set to true, it walks the operations top-down, which is
  /// generally more efficient in compile time.  When set to false, its initial
  /// traversal of the region tree is bottom up on each block, which may match
  /// larger patterns when given an ambiguous pattern set.
  ///
  /// Note: Only applicable when simplifying entire regions.
  bool useTopDownTraversal = false;

  /// Perform control flow optimizations to the region tree after applying all
  /// patterns.
  ///
  /// Note: Only applicable when simplifying entire regions.
  bool enableRegionSimplification = true;

  /// This specifies the maximum number of times the rewriter will iterate
  /// between applying patterns and simplifying regions. Use `kNoLimit` to
  /// disable this iteration limit.
  ///
  /// Note: Only applicable when simplifying entire regions.
  int64_t maxIterations = 10;

  /// This specifies the maximum number of rewrites within an iteration. Use
  /// `kNoLimit` to disable this limit.
  int64_t maxNumRewrites = kNoLimit;

  static constexpr int64_t kNoLimit = -1;

  /// Only ops within the scope are added to the worklist. If no scope is
  /// specified, the closest enclosing region around the initial list of ops
  /// is used as a scope.
  Region *scope = nullptr;

  /// Strict mode can restrict the ops that are added to the worklist during
  /// the rewrite.
  ///
  /// * GreedyRewriteStrictness::AnyOp: No ops are excluded.
  /// * GreedyRewriteStrictness::ExistingAndNewOps: Only pre-existing ops (that
  ///   were on the worklist at the very beginning) and newly created ops are
  ///   enqueued. All other ops are excluded.
  /// * GreedyRewriteStrictness::ExistingOps: Only pre-existing ops (that were
  ///   were on the worklist at the very beginning) enqueued. All other ops are
  ///   excluded.
  GreedyRewriteStrictness strictMode = GreedyRewriteStrictness::AnyOp;

  /// An optional listener that should be notified about IR modifications.
  RewriterBase::Listener *listener = nullptr;
};

//===----------------------------------------------------------------------===//
// applyPatternsGreedily
//===----------------------------------------------------------------------===//

/// Rewrite ops in the given region, which must be isolated from above, by
/// repeatedly applying the highest benefit patterns in a greedy work-list
/// driven manner.
///
/// This variant may stop after a predefined number of iterations, see the
/// alternative below to provide a specific number of iterations before stopping
/// in absence of convergence.
///
/// Return success if the iterative process converged and no more patterns can
/// be matched in the result operation regions. `changed` is set to true if the
/// IR was modified at all.
///
/// Note: This does not apply patterns to the top-level operation itself.
///       These methods also perform folding and simple dead-code elimination
///       before attempting to match any of the provided patterns.
///
/// You may configure several aspects of this with GreedyRewriteConfig.
LogicalResult
applyPatternsAndFoldGreedily(Region &region,
                             const FrozenRewritePatternSet &patterns,
                             GreedyRewriteConfig config = GreedyRewriteConfig(),
                             bool *changed = nullptr);

/// Rewrite ops in all regions of the given op, which must be isolated from
/// above.
inline LogicalResult
applyPatternsAndFoldGreedily(Operation *op,
                             const FrozenRewritePatternSet &patterns,
                             GreedyRewriteConfig config = GreedyRewriteConfig(),
                             bool *changed = nullptr) {
  bool failed = false;
  for (Region &region : op->getRegions()) {
    bool regionChanged;
    failed |=
        applyPatternsAndFoldGreedily(region, patterns, config, &regionChanged)
            .failed();
    if (changed)
      *changed |= regionChanged;
  }
  return failure(failed);
}

/// Applies the specified rewrite patterns on `ops` while also trying to fold
/// these ops.
///
/// Newly created ops and other pre-existing ops that use results of rewritten
/// ops or supply operands to such ops are simplified, unless such ops are
/// excluded via `config.strictMode`. Any other ops remain unmodified (i.e.,
/// regardless of `strictMode`).
///
/// In addition to strictness, a region scope can be specified. Only ops within
/// the scope are simplified. This is similar to `applyPatternsAndFoldGreedily`,
/// where only ops within the given regions are simplified. If no scope is
/// specified, it is assumed to be the first common enclosing region of the
/// given ops.
///
/// Note that ops in `ops` could be erased as result of folding, becoming dead,
/// or via pattern rewrites. If more far reaching simplification is desired,
/// applyPatternsAndFoldGreedily should be used.
///
/// Returns success if the iterative process converged and no more patterns can
/// be matched. `changed` is set to true if the IR was modified at all.
/// `allOpsErased` is set to true if all ops in `ops` were erased.
LogicalResult
applyOpPatternsAndFold(ArrayRef<Operation *> ops,
                       const FrozenRewritePatternSet &patterns,
                       GreedyRewriteConfig config = GreedyRewriteConfig(),
                       bool *changed = nullptr, bool *allErased = nullptr);

} // namespace mlir

#endif // MLIR_TRANSFORMS_GREEDYPATTERNREWRITEDRIVER_H_
