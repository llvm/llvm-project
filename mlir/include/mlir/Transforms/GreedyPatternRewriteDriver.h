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

/// This class allows control over how the GreedyPatternRewriteDriver works.
class GreedyRewriteConfig {
public:
  /// This specifies the order of initial traversal that populates the rewriters
  /// worklist.  When set to true, it walks the operations top-down, which is
  /// generally more efficient in compile time.  When set to false, its initial
  /// traversal of the region tree is bottom up on each block, which may match
  /// larger patterns when given an ambiguous pattern set.
  bool useTopDownTraversal = false;

  // Perform control flow optimizations to the region tree after applying all
  // patterns.
  bool enableRegionSimplification = true;

  /// This specifies the maximum number of times the rewriter will iterate
  /// between applying patterns and simplifying regions. Use `kNoIterationLimit`
  /// to disable this iteration limit.
  int64_t maxIterations = 10;

  static constexpr int64_t kNoIterationLimit = -1;
};

//===----------------------------------------------------------------------===//
// applyPatternsGreedily
//===----------------------------------------------------------------------===//

/// Rewrite the regions of the specified operation, which must be isolated from
/// above, by repeatedly applying the highest benefit patterns in a greedy
/// work-list driven manner.
///
/// This variant may stop after a predefined number of iterations, see the
/// alternative below to provide a specific number of iterations before stopping
/// in absence of convergence.
///
/// Return success if the iterative process converged and no more patterns can
/// be matched in the result operation regions.
///
/// Note: This does not apply patterns to the top-level operation itself.
///       These methods also perform folding and simple dead-code elimination
///       before attempting to match any of the provided patterns.
///
/// You may configure several aspects of this with GreedyRewriteConfig.
LogicalResult applyPatternsAndFoldGreedily(
    MutableArrayRef<Region> regions, const FrozenRewritePatternSet &patterns,
    GreedyRewriteConfig config = GreedyRewriteConfig());

/// Rewrite the given regions, which must be isolated from above.
inline LogicalResult applyPatternsAndFoldGreedily(
    Operation *op, const FrozenRewritePatternSet &patterns,
    GreedyRewriteConfig config = GreedyRewriteConfig()) {
  return applyPatternsAndFoldGreedily(op->getRegions(), patterns, config);
}

/// Applies the specified patterns on `op` alone while also trying to fold it,
/// by selecting the highest benefits patterns in a greedy manner. Returns
/// success if no more patterns can be matched. `erased` is set to true if `op`
/// was folded away or erased as a result of becoming dead. Note: This does not
/// apply any patterns recursively to the regions of `op`.
LogicalResult applyOpPatternsAndFold(Operation *op,
                                     const FrozenRewritePatternSet &patterns,
                                     bool *erased = nullptr);

/// Applies the specified rewrite patterns on `ops` while also trying to fold
/// these ops as well as any other ops that were in turn created due to such
/// rewrites. Furthermore, any pre-existing ops in the IR outside of `ops`
/// remain completely unmodified if `strict` is set to true. If `strict` is
/// false, other operations that use results of rewritten ops or supply operands
/// to such ops are in turn simplified; any other ops still remain unmodified
/// (i.e., regardless of `strict`). Note that ops in `ops` could be erased as a
/// result of folding, becoming dead, or via pattern rewrites. If more far
/// reaching simplification is desired, applyPatternsAndFoldGreedily should be
/// used. Returns true if at all any IR was rewritten.
bool applyOpPatternsAndFold(ArrayRef<Operation *> ops,
                            const FrozenRewritePatternSet &patterns,
                            bool strict);

} // end namespace mlir

#endif // MLIR_TRANSFORMS_GREEDYPATTERNREWRITEDRIVER_H_
