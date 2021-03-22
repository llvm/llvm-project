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

#include "mlir/Rewrite/FrozenRewritePatternList.h"

namespace mlir {

//===----------------------------------------------------------------------===//
// applyPatternsGreedily
//===----------------------------------------------------------------------===//

/// Rewrite the regions of the specified operation, which must be isolated from
/// above, by repeatedly applying the highest benefit patterns in a greedy
/// work-list driven manner.
/// This variant may stop after a predefined number of iterations, see the
/// alternative below to provide a specific number of iterations before stopping
/// in absence of convergence.
/// Return success if the iterative process converged and no more patterns can
/// be matched in the result operation regions.
/// Note: This does not apply patterns to the top-level operation itself.
///       These methods also perform folding and simple dead-code elimination
///       before attempting to match any of the provided patterns.
LogicalResult
applyPatternsAndFoldGreedily(Operation *op,
                             const FrozenRewritePatternList &patterns,
                             bool useTopDownTraversal = true);

/// Rewrite the regions of the specified operation, with a user-provided limit
/// on iterations to attempt before reaching convergence.
LogicalResult applyPatternsAndFoldGreedily(
    Operation *op, const FrozenRewritePatternList &patterns,
    unsigned maxIterations, bool useTopDownTraversal = true);

/// Rewrite the given regions, which must be isolated from above.
LogicalResult
applyPatternsAndFoldGreedily(MutableArrayRef<Region> regions,
                             const FrozenRewritePatternList &patterns,
                             bool useTopDownTraversal = true);

/// Rewrite the given regions, with a user-provided limit on iterations to
/// attempt before reaching convergence.
LogicalResult applyPatternsAndFoldGreedily(
    MutableArrayRef<Region> regions, const FrozenRewritePatternList &patterns,
    unsigned maxIterations, bool useTopDownTraversal = true);

/// Applies the specified patterns on `op` alone while also trying to fold it,
/// by selecting the highest benefits patterns in a greedy manner. Returns
/// success if no more patterns can be matched. `erased` is set to true if `op`
/// was folded away or erased as a result of becoming dead. Note: This does not
/// apply any patterns recursively to the regions of `op`.
LogicalResult applyOpPatternsAndFold(Operation *op,
                                     const FrozenRewritePatternList &patterns,
                                     bool *erased = nullptr);

} // end namespace mlir

#endif // MLIR_TRANSFORMS_GREEDYPATTERNREWRITEDRIVER_H_
