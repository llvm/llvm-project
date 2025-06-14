//===- SliceMatchers.h - Matchers for slicing analysis ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides matchers for MLIRQuery that peform slicing analysis
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIRQUERY_MATCHERS_SLICEMATCHERS_H
#define MLIR_TOOLS_MLIRQUERY_MATCHERS_SLICEMATCHERS_H

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/IR/Operation.h"

/// A matcher encapsulating `getBackwardSlice` method from SliceAnalysis.h.
/// Additionally, it limits the slice computation to a certain depth level using
/// a custom filter.
///
/// Example: starting from node 9, assuming the matcher
/// computes the slice for the first two depth levels:
/// ============================
///    1       2      3      4
///    |_______|      |______|
///    |   |             |
///    |   5             6
///    |___|_____________|
///      |               |
///      7               8
///      |_______________|
///              |
///              9
///
/// Assuming all local orders match the numbering order:
///     {5, 7, 6, 8, 9}
namespace mlir::query::matcher {

template <typename Matcher>
class BackwardSliceMatcher {
public:
  BackwardSliceMatcher(Matcher innerMatcher, int64_t maxDepth, bool inclusive,
                       bool omitBlockArguments, bool omitUsesFromAbove)
      : innerMatcher(std::move(innerMatcher)), maxDepth(maxDepth),
        inclusive(inclusive), omitBlockArguments(omitBlockArguments),
        omitUsesFromAbove(omitUsesFromAbove) {}

  bool match(Operation *rootOp, SetVector<Operation *> &backwardSlice) {
    BackwardSliceOptions options;
    options.inclusive = inclusive;
    options.omitUsesFromAbove = omitUsesFromAbove;
    options.omitBlockArguments = omitBlockArguments;
    return (innerMatcher.match(rootOp) &&
            matches(rootOp, backwardSlice, options, maxDepth));
  }

private:
  bool matches(Operation *rootOp, llvm::SetVector<Operation *> &backwardSlice,
               BackwardSliceOptions &options, int64_t maxDepth);

private:
  // The outer matcher (e.g., BackwardSliceMatcher) relies on the innerMatcher
  // to determine whether we want to traverse the IR or not. For example, we
  // want to explore the IR only if the top-level operation name is
  // `"arith.addf"`.
  Matcher innerMatcher;
  // `maxDepth` specifies the maximum depth that the matcher can traverse the
  // IR. For example, if `maxDepth` is 2, the matcher will explore the defining
  // operations of the top-level op up to 2 levels.
  int64_t maxDepth;
  bool inclusive;
  bool omitBlockArguments;
  bool omitUsesFromAbove;
};

template <typename Matcher>
bool BackwardSliceMatcher<Matcher>::matches(
    Operation *rootOp, llvm::SetVector<Operation *> &backwardSlice,
    BackwardSliceOptions &options, int64_t maxDepth) {
  backwardSlice.clear();
  llvm::DenseMap<Operation *, int64_t> opDepths;
  // Initializing the root op with a depth of 0
  opDepths[rootOp] = 0;
  options.filter = [&](Operation *subOp) {
    // If the subOp hasn't been recorded in opDepths, it is deeper than
    // maxDepth.
    if (!opDepths.contains(subOp))
      return false;
    // Examine subOp's operands to compute depths of their defining operations.
    for (auto operand : subOp->getOperands()) {
      int64_t newDepth = opDepths[subOp] + 1;
      // If the newDepth is greater than maxDepth, further computation can be
      // skipped.
      if (newDepth > maxDepth)
        continue;

      if (auto definingOp = operand.getDefiningOp()) {
        // Registers the minimum depth
        if (!opDepths.contains(definingOp) || newDepth < opDepths[definingOp])
          opDepths[definingOp] = newDepth;
      } else {
        auto blockArgument = cast<BlockArgument>(operand);
        Operation *parentOp = blockArgument.getOwner()->getParentOp();
        if (!parentOp)
          continue;

        if (!opDepths.contains(parentOp) || newDepth < opDepths[parentOp])
          opDepths[parentOp] = newDepth;
      }
    }
    return true;
  };
  LogicalResult result = getBackwardSlice(rootOp, &backwardSlice, options);
  assert(result.succeeded() && "expected backward slice to succeed");
  (void)result;
  return options.inclusive ? backwardSlice.size() > 1
                           : backwardSlice.size() >= 1;
}

/// Matches transitive defs of a top-level operation up to N levels.
template <typename Matcher>
inline BackwardSliceMatcher<Matcher>
m_GetDefinitions(Matcher innerMatcher, int64_t maxDepth, bool inclusive,
                 bool omitBlockArguments, bool omitUsesFromAbove) {
  assert(maxDepth >= 0 && "maxDepth must be non-negative");
  return BackwardSliceMatcher<Matcher>(std::move(innerMatcher), maxDepth,
                                       inclusive, omitBlockArguments,
                                       omitUsesFromAbove);
}

/// Matches all transitive defs of a top-level operation up to N levels
template <typename Matcher>
inline BackwardSliceMatcher<Matcher> m_GetAllDefinitions(Matcher innerMatcher,
                                                         int64_t maxDepth) {
  assert(maxDepth >= 0 && "maxDepth must be non-negative");
  return BackwardSliceMatcher<Matcher>(std::move(innerMatcher), maxDepth, true,
                                       false, false);
}

} // namespace mlir::query::matcher

#endif // MLIR_TOOLS_MLIRQUERY_MATCHERS_SLICEMATCHERS_H
