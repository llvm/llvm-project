//===- ExtraMatchers.h - Various common matchers --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides matchers that depend on Query.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIRQUERY_MATCHERS_EXTRAMATCHERS_H
#define MLIR_TOOLS_MLIRQUERY_MATCHERS_EXTRAMATCHERS_H
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Query/Matcher/MatchersInternal.h"

/// A matcher encapsulating the initial `getBackwardSlice` method from
/// SliceAnalysis.h
/// Additionally, it limits the slice computation to a certain depth level using
/// a custom filter
///
/// Example starting from node 9, assuming the matcher
/// computes the slice for the first two depth levels
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
class BackwardSliceMatcher {
public:
  explicit BackwardSliceMatcher(query::matcher::DynMatcher &&innerMatcher,
                                int64_t maxDepth, bool inclusive,
                                bool omitBlockArguments, bool omitUsesFromAbove)
      : innerMatcher(std::move(innerMatcher)), maxDepth(maxDepth),
        inclusive(inclusive), omitBlockArguments(omitBlockArguments),
        omitUsesFromAbove(omitUsesFromAbove) {}
  bool match(Operation *op, SetVector<Operation *> &backwardSlice) {
    BackwardSliceOptions options;
    return (innerMatcher.match(op) &&
            matches(op, backwardSlice, options, maxDepth));
  }

private:
  bool matches(Operation *rootOp, llvm::SetVector<Operation *> &backwardSlice,
               BackwardSliceOptions &options, int64_t maxDepth);

private:
  // The outer matcher (e.g., BackwardSliceMatcher) relies on the innerMatcher
  // to determine whether we want to traverse the DAG or not. For example, we
  // want to explore the DAG only if the top-level operation name is
  // "arith.addf".
  query::matcher::DynMatcher innerMatcher;
  // maxDepth specifies the maximum depth that the matcher can traverse in the
  // DAG. For example, if maxDepth is 2, the matcher will explore the defining
  // operations of the top-level op up to 2 levels.
  int64_t maxDepth;

  bool inclusive;
  bool omitBlockArguments;
  bool omitUsesFromAbove;
};

// Matches transitive defs of a top level operation up to N levels
inline BackwardSliceMatcher
m_GetDefinitions(query::matcher::DynMatcher innerMatcher, int64_t maxDepth,
                 bool inclusive, bool omitBlockArguments,
                 bool omitUsesFromAbove) {
  assert(maxDepth >= 0 && "maxDepth must be non-negative");
  return BackwardSliceMatcher(std::move(innerMatcher), maxDepth, inclusive,
                              omitBlockArguments, omitUsesFromAbove);
}
} // namespace mlir::query::matcher

#endif // MLIR_TOOLS_MLIRQUERY_MATCHERS_EXTRAMATCHERS_H
