//===- MatchFinder.h - ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the MatchFinder class, which is used to find operations
// that match a given matcher and print them.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIRQUERY_MATCHER_MATCHERFINDER_H
#define MLIR_TOOLS_MLIRQUERY_MATCHER_MATCHERFINDER_H

#include "MatchersInternal.h"
#include "mlir/Query/Query.h"
#include "mlir/Query/QuerySession.h"
#include "llvm/ADT/SetVector.h"

namespace mlir::query::matcher {

/// Finds and collects matches from the IR. After construction
/// `collectMatches` can be used to traverse the IR and apply
/// matchers.
class MatchFinder {

public:
  /// A subclass which preserves the matching information. Each instance
  /// contains the `rootOp` along with the matching environment.
  struct MatchResult {
    MatchResult() = default;
    MatchResult(Operation *rootOp, std::vector<Operation *> matchedOps);

    Operation *rootOp = nullptr;
    /// Contains the matching environment.
    std::vector<Operation *> matchedOps;
  };

  /// Traverses the IR and returns a vector of `MatchResult` for each match of
  /// the `matcher`.
  std::vector<MatchResult> collectMatches(Operation *root,
                                          DynMatcher matcher) const;

  /// Prints the matched operation.
  void printMatch(llvm::raw_ostream &os, QuerySession &qs, Operation *op) const;

  /// Labels the matched operation with the given binding (e.g., `"root"`) and
  /// prints it.
  void printMatch(llvm::raw_ostream &os, QuerySession &qs, Operation *op,
                  const std::string &binding) const;

  /// Flattens a vector of `MatchResult` into a vector of operations.
  std::vector<Operation *>
  flattenMatchedOps(std::vector<MatchResult> &matches) const;
};

} // namespace mlir::query::matcher

#endif // MLIR_TOOLS_MLIRQUERY_MATCHER_MATCHERFINDER_H
