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

/// A class that provides utilities to find operations in a DAG
class MatchFinder {

public:
  /// A subclass which preserves the matching information
  struct MatchResult {
    MatchResult() = default;
    MatchResult(Operation *rootOp, std::vector<Operation *> matchedOps);

    /// Contains the root operation of the matching environment
    Operation *rootOp = nullptr;
    /// Contains the matching enviroment. This allows the user to easily
    /// extract the matched operations
    std::vector<Operation *> matchedOps;
  };
  /// Traverses the DAG and collects the "rootOp" + "matching enviroment" for
  /// a given Matcher
  std::vector<MatchResult> collectMatches(Operation *root,
                                          DynMatcher matcher) const;
  /// Prints the matched operation
  void printMatch(llvm::raw_ostream &os, QuerySession &qs, Operation *op) const;
  /// Labels the matched operation with the given binding (e.g., "root") and
  /// prints it
  void printMatch(llvm::raw_ostream &os, QuerySession &qs, Operation *op,
                  const std::string &binding) const;
  /// Flattens a vector of MatchResults into a vector of operations
  std::vector<Operation *>
  flattenMatchedOps(std::vector<MatchResult> &matches) const;
};

} // namespace mlir::query::matcher

#endif // MLIR_TOOLS_MLIRQUERY_MATCHER_MATCHERFINDER_H
