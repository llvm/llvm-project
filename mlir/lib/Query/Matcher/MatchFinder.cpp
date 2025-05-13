//===- MatchFinder.cpp - --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the method definitions for the `MatchFinder` class
//
//===----------------------------------------------------------------------===//

#include "mlir/Query/Matcher/MatchFinder.h"
namespace mlir::query::matcher {

MatchFinder::MatchResult::MatchResult(Operation *rootOp,
                                      std::vector<Operation *> matchedOps)
    : rootOp(rootOp), matchedOps(std::move(matchedOps)) {}

std::vector<MatchFinder::MatchResult>
MatchFinder::collectMatches(Operation *root, DynMatcher matcher) const {
  std::vector<MatchResult> results;
  llvm::SetVector<Operation *> tempStorage;
  root->walk([&](Operation *subOp) {
    if (matcher.match(subOp)) {
      MatchResult match;
      match.rootOp = subOp;
      match.matchedOps.push_back(subOp);
      results.push_back(std::move(match));
    } else if (matcher.match(subOp, tempStorage)) {
      results.emplace_back(subOp, std::vector<Operation *>(tempStorage.begin(),
                                                           tempStorage.end()));
    }
    tempStorage.clear();
  });
  return results;
}

void MatchFinder::printMatch(llvm::raw_ostream &os, QuerySession &qs,
                             Operation *op) const {
  auto fileLoc = cast<FileLineColLoc>(op->getLoc());
  SMLoc smloc = qs.getSourceManager().FindLocForLineAndColumn(
      qs.getBufferId(), fileLoc.getLine(), fileLoc.getColumn());
  llvm::SMDiagnostic diag =
      qs.getSourceManager().GetMessage(smloc, llvm::SourceMgr::DK_Note, "");
  diag.print("", os, true, false, true);
}

void MatchFinder::printMatch(llvm::raw_ostream &os, QuerySession &qs,
                             Operation *op, const std::string &binding) const {
  auto fileLoc = cast<FileLineColLoc>(op->getLoc());
  auto smloc = qs.getSourceManager().FindLocForLineAndColumn(
      qs.getBufferId(), fileLoc.getLine(), fileLoc.getColumn());
  qs.getSourceManager().PrintMessage(os, smloc, llvm::SourceMgr::DK_Note,
                                     "\"" + binding + "\" binds here");
}

std::vector<Operation *>
MatchFinder::flattenMatchedOps(std::vector<MatchResult> &matches) const {
  std::vector<Operation *> newVector;
  for (auto &result : matches) {
    newVector.insert(newVector.end(), result.matchedOps.begin(),
                     result.matchedOps.end());
  }
  return newVector;
}

} // namespace mlir::query::matcher
