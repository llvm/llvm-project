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
#include "mlir/Query/QuerySession.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::query::matcher {

class MatchFinder {
private:
  static void printMatch(llvm::raw_ostream &os, QuerySession &qs,
                         mlir::Operation *op, const std::string &binding) {
    auto fileLoc = op->getLoc()->findInstanceOf<FileLineColLoc>();
    auto smloc = qs.getSourceManager().FindLocForLineAndColumn(
        qs.getBufferId(), fileLoc.getLine(), fileLoc.getColumn());
    qs.getSourceManager().PrintMessage(os, smloc, llvm::SourceMgr::DK_Note,
                                       "\"" + binding + "\" binds here");
  };

public:
  static std::vector<Operation *>
  getMatches(Operation *root, QueryOptions &options, DynMatcher matcher,
             llvm::raw_ostream &os, QuerySession &qs) {
    unsigned matchCount = 0;
    std::vector<Operation *> matchedOps;
    SetVector<Operation *> tempStorage;
    os << "\n";
    root->walk([&](Operation *subOp) {
      if (matcher.match(subOp)) {
        matchedOps.push_back(subOp);
        os << "Match #" << ++matchCount << ":\n\n";
        printMatch(os, qs, subOp, "root");
      } else {
        SmallVector<Operation *> printingOps;
        if (matcher.match(subOp, tempStorage, options)) {
          os << "Match #" << ++matchCount << ":\n\n";
          SmallVector<Operation *> printingOps(tempStorage.takeVector());
          for (auto op : printingOps) {
            printMatch(os, qs, op, "root");
            matchedOps.push_back(op);
          }
          printingOps.clear();
        }
      }
    });
    os << matchCount << (matchCount == 1 ? " match.\n\n" : " matches.\n\n");
    return matchedOps;
  }
};

} // namespace mlir::query::matcher

#endif // MLIR_TOOLS_MLIRQUERY_MATCHER_MATCHERFINDER_H
