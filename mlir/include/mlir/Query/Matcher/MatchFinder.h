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

public:
  //
  // getMatches walks the IR and prints operations as soon as it matches them
  // if a matcher is to be further extracted into the function, then it does not
  // print operations
  //
  static std::vector<Operation *>
  getMatches(Operation *root, QueryOptions &options, DynMatcher matcher,
             llvm::raw_ostream &os, QuerySession &qs) {
    int matchCount = 0;
    bool printMatchingOps = true;
    // If matcher is to be extracted to a function, we don't want to print
    // matching ops to sdout
    if (matcher.hasFunctionName()) {
      printMatchingOps = false;
    }
    std::vector<Operation *> matchedOps;
    SetVector<Operation *> tempStorage;
    os << "\n";
    root->walk([&](Operation *subOp) {
      if (matcher.match(subOp)) {
        matchedOps.push_back(subOp);
        if (printMatchingOps) {
          os << "Match #" << ++matchCount << ":\n\n";
          printMatch(os, qs, subOp, "root");
        }
      } else {
        SmallVector<Operation *> printingOps;
        if (matcher.match(subOp, tempStorage, options)) {
          if (printMatchingOps) {
            os << "Match #" << ++matchCount << ":\n\n";
          }
          SmallVector<Operation *> printingOps(tempStorage.takeVector());
          for (auto op : printingOps) {
            if (printMatchingOps) {
              printMatch(os, qs, op, "root");
            }
            matchedOps.push_back(op);
          }
          printingOps.clear();
        }
      }
    });
    if (printMatchingOps) {
      os << matchCount << (matchCount == 1 ? " match.\n\n" : " matches.\n\n");
    }
    return matchedOps;
  }

private:
  // Overloaded version that doesn't print the binding
  static void printMatch(llvm::raw_ostream &os, QuerySession &qs,
                         mlir::Operation *op) {
    auto fileLoc = op->getLoc()->dyn_cast<FileLineColLoc>();
    SMLoc smloc = qs.getSourceManager().FindLocForLineAndColumn(
        qs.getBufferId(), fileLoc.getLine(), fileLoc.getColumn());

    llvm::SMDiagnostic diag =
        qs.getSourceManager().GetMessage(smloc, llvm::SourceMgr::DK_Note,

                                         "");
    diag.print("", os, true, false, true);
  }
  static void printMatch(llvm::raw_ostream &os, QuerySession &qs,
                         mlir::Operation *op, const std::string &binding) {
    auto fileLoc = op->getLoc()->findInstanceOf<FileLineColLoc>();
    auto smloc = qs.getSourceManager().FindLocForLineAndColumn(
        qs.getBufferId(), fileLoc.getLine(), fileLoc.getColumn());
    qs.getSourceManager().PrintMessage(os, smloc, llvm::SourceMgr::DK_Note,
                                       "\"" + binding + "\" binds here");
  }
};

} // namespace mlir::query::matcher

#endif // MLIR_TOOLS_MLIRQUERY_MATCHER_MATCHERFINDER_H
