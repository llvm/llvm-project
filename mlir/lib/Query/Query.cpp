//===---- Query.cpp - -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Query/Query.h"
#include "QueryParser.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Utils/Utils.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Query/Matcher/MatchFinder.h"
#include "mlir/Query/QuerySession.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::query {

QueryRef parse(llvm::StringRef line, const QuerySession &qs) {
  return QueryParser::parse(line, qs);
}

std::vector<llvm::LineEditor::Completion>
complete(llvm::StringRef line, size_t pos, const QuerySession &qs) {
  return QueryParser::complete(line, pos, qs);
}

Query::~Query() = default;

LogicalResult InvalidQuery::run(llvm::raw_ostream &os, QuerySession &qs) const {
  os << errStr << "\n";
  return mlir::failure();
}

LogicalResult NoOpQuery::run(llvm::raw_ostream &os, QuerySession &qs) const {
  return mlir::success();
}

LogicalResult HelpQuery::run(llvm::raw_ostream &os, QuerySession &qs) const {
  os << "Available commands:\n\n"
        "  match MATCHER, m MATCHER      "
        "Match the mlir against the given matcher.\n"
        "  quit                              "
        "Terminates the query session.\n\n";
  return mlir::success();
}

LogicalResult QuitQuery::run(llvm::raw_ostream &os, QuerySession &qs) const {
  qs.terminate = true;
  return mlir::success();
}

LogicalResult MatchQuery::run(llvm::raw_ostream &os, QuerySession &qs) const {
  Operation *rootOp = qs.getRootOp();
  int matchCount = 0;
  matcher::MatchFinder finder;

  StringRef functionName = matcher.getFunctionName();
  auto matches = finder.collectMatches(rootOp, std::move(matcher));

  // An extract call is recognized by considering if the matcher has a name.
  // TODO: Consider making the extract more explicit.
  if (!functionName.empty()) {
    std::vector<Operation *> flattenedMatches =
        finder.flattenMatchedOps(matches);
    func::FuncOp function = func::extractOperationsIntoFunction(
        flattenedMatches, rootOp->getContext(), functionName);
    if (failed(verify(function)))
      return mlir::failure();
    os << "\n" << *function << "\n\n";
    function->erase();
    return mlir::success();
  }

  os << "\n";
  for (auto &results : matches) {
    os << "Match #" << ++matchCount << ":\n\n";
    for (Operation *op : results.matchedOps) {
      if (op == results.rootOp) {
        finder.printMatch(os, qs, op, "root");
      } else {
        finder.printMatch(os, qs, op);
      }
    }
  }
  os << matchCount << (matchCount == 1 ? " match.\n\n" : " matches.\n\n");
  return mlir::success();
}

} // namespace mlir::query
