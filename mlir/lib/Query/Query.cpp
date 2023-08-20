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
#include "mlir/IR/IRMapping.h"
#include "mlir/Query/Matcher/MatchFinder.h"
#include "mlir/Query/QuerySession.h"
#include "mlir/Support/LogicalResult.h"
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

static void printMatch(llvm::raw_ostream &os, QuerySession &qs, Operation *op,
                       const std::string &binding) {
  auto fileLoc = op->getLoc()->findInstanceOf<FileLineColLoc>();
  auto smloc = qs.getSourceManager().FindLocForLineAndColumn(
      qs.getBufferId(), fileLoc.getLine(), fileLoc.getColumn());
  qs.getSourceManager().PrintMessage(os, smloc, llvm::SourceMgr::DK_Note,
                                     "\"" + binding + "\" binds here");
}

static Operation *extractFunction(std::vector<Operation *> &ops,
                                  MLIRContext *context,
                                  llvm::StringRef functionName) {
  context->loadDialect<func::FuncDialect>();
  OpBuilder builder(context);
  std::vector<Operation *> slice;
  std::vector<Value> values;

  bool hasReturn = false;
  TypeRange resultType = std::nullopt;

  for (auto *op : ops) {
    slice.push_back(op);
    if (auto returnOp = dyn_cast<func::ReturnOp>(op)) {
      resultType = returnOp.getOperands().getTypes();
      hasReturn = true;
    } else {
      // Extract all values that might potentially be needed as func
      // arguments.
      for (Value value : op->getOperands()) {
        values.push_back(value);
      }
    }
  }

  auto loc = builder.getUnknownLoc();

  if (!hasReturn) {
    resultType = slice.back()->getResults().getTypes();
  }

  func::FuncOp funcOp = func::FuncOp::create(
      loc, functionName,
      builder.getFunctionType(ValueRange(values), resultType));

  loc = funcOp.getLoc();
  builder.setInsertionPointToEnd(funcOp.addEntryBlock());
  builder.setInsertionPointToEnd(&funcOp.getBody().front());

  IRMapping mapper;
  for (const auto &arg : llvm::enumerate(values))
    mapper.map(arg.value(), funcOp.getArgument(arg.index()));

  std::vector<Operation *> clonedOps;
  for (Operation *slicedOp : slice)
    clonedOps.push_back(builder.clone(*slicedOp, mapper));

  // Remove func arguments that are not used.
  unsigned currentIndex = 0;
  while (currentIndex < funcOp.getNumArguments()) {
    if (funcOp.getArgument(currentIndex).getUses().empty()) {
      funcOp.eraseArgument(currentIndex);
    } else {
      currentIndex++;
    }
  }

  // Add an extra return operation with the result of the final operation
  if (!hasReturn) {
    builder.create<func::ReturnOp>(loc, clonedOps.back()->getResults());
  }

  return funcOp;
}

Query::~Query() = default;

mlir::LogicalResult InvalidQuery::run(llvm::raw_ostream &os,
                                      QuerySession &qs) const {
  os << errStr << "\n";
  return mlir::failure();
}

mlir::LogicalResult NoOpQuery::run(llvm::raw_ostream &os,
                                   QuerySession &qs) const {
  return mlir::success();
}

mlir::LogicalResult HelpQuery::run(llvm::raw_ostream &os,
                                   QuerySession &qs) const {
  os << "Available commands:\n\n"
        "  match MATCHER, m MATCHER      "
        "Match the mlir against the given matcher.\n"
        "  quit                              "
        "Terminates the query session.\n\n";
  return mlir::success();
}

mlir::LogicalResult QuitQuery::run(llvm::raw_ostream &os,
                                   QuerySession &qs) const {
  qs.terminate = true;
  return mlir::success();
}

mlir::LogicalResult MatchQuery::run(llvm::raw_ostream &os,
                                    QuerySession &qs) const {
  Operation *rootOp = qs.getRootOp();
  int matchCount = 0;
  std::vector<Operation *> matches =
      matcher::MatchFinder().getMatches(rootOp, matcher);

  if (matcher.hasFunctionName()) {
    auto functionName = matcher.getFunctionName();
    Operation *function =
        extractFunction(matches, rootOp->getContext(), functionName);
    os << "\n" << *function << "\n\n";
    return mlir::success();
  }

  os << "\n";
  for (Operation *op : matches) {
    os << "Match #" << ++matchCount << ":\n\n";
    // Placeholder "root" binding for the initial draft.
    printMatch(os, qs, op, "root");
  }
  os << matchCount << (matchCount == 1 ? " match.\n\n" : " matches.\n\n");

  return mlir::success();
}

} // namespace mlir::query
