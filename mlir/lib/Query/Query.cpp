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

// TODO: Extract into a helper function that can be reused outside query
// context.
static Operation *extractFunction(std::vector<Operation *> &ops,
                                  MLIRContext *context,
                                  llvm::StringRef functionName) {
  context->loadDialect<func::FuncDialect>();
  OpBuilder builder(context);

  // Collect data for function creation
  std::vector<Operation *> slice;
  std::vector<Value> values;
  std::vector<Type> outputTypes;

  for (auto *op : ops) {
    // Return op's operands are propagated, but the op itself isn't needed.
    if (!isa<func::ReturnOp>(op))
      slice.push_back(op);

    // All results are returned by the extracted function.
    llvm::append_range(outputTypes, op->getResults().getTypes());

    // Track all values that need to be taken as input to function.
    llvm::append_range(values, op->getOperands());
  }

  // Create the function
  FunctionType funcType =
      builder.getFunctionType(TypeRange(ValueRange(values)), outputTypes);
  auto loc = builder.getUnknownLoc();
  func::FuncOp funcOp = func::FuncOp::create(loc, functionName, funcType);

  builder.setInsertionPointToEnd(funcOp.addEntryBlock());

  // Map original values to function arguments
  IRMapping mapper;
  for (const auto &arg : llvm::enumerate(values))
    mapper.map(arg.value(), funcOp.getArgument(arg.index()));

  // Clone operations and build function body
  std::vector<Operation *> clonedOps;
  std::vector<Value> clonedVals;
  // TODO: Handle extraction of operations with compute payloads defined via
  // regions.
  for (Operation *slicedOp : slice) {
    Operation *clonedOp =
        clonedOps.emplace_back(builder.clone(*slicedOp, mapper));
    clonedVals.insert(clonedVals.end(), clonedOp->result_begin(),
                      clonedOp->result_end());
  }
  // Add return operation
  func::ReturnOp::create(builder, loc, clonedVals);

  // Remove unused function arguments
  size_t currentIndex = 0;
  while (currentIndex < funcOp.getNumArguments()) {
    // Erase if possible.
    if (funcOp.getArgument(currentIndex).use_empty())
      if (succeeded(funcOp.eraseArgument(currentIndex)))
        continue;
    ++currentIndex;
  }

  return funcOp;
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
  auto matches = finder.collectMatches(rootOp, std::move(matcher));

  // An extract call is recognized by considering if the matcher has a name.
  // TODO: Consider making the extract more explicit.
  if (matcher.hasFunctionName()) {
    auto functionName = matcher.getFunctionName();
    std::vector<Operation *> flattenedMatches =
        finder.flattenMatchedOps(matches);
    Operation *function =
        extractFunction(flattenedMatches, rootOp->getContext(), functionName);
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
