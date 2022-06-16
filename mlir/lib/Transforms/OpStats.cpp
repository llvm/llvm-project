//===- OpStats.cpp - Prints stats of operations in module -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {
struct PrintOpStatsPass : public PrintOpStatsBase<PrintOpStatsPass> {
  explicit PrintOpStatsPass(raw_ostream &os) : os(os) {}

  explicit PrintOpStatsPass(raw_ostream &os, bool printAsJSON) : os(os) {
    this->printAsJSON = printAsJSON;
  }

  // Prints the resultant operation statistics post iterating over the module.
  void runOnOperation() override;

  // Print summary of op stats.
  void printSummary();

  // Print symmary of op stats in JSON.
  void printSummaryInJSON();

private:
  llvm::StringMap<int64_t> opCount;
  raw_ostream &os;
};
} // namespace

void PrintOpStatsPass::runOnOperation() {
  opCount.clear();

  // Compute the operation statistics for the currently visited operation.
  getOperation()->walk(
      [&](Operation *op) { ++opCount[op->getName().getStringRef()]; });
  if (printAsJSON) {
    printSummaryInJSON();
  } else
    printSummary();
}

void PrintOpStatsPass::printSummary() {
  os << "Operations encountered:\n";
  os << "-----------------------\n";
  SmallVector<StringRef, 64> sorted(opCount.keys());
  llvm::sort(sorted);

  // Split an operation name from its dialect prefix.
  auto splitOperationName = [](StringRef opName) {
    auto splitName = opName.split('.');
    return splitName.second.empty() ? std::make_pair("", splitName.first)
                                    : splitName;
  };

  // Compute the largest dialect and operation name.
  StringRef dialectName, opName;
  size_t maxLenOpName = 0, maxLenDialect = 0;
  for (const auto &key : sorted) {
    std::tie(dialectName, opName) = splitOperationName(key);
    maxLenDialect = std::max(maxLenDialect, dialectName.size());
    maxLenOpName = std::max(maxLenOpName, opName.size());
  }

  for (const auto &key : sorted) {
    std::tie(dialectName, opName) = splitOperationName(key);

    // Left-align the names (aligning on the dialect) and right-align the count
    // below. The alignment is for readability and does not affect CSV/FileCheck
    // parsing.
    if (dialectName.empty())
      os.indent(maxLenDialect + 3);
    else
      os << llvm::right_justify(dialectName, maxLenDialect + 2) << '.';

    // Left justify the operation name.
    os << llvm::left_justify(opName, maxLenOpName) << " , " << opCount[key]
       << '\n';
  }
}

void PrintOpStatsPass::printSummaryInJSON() {
  SmallVector<StringRef, 64> sorted(opCount.keys());
  llvm::sort(sorted);

  os << "{\n";

  for (unsigned i = 0, e = sorted.size(); i != e; ++i) {
    const auto &key = sorted[i];
    os << "  \"" << key << "\" : " << opCount[key];
    if (i != e - 1)
      os << ",\n";
    else
      os << "\n";
  }
  os << "}\n";
}

std::unique_ptr<Pass> mlir::createPrintOpStatsPass(raw_ostream &os) {
  return std::make_unique<PrintOpStatsPass>(os);
}

std::unique_ptr<Pass> mlir::createPrintOpStatsPass(raw_ostream &os,
                                                   bool printAsJSON) {
  return std::make_unique<PrintOpStatsPass>(os, printAsJSON);
}
