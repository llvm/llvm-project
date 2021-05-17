//===- TestAliasAnalysis.cpp - Test alias analysis results ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains test passes for constructing and testing alias analysis
// results.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
struct TestAliasAnalysisPass
    : public PassWrapper<TestAliasAnalysisPass, OperationPass<>> {
  void runOnOperation() override {
    llvm::errs() << "Testing : ";
    if (Attribute testName = getOperation()->getAttr("test.name"))
      llvm::errs() << testName << "\n";
    else
      llvm::errs() << getOperation()->getAttr("sym_name") << "\n";

    // Collect all of the values to check for aliasing behavior.
    AliasAnalysis &aliasAnalysis = getAnalysis<AliasAnalysis>();
    SmallVector<Value, 32> valsToCheck;
    getOperation()->walk([&](Operation *op) {
      if (!op->getAttr("test.ptr"))
        return;
      valsToCheck.append(op->result_begin(), op->result_end());
      for (Region &region : op->getRegions())
        for (Block &block : region)
          valsToCheck.append(block.args_begin(), block.args_end());
    });

    // Check for aliasing behavior between each of the values.
    for (auto it = valsToCheck.begin(), e = valsToCheck.end(); it != e; ++it)
      for (auto innerIt = valsToCheck.begin(); innerIt != it; ++innerIt)
        printAliasResult(aliasAnalysis.alias(*innerIt, *it), *innerIt, *it);
  }

  /// Print the result of an alias query.
  void printAliasResult(AliasResult result, Value lhs, Value rhs) {
    printAliasOperand(lhs);
    llvm::errs() << " <-> ";
    printAliasOperand(rhs);
    llvm::errs() << ": ";

    switch (result.getKind()) {
    case AliasResult::NoAlias:
      llvm::errs() << "NoAlias";
      break;
    case AliasResult::MayAlias:
      llvm::errs() << "MayAlias";
      break;
    case AliasResult::PartialAlias:
      llvm::errs() << "PartialAlias";
      break;
    case AliasResult::MustAlias:
      llvm::errs() << "MustAlias";
      break;
    }
    llvm::errs() << "\n";
  }
  /// Print a value that is used as an operand of an alias query.
  void printAliasOperand(Value value) {
    if (BlockArgument arg = value.dyn_cast<BlockArgument>()) {
      Region *region = arg.getParentRegion();
      unsigned parentBlockNumber =
          std::distance(region->begin(), arg.getOwner()->getIterator());
      llvm::errs() << region->getParentOp()
                          ->getAttrOfType<StringAttr>("test.ptr")
                          .getValue()
                   << ".region" << region->getRegionNumber();
      if (parentBlockNumber != 0)
        llvm::errs() << ".block" << parentBlockNumber;
      llvm::errs() << "#" << arg.getArgNumber();
      return;
    }
    OpResult result = value.cast<OpResult>();
    llvm::errs()
        << result.getOwner()->getAttrOfType<StringAttr>("test.ptr").getValue()
        << "#" << result.getResultNumber();
  }
};
} // end anonymous namespace

namespace mlir {
namespace test {
void registerTestAliasAnalysisPass() {
  PassRegistration<TestAliasAnalysisPass> pass("test-alias-analysis",
                                               "Test alias analysis results.");
}
} // namespace test
} // namespace mlir
