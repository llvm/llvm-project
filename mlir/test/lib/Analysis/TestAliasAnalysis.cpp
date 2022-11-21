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

#include "TestAliasAnalysis.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

/// Print a value that is used as an operand of an alias query.
static void printAliasOperand(Operation *op) {
  llvm::errs() << op->getAttrOfType<StringAttr>("test.ptr").getValue();
}
static void printAliasOperand(Value value) {
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
  printAliasOperand(result.getOwner());
  llvm::errs() << "#" << result.getResultNumber();
}

namespace mlir {
namespace test {
void printAliasResult(AliasResult result, Value lhs, Value rhs) {
  printAliasOperand(lhs);
  llvm::errs() << " <-> ";
  printAliasOperand(rhs);
  llvm::errs() << ": " << result << "\n";
}

/// Print the result of an alias query.
void printModRefResult(ModRefResult result, Operation *op, Value location) {
  printAliasOperand(op);
  llvm::errs() << " -> ";
  printAliasOperand(location);
  llvm::errs() << ": " << result << "\n";
}

void TestAliasAnalysisBase::runAliasAnalysisOnOperation(
    Operation *op, AliasAnalysis &aliasAnalysis) {
  llvm::errs() << "Testing : " << op->getAttr("sym_name") << "\n";

  // Collect all of the values to check for aliasing behavior.
  SmallVector<Value, 32> valsToCheck;
  op->walk([&](Operation *op) {
    if (!op->getAttr("test.ptr"))
      return;
    valsToCheck.append(op->result_begin(), op->result_end());
    for (Region &region : op->getRegions())
      for (Block &block : region)
        valsToCheck.append(block.args_begin(), block.args_end());
  });

  // Check for aliasing behavior between each of the values.
  for (auto it = valsToCheck.begin(), e = valsToCheck.end(); it != e; ++it)
    for (auto *innerIt = valsToCheck.begin(); innerIt != it; ++innerIt)
      printAliasResult(aliasAnalysis.alias(*innerIt, *it), *innerIt, *it);
}

void TestAliasAnalysisModRefBase::runAliasAnalysisOnOperation(
    Operation *op, AliasAnalysis &aliasAnalysis) {
  llvm::errs() << "Testing : " << op->getAttr("sym_name") << "\n";

  // Collect all of the values to check for aliasing behavior.
  SmallVector<Value, 32> valsToCheck;
  op->walk([&](Operation *op) {
    if (!op->getAttr("test.ptr"))
      return;
    valsToCheck.append(op->result_begin(), op->result_end());
    for (Region &region : op->getRegions())
      for (Block &block : region)
        valsToCheck.append(block.args_begin(), block.args_end());
  });

  // Check for aliasing behavior between each of the values.
  for (auto &it : valsToCheck) {
    op->walk([&](Operation *op) {
      if (!op->getAttr("test.ptr"))
        return;
      printModRefResult(aliasAnalysis.getModRef(op, it), op, it);
    });
  }
}

} // namespace test
} // namespace mlir

//===----------------------------------------------------------------------===//
// Testing AliasResult
//===----------------------------------------------------------------------===//

namespace {
struct TestAliasAnalysisPass
    : public test::TestAliasAnalysisBase,
      PassWrapper<TestAliasAnalysisPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestAliasAnalysisPass)

  StringRef getArgument() const final { return "test-alias-analysis"; }
  StringRef getDescription() const final {
    return "Test alias analysis results.";
  }
  void runOnOperation() override {
    AliasAnalysis &aliasAnalysis = getAnalysis<AliasAnalysis>();
    runAliasAnalysisOnOperation(getOperation(), aliasAnalysis);
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Testing ModRefResult
//===----------------------------------------------------------------------===//

namespace {
struct TestAliasAnalysisModRefPass
    : public test::TestAliasAnalysisModRefBase,
      PassWrapper<TestAliasAnalysisModRefPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestAliasAnalysisModRefPass)

  StringRef getArgument() const final { return "test-alias-analysis-modref"; }
  StringRef getDescription() const final {
    return "Test alias analysis ModRef results.";
  }
  void runOnOperation() override {
    AliasAnalysis &aliasAnalysis = getAnalysis<AliasAnalysis>();
    runAliasAnalysisOnOperation(getOperation(), aliasAnalysis);
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

namespace mlir {
namespace test {
void registerTestAliasAnalysisPass() {
  PassRegistration<TestAliasAnalysisPass>();
  PassRegistration<TestAliasAnalysisModRefPass>();
}
} // namespace test
} // namespace mlir
