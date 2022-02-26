//===- TestPrintDefUse.cpp - Passes to illustrate the IR def-use chains ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
/// This pass illustrates the IR def-use chains through printing.
struct TestPrintDefUsePass
    : public PassWrapper<TestPrintDefUsePass, OperationPass<>> {
  StringRef getArgument() const final { return "test-print-defuse"; }
  StringRef getDescription() const final { return "Test various printing."; }
  void runOnOperation() override {
    // Recursively traverse the IR nested under the current operation and print
    // every single operation and their operands and users.
    getOperation()->walk([](Operation *op) {
      llvm::outs() << "Visiting op '" << op->getName() << "' with "
                   << op->getNumOperands() << " operands:\n";

      // Print information about the producer of each of the operands.
      for (Value operand : op->getOperands()) {
        if (Operation *producer = operand.getDefiningOp()) {
          llvm::outs() << "  - Operand produced by operation '"
                       << producer->getName() << "'\n";
        } else {
          // If there is no defining op, the Value is necessarily a Block
          // argument.
          auto blockArg = operand.cast<BlockArgument>();
          llvm::outs() << "  - Operand produced by Block argument, number "
                       << blockArg.getArgNumber() << "\n";
        }
      }

      // Print information about the user of each of the result.
      llvm::outs() << "Has " << op->getNumResults() << " results:\n";
      for (const auto &indexedResult : llvm::enumerate(op->getResults())) {
        Value result = indexedResult.value();
        llvm::outs() << "  - Result " << indexedResult.index();
        if (result.use_empty()) {
          llvm::outs() << " has no uses\n";
          continue;
        }
        if (result.hasOneUse()) {
          llvm::outs() << " has a single use: ";
        } else {
          llvm::outs() << " has "
                       << std::distance(result.getUses().begin(),
                                        result.getUses().end())
                       << " uses:\n";
        }
        for (Operation *userOp : result.getUsers()) {
          llvm::outs() << "    - " << userOp->getName() << "\n";
        }
      }
    });
  }
};
} // namespace

namespace mlir {
void registerTestPrintDefUsePass() { PassRegistration<TestPrintDefUsePass>(); }
} // namespace mlir
