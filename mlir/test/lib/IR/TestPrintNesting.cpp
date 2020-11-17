//===- TestPrintNesting.cpp - Passes to illustrate the IR nesting ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
/// This pass illustrates the IR nesting through printing.
struct TestPrintNestingPass
    : public PassWrapper<TestPrintNestingPass, OperationPass<>> {
  // Entry point for the pass.
  void runOnOperation() override {
    Operation *op = getOperation();
    resetIndent();
    printOperation(op);
  }

  /// The three methods below are mutually recursive and follow the nesting of
  /// the IR: operation->region->block->operation->...

  void printOperation(Operation *op) {
    // Print the operation itself and some of its properties
    printIndent() << "visiting op: '" << op->getName() << "' with "
                  << op->getNumOperands() << " operands and "
                  << op->getNumResults() << " results\n";
    // Print the operation attributes
    if (!op->getAttrs().empty()) {
      printIndent() << op->getAttrs().size() << " attributes:\n";
      for (NamedAttribute attr : op->getAttrs())
        printIndent() << " - '" << attr.first << "' : '" << attr.second
                      << "'\n";
    }

    // Recurse into each of the regions attached to the operation.
    printIndent() << " " << op->getNumRegions() << " nested regions:\n";
    auto indent = pushIndent();
    for (Region &region : op->getRegions())
      printRegion(region);
  }

  void printRegion(Region &region) {
    // A region does not hold anything by itself other than a list of blocks.
    printIndent() << "Region with " << region.getBlocks().size()
                  << " blocks:\n";
    auto indent = pushIndent();
    for (Block &block : region.getBlocks())
      printBlock(block);
  }

  void printBlock(Block &block) {
    // Print the block intrinsics properties (basically: argument list)
    printIndent()
        << "Block with " << block.getNumArguments() << " arguments, "
        << block.getNumSuccessors()
        << " successors, and "
        // Note, this `.size()` is traversing a linked-list and is O(n).
        << block.getOperations().size() << " operations\n";

    // Block main role is to hold a list of Operations: let's recurse.
    auto indent = pushIndent();
    for (Operation &op : block.getOperations())
      printOperation(&op);
  }

  /// Manages the indentation as we traverse the IR nesting.
  int indent;
  struct IdentRAII {
    int &indent;
    IdentRAII(int &indent) : indent(indent) {}
    ~IdentRAII() { --indent; }
  };
  void resetIndent() { indent = 0; }
  IdentRAII pushIndent() { return IdentRAII(++indent); }

  llvm::raw_ostream &printIndent() {
    for (int i = 0; i < indent; ++i)
      llvm::outs() << "  ";
    return llvm::outs();
  }
};
} // end anonymous namespace

namespace mlir {
void registerTestPrintNestingPass() {
  PassRegistration<TestPrintNestingPass>("test-print-nesting",
                                         "Test various printing.");
}
} // namespace mlir
