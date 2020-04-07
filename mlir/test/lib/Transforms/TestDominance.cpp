//===- TestDominance.cpp - Test dominance construction and information
//-------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains test passes for constructing and resolving dominance
// information.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Dominance.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

/// Helper class to print dominance information.
class DominanceTest {
public:
  /// Constructs a new test instance using the given operation.
  DominanceTest(Operation *operation) : operation(operation) {
    // Create unique ids for each block.
    operation->walk([&](Operation *nested) {
      if (blockIds.count(nested->getBlock()) > 0)
        return;
      blockIds.insert({nested->getBlock(), blockIds.size()});
    });
  }

  /// Prints dominance information of all blocks.
  template <typename DominanceT>
  void printDominance(DominanceT &dominanceInfo) {
    DenseSet<Block *> parentVisited;
    operation->walk([&](Operation *op) {
      Block *block = op->getBlock();
      if (!parentVisited.insert(block).second)
        return;

      DenseSet<Block *> visited;
      operation->walk([&](Operation *nested) {
        Block *nestedBlock = nested->getBlock();
        if (!visited.insert(nestedBlock).second)
          return;
        llvm::errs() << "Nearest(" << blockIds[block] << ", "
                     << blockIds[nestedBlock] << ") = ";
        Block *dom =
            dominanceInfo.findNearestCommonDominator(block, nestedBlock);
        if (dom)
          llvm::errs() << blockIds[dom];
        else
          llvm::errs() << "<no dom>";
        llvm::errs() << "\n";
      });
    });
  }

private:
  Operation *operation;
  DenseMap<Block *, size_t> blockIds;
};

struct TestDominancePass : public PassWrapper<TestDominancePass, FunctionPass> {

  void runOnFunction() override {
    llvm::errs() << "Testing : " << getFunction().getName() << "\n";
    DominanceTest dominanceTest(getFunction());

    // Print dominance information.
    llvm::errs() << "--- DominanceInfo ---\n";
    dominanceTest.printDominance(getAnalysis<DominanceInfo>());

    llvm::errs() << "--- PostDominanceInfo ---\n";
    dominanceTest.printDominance(getAnalysis<PostDominanceInfo>());
  }
};

} // end anonymous namespace

namespace mlir {
void registerTestDominancePass() {
  PassRegistration<TestDominancePass>(
      "test-print-dominance",
      "Print the dominance information for multiple regions.");
}
} // namespace mlir
