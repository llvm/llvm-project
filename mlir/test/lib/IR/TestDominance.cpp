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

#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

/// Overloaded helper to call the right function based on whether we are testing
/// dominance or post-dominance.
static bool dominatesOrPostDominates(DominanceInfo &dominanceInfo, Block *a,
                                     Block *b) {
  return dominanceInfo.dominates(a, b);
}

static bool dominatesOrPostDominates(PostDominanceInfo &dominanceInfo, Block *a,
                                     Block *b) {
  return dominanceInfo.postDominates(a, b);
}

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
  void printDominance(DominanceT &dominanceInfo,
                      bool printCommonDominatorInfo) {
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
        if (printCommonDominatorInfo) {
          llvm::errs() << "Nearest(" << blockIds[block] << ", "
                       << blockIds[nestedBlock] << ") = ";
          Block *dom =
              dominanceInfo.findNearestCommonDominator(block, nestedBlock);
          if (dom)
            llvm::errs() << blockIds[dom];
          else
            llvm::errs() << "<no dom>";
          llvm::errs() << "\n";
        } else {
          if (std::is_same<DominanceInfo, DominanceT>::value)
            llvm::errs() << "dominates(";
          else
            llvm::errs() << "postdominates(";
          llvm::errs() << blockIds[block] << ", " << blockIds[nestedBlock]
                       << ") = ";
          if (dominatesOrPostDominates(dominanceInfo, block, nestedBlock))
            llvm::errs() << "true\n";
          else
            llvm::errs() << "false\n";
        }
      });
    });
  }

private:
  Operation *operation;
  DenseMap<Block *, size_t> blockIds;
};

struct TestDominancePass
    : public PassWrapper<TestDominancePass, InterfacePass<SymbolOpInterface>> {
  StringRef getArgument() const final { return "test-print-dominance"; }
  StringRef getDescription() const final {
    return "Print the dominance information for multiple regions.";
  }

  void runOnOperation() override {
    llvm::errs() << "Testing : " << getOperation().getName() << "\n";
    DominanceTest dominanceTest(getOperation());

    // Print dominance information.
    llvm::errs() << "--- DominanceInfo ---\n";
    dominanceTest.printDominance(getAnalysis<DominanceInfo>(),
                                 /*printCommonDominatorInfo=*/true);

    llvm::errs() << "--- PostDominanceInfo ---\n";
    dominanceTest.printDominance(getAnalysis<PostDominanceInfo>(),
                                 /*printCommonDominatorInfo=*/true);

    // Print dominance relationship between blocks.
    llvm::errs() << "--- Block Dominance relationship ---\n";
    dominanceTest.printDominance(getAnalysis<DominanceInfo>(),
                                 /*printCommonDominatorInfo=*/false);

    llvm::errs() << "--- Block PostDominance relationship ---\n";
    dominanceTest.printDominance(getAnalysis<PostDominanceInfo>(),
                                 /*printCommonDominatorInfo=*/false);
  }
};

} // namespace

namespace mlir {
namespace test {
void registerTestDominancePass() { PassRegistration<TestDominancePass>(); }
} // namespace test
} // namespace mlir
