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

#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/SymbolTable.h"
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
static bool properlyDominatesOrPostDominates(DominanceInfo &dominanceInfo,
                                             Block *a, Block *b) {
  return dominanceInfo.properlyDominates(a, b);
}
static bool properlyDominatesOrPostDominates(PostDominanceInfo &dominanceInfo,
                                             Block *a, Block *b) {
  return dominanceInfo.properlyPostDominates(a, b);
}

namespace {

/// Helper class to print dominance information.
class DominanceTest {
public:
  static constexpr StringRef kBlockIdsAttrName = "test.block_ids";

  /// Constructs a new test instance using the given operation.
  DominanceTest(Operation *operation) : operation(operation) {
    Builder b(operation->getContext());

    // Helper function that annotates the IR with block IDs.
    auto annotateBlockId = [&](Operation *op, int64_t blockId) {
      auto idAttr = op->getAttrOfType<DenseI64ArrayAttr>(kBlockIdsAttrName);
      SmallVector<int64_t> ids;
      if (idAttr)
        ids = llvm::to_vector(idAttr.asArrayRef());
      ids.push_back(blockId);
      op->setAttr(kBlockIdsAttrName, b.getDenseI64ArrayAttr(ids));
    };

    // Create unique IDs for each block.
    operation->walk([&](Operation *nested) {
      if (blockIds.count(nested->getBlock()) > 0)
        return;
      blockIds.insert({nested->getBlock(), blockIds.size()});
      annotateBlockId(nested->getBlock()->getParentOp(), blockIds.size() - 1);
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
          llvm::outs() << "Nearest(" << blockIds[block] << ", "
                       << blockIds[nestedBlock] << ") = ";
          Block *dom =
              dominanceInfo.findNearestCommonDominator(block, nestedBlock);
          if (dom)
            llvm::outs() << blockIds[dom];
          else
            llvm::outs() << "<no dom>";
          llvm::outs() << "\n";
        } else {
          if (std::is_same<DominanceInfo, DominanceT>::value)
            llvm::outs() << "dominates(";
          else
            llvm::outs() << "postdominates(";
          llvm::outs() << blockIds[block] << ", " << blockIds[nestedBlock]
                       << ") = "
                       << std::to_string(dominatesOrPostDominates(
                              dominanceInfo, block, nestedBlock))
                       << " (properly = "
                       << std::to_string(properlyDominatesOrPostDominates(
                              dominanceInfo, block, nestedBlock))
                       << ")\n";
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
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestDominancePass)

  StringRef getArgument() const final { return "test-print-dominance"; }
  StringRef getDescription() const final {
    return "Print the dominance information for multiple regions.";
  }

  void runOnOperation() override {
    llvm::outs() << "Testing : " << getOperation().getName() << "\n";
    DominanceTest dominanceTest(getOperation());

    // Print dominance information.
    llvm::outs() << "--- DominanceInfo ---\n";
    dominanceTest.printDominance(getAnalysis<DominanceInfo>(),
                                 /*printCommonDominatorInfo=*/true);

    llvm::outs() << "--- PostDominanceInfo ---\n";
    dominanceTest.printDominance(getAnalysis<PostDominanceInfo>(),
                                 /*printCommonDominatorInfo=*/true);

    // Print dominance relationship between blocks.
    llvm::outs() << "--- Block Dominance relationship ---\n";
    dominanceTest.printDominance(getAnalysis<DominanceInfo>(),
                                 /*printCommonDominatorInfo=*/false);

    llvm::outs() << "--- Block PostDominance relationship ---\n";
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
