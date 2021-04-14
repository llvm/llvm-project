//===- ReductionTreePass.h - Reduction Tree Pass Implementation -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the Reduction Tree Pass class. It provides a framework for
// the implementation of different reduction passes in the MLIR Reduce tool. It
// allows for custom specification of the variant generation behavior. It
// implements methods that define the different possible traversals of the
// reduction tree.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_REDUCER_REDUCTIONTREEPASS_H
#define MLIR_REDUCER_REDUCTIONTREEPASS_H

#include <vector>

#include "PassDetail.h"
#include "ReductionNode.h"
#include "mlir/Reducer/Passes/OpReducer.h"
#include "mlir/Reducer/ReductionTreeUtils.h"
#include "mlir/Reducer/Tester.h"

#define DEBUG_TYPE "mlir-reduce"

namespace mlir {

// Defines the traversal method options to be used in the reduction tree
/// traversal.
enum TraversalMode { SinglePath, Backtrack, MultiPath };

/// This class defines the Reduction Tree Pass. It provides a framework to
/// to implement a reduction pass using a tree structure to keep track of the
/// generated reduced variants.
template <typename Reducer, TraversalMode mode>
class ReductionTreePass
    : public ReductionTreeBase<ReductionTreePass<Reducer, mode>> {
public:
  ReductionTreePass(const ReductionTreePass &pass)
      : ReductionTreeBase<ReductionTreePass<Reducer, mode>>(pass),
        root(new ReductionNode(pass.root->getModule().clone(), nullptr)),
        test(pass.test) {}

  ReductionTreePass(const Tester &test) : test(test) {}

  /// Runs the pass instance in the pass pipeline.
  void runOnOperation() override {
    ModuleOp module = this->getOperation();
    Reducer reducer;
    std::vector<bool> transformSpace = reducer.initTransformSpace(module);
    ReductionNode *reduced;

    this->root =
        std::make_unique<ReductionNode>(module, nullptr, transformSpace);

    root->measureAndTest(test);

    LLVM_DEBUG(llvm::dbgs() << "\nReduction Tree Pass: " << reducer.getName(););
    switch (mode) {
    case SinglePath:
      LLVM_DEBUG(llvm::dbgs() << " (Single Path)\n";);
      reduced = singlePathTraversal();
      break;
    default:
      llvm::report_fatal_error("Traversal method not currently supported.");
    }

    ReductionTreeUtils::updateGoldenModule(module,
                                           reduced->getModule().clone());
  }

private:
  // Points to the root node in this reduction tree.
  std::unique_ptr<ReductionNode> root;

  // This object defines the variant generation at each level of the reduction
  // tree.
  Reducer reducer;

  // This is used to test the interesting behavior of the reduction nodes in the
  // tree.
  const Tester &test;

  /// Traverse the most reduced path in the reduction tree by generating the
  /// variants at each level using the Reducer parameter's generateVariants
  /// function. Stops when no new successful variants can be created at the
  /// current level.
  ReductionNode *singlePathTraversal() {
    ReductionNode *currNode = root.get();
    ReductionNode *smallestNode = currNode;
    int tSpaceSize = currNode->transformSpaceSize();
    std::vector<int> path;

    ReductionTreeUtils::updateSmallestNode(currNode, smallestNode, path);

    LLVM_DEBUG(llvm::dbgs() << "\nGenerating 1 variant: applying the ");
    LLVM_DEBUG(llvm::dbgs() << "transformation to the entire module\n");

    reducer.generateVariants(currNode, test, 1);
    LLVM_DEBUG(llvm::dbgs() << "Testing\n");
    currNode->organizeVariants(test);

    if (!currNode->variantsEmpty())
      return currNode->getVariant(0);

    while (tSpaceSize != 1) {
      ReductionTreeUtils::updateSmallestNode(currNode, smallestNode, path);

      LLVM_DEBUG(llvm::dbgs() << "\nGenerating 2 variants: applying the ");
      LLVM_DEBUG(llvm::dbgs() << "transformation to two different sections ");
      LLVM_DEBUG(llvm::dbgs() << "of transformable indices\n");

      reducer.generateVariants(currNode, test, 2);
      LLVM_DEBUG(llvm::dbgs() << "Testing\n");
      currNode->organizeVariants(test);

      if (currNode->variantsEmpty())
        break;

      currNode = currNode->getVariant(0);
      tSpaceSize = currNode->transformSpaceSize();
      path.push_back(0);
    }

    if (tSpaceSize == 1) {
      ReductionTreeUtils::updateSmallestNode(currNode, smallestNode, path);

      LLVM_DEBUG(llvm::dbgs() << "\nGenerating 1 variants: applying the ");
      LLVM_DEBUG(llvm::dbgs() << "transformation to the only transformable");
      LLVM_DEBUG(llvm::dbgs() << "index\n");

      reducer.generateVariants(currNode, test, 1);
      LLVM_DEBUG(llvm::dbgs() << "Testing\n");
      currNode->organizeVariants(test);

      if (!currNode->variantsEmpty()) {
        currNode = currNode->getVariant(0);
        path.push_back(0);

        ReductionTreeUtils::updateSmallestNode(currNode, smallestNode, path);
      }
    }

    return currNode;
  }
};

} // end namespace mlir

#endif
