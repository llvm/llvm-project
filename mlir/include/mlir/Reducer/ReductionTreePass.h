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
#include "mlir/Reducer/Passes/FunctionReducer.h"
#include "mlir/Reducer/Tester.h"

namespace mlir {

/// Defines the traversal method options to be used in the reduction tree
/// traversal.
enum TraversalMode { SinglePath, MultiPath, Concurrent, Backtrack };

// This class defines the non- templated utilities used by the ReductionTreePass
// class.
class ReductionTreeUtils {
public:
  void updateGoldenModule(ModuleOp &golden, ModuleOp reduced);
};

/// This class defines the Reduction Tree Pass. It provides a framework to
/// to implement a reduction pass using a tree structure to keep track of the
/// generated reduced variants.
template <typename Reducer, TraversalMode mode>
class ReductionTreePass
    : public ReductionTreeBase<ReductionTreePass<Reducer, mode>> {
public:
  ReductionTreePass(const Tester *test) : test(test) {}

  ReductionTreePass(const ReductionTreePass &pass)
      : root(new ReductionNode(pass.root->getModule().clone(), nullptr)),
        test(pass.test) {}

  /// Runs the pass instance in the pass pipeline.
  void runOnOperation() override {
    ModuleOp module = this->getOperation();
    this->root = std::make_unique<ReductionNode>(module, nullptr);
    ReductionNode *reduced;

    switch (mode) {
    case SinglePath:
      reduced = singlePathTraversal();
      break;
    default:
      llvm::report_fatal_error("Traversal method not currently supported.");
    }

    ReductionTreeUtils utils;
    utils.updateGoldenModule(module, reduced->getModule());
  }

private:
  // Points to the root node in this reduction tree.
  std::unique_ptr<ReductionNode> root;

  // This object defines the variant generation at each level of the reduction
  // tree.
  Reducer reducer;

  // This is used to test the interesting behavior of the reduction nodes in the
  // tree.
  const Tester *test;

  /// Traverse the most reduced path in the reduction tree by generating the
  /// variants at each level using the Reducer parameter's generateVariants
  /// function. Stops when no new successful variants can be created at the
  /// current level.
  ReductionNode *singlePathTraversal() {
    ReductionNode *currLevel = root.get();

    while (true) {
      reducer.generateVariants(currLevel, test);
      currLevel->organizeVariants(test);

      if (currLevel->variantsEmpty())
        break;

      currLevel = currLevel->getVariant(0);
    }

    return currLevel;
  }
};

} // end namespace mlir

#endif
