//===- ReductionTreePass.cpp - ReductionTreePass Implementation -----------===//
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

#include "mlir/Reducer/ReductionTreePass.h"
#include "mlir/Reducer/Passes.h"

#include "llvm/Support/Allocator.h"

using namespace mlir;

static std::unique_ptr<OpReducer> getOpReducer(llvm::StringRef opType) {
  if (opType == ModuleOp::getOperationName())
    return std::make_unique<Reducer<ModuleOp>>();
  else if (opType == FuncOp::getOperationName())
    return std::make_unique<Reducer<FuncOp>>();
  llvm_unreachable("Now only supports two built-in ops");
}

void ReductionTreePass::runOnOperation() {
  ModuleOp module = this->getOperation();
  std::unique_ptr<OpReducer> reducer = getOpReducer(opReducerName);
  std::vector<std::pair<int, int>> ranges = {
      {0, reducer->getNumTargetOps(module)}};

  llvm::SpecificBumpPtrAllocator<ReductionNode> allocator;

  ReductionNode *root = allocator.Allocate();
  new (root) ReductionNode(nullptr, ranges, allocator);

  ModuleOp golden = module;
  switch (traversalModeId) {
  case TraversalMode::SinglePath:
    golden = findOptimal<ReductionNode::iterator<TraversalMode::SinglePath>>(
        module, std::move(reducer), root);
    break;
  default:
    llvm_unreachable("Unsupported mode");
  }

  if (golden != module) {
    module.getBody()->clear();
    module.getBody()->getOperations().splice(module.getBody()->begin(),
                                             golden.getBody()->getOperations());
    golden->destroy();
  }
}

template <typename IteratorType>
ModuleOp ReductionTreePass::findOptimal(ModuleOp module,
                                        std::unique_ptr<OpReducer> reducer,
                                        ReductionNode *root) {
  Tester test(testerName, testerArgs);
  std::pair<Tester::Interestingness, size_t> initStatus =
      test.isInteresting(module);

  if (initStatus.first != Tester::Interestingness::True) {
    LLVM_DEBUG(llvm::dbgs() << "\nThe original input is not interested");
    return module;
  }

  root->update(initStatus);

  ReductionNode *smallestNode = root;
  ModuleOp golden = module;

  IteratorType iter(root);

  while (iter != IteratorType::end()) {
    ModuleOp cloneModule = module.clone();

    ReductionNode &currentNode = *iter;
    reducer->reduce(cloneModule, currentNode.getRanges());

    std::pair<Tester::Interestingness, size_t> result =
        test.isInteresting(cloneModule);
    currentNode.update(result);

    if (result.first == Tester::Interestingness::True &&
        result.second < smallestNode->getSize()) {
      smallestNode = &currentNode;
      golden = cloneModule;
    } else {
      cloneModule->destroy();
    }

    ++iter;
  }

  return golden;
}

std::unique_ptr<Pass> mlir::createReductionTreePass() {
  return std::make_unique<ReductionTreePass>();
}
