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

#include "mlir/IR/DialectInterface.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Reducer/Passes.h"
#include "mlir/Reducer/ReductionNode.h"
#include "mlir/Reducer/ReductionPatternInterface.h"
#include "mlir/Reducer/Tester.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/ManagedStatic.h"

namespace mlir {
#define GEN_PASS_DEF_REDUCTIONTREE
#include "mlir/Reducer/Passes.h.inc"
} // namespace mlir

using namespace mlir;

/// We implicitly number each operation in the region and if an operation's
/// number falls into rangeToKeep, we need to keep it and apply the given
/// rewrite patterns on it.
static void applyPatterns(Region &region,
                          const FrozenRewritePatternSet &patterns,
                          ArrayRef<ReductionNode::Range> rangeToKeep,
                          bool eraseOpNotInRange) {
  std::vector<Operation *> opsNotInRange;
  std::vector<Operation *> opsInRange;
  size_t keepIndex = 0;
  for (const auto &op : enumerate(region.getOps())) {
    int index = op.index();
    if (keepIndex < rangeToKeep.size() &&
        index == rangeToKeep[keepIndex].second)
      ++keepIndex;
    if (keepIndex == rangeToKeep.size() || index < rangeToKeep[keepIndex].first)
      opsNotInRange.push_back(&op.value());
    else
      opsInRange.push_back(&op.value());
  }

  // `applyOpPatternsAndFold` may erase the ops so we can't do the pattern
  // matching in above iteration. Besides, erase op not-in-range may end up in
  // invalid module, so `applyOpPatternsAndFold` should come before that
  // transform.
  for (Operation *op : opsInRange) {
    // `applyOpPatternsAndFold` returns whether the op is convered. Omit it
    // because we don't have expectation this reduction will be success or not.
    GreedyRewriteConfig config;
    config.strictMode = GreedyRewriteStrictness::ExistingOps;
    (void)applyOpPatternsGreedily(op, patterns, config);
  }

  if (eraseOpNotInRange)
    for (Operation *op : opsNotInRange) {
      op->dropAllUses();
      op->erase();
    }
}

/// We will apply the reducer patterns to the operations in the ranges specified
/// by ReductionNode. Note that we are not able to remove an operation without
/// replacing it with another valid operation. However, The validity of module
/// reduction is based on the Tester provided by the user and that means certain
/// invalid module is still interested by the use. Thus we provide an
/// alternative way to remove operations, which is using `eraseOpNotInRange` to
/// erase the operations not in the range specified by ReductionNode.
template <typename IteratorType>
static LogicalResult findOptimal(ModuleOp module, Region &region,
                                 const FrozenRewritePatternSet &patterns,
                                 const Tester &test, bool eraseOpNotInRange) {
  std::pair<Tester::Interestingness, size_t> initStatus =
      test.isInteresting(module);
  // While exploring the reduction tree, we always branch from an interesting
  // node. Thus the root node must be interesting.
  if (initStatus.first != Tester::Interestingness::True)
    return module.emitWarning() << "uninterested module will not be reduced";

  llvm::SpecificBumpPtrAllocator<ReductionNode> allocator;

  std::vector<ReductionNode::Range> ranges{
      {0, std::distance(region.op_begin(), region.op_end())}};

  ReductionNode *root = allocator.Allocate();
  new (root) ReductionNode(nullptr, ranges, allocator);
  // Duplicate the module for root node and locate the region in the copy.
  if (failed(root->initialize(module, region)))
    llvm_unreachable("unexpected initialization failure");
  root->update(initStatus);

  ReductionNode *smallestNode = root;
  IteratorType iter(root);

  while (iter != IteratorType::end()) {
    ReductionNode &currentNode = *iter;
    Region &curRegion = currentNode.getRegion();

    applyPatterns(curRegion, patterns, currentNode.getRanges(),
                  eraseOpNotInRange);
    currentNode.update(test.isInteresting(currentNode.getModule()));

    if (currentNode.isInteresting() == Tester::Interestingness::True &&
        currentNode.getSize() < smallestNode->getSize())
      smallestNode = &currentNode;

    ++iter;
  }

  // At here, we have found an optimal path to reduce the given region. Retrieve
  // the path and apply the reducer to it.
  SmallVector<ReductionNode *> trace;
  ReductionNode *curNode = smallestNode;
  trace.push_back(curNode);
  while (curNode != root) {
    curNode = curNode->getParent();
    trace.push_back(curNode);
  }

  // Reduce the region through the optimal path.
  while (!trace.empty()) {
    ReductionNode *top = trace.pop_back_val();
    applyPatterns(region, patterns, top->getStartRanges(), eraseOpNotInRange);
  }

  if (test.isInteresting(module).first != Tester::Interestingness::True)
    llvm::report_fatal_error("Reduced module is not interesting");
  if (test.isInteresting(module).second != smallestNode->getSize())
    llvm::report_fatal_error(
        "Reduced module doesn't have consistent size with smallestNode");
  return success();
}

template <typename IteratorType>
static LogicalResult findOptimal(ModuleOp module, Region &region,
                                 const FrozenRewritePatternSet &patterns,
                                 const Tester &test) {
  // We separate the reduction process into 2 steps, the first one is to erase
  // redundant operations and the second one is to apply the reducer patterns.

  // In the first phase, we don't apply any patterns so that we only select the
  // range of operations to keep to the module stay interesting.
  if (failed(findOptimal<IteratorType>(module, region, /*patterns=*/{}, test,
                                       /*eraseOpNotInRange=*/true)))
    return failure();
  // In the second phase, we suppose that no operation is redundant, so we try
  // to rewrite the operation into simpler form.
  return findOptimal<IteratorType>(module, region, patterns, test,
                                   /*eraseOpNotInRange=*/false);
}

namespace {

//===----------------------------------------------------------------------===//
// Reduction Pattern Interface Collection
//===----------------------------------------------------------------------===//

class ReductionPatternInterfaceCollection
    : public DialectInterfaceCollection<DialectReductionPatternInterface> {
public:
  using Base::Base;

  // Collect the reduce patterns defined by each dialect.
  void populateReductionPatterns(RewritePatternSet &pattern) const {
    for (const DialectReductionPatternInterface &interface : *this)
      interface.populateReductionPatterns(pattern);
  }
};

//===----------------------------------------------------------------------===//
// ReductionTreePass
//===----------------------------------------------------------------------===//

/// This class defines the Reduction Tree Pass. It provides a framework to
/// to implement a reduction pass using a tree structure to keep track of the
/// generated reduced variants.
class ReductionTreePass : public impl::ReductionTreeBase<ReductionTreePass> {
public:
  ReductionTreePass() = default;
  ReductionTreePass(const ReductionTreePass &pass) = default;

  LogicalResult initialize(MLIRContext *context) override;

  /// Runs the pass instance in the pass pipeline.
  void runOnOperation() override;

private:
  LogicalResult reduceOp(ModuleOp module, Region &region);

  FrozenRewritePatternSet reducerPatterns;
};

} // namespace

LogicalResult ReductionTreePass::initialize(MLIRContext *context) {
  RewritePatternSet patterns(context);
  ReductionPatternInterfaceCollection reducePatternCollection(context);
  reducePatternCollection.populateReductionPatterns(patterns);
  reducerPatterns = std::move(patterns);
  return success();
}

void ReductionTreePass::runOnOperation() {
  Operation *topOperation = getOperation();
  while (topOperation->getParentOp() != nullptr)
    topOperation = topOperation->getParentOp();
  ModuleOp module = dyn_cast<ModuleOp>(topOperation);
  if (!module) {
    emitError(getOperation()->getLoc())
        << "top-level op must be 'builtin.module'";
    return signalPassFailure();
  }

  SmallVector<Operation *, 8> workList;
  workList.push_back(getOperation());

  do {
    Operation *op = workList.pop_back_val();

    for (Region &region : op->getRegions())
      if (!region.empty())
        if (failed(reduceOp(module, region)))
          return signalPassFailure();

    for (Region &region : op->getRegions())
      for (Operation &op : region.getOps())
        if (op.getNumRegions() != 0)
          workList.push_back(&op);
  } while (!workList.empty());
}

LogicalResult ReductionTreePass::reduceOp(ModuleOp module, Region &region) {
  Tester test(testerName, testerArgs);
  switch (traversalModeId) {
  case TraversalMode::SinglePath:
    return findOptimal<ReductionNode::iterator<TraversalMode::SinglePath>>(
        module, region, reducerPatterns, test);
  default:
    return module.emitError() << "unsupported traversal mode detected";
  }
}

std::unique_ptr<Pass> mlir::createReductionTreePass() {
  return std::make_unique<ReductionTreePass>();
}
