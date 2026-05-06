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
#include "mlir/Reducer/Passes.h"
#include "mlir/Reducer/ReductionNode.h"
#include "mlir/Reducer/ReductionPatternInterface.h"
#include "mlir/Reducer/Tester.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Allocator.h"

namespace mlir {
#define GEN_PASS_DEF_REDUCTIONTREEPASS
#include "mlir/Reducer/Passes.h.inc"
} // namespace mlir

using namespace mlir;

/// We will apply `applyFn` to the operations in the ranges specified by
/// ReductionNode.
template <typename IteratorType, typename ApplyFn>
static LogicalResult findOptimalUsing(ModuleOp module, Region &region,
                                      const Tester &test, ApplyFn applyFn) {

  llvm::SpecificBumpPtrAllocator<ReductionNode> allocator;

  std::vector<ReductionNode::Range> ranges{
      {0, std::distance(region.op_begin(), region.op_end())}};

  ReductionNode *root = allocator.Allocate();
  new (root) ReductionNode(nullptr, ranges, allocator);
  // Duplicate the module for root node and locate the region in the copy.
  if (failed(root->initialize(module, region)))
    llvm_unreachable("unexpected initialization failure");

  // Create a duplicate of the root node as the first variant of the root.
  // This will keep the root intact when applying `applyFn`.
  ReductionNode *firstVariant = allocator.Allocate();
  new (firstVariant) ReductionNode(root, ranges, allocator);
  root->addVariant(firstVariant);

  ReductionNode *smallestNode = root;
  IteratorType iter(firstVariant);

  while (iter != IteratorType::end()) {
    ReductionNode &currentNode = *iter;

    applyFn(currentNode.getRegion(), currentNode.getRanges());
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
  while (curNode != root) {
    trace.push_back(curNode);
    curNode = curNode->getParent();
  }

  if (trace.empty())
    // If trace is empty, then the smallestNode == root and therefore we were
    // not successful in reducing the module
    return failure();

  // Reduce the region through the optimal path.
  while (!trace.empty()) {
    ReductionNode *top = trace.pop_back_val();
    applyFn(region, top->getStartRanges());
  }

  std::pair<Tester::Interestingness, size_t> finalStatus =
      test.isInteresting(module);

  if (finalStatus.first != Tester::Interestingness::True)
    llvm::report_fatal_error("Reduced module is not interesting");
  if (finalStatus.second != smallestNode->getSize())
    llvm::report_fatal_error(
        "Reduced module doesn't have consistent size with smallestNode");
  return success();
}

/// We implicitly number each operation in the region and if an operation's
/// number falls into rangeToKeep, we'll keep it.
static void eliminateOperations(Region &region,
                                ArrayRef<ReductionNode::Range> rangeToKeep) {
  std::vector<Operation *> opsNotInRange;
  size_t keepIndex = 0;
  for (const auto &op : enumerate(region.getOps())) {
    int index = op.index();
    if (keepIndex < rangeToKeep.size() &&
        index == rangeToKeep[keepIndex].second)
      ++keepIndex;
    if (keepIndex == rangeToKeep.size() || index < rangeToKeep[keepIndex].first)
      opsNotInRange.push_back(&op.value());
  }

  for (Operation *op : opsNotInRange) {
    op->dropAllUses();
    op->erase();
  }
}

/// We implicitly number each operation in the region and if an operation's
/// number falls into rangeToApply, we'll apply the given rewrite patterns on
/// it.
static void applyPatterns(Region &region,
                          const FrozenRewritePatternSet &patterns,
                          ArrayRef<ReductionNode::Range> rangeToApply) {
  size_t rangeIndex = 0;
  std::vector<Operation *> opsInRange;
  for (const auto &op : enumerate(region.getOps())) {
    int index = op.index();
    if (rangeIndex < rangeToApply.size() &&
        index == rangeToApply[rangeIndex].second)
      ++rangeIndex;
    if (rangeIndex < rangeToApply.size() &&
        index >= rangeToApply[rangeIndex].first)
      opsInRange.push_back(&op.value());
  }

  for (auto *op : opsInRange)
    // `applyOpPatternsGreedily` with folding returns whether the op is
    // converted. Omit it because we don't have expectation this reduction
    // will be success or not.
    (void)applyOpPatternsGreedily(op, patterns,
                                  GreedyRewriteConfig().setStrictness(
                                      GreedyRewriteStrictness::ExistingOps));
}

template <typename IteratorType>
static LogicalResult findOptimal(ModuleOp module, Region &region,
                                 const FrozenRewritePatternSet &patterns,
                                 const Tester &test) {

  // We first test the interstingness of the module passed to findOptimal.
  std::pair<Tester::Interestingness, size_t> initStatus =
      test.isInteresting(module);
  if (initStatus.first != Tester::Interestingness::True)
    // If the module is not interesting, we can return failure
    return module.emitError() << "uninterested module will not be reduced";

  // We separate the reduction process into 3 steps:
  // In the first step, we attempt to erase all operations within the
  // entire region.
  if (succeeded(findOptimalUsing<IteratorType>(
          module, region, test, [](auto &region, auto) {
            for (auto &block : region.getBlocks())
              block.clear();
          })))
    // If clearing the entire region kept the module interesting
    // we will return success.
    return success();

  // In the second step, we eliminate redundant operations from the region to
  // select those that keep the module interesting
  auto eliminationResult =
      findOptimalUsing<IteratorType>(module, region, test, eliminateOperations);

  // In the third step, we suppose that no operation is redundant, so we try
  // to rewrite the operation into simpler form by applying patterns.
  auto applyPatternsResult = findOptimalUsing<IteratorType>(
      module, region, test, [&](auto &region, auto ranges) {
        applyPatterns(region, patterns, ranges);
      });

  if (succeeded(eliminationResult) || succeeded(applyPatternsResult))
    // if step 2 or 3 was successful, then we return success.
    return success();
  return failure();
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
  void populateReductionPatterns(RewritePatternSet &pattern,
                                 Tester &tester) const {
    for (const DialectReductionPatternInterface &interface : *this) {
      interface.populateReductionPatterns(pattern);
      interface.populateReductionPatternsWithTester(pattern, tester);
    }
  }
};

//===----------------------------------------------------------------------===//
// ReductionTreePass
//===----------------------------------------------------------------------===//

/// This class defines the Reduction Tree Pass. It provides a framework to
/// to implement a reduction pass using a tree structure to keep track of the
/// generated reduced variants.
class ReductionTreePass
    : public impl::ReductionTreePassBase<ReductionTreePass> {
public:
  using Base::Base;

  LogicalResult initialize(MLIRContext *context) override;

  /// Runs the pass instance in the pass pipeline.
  void runOnOperation() override;

private:
  LogicalResult reduceOp(ModuleOp module, Region &region);

  Tester tester;
  FrozenRewritePatternSet reducerPatterns;
};

} // namespace

LogicalResult ReductionTreePass::initialize(MLIRContext *context) {
  tester.setTestScript(testerName);
  tester.setTestScriptArgs(testerArgs);

  RewritePatternSet patterns(context);

  ReductionPatternInterfaceCollection reducePatternCollection(context);
  reducePatternCollection.populateReductionPatterns(patterns, tester);

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
  switch (traversalModeId) {
  case TraversalMode::SinglePath:
    return findOptimal<ReductionNode::iterator<TraversalMode::SinglePath>>(
        module, region, reducerPatterns, tester);
  default:
    return module.emitError() << "unsupported traversal mode detected";
  }
}
