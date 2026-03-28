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

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/DialectInterface.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Reducer/Passes.h"
#include "mlir/Reducer/ReductionNode.h"
#include "mlir/Reducer/ReductionPatternInterface.h"
#include "mlir/Reducer/Tester.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "reduction-tree"

namespace mlir {
#define GEN_PASS_DEF_REDUCTIONTREEPASS
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

  // `applyOpPatternsGreedily` with folding may erase the ops so we can't do the
  // pattern matching in above iteration. Besides, erase op not-in-range may end
  // up in invalid module, so `applyOpPatternsGreedily` with folding should come
  // before that transform.
  for (Operation *op : opsInRange) {
    // `applyOpPatternsGreedily` with folding returns whether the op is
    // converted. Omit it because we don't have expectation this reduction will
    // be success or not.
    (void)applyOpPatternsGreedily(op, patterns,
                                  GreedyRewriteConfig().setStrictness(
                                      GreedyRewriteStrictness::ExistingOps));
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

/// This function attempts to erase all operations within the region currently
/// being processed.
static LogicalResult eraseAllOpsInRegion(ModuleOp module, Region &region,
                                         const Tester &test) {
  std::pair<Tester::Interestingness, size_t> initStatus =
      test.isInteresting(module);

  // While exploring the reduction tree, we always branch from an interesting
  // node. Thus the root node must be interesting.
  if (initStatus.first != Tester::Interestingness::True)
    return module.emitWarning() << "uninterested module will not be reduced";
  llvm::SpecificBumpPtrAllocator<ReductionNode> allocator;

  // Setting the ranges to {{0, 0}} will result in the deletion of all ops
  // within the region.
  std::vector<ReductionNode::Range> ranges{{0, 0}};

  // We allocate memory on the stack, and the 'allocator' is only used to
  // construct the 'root node'. Since we won't be constructing any child nodes
  // for emptyRegionNode, it is only used within the current scope.
  ReductionNode emptyRegionNode(nullptr, ranges, allocator);
  ReductionNode *root = &emptyRegionNode;

  // Create a copy of the current IR.
  if (failed(root->initialize(module, region)))
    llvm_unreachable("unexpected initialization failure");

  // Erase all operations within the corresponding region of the clone.
  applyPatterns(root->getRegion(), {}, root->getRanges(), true);
  root->update(test.isInteresting(root->getModule()));
  if (root->isInteresting() == Tester::Interestingness::True) {
    // If we can successfully remove all ops in the region, we apply the same
    // transformation to the original IR and return success.
    applyPatterns(region, {}, root->getRanges(), true);
    return success();
  }
  return failure();
}

// Returns the first branching terminator (cond_br, switch, etc.) found in the
// region.
static Operation *getBranchTerminatorInRegion(Region &region) {
  for (Block &block : region.getBlocks()) {
    if (block.getNumSuccessors() > 1)
      return block.getTerminator();
  }
  return {};
}

/// Reduces the control flow in a region by iteratively forcing branching
/// terminators to point to a single successor. It evaluates each potential
/// branch path and commits the reduction that results in the smallest
/// "interesting" module.
static LogicalResult eraseRedundantBlocksInRegion(ModuleOp module,
                                                  Region &region,
                                                  const Tester &test) {
  std::pair<Tester::Interestingness, size_t> initStatus =
      test.isInteresting(module);

  // While exploring the reduction tree, we always branch from an interesting
  // node. Thus the root node must be interesting.
  if (initStatus.first != Tester::Interestingness::True)
    return module.emitWarning() << "uninterested module will not be reduced";
  llvm::SpecificBumpPtrAllocator<ReductionNode> allocator;

  // We set the simplification level to Aggressive to enable block merging.
  GreedyRewriteConfig config;
  config.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Aggressive);
  config.setUseTopDownTraversal(true);

  // Populate canonicalization patterns for cf ops. When all targets of a
  // 'cf.cond_br' or 'cf.switch' point to the same block, they will be
  // canonicalized into a 'cf.br'.
  auto context = region.getContext();
  RewritePatternSet patterns(context);
  cf::BranchOp::getCanonicalizationPatterns(patterns, context);
  cf::CondBranchOp::getCanonicalizationPatterns(patterns, context);
  cf::SwitchOp::getCanonicalizationPatterns(patterns, context);
  FrozenRewritePatternSet fPatterns = std::move(patterns);

  ReductionNode *smallestNode = nullptr;
  mlir::OpBuilder b(context);
  while (Operation *branchTerminator = getBranchTerminatorInRegion(region)) {
    size_t numSuccessor = branchTerminator->getNumSuccessors();
    // We allocate memory on the heap because the object will be assigned to
    // 'smallestNode'.
    ReductionNode *root = allocator.Allocate();
    std::vector<ReductionNode::Range> ranges{
        {0, std::distance(region.op_begin(), region.op_end())}};

    // Iterate through each successor of the branching terminator to try
    // reducing the control flow to a single-path execution.
    int branchIdx = -1;
    for (int i = 0, e = numSuccessor; i < e; ++i) {
      new (root) ReductionNode(nullptr, ranges, allocator);
      mlir::IRMapping mapper;
      if (failed(root->initialize(module, region, mapper)))
        llvm_unreachable("unexpected initialization failure");
      Operation *tergetTerminator = mapper.lookup(branchTerminator);
      Block *selectedBlock = tergetTerminator->getSuccessor(i);
      auto branchOp = cast<BranchOpInterface>(tergetTerminator);
      ValueRange selectedBlockOperands =
          branchOp.getSuccessorForwardOperands(selectedBlock);
      b.setInsertionPointAfter(tergetTerminator);
      cf::BranchOp::create(b, tergetTerminator->getLoc(), selectedBlock,
                           selectedBlockOperands);
      tergetTerminator->erase();

      // Apply canonicalization patterns to collapse the now-redundant branches
      (void)applyPatternsGreedily(root->getRegion().getParentOp(), fPatterns,
                                  config);
      root->update(test.isInteresting(root->getModule()));

      // Track the smallest "interesting" version of the IR found so far.
      if (root->isInteresting() == Tester::Interestingness::True &&
          (smallestNode == nullptr ||
           root->getSize() < smallestNode->getSize())) {
        smallestNode = root;
        branchIdx = i;
      }
    }

    // If an interesting reduced branch was found, commit the change to the
    // original region and re-apply patterns for a final cleanup.
    if (branchIdx != -1) {
      Block *selectedBlock = branchTerminator->getSuccessor(branchIdx);
      auto branchOp = cast<BranchOpInterface>(branchTerminator);
      ValueRange selectedBlockOperands =
          branchOp.getSuccessorForwardOperands(selectedBlock);
      b.setInsertionPointAfter(branchTerminator);
      cf::BranchOp::create(b, branchTerminator->getLoc(), selectedBlock,
                           selectedBlockOperands);
      branchTerminator->erase();
      (void)applyPatternsGreedily(region.getParentOp(), fPatterns, config);
    }
  }

  // If no branching terminators were found (skipping the while loop),
  // there might still be opportunities for linear block merging or
  // We apply patterns here as a final cleanup to ensure the region is fully
  // simplified.
  if (smallestNode == nullptr)
    (void)applyPatternsGreedily(region.getParentOp(), fPatterns, config);
  return success();
}

template <typename IteratorType>
static LogicalResult findOptimal(ModuleOp module, Region &region,
                                 const FrozenRewritePatternSet &patterns,
                                 const Tester &test) {
  // We separate the reduction process into 3 steps, the first one is to erase
  // redundant operations and the second one is to apply the reducer patterns.

  // In the first phase, we attempt to erase all operations within the entire
  // region.
  if (succeeded(eraseAllOpsInRegion(module, region, test)))
    return success();

  (void)eraseRedundantBlocksInRegion(module, region, test);

  // In the second phase, we don't apply any patterns so that we only select the
  // range of operations to keep to the module stay interesting.
  if (failed(findOptimal<IteratorType>(module, region, /*patterns=*/{}, test,
                                       /*eraseOpNotInRange=*/true)))
    return failure();
  // In the third phase, we suppose that no operation is redundant, so we try
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
