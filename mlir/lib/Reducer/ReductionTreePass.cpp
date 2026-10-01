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
#include "mlir/Transforms/RegionUtils.h"

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
  size_t keepIndex = 0;
  for (const auto &op : enumerate(region.getOps())) {
    int index = op.index();
    if (keepIndex < rangeToKeep.size() &&
        index == rangeToKeep[keepIndex].second)
      ++keepIndex;
    if (keepIndex == rangeToKeep.size() || index < rangeToKeep[keepIndex].first)
      opsNotInRange.push_back(&op.value());
  }

  // `applyOpPatternsGreedily` with folding may erase the ops so we can't do the
  // pattern matching in above iteration. Besides, erase op not-in-range may end
  // up in invalid module, so `applyOpPatternsGreedily` with folding should come
  // before that transform.
  if (!eraseOpNotInRange)
    for (Operation *op : opsNotInRange) {
      // `applyOpPatternsGreedily` with folding returns whether the op is
      // converted. Omit it because we don't have expectation this reduction
      // will be success or not.
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
    return module.emitError() << "uninterested module will not be reduced";

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
    return module.emitError() << "uninterested module will not be reduced";
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

/// Searches for an unvisited branch terminator within the given region based on
/// the specified conditionality. This helper scans blocks in the \p region to
/// find a terminator that has not yet been processed (not in \p visited). If
/// \p isConditional is true, it looks for terminators with multiple successors
/// (e.g., cf.cond_br). Otherwise, it looks for single-successor terminators
/// (e.g., cf.br).
static Operation *getBranchTerminatorInRegion(Region &region,
                                              DenseSet<Operation *> &visited,
                                              bool isConditional = true) {
  auto it = llvm::find_if(region.getBlocks(), [&](Block &block) {
    if (!block.mightHaveTerminator())
      return false;
    size_t numSucc = block.getNumSuccessors();
    Operation *term = block.getTerminator();
    return !visited.contains(term) &&
           (isConditional ? numSucc > 1 : numSucc == 1);
  });
  return it != region.end() ? it->getTerminator() : nullptr;
}

/// Prunes unreachable blocks from the CFG using the \p worklist. This function
/// iteratively removes blocks that have no predecessors. When a block is
/// erased, its successors are added to the worklist as they may consequently
/// become unreachable. This ensures a cascading deletion of dead-end paths in
/// the control flow graph.
static void pruneCFGEdges(SetVector<Block *> &workList, IRRewriter &rewriter) {
  while (!workList.empty()) {
    Block *b = workList.front();
    workList.erase(workList.begin());
    if (b->hasNoPredecessors()) {
      for (Block *it : b->getSuccessors())
        workList.insert(it);
      rewriter.eraseBlock(b);
    }
  }
}

/// Reduces the control flow in a region by iteratively forcing branching
/// terminators to point to a single successor. It evaluates each potential
/// branch path and commits the reduction that results in the smallest
/// "interesting" module.
static LogicalResult reduceConditionalsInRegion(ModuleOp module, Region &region,
                                                const Tester &test) {
  std::pair<Tester::Interestingness, size_t> initStatus =
      test.isInteresting(module);

  if (initStatus.first != Tester::Interestingness::True)
    return module.emitWarning() << "uninterested module will not be reduced";
  llvm::SpecificBumpPtrAllocator<ReductionNode> allocator;

  ReductionNode *smallestNode = nullptr;
  mlir::IRRewriter rewriter(region.getContext());
  DenseSet<Operation *> visited;

  // This loop attempts to convert conditional branch operations into
  // unconditional ones.
  while (Operation *branchTerminator =
             getBranchTerminatorInRegion(region, visited)) {
    size_t numSuccessor = branchTerminator->getNumSuccessors();
    std::vector<ReductionNode::Range> ranges{
        {0, std::distance(region.op_begin(), region.op_end())}};
    // Iterate through each successor of the branching terminator to try
    // reducing the control flow to a single-path execution.
    int branchIdx = -1;
    for (int i = 0, e = numSuccessor; i < e; ++i) {
      // We allocate memory on the heap because the object will be assigned to
      // 'smallestNode'.
      ReductionNode *root = allocator.Allocate();
      new (root) ReductionNode(nullptr, ranges, allocator);
      mlir::IRMapping mapper;
      if (failed(root->initialize(module, region, mapper)))
        llvm_unreachable("unexpected initialization failure");

      Operation *tergetTerminator = mapper.lookup(branchTerminator);
      Block *selectedBlock = tergetTerminator->getSuccessor(i);
      auto branchOp = cast<BranchOpInterface>(tergetTerminator);
      mlir::SuccessorOperands selectedBlockOperands =
          branchOp.getSuccessorOperands(i);
      rewriter.setInsertionPointAfter(tergetTerminator);
      cf::BranchOp::create(rewriter, tergetTerminator->getLoc(), selectedBlock,
                           selectedBlockOperands.getForwardedOperands());
      auto succs = llvm::to_vector(tergetTerminator->getSuccessors());
      succs.erase(succs.begin() + i);
      SetVector<Block *> workList(succs.begin(), succs.end());
      rewriter.eraseOp(tergetTerminator);
      pruneCFGEdges(workList, rewriter);
      root->update(test.isInteresting(root->getModule()));
      if (root->isInteresting() == Tester::Interestingness::True &&
          (smallestNode == nullptr ||
           root->getSize() < smallestNode->getSize())) {
        smallestNode = root;
        branchIdx = i;
      }
    }

    if (branchIdx != -1) {
      Block *selectedBlock = branchTerminator->getSuccessor(branchIdx);
      auto branchOp = cast<BranchOpInterface>(branchTerminator);
      mlir::SuccessorOperands selectedBlockOperands =
          branchOp.getSuccessorOperands(branchIdx);
      rewriter.setInsertionPointAfter(branchTerminator);
      cf::BranchOp::create(rewriter, branchTerminator->getLoc(), selectedBlock,
                           selectedBlockOperands.getForwardedOperands());

      auto succs = llvm::to_vector(branchOp->getSuccessors());
      succs.erase(succs.begin() + branchIdx);
      SetVector<Block *> workList(succs.begin(), succs.end());
      rewriter.eraseOp(branchOp);
      pruneCFGEdges(workList, rewriter);
    } else {
      // Insert 'branchTerminator' into visited to prevent it from being
      // processed again.
      visited.insert(branchTerminator);
    }
  }
  return success();
}

/// Simplifies the Control Flow Graph (CFG) by merging blocks that have a
/// single-successor / single-predecessor relationship. This function leverages
/// the canonicalization patterns of 'cf.br' to perform the merge
static LogicalResult reduceBlockMergeInRegion(ModuleOp module, Region &region,
                                              const Tester &test) {
  std::pair<Tester::Interestingness, size_t> initStatus =
      test.isInteresting(module);

  if (initStatus.first != Tester::Interestingness::True)
    return module.emitWarning() << "uninterested module will not be reduced";
  llvm::SpecificBumpPtrAllocator<ReductionNode> allocator;

  GreedyRewriteConfig config;
  auto context = region.getContext();
  RewritePatternSet patterns(context);
  cf::BranchOp::getCanonicalizationPatterns(patterns, context);
  FrozenRewritePatternSet fPatterns = std::move(patterns);

  mlir::IRRewriter rewriter(context);
  DenseSet<Operation *> visited;
  while (Operation *branchTerminator =
             getBranchTerminatorInRegion(region, visited, false)) {
    std::vector<ReductionNode::Range> ranges{
        {0, std::distance(region.op_begin(), region.op_end())}};
    ReductionNode *root = allocator.Allocate();
    new (root) ReductionNode(nullptr, ranges, allocator);
    mlir::IRMapping mapper;
    if (failed(root->initialize(module, region, mapper)))
      llvm_unreachable("unexpected initialization failure");
    Operation *tergetTerminator = mapper.lookup(branchTerminator);
    bool changed = false;
    (void)applyOpPatternsGreedily(tergetTerminator, fPatterns, config,
                                  &changed);
    root->update(test.isInteresting(root->getModule()));

    // If the changed variable is false, it indicates that the pattern failed to
    // apply. We should insert it into visited to prevent it from being
    // processed again.
    if (changed && root->isInteresting() == Tester::Interestingness::True)
      (void)applyOpPatternsGreedily(branchTerminator, fPatterns, config);
    else
      visited.insert(branchTerminator);
  }
  return success();
}

static LogicalResult eraseRedundantBlocksInRegion(ModuleOp module,
                                                  Region &region,
                                                  const Tester &test) {
  /// We separate the reduction control flow graph process into 2 steps.

  // we attempts to simplify conditional branches into unconditional ones by
  // picking the "interesting" path.
  (void)reduceConditionalsInRegion(module, region, test);

  // We merge redundant blocks that have single-successor/single-predecessor
  // relationships using canonicalization patterns.
  (void)reduceBlockMergeInRegion(module, region, test);
  return success();
}

template <typename IteratorType>
static LogicalResult findOptimal(ModuleOp module, Region &region,
                                 const FrozenRewritePatternSet &patterns,
                                 const Tester &test) {
  // We separate the reduction process into 4 steps, the first one is to erase
  // redundant operations and the second one is to apply the reducer patterns.

  // In the first phase, we attempt to erase all operations within the entire
  // region.
  if (succeeded(eraseAllOpsInRegion(module, region, test)))
    return success();

  // In the second phase, we attempt to eliminate redundant blocks. This reduces
  // the program's execution paths.
  (void)eraseRedundantBlocksInRegion(module, region, test);

  // In the third phase, we don't apply any patterns so that we only select the
  // range of operations to keep to the module stay interesting.
  if (failed(findOptimal<IteratorType>(module, region, /*patterns=*/{}, test,
                                       /*eraseOpNotInRange=*/true)))
    return failure();
  // In the fourth phase, we suppose that no operation is redundant, so we try
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
