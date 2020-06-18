//===- PatternMatch.cpp - Base classes for pattern match ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

using namespace mlir;

PatternBenefit::PatternBenefit(unsigned benefit) : representation(benefit) {
  assert(representation == benefit && benefit != ImpossibleToMatchSentinel &&
         "This pattern match benefit is too large to represent");
}

unsigned short PatternBenefit::getBenefit() const {
  assert(!isImpossibleToMatch() && "Pattern doesn't match");
  return representation;
}

//===----------------------------------------------------------------------===//
// Pattern implementation
//===----------------------------------------------------------------------===//

Pattern::Pattern(StringRef rootName, PatternBenefit benefit,
                 MLIRContext *context)
    : rootKind(OperationName(rootName, context)), benefit(benefit) {}

// Out-of-line vtable anchor.
void Pattern::anchor() {}

//===----------------------------------------------------------------------===//
// RewritePattern and PatternRewriter implementation
//===----------------------------------------------------------------------===//

void RewritePattern::rewrite(Operation *op, PatternRewriter &rewriter) const {
  llvm_unreachable("need to implement either matchAndRewrite or one of the "
                   "rewrite functions!");
}

LogicalResult RewritePattern::match(Operation *op) const {
  llvm_unreachable("need to implement either match or matchAndRewrite!");
}

/// Patterns must specify the root operation name they match against, and can
/// also specify the benefit of the pattern matching. They can also specify the
/// names of operations that may be generated during a successful rewrite.
RewritePattern::RewritePattern(StringRef rootName,
                               ArrayRef<StringRef> generatedNames,
                               PatternBenefit benefit, MLIRContext *context)
    : Pattern(rootName, benefit, context) {
  generatedOps.reserve(generatedNames.size());
  std::transform(generatedNames.begin(), generatedNames.end(),
                 std::back_inserter(generatedOps), [context](StringRef name) {
                   return OperationName(name, context);
                 });
}

PatternRewriter::~PatternRewriter() {
  // Out of line to provide a vtable anchor for the class.
}

/// This method performs the final replacement for a pattern, where the
/// results of the operation are updated to use the specified list of SSA
/// values.
void PatternRewriter::replaceOp(Operation *op, ValueRange newValues) {
  // Notify the rewriter subclass that we're about to replace this root.
  notifyRootReplaced(op);

  assert(op->getNumResults() == newValues.size() &&
         "incorrect # of replacement values");
  op->replaceAllUsesWith(newValues);

  notifyOperationRemoved(op);
  op->erase();
}

/// This method erases an operation that is known to have no uses. The uses of
/// the given operation *must* be known to be dead.
void PatternRewriter::eraseOp(Operation *op) {
  assert(op->use_empty() && "expected 'op' to have no uses");
  notifyOperationRemoved(op);
  op->erase();
}

void PatternRewriter::eraseBlock(Block *block) {
  for (auto &op : llvm::make_early_inc_range(llvm::reverse(*block))) {
    assert(op.use_empty() && "expected 'op' to have no uses");
    eraseOp(&op);
  }
  block->erase();
}

/// Merge the operations of block 'source' into the end of block 'dest'.
/// 'source's predecessors must be empty or only contain 'dest`.
/// 'argValues' is used to replace the block arguments of 'source' after
/// merging.
void PatternRewriter::mergeBlocks(Block *source, Block *dest,
                                  ValueRange argValues) {
  assert(llvm::all_of(source->getPredecessors(),
                      [dest](Block *succ) { return succ == dest; }) &&
         "expected 'source' to have no predecessors or only 'dest'");
  assert(argValues.size() == source->getNumArguments() &&
         "incorrect # of argument replacement values");

  // Replace all of the successor arguments with the provided values.
  for (auto it : llvm::zip(source->getArguments(), argValues))
    std::get<0>(it).replaceAllUsesWith(std::get<1>(it));

  // Splice the operations of the 'source' block into the 'dest' block and erase
  // it.
  dest->getOperations().splice(dest->end(), source->getOperations());
  source->dropAllUses();
  source->erase();
}

/// Split the operations starting at "before" (inclusive) out of the given
/// block into a new block, and return it.
Block *PatternRewriter::splitBlock(Block *block, Block::iterator before) {
  return block->splitBlock(before);
}

/// 'op' and 'newOp' are known to have the same number of results, replace the
/// uses of op with uses of newOp
void PatternRewriter::replaceOpWithResultsOfAnotherOp(Operation *op,
                                                      Operation *newOp) {
  assert(op->getNumResults() == newOp->getNumResults() &&
         "replacement op doesn't match results of original op");
  if (op->getNumResults() == 1)
    return replaceOp(op, newOp->getResult(0));
  return replaceOp(op, newOp->getResults());
}

/// Move the blocks that belong to "region" before the given position in
/// another region.  The two regions must be different.  The caller is in
/// charge to update create the operation transferring the control flow to the
/// region and pass it the correct block arguments.
void PatternRewriter::inlineRegionBefore(Region &region, Region &parent,
                                         Region::iterator before) {
  parent.getBlocks().splice(before, region.getBlocks());
}
void PatternRewriter::inlineRegionBefore(Region &region, Block *before) {
  inlineRegionBefore(region, *before->getParent(), before->getIterator());
}

/// Clone the blocks that belong to "region" before the given position in
/// another region "parent". The two regions must be different. The caller is
/// responsible for creating or updating the operation transferring flow of
/// control to the region and passing it the correct block arguments.
void PatternRewriter::cloneRegionBefore(Region &region, Region &parent,
                                        Region::iterator before,
                                        BlockAndValueMapping &mapping) {
  region.cloneInto(&parent, before, mapping);
}
void PatternRewriter::cloneRegionBefore(Region &region, Region &parent,
                                        Region::iterator before) {
  BlockAndValueMapping mapping;
  cloneRegionBefore(region, parent, before, mapping);
}
void PatternRewriter::cloneRegionBefore(Region &region, Block *before) {
  cloneRegionBefore(region, *before->getParent(), before->getIterator());
}

//===----------------------------------------------------------------------===//
// PatternMatcher implementation
//===----------------------------------------------------------------------===//

void PatternApplicator::applyCostModel(CostModel model) {
  // Separate patterns by root kind to simplify lookup later on.
  patterns.clear();
  for (const auto &pat : owningPatternList)
    patterns[pat->getRootKind()].push_back(pat.get());

  // Sort the patterns using the provided cost model.
  llvm::SmallDenseMap<RewritePattern *, PatternBenefit> benefits;
  auto cmp = [&benefits](RewritePattern *lhs, RewritePattern *rhs) {
    return benefits[lhs] > benefits[rhs];
  };
  for (auto &it : patterns) {
    SmallVectorImpl<RewritePattern *> &list = it.second;

    // Special case for one pattern in the list, which is the most common case.
    if (list.size() == 1) {
      if (model(*list.front()).isImpossibleToMatch())
        list.clear();
      continue;
    }

    // Collect the dynamic benefits for the current pattern list.
    benefits.clear();
    for (RewritePattern *pat : list)
      benefits.try_emplace(pat, model(*pat));

    // Sort patterns with highest benefit first, and remove those that are
    // impossible to match.
    std::stable_sort(list.begin(), list.end(), cmp);
    while (!list.empty() && benefits[list.back()].isImpossibleToMatch())
      list.pop_back();
  }
}

void PatternApplicator::walkAllPatterns(
    function_ref<void(const RewritePattern &)> walk) {
  for (auto &it : owningPatternList)
    walk(*it);
}

/// Try to match the given operation to a pattern and rewrite it.
LogicalResult PatternApplicator::matchAndRewrite(
    Operation *op, PatternRewriter &rewriter,
    function_ref<bool(const RewritePattern &)> canApply,
    function_ref<void(const RewritePattern &)> onFailure,
    function_ref<LogicalResult(const RewritePattern &)> onSuccess) {
  auto patternIt = patterns.find(op->getName());
  if (patternIt == patterns.end())
    return failure();

  for (auto *pattern : patternIt->second) {
    // Check that the pattern can be applied.
    if (canApply && !canApply(*pattern))
      continue;

    // Try to match and rewrite this pattern. The patterns are sorted by
    // benefit, so if we match we can immediately rewrite.
    rewriter.setInsertionPoint(op);
    if (succeeded(pattern->matchAndRewrite(op, rewriter))) {
      if (!onSuccess || succeeded(onSuccess(*pattern)))
        return success();
      continue;
    }

    if (onFailure)
      onFailure(*pattern);
  }
  return failure();
}
