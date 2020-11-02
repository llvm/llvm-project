//===- PatternApplicator.cpp - Pattern Application Engine -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements an applicator that applies pattern rewrites based upon a
// user defined cost model.
//
//===----------------------------------------------------------------------===//

#include "mlir/Rewrite/PatternApplicator.h"
#include "llvm/Support/Debug.h"

using namespace mlir;

#define DEBUG_TYPE "pattern-match"

void PatternApplicator::applyCostModel(CostModel model) {
  // Separate patterns by root kind to simplify lookup later on.
  patterns.clear();
  anyOpPatterns.clear();
  for (const auto &pat : frozenPatternList.getPatterns()) {
    // If the pattern is always impossible to match, just ignore it.
    if (pat.getBenefit().isImpossibleToMatch()) {
      LLVM_DEBUG({
        llvm::dbgs()
            << "Ignoring pattern '" << pat.getRootKind()
            << "' because it is impossible to match (by pattern benefit)\n";
      });
      continue;
    }
    if (Optional<OperationName> opName = pat.getRootKind())
      patterns[*opName].push_back(&pat);
    else
      anyOpPatterns.push_back(&pat);
  }

  // Sort the patterns using the provided cost model.
  llvm::SmallDenseMap<const Pattern *, PatternBenefit> benefits;
  auto cmp = [&benefits](const Pattern *lhs, const Pattern *rhs) {
    return benefits[lhs] > benefits[rhs];
  };
  auto processPatternList = [&](SmallVectorImpl<const RewritePattern *> &list) {
    // Special case for one pattern in the list, which is the most common case.
    if (list.size() == 1) {
      if (model(*list.front()).isImpossibleToMatch()) {
        LLVM_DEBUG({
          llvm::dbgs() << "Ignoring pattern '" << list.front()->getRootKind()
                       << "' because it is impossible to match or cannot lead "
                          "to legal IR (by cost model)\n";
        });
        list.clear();
      }
      return;
    }

    // Collect the dynamic benefits for the current pattern list.
    benefits.clear();
    for (const Pattern *pat : list)
      benefits.try_emplace(pat, model(*pat));

    // Sort patterns with highest benefit first, and remove those that are
    // impossible to match.
    std::stable_sort(list.begin(), list.end(), cmp);
    while (!list.empty() && benefits[list.back()].isImpossibleToMatch()) {
      LLVM_DEBUG({
        llvm::dbgs() << "Ignoring pattern '" << list.back()->getRootKind()
                     << "' because it is impossible to match or cannot lead to "
                        "legal IR (by cost model)\n";
      });
      list.pop_back();
    }
  };
  for (auto &it : patterns)
    processPatternList(it.second);
  processPatternList(anyOpPatterns);
}

void PatternApplicator::walkAllPatterns(
    function_ref<void(const Pattern &)> walk) {
  for (auto &it : frozenPatternList.getPatterns())
    walk(it);
}

LogicalResult PatternApplicator::matchAndRewrite(
    Operation *op, PatternRewriter &rewriter,
    function_ref<bool(const Pattern &)> canApply,
    function_ref<void(const Pattern &)> onFailure,
    function_ref<LogicalResult(const Pattern &)> onSuccess) {
  // Check to see if there are patterns matching this specific operation type.
  MutableArrayRef<const RewritePattern *> opPatterns;
  auto patternIt = patterns.find(op->getName());
  if (patternIt != patterns.end())
    opPatterns = patternIt->second;

  // Process the patterns for that match the specific operation type, and any
  // operation type in an interleaved fashion.
  // FIXME: It'd be nice to just write an llvm::make_merge_range utility
  // and pass in a comparison function. That would make this code trivial.
  auto opIt = opPatterns.begin(), opE = opPatterns.end();
  auto anyIt = anyOpPatterns.begin(), anyE = anyOpPatterns.end();
  while (opIt != opE && anyIt != anyE) {
    // Try to match the pattern providing the most benefit.
    const RewritePattern *pattern;
    if ((*opIt)->getBenefit() >= (*anyIt)->getBenefit())
      pattern = *(opIt++);
    else
      pattern = *(anyIt++);

    // Otherwise, try to match the generic pattern.
    if (succeeded(matchAndRewrite(op, *pattern, rewriter, canApply, onFailure,
                                  onSuccess)))
      return success();
  }
  // If we break from the loop, then only one of the ranges can still have
  // elements. Loop over both without checking given that we don't need to
  // interleave anymore.
  for (const RewritePattern *pattern : llvm::concat<const RewritePattern *>(
           llvm::make_range(opIt, opE), llvm::make_range(anyIt, anyE))) {
    if (succeeded(matchAndRewrite(op, *pattern, rewriter, canApply, onFailure,
                                  onSuccess)))
      return success();
  }
  return failure();
}

LogicalResult PatternApplicator::matchAndRewrite(
    Operation *op, const RewritePattern &pattern, PatternRewriter &rewriter,
    function_ref<bool(const Pattern &)> canApply,
    function_ref<void(const Pattern &)> onFailure,
    function_ref<LogicalResult(const Pattern &)> onSuccess) {
  // Check that the pattern can be applied.
  if (canApply && !canApply(pattern))
    return failure();

  // Try to match and rewrite this pattern. The patterns are sorted by
  // benefit, so if we match we can immediately rewrite.
  rewriter.setInsertionPoint(op);
  if (succeeded(pattern.matchAndRewrite(op, rewriter)))
    return success(!onSuccess || succeeded(onSuccess(pattern)));

  if (onFailure)
    onFailure(pattern);
  return failure();
}
