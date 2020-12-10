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
#include "ByteCode.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::detail;

PatternApplicator::PatternApplicator(
    const FrozenRewritePatternList &frozenPatternList)
    : frozenPatternList(frozenPatternList) {
  if (const PDLByteCode *bytecode = frozenPatternList.getPDLByteCode()) {
    mutableByteCodeState = std::make_unique<PDLByteCodeMutableState>();
    bytecode->initializeMutableState(*mutableByteCodeState);
  }
}
PatternApplicator::~PatternApplicator() {}

#define DEBUG_TYPE "pattern-match"

void PatternApplicator::applyCostModel(CostModel model) {
  // Apply the cost model to the bytecode patterns first, and then the native
  // patterns.
  if (const PDLByteCode *bytecode = frozenPatternList.getPDLByteCode()) {
    for (auto it : llvm::enumerate(bytecode->getPatterns()))
      mutableByteCodeState->updatePatternBenefit(it.index(), model(it.value()));
  }

  // Separate patterns by root kind to simplify lookup later on.
  patterns.clear();
  anyOpPatterns.clear();
  for (const auto &pat : frozenPatternList.getNativePatterns()) {
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
  for (const Pattern &it : frozenPatternList.getNativePatterns())
    walk(it);
  if (const PDLByteCode *bytecode = frozenPatternList.getPDLByteCode()) {
    for (const Pattern &it : bytecode->getPatterns())
      walk(it);
  }
}

LogicalResult PatternApplicator::matchAndRewrite(
    Operation *op, PatternRewriter &rewriter,
    function_ref<bool(const Pattern &)> canApply,
    function_ref<void(const Pattern &)> onFailure,
    function_ref<LogicalResult(const Pattern &)> onSuccess) {
  // Before checking native patterns, first match against the bytecode. This
  // won't automatically perform any rewrites so there is no need to worry about
  // conflicts.
  SmallVector<PDLByteCode::MatchResult, 4> pdlMatches;
  const PDLByteCode *bytecode = frozenPatternList.getPDLByteCode();
  if (bytecode)
    bytecode->match(op, rewriter, pdlMatches, *mutableByteCodeState);

  // Check to see if there are patterns matching this specific operation type.
  MutableArrayRef<const RewritePattern *> opPatterns;
  auto patternIt = patterns.find(op->getName());
  if (patternIt != patterns.end())
    opPatterns = patternIt->second;

  // Process the patterns for that match the specific operation type, and any
  // operation type in an interleaved fashion.
  auto opIt = opPatterns.begin(), opE = opPatterns.end();
  auto anyIt = anyOpPatterns.begin(), anyE = anyOpPatterns.end();
  auto pdlIt = pdlMatches.begin(), pdlE = pdlMatches.end();
  while (true) {
    // Find the next pattern with the highest benefit.
    const Pattern *bestPattern = nullptr;
    const PDLByteCode::MatchResult *pdlMatch = nullptr;
    /// Operation specific patterns.
    if (opIt != opE)
      bestPattern = *(opIt++);
    /// Operation agnostic patterns.
    if (anyIt != anyE &&
        (!bestPattern || bestPattern->getBenefit() < (*anyIt)->getBenefit()))
      bestPattern = *(anyIt++);
    /// PDL patterns.
    if (pdlIt != pdlE &&
        (!bestPattern || bestPattern->getBenefit() < pdlIt->benefit)) {
      pdlMatch = pdlIt;
      bestPattern = (pdlIt++)->pattern;
    }
    if (!bestPattern)
      break;

    // Check that the pattern can be applied.
    if (canApply && !canApply(*bestPattern))
      continue;

    // Try to match and rewrite this pattern. The patterns are sorted by
    // benefit, so if we match we can immediately rewrite. For PDL patterns, the
    // match has already been performed, we just need to rewrite.
    rewriter.setInsertionPoint(op);
    LogicalResult result = success();
    if (pdlMatch) {
      bytecode->rewrite(rewriter, *pdlMatch, *mutableByteCodeState);
    } else {
      result = static_cast<const RewritePattern *>(bestPattern)
                   ->matchAndRewrite(op, rewriter);
    }
    if (succeeded(result) && (!onSuccess || succeeded(onSuccess(*bestPattern))))
      return success();

    // Perform any necessary cleanups.
    if (onFailure)
      onFailure(*bestPattern);
  }
  return failure();
}
