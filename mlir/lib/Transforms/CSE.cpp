//===- CSE.cpp - Common Sub-expression Elimination ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the CSE pass. The actual CSE algorithm lives in
// mlir/lib/Transforms/Utils/CSE.cpp so that it can be invoked from other
// utilities (e.g. the greedy pattern rewrite driver).
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/CSE.h"

#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DEF_CSEPASS
#include "mlir/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
/// CSE pass.
struct CSE : public impl::CSEPassBase<CSE> {
  void runOnOperation() override;
};
} // namespace

void CSE::runOnOperation() {
  IRRewriter rewriter(&getContext());
  auto &domInfo = getAnalysis<DominanceInfo>();
  bool changed = false;
  // `numCSE` / `numDCE` are `llvm::Statistic` objects, not raw `int64_t`, so
  // the public API's out-parameters cannot point at them directly.
  int64_t cseCount = 0;
  int64_t dceCount = 0;
  eliminateCommonSubExpressions(rewriter, domInfo, getOperation(), &changed,
                                &cseCount, &dceCount);

  numCSE = cseCount;
  numDCE = dceCount;

  // If there was no change to the IR, we mark all analyses as preserved.
  if (!changed)
    return markAllAnalysesPreserved();

  // We only delete redundant operations without moving any operation to a
  // different block, so the dominance tree structure remains unchanged and
  // DominanceInfo/PostDominanceInfo can be safely preserved.
  markAnalysesPreserved<DominanceInfo, PostDominanceInfo>();
}
