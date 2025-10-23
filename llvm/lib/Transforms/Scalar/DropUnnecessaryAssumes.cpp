//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/DropUnnecessaryAssumes.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Transforms/Utils/Local.h"

using namespace llvm;
using namespace llvm::PatternMatch;

PreservedAnalyses
DropUnnecessaryAssumesPass::run(Function &F, FunctionAnalysisManager &FAM) {
  AssumptionCache &AC = FAM.getResult<AssumptionAnalysis>(F);
  bool Changed = false;

  for (AssumptionCache::ResultElem &Elem : AC.assumptions()) {
    auto *Assume = cast_or_null<AssumeInst>(Elem.Assume);
    if (!Assume)
      continue;

    // TODO: Handle assumes with operand bundles.
    if (Assume->hasOperandBundles())
      continue;

    Value *Cond = Assume->getArgOperand(0);
    // Don't drop type tests, which have special semantics.
    if (match(Cond, m_Intrinsic<Intrinsic::type_test>()))
      continue;

    SmallPtrSet<Value *, 8> Affected;
    findValuesAffectedByCondition(Cond, /*IsAssume=*/true,
                                  [&](Value *A) { Affected.insert(A); });

    // If all the affected uses have only one use (part of the assume), then
    // the assume does not provide useful information. Note that additional
    // users may appear as a result of inlining and CSE, so we should only
    // make this assumption late in the optimization pipeline.
    // TODO: Handle dead cyclic usages.
    // TODO: Handle multiple dead assumes on the same value.
    if (!all_of(Affected, match_fn(m_OneUse(m_Value()))))
      continue;

    Assume->eraseFromParent();
    RecursivelyDeleteTriviallyDeadInstructions(Cond);
    Changed = true;
  }

  if (Changed) {
    PreservedAnalyses PA;
    PA.preserveSet<CFGAnalyses>();
    return PA;
  }
  return PreservedAnalyses::all();
}
