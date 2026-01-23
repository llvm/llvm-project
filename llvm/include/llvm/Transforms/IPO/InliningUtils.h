//===- InliningUtils.h - Shared inlining utilities -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines shared utilities used by the inliner passes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_IPO_INLININGUTILS_H
#define LLVM_TRANSFORMS_IPO_INLININGUTILS_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"

namespace llvm {

/// Check if Function F appears in the inline history chain.
/// InlineHistory is a vector of (Function, ParentHistoryID) pairs.
/// Returns true if F was already inlined in the chain leading to
/// InlineHistoryID.
inline bool inlineHistoryIncludes(
    Function *F, int InlineHistoryID,
    const SmallVectorImpl<std::pair<Function *, int>> &InlineHistory) {
  while (InlineHistoryID != -1) {
    assert(unsigned(InlineHistoryID) < InlineHistory.size() &&
           "Invalid inline history ID");
    if (InlineHistory[InlineHistoryID].first == F)
      return true;
    InlineHistoryID = InlineHistory[InlineHistoryID].second;
  }
  return false;
}

/// Flatten a function by inlining all calls recursively.
///
/// PolicyT must provide:
///   - bool canInlineCall(Function &F, CallBase &CB): Check if call can be
///       inlined into F
///   - bool doInline(Function &F, CallBase &CB, Function &Callee): Perform
///       the inline, return true on success
///   - ArrayRef<CallBase *> getNewCallSites(): Get call sites from last inline
///
/// Returns true if any inlining was performed.
template <typename PolicyT>
bool flattenFunction(Function &F, PolicyT &Policy,
                     OptimizationRemarkEmitter &ORE) {
  SmallVector<std::pair<CallBase *, int>, 16> Worklist;
  SmallVector<std::pair<Function *, int>, 16> InlineHistory;

  // Collect initial calls.
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      if (auto *CB = dyn_cast<CallBase>(&I)) {
        if (CB->getAttributes().hasFnAttr(Attribute::NoInline))
          continue;
        Function *Callee = CB->getCalledFunction();
        if (!Callee || Callee->isDeclaration())
          continue;
        Worklist.push_back({CB, -1});
      }
    }
  }

  bool Changed = false;
  while (!Worklist.empty()) {
    auto Item = Worklist.pop_back_val();
    CallBase *CB = Item.first;
    int InlineHistoryID = Item.second;
    Function *Callee = CB->getCalledFunction();
    if (!Callee)
      continue;

    // Detect recursion.
    if (Callee == &F ||
        inlineHistoryIncludes(Callee, InlineHistoryID, InlineHistory)) {
      ORE.emit([&]() {
        return OptimizationRemarkMissed("inline", "NotInlined",
                                        CB->getDebugLoc(), CB->getParent())
               << "'" << ore::NV("Callee", Callee) << "' is not inlined into '"
               << ore::NV("Caller", CB->getCaller())
               << "': recursive call during flattening";
      });
      continue;
    }

    if (!Policy.canInlineCall(F, *CB))
      continue;

    if (!Policy.doInline(F, *CB, *Callee))
      continue;

    Changed = true;

    // Add new call sites from the inlined function to the worklist.
    ArrayRef<CallBase *> NewCallSites = Policy.getNewCallSites();
    if (!NewCallSites.empty()) {
      int NewHistoryID = InlineHistory.size();
      InlineHistory.push_back({Callee, InlineHistoryID});
      for (CallBase *NewCB : NewCallSites) {
        Function *NewCallee = NewCB->getCalledFunction();
        if (NewCallee && !NewCallee->isDeclaration() &&
            !NewCB->getAttributes().hasFnAttr(Attribute::NoInline))
          Worklist.push_back({NewCB, NewHistoryID});
      }
    }
  }

  return Changed;
}

} // namespace llvm

#endif // LLVM_TRANSFORMS_IPO_INLININGUTILS_H
