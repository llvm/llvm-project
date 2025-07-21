//===-- SimplifySwitchVar.cpp - Switch Variable simplification ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// This file implements switch variable simplification. It looks for a
/// linear relationship between the case value of a switch and the constant
/// offset of an operation. Knowing this relationship, we can simplify
/// multiple individual operations into a single, more generic one, which
/// can help with further optimizations.
///
/// It is similar to SimplifyIndVar, but instead of looking at an
/// induction variable and modeling its scalar evolution over
/// multiple iterations, it analyzes the switch variable and
/// models how it affects constant offsets.
///
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/SimplifySwitchVar.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

/// Return the BB, where (most of) the cases meet.
/// In that BB are phi nodes, that contain the case BBs.
static BasicBlock *findMostCommonSuccessor(SwitchInst *Switch) {
  uint64_t Max = 0;
  BasicBlock *MostCommonSuccessor = nullptr;

  for (auto &Case : Switch->cases()) {
    auto *CaseBB = Case.getCaseSuccessor();
    auto GetNumPredecessors = [](BasicBlock *BB) -> uint64_t {
      return std::distance(predecessors(BB).begin(), predecessors(BB).end());
    };

    auto Length = GetNumPredecessors(CaseBB);

    if (Length > Max) {
      Max = Length;
      MostCommonSuccessor = CaseBB;
    }

    for (auto *Successor : successors(CaseBB)) {
      auto Length = GetNumPredecessors(Successor);
      if (Length > Max) {
        Max = Length;
        MostCommonSuccessor = Successor;
      }
    }
  }

  return MostCommonSuccessor;
}

PreservedAnalyses SimplifySwitchVarPass::run(Function &F,
                                             FunctionAnalysisManager &AM) {
  bool Changed = false;
  BasicBlock *MostCommonSuccessor;
  // collect switch insts
  for (auto &BB : F) {
    if (auto *Switch = dyn_cast<SwitchInst>(BB.getTerminator())) {
      // get the most common successor for the phi nodes
      MostCommonSuccessor = findMostCommonSuccessor(Switch);
    }
  }

  if (!Changed)
    return PreservedAnalyses::all();

  return PreservedAnalyses::allInSet<CFGAnalyses>();
}
