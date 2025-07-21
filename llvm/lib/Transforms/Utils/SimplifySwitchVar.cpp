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

using namespace llvm;

PreservedAnalyses SimplifySwitchVarPass::run(Function &F,
                                             FunctionAnalysisManager &AM) {
  bool Changed = false;
  if (!Changed)
    return PreservedAnalyses::all();

  return PreservedAnalyses::allInSet<CFGAnalyses>();
}
