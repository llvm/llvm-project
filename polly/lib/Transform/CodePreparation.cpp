//===---- CodePreparation.cpp - Code preparation for Scop Detection -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The Polly code preparation pass is executed before SCoP detection. Its
// currently only splits the entry block of the SCoP to make room for alloc
// instructions as they are generated during code generation.
//
// XXX: In the future, we should remove the need for this pass entirely and
// instead add this spitting to the code generation pass.
//
//===----------------------------------------------------------------------===//

#include "polly/CodePreparation.h"
#include "polly/Support/ScopHelper.h"
#include "llvm/Analysis/DominanceFrontier.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/RegionInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"

using namespace llvm;
using namespace polly;

static bool runCodePreprationImpl(Function &F, DominatorTree *DT, LoopInfo *LI,
                                  RegionInfo *RI) {
  // Find first non-alloca instruction. Every basic block has a non-alloca
  // instruction, as every well formed basic block has a terminator.
  auto &EntryBlock = F.getEntryBlock();
  BasicBlock::iterator I = EntryBlock.begin();
  while (isa<AllocaInst>(I))
    ++I;

  // Abort if not necessary to split
  if (I->isTerminator() && isa<BranchInst>(I) &&
      cast<BranchInst>(I)->isUnconditional())
    return false;

  // splitBlock updates DT, LI and RI.
  splitEntryBlockForAlloca(&EntryBlock, DT, LI, RI);

  return true;
}

bool polly::runCodePreparation(Function &F, DominatorTree *DT, LoopInfo *LI,
                               RegionInfo *RI) {
  return runCodePreprationImpl(F, DT, LI, RI);
}
