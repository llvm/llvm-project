//===- LoopExtractor.cpp - Extract each loop into a new function ----------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// A pass wrapper around the ExtractLoop() scalar transformation to extract each
// top-level loop into its own new function. If the loop is the ONLY loop in a
// given function, it is not touched. This is a pass most useful for debugging
// via bugpoint.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO.h"
#include "llvm/iTerminators.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/FunctionUtils.h"
#include "Support/Statistic.h"
using namespace llvm;

namespace {
  Statistic<> NumExtracted("loop-extract", "Number of loops extracted");
  
  // FIXME: This is not a function pass, but the PassManager doesn't allow
  // Module passes to require FunctionPasses, so we can't get loop info if we're
  // not a function pass.
  struct LoopExtractor : public FunctionPass {
    unsigned NumLoops;

    LoopExtractor(unsigned numLoops = ~0) : NumLoops(numLoops) {}

    virtual bool runOnFunction(Function &F);
    
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequiredID(BreakCriticalEdgesID);
      AU.addRequiredID(LoopSimplifyID);
      AU.addRequired<DominatorSet>();
      AU.addRequired<LoopInfo>();
    }
  };

  RegisterOpt<LoopExtractor> 
  X("loop-extract", "Extract loops into new functions");

  /// SingleLoopExtractor - For bugpoint.
  struct SingleLoopExtractor : public LoopExtractor {
    SingleLoopExtractor() : LoopExtractor(1) {}
  };

  RegisterOpt<SingleLoopExtractor> 
  Y("loop-extract-single", "Extract at most one loop into a new function");
} // End anonymous namespace 

bool LoopExtractor::runOnFunction(Function &F) {
  LoopInfo &LI = getAnalysis<LoopInfo>();

  // If this function has no loops, there is nothing to do.
  if (LI.begin() == LI.end())
    return false;

  DominatorSet &DS = getAnalysis<DominatorSet>();

  // If there is more than one top-level loop in this function, extract all of
  // the loops.
  bool Changed = false;
  if (LI.end()-LI.begin() > 1) {
    for (LoopInfo::iterator i = LI.begin(), e = LI.end(); i != e; ++i) {
      if (NumLoops == 0) return Changed;
      --NumLoops;
      Changed |= ExtractLoop(DS, *i) != 0;
      ++NumExtracted;
    }
  } else {
    // Otherwise there is exactly one top-level loop.  If this function is more
    // than a minimal wrapper around the loop, extract the loop.
    Loop *TLL = *LI.begin();
    bool ShouldExtractLoop = false;
    
    // Extract the loop if the entry block doesn't branch to the loop header.
    TerminatorInst *EntryTI = F.getEntryBlock().getTerminator();
    if (!isa<BranchInst>(EntryTI) ||
        !cast<BranchInst>(EntryTI)->isUnconditional() || 
        EntryTI->getSuccessor(0) != TLL->getHeader())
      ShouldExtractLoop = true;
    else {
      // Check to see if any exits from the loop are more than just return
      // blocks.
      for (unsigned i = 0, e = TLL->getExitBlocks().size(); i != e; ++i)
        if (!isa<ReturnInst>(TLL->getExitBlocks()[i]->getTerminator())) {
          ShouldExtractLoop = true;
          break;
        }
    }
    
    if (ShouldExtractLoop) {
      if (NumLoops == 0) return Changed;
      --NumLoops;
      Changed |= ExtractLoop(DS, TLL) != 0;
      ++NumExtracted;
    } else {
      // Okay, this function is a minimal container around the specified loop.
      // If we extract the loop, we will continue to just keep extracting it
      // infinitely... so don't extract it.  However, if the loop contains any
      // subloops, extract them.
      for (Loop::iterator i = TLL->begin(), e = TLL->end(); i != e; ++i) {
        if (NumLoops == 0) return Changed;
        --NumLoops;
        Changed |= ExtractLoop(DS, *i) != 0;
        ++NumExtracted;
      }
    }
  }

  return Changed;
}

// createSingleLoopExtractorPass - This pass extracts one natural loop from the
// program into a function if it can.  This is used by bugpoint.
//
Pass *llvm::createSingleLoopExtractorPass() {
  return new SingleLoopExtractor();
}
