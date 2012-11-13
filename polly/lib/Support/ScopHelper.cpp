//===- ScopHelper.cpp - Some Helper Functions for Scop.  ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Small functions that help with Scop and LLVM-IR.
//
//===----------------------------------------------------------------------===//

#include "polly/Support/ScopHelper.h"

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/RegionInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Support/CFG.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#define DEBUG_TYPE "polly-scop-helper"
#include "llvm/Support/Debug.h"

using namespace llvm;

// Helper function for Scop
// TODO: Add assertion to not allow parameter to be null
//===----------------------------------------------------------------------===//
// Temporary Hack for extended region tree.
// Cast the region to loop if there is a loop have the same header and exit.
Loop *polly::castToLoop(const Region &R, LoopInfo &LI) {
  BasicBlock *entry = R.getEntry();

  if (!LI.isLoopHeader(entry))
    return 0;

  Loop *L = LI.getLoopFor(entry);

  BasicBlock *exit = L->getExitBlock();

  // Is the loop with multiple exits?
  if (!exit) return 0;

  if (exit != R.getExit()) {
    // SubRegion/ParentRegion with the same entry.
    assert((R.getNode(R.getEntry())->isSubRegion()
            || R.getParent()->getEntry() == entry)
           && "Expect the loop is the smaller or bigger region");
    return 0;
  }

  return L;
}

Value *polly::getPointerOperand(Instruction &Inst) {
  if (LoadInst *load = dyn_cast<LoadInst>(&Inst))
    return load->getPointerOperand();
  else if (StoreInst *store = dyn_cast<StoreInst>(&Inst))
    return store->getPointerOperand();
  else if (GetElementPtrInst *gep = dyn_cast<GetElementPtrInst>(&Inst))
    return gep->getPointerOperand();

  return 0;
}

//===----------------------------------------------------------------------===//
// Helper functions
bool polly::isIndVar(const SCEV *Var, Region &RefRegion,
                     LoopInfo &LI, ScalarEvolution &SE) {
  const SCEVAddRecExpr *AddRec = dyn_cast<SCEVAddRecExpr>(Var);

  // AddRecExprs are no induction variables.
  if (!AddRec) return false;

  Loop *L = const_cast<Loop*>(AddRec->getLoop());

  // Is the addrec an induction variable of a loop contained in the current
  // region.
  if (!RefRegion.contains(L))
    return false;

  DEBUG(dbgs() << "Find AddRec: " << *AddRec
        << " at region: " << RefRegion.getNameStr() << " as indvar\n");
  return true;
}

bool polly::isIndVar(const Instruction *I, const LoopInfo *LI) {
  Loop *L = LI->getLoopFor(I->getParent());

  return L && I == L->getCanonicalInductionVariable();
}

bool polly::hasInvokeEdge(const PHINode *PN) {
  for (unsigned i = 0, e = PN->getNumIncomingValues(); i < e; ++i)
    if (InvokeInst *II = dyn_cast<InvokeInst>(PN->getIncomingValue(i)))
      if (II->getParent() == PN->getIncomingBlock(i))
        return true;

  return false;
}

BasicBlock *polly::createSingleExitEdge(Region *R, Pass *P) {
  BasicBlock *BB = R->getExit();

  SmallVector<BasicBlock*, 4> Preds;
  for (pred_iterator PI = pred_begin(BB), PE = pred_end(BB); PI != PE; ++PI)
    if (R->contains(*PI))
      Preds.push_back(*PI);

  return SplitBlockPredecessors(BB, Preds, ".region", P);
}

void polly::splitEntryBlockForAlloca(BasicBlock *EntryBlock, Pass *P) {
  // Find first non-alloca instruction. Every basic block has a non-alloc
  // instruction, as every well formed basic block has a terminator.
  BasicBlock::iterator I = EntryBlock->begin();
  while (isa<AllocaInst>(I)) ++I;

  // SplitBlock updates DT, DF and LI.
  BasicBlock *NewEntry = SplitBlock(EntryBlock, I, P);
  if (RegionInfo *RI = P->getAnalysisIfAvailable<RegionInfo>())
    RI->splitBlock(NewEntry, EntryBlock);
}
