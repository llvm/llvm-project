//===-- Sink.cpp - Code Sinking -------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass moves instructions into successor blocks, when possible, so that
// they aren't executed on paths where their results aren't needed.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "sink"
#include "llvm/Transforms/Scalar.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

STATISTIC(NumSunk, "Number of instructions sunk");
STATISTIC(NumSinkIter, "Number of sinking iterations");

namespace {
  class Sinking : public FunctionPass {
    DominatorTree *DT;
    LoopInfo *LI;
    AliasAnalysis *AA;

  public:
    static char ID; // Pass identification
    Sinking() : FunctionPass(ID) {
      initializeSinkingPass(*PassRegistry::getPassRegistry());
    }

    virtual bool runOnFunction(Function &F);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      FunctionPass::getAnalysisUsage(AU);
      AU.addRequired<AliasAnalysis>();
      AU.addRequired<DominatorTree>();
      AU.addRequired<LoopInfo>();
      AU.addPreserved<DominatorTree>();
      AU.addPreserved<LoopInfo>();
    }
  private:
    bool ProcessBlock(BasicBlock &BB);
    bool SinkInstruction(Instruction *I, SmallPtrSet<Instruction *, 8> &Stores);
    bool AllUsesDominatedByBlock(Instruction *Inst, BasicBlock *BB) const;
    bool IsAcceptableTarget(Instruction *Inst, BasicBlock *SuccToSinkTo) const;
  };
} // end anonymous namespace

char Sinking::ID = 0;
INITIALIZE_PASS_BEGIN(Sinking, "sink", "Code sinking", false, false)
INITIALIZE_PASS_DEPENDENCY(LoopInfo)
INITIALIZE_PASS_DEPENDENCY(DominatorTree)
INITIALIZE_AG_DEPENDENCY(AliasAnalysis)
INITIALIZE_PASS_END(Sinking, "sink", "Code sinking", false, false)

FunctionPass *llvm::createSinkingPass() { return new Sinking(); }

/// AllUsesDominatedByBlock - Return true if all uses of the specified value
/// occur in blocks dominated by the specified block.
bool Sinking::AllUsesDominatedByBlock(Instruction *Inst,
                                      BasicBlock *BB) const {
  // Ignoring debug uses is necessary so debug info doesn't affect the code.
  // This may leave a referencing dbg_value in the original block, before
  // the definition of the vreg.  Dwarf generator handles this although the
  // user might not get the right info at runtime.
  for (Value::use_iterator I = Inst->use_begin(),
       E = Inst->use_end(); I != E; ++I) {
    // Determine the block of the use.
    Instruction *UseInst = cast<Instruction>(*I);
    BasicBlock *UseBlock = UseInst->getParent();
    if (PHINode *PN = dyn_cast<PHINode>(UseInst)) {
      // PHI nodes use the operand in the predecessor block, not the block with
      // the PHI.
      unsigned Num = PHINode::getIncomingValueNumForOperand(I.getOperandNo());
      UseBlock = PN->getIncomingBlock(Num);
    }
    // Check that it dominates.
    if (!DT->dominates(BB, UseBlock))
      return false;
  }
  return true;
}

bool Sinking::runOnFunction(Function &F) {
  DT = &getAnalysis<DominatorTree>();
  LI = &getAnalysis<LoopInfo>();
  AA = &getAnalysis<AliasAnalysis>();

  bool MadeChange, EverMadeChange = false;

  do {
    MadeChange = false;
    DEBUG(dbgs() << "Sinking iteration " << NumSinkIter << "\n");
    // Process all basic blocks.
    for (Function::iterator I = F.begin(), E = F.end();
         I != E; ++I)
      MadeChange |= ProcessBlock(*I);
    EverMadeChange |= MadeChange;
    NumSinkIter++;
  } while (MadeChange);

  return EverMadeChange;
}

bool Sinking::ProcessBlock(BasicBlock &BB) {
  // Can't sink anything out of a block that has less than two successors.
  if (BB.getTerminator()->getNumSuccessors() <= 1 || BB.empty()) return false;

  // Don't bother sinking code out of unreachable blocks. In addition to being
  // unprofitable, it can also lead to infinite looping, because in an
  // unreachable loop there may be nowhere to stop.
  if (!DT->isReachableFromEntry(&BB)) return false;

  bool MadeChange = false;

  // Walk the basic block bottom-up.  Remember if we saw a store.
  BasicBlock::iterator I = BB.end();
  --I;
  bool ProcessedBegin = false;
  SmallPtrSet<Instruction *, 8> Stores;
  do {
    Instruction *Inst = I;  // The instruction to sink.

    // Predecrement I (if it's not begin) so that it isn't invalidated by
    // sinking.
    ProcessedBegin = I == BB.begin();
    if (!ProcessedBegin)
      --I;

    if (isa<DbgInfoIntrinsic>(Inst))
      continue;

    if (SinkInstruction(Inst, Stores))
      ++NumSunk, MadeChange = true;

    // If we just processed the first instruction in the block, we're done.
  } while (!ProcessedBegin);

  return MadeChange;
}

static bool isSafeToMove(Instruction *Inst, AliasAnalysis *AA,
                         SmallPtrSet<Instruction *, 8> &Stores) {

  if (Inst->mayWriteToMemory()) {
    Stores.insert(Inst);
    return false;
  }

  if (LoadInst *L = dyn_cast<LoadInst>(Inst)) {
    AliasAnalysis::Location Loc = AA->getLocation(L);
    for (SmallPtrSet<Instruction *, 8>::iterator I = Stores.begin(),
         E = Stores.end(); I != E; ++I)
      if (AA->getModRefInfo(*I, Loc) & AliasAnalysis::Mod)
        return false;
  }

  if (isa<TerminatorInst>(Inst) || isa<PHINode>(Inst))
    return false;

  return true;
}

/// IsAcceptableTarget - Return true if it is possible to sink the instruction
/// in the specified basic block.
bool Sinking::IsAcceptableTarget(Instruction *Inst,
                                 BasicBlock *SuccToSinkTo) const {
  assert(Inst && "Instruction to be sunk is null");
  assert(SuccToSinkTo && "Candidate sink target is null");

  // It is not possible to sink an instruction into its own block.  This can
  // happen with loops.
  if (Inst->getParent() == SuccToSinkTo)
    return false;

  // If the block has multiple predecessors, this would introduce computation
  // on different code paths.  We could split the critical edge, but for now we
  // just punt.
  // FIXME: Split critical edges if not backedges.
  if (SuccToSinkTo->getUniquePredecessor() != Inst->getParent()) {
    // We cannot sink a load across a critical edge - there may be stores in
    // other code paths.
    if (!isSafeToSpeculativelyExecute(Inst))
      return false;

    // We don't want to sink across a critical edge if we don't dominate the
    // successor. We could be introducing calculations to new code paths.
    if (!DT->dominates(Inst->getParent(), SuccToSinkTo))
      return false;

    // Don't sink instructions into a loop.
    Loop *succ = LI->getLoopFor(SuccToSinkTo);
    Loop *cur = LI->getLoopFor(Inst->getParent());
    if (succ != 0 && succ != cur)
      return false;
  }

  // Finally, check that all the uses of the instruction are actually
  // dominated by the candidate
  return AllUsesDominatedByBlock(Inst, SuccToSinkTo);
}

/// SinkInstruction - Determine whether it is safe to sink the specified machine
/// instruction out of its current block into a successor.
bool Sinking::SinkInstruction(Instruction *Inst,
                              SmallPtrSet<Instruction *, 8> &Stores) {
  // Check if it's safe to move the instruction.
  if (!isSafeToMove(Inst, AA, Stores))
    return false;

  // FIXME: This should include support for sinking instructions within the
  // block they are currently in to shorten the live ranges.  We often get
  // instructions sunk into the top of a large block, but it would be better to
  // also sink them down before their first use in the block.  This xform has to
  // be careful not to *increase* register pressure though, e.g. sinking
  // "x = y + z" down if it kills y and z would increase the live ranges of y
  // and z and only shrink the live range of x.

  // SuccToSinkTo - This is the successor to sink this instruction to, once we
  // decide.
  BasicBlock *SuccToSinkTo = 0;

  // Instructions can only be sunk if all their uses are in blocks
  // dominated by one of the successors.
  // Look at all the postdominators and see if we can sink it in one.
  DomTreeNode *DTN = DT->getNode(Inst->getParent());
  for (DomTreeNode::iterator I = DTN->begin(), E = DTN->end();
      I != E && SuccToSinkTo == 0; ++I) {
    BasicBlock *Candidate = (*I)->getBlock();
    if ((*I)->getIDom()->getBlock() == Inst->getParent() &&
        IsAcceptableTarget(Inst, Candidate))
      SuccToSinkTo = Candidate;
  }

  // If no suitable postdominator was found, look at all the successors and
  // decide which one we should sink to, if any.
  for (succ_iterator I = succ_begin(Inst->getParent()),
      E = succ_end(Inst->getParent()); I != E && SuccToSinkTo == 0; ++I) {
    if (IsAcceptableTarget(Inst, *I))
      SuccToSinkTo = *I;
  }

  // If we couldn't find a block to sink to, ignore this instruction.
  if (SuccToSinkTo == 0)
    return false;

  DEBUG(dbgs() << "Sink" << *Inst << " (";
        WriteAsOperand(dbgs(), Inst->getParent(), false);
        dbgs() << " -> ";
        WriteAsOperand(dbgs(), SuccToSinkTo, false);
        dbgs() << ")\n");

  // Move the instruction.
  Inst->moveBefore(SuccToSinkTo->getFirstInsertionPt());
  return true;
}
