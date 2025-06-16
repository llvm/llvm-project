//===-- Sink.cpp - Code Sinking -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass moves instructions into successor blocks, when possible, so that
// they aren't executed on paths where their results aren't needed.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/Sink.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/Analysis/MemorySSAUpdater.h"
#include "llvm/IR/Dominators.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar.h"
using namespace llvm;

#define DEBUG_TYPE "sink"

STATISTIC(NumSunk, "Number of instructions sunk");
STATISTIC(NumSinkIter, "Number of sinking iterations");

static bool isSafeToMove(Instruction *Inst, AliasAnalysis &AA,
                         SmallPtrSetImpl<Instruction *> &Stores) {

  if (Inst->mayWriteToMemory()) {
    Stores.insert(Inst);
    return false;
  }

  // Don't sink static alloca instructions.  CodeGen assumes allocas outside the
  // entry block are dynamically sized stack objects.
  if (AllocaInst *AI = dyn_cast<AllocaInst>(Inst))
    if (AI->isStaticAlloca())
      return false;

  if (LoadInst *L = dyn_cast<LoadInst>(Inst)) {
    MemoryLocation Loc = MemoryLocation::get(L);
    for (Instruction *S : Stores)
      if (isModSet(AA.getModRefInfo(S, Loc)))
        return false;
  }

  if (Inst->isTerminator() || isa<PHINode>(Inst) || Inst->isEHPad() ||
      Inst->mayThrow() || !Inst->willReturn())
    return false;

  if (auto *Call = dyn_cast<CallBase>(Inst)) {
    // Convergent operations cannot be made control-dependent on additional
    // values.
    if (Call->isConvergent())
      return false;

    for (Instruction *S : Stores)
      if (isModSet(AA.getModRefInfo(S, Call)))
        return false;
  }

  return true;
}

static cl::opt<unsigned> SinkLoadStoreLimit(
    "sink-load-store-limit", cl::Hidden, cl::init(4),
    cl::desc("Maximum number of stores in descendant blocks that will be "
             "analyzed when attempting to sink a load."));

using BlocksSet = SmallPtrSet<BasicBlock *, 8>;
static bool hasStoreConflict(BasicBlock *LoadBB, BasicBlock *BB,
                             BlocksSet &VisitedBlocksSet,
                             MemorySSAUpdater &MSSAU, BatchAAResults &BAA,
                             Instruction *ReadMemInst, unsigned &StoreCnt) {
  if (BB == LoadBB || !VisitedBlocksSet.insert(BB).second)
    return false;
  if (auto *Accesses = MSSAU.getMemorySSA()->getBlockDefs(BB)) {
    StoreCnt += Accesses->size();
    if (StoreCnt > SinkLoadStoreLimit)
      return true;
    for (auto &MA : *Accesses) {
      if (auto *MD = dyn_cast<MemoryDef>(&MA)) {
        Instruction *S = MD->getMemoryInst();
        if (LoadInst *L = dyn_cast<LoadInst>(ReadMemInst)) {
          MemoryLocation Loc = MemoryLocation::get(L);
          if (isModSet(BAA.getModRefInfo(S, Loc)))
            return true;
        } else if (auto *Call = dyn_cast<CallBase>(ReadMemInst)) {
          if (isModSet(BAA.getModRefInfo(S, Call)))
            return true;
        }
      }
    }
  }
  for (BasicBlock *Pred : predecessors(BB)) {
    if (hasStoreConflict(LoadBB, Pred, VisitedBlocksSet, MSSAU, BAA,
                         ReadMemInst, StoreCnt))
      return true;
  }
  return false;
}

static bool hasConflictingStoreBeforeSuccToSinkTo(Instruction *ReadMemInst,
                                                  BasicBlock *SuccToSinkTo,
                                                  MemorySSAUpdater &MSSAU,
                                                  BatchAAResults &BAA) {
  BlocksSet VisitedBlocksSet;
  BasicBlock *LoadBB = ReadMemInst->getParent();
  unsigned StoreCnt{0};

  for (BasicBlock *Pred : predecessors(SuccToSinkTo))
    if (hasStoreConflict(LoadBB, Pred, VisitedBlocksSet, MSSAU, BAA,
                         ReadMemInst, StoreCnt))
      return true;
  return false;
}

/// IsAcceptableTarget - Return true if it is possible to sink the instruction
/// in the specified basic block.
static bool IsAcceptableTarget(Instruction *Inst, BasicBlock *SuccToSinkTo,
                               DominatorTree &DT, LoopInfo &LI,
                               MemorySSAUpdater &MSSAU, BatchAAResults &BAA) {
  assert(Inst && "Instruction to be sunk is null");
  assert(SuccToSinkTo && "Candidate sink target is null");

  // It's never legal to sink an instruction into an EH-pad block.
  if (SuccToSinkTo->isEHPad())
    return false;

  // If the block has multiple predecessors, this would introduce computation
  // on different code paths.  We could split the critical edge, but for now we
  // just punt.
  // FIXME: Split critical edges if not backedges.
  if (SuccToSinkTo->getUniquePredecessor() != Inst->getParent()) {
    // Ensure that there is no conflicting store on any path to SuccToSinkTo.
    if (Inst->mayReadFromMemory() &&
        !Inst->hasMetadata(LLVMContext::MD_invariant_load) &&
        hasConflictingStoreBeforeSuccToSinkTo(Inst, SuccToSinkTo, MSSAU, BAA))
      return false;

    // Don't sink instructions into a loop.
    Loop *succ = LI.getLoopFor(SuccToSinkTo);
    Loop *cur = LI.getLoopFor(Inst->getParent());
    if (succ != nullptr && succ != cur)
      return false;
  }

  return true;
}

/// SinkInstruction - Determine whether it is safe to sink the specified machine
/// instruction out of its current block into a successor.
static bool SinkInstruction(Instruction *Inst,
                            SmallPtrSetImpl<Instruction *> &Stores,
                            DominatorTree &DT, LoopInfo &LI, AAResults &AA,
                            MemorySSAUpdater &MSSAU) {

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
  BasicBlock *SuccToSinkTo = nullptr;

  // Find the nearest common dominator of all users as the candidate.
  BasicBlock *BB = Inst->getParent();
  for (Use &U : Inst->uses()) {
    Instruction *UseInst = cast<Instruction>(U.getUser());
    BasicBlock *UseBlock = UseInst->getParent();
    if (PHINode *PN = dyn_cast<PHINode>(UseInst)) {
      // PHI nodes use the operand in the predecessor block, not the block with
      // the PHI.
      unsigned Num = PHINode::getIncomingValueNumForOperand(U.getOperandNo());
      UseBlock = PN->getIncomingBlock(Num);
    }
    // Don't worry about dead users.
    if (!DT.isReachableFromEntry(UseBlock))
      continue;

    if (SuccToSinkTo)
      SuccToSinkTo = DT.findNearestCommonDominator(SuccToSinkTo, UseBlock);
    else
      SuccToSinkTo = UseBlock;
  }

  if (SuccToSinkTo) {
    // The nearest common dominator may be in a parent loop of BB, which may not
    // be beneficial. Find an ancestor.
    BatchAAResults BAA(AA);
    while (SuccToSinkTo != BB &&
           !IsAcceptableTarget(Inst, SuccToSinkTo, DT, LI, MSSAU, BAA))
      SuccToSinkTo = DT.getNode(SuccToSinkTo)->getIDom()->getBlock();
    if (SuccToSinkTo == BB)
      SuccToSinkTo = nullptr;
  }

  // If we couldn't find a block to sink to, ignore this instruction.
  if (!SuccToSinkTo)
    return false;

  LLVM_DEBUG(dbgs() << "Sink" << *Inst << " (";
             Inst->getParent()->printAsOperand(dbgs(), false); dbgs() << " -> ";
             SuccToSinkTo->printAsOperand(dbgs(), false); dbgs() << ")\n");

  // The current location of Inst dominates all uses, thus it must dominate
  // SuccToSinkTo, which is on the IDom chain between the nearest common
  // dominator to all uses and the current location.
  assert(DT.dominates(BB, SuccToSinkTo) &&
         "SuccToSinkTo must be dominated by current Inst location!");

  // Move the instruction.
  Inst->moveBefore(SuccToSinkTo->getFirstInsertionPt());
  if (MemoryUseOrDef *OldMemAcc = cast_or_null<MemoryUseOrDef>(
          MSSAU.getMemorySSA()->getMemoryAccess(Inst)))
    MSSAU.moveToPlace(OldMemAcc, SuccToSinkTo, MemorySSA::Beginning);

  return true;
}

static bool ProcessBlock(BasicBlock &BB, DominatorTree &DT, LoopInfo &LI,
                         AAResults &AA, MemorySSAUpdater &MSSAU) {
  // Don't bother sinking code out of unreachable blocks. In addition to being
  // unprofitable, it can also lead to infinite looping, because in an
  // unreachable loop there may be nowhere to stop.
  if (!DT.isReachableFromEntry(&BB)) return false;

  bool MadeChange = false;

  // Walk the basic block bottom-up.  Remember if we saw a store.
  BasicBlock::iterator I = BB.end();
  --I;
  bool ProcessedBegin = false;
  SmallPtrSet<Instruction *, 8> Stores;
  do {
    Instruction *Inst = &*I; // The instruction to sink.

    // Predecrement I (if it's not begin) so that it isn't invalidated by
    // sinking.
    ProcessedBegin = I == BB.begin();
    if (!ProcessedBegin)
      --I;

    if (Inst->isDebugOrPseudoInst())
      continue;

    if (SinkInstruction(Inst, Stores, DT, LI, AA, MSSAU)) {
      ++NumSunk;
      MadeChange = true;
    }

    // If we just processed the first instruction in the block, we're done.
  } while (!ProcessedBegin);

  return MadeChange;
}

static bool iterativelySinkInstructions(Function &F, DominatorTree &DT,
                                        LoopInfo &LI, AAResults &AA,
                                        MemorySSAUpdater &MSSAU) {
  bool MadeChange, EverMadeChange = false;

  do {
    MadeChange = false;
    LLVM_DEBUG(dbgs() << "Sinking iteration " << NumSinkIter << "\n");
    // Process all basic blocks.
    for (BasicBlock &I : F)
      MadeChange |= ProcessBlock(I, DT, LI, AA, MSSAU);
    EverMadeChange |= MadeChange;
    NumSinkIter++;
  } while (MadeChange);

  return EverMadeChange;
}

PreservedAnalyses SinkingPass::run(Function &F, FunctionAnalysisManager &AM) {
  auto &DT = AM.getResult<DominatorTreeAnalysis>(F);
  auto &LI = AM.getResult<LoopAnalysis>(F);
  auto &AA = AM.getResult<AAManager>(F);
  MemorySSA &MSSA = AM.getResult<MemorySSAAnalysis>(F).getMSSA();
  MemorySSAUpdater MSSAU(&MSSA);

  if (!iterativelySinkInstructions(F, DT, LI, AA, MSSAU))
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  PA.preserve<MemorySSAAnalysis>();
  return PA;
}

namespace {
  class SinkingLegacyPass : public FunctionPass {
  public:
    static char ID; // Pass identification
    SinkingLegacyPass() : FunctionPass(ID) {
      initializeSinkingLegacyPassPass(*PassRegistry::getPassRegistry());
    }

    bool runOnFunction(Function &F) override {
      auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
      auto &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
      auto &AA = getAnalysis<AAResultsWrapperPass>().getAAResults();
      MemorySSA *MSSA = &getAnalysis<MemorySSAWrapperPass>().getMSSA();
      MemorySSAUpdater MSSAU(MSSA);
      return iterativelySinkInstructions(F, DT, LI, AA, MSSAU);
    }

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.setPreservesCFG();
      FunctionPass::getAnalysisUsage(AU);
      AU.addRequired<AAResultsWrapperPass>();
      AU.addRequired<DominatorTreeWrapperPass>();
      AU.addRequired<LoopInfoWrapperPass>();
      AU.addRequired<MemorySSAWrapperPass>();
      AU.addPreserved<DominatorTreeWrapperPass>();
      AU.addPreserved<LoopInfoWrapperPass>();
      AU.addPreserved<MemorySSAWrapperPass>();
    }
  };
} // end anonymous namespace

char SinkingLegacyPass::ID = 0;
INITIALIZE_PASS_BEGIN(SinkingLegacyPass, "sink", "Code sinking", false, false)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MemorySSAWrapperPass)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_END(SinkingLegacyPass, "sink", "Code sinking", false, false)

FunctionPass *llvm::createSinkingPass() { return new SinkingLegacyPass(); }
