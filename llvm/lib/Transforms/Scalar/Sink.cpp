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
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/UniformityAnalysis.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar.h"
using namespace llvm;

#define DEBUG_TYPE "sink"

STATISTIC(NumSunk, "Number of instructions sunk");
STATISTIC(NumSinkIter, "Number of sinking iterations");

static cl::opt<unsigned> SinkMaxUsesToScan(
    "sink-max-uses-to-scan", cl::init(32), cl::Hidden,
    cl::desc("Maximum number of uses of an operand to scan when checking "
             "whether sinking into a postdominator extends its live range"));

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

/// IsAcceptableTarget - Return true if it is possible to sink the instruction
/// from \p From into the specified basic block.
static bool IsAcceptableTarget(Instruction *Inst, BasicBlock *From,
                               BasicBlock *SuccToSinkTo, DominatorTree &DT,
                               LoopInfo &LI) {
  assert(Inst && "Instruction to be sunk is null");
  assert(SuccToSinkTo && "Candidate sink target is null");

  // It's never legal to sink an instruction into an EH-pad block.
  if (SuccToSinkTo->isEHPad())
    return false;

  // If the block has multiple predecessors, this would introduce computation
  // on different code paths.  We could split the critical edge, but for now we
  // just punt.
  // FIXME: Split critical edges if not backedges.
  if (SuccToSinkTo->getUniquePredecessor() != From) {
    // We cannot sink a load across a critical edge - there may be stores in
    // other code paths.
    if (Inst->mayReadFromMemory() &&
        !Inst->hasMetadata(LLVMContext::MD_invariant_load))
      return false;

    // Don't sink instructions into a loop.
    Loop *succ = LI.getLoopFor(SuccToSinkTo);
    Loop *cur = LI.getLoopFor(From);
    if (succ != nullptr && succ != cur)
      return false;
  }

  return true;
}

/// Return the block between \p From and the nearest common dominator of the
/// uses, \p UsesDom, that \p Inst can be sunk into from \p From, or null if
/// there is none.
static BasicBlock *findSinkTarget(Instruction *Inst, BasicBlock *From,
                                  BasicBlock *UsesDom, DominatorTree &DT,
                                  LoopInfo &LI) {
  // The nearest common dominator may be in a parent loop of From, which may
  // not be beneficial. Find an ancestor.
  BasicBlock *Target = UsesDom;
  while (Target != From && !IsAcceptableTarget(Inst, From, Target, DT, LI))
    Target = DT.getNode(Target)->getIDom()->getBlock();
  return Target == From ? nullptr : Target;
}

/// Return true if \p Op is live into \p Dest because of a use other than
/// \p Inst, i.e. such a use can be reached from \p Dest without executing the
/// definition of \p Op again. Conservatively returns true if \p Op has too many
/// uses to scan.
static bool isLiveInto(Value *Op, Instruction *Inst, BasicBlock *Dest,
                       DominatorTree &DT, LoopInfo &LI) {
  BasicBlock *DefBlock = nullptr;
  if (auto *I = dyn_cast<Instruction>(Op))
    DefBlock = I->getParent();
  if (Op->hasNUsesOrMore(SinkMaxUsesToScan + 1))
    return true;
  SmallPtrSet<const BasicBlock *, 8> UseBlocks;
  for (Use &U : Op->uses()) {
    auto *User = cast<Instruction>(U.getUser());
    if (User == Inst)
      continue;
    BasicBlock *UseBlock = User->getParent();
    // PHI nodes use the operand in the predecessor block.
    if (auto *PN = dyn_cast<PHINode>(User))
      UseBlock = PN->getIncomingBlock(U);
    // Reaching a use in the defining block means passing the definition first,
    // since that block dominates Dest.
    if (UseBlock == DefBlock || !DT.isReachableFromEntry(UseBlock))
      continue;
    if (DT.dominates(Dest, UseBlock))
      return true;
    UseBlocks.insert(UseBlock);
  }
  if (UseBlocks.empty())
    return false;
  SmallVector<BasicBlock *, 1> Worklist = {Dest};
  SmallPtrSet<BasicBlock *, 1> ExclusionSet;
  if (DefBlock)
    ExclusionSet.insert(DefBlock);
  return isManyPotentiallyReachableFromMany(Worklist, UseBlocks, &ExclusionSet,
                                            &DT, &LI);
}

/// Return true if sinking \p Inst into \p Dest would keep more bits live than
/// it frees: the operands that are not otherwise live into \p Dest would have
/// their live ranges extended, while only the result's live range shrinks.
/// With \p UI, only divergent values are counted, since uniform values are
/// kept in scalar registers and do not add to per-lane register pressure.
static bool increasesLiveBits(Instruction *Inst, BasicBlock *Dest,
                              DominatorTree &DT, LoopInfo &LI,
                              const UniformityInfo *UI) {
  const DataLayout &DL = Inst->getDataLayout();
  auto SizeInBits = [&](const Value *V) -> uint64_t {
    if (!V->getType()->isSized())
      return 0;
    return DL.getTypeSizeInBits(V->getType()).getKnownMinValue();
  };
  // A value that is uniform at its definition is still divergent where it is
  // used outside of a cycle with a divergent exit, so check the uses.
  auto IsDivergentUse = [&](const Use &U) {
    return !UI || UI->isDivergentAtUse(U);
  };
  uint64_t FreedBits =
      any_of(Inst->uses(), IsDivergentUse) ? SizeInBits(Inst) : 0;
  uint64_t ExtendedBits = 0;
  SmallPtrSet<Value *, 4> Seen;
  for (Use &U : Inst->operands()) {
    Value *Op = U.get();
    if (!isa<Instruction, Argument>(Op) || !Seen.insert(Op).second)
      continue;
    uint64_t Bits = SizeInBits(Op);
    if (Bits && IsDivergentUse(U) && !isLiveInto(Op, Inst, Dest, DT, LI)) {
      ExtendedBits += Bits;
      if (ExtendedBits > FreedBits)
        return true;
    }
  }
  return false;
}

/// SinkInstruction - Determine whether it is safe to sink the specified machine
/// instruction out of its current block into a successor.
static bool SinkInstruction(Instruction *Inst,
                            SmallPtrSetImpl<Instruction *> &Stores,
                            DominatorTree &DT, LoopInfo &LI, AAResults &AA,
                            PostDominatorTree *PDT, const UniformityInfo *UI) {

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

  BasicBlock *UsesDom = SuccToSinkTo;
  if (UsesDom)
    SuccToSinkTo = findSinkTarget(Inst, BB, UsesDom, DT, LI);

  // If we couldn't find a block to sink to, ignore this instruction.
  if (!SuccToSinkTo)
    return false;

  if (PDT) {
    // Sinking is repeated until nothing changes, so Inst may later move on
    // from SuccToSinkTo towards UsesDom. Find where it ends up.
    BasicBlock *Dest = SuccToSinkTo;
    while (BasicBlock *Next = findSinkTarget(Inst, Dest, UsesDom, DT, LI)) {
      // Inst would be placed above every write in Dest, which may then keep
      // it from moving on.
      SmallPtrSet<Instruction *, 8> Writes;
      for (Instruction &I : *Dest)
        if (I.mayWriteToMemory())
          Writes.insert(&I);
      if (!isSafeToMove(Inst, AA, Writes))
        break;
      Dest = Next;
    }
    // If Dest postdominates BB, it executes whenever BB does, so sinking saves
    // no execution. Don't sink if that would keep more values live across the
    // region than it frees.
    if (PDT->dominates(Dest, BB) && increasesLiveBits(Inst, Dest, DT, LI, UI)) {
      LLVM_DEBUG(dbgs() << "Not sinking into postdominator" << *Inst << "\n");
      return false;
    }
  }

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
  return true;
}

static bool ProcessBlock(BasicBlock &BB, DominatorTree &DT, LoopInfo &LI,
                         AAResults &AA, PostDominatorTree *PDT,
                         const UniformityInfo *UI) {
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

    if (isa<PseudoProbeInst>(Inst))
      continue;

    if (SinkInstruction(Inst, Stores, DT, LI, AA, PDT, UI)) {
      ++NumSunk;
      MadeChange = true;
    }

    // If we just processed the first instruction in the block, we're done.
  } while (!ProcessedBegin);

  return MadeChange;
}

static bool iterativelySinkInstructions(Function &F, DominatorTree &DT,
                                        LoopInfo &LI, AAResults &AA,
                                        PostDominatorTree *PDT,
                                        const UniformityInfo *UI) {
  bool MadeChange, EverMadeChange = false;

  do {
    MadeChange = false;
    LLVM_DEBUG(dbgs() << "Sinking iteration " << NumSinkIter << "\n");
    // Process all basic blocks.
    for (BasicBlock &I : F)
      MadeChange |= ProcessBlock(I, DT, LI, AA, PDT, UI);
    EverMadeChange |= MadeChange;
    NumSinkIter++;
  } while (MadeChange);

  return EverMadeChange;
}

PreservedAnalyses SinkingPass::run(Function &F, FunctionAnalysisManager &AM) {
  auto &DT = AM.getResult<DominatorTreeAnalysis>(F);
  auto &LI = AM.getResult<LoopAnalysis>(F);
  auto &AA = AM.getResult<AAManager>(F);
  PostDominatorTree *PDT = nullptr;
  const UniformityInfo *UI = nullptr;
  if (AvoidPostDominators) {
    PDT = &AM.getResult<PostDominatorTreeAnalysis>(F);
    if (AM.getResult<TargetIRAnalysis>(F).hasBranchDivergence(&F))
      UI = &AM.getResult<UniformityInfoAnalysis>(F);
  }

  if (!iterativelySinkInstructions(F, DT, LI, AA, PDT, UI))
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}

void SinkingPass::printPipeline(
    raw_ostream &OS, function_ref<StringRef(StringRef)> MapClassName2PassName) {
  static_cast<PassInfoMixin<SinkingPass> *>(this)->printPipeline(
      OS, MapClassName2PassName);
  if (AvoidPostDominators)
    OS << "<avoid-postdominators>";
}

namespace {
  class SinkingLegacyPass : public FunctionPass {
    bool AvoidPostDominators;

  public:
    static char ID; // Pass identification
    SinkingLegacyPass(bool AvoidPostDominators = false)
        : FunctionPass(ID), AvoidPostDominators(AvoidPostDominators) {
      initializeSinkingLegacyPassPass(*PassRegistry::getPassRegistry());
    }

    bool runOnFunction(Function &F) override {
      auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
      auto &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
      auto &AA = getAnalysis<AAResultsWrapperPass>().getAAResults();
      PostDominatorTree *PDT = nullptr;
      const UniformityInfo *UI = nullptr;
      if (AvoidPostDominators) {
        PDT = &getAnalysis<PostDominatorTreeWrapperPass>().getPostDomTree();
        if (getAnalysis<TargetTransformInfoWrapperPass>()
                .getTTI(F)
                .hasBranchDivergence(&F))
          UI = &getAnalysis<UniformityInfoWrapperPass>().getUniformityInfo();
      }

      return iterativelySinkInstructions(F, DT, LI, AA, PDT, UI);
    }

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.setPreservesCFG();
      FunctionPass::getAnalysisUsage(AU);
      AU.addRequired<AAResultsWrapperPass>();
      AU.addRequired<DominatorTreeWrapperPass>();
      AU.addRequired<LoopInfoWrapperPass>();
      if (AvoidPostDominators) {
        AU.addRequired<PostDominatorTreeWrapperPass>();
        AU.addRequired<TargetTransformInfoWrapperPass>();
        AU.addRequired<UniformityInfoWrapperPass>();
      }
    }
  };
} // end anonymous namespace

char SinkingLegacyPass::ID = 0;
INITIALIZE_PASS_BEGIN(SinkingLegacyPass, "sink", "Code sinking", false, false)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(PostDominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(UniformityInfoWrapperPass)
INITIALIZE_PASS_END(SinkingLegacyPass, "sink", "Code sinking", false, false)

FunctionPass *llvm::createSinkingPass(bool AvoidPostDominators) {
  return new SinkingLegacyPass(AvoidPostDominators);
}
