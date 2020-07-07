//===-- UnrollLoop.cpp - Loop unrolling utilities -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements some loop unrolling utilities. It does not define any
// actual pass or policy, but provides a single function to perform loop
// unrolling.
//
// The process of unrolling can produce extraneous basic blocks linked with
// unconditional branches.  This will be corrected in the future.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/ilist_iterator.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopIterator.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Use.h"
#include "llvm/IR/User.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/IR/ValueMap.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/GenericDomTree.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/LoopSimplify.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/SimplifyIndVar.h"
#include "llvm/Transforms/Utils/UnrollLoop.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include <algorithm>
#include <assert.h>
#include <type_traits>
#include <vector>

namespace llvm {
class DataLayout;
class Value;
} // namespace llvm

using namespace llvm;

#define DEBUG_TYPE "loop-unroll"

// TODO: Should these be here or in LoopUnroll?
STATISTIC(NumCompletelyUnrolled, "Number of loops completely unrolled");
STATISTIC(NumUnrolled, "Number of loops unrolled (completely or otherwise)");
STATISTIC(NumUnrolledNotLatch, "Number of loops unrolled without a conditional "
                               "latch (completely or otherwise)");

static cl::opt<bool>
UnrollRuntimeEpilog("unroll-runtime-epilog", cl::init(false), cl::Hidden,
                    cl::desc("Allow runtime unrolled loops to be unrolled "
                             "with epilog instead of prolog."));

static cl::opt<bool>
UnrollVerifyDomtree("unroll-verify-domtree", cl::Hidden,
                    cl::desc("Verify domtree after unrolling"),
#ifdef EXPENSIVE_CHECKS
    cl::init(true)
#else
    cl::init(false)
#endif
                    );

/// Check if unrolling created a situation where we need to insert phi nodes to
/// preserve LCSSA form.
/// \param Blocks is a vector of basic blocks representing unrolled loop.
/// \param L is the outer loop.
/// It's possible that some of the blocks are in L, and some are not. In this
/// case, if there is a use is outside L, and definition is inside L, we need to
/// insert a phi-node, otherwise LCSSA will be broken.
/// The function is just a helper function for llvm::UnrollLoop that returns
/// true if this situation occurs, indicating that LCSSA needs to be fixed.
static bool needToInsertPhisForLCSSA(Loop *L, std::vector<BasicBlock *> Blocks,
                                     LoopInfo *LI) {
  for (BasicBlock *BB : Blocks) {
    if (LI->getLoopFor(BB) == L)
      continue;
    for (Instruction &I : *BB) {
      for (Use &U : I.operands()) {
        if (auto Def = dyn_cast<Instruction>(U)) {
          Loop *DefLoop = LI->getLoopFor(Def->getParent());
          if (!DefLoop)
            continue;
          if (DefLoop->contains(L))
            return true;
        }
      }
    }
  }
  return false;
}

/// Adds ClonedBB to LoopInfo, creates a new loop for ClonedBB if necessary
/// and adds a mapping from the original loop to the new loop to NewLoops.
/// Returns nullptr if no new loop was created and a pointer to the
/// original loop OriginalBB was part of otherwise.
const Loop* llvm::addClonedBlockToLoopInfo(BasicBlock *OriginalBB,
                                           BasicBlock *ClonedBB, LoopInfo *LI,
                                           NewLoopsMap &NewLoops) {
  // Figure out which loop New is in.
  const Loop *OldLoop = LI->getLoopFor(OriginalBB);
  assert(OldLoop && "Should (at least) be in the loop being unrolled!");

  Loop *&NewLoop = NewLoops[OldLoop];
  if (!NewLoop) {
    // Found a new sub-loop.
    assert(OriginalBB == OldLoop->getHeader() &&
           "Header should be first in RPO");

    NewLoop = LI->AllocateLoop();
    Loop *NewLoopParent = NewLoops.lookup(OldLoop->getParentLoop());

    if (NewLoopParent)
      NewLoopParent->addChildLoop(NewLoop);
    else
      LI->addTopLevelLoop(NewLoop);

    NewLoop->addBasicBlockToLoop(ClonedBB, *LI);
    return OldLoop;
  } else {
    NewLoop->addBasicBlockToLoop(ClonedBB, *LI);
    return nullptr;
  }
}

/// The function chooses which type of unroll (epilog or prolog) is more
/// profitabale.
/// Epilog unroll is more profitable when there is PHI that starts from
/// constant.  In this case epilog will leave PHI start from constant,
/// but prolog will convert it to non-constant.
///
/// loop:
///   PN = PHI [I, Latch], [CI, PreHeader]
///   I = foo(PN)
///   ...
///
/// Epilog unroll case.
/// loop:
///   PN = PHI [I2, Latch], [CI, PreHeader]
///   I1 = foo(PN)
///   I2 = foo(I1)
///   ...
/// Prolog unroll case.
///   NewPN = PHI [PrologI, Prolog], [CI, PreHeader]
/// loop:
///   PN = PHI [I2, Latch], [NewPN, PreHeader]
///   I1 = foo(PN)
///   I2 = foo(I1)
///   ...
///
static bool isEpilogProfitable(Loop *L) {
  BasicBlock *PreHeader = L->getLoopPreheader();
  BasicBlock *Header = L->getHeader();
  assert(PreHeader && Header);
  for (const PHINode &PN : Header->phis()) {
    if (isa<ConstantInt>(PN.getIncomingValueForBlock(PreHeader)))
      return true;
  }
  return false;
}

/// Perform some cleanup and simplifications on loops after unrolling. It is
/// useful to simplify the IV's in the new loop, as well as do a quick
/// simplify/dce pass of the instructions.
void llvm::simplifyLoopAfterUnroll(Loop *L, bool SimplifyIVs, LoopInfo *LI,
                                   ScalarEvolution *SE, DominatorTree *DT,
                                   AssumptionCache *AC,
                                   const TargetTransformInfo *TTI) {
  // Simplify any new induction variables in the partially unrolled loop.
  if (SE && SimplifyIVs) {
    SmallVector<WeakTrackingVH, 16> DeadInsts;
    simplifyLoopIVs(L, SE, DT, LI, TTI, DeadInsts);

    // Aggressively clean up dead instructions that simplifyLoopIVs already
    // identified. Any remaining should be cleaned up below.
    while (!DeadInsts.empty()) {
      Value *V = DeadInsts.pop_back_val();
      if (Instruction *Inst = dyn_cast_or_null<Instruction>(V))
        RecursivelyDeleteTriviallyDeadInstructions(Inst);
    }
  }

  // At this point, the code is well formed.  We now do a quick sweep over the
  // inserted code, doing constant propagation and dead code elimination as we
  // go.
  const DataLayout &DL = L->getHeader()->getModule()->getDataLayout();
  for (BasicBlock *BB : L->getBlocks()) {
    for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E;) {
      Instruction *Inst = &*I++;

      if (Value *V = SimplifyInstruction(Inst, {DL, nullptr, DT, AC}))
        if (LI->replacementPreservesLCSSAForm(Inst, V))
          Inst->replaceAllUsesWith(V);
      if (isInstructionTriviallyDead(Inst))
        BB->getInstList().erase(Inst);
    }
  }

  // TODO: after peeling or unrolling, previously loop variant conditions are
  // likely to fold to constants, eagerly propagating those here will require
  // fewer cleanup passes to be run.  Alternatively, a LoopEarlyCSE might be
  // appropriate.
}

/// Unroll the given loop by Count. The loop must be in LCSSA form.  Unrolling
/// can only fail when the loop's latch block is not terminated by a conditional
/// branch instruction. However, if the trip count (and multiple) are not known,
/// loop unrolling will mostly produce more code that is no faster.
///
/// TripCount is the upper bound of the iteration on which control exits
/// LatchBlock. Control may exit the loop prior to TripCount iterations either
/// via an early branch in other loop block or via LatchBlock terminator. This
/// is relaxed from the general definition of trip count which is the number of
/// times the loop header executes. Note that UnrollLoop assumes that the loop
/// counter test is in LatchBlock in order to remove unnecesssary instances of
/// the test.  If control can exit the loop from the LatchBlock's terminator
/// prior to TripCount iterations, flag PreserveCondBr needs to be set.
///
/// PreserveCondBr indicates whether the conditional branch of the LatchBlock
/// needs to be preserved.  It is needed when we use trip count upper bound to
/// fully unroll the loop. If PreserveOnlyFirst is also set then only the first
/// conditional branch needs to be preserved.
///
/// Similarly, TripMultiple divides the number of times that the LatchBlock may
/// execute without exiting the loop.
///
/// If AllowRuntime is true then UnrollLoop will consider unrolling loops that
/// have a runtime (i.e. not compile time constant) trip count.  Unrolling these
/// loops require a unroll "prologue" that runs "RuntimeTripCount % Count"
/// iterations before branching into the unrolled loop.  UnrollLoop will not
/// runtime-unroll the loop if computing RuntimeTripCount will be expensive and
/// AllowExpensiveTripCount is false.
///
/// If we want to perform PGO-based loop peeling, PeelCount is set to the
/// number of iterations we want to peel off.
///
/// The LoopInfo Analysis that is passed will be kept consistent.
///
/// This utility preserves LoopInfo. It will also preserve ScalarEvolution and
/// DominatorTree if they are non-null.
///
/// If RemainderLoop is non-null, it will receive the remainder loop (if
/// required and not fully unrolled).
LoopUnrollResult llvm::UnrollLoop(Loop *L, UnrollLoopOptions ULO, LoopInfo *LI,
                                  ScalarEvolution *SE, DominatorTree *DT,
                                  AssumptionCache *AC,
                                  const TargetTransformInfo *TTI,
                                  OptimizationRemarkEmitter *ORE,
                                  bool PreserveLCSSA, Loop **RemainderLoop) {

  BasicBlock *Preheader = L->getLoopPreheader();
  if (!Preheader) {
    LLVM_DEBUG(dbgs() << "  Can't unroll; loop preheader-insertion failed.\n");
    return LoopUnrollResult::Unmodified;
  }

  BasicBlock *LatchBlock = L->getLoopLatch();
  if (!LatchBlock) {
    LLVM_DEBUG(dbgs() << "  Can't unroll; loop exit-block-insertion failed.\n");
    return LoopUnrollResult::Unmodified;
  }

  // Loops with indirectbr cannot be cloned.
  if (!L->isSafeToClone()) {
    LLVM_DEBUG(dbgs() << "  Can't unroll; Loop body cannot be cloned.\n");
    return LoopUnrollResult::Unmodified;
  }

  // The current loop unroll pass can unroll loops that have
  // (1) single latch; and
  // (2a) latch is unconditional; or
  // (2b) latch is conditional and is an exiting block
  // FIXME: The implementation can be extended to work with more complicated
  // cases, e.g. loops with multiple latches.
  BasicBlock *Header = L->getHeader();
  BranchInst *LatchBI = dyn_cast<BranchInst>(LatchBlock->getTerminator());

  // A conditional branch which exits the loop, which can be optimized to an
  // unconditional branch in the unrolled loop in some cases.
  BranchInst *ExitingBI = nullptr;
  bool LatchIsExiting = L->isLoopExiting(LatchBlock);
  if (LatchIsExiting)
    ExitingBI = LatchBI;
  else if (BasicBlock *ExitingBlock = L->getExitingBlock())
    ExitingBI = dyn_cast<BranchInst>(ExitingBlock->getTerminator());
  if (!LatchBI || (LatchBI->isConditional() && !LatchIsExiting)) {
    LLVM_DEBUG(
        dbgs() << "Can't unroll; a conditional latch must exit the loop");
    return LoopUnrollResult::Unmodified;
  }
  LLVM_DEBUG({
    if (ExitingBI)
      dbgs() << "  Exiting Block = " << ExitingBI->getParent()->getName()
             << "\n";
    else
      dbgs() << "  No single exiting block\n";
  });

  if (Header->hasAddressTaken()) {
    // The loop-rotate pass can be helpful to avoid this in many cases.
    LLVM_DEBUG(
        dbgs() << "  Won't unroll loop: address of header block is taken.\n");
    return LoopUnrollResult::Unmodified;
  }

  if (ULO.TripCount != 0)
    LLVM_DEBUG(dbgs() << "  Trip Count = " << ULO.TripCount << "\n");
  if (ULO.TripMultiple != 1)
    LLVM_DEBUG(dbgs() << "  Trip Multiple = " << ULO.TripMultiple << "\n");

  // Effectively "DCE" unrolled iterations that are beyond the tripcount
  // and will never be executed.
  if (ULO.TripCount != 0 && ULO.Count > ULO.TripCount)
    ULO.Count = ULO.TripCount;

  // Don't enter the unroll code if there is nothing to do.
  if (ULO.TripCount == 0 && ULO.Count < 2 && ULO.PeelCount == 0) {
    LLVM_DEBUG(dbgs() << "Won't unroll; almost nothing to do\n");
    return LoopUnrollResult::Unmodified;
  }

  assert(ULO.Count > 0);
  assert(ULO.TripMultiple > 0);
  assert(ULO.TripCount == 0 || ULO.TripCount % ULO.TripMultiple == 0);

  // Are we eliminating the loop control altogether?
  bool CompletelyUnroll = ULO.Count == ULO.TripCount;
  SmallVector<BasicBlock *, 4> ExitBlocks;
  L->getExitBlocks(ExitBlocks);
  std::vector<BasicBlock*> OriginalLoopBlocks = L->getBlocks();

  // Go through all exits of L and see if there are any phi-nodes there. We just
  // conservatively assume that they're inserted to preserve LCSSA form, which
  // means that complete unrolling might break this form. We need to either fix
  // it in-place after the transformation, or entirely rebuild LCSSA. TODO: For
  // now we just recompute LCSSA for the outer loop, but it should be possible
  // to fix it in-place.
  bool NeedToFixLCSSA = PreserveLCSSA && CompletelyUnroll &&
                        any_of(ExitBlocks, [](const BasicBlock *BB) {
                          return isa<PHINode>(BB->begin());
                        });

  // We assume a run-time trip count if the compiler cannot
  // figure out the loop trip count and the unroll-runtime
  // flag is specified.
  bool RuntimeTripCount =
      (ULO.TripCount == 0 && ULO.Count > 0 && ULO.AllowRuntime);

  assert((!RuntimeTripCount || !ULO.PeelCount) &&
         "Did not expect runtime trip-count unrolling "
         "and peeling for the same loop");

  bool Peeled = false;
  if (ULO.PeelCount) {
    Peeled = peelLoop(L, ULO.PeelCount, LI, SE, DT, AC, PreserveLCSSA);

    // Successful peeling may result in a change in the loop preheader/trip
    // counts. If we later unroll the loop, we want these to be updated.
    if (Peeled) {
      // According to our guards and profitability checks the only
      // meaningful exit should be latch block. Other exits go to deopt,
      // so we do not worry about them.
      BasicBlock *ExitingBlock = L->getLoopLatch();
      assert(ExitingBlock && "Loop without exiting block?");
      assert(L->isLoopExiting(ExitingBlock) && "Latch is not exiting?");
      Preheader = L->getLoopPreheader();
      ULO.TripCount = SE->getSmallConstantTripCount(L, ExitingBlock);
      ULO.TripMultiple = SE->getSmallConstantTripMultiple(L, ExitingBlock);
    }
  }

  // Loops containing convergent instructions must have a count that divides
  // their TripMultiple.
  LLVM_DEBUG(
      {
        bool HasConvergent = false;
        for (auto &BB : L->blocks())
          for (auto &I : *BB)
            if (auto *CB = dyn_cast<CallBase>(&I))
              HasConvergent |= CB->isConvergent();
        assert((!HasConvergent || ULO.TripMultiple % ULO.Count == 0) &&
               "Unroll count must divide trip multiple if loop contains a "
               "convergent operation.");
      });

  bool EpilogProfitability =
      UnrollRuntimeEpilog.getNumOccurrences() ? UnrollRuntimeEpilog
                                              : isEpilogProfitable(L);

  if (RuntimeTripCount && ULO.TripMultiple % ULO.Count != 0 &&
      !UnrollRuntimeLoopRemainder(L, ULO.Count, ULO.AllowExpensiveTripCount,
                                  EpilogProfitability, ULO.UnrollRemainder,
                                  ULO.ForgetAllSCEV, LI, SE, DT, AC, TTI,
                                  PreserveLCSSA, RemainderLoop)) {
    if (ULO.Force)
      RuntimeTripCount = false;
    else {
      LLVM_DEBUG(dbgs() << "Won't unroll; remainder loop could not be "
                           "generated when assuming runtime trip count\n");
      return LoopUnrollResult::Unmodified;
    }
  }

  // If we know the trip count, we know the multiple...
  unsigned BreakoutTrip = 0;
  if (ULO.TripCount != 0) {
    BreakoutTrip = ULO.TripCount % ULO.Count;
    ULO.TripMultiple = 0;
  } else {
    // Figure out what multiple to use.
    BreakoutTrip = ULO.TripMultiple =
        (unsigned)GreatestCommonDivisor64(ULO.Count, ULO.TripMultiple);
  }

  using namespace ore;
  // Report the unrolling decision.
  if (CompletelyUnroll) {
    LLVM_DEBUG(dbgs() << "COMPLETELY UNROLLING loop %" << Header->getName()
                      << " with trip count " << ULO.TripCount << "!\n");
    if (ORE)
      ORE->emit([&]() {
        return OptimizationRemark(DEBUG_TYPE, "FullyUnrolled", L->getStartLoc(),
                                  L->getHeader())
               << "completely unrolled loop with "
               << NV("UnrollCount", ULO.TripCount) << " iterations";
      });
  } else if (ULO.PeelCount) {
    LLVM_DEBUG(dbgs() << "PEELING loop %" << Header->getName()
                      << " with iteration count " << ULO.PeelCount << "!\n");
    if (ORE)
      ORE->emit([&]() {
        return OptimizationRemark(DEBUG_TYPE, "Peeled", L->getStartLoc(),
                                  L->getHeader())
               << " peeled loop by " << NV("PeelCount", ULO.PeelCount)
               << " iterations";
      });
  } else {
    auto DiagBuilder = [&]() {
      OptimizationRemark Diag(DEBUG_TYPE, "PartialUnrolled", L->getStartLoc(),
                              L->getHeader());
      return Diag << "unrolled loop by a factor of "
                  << NV("UnrollCount", ULO.Count);
    };

    LLVM_DEBUG(dbgs() << "UNROLLING loop %" << Header->getName() << " by "
                      << ULO.Count);
    if (ULO.TripMultiple == 0 || BreakoutTrip != ULO.TripMultiple) {
      LLVM_DEBUG(dbgs() << " with a breakout at trip " << BreakoutTrip);
      if (ORE)
        ORE->emit([&]() {
          return DiagBuilder() << " with a breakout at trip "
                               << NV("BreakoutTrip", BreakoutTrip);
        });
    } else if (ULO.TripMultiple != 1) {
      LLVM_DEBUG(dbgs() << " with " << ULO.TripMultiple << " trips per branch");
      if (ORE)
        ORE->emit([&]() {
          return DiagBuilder()
                 << " with " << NV("TripMultiple", ULO.TripMultiple)
                 << " trips per branch";
        });
    } else if (RuntimeTripCount) {
      LLVM_DEBUG(dbgs() << " with run-time trip count");
      if (ORE)
        ORE->emit(
            [&]() { return DiagBuilder() << " with run-time trip count"; });
    }
    LLVM_DEBUG(dbgs() << "!\n");
  }

  // We are going to make changes to this loop. SCEV may be keeping cached info
  // about it, in particular about backedge taken count. The changes we make
  // are guaranteed to invalidate this information for our loop. It is tempting
  // to only invalidate the loop being unrolled, but it is incorrect as long as
  // all exiting branches from all inner loops have impact on the outer loops,
  // and if something changes inside them then any of outer loops may also
  // change. When we forget outermost loop, we also forget all contained loops
  // and this is what we need here.
  if (SE) {
    if (ULO.ForgetAllSCEV)
      SE->forgetAllLoops();
    else
      SE->forgetTopmostLoop(L);
  }

  if (!LatchIsExiting)
    ++NumUnrolledNotLatch;
  Optional<bool> ContinueOnTrue = None;
  BasicBlock *LoopExit = nullptr;
  if (ExitingBI) {
    ContinueOnTrue = L->contains(ExitingBI->getSuccessor(0));
    LoopExit = ExitingBI->getSuccessor(*ContinueOnTrue);
  }

  // For the first iteration of the loop, we should use the precloned values for
  // PHI nodes.  Insert associations now.
  ValueToValueMapTy LastValueMap;
  std::vector<PHINode*> OrigPHINode;
  for (BasicBlock::iterator I = Header->begin(); isa<PHINode>(I); ++I) {
    OrigPHINode.push_back(cast<PHINode>(I));
  }

  std::vector<BasicBlock *> Headers;
  std::vector<BasicBlock *> ExitingBlocks;
  std::vector<BasicBlock *> ExitingSucc;
  std::vector<BasicBlock *> Latches;
  Headers.push_back(Header);
  Latches.push_back(LatchBlock);
  if (ExitingBI) {
    ExitingBlocks.push_back(ExitingBI->getParent());
    ExitingSucc.push_back(ExitingBI->getSuccessor(!(*ContinueOnTrue)));
  }

  // The current on-the-fly SSA update requires blocks to be processed in
  // reverse postorder so that LastValueMap contains the correct value at each
  // exit.
  LoopBlocksDFS DFS(L);
  DFS.perform(LI);

  // Stash the DFS iterators before adding blocks to the loop.
  LoopBlocksDFS::RPOIterator BlockBegin = DFS.beginRPO();
  LoopBlocksDFS::RPOIterator BlockEnd = DFS.endRPO();

  std::vector<BasicBlock*> UnrolledLoopBlocks = L->getBlocks();

  // Loop Unrolling might create new loops. While we do preserve LoopInfo, we
  // might break loop-simplified form for these loops (as they, e.g., would
  // share the same exit blocks). We'll keep track of loops for which we can
  // break this so that later we can re-simplify them.
  SmallSetVector<Loop *, 4> LoopsToSimplify;
  for (Loop *SubLoop : *L)
    LoopsToSimplify.insert(SubLoop);

  if (Header->getParent()->isDebugInfoForProfiling())
    for (BasicBlock *BB : L->getBlocks())
      for (Instruction &I : *BB)
        if (!isa<DbgInfoIntrinsic>(&I))
          if (const DILocation *DIL = I.getDebugLoc()) {
            auto NewDIL = DIL->cloneByMultiplyingDuplicationFactor(ULO.Count);
            if (NewDIL)
              I.setDebugLoc(NewDIL.getValue());
            else
              LLVM_DEBUG(dbgs()
                         << "Failed to create new discriminator: "
                         << DIL->getFilename() << " Line: " << DIL->getLine());
          }

  for (unsigned It = 1; It != ULO.Count; ++It) {
    SmallVector<BasicBlock *, 8> NewBlocks;
    SmallDenseMap<const Loop *, Loop *, 4> NewLoops;
    NewLoops[L] = L;

    for (LoopBlocksDFS::RPOIterator BB = BlockBegin; BB != BlockEnd; ++BB) {
      ValueToValueMapTy VMap;
      BasicBlock *New = CloneBasicBlock(*BB, VMap, "." + Twine(It));
      Header->getParent()->getBasicBlockList().push_back(New);

      assert((*BB != Header || LI->getLoopFor(*BB) == L) &&
             "Header should not be in a sub-loop");
      // Tell LI about New.
      const Loop *OldLoop = addClonedBlockToLoopInfo(*BB, New, LI, NewLoops);
      if (OldLoop)
        LoopsToSimplify.insert(NewLoops[OldLoop]);

      if (*BB == Header)
        // Loop over all of the PHI nodes in the block, changing them to use
        // the incoming values from the previous block.
        for (PHINode *OrigPHI : OrigPHINode) {
          PHINode *NewPHI = cast<PHINode>(VMap[OrigPHI]);
          Value *InVal = NewPHI->getIncomingValueForBlock(LatchBlock);
          if (Instruction *InValI = dyn_cast<Instruction>(InVal))
            if (It > 1 && L->contains(InValI))
              InVal = LastValueMap[InValI];
          VMap[OrigPHI] = InVal;
          New->getInstList().erase(NewPHI);
        }

      // Update our running map of newest clones
      LastValueMap[*BB] = New;
      for (ValueToValueMapTy::iterator VI = VMap.begin(), VE = VMap.end();
           VI != VE; ++VI)
        LastValueMap[VI->first] = VI->second;

      // Add phi entries for newly created values to all exit blocks.
      for (BasicBlock *Succ : successors(*BB)) {
        if (L->contains(Succ))
          continue;
        for (PHINode &PHI : Succ->phis()) {
          Value *Incoming = PHI.getIncomingValueForBlock(*BB);
          ValueToValueMapTy::iterator It = LastValueMap.find(Incoming);
          if (It != LastValueMap.end())
            Incoming = It->second;
          PHI.addIncoming(Incoming, New);
        }
      }
      // Keep track of new headers and latches as we create them, so that
      // we can insert the proper branches later.
      if (*BB == Header)
        Headers.push_back(New);
      if (*BB == LatchBlock)
        Latches.push_back(New);

      // Keep track of the exiting block and its successor block contained in
      // the loop for the current iteration.
      if (ExitingBI) {
        if (*BB == ExitingBlocks[0])
          ExitingBlocks.push_back(New);
        if (*BB == ExitingSucc[0])
          ExitingSucc.push_back(New);
      }

      NewBlocks.push_back(New);
      UnrolledLoopBlocks.push_back(New);

      // Update DomTree: since we just copy the loop body, and each copy has a
      // dedicated entry block (copy of the header block), this header's copy
      // dominates all copied blocks. That means, dominance relations in the
      // copied body are the same as in the original body.
      if (DT) {
        if (*BB == Header)
          DT->addNewBlock(New, Latches[It - 1]);
        else {
          auto BBDomNode = DT->getNode(*BB);
          auto BBIDom = BBDomNode->getIDom();
          BasicBlock *OriginalBBIDom = BBIDom->getBlock();
          DT->addNewBlock(
              New, cast<BasicBlock>(LastValueMap[cast<Value>(OriginalBBIDom)]));
        }
      }
    }

    // Remap all instructions in the most recent iteration
    remapInstructionsInBlocks(NewBlocks, LastValueMap);
    for (BasicBlock *NewBlock : NewBlocks) {
      for (Instruction &I : *NewBlock) {
        if (auto *II = dyn_cast<IntrinsicInst>(&I))
          if (II->getIntrinsicID() == Intrinsic::assume)
            AC->registerAssumption(II);
      }
    }
  }

  // Loop over the PHI nodes in the original block, setting incoming values.
  for (PHINode *PN : OrigPHINode) {
    if (CompletelyUnroll) {
      PN->replaceAllUsesWith(PN->getIncomingValueForBlock(Preheader));
      Header->getInstList().erase(PN);
    } else if (ULO.Count > 1) {
      Value *InVal = PN->removeIncomingValue(LatchBlock, false);
      // If this value was defined in the loop, take the value defined by the
      // last iteration of the loop.
      if (Instruction *InValI = dyn_cast<Instruction>(InVal)) {
        if (L->contains(InValI))
          InVal = LastValueMap[InVal];
      }
      assert(Latches.back() == LastValueMap[LatchBlock] && "bad last latch");
      PN->addIncoming(InVal, Latches.back());
    }
  }

  auto setDest = [](BasicBlock *Src, BasicBlock *Dest, BasicBlock *BlockInLoop,
                    bool NeedConditional, Optional<bool> ContinueOnTrue,
                    bool IsDestLoopExit) {
    auto *Term = cast<BranchInst>(Src->getTerminator());
    if (NeedConditional) {
      // Update the conditional branch's successor for the following
      // iteration.
      assert(ContinueOnTrue.hasValue() &&
             "Expecting valid ContinueOnTrue when NeedConditional is true");
      Term->setSuccessor(!(*ContinueOnTrue), Dest);
    } else {
      // Remove phi operands at this loop exit
      if (!IsDestLoopExit) {
        BasicBlock *BB = Src;
        for (BasicBlock *Succ : successors(BB)) {
          // Preserve the incoming value from BB if we are jumping to the block
          // in the current loop.
          if (Succ == BlockInLoop)
            continue;
          for (PHINode &Phi : Succ->phis())
            Phi.removeIncomingValue(BB, false);
        }
      }
      // Replace the conditional branch with an unconditional one.
      BranchInst::Create(Dest, Term);
      Term->eraseFromParent();
    }
  };

  // Connect latches of the unrolled iterations to the headers of the next
  // iteration. If the latch is also the exiting block, the conditional branch
  // may have to be preserved.
  for (unsigned i = 0, e = Latches.size(); i != e; ++i) {
    // The branch destination.
    unsigned j = (i + 1) % e;
    BasicBlock *Dest = Headers[j];
    bool NeedConditional = LatchIsExiting;

    if (LatchIsExiting) {
      if (RuntimeTripCount && j != 0)
        NeedConditional = false;

      // For a complete unroll, make the last iteration end with a branch
      // to the exit block.
      if (CompletelyUnroll) {
        if (j == 0)
          Dest = LoopExit;
        // If using trip count upper bound to completely unroll, we need to
        // keep the conditional branch except the last one because the loop
        // may exit after any iteration.
        assert(NeedConditional &&
               "NeedCondition cannot be modified by both complete "
               "unrolling and runtime unrolling");
        NeedConditional =
            (ULO.PreserveCondBr && j && !(ULO.PreserveOnlyFirst && i != 0));
      } else if (j != BreakoutTrip &&
                 (ULO.TripMultiple == 0 || j % ULO.TripMultiple != 0)) {
        // If we know the trip count or a multiple of it, we can safely use an
        // unconditional branch for some iterations.
        NeedConditional = false;
      }
    }

    setDest(Latches[i], Dest, Headers[i], NeedConditional, ContinueOnTrue,
            Dest == LoopExit);
  }

  if (!LatchIsExiting) {
    // If the latch is not exiting, we may be able to simplify the conditional
    // branches in the unrolled exiting blocks.
    for (unsigned i = 0, e = ExitingBlocks.size(); i != e; ++i) {
      // The branch destination.
      unsigned j = (i + 1) % e;
      bool NeedConditional = true;

      if (RuntimeTripCount && j != 0)
        NeedConditional = false;

      if (CompletelyUnroll)
        // We cannot drop the conditional branch for the last condition, as we
        // may have to execute the loop body depending on the condition.
        NeedConditional = j == 0 || ULO.PreserveCondBr;
      else if (j != BreakoutTrip &&
               (ULO.TripMultiple == 0 || j % ULO.TripMultiple != 0))
        // If we know the trip count or a multiple of it, we can safely use an
        // unconditional branch for some iterations.
        NeedConditional = false;

      // Conditional branches from non-latch exiting block have successors
      // either in the same loop iteration or outside the loop. The branches are
      // already correct.
      if (NeedConditional)
        continue;
      setDest(ExitingBlocks[i], ExitingSucc[i], ExitingSucc[i], NeedConditional,
              None, false);
    }

    // When completely unrolling, the last latch becomes unreachable.
    if (CompletelyUnroll) {
      BranchInst *Term = cast<BranchInst>(Latches.back()->getTerminator());
      new UnreachableInst(Term->getContext(), Term);
      Term->eraseFromParent();
    }
  }

  // Update dominators of blocks we might reach through exits.
  // Immediate dominator of such block might change, because we add more
  // routes which can lead to the exit: we can now reach it from the copied
  // iterations too.
  if (DT && ULO.Count > 1) {
    for (auto *BB : OriginalLoopBlocks) {
      auto *BBDomNode = DT->getNode(BB);
      SmallVector<BasicBlock *, 16> ChildrenToUpdate;
      for (auto *ChildDomNode : BBDomNode->children()) {
        auto *ChildBB = ChildDomNode->getBlock();
        if (!L->contains(ChildBB))
          ChildrenToUpdate.push_back(ChildBB);
      }
      BasicBlock *NewIDom;
      if (ExitingBI && BB == ExitingBlocks[0]) {
        // The latch is special because we emit unconditional branches in
        // some cases where the original loop contained a conditional branch.
        // Since the latch is always at the bottom of the loop, if the latch
        // dominated an exit before unrolling, the new dominator of that exit
        // must also be a latch.  Specifically, the dominator is the first
        // latch which ends in a conditional branch, or the last latch if
        // there is no such latch.
        // For loops exiting from non latch exiting block, we limit the
        // branch simplification to single exiting block loops.
        NewIDom = ExitingBlocks.back();
        for (unsigned i = 0, e = ExitingBlocks.size(); i != e; ++i) {
          Instruction *Term = ExitingBlocks[i]->getTerminator();
          if (isa<BranchInst>(Term) && cast<BranchInst>(Term)->isConditional()) {
            NewIDom =
                DT->findNearestCommonDominator(ExitingBlocks[i], Latches[i]);
            break;
          }
        }
      } else {
        // The new idom of the block will be the nearest common dominator
        // of all copies of the previous idom. This is equivalent to the
        // nearest common dominator of the previous idom and the first latch,
        // which dominates all copies of the previous idom.
        NewIDom = DT->findNearestCommonDominator(BB, LatchBlock);
      }
      for (auto *ChildBB : ChildrenToUpdate)
        DT->changeImmediateDominator(ChildBB, NewIDom);
    }
  }

  assert(!DT || !UnrollVerifyDomtree ||
         DT->verify(DominatorTree::VerificationLevel::Fast));

  DomTreeUpdater DTU(DT, DomTreeUpdater::UpdateStrategy::Lazy);
  // Merge adjacent basic blocks, if possible.
  for (BasicBlock *Latch : Latches) {
    BranchInst *Term = dyn_cast<BranchInst>(Latch->getTerminator());
    assert((Term ||
            (CompletelyUnroll && !LatchIsExiting && Latch == Latches.back())) &&
           "Need a branch as terminator, except when fully unrolling with "
           "unconditional latch");
    if (Term && Term->isUnconditional()) {
      BasicBlock *Dest = Term->getSuccessor(0);
      BasicBlock *Fold = Dest->getUniquePredecessor();
      if (MergeBlockIntoPredecessor(Dest, &DTU, LI)) {
        // Dest has been folded into Fold. Update our worklists accordingly.
        std::replace(Latches.begin(), Latches.end(), Dest, Fold);
        UnrolledLoopBlocks.erase(std::remove(UnrolledLoopBlocks.begin(),
                                             UnrolledLoopBlocks.end(), Dest),
                                 UnrolledLoopBlocks.end());
      }
    }
  }
  // Apply updates to the DomTree.
  DT = &DTU.getDomTree();

  // At this point, the code is well formed.  We now simplify the unrolled loop,
  // doing constant propagation and dead code elimination as we go.
  simplifyLoopAfterUnroll(L, !CompletelyUnroll && (ULO.Count > 1 || Peeled), LI,
                          SE, DT, AC, TTI);

  NumCompletelyUnrolled += CompletelyUnroll;
  ++NumUnrolled;

  Loop *OuterL = L->getParentLoop();
  // Update LoopInfo if the loop is completely removed.
  if (CompletelyUnroll)
    LI->erase(L);

  // After complete unrolling most of the blocks should be contained in OuterL.
  // However, some of them might happen to be out of OuterL (e.g. if they
  // precede a loop exit). In this case we might need to insert PHI nodes in
  // order to preserve LCSSA form.
  // We don't need to check this if we already know that we need to fix LCSSA
  // form.
  // TODO: For now we just recompute LCSSA for the outer loop in this case, but
  // it should be possible to fix it in-place.
  if (PreserveLCSSA && OuterL && CompletelyUnroll && !NeedToFixLCSSA)
    NeedToFixLCSSA |= ::needToInsertPhisForLCSSA(OuterL, UnrolledLoopBlocks, LI);

  // If we have a pass and a DominatorTree we should re-simplify impacted loops
  // to ensure subsequent analyses can rely on this form. We want to simplify
  // at least one layer outside of the loop that was unrolled so that any
  // changes to the parent loop exposed by the unrolling are considered.
  if (DT) {
    if (OuterL) {
      // OuterL includes all loops for which we can break loop-simplify, so
      // it's sufficient to simplify only it (it'll recursively simplify inner
      // loops too).
      if (NeedToFixLCSSA) {
        // LCSSA must be performed on the outermost affected loop. The unrolled
        // loop's last loop latch is guaranteed to be in the outermost loop
        // after LoopInfo's been updated by LoopInfo::erase.
        Loop *LatchLoop = LI->getLoopFor(Latches.back());
        Loop *FixLCSSALoop = OuterL;
        if (!FixLCSSALoop->contains(LatchLoop))
          while (FixLCSSALoop->getParentLoop() != LatchLoop)
            FixLCSSALoop = FixLCSSALoop->getParentLoop();

        formLCSSARecursively(*FixLCSSALoop, *DT, LI, SE);
      } else if (PreserveLCSSA) {
        assert(OuterL->isLCSSAForm(*DT) &&
               "Loops should be in LCSSA form after loop-unroll.");
      }

      // TODO: That potentially might be compile-time expensive. We should try
      // to fix the loop-simplified form incrementally.
      simplifyLoop(OuterL, DT, LI, SE, AC, nullptr, PreserveLCSSA);
    } else {
      // Simplify loops for which we might've broken loop-simplify form.
      for (Loop *SubLoop : LoopsToSimplify)
        simplifyLoop(SubLoop, DT, LI, SE, AC, nullptr, PreserveLCSSA);
    }
  }

  return CompletelyUnroll ? LoopUnrollResult::FullyUnrolled
                          : LoopUnrollResult::PartiallyUnrolled;
}

/// Given an llvm.loop loop id metadata node, returns the loop hint metadata
/// node with the given name (for example, "llvm.loop.unroll.count"). If no
/// such metadata node exists, then nullptr is returned.
MDNode *llvm::GetUnrollMetadata(MDNode *LoopID, StringRef Name) {
  // First operand should refer to the loop id itself.
  assert(LoopID->getNumOperands() > 0 && "requires at least one operand");
  assert(LoopID->getOperand(0) == LoopID && "invalid loop id");

  for (unsigned i = 1, e = LoopID->getNumOperands(); i < e; ++i) {
    MDNode *MD = dyn_cast<MDNode>(LoopID->getOperand(i));
    if (!MD)
      continue;

    MDString *S = dyn_cast<MDString>(MD->getOperand(0));
    if (!S)
      continue;

    if (Name.equals(S->getString()))
      return MD;
  }
  return nullptr;
}
