//===- LoopPeel.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Loop Peeling Utilities.
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/LoopPeel.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopIterator.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/LoopSimplify.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/UnrollLoop.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>

using namespace llvm;
using namespace llvm::PatternMatch;

#define DEBUG_TYPE "loop-peel"

STATISTIC(NumPeeled, "Number of loops peeled");

static cl::opt<unsigned> UnrollPeelCount(
    "unroll-peel-count", cl::Hidden,
    cl::desc("Set the unroll peeling count, for testing purposes"));

static cl::opt<bool>
    UnrollAllowPeeling("unroll-allow-peeling", cl::init(true), cl::Hidden,
                       cl::desc("Allows loops to be peeled when the dynamic "
                                "trip count is known to be low."));

static cl::opt<bool>
    UnrollAllowLoopNestsPeeling("unroll-allow-loop-nests-peeling",
                                cl::init(false), cl::Hidden,
                                cl::desc("Allows loop nests to be peeled."));

static cl::opt<unsigned> UnrollPeelMaxCount(
    "unroll-peel-max-count", cl::init(7), cl::Hidden,
    cl::desc("Max average trip count which will cause loop peeling."));

static cl::opt<unsigned> UnrollForcePeelCount(
    "unroll-force-peel-count", cl::init(0), cl::Hidden,
    cl::desc("Force a peel count regardless of profiling information."));

static cl::opt<bool> UnrollPeelMultiDeoptExit(
    "unroll-peel-multi-deopt-exit", cl::init(true), cl::Hidden,
    cl::desc("Allow peeling of loops with multiple deopt exits."));

static const char *PeeledCountMetaData = "llvm.loop.peeled.count";

// Designates that a Phi is estimated to become invariant after an "infinite"
// number of loop iterations (i.e. only may become an invariant if the loop is
// fully unrolled).
static const unsigned InfiniteIterationsToInvariance =
    std::numeric_limits<unsigned>::max();

// Check whether we are capable of peeling this loop.
bool llvm::canPeel(Loop *L) {
  // Make sure the loop is in simplified form
  if (!L->isLoopSimplifyForm())
    return false;

  if (UnrollPeelMultiDeoptExit) {
    SmallVector<BasicBlock *, 4> Exits;
    L->getUniqueNonLatchExitBlocks(Exits);

    if (!Exits.empty()) {
      // Latch's terminator is a conditional branch, Latch is exiting and
      // all non Latch exits ends up with deoptimize.
      const BasicBlock *Latch = L->getLoopLatch();
      const BranchInst *T = dyn_cast<BranchInst>(Latch->getTerminator());
      return T && T->isConditional() && L->isLoopExiting(Latch) &&
             all_of(Exits, [](const BasicBlock *BB) {
               return BB->getTerminatingDeoptimizeCall();
             });
    }
  }

  // Only peel loops that contain a single exit
  if (!L->getExitingBlock() || !L->getUniqueExitBlock())
    return false;

  // Don't try to peel loops where the latch is not the exiting block.
  // This can be an indication of two different things:
  // 1) The loop is not rotated.
  // 2) The loop contains irreducible control flow that involves the latch.
  const BasicBlock *Latch = L->getLoopLatch();
  if (Latch != L->getExitingBlock())
    return false;

  // Peeling is only supported if the latch is a branch.
  if (!isa<BranchInst>(Latch->getTerminator()))
    return false;

  return true;
}

// This function calculates the number of iterations after which the given Phi
// becomes an invariant. The pre-calculated values are memorized in the map. The
// function (shortcut is I) is calculated according to the following definition:
// Given %x = phi <Inputs from above the loop>, ..., [%y, %back.edge].
//   If %y is a loop invariant, then I(%x) = 1.
//   If %y is a Phi from the loop header, I(%x) = I(%y) + 1.
//   Otherwise, I(%x) is infinite.
// TODO: Actually if %y is an expression that depends only on Phi %z and some
//       loop invariants, we can estimate I(%x) = I(%z) + 1. The example
//       looks like:
//         %x = phi(0, %a),  <-- becomes invariant starting from 3rd iteration.
//         %y = phi(0, 5),
//         %a = %y + 1.
static unsigned calculateIterationsToInvariance(
    PHINode *Phi, Loop *L, BasicBlock *BackEdge,
    SmallDenseMap<PHINode *, unsigned> &IterationsToInvariance) {
  assert(Phi->getParent() == L->getHeader() &&
         "Non-loop Phi should not be checked for turning into invariant.");
  assert(BackEdge == L->getLoopLatch() && "Wrong latch?");
  // If we already know the answer, take it from the map.
  auto I = IterationsToInvariance.find(Phi);
  if (I != IterationsToInvariance.end())
    return I->second;

  // Otherwise we need to analyze the input from the back edge.
  Value *Input = Phi->getIncomingValueForBlock(BackEdge);
  // Place infinity to map to avoid infinite recursion for cycled Phis. Such
  // cycles can never stop on an invariant.
  IterationsToInvariance[Phi] = InfiniteIterationsToInvariance;
  unsigned ToInvariance = InfiniteIterationsToInvariance;

  if (L->isLoopInvariant(Input))
    ToInvariance = 1u;
  else if (PHINode *IncPhi = dyn_cast<PHINode>(Input)) {
    // Only consider Phis in header block.
    if (IncPhi->getParent() != L->getHeader())
      return InfiniteIterationsToInvariance;
    // If the input becomes an invariant after X iterations, then our Phi
    // becomes an invariant after X + 1 iterations.
    unsigned InputToInvariance = calculateIterationsToInvariance(
        IncPhi, L, BackEdge, IterationsToInvariance);
    if (InputToInvariance != InfiniteIterationsToInvariance)
      ToInvariance = InputToInvariance + 1u;
  }

  // If we found that this Phi lies in an invariant chain, update the map.
  if (ToInvariance != InfiniteIterationsToInvariance)
    IterationsToInvariance[Phi] = ToInvariance;
  return ToInvariance;
}

// Return the number of iterations to peel off that make conditions in the
// body true/false. For example, if we peel 2 iterations off the loop below,
// the condition i < 2 can be evaluated at compile time.
//  for (i = 0; i < n; i++)
//    if (i < 2)
//      ..
//    else
//      ..
//   }
static unsigned countToEliminateCompares(Loop &L, unsigned MaxPeelCount,
                                         ScalarEvolution &SE) {
  assert(L.isLoopSimplifyForm() && "Loop needs to be in loop simplify form");
  unsigned DesiredPeelCount = 0;

  for (auto *BB : L.blocks()) {
    auto *BI = dyn_cast<BranchInst>(BB->getTerminator());
    if (!BI || BI->isUnconditional())
      continue;

    // Ignore loop exit condition.
    if (L.getLoopLatch() == BB)
      continue;

    Value *Condition = BI->getCondition();
    Value *LeftVal, *RightVal;
    CmpInst::Predicate Pred;
    if (!match(Condition, m_ICmp(Pred, m_Value(LeftVal), m_Value(RightVal))))
      continue;

    const SCEV *LeftSCEV = SE.getSCEV(LeftVal);
    const SCEV *RightSCEV = SE.getSCEV(RightVal);

    // Do not consider predicates that are known to be true or false
    // independently of the loop iteration.
    if (SE.isKnownPredicate(Pred, LeftSCEV, RightSCEV) ||
        SE.isKnownPredicate(ICmpInst::getInversePredicate(Pred), LeftSCEV,
                            RightSCEV))
      continue;

    // Check if we have a condition with one AddRec and one non AddRec
    // expression. Normalize LeftSCEV to be the AddRec.
    if (!isa<SCEVAddRecExpr>(LeftSCEV)) {
      if (isa<SCEVAddRecExpr>(RightSCEV)) {
        std::swap(LeftSCEV, RightSCEV);
        Pred = ICmpInst::getSwappedPredicate(Pred);
      } else
        continue;
    }

    const SCEVAddRecExpr *LeftAR = cast<SCEVAddRecExpr>(LeftSCEV);

    // Avoid huge SCEV computations in the loop below, make sure we only
    // consider AddRecs of the loop we are trying to peel.
    if (!LeftAR->isAffine() || LeftAR->getLoop() != &L)
      continue;
    if (!(ICmpInst::isEquality(Pred) && LeftAR->hasNoSelfWrap()) &&
        !SE.getMonotonicPredicateType(LeftAR, Pred))
      continue;

    // Check if extending the current DesiredPeelCount lets us evaluate Pred
    // or !Pred in the loop body statically.
    unsigned NewPeelCount = DesiredPeelCount;

    const SCEV *IterVal = LeftAR->evaluateAtIteration(
        SE.getConstant(LeftSCEV->getType(), NewPeelCount), SE);

    // If the original condition is not known, get the negated predicate
    // (which holds on the else branch) and check if it is known. This allows
    // us to peel of iterations that make the original condition false.
    if (!SE.isKnownPredicate(Pred, IterVal, RightSCEV))
      Pred = ICmpInst::getInversePredicate(Pred);

    const SCEV *Step = LeftAR->getStepRecurrence(SE);
    const SCEV *NextIterVal = SE.getAddExpr(IterVal, Step);
    auto PeelOneMoreIteration = [&IterVal, &NextIterVal, &SE, Step,
                                 &NewPeelCount]() {
      IterVal = NextIterVal;
      NextIterVal = SE.getAddExpr(IterVal, Step);
      NewPeelCount++;
    };

    auto CanPeelOneMoreIteration = [&NewPeelCount, &MaxPeelCount]() {
      return NewPeelCount < MaxPeelCount;
    };

    while (CanPeelOneMoreIteration() &&
           SE.isKnownPredicate(Pred, IterVal, RightSCEV))
      PeelOneMoreIteration();

    // With *that* peel count, does the predicate !Pred become known in the
    // first iteration of the loop body after peeling?
    if (!SE.isKnownPredicate(ICmpInst::getInversePredicate(Pred), IterVal,
                             RightSCEV))
      continue; // If not, give up.

    // However, for equality comparisons, that isn't always sufficient to
    // eliminate the comparsion in loop body, we may need to peel one more
    // iteration. See if that makes !Pred become unknown again.
    if (ICmpInst::isEquality(Pred) &&
        !SE.isKnownPredicate(ICmpInst::getInversePredicate(Pred), NextIterVal,
                             RightSCEV) &&
        !SE.isKnownPredicate(Pred, IterVal, RightSCEV) &&
        SE.isKnownPredicate(Pred, NextIterVal, RightSCEV)) {
      if (!CanPeelOneMoreIteration())
        continue; // Need to peel one more iteration, but can't. Give up.
      PeelOneMoreIteration(); // Great!
    }

    DesiredPeelCount = std::max(DesiredPeelCount, NewPeelCount);
  }

  return DesiredPeelCount;
}

// Return the number of iterations we want to peel off.
void llvm::computePeelCount(Loop *L, unsigned LoopSize,
                            TargetTransformInfo::PeelingPreferences &PP,
                            unsigned &TripCount, ScalarEvolution &SE,
                            unsigned Threshold) {
  assert(LoopSize > 0 && "Zero loop size is not allowed!");
  // Save the PP.PeelCount value set by the target in
  // TTI.getPeelingPreferences or by the flag -unroll-peel-count.
  unsigned TargetPeelCount = PP.PeelCount;
  PP.PeelCount = 0;
  if (!canPeel(L))
    return;

  // Only try to peel innermost loops by default.
  // The constraint can be relaxed by the target in TTI.getUnrollingPreferences
  // or by the flag -unroll-allow-loop-nests-peeling.
  if (!PP.AllowLoopNestsPeeling && !L->isInnermost())
    return;

  // If the user provided a peel count, use that.
  bool UserPeelCount = UnrollForcePeelCount.getNumOccurrences() > 0;
  if (UserPeelCount) {
    LLVM_DEBUG(dbgs() << "Force-peeling first " << UnrollForcePeelCount
                      << " iterations.\n");
    PP.PeelCount = UnrollForcePeelCount;
    PP.PeelProfiledIterations = true;
    return;
  }

  // Skip peeling if it's disabled.
  if (!PP.AllowPeeling)
    return;

  unsigned AlreadyPeeled = 0;
  if (auto Peeled = getOptionalIntLoopAttribute(L, PeeledCountMetaData))
    AlreadyPeeled = *Peeled;
  // Stop if we already peeled off the maximum number of iterations.
  if (AlreadyPeeled >= UnrollPeelMaxCount)
    return;

  // Here we try to get rid of Phis which become invariants after 1, 2, ..., N
  // iterations of the loop. For this we compute the number for iterations after
  // which every Phi is guaranteed to become an invariant, and try to peel the
  // maximum number of iterations among these values, thus turning all those
  // Phis into invariants.
  // First, check that we can peel at least one iteration.
  if (2 * LoopSize <= Threshold && UnrollPeelMaxCount > 0) {
    // Store the pre-calculated values here.
    SmallDenseMap<PHINode *, unsigned> IterationsToInvariance;
    // Now go through all Phis to calculate their the number of iterations they
    // need to become invariants.
    // Start the max computation with the UP.PeelCount value set by the target
    // in TTI.getUnrollingPreferences or by the flag -unroll-peel-count.
    unsigned DesiredPeelCount = TargetPeelCount;
    BasicBlock *BackEdge = L->getLoopLatch();
    assert(BackEdge && "Loop is not in simplified form?");
    for (auto BI = L->getHeader()->begin(); isa<PHINode>(&*BI); ++BI) {
      PHINode *Phi = cast<PHINode>(&*BI);
      unsigned ToInvariance = calculateIterationsToInvariance(
          Phi, L, BackEdge, IterationsToInvariance);
      if (ToInvariance != InfiniteIterationsToInvariance)
        DesiredPeelCount = std::max(DesiredPeelCount, ToInvariance);
    }

    // Pay respect to limitations implied by loop size and the max peel count.
    unsigned MaxPeelCount = UnrollPeelMaxCount;
    MaxPeelCount = std::min(MaxPeelCount, Threshold / LoopSize - 1);

    DesiredPeelCount = std::max(DesiredPeelCount,
                                countToEliminateCompares(*L, MaxPeelCount, SE));

    if (DesiredPeelCount > 0) {
      DesiredPeelCount = std::min(DesiredPeelCount, MaxPeelCount);
      // Consider max peel count limitation.
      assert(DesiredPeelCount > 0 && "Wrong loop size estimation?");
      if (DesiredPeelCount + AlreadyPeeled <= UnrollPeelMaxCount) {
        LLVM_DEBUG(dbgs() << "Peel " << DesiredPeelCount
                          << " iteration(s) to turn"
                          << " some Phis into invariants.\n");
        PP.PeelCount = DesiredPeelCount;
        PP.PeelProfiledIterations = false;
        return;
      }
    }
  }

  // Bail if we know the statically calculated trip count.
  // In this case we rather prefer partial unrolling.
  if (TripCount)
    return;

  // Do not apply profile base peeling if it is disabled.
  if (!PP.PeelProfiledIterations)
    return;
  // If we don't know the trip count, but have reason to believe the average
  // trip count is low, peeling should be beneficial, since we will usually
  // hit the peeled section.
  // We only do this in the presence of profile information, since otherwise
  // our estimates of the trip count are not reliable enough.
  if (L->getHeader()->getParent()->hasProfileData()) {
    Optional<unsigned> PeelCount = getLoopEstimatedTripCount(L);
    if (!PeelCount)
      return;

    LLVM_DEBUG(dbgs() << "Profile-based estimated trip count is " << *PeelCount
                      << "\n");

    if (*PeelCount) {
      if ((*PeelCount + AlreadyPeeled <= UnrollPeelMaxCount) &&
          (LoopSize * (*PeelCount + 1) <= Threshold)) {
        LLVM_DEBUG(dbgs() << "Peeling first " << *PeelCount
                          << " iterations.\n");
        PP.PeelCount = *PeelCount;
        return;
      }
      LLVM_DEBUG(dbgs() << "Requested peel count: " << *PeelCount << "\n");
      LLVM_DEBUG(dbgs() << "Already peel count: " << AlreadyPeeled << "\n");
      LLVM_DEBUG(dbgs() << "Max peel count: " << UnrollPeelMaxCount << "\n");
      LLVM_DEBUG(dbgs() << "Peel cost: " << LoopSize * (*PeelCount + 1)
                        << "\n");
      LLVM_DEBUG(dbgs() << "Max peel cost: " << Threshold << "\n");
    }
  }
}

/// Update the branch weights of the latch of a peeled-off loop
/// iteration.
/// This sets the branch weights for the latch of the recently peeled off loop
/// iteration correctly.
/// Let F is a weight of the edge from latch to header.
/// Let E is a weight of the edge from latch to exit.
/// F/(F+E) is a probability to go to loop and E/(F+E) is a probability to
/// go to exit.
/// Then, Estimated TripCount = F / E.
/// For I-th (counting from 0) peeled off iteration we set the the weights for
/// the peeled latch as (TC - I, 1). It gives us reasonable distribution,
/// The probability to go to exit 1/(TC-I) increases. At the same time
/// the estimated trip count of remaining loop reduces by I.
/// To avoid dealing with division rounding we can just multiple both part
/// of weights to E and use weight as (F - I * E, E).
///
/// \param Header The copy of the header block that belongs to next iteration.
/// \param LatchBR The copy of the latch branch that belongs to this iteration.
/// \param[in,out] FallThroughWeight The weight of the edge from latch to
/// header before peeling (in) and after peeled off one iteration (out).
static void updateBranchWeights(BasicBlock *Header, BranchInst *LatchBR,
                                uint64_t ExitWeight,
                                uint64_t &FallThroughWeight) {
  // FallThroughWeight is 0 means that there is no branch weights on original
  // latch block or estimated trip count is zero.
  if (!FallThroughWeight)
    return;

  unsigned HeaderIdx = (LatchBR->getSuccessor(0) == Header ? 0 : 1);
  MDBuilder MDB(LatchBR->getContext());
  MDNode *WeightNode =
      HeaderIdx ? MDB.createBranchWeights(ExitWeight, FallThroughWeight)
                : MDB.createBranchWeights(FallThroughWeight, ExitWeight);
  LatchBR->setMetadata(LLVMContext::MD_prof, WeightNode);
  FallThroughWeight =
      FallThroughWeight > ExitWeight ? FallThroughWeight - ExitWeight : 1;
}

/// Initialize the weights.
///
/// \param Header The header block.
/// \param LatchBR The latch branch.
/// \param[out] ExitWeight The weight of the edge from Latch to Exit.
/// \param[out] FallThroughWeight The weight of the edge from Latch to Header.
static void initBranchWeights(BasicBlock *Header, BranchInst *LatchBR,
                              uint64_t &ExitWeight,
                              uint64_t &FallThroughWeight) {
  uint64_t TrueWeight, FalseWeight;
  if (!LatchBR->extractProfMetadata(TrueWeight, FalseWeight))
    return;
  unsigned HeaderIdx = LatchBR->getSuccessor(0) == Header ? 0 : 1;
  ExitWeight = HeaderIdx ? TrueWeight : FalseWeight;
  FallThroughWeight = HeaderIdx ? FalseWeight : TrueWeight;
}

/// Update the weights of original Latch block after peeling off all iterations.
///
/// \param Header The header block.
/// \param LatchBR The latch branch.
/// \param ExitWeight The weight of the edge from Latch to Exit.
/// \param FallThroughWeight The weight of the edge from Latch to Header.
static void fixupBranchWeights(BasicBlock *Header, BranchInst *LatchBR,
                               uint64_t ExitWeight,
                               uint64_t FallThroughWeight) {
  // FallThroughWeight is 0 means that there is no branch weights on original
  // latch block or estimated trip count is zero.
  if (!FallThroughWeight)
    return;

  // Sets the branch weights on the loop exit.
  MDBuilder MDB(LatchBR->getContext());
  unsigned HeaderIdx = LatchBR->getSuccessor(0) == Header ? 0 : 1;
  MDNode *WeightNode =
      HeaderIdx ? MDB.createBranchWeights(ExitWeight, FallThroughWeight)
                : MDB.createBranchWeights(FallThroughWeight, ExitWeight);
  LatchBR->setMetadata(LLVMContext::MD_prof, WeightNode);
}

/// Clones the body of the loop L, putting it between \p InsertTop and \p
/// InsertBot.
/// \param IterNumber The serial number of the iteration currently being
/// peeled off.
/// \param ExitEdges The exit edges of the original loop.
/// \param[out] NewBlocks A list of the blocks in the newly created clone
/// \param[out] VMap The value map between the loop and the new clone.
/// \param LoopBlocks A helper for DFS-traversal of the loop.
/// \param LVMap A value-map that maps instructions from the original loop to
/// instructions in the last peeled-off iteration.
static void cloneLoopBlocks(
    Loop *L, unsigned IterNumber, BasicBlock *InsertTop, BasicBlock *InsertBot,
    SmallVectorImpl<std::pair<BasicBlock *, BasicBlock *>> &ExitEdges,
    SmallVectorImpl<BasicBlock *> &NewBlocks, LoopBlocksDFS &LoopBlocks,
    ValueToValueMapTy &VMap, ValueToValueMapTy &LVMap, DominatorTree *DT,
    LoopInfo *LI, ArrayRef<MDNode *> LoopLocalNoAliasDeclScopes) {
  BasicBlock *Header = L->getHeader();
  BasicBlock *Latch = L->getLoopLatch();
  BasicBlock *PreHeader = L->getLoopPreheader();

  Function *F = Header->getParent();
  LoopBlocksDFS::RPOIterator BlockBegin = LoopBlocks.beginRPO();
  LoopBlocksDFS::RPOIterator BlockEnd = LoopBlocks.endRPO();
  Loop *ParentLoop = L->getParentLoop();

  // For each block in the original loop, create a new copy,
  // and update the value map with the newly created values.
  for (LoopBlocksDFS::RPOIterator BB = BlockBegin; BB != BlockEnd; ++BB) {
    BasicBlock *NewBB = CloneBasicBlock(*BB, VMap, ".peel", F);
    NewBlocks.push_back(NewBB);

    // If an original block is an immediate child of the loop L, its copy
    // is a child of a ParentLoop after peeling. If a block is a child of
    // a nested loop, it is handled in the cloneLoop() call below.
    if (ParentLoop && LI->getLoopFor(*BB) == L)
      ParentLoop->addBasicBlockToLoop(NewBB, *LI);

    VMap[*BB] = NewBB;

    // If dominator tree is available, insert nodes to represent cloned blocks.
    if (DT) {
      if (Header == *BB)
        DT->addNewBlock(NewBB, InsertTop);
      else {
        DomTreeNode *IDom = DT->getNode(*BB)->getIDom();
        // VMap must contain entry for IDom, as the iteration order is RPO.
        DT->addNewBlock(NewBB, cast<BasicBlock>(VMap[IDom->getBlock()]));
      }
    }
  }

  {
    // Identify what other metadata depends on the cloned version. After
    // cloning, replace the metadata with the corrected version for both
    // memory instructions and noalias intrinsics.
    std::string Ext = (Twine("Peel") + Twine(IterNumber)).str();
    cloneAndAdaptNoAliasScopes(LoopLocalNoAliasDeclScopes, NewBlocks,
                               Header->getContext(), Ext);
  }

  // Recursively create the new Loop objects for nested loops, if any,
  // to preserve LoopInfo.
  for (Loop *ChildLoop : *L) {
    cloneLoop(ChildLoop, ParentLoop, VMap, LI, nullptr);
  }

  // Hook-up the control flow for the newly inserted blocks.
  // The new header is hooked up directly to the "top", which is either
  // the original loop preheader (for the first iteration) or the previous
  // iteration's exiting block (for every other iteration)
  InsertTop->getTerminator()->setSuccessor(0, cast<BasicBlock>(VMap[Header]));

  // Similarly, for the latch:
  // The original exiting edge is still hooked up to the loop exit.
  // The backedge now goes to the "bottom", which is either the loop's real
  // header (for the last peeled iteration) or the copied header of the next
  // iteration (for every other iteration)
  BasicBlock *NewLatch = cast<BasicBlock>(VMap[Latch]);
  BranchInst *LatchBR = cast<BranchInst>(NewLatch->getTerminator());
  for (unsigned idx = 0, e = LatchBR->getNumSuccessors(); idx < e; ++idx)
    if (LatchBR->getSuccessor(idx) == Header) {
      LatchBR->setSuccessor(idx, InsertBot);
      break;
    }
  if (DT)
    DT->changeImmediateDominator(InsertBot, NewLatch);

  // The new copy of the loop body starts with a bunch of PHI nodes
  // that pick an incoming value from either the preheader, or the previous
  // loop iteration. Since this copy is no longer part of the loop, we
  // resolve this statically:
  // For the first iteration, we use the value from the preheader directly.
  // For any other iteration, we replace the phi with the value generated by
  // the immediately preceding clone of the loop body (which represents
  // the previous iteration).
  for (BasicBlock::iterator I = Header->begin(); isa<PHINode>(I); ++I) {
    PHINode *NewPHI = cast<PHINode>(VMap[&*I]);
    if (IterNumber == 0) {
      VMap[&*I] = NewPHI->getIncomingValueForBlock(PreHeader);
    } else {
      Value *LatchVal = NewPHI->getIncomingValueForBlock(Latch);
      Instruction *LatchInst = dyn_cast<Instruction>(LatchVal);
      if (LatchInst && L->contains(LatchInst))
        VMap[&*I] = LVMap[LatchInst];
      else
        VMap[&*I] = LatchVal;
    }
    cast<BasicBlock>(VMap[Header])->getInstList().erase(NewPHI);
  }

  // Fix up the outgoing values - we need to add a value for the iteration
  // we've just created. Note that this must happen *after* the incoming
  // values are adjusted, since the value going out of the latch may also be
  // a value coming into the header.
  for (auto Edge : ExitEdges)
    for (PHINode &PHI : Edge.second->phis()) {
      Value *LatchVal = PHI.getIncomingValueForBlock(Edge.first);
      Instruction *LatchInst = dyn_cast<Instruction>(LatchVal);
      if (LatchInst && L->contains(LatchInst))
        LatchVal = VMap[LatchVal];
      PHI.addIncoming(LatchVal, cast<BasicBlock>(VMap[Edge.first]));
    }

  // LastValueMap is updated with the values for the current loop
  // which are used the next time this function is called.
  for (auto KV : VMap)
    LVMap[KV.first] = KV.second;
}

TargetTransformInfo::PeelingPreferences llvm::gatherPeelingPreferences(
    Loop *L, ScalarEvolution &SE, const TargetTransformInfo &TTI,
    Optional<bool> UserAllowPeeling,
    Optional<bool> UserAllowProfileBasedPeeling, bool UnrollingSpecficValues) {
  TargetTransformInfo::PeelingPreferences PP;

  // Set the default values.
  PP.PeelCount = 0;
  PP.AllowPeeling = true;
  PP.AllowLoopNestsPeeling = false;
  PP.PeelProfiledIterations = true;

  // Get the target specifc values.
  TTI.getPeelingPreferences(L, SE, PP);

  // User specified values using cl::opt.
  if (UnrollingSpecficValues) {
    if (UnrollPeelCount.getNumOccurrences() > 0)
      PP.PeelCount = UnrollPeelCount;
    if (UnrollAllowPeeling.getNumOccurrences() > 0)
      PP.AllowPeeling = UnrollAllowPeeling;
    if (UnrollAllowLoopNestsPeeling.getNumOccurrences() > 0)
      PP.AllowLoopNestsPeeling = UnrollAllowLoopNestsPeeling;
  }

  // User specifed values provided by argument.
  if (UserAllowPeeling.hasValue())
    PP.AllowPeeling = *UserAllowPeeling;
  if (UserAllowProfileBasedPeeling.hasValue())
    PP.PeelProfiledIterations = *UserAllowProfileBasedPeeling;

  return PP;
}

/// Peel off the first \p PeelCount iterations of loop \p L.
///
/// Note that this does not peel them off as a single straight-line block.
/// Rather, each iteration is peeled off separately, and needs to check the
/// exit condition.
/// For loops that dynamically execute \p PeelCount iterations or less
/// this provides a benefit, since the peeled off iterations, which account
/// for the bulk of dynamic execution, can be further simplified by scalar
/// optimizations.
bool llvm::peelLoop(Loop *L, unsigned PeelCount, LoopInfo *LI,
                    ScalarEvolution *SE, DominatorTree *DT, AssumptionCache *AC,
                    bool PreserveLCSSA) {
  assert(PeelCount > 0 && "Attempt to peel out zero iterations?");
  assert(canPeel(L) && "Attempt to peel a loop which is not peelable?");

  LoopBlocksDFS LoopBlocks(L);
  LoopBlocks.perform(LI);

  BasicBlock *Header = L->getHeader();
  BasicBlock *PreHeader = L->getLoopPreheader();
  BasicBlock *Latch = L->getLoopLatch();
  SmallVector<std::pair<BasicBlock *, BasicBlock *>, 4> ExitEdges;
  L->getExitEdges(ExitEdges);

  DenseMap<BasicBlock *, BasicBlock *> ExitIDom;
  if (DT) {
    // We'd like to determine the idom of exit block after peeling one
    // iteration.
    // Let Exit is exit block.
    // Let ExitingSet - is a set of predecessors of Exit block. They are exiting
    // blocks.
    // Let Latch' and ExitingSet' are copies after a peeling.
    // We'd like to find an idom'(Exit) - idom of Exit after peeling.
    // It is an evident that idom'(Exit) will be the nearest common dominator
    // of ExitingSet and ExitingSet'.
    // idom(Exit) is a nearest common dominator of ExitingSet.
    // idom(Exit)' is a nearest common dominator of ExitingSet'.
    // Taking into account that we have a single Latch, Latch' will dominate
    // Header and idom(Exit).
    // So the idom'(Exit) is nearest common dominator of idom(Exit)' and Latch'.
    // All these basic blocks are in the same loop, so what we find is
    // (nearest common dominator of idom(Exit) and Latch)'.
    // In the loop below we remember nearest common dominator of idom(Exit) and
    // Latch to update idom of Exit later.
    assert(L->hasDedicatedExits() && "No dedicated exits?");
    for (auto Edge : ExitEdges) {
      if (ExitIDom.count(Edge.second))
        continue;
      BasicBlock *BB = DT->findNearestCommonDominator(
          DT->getNode(Edge.second)->getIDom()->getBlock(), Latch);
      assert(L->contains(BB) && "IDom is not in a loop");
      ExitIDom[Edge.second] = BB;
    }
  }

  Function *F = Header->getParent();

  // Set up all the necessary basic blocks. It is convenient to split the
  // preheader into 3 parts - two blocks to anchor the peeled copy of the loop
  // body, and a new preheader for the "real" loop.

  // Peeling the first iteration transforms.
  //
  // PreHeader:
  // ...
  // Header:
  //   LoopBody
  //   If (cond) goto Header
  // Exit:
  //
  // into
  //
  // InsertTop:
  //   LoopBody
  //   If (!cond) goto Exit
  // InsertBot:
  // NewPreHeader:
  // ...
  // Header:
  //  LoopBody
  //  If (cond) goto Header
  // Exit:
  //
  // Each following iteration will split the current bottom anchor in two,
  // and put the new copy of the loop body between these two blocks. That is,
  // after peeling another iteration from the example above, we'll split
  // InsertBot, and get:
  //
  // InsertTop:
  //   LoopBody
  //   If (!cond) goto Exit
  // InsertBot:
  //   LoopBody
  //   If (!cond) goto Exit
  // InsertBot.next:
  // NewPreHeader:
  // ...
  // Header:
  //  LoopBody
  //  If (cond) goto Header
  // Exit:

  BasicBlock *InsertTop = SplitEdge(PreHeader, Header, DT, LI);
  BasicBlock *InsertBot =
      SplitBlock(InsertTop, InsertTop->getTerminator(), DT, LI);
  BasicBlock *NewPreHeader =
      SplitBlock(InsertBot, InsertBot->getTerminator(), DT, LI);

  InsertTop->setName(Header->getName() + ".peel.begin");
  InsertBot->setName(Header->getName() + ".peel.next");
  NewPreHeader->setName(PreHeader->getName() + ".peel.newph");

  ValueToValueMapTy LVMap;

  // If we have branch weight information, we'll want to update it for the
  // newly created branches.
  BranchInst *LatchBR =
      cast<BranchInst>(cast<BasicBlock>(Latch)->getTerminator());
  uint64_t ExitWeight = 0, FallThroughWeight = 0;
  initBranchWeights(Header, LatchBR, ExitWeight, FallThroughWeight);

  // Identify what noalias metadata is inside the loop: if it is inside the
  // loop, the associated metadata must be cloned for each iteration.
  SmallVector<MDNode *, 6> LoopLocalNoAliasDeclScopes;
  identifyNoAliasScopesToClone(L->getBlocks(), LoopLocalNoAliasDeclScopes);

  // For each peeled-off iteration, make a copy of the loop.
  for (unsigned Iter = 0; Iter < PeelCount; ++Iter) {
    SmallVector<BasicBlock *, 8> NewBlocks;
    ValueToValueMapTy VMap;

    cloneLoopBlocks(L, Iter, InsertTop, InsertBot, ExitEdges, NewBlocks,
                    LoopBlocks, VMap, LVMap, DT, LI,
                    LoopLocalNoAliasDeclScopes);

    // Remap to use values from the current iteration instead of the
    // previous one.
    remapInstructionsInBlocks(NewBlocks, VMap);

    if (DT) {
      // Latches of the cloned loops dominate over the loop exit, so idom of the
      // latter is the first cloned loop body, as original PreHeader dominates
      // the original loop body.
      if (Iter == 0)
        for (auto Exit : ExitIDom)
          DT->changeImmediateDominator(Exit.first,
                                       cast<BasicBlock>(LVMap[Exit.second]));
#ifdef EXPENSIVE_CHECKS
      assert(DT->verify(DominatorTree::VerificationLevel::Fast));
#endif
    }

    auto *LatchBRCopy = cast<BranchInst>(VMap[LatchBR]);
    updateBranchWeights(InsertBot, LatchBRCopy, ExitWeight, FallThroughWeight);
    // Remove Loop metadata from the latch branch instruction
    // because it is not the Loop's latch branch anymore.
    LatchBRCopy->setMetadata(LLVMContext::MD_loop, nullptr);

    InsertTop = InsertBot;
    InsertBot = SplitBlock(InsertBot, InsertBot->getTerminator(), DT, LI);
    InsertBot->setName(Header->getName() + ".peel.next");

    F->getBasicBlockList().splice(InsertTop->getIterator(),
                                  F->getBasicBlockList(),
                                  NewBlocks[0]->getIterator(), F->end());
  }

  // Now adjust the phi nodes in the loop header to get their initial values
  // from the last peeled-off iteration instead of the preheader.
  for (BasicBlock::iterator I = Header->begin(); isa<PHINode>(I); ++I) {
    PHINode *PHI = cast<PHINode>(I);
    Value *NewVal = PHI->getIncomingValueForBlock(Latch);
    Instruction *LatchInst = dyn_cast<Instruction>(NewVal);
    if (LatchInst && L->contains(LatchInst))
      NewVal = LVMap[LatchInst];

    PHI->setIncomingValueForBlock(NewPreHeader, NewVal);
  }

  fixupBranchWeights(Header, LatchBR, ExitWeight, FallThroughWeight);

  // Update Metadata for count of peeled off iterations.
  unsigned AlreadyPeeled = 0;
  if (auto Peeled = getOptionalIntLoopAttribute(L, PeeledCountMetaData))
    AlreadyPeeled = *Peeled;
  addStringMetadataToLoop(L, PeeledCountMetaData, AlreadyPeeled + PeelCount);

  if (Loop *ParentLoop = L->getParentLoop())
    L = ParentLoop;

  // We modified the loop, update SE.
  SE->forgetTopmostLoop(L);

  // Finally DomtTree must be correct.
  assert(DT->verify(DominatorTree::VerificationLevel::Fast));

  // FIXME: Incrementally update loop-simplify
  simplifyLoop(L, DT, LI, SE, AC, nullptr, PreserveLCSSA);

  NumPeeled++;

  return true;
}
