#include "llvm/Transforms/Scalar/LoopUnrollForVectorization.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/LoopSimplify.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/UnrollLoop.h"

using namespace llvm;

#define DEBUG_TYPE "loop-unroll-for-vectorization"

static cl::opt<unsigned> UnrollForVecMaxTripCount(
    "unroll-for-vec-max-trip-count", cl::init(16), cl::Hidden,
    cl::desc("Maximum trip count of inner loops to unroll for outer loop "
             "vectorization"));

static cl::opt<unsigned> UnrollForVecMaxUnrolledSize(
    "unroll-for-vec-max-unrolled-size", cl::init(4096), cl::Hidden,
    cl::desc("Maximum total unrolled instruction count for inner loop "
             "unrolling to enable outer loop vectorization"));

/// Collect ALL sub-loops under Parent that should be unrolled,
/// skipping loops that carry the vectorize.enable hint (those are targets, not
/// candidates). Loops are collected in post-order (innermost first) so they
/// can be unrolled bottom-up.
/// Returns false if any sub-loop has a non-constant or too-large trip count,
/// meaning the nest cannot be fully unrolled.
static bool collectAllUnrollCandidates(Loop *Parent, ScalarEvolution &SE,
                                       SmallVectorImpl<Loop *> &Candidates) {
  for (Loop *Sub : *Parent) {
    // Vectorization target itself is not an unroll candidate.
    if (getBooleanLoopAttribute(Sub, "llvm.loop.vectorize.enable"))
      continue;

    if (!Sub->isInnermost()) {
      if (!collectAllUnrollCandidates(Sub, SE, Candidates))
        return false;
    }

    unsigned TripCount = SE.getSmallConstantTripCount(Sub);
    if (TripCount == 0 || TripCount > UnrollForVecMaxTripCount)
      return false;

    Candidates.push_back(Sub);
  }
  return true;
}

/// Estimate the instruction count of a single loop's own body blocks
static unsigned estimateLoopOwnSize(const Loop *L) {
  unsigned Size = 0;
  for (BasicBlock *BB : L->blocks()) {
    // Only count blocks at this loop level, not in sub-loops.
    if (L->isInnermost() ||
        std::none_of(L->begin(), L->end(),
                     [BB](const Loop *Sub) { return Sub->contains(BB); }))
      Size += BB->size();
  }
  return Size;
}

/// Compute an estimated total instruction count after fully unrolling all
/// candidates.
static unsigned
computeCompoundUnrolledSize(const SmallVectorImpl<Loop *> &Candidates,
                            ScalarEvolution &SE) {
  DenseMap<const Loop *, unsigned> UnrolledSize;

  for (Loop *L : Candidates) {
    unsigned OwnSize = estimateLoopOwnSize(L);
    unsigned ChildrenSize = 0;
    for (Loop *Child : *L) {
      if (UnrolledSize.count(Child)) {
        ChildrenSize += UnrolledSize[Child];
      } else {
        // Child was not a candidate (e.g. has vectorize.enable hint).
        // Its blocks are excluded from OwnSize, so account for them here.
        // When outer loop vectorization is further along we may want to do
        // additional analysis here
        LLVM_DEBUG(
            dbgs() << "Child loop " << Child->getHeader()->getName()
                   << " is not a candidate (possible vectorize.enable hint),"
                   << " adding its blocks to children size\n");
        unsigned ChildBlockSize = 0;
        for (BasicBlock *BB : Child->blocks())
          ChildBlockSize += BB->size();
        ChildrenSize += ChildBlockSize;
      }
    }
    unsigned TripCount = SE.getSmallConstantTripCount(L);
    UnrolledSize[L] = (OwnSize + ChildrenSize) * TripCount;
  }

  SmallPtrSet<const Loop *, 8> CandidateSet(Candidates.begin(),
                                            Candidates.end());
  unsigned Total = 0;
  for (Loop *L : Candidates) {
    if (!CandidateSet.count(L->getParentLoop()))
      Total += UnrolledSize[L];
  }
  return Total;
}

PreservedAnalyses
LoopUnrollForVectorizationPass::run(Function &F, FunctionAnalysisManager &AM) {
  auto &LI = AM.getResult<LoopAnalysis>(F);
  auto &SE = AM.getResult<ScalarEvolutionAnalysis>(F);
  auto &DT = AM.getResult<DominatorTreeAnalysis>(F);
  auto &AC = AM.getResult<AssumptionAnalysis>(F);
  auto &TTI = AM.getResult<TargetIRAnalysis>(F);
  auto &ORE = AM.getResult<OptimizationRemarkEmitterAnalysis>(F);

  bool Changed = false;

  // Find loops with explicit vectorization hints. These are the targets
  // whose inner loops we want to fully unroll.
  SmallVector<Loop *, 4> VecTargets;
  for (Loop *TopLevel : LI) {
    SmallVector<Loop *, 8> Worklist;
    Worklist.push_back(TopLevel);
    while (!Worklist.empty()) {
      Loop *L = Worklist.pop_back_val();
      if (getBooleanLoopAttribute(L, "llvm.loop.vectorize.enable"))
        VecTargets.push_back(L);
      for (Loop *Sub : *L)
        Worklist.push_back(Sub);
    }
  }

  if (VecTargets.empty())
    return PreservedAnalyses::all();

  LLVM_DEBUG(dbgs() << "LoopUnrollForVec: Found " << VecTargets.size()
                    << " outer loop(s) with unrollable inner loops in "
                    << F.getName() << "\n");

  // For each vectorization target, collect all inner loops to unroll,
  // compute the compound cost, and either unroll all or none.
  for (Loop *Target : VecTargets) {
    if (Target->isInnermost())
      continue; // Nothing to unroll inside an innermost loop.

    SmallVector<Loop *, 8> Candidates;
    if (!collectAllUnrollCandidates(Target, SE, Candidates)) {
      LLVM_DEBUG(dbgs() << "LoopUnrollForVec: Cannot fully unroll nest at "
                        << Target->getHeader()->getName()
                        << " (non-constant or too-large trip count)\n");
      continue;
    }

    if (Candidates.empty())
      continue;

    unsigned TotalSize = computeCompoundUnrolledSize(Candidates, SE);
    if (TotalSize > UnrollForVecMaxUnrolledSize) {
      LLVM_DEBUG(dbgs() << "LoopUnrollForVec: Skipping nest at "
                        << Target->getHeader()->getName()
                        << " (compound unrolled size " << TotalSize
                        << " exceeds limit " << UnrollForVecMaxUnrolledSize
                        << ")\n");
      continue;
    }

    LLVM_DEBUG(dbgs() << "LoopUnrollForVec: " << Candidates.size()
                      << " inner loop candidate(s) in outer loop at "
                      << Target->getHeader()->getName() << " (compound size "
                      << TotalSize << ")\n");

    // Unroll all candidates bottom-up (they are already in post-order).
    // Sibling loops are independent so unrolling one doesn't invalidate
    // another. Each loop is re-simplified right before unrolling it.
    for (Loop *L : Candidates) {
      unsigned TripCount = SE.getSmallConstantTripCount(L);
      if (TripCount == 0)
        continue;

      LLVM_DEBUG(dbgs() << "LoopUnrollForVec: Unrolling loop at "
                        << L->getHeader()->getName() << " (trip count "
                        << TripCount << ")\n");

      UnrollLoopOptions ULO;
      ULO.Count = TripCount;
      ULO.Force = true;
      ULO.Runtime = false;
      ULO.AllowExpensiveTripCount = false;
      ULO.UnrollRemainder = false;
      ULO.ForgetAllSCEV = true;
      ULO.SCEVExpansionBudget = 0;

      // Re-simplify this loop in case a child's unroll broke its form.
      Changed |= simplifyLoop(L, &DT, &LI, &SE, &AC, nullptr, false);
      Changed |= formLCSSARecursively(*L, DT, &LI, &SE);

      LoopUnrollResult Result =
          UnrollLoop(L, ULO, &LI, &SE, &DT, &AC, &TTI, &ORE,
                     /*PreserveLCSSA=*/true);

      if (Result != LoopUnrollResult::Unmodified)
        Changed = true;
    }
  }

  if (!Changed)
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserve<LoopAnalysis>();
  PA.preserve<DominatorTreeAnalysis>();
  return PA;
}
