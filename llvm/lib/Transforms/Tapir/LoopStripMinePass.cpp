//===-- LoopStripMinePass.cpp - Loop strip-mining pass --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to perform Tapir loop strip-mining.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Tapir/LoopStripMinePass.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/CodeMetrics.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TapirTaskInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Tapir.h"
#include "llvm/Transforms/Tapir/LoopStripMine.h"
#include "llvm/Transforms/Utils.h"
#include "llvm/Transforms/Utils/LoopSimplify.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/TapirUtils.h"

using namespace llvm;

#define DEBUG_TYPE "loop-stripmine"

static cl::opt<unsigned> StripMineCount(
    "stripmine-count", cl::Hidden,
    cl::desc("Use this stripmine count for all loops, for testing purposes"));

static cl::opt<unsigned> StripMineCoarseningFactor(
    "stripmine-coarsen-factor", cl::Hidden,
    cl::desc("Use this coarsening factor for stripmining"));

static cl::opt<bool> StripMineUnrollRemainder(
  "stripmine-unroll-remainder", cl::Hidden,
  cl::desc("Allow the loop remainder after stripmining to be unrolled."));

/// Constants for stripmining cost analysis.
namespace StripMineConstants {
/// Default coarsening factor for strpimined Tapir loops.
const unsigned DefaultCoarseningFactor = 2048;
}

/// Create an analysis remark that explains why stripmining failed
///
/// \p RemarkName is the identifier for the remark.  If \p I is passed it is an
/// instruction that prevents vectorization.  Otherwise \p TheLoop is used for
/// the location of the remark.  \return the remark object that can be streamed
/// to.
static OptimizationRemarkAnalysis
createMissedAnalysis(StringRef RemarkName, const Loop *TheLoop,
                     Instruction *I = nullptr) {
  const Value *CodeRegion = TheLoop->getHeader();
  DebugLoc DL = TheLoop->getStartLoc();

  if (I) {
    CodeRegion = I->getParent();
    // If there is no debug location attached to the instruction, revert back to
    // using the loop's.
    if (I->getDebugLoc())
      DL = I->getDebugLoc();
  }

  OptimizationRemarkAnalysis R("loop-stripmine", RemarkName, DL, CodeRegion);
  R << "Tapir loop not transformed: ";
  return R;
}

/// Gather the various unrolling parameters based on the defaults, compiler
/// flags, TTI overrides and user specified parameters.
TargetTransformInfo::StripMiningPreferences llvm::gatherStripMiningPreferences(
    Loop *L, ScalarEvolution &SE, const TargetTransformInfo &TTI,
    Optional<unsigned> UserCount) {
  TargetTransformInfo::StripMiningPreferences SMP;

  // Set up the defaults
  SMP.Count = 0;
  SMP.AllowExpensiveTripCount = false;
  SMP.DefaultCoarseningFactor =
    (StripMineCoarseningFactor.getNumOccurrences() > 0) ?
    StripMineCoarseningFactor : StripMineConstants::DefaultCoarseningFactor;
  SMP.UnrollRemainder = false;

  // Override with any target specific settings
  TTI.getStripMiningPreferences(L, SE, SMP);

  // Apply any user values specified by cl::opt
  if (UserCount.hasValue())
    SMP.Count = *UserCount;
  if (StripMineUnrollRemainder.getNumOccurrences() > 0)
    SMP.UnrollRemainder = StripMineUnrollRemainder;

  return SMP;
}

// Helper method to get a constant trip count for the given loop.
static unsigned getConstTripCount(const Loop *L, ScalarEvolution &SE) {
  unsigned ConstTripCount = 0;
  // If there are multiple exiting blocks but one of them is the latch, use
  // the latch for the trip count estimation. Otherwise insist on a single
  // exiting block for the trip count estimation.
  BasicBlock *ExitingBlock = L->getLoopLatch();
  if (!ExitingBlock || !L->isLoopExiting(ExitingBlock))
    ExitingBlock = L->getExitingBlock();
  if (ExitingBlock)
    ConstTripCount = SE.getSmallConstantTripCount(L, ExitingBlock);
  return ConstTripCount;
}

/// Recursive helper routine to estimate the amount of work in a loop.
static unsigned ApproximateLoopSizeHelper(const Loop *L, CodeMetrics &Metrics,
                                          bool &UnknownSize, LoopInfo *LI,
                                          ScalarEvolution &SE) {
  if (UnknownSize)
    return std::numeric_limits<unsigned>::max();

  // TODO: Handle control flow within the loop more intelligently.
  unsigned LoopSize = 0;
  for (Loop *SubL : *L) {
    unsigned SubLoopSize = ApproximateLoopSizeHelper(SubL, Metrics, UnknownSize,
                                                     LI, SE);
    // Quit early if the size of this subloop is already too big.
    if (std::numeric_limits<unsigned>::max() == SubLoopSize)
      LoopSize = std::numeric_limits<unsigned>::max();

    // Find a constant trip count if available
    unsigned ConstTripCount = getConstTripCount(SubL, SE);
    // TODO: Use a more precise analysis to account for non-constant trip
    // counts.
    if (!ConstTripCount) {
      UnknownSize = true;
      // If we cannot compute a constant trip count, assume this subloop
      // executes at least once.
      ConstTripCount = 1;
    }

    // Check if the total size of this subloop is huge.
    if (std::numeric_limits<unsigned>::max() / ConstTripCount > SubLoopSize)
      LoopSize = std::numeric_limits<unsigned>::max();

    // Check if this subloop suffices to make loop L huge.
    if (std::numeric_limits<unsigned>::max() - LoopSize <
        (SubLoopSize * ConstTripCount))
      LoopSize = std::numeric_limits<unsigned>::max();

    // Add in the size of this subloop.
    LoopSize += (SubLoopSize * ConstTripCount);
  }

  // After looking at all subloops, if we've concluded we have a huge loop size,
  // return early.
  if (std::numeric_limits<unsigned>::max() == LoopSize)
    return LoopSize;

  for (BasicBlock *BB : L->blocks())
    if (LI->getLoopFor(BB) == L) {
      // Check if this BB suffices to make loop L huge.
      if (std::numeric_limits<unsigned>::max() - LoopSize <
          Metrics.NumBBInsts[BB])
        return std::numeric_limits<unsigned>::max();
      LoopSize += Metrics.NumBBInsts[BB];
    }

  return LoopSize;
}

/// ApproximateLoopSize - Approximate the size of the loop.
unsigned llvm::ApproximateLoopSize(
    const Loop *L, unsigned &NumCalls, bool &NotDuplicatable,
    bool &Convergent, bool &IsRecursive, bool &UnknownSize,
    const TargetTransformInfo &TTI, LoopInfo *LI, ScalarEvolution &SE,
    const SmallPtrSetImpl<const Value *> &EphValues,
    TargetLibraryInfo *TLI) {

  // TODO: Use more precise analysis to estimate the work in each call.
  // TODO: Use vectorizability to enhance cost analysis.

  CodeMetrics Metrics;
  for (BasicBlock *BB : L->blocks())
    Metrics.analyzeBasicBlock(BB, TTI, EphValues, TLI);
  // Exclude calls to builtins when counting the calls.  This assumes that all
  // builtin functions are cheap.
  NumCalls = Metrics.NumCalls - Metrics.NumBuiltinCalls;
  NotDuplicatable = Metrics.notDuplicatable;
  Convergent = Metrics.convergent;
  IsRecursive = Metrics.isRecursive;

  unsigned LoopSize = ApproximateLoopSizeHelper(L, Metrics, UnknownSize, LI,
                                                SE);

  return LoopSize;
}

// Returns the loop hint metadata node with the given name (for example,
// "tapir.loop.stripmine.count").  If no such metadata node exists, then nullptr
// is returned.
static MDNode *GetStripMineMetadataForLoop(const Loop *L, StringRef Name) {
  if (MDNode *LoopID = L->getLoopID())
    return GetStripMineMetadata(LoopID, Name);
  return nullptr;
}

// Returns true if the loop has an stripmine(disable) pragma.
static bool HasStripMineDisablePragma(const Loop *L) {
  return GetStripMineMetadataForLoop(L, "tapir.loop.stripmine.disable");
}

// Returns true if the loop has an stripmine(enable) pragma.
static bool HasStripMineEnablePragma(const Loop *L) {
  return GetStripMineMetadataForLoop(L, "tapir.loop.stripmine.enable");
}

// If loop has an grainsize pragma return the (necessarily positive) value from
// the pragma for stripmining.  Otherwise return 0.
static unsigned StripMineCountPragmaValue(const Loop *L) {
  TapirLoopHints Hints(L);
  return Hints.getGrainsize();
}

// Returns true if stripmine count was set explicitly.
// Calculates stripmine count and writes it to SMP.Count.
bool llvm::computeStripMineCount(
    Loop *L, const TargetTransformInfo &TTI, unsigned LoopSize,
    TargetTransformInfo::StripMiningPreferences &SMP) {
  // Check for explicit Count.
  // 1st priority is stripmine count set by "stripmine-count" option.
  bool UserStripMineCount = StripMineCount.getNumOccurrences() > 0;
  if (UserStripMineCount) {
    SMP.Count = StripMineCount;
    SMP.AllowExpensiveTripCount = true;
    return true;
  }

  // 2nd priority is stripmine count set by pragma.
  unsigned PragmaCount = StripMineCountPragmaValue(L);
  if (PragmaCount > 0) {
    SMP.Count = PragmaCount;
    SMP.AllowExpensiveTripCount = true;
    return true;
  }

  // 3rd priority is computed stripmine count.
  //
  // We want to coarsen the loop such that the work of detaching a loop
  // iteration is tiny compared to the work of the loop body.  Specifically, we
  // want the total cost of the parallel loop to be at most (1 + \eps) times the
  // cost of its serial projection.  Let G is the grainsize, n the number of
  // loop iterations, d the cost of a detach, and S the work of the loop body.
  // Then we want
  //
  //   (n/G)(G*S + d) <= (1 + \eps)(n * S)
  //
  // Solving for G yeilds G >= d/(\eps * S).  Substituting in \eps = 1/C for a
  // given coarsening factor C gives the equation below.
  Instruction *DetachI = L->getHeader()->getTerminator();
  SMP.Count = SMP.DefaultCoarseningFactor * TTI.getUserCost(DetachI) / LoopSize;

  return false;
}

static bool tryToStripMineLoop(
    Loop *L, DominatorTree &DT, LoopInfo *LI, ScalarEvolution &SE,
    const TargetTransformInfo &TTI, AssumptionCache &AC, TaskInfo *TI,
    OptimizationRemarkEmitter &ORE, bool PreserveLCSSA,
    Optional<unsigned> ProvidedCount) {
  if (!getTaskIfTapirLoop(L, TI))
    return false;
  TapirLoopHints Hints(L);

  LLVM_DEBUG(dbgs() << "Loop Strip Mine: F["
                    << L->getHeader()->getParent()->getName() << "] Loop %"
                    << L->getHeader()->getName() << "\n");
  if (HasStripMineDisablePragma(L))
    return false;
  if (!L->isLoopSimplifyForm()) {
    LLVM_DEBUG(
        dbgs() << "  Not stripmining loop which is not in loop-simplify form.\n");
    return false;
  }
  bool StripMiningRequested = HasStripMineEnablePragma(L);
  TargetTransformInfo::StripMiningPreferences SMP = gatherStripMiningPreferences(
      L, SE, TTI, ProvidedCount);

  unsigned NumCalls = 0;
  bool NotDuplicatable = false;
  bool Convergent = false;
  bool IsRecursive = false;
  bool UnknownSize = false;

  SmallPtrSet<const Value *, 32> EphValues;
  CodeMetrics::collectEphemeralValues(L, &AC, EphValues);

  unsigned LoopSize =
      ApproximateLoopSize(L, NumCalls, NotDuplicatable, Convergent, IsRecursive,
                          UnknownSize, TTI, LI, SE, EphValues);
  // computeStripMineCount() determines the count to stripmine the loop.
  bool explicitCount = computeStripMineCount(L, TTI, LoopSize, SMP);

  // If the loop size is unknown, then we cannot compute a stripmining count for
  // it.
  if (!explicitCount && UnknownSize) {
    LLVM_DEBUG(dbgs() << "  Not stripmining loop with unknown size.\n");
    if (StripMiningRequested)
      ORE.emit(DiagnosticInfoOptimizationFailure(
                   DEBUG_TYPE, "UnknownSize",
                   L->getStartLoc(), L->getHeader())
               << "Cannot stripmine loop with unknown size.");
    return false;
  }

  // If the loop size is enormous, then we might want to use a stripmining count
  // of 1 for it.
  LLVM_DEBUG(dbgs() << "  Loop Size = " << LoopSize << "\n");
  if (!explicitCount && std::numeric_limits<unsigned>::max() == LoopSize) {
    LLVM_DEBUG(dbgs() << "  Not stripmining loop with very large size.\n");
    if (Hints.getGrainsize() == 1)
      return false;
    ORE.emit([&]() {
               return OptimizationRemark("loop-stripmine", "HugeLoop",
                                         L->getStartLoc(), L->getHeader())
                 << "using grainsize 1 for huge loop";
             });
    Hints.setAlreadyStripMined();
    return true;
  }

  // If the loop is recursive, set the stripmine factor to be 1.
  if (!explicitCount && IsRecursive) {
    LLVM_DEBUG(dbgs() << "  Not stripmining loop that recursively calls the "
                      << "containing function.\n");
    if (Hints.getGrainsize() == 1)
      return false;
    ORE.emit([&]() {
               return OptimizationRemark("loop-stripmine", "RecursiveCalls",
                                         L->getStartLoc(), L->getHeader())
                 << "using grainsize 1 for loop with recursive calls";
             });
    Hints.setAlreadyStripMined();
    return true;
  }

  // TODO: We can stripmine loops if the stripmined version does not require a
  // prolog or epilog.
  if (NotDuplicatable) {
    LLVM_DEBUG(dbgs() << "  Not stripmining loop which contains "
                      << "non-duplicatable instructions.\n");
    if (explicitCount || StripMiningRequested)
      ORE.emit(DiagnosticInfoOptimizationFailure(
                   DEBUG_TYPE, "NotDuplicatable",
                   L->getStartLoc(), L->getHeader())
               << "Cannot stripmine loop with non-duplicatable instructions.");
    return false;
  }

  // If the loop contains a convergent operation, then the control flow
  // introduced between the stripmined loop and epilog is unsafe -- it adds a
  // control-flow dependency to the convergent operation.
  if (Convergent) {
    LLVM_DEBUG(dbgs() << "  Skipping loop with convergent operations.\n");
    if (explicitCount || StripMiningRequested)
      ORE.emit(DiagnosticInfoOptimizationFailure(
                   DEBUG_TYPE, "Convergent",
                   L->getStartLoc(), L->getHeader())
               << "Cannot stripmine loop with convergent instructions.");
    return false;
  }

  // If the loop contains potentially expensive function calls, then we don't
  // want to stripmine it.
  if (NumCalls > 0 && !explicitCount && !StripMiningRequested) {
    LLVM_DEBUG(dbgs() << "  Skipping loop with expensive function calls.\n");
    ORE.emit(createMissedAnalysis("ExpensiveCalls", L)
             << "Not stripmining loop with potentially expensive calls.");
    return false;
  }

  // Make sure the count is a power of 2.
  if (!isPowerOf2_32(SMP.Count))
    SMP.Count = NextPowerOf2(SMP.Count);
  if (SMP.Count < 2) {
    if (Hints.getGrainsize() == 1)
      return false;
    ORE.emit([&]() {
               return OptimizationRemark("loop-stripmine", "LargeLoop",
                                         L->getStartLoc(), L->getHeader())
                 << "using grainsize 1 for large loop";
             });
    Hints.setAlreadyStripMined();
    return true;
  }

  // Find a constant trip count if available
  unsigned ConstTripCount = getConstTripCount(L, SE);

  // Stripmining factor (Count) must be less or equal to TripCount.
  if (ConstTripCount && SMP.Count >= ConstTripCount) {
    ORE.emit(createMissedAnalysis("FullStripMine", L)
             << "Stripmining count larger than loop trip count.");
    ORE.emit(DiagnosticInfoOptimizationFailure(
                 DEBUG_TYPE, "UnprofitableParallelLoop",
                 L->getStartLoc(), L->getHeader())
             << "Parallel loop appears to be unprofitable to parallelize.");
    return false;
  }

  // When is it worthwhile to allow the epilog to run in parallel with the
  // stripmined loop?  We expect the epilog to perform G/2 iterations on
  // average, where G is the selected grainsize.  Our goal is to ensure that
  // these G/2 iterations offset the cost of an additional detach.  We have
  // already computed G = \pow_2_floor(C * d / S), where C is the coarsening
  // factor (a power of 2), d is the cost of a detach, and S is the work of the
  // loop body, and \pow_2_floor gives the floow of the value rounded to the
  // next power of 2.  We distinguish two cases.
  //
  // 1) If G < C, then due to rounding, have that d / S < 1/2, implying that
  // S > 2d.  Hence it's profitable to allow the epilog to execute in parallel.
  //
  // 2) If G == C, then to satisfy (G/2) * S + d <= (1 + \eps) * G/2 * S, where
  // \eps = 1/C, we require S >= 2 * d.
  //
  // We check for these two cases and encode the result in ParallelEpilog.
  bool ParallelEpilog = (SMP.Count < SMP.DefaultCoarseningFactor) ||
    (LoopSize >= 2 * TTI.getUserCost(L->getHeader()->getTerminator()));
  Loop *NewLoop = StripMineLoop(L, SMP.Count, SMP.AllowExpensiveTripCount,
                                SMP.UnrollRemainder, LI, &SE, &DT, &AC, TI,
                                &ORE, PreserveLCSSA, ParallelEpilog);
  if (!NewLoop)
    return false;

  // Mark the new loop as stripmined.
  TapirLoopHints NewHints(NewLoop);
  NewHints.setAlreadyStripMined();

  return true;
}

namespace {

class LoopStripMine : public LoopPass {
public:
  static char ID; // Pass ID, replacement for typeid

  Optional<unsigned> ProvidedCount;

  LoopStripMine(Optional<unsigned> Count = None)
      : LoopPass(ID), ProvidedCount(Count) {
    initializeLoopStripMinePass(*PassRegistry::getPassRegistry());
  }

  bool runOnLoop(Loop *L, LPPassManager &LPM) override {
    if (skipLoop(L))
      return false;

    Function &F = *L->getHeader()->getParent();

    auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    LoopInfo *LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
    TaskInfo *TI = &getAnalysis<TaskInfoWrapperPass>().getTaskInfo();
    ScalarEvolution &SE = getAnalysis<ScalarEvolutionWrapperPass>().getSE();
    const TargetTransformInfo &TTI =
        getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);
    auto &AC = getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F);
    // For the old PM, we can't use OptimizationRemarkEmitter as an analysis
    // pass.  Function analyses need to be preserved across loop transformations
    // but ORE cannot be preserved (see comment before the pass definition).
    OptimizationRemarkEmitter ORE(&F);
    bool PreserveLCSSA = mustPreserveAnalysisID(LCSSAID);

    return tryToStripMineLoop(L, DT, LI, SE, TTI, AC, TI, ORE, PreserveLCSSA,
                              ProvidedCount);
  }

  /// This transformation requires natural loop information & requires that
  /// loop preheaders be inserted into the CFG...
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<AssumptionCacheTracker>();
    AU.addRequired<TargetTransformInfoWrapperPass>();
    getLoopAnalysisUsage(AU);
  }
};

} // end anonymous namespace

char LoopStripMine::ID = 0;

INITIALIZE_PASS_BEGIN(LoopStripMine, "loop-stripmine", "Stripmine Tapir loops",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_DEPENDENCY(LoopPass)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_END(LoopStripMine, "loop-stripmine", "Stripmine Tapir loops",
                    false, false)

Pass *llvm::createLoopStripMinePass(int Count) {
  // TODO: It would make more sense for this function to take the optionals
  // directly, but that's dangerous since it would silently break out of tree
  // callers.
  return new LoopStripMine(
      Count == -1 ? None : Optional<unsigned>(Count));
}

template <typename RangeT>
static SmallVector<Loop *, 8> appendLoopsToWorklist(RangeT &&Loops) {
  SmallVector<Loop *, 8> Worklist;
  // We use an internal worklist to build up the preorder traversal without
  // recursion.
  SmallVector<Loop *, 4> PreOrderLoops, PreOrderWorklist;

  for (Loop *RootL : Loops) {
    assert(PreOrderLoops.empty() && "Must start with an empty preorder walk.");
    assert(PreOrderWorklist.empty() &&
           "Must start with an empty preorder walk worklist.");
    PreOrderWorklist.push_back(RootL);
    do {
      Loop *L = PreOrderWorklist.pop_back_val();
      PreOrderWorklist.append(L->begin(), L->end());
      PreOrderLoops.push_back(L);
    } while (!PreOrderWorklist.empty());

    Worklist.append(PreOrderLoops.begin(), PreOrderLoops.end());
    PreOrderLoops.clear();
  }
  return Worklist;
}

PreservedAnalyses LoopStripMinePass::run(Function &F,
                                         FunctionAnalysisManager &AM) {
  auto &SE = AM.getResult<ScalarEvolutionAnalysis>(F);
  auto &LI = AM.getResult<LoopAnalysis>(F);
  auto &TTI = AM.getResult<TargetIRAnalysis>(F);
  auto &DT = AM.getResult<DominatorTreeAnalysis>(F);
  auto &AC = AM.getResult<AssumptionAnalysis>(F);
  auto &TI = AM.getResult<TaskAnalysis>(F);
  auto &ORE = AM.getResult<OptimizationRemarkEmitterAnalysis>(F);

  LoopAnalysisManager *LAM = nullptr;
  if (auto *LAMProxy = AM.getCachedResult<LoopAnalysisManagerFunctionProxy>(F))
    LAM = &LAMProxy->getManager();

  // const ModuleAnalysisManager &MAM =
  //     AM.getResult<ModuleAnalysisManagerFunctionProxy>(F).getManager();
  // ProfileSummaryInfo *PSI =
  //     MAM.getCachedResult<ProfileSummaryAnalysis>(*F.getParent());

  bool Changed = false;

  // The stripminer requires loops to be in simplified form, and also needs
  // LCSSA.  Since simplification may add new inner loops, it has to run before
  // the legality and profitability checks. This means running the loop
  // stripminer will simplify all loops, regardless of whether anything end up
  // being stripmined.
  for (auto &L : LI) {
    Changed |= simplifyLoop(L, &DT, &LI, &SE, &AC, false /* PreserveLCSSA */);
    Changed |= formLCSSARecursively(*L, DT, &LI, &SE);
  }

  SmallVector<Loop *, 8> Worklist = appendLoopsToWorklist(LI);

  while (!Worklist.empty()) {
    // Because the LoopInfo stores the loops in RPO, we walk the worklist from
    // back to front so that we work forward across the CFG, which for
    // stripmining is only needed to get optimization remarks emitted in a
    // forward order.
    Loop &L = *Worklist.pop_back_val();
#ifndef NDEBUG
    Loop *ParentL = L.getParentLoop();
#endif

    // // Check if the profile summary indicates that the profiled application
    // // has a huge working set size, in which case we disable peeling to avoid
    // // bloating it further.
    // if (PSI && PSI->hasHugeWorkingSetSize())
    //   AllowPeeling = false;
    std::string LoopName = L.getName();
    bool LoopChanged =
      tryToStripMineLoop(&L, DT, &LI, SE, TTI, AC, &TI, ORE,
                         /*PreserveLCSSA*/ true, /*Count*/ None);
    Changed |= LoopChanged;

    // The parent must not be damaged by stripmining!
#ifndef NDEBUG
    if (LoopChanged && ParentL)
      ParentL->verifyLoop();
#endif

    // Clear any cached analysis results for L if we removed it completely.
    if (LAM && LoopChanged)
      LAM->clear(L, LoopName);
  }

  if (!Changed)
    return PreservedAnalyses::all();

  return getLoopPassPreservedAnalyses();
}
