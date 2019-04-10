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
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/WorkSpanAnalysis.h"
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

static cl::opt<bool> AllowParallelEpilog(
  "allow-parallel-epilog", cl::Hidden, cl::init(true),
  cl::desc("Allow stripmined Tapir loops to execute their epilogs in parallel."));

static cl::opt<bool> IncludeNestedSync(
  "include-nested-sync", cl::Hidden, cl::init(true),
  cl::desc("If the epilog is allowed to execute in parallel, include a sync "
           "instruction in the nested task."));

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


/// Approximate the work of the body of the loop L.  Returns several relevant
/// properties of loop L via by-reference arguments.
static int64_t ApproximateLoopCost(
    const Loop *L, unsigned &NumCalls, bool &NotDuplicatable,
    bool &Convergent, bool &IsRecursive, bool &UnknownSize,
    const TargetTransformInfo &TTI, LoopInfo *LI, ScalarEvolution &SE,
    const SmallPtrSetImpl<const Value *> &EphValues,
    TargetLibraryInfo *TLI) {

  WSCost LoopCost;
  estimateLoopCost(LoopCost, L, LI, &SE, TTI, TLI, EphValues);

  // Exclude calls to builtins when counting the calls.  This assumes that all
  // builtin functions are cheap.
  NumCalls = LoopCost.Metrics.NumCalls - LoopCost.Metrics.NumBuiltinCalls;
  NotDuplicatable = LoopCost.Metrics.notDuplicatable;
  Convergent = LoopCost.Metrics.convergent;
  IsRecursive = LoopCost.Metrics.isRecursive;
  UnknownSize = LoopCost.UnknownCost;

  return LoopCost.Work;
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

static bool tryToStripMineLoop(
    Loop *L, DominatorTree &DT, LoopInfo *LI, ScalarEvolution &SE,
    const TargetTransformInfo &TTI, AssumptionCache &AC, TaskInfo *TI,
    OptimizationRemarkEmitter &ORE, TargetLibraryInfo *TLI, bool PreserveLCSSA,
    Optional<unsigned> ProvidedCount) {
  Task *T = getTaskIfTapirLoop(L, TI);
  if (!T)
    return false;
  TapirLoopHints Hints(L);

  if (HasStripMineDisablePragma(L))
    return false;

  LLVM_DEBUG(dbgs() << "Loop Strip Mine: F["
                    << L->getHeader()->getParent()->getName() << "] Loop %"
                    << L->getHeader()->getName() << "\n");

  if (!L->isLoopSimplifyForm()) {
    LLVM_DEBUG(
        dbgs() << "  Not stripmining loop which is not in loop-simplify form.\n");
    return false;
  }
  bool StripMiningRequested = HasStripMineEnablePragma(L);
  TargetTransformInfo::StripMiningPreferences SMP =
    gatherStripMiningPreferences(L, SE, TTI, ProvidedCount);

  unsigned NumCalls = 0;
  bool NotDuplicatable = false;
  bool Convergent = false;
  bool IsRecursive = false;
  bool UnknownSize = false;

  SmallPtrSet<const Value *, 32> EphValues;
  CodeMetrics::collectEphemeralValues(L, &AC, EphValues);

  int64_t LoopCost =
      ApproximateLoopCost(L, NumCalls, NotDuplicatable, Convergent, IsRecursive,
                          UnknownSize, TTI, LI, SE, EphValues, TLI);
  // Determine the iteration count of the eventual stripmined the loop.
  bool explicitCount = computeStripMineCount(L, TTI, LoopCost, SMP);

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
  LLVM_DEBUG(dbgs() << "  Loop Cost = " << LoopCost << "\n");
  if (!explicitCount && std::numeric_limits<int64_t>::max() == LoopCost) {
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
  // these G/2 iterations offset the cost of an additional detach.
  // Mathematically, this means
  //
  // (G/2) * S + d <= (1 + \eps) * G/2 * S ,
  //
  // where S is the work of one loop iteration, d is the cost of a detach, and
  // \eps is a sufficiently small constant, e.g., 1/C for a coarsening factor C.
  // We assume that the choice of G is chosen such that G * \eps <= 1, which is
  // true for the automatic computation of G aimed at ensuring the stripmined
  // loop performs at most a (1 + \eps) factor more work than its serial
  // projection.  Solving the above equation thus shows that the epilog should
  // be allowed to run in parallel when S >= 2 * d.  We check for this case and
  // encode the result in ParallelEpilog.
  Instruction *DetachI = L->getHeader()->getTerminator();
  bool ParallelEpilog = AllowParallelEpilog &&
    ((SMP.Count < SMP.DefaultCoarseningFactor) ||
     (LoopCost >= static_cast<unsigned>(2 * TTI.getUserCost(DetachI))));

  // Some parallel runtimes, such as Cilk, require nested parallel tasks to be
  // synchronized.
  bool NeedNestedSync = IncludeNestedSync;
  if (!NeedNestedSync && TLI)
    NeedNestedSync = (TLI->getTapirTarget() == TapirTargetID::Cilk);

  Loop *NewLoop = StripMineLoop(L, SMP.Count, SMP.AllowExpensiveTripCount,
                                SMP.UnrollRemainder, LI, &SE, &DT, &AC, TI,
                                &ORE, PreserveLCSSA, ParallelEpilog,
                                NeedNestedSync);
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

    auto &TLI = getAnalysis<TargetLibraryInfoWrapperPass>().getTLI();
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

    return tryToStripMineLoop(L, DT, LI, SE, TTI, AC, TI, ORE, &TLI,
                              PreserveLCSSA, ProvidedCount);
  }

  /// This transformation requires natural loop information & requires that
  /// loop preheaders be inserted into the CFG...
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<AssumptionCacheTracker>();
    AU.addRequired<TargetTransformInfoWrapperPass>();
    AU.addRequired<TargetLibraryInfoWrapperPass>();
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
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
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
  auto &TLI = AM.getResult<TargetLibraryAnalysis>(F);
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
      tryToStripMineLoop(&L, DT, &LI, SE, TTI, AC, &TI, ORE, &TLI,
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
