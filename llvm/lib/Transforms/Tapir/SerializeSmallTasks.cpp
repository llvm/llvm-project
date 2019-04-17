//===- SerializeSmallTasks.cpp - Serialize small Tapir tasks --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass serializes Tapir tasks with too little work to justify spawning.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Tapir/SerializeSmallTasks.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/CodeMetrics.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TapirTaskInfo.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/WorkSpanAnalysis.h"
#include "llvm/Transforms/Tapir/LoopStripMine.h"
#include "llvm/Transforms/Utils/TapirUtils.h"

using namespace llvm;

#define DEBUG_TYPE "serialize-small-tasks"

static cl::opt<bool> SerializeUnprofitableLoops(
  "serialize-unprofitable-loops", cl::Hidden, cl::init(true),
  cl::desc("Serialize any Tapir tasks found to be unprofitable."));

// TODO: Remove this duplicated code from LoopStripMinePass.

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

static bool trySerializeSmallLoop(
    Loop *L, DominatorTree &DT, LoopInfo *LI, ScalarEvolution &SE,
    const TargetTransformInfo &TTI, AssumptionCache &AC, TaskInfo *TI,
    OptimizationRemarkEmitter &ORE, TargetLibraryInfo *TLI) {
  bool Changed = false;
  for (Loop *SubL : *L)
    Changed |= trySerializeSmallLoop(SubL, DT, LI, SE, TTI, AC, TI, ORE, TLI);

  Task *T = getTaskIfTapirLoop(L, TI);
  if (!T)
    return Changed;

  // Skip any loop for which stripmining is explicitly disabled.
  if (HasStripMineDisablePragma(L))
    return Changed;

  TapirLoopHints Hints(L);

  TargetTransformInfo::StripMiningPreferences SMP =
    gatherStripMiningPreferences(L, SE, TTI, None);

  SmallPtrSet<const Value *, 32> EphValues;
  CodeMetrics::collectEphemeralValues(L, &AC, EphValues);

  WSCost LoopCost;
  estimateLoopCost(LoopCost, L, LI, &SE, TTI, TLI, EphValues);

  // If the work in the loop is larger than the maximum value we can deal with,
  // then it's not small.
  if (LoopCost.UnknownCost)
    return Changed;

  computeStripMineCount(L, TTI, LoopCost.Work, SMP);
  // Make sure the count is a power of 2.
  if (!isPowerOf2_32(SMP.Count))
    SMP.Count = NextPowerOf2(SMP.Count);

  // Find a constant trip count if available
  unsigned ConstTripCount = getConstTripCount(L, SE);

  if (!ConstTripCount || SMP.Count < ConstTripCount)
    return Changed;

  ORE.emit([&]() {
             return OptimizationRemark("serialize-small-tasks",
                                       "SerializingSmallLoop",
                                       L->getStartLoc(), L->getHeader())
               << "Serializing parallel loop that appears to be unprofitable "
               << "to parallelize.";
           });
  SerializeDetach(cast<DetachInst>(L->getHeader()->getTerminator()), T, &DT);
  Hints.clearHintsMetadata();
  L->setDerivedFromTapirLoop();
  return true;
}

namespace {
struct SerializeSmallTasks : public FunctionPass {
  static char ID; // Pass identification, replacement for typeid
  SerializeSmallTasks() : FunctionPass(ID) {
    initializeSerializeSmallTasksPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<AssumptionCacheTracker>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addRequired<ScalarEvolutionWrapperPass>();
    AU.addRequired<TargetLibraryInfoWrapperPass>();
    AU.addRequired<TargetTransformInfoWrapperPass>();
    AU.addRequired<TaskInfoWrapperPass>();
    AU.addRequired<OptimizationRemarkEmitterWrapperPass>();
    AU.addPreserved<GlobalsAAWrapperPass>();
  }
};
}

char SerializeSmallTasks::ID = 0;
INITIALIZE_PASS_BEGIN(SerializeSmallTasks, "serialize-small-tasks",
                "Serialize small Tapir tasks", false, false)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolutionWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TaskInfoWrapperPass)
INITIALIZE_PASS_END(SerializeSmallTasks, "serialize-small-tasks",
                "Serialize small Tapir tasks", false, false)

namespace llvm {
FunctionPass *createSerializeSmallTasksPass() {
  return new SerializeSmallTasks();
}
} // end namespace llvm

/// runOnFunction - Run through all tasks in the function and simplify them in
/// post order.
///
bool SerializeSmallTasks::runOnFunction(Function &F) {
  if (skipFunction(F))
    return false;

  TaskInfo &TI = getAnalysis<TaskInfoWrapperPass>().getTaskInfo();
  if (TI.isSerial())
    return false;

  auto &TLI = getAnalysis<TargetLibraryInfoWrapperPass>().getTLI();
  auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  LoopInfo *LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  ScalarEvolution &SE = getAnalysis<ScalarEvolutionWrapperPass>().getSE();
  const TargetTransformInfo &TTI =
    getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);
  auto &AC = getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F);
  // For the old PM, we can't use OptimizationRemarkEmitter as an analysis
  // pass.  Function analyses need to be preserved across loop transformations
  // but ORE cannot be preserved (see comment before the pass definition).
  OptimizationRemarkEmitter ORE(&F);

  LLVM_DEBUG(dbgs() << "SerializeSmallTasks running on function " << F.getName()
             << "\n");

  bool Changed = false;
  if (SerializeUnprofitableLoops)
    for (Loop *L : *LI)
      Changed |= trySerializeSmallLoop(L, DT, LI, SE, TTI, AC, &TI, ORE, &TLI);

  if (Changed)
    // Recalculate TaskInfo
    TI.recalculate(*DT.getRoot()->getParent(), DT);

  return Changed;
}

PreservedAnalyses SerializeSmallTasksPass::run(Function &F,
                                               FunctionAnalysisManager &AM) {
  if (F.empty())
    return PreservedAnalyses::all();

  TaskInfo &TI = AM.getResult<TaskAnalysis>(F);
  if (TI.isSerial())
    return PreservedAnalyses::all();

  auto &TLI = AM.getResult<TargetLibraryAnalysis>(F);
  auto &SE = AM.getResult<ScalarEvolutionAnalysis>(F);
  auto &LI = AM.getResult<LoopAnalysis>(F);
  auto &TTI = AM.getResult<TargetIRAnalysis>(F);
  auto &DT = AM.getResult<DominatorTreeAnalysis>(F);
  auto &AC = AM.getResult<AssumptionAnalysis>(F);
  auto &ORE = AM.getResult<OptimizationRemarkEmitterAnalysis>(F);


  LLVM_DEBUG(dbgs() << "SerializeSmallTasks running on function " << F.getName()
             << "\n");

  bool Changed = false;
  if (SerializeUnprofitableLoops)
    for (Loop *L : LI)
      Changed |= trySerializeSmallLoop(L, DT, &LI, SE, TTI, AC, &TI, ORE, &TLI);

  if (!Changed)
    return PreservedAnalyses::all();

  // Recalculate TaskInfo
  TI.recalculate(*DT.getRoot()->getParent(), DT);

  PreservedAnalyses PA;
  PA.preserve<DominatorTreeAnalysis>();
  PA.preserve<LoopAnalysis>();
  PA.preserve<ScalarEvolutionAnalysis>();
  PA.preserve<TaskAnalysis>();
  PA.preserve<GlobalsAA>();
  // TODO: Add more preserved analyses here.
  return PA;
}
