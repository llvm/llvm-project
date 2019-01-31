//===- AnalyzeTapir.cpp - Analyze Tapir tasks -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/TapirTaskInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Tapir.h"
#include "llvm/Transforms/Tapir/LoweringUtils.h"

#define DEBUG_TYPE "analyzetapir"

using namespace llvm;

namespace {

struct AnalyzeTapir : public FunctionPass {
  static char ID; // Pass identification, replacement for typeid
  explicit AnalyzeTapir() : FunctionPass(ID) {
    initializeAnalyzeTapirPass(*PassRegistry::getPassRegistry());
  }

  StringRef getPassName() const override {
    return "Analysis of Tapir tasks";
  }

  bool runOnFunction(Function &F) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<AssumptionCacheTracker>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<MemorySSAWrapperPass>();
    AU.addRequired<TaskInfoWrapperPass>();
    AU.addRequired<OptimizationRemarkEmitterWrapperPass>();
  }
};
}  // End of anonymous namespace

char AnalyzeTapir::ID = 0;
INITIALIZE_PASS_BEGIN(AnalyzeTapir, "analyzetapir",
                      "Analyze Tapir tasks", false, false)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MemorySSAWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TaskInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(OptimizationRemarkEmitterWrapperPass)
INITIALIZE_PASS_END(AnalyzeTapir, "analyzetapir",
                    "Analyze Tapir tasks", false, false)

static bool hasPredecessorSync(const Spindle *S) {
  for (const BasicBlock *Pred : predecessors(S->getEntry()))
    if (isa<SyncInst>(Pred->getTerminator()))
      return true;
  return false;
}

bool AnalyzeTapir::runOnFunction(Function &F) {
  if (skipFunction(F))
    return false;

  TaskInfo &TI = getAnalysis<TaskInfoWrapperPass>().getTaskInfo();
  auto &ORE =
    getAnalysis<OptimizationRemarkEmitterWrapperPass>().getORE();

  if (TI.getRootTask()->isSerial())
    return false;

  // dbgs() << "Analyzing function " << F.getName() << "\n";

  // LLVM_DEBUG({
  //     ReversePostOrderTraversal<Spindle *>
  //       RPOT(TI.getRootTask()->getEntrySpindle());
  //     // SmallPtrSet<Spindle *, 8> Visited;

  //     dbgs() << "RPOT of task:\n";
  //     for (Spindle *S : RPOT) {
  //       // if (!Visited.insert(S).second)
  //       //   dbgs() << "<RETURN VISIT>";
  //       S->print(dbgs());
  //     }
  //     dbgs() << "\n";
  //   });

  // Evaluate whether each spindle is synced, and count number of partially
  // redundant syncs.
  IsSyncedState State;
  TI.evaluateParallelState<IsSyncedState>(State);
  unsigned NumPRSyncs = 0;
  for (const Task *T : depth_first(TI.getRootTask()))
    for (const Spindle *S : T->spindles())
      if ((S->isPhi() || S->isSync()) &&
          IsSyncedState::isSynced(State.SyncedState[S]))
        if (hasPredecessorSync(S) && !S->getEntry()->getUniquePredecessor()) {
          dbgs() << "PR Sync: " << S->getEntry()->getName() << "\n";
          NumPRSyncs++;
        }
  if (NumPRSyncs > 0)
    dbgs() << "In " << F.getName() << " found " << NumPRSyncs
           << " partially redundant syncs.\n";

  // Evaluate the tasks that might be in parallel with each spindle, and
  // determine number of discriminating syncs: syncs that sync a subset of the
  // detached tasks, based on sync regions.
  MaybeParallelTasks MPTasks;
  TI.evaluateParallelState<MaybeParallelTasks>(MPTasks);
  // DEBUGGING: Print out the non-empty may-parallel task lists.
  LLVM_DEBUG({
      for (const Task *T : depth_first(TI.getRootTask())) {
        // Skip tasks with no subtasks.
        if (T->isSerial()) continue;

        for (const Spindle *S : T->spindles()) {
          // Only conider spindles that might have tasks in parallel.
          if (MPTasks.TaskList[S].empty()) continue;

          dbgs() << "MPT's of spindle with entry " << S->getEntry()->getName() << ":\n";
          for (const Task *MPT : MPTasks.TaskList[S])
            dbgs() << "\t" << MPT->getEntry()->getName() << "\n";
        }
      }
    });

  for (const Task *T : depth_first(TI.getRootTask())) {
    // Skip tasks with no subtasks.
    if (T->isSerial()) continue;

    // Find unique sync regions in this task.
    SmallPtrSet<const Value *, 1> UniqueSyncRegs;
    for (const Task *SubT : T->subtasks())
      UniqueSyncRegs.insert(SubT->getDetach()->getSyncRegion());
    // Skip tasks with just one sync region.
    if (UniqueSyncRegs.size() < 2) continue;

    // Count number of discriminating syncs in this task.
    unsigned NumDiscrimSyncs = 0;

    for (const Spindle *S : T->spindles()) {
      // Only conider spindles that might have tasks in parallel.
      if (MPTasks.TaskList[S].empty()) continue;

      // Iterate over the outgoing edges of this spindle.
      for (const Spindle::SpindleEdge &Edge : S->out_edges()) {
        // Examine all sync instructions leaving this spindle.
        if (const SyncInst *Y =
            dyn_cast<SyncInst>(Edge.second->getTerminator())) {
          // Get the sync region for this sync.
          const Value *SyncReg = Y->getSyncRegion();

          // If that sync region does not match the sync region for the detach
          // that creates some maybe-parallel task, we have a discriminating
          // sync region.
          for (const Task *MPT : MPTasks.TaskList[S])
            if (SyncReg != MPT->getDetach()->getSyncRegion()) {
              NumDiscrimSyncs++;
              break;
            }
        }
      }
    }
    dbgs() << "In " << F.getName() << " in task @ " << T->getEntry()->getName()
           << " found " << UniqueSyncRegs.size() << " sync regions and "
           << NumDiscrimSyncs << " discriminating syncs.\n";
  }

  // // Collect the loads and stores in this function
  // SmallVector<Instruction *, 4> Accesses;
  // for (BasicBlock &BB : F)
  //   for (Instruction &I : BB)
  //     if (isa<LoadInst>(I) || isa<StoreInst>(I))
  //       Accesses.push_back(&I);

  // MemorySSA &MSSA = getAnalysis<MemorySSAWrapperPass>().getMSSA();
  // MemorySSAWalker *Walker = MSSA.getWalker();
  // for (Instruction *Acc : Accesses) {
  //   dbgs() << "Access " << *Acc << " (block " << Acc->getParent()->getName() << ")\n";
  //   MemoryUseOrDef *MemAcc = MSSA.getMemoryAccess(Acc);
  //   for (User *U : MemAcc->users())
  //     if (MemoryAccess *Use = dyn_cast<MemoryAccess>(U))
  //       dbgs() << "\tUser " << *Use << "\n";
  //     else
  //       dbgs() << "\tUser is not a memory access\n";
  //   MemoryAccess *Clobber =
  //     Walker->getClobberingMemoryAccess(MemAcc, MemoryLocation::get(Acc));
  //   dbgs() << "\tClobbering access " << *Clobber << "\n";
  // }

  return false;
}

// createAnalyzeTapirPass - Provide an entry point to create this pass.
//
namespace llvm {
FunctionPass *createAnalyzeTapirPass() {
  return new AnalyzeTapir();
}
}
