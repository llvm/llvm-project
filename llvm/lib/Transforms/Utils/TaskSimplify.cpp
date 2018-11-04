//===- TaskSimplify.cpp - Tapir task simplification pass ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass performs several transformations to simplify Tapir tasks.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/TaskSimplify.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/TapirTaskInfo.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/TapirUtils.h"

using namespace llvm;

#define DEBUG_TYPE "task-simplify"

// Structure to record the synced state of each spindle.
struct IsSyncedState {
  enum class SyncInfo {
    Unsynced = 0,
    Synced = 1,
    TaskEntry = 2,
    NoUnsync = Synced | TaskEntry,
    Incomplete = 4,
  };

  static inline bool isUnsynced(const SyncInfo SyncI) {
    return (static_cast<int>(SyncI) & static_cast<int>(SyncInfo::NoUnsync)) ==
      static_cast<int>(SyncInfo::Unsynced);
  }
  static inline bool isSynced(const SyncInfo SyncI) {
    return !isUnsynced(SyncI);
  }
  static inline bool isIncomplete(const SyncInfo SyncI) {
    return (static_cast<int>(SyncI) & static_cast<int>(SyncInfo::Incomplete)) ==
      static_cast<int>(SyncInfo::Incomplete);
  }
  static inline SyncInfo setUnsynced(const SyncInfo SyncI) {
    // Once a sync state is set to unsynced, it's complete.
    return SyncInfo(static_cast<int>(SyncI) &
                    static_cast<int>(SyncInfo::Unsynced));
  }
  static inline SyncInfo setIncomplete(const SyncInfo SyncI) {
    return SyncInfo(static_cast<int>(SyncI) |
                    static_cast<int>(SyncInfo::Incomplete));
  }
  static inline SyncInfo setComplete(const SyncInfo SyncI) {
    return SyncInfo(static_cast<int>(SyncI) &
                    ~static_cast<int>(SyncInfo::Incomplete));
  }

  DenseMap<const Spindle *, SyncInfo> SyncedState;

  // This method is called once per spindle during an initial DFS traversal of
  // the spindle graph.
  bool markDefiningSpindle(const Spindle *S) {
    LLVM_DEBUG(dbgs() << "markDefiningSpindle @ " << *S << "\n");
    // Entry spindles, detach spindles, sync spindles, and continuation-Phi
    // spindles all define their sync state directly.  Other Phi spindles
    // determine their sync state based on their predecessors.
    switch (S->getType()) {
    case Spindle::SPType::Entry:
    case Spindle::SPType::Detach:
      SyncedState[S] = SyncInfo::TaskEntry;
      return true;
    case Spindle::SPType::Sync:
      SyncedState[S] = SyncInfo::Synced;
      return true;
    case Spindle::SPType::Phi:
      if (S->isTaskContinuation()) {
        SyncedState[S] = SyncInfo::Unsynced;
        return true;
      }
    }
    return false;
  }

  // This method is called once per unevaluated spindle in an inverse-post-order
  // walk of the spindle graph.
  bool evaluate(const Spindle *S, unsigned EvalNum) {
    LLVM_DEBUG(dbgs() << "evaluate @ " << *S << "\n");

    // if (!EvalNum && SyncedState.count(S)) return;

    // For the first evaluation, optimistically assume that we are synced.  Any
    // unsynced predecessor will clear this bit.
    if (!EvalNum) {
      assert(!SyncedState.count(S) &&
             "Evaluating a spindle whose sync state is already determined.");
      SyncedState[S] = SyncInfo::Synced;
    }

    for (const Spindle::SpindleEdge &PredEdge : S->in_edges()) {
      const Spindle *Pred = PredEdge.first;
      const BasicBlock *Inc = PredEdge.second;

      // During the first evaluation, if we have a loop amongst Phi spindles,
      // then the predecessor might not be defined.  Skip predecessors that
      // aren't defined.
      if (!EvalNum && !SyncedState.count(Pred)) {
        SyncedState[S] = setIncomplete(SyncedState[S]);
        continue;
      } else
        assert(SyncedState.count(Pred) &&
               "All predecessors should have synced states after first eval.");

      // If we find an unsynced predecessor that is not terminated by a sync
      // instruction, then we must be unsynced.
      if (isUnsynced(SyncedState[Pred]) &&
          !isa<SyncInst>(Inc->getTerminator())) {
        SyncedState[S] = setUnsynced(SyncedState[S]);
        break;
      }
    }
    // Because spindles are evaluated in each round in an inverse post-order
    // traversal, two evaluations should suffice.  If we have an incomplete
    // synced state at the end of the first evaluation, then we conclude that
    // it's synced at set it complete.
    if (EvalNum && isIncomplete(SyncedState[S])) {
      SyncedState[S] = setComplete(SyncedState[S]);
      return true;
    }
    return !isIncomplete(SyncedState[S]);
  }
};

static bool hasPredecessorSync(const Spindle *S) {
  for (const BasicBlock *Pred : predecessors(S->getEntry()))
    if (isa<SyncInst>(Pred->getTerminator()))
      return true;
  return false;
}

// Structure to record the set of child tasks that might be in parallel with
// this spindle.
struct MaybeParallelTasks {
  DenseMap<const Spindle *, SmallPtrSet<const Task *, 2>> TaskList;

  // This method is called once per spindle during an initial DFS traversal of
  // the spindle graph.
  bool markDefiningSpindle(const Spindle *S) {
    LLVM_DEBUG(dbgs() << "markDefiningSpindle @ " << *S << "\n");
    switch (S->getType()) {
      // Emplace empty task lists for Entry, Detach, and Sync spindles.
    case Spindle::SPType::Entry:
    case Spindle::SPType::Detach:
      TaskList.try_emplace(S);
      return true;
    case Spindle::SPType::Sync:
      // TaskList.try_emplace(S);
      // return true;
      return false;
    case Spindle::SPType::Phi:
      {
        // At task-continuation Phi's, initialize the task list with the
        // detached task that reattaches to this continuation.
        if (S->isTaskContinuation()) {
          LLVM_DEBUG(dbgs() << "TaskCont spindle " << *S << "\n");
          bool Complete = true;
          for (const Spindle *Pred : predecessors(S)) {
            LLVM_DEBUG(dbgs() << "pred spindle " << *Pred << "\n");
            if (S->predInDifferentTask(Pred))
              TaskList[S].insert(Pred->getParentTask());
            // If we have a Phi or Sync predecessor of this spindle, we'll want
            // to re-evaluate it.
            if (Pred->isPhi() || Pred->isSync())
              Complete = false;
          }
          LLVM_DEBUG({
              for (const Task *MPT : TaskList[S])
                dbgs() << "Added MPT " << MPT->getEntry()->getName() << "\n";
            });
          return Complete;
        }
        return false;
      }
    }
    return false;
  }

  // This method is called once per unevaluated spindle in an inverse-post-order
  // walk of the spindle graph.
  bool evaluate(const Spindle *S, unsigned EvalNum) {
    LLVM_DEBUG(dbgs() << "evaluate @ " << *S << "\n");
    if (!TaskList.count(S))
      TaskList.try_emplace(S);

    bool Complete = true;
    for (const Spindle::SpindleEdge &PredEdge : S->in_edges()) {
      const Spindle *Pred = PredEdge.first;
      const BasicBlock *Inc = PredEdge.second;

      // If the incoming edge is a sync edge, get the associated sync region.
      const Value *SyncRegSynced = nullptr;
      if (const SyncInst *SI = dyn_cast<SyncInst>(Inc->getTerminator()))
        SyncRegSynced = SI->getSyncRegion();

      // Iterate through the tasks in the task list for Pred.
      for (const Task *MP : TaskList[Pred]) {
        // Filter out any tasks that are synced by the sync region.
        if (const DetachInst *DI = MP->getDetach())
          if (SyncRegSynced == DI->getSyncRegion())
            continue;
        // Insert the task into this spindle's task list.  If this task is a new
        // addition, then we haven't yet reached the fixed point of this
        // analysis.
        if (TaskList[S].insert(MP).second)
          Complete = false;
      }
    }
    LLVM_DEBUG({
        dbgs() << "New MPT list for " << *S << "(Complete? " << Complete << ")\n";
        for (const Task *MP : TaskList[S])
          dbgs() << "\t" << MP->getEntry()->getName() << "\n";
      });
    return Complete;
  }
};

static bool syncMatchesReachingTask(const Value *SyncSR,
                                    SmallPtrSetImpl<const Task *> &MPTasks) {
  if (MPTasks.empty())
    return false;
  for (const Task *MPTask : MPTasks)
    if (SyncSR == MPTask->getDetach()->getSyncRegion())
      return true;
  return false;
}

static bool removeRedundantSyncs(MaybeParallelTasks &MPTasks, Task *T) {
  // Skip tasks with no subtasks.
  if (T->isSerial())
    return false;

  bool Changed = false;
  SmallPtrSet<SyncInst *, 1> RedundantSyncs;
  for (Spindle *S : T->spindles())
    // Iterate over outgoing edges of S to find redundant syncs.
    for (Spindle::SpindleEdge &Edge : S->out_edges())
      if (SyncInst *Y = dyn_cast<SyncInst>(Edge.second->getTerminator()))
        if (!syncMatchesReachingTask(Y->getSyncRegion(), MPTasks.TaskList[S]))
          RedundantSyncs.insert(Y);

  // Replace all unnecesary syncs with unconditional branches.
  for (SyncInst *Y : RedundantSyncs)
    ReplaceInstWithInst(Y, BranchInst::Create(Y->getSuccessor(0)));

  Changed |= !RedundantSyncs.empty();

  return Changed;
}

static bool syncIsDiscriminating(const Value *SyncSR,
                                 SmallPtrSetImpl<const Task *> &MPTasks) {
  for (const Task *MPTask : MPTasks)
    if (SyncSR != MPTask->getDetach()->getSyncRegion())
      return true;
  return false;
}

static bool removeRedundantSyncRegions(MaybeParallelTasks &MPTasks, Task *T) {
  if (T->isSerial())
    return false;

  // Find the unique sync regions in this task.
  SmallPtrSet<Value *, 1> UniqueSyncRegs;
  Value *FirstSyncRegion = nullptr;
  for (Task *SubT : T->subtasks()) {
    UniqueSyncRegs.insert(SubT->getDetach()->getSyncRegion());
    if (!FirstSyncRegion)
      FirstSyncRegion = SubT->getDetach()->getSyncRegion();
  }
  // Skip this task if there's only one unique sync region.
  if (UniqueSyncRegs.size() < 2)
    return false;

  bool Changed = false;
  SmallPtrSet<Value *, 1> NonRedundantSyncRegs;
  for (Spindle *S : T->spindles()) {
    // Only consider spindles that might have tasks in parallel.
    if (MPTasks.TaskList[S].empty()) continue;

    // Iterate over outgoing edges of S to find discriminating syncs.
    for (Spindle::SpindleEdge &Edge : S->out_edges())
      if (const SyncInst *Y = dyn_cast<SyncInst>(Edge.second->getTerminator()))
        if (syncIsDiscriminating(Y->getSyncRegion(), MPTasks.TaskList[S]))
          NonRedundantSyncRegs.insert(Y->getSyncRegion());

    // Replace all redundant sync regions with the first sync region.
    for (Value *SR : UniqueSyncRegs) {
      if (!NonRedundantSyncRegs.count(SR) && SR != FirstSyncRegion) {
        Changed = true;
        SR->replaceAllUsesWith(FirstSyncRegion);
      }
    }
  }
  return Changed;
}

bool llvm::simplifySyncs(Task *T, TaskInfo &TI) {
  bool Changed = false;

  LLVM_DEBUG(dbgs() <<
             "Simplifying syncs in task @ " << T->getEntry()->getName() <<
             "\n");

  // Evaluate the tasks that might be in parallel with each spindle, and
  // determine number of discriminating syncs: syncs that sync a subset of the
  // detached tasks, based on sync regions.
  MaybeParallelTasks MPTasks;
  TI.evaluateParallelState<MaybeParallelTasks>(MPTasks);

  // Remove redundant syncs.  This optimization might not be necessary here,
  // because SimplifyCFG seems to do a good job removing syncs that cannot sync
  // anything.
  Changed |= removeRedundantSyncs(MPTasks, T);

  // Remove redundant sync regions.
  Changed |= removeRedundantSyncRegions(MPTasks, T);

  return Changed;
}  

static bool taskCanThrow(const Task *T) {
  for (const Spindle *S : T->spindles())
    for (const BasicBlock *BB : S->blocks())
      if (isa<InvokeInst>(BB->getTerminator()))
        return true;
  return false;
}

static bool taskCanReachContinuation(Task *T) {
  if (T->isRootTask())
    return true;

  DetachInst *DI = T->getDetach();
  BasicBlock *Continue = DI->getContinue();
  for (BasicBlock *Pred : predecessors(Continue)) {
    if (ReattachInst *RI = dyn_cast<ReattachInst>(Pred->getTerminator()))
      if (T->encloses(RI->getParent()))
        return true;
  }

  return false;
}

static bool detachImmediatelySyncs(DetachInst *DI) {
  Instruction *I = DI->getParent()->getFirstNonPHIOrDbgOrLifetime();
  return isa<SyncInst>(I);
}

bool llvm::simplifyTask(Task *T, TaskInfo &TI) {
  if (T->isRootTask())
    return false;

  LLVM_DEBUG(dbgs() <<
             "Simplifying task @ " << T->getEntry()->getName() << "\n");

  bool Changed = false;
  DetachInst *DI = T->getDetach();

  // If T's detach has an unwind dest and T cannot throw, remove the unwind
  // destination from T's detach.
  if (DI->hasUnwindDest()) {
    if (!taskCanThrow(T)) {
      removeUnwindEdge(DI->getParent());
      // removeUnwindEdge will invalidate the DI pointer.  Get the new DI
      // pointer.
      DI = T->getDetach();
      Changed = true;
    }
  }

  if (!taskCanReachContinuation(T)) {
    // This optimization assumes that if a task cannot reach its continuation
    // then we shouldn't bother spawning it.  The task might perform code that
    // can reach the unwind destination, however.
    SerializeDetach(DI, T);
    Changed = true;
  } else if (detachImmediatelySyncs(DI)) {
    SerializeDetach(DI, T);
    Changed = true;
  }

  return Changed;
}

static void simplifyCFG(Function &F) {
  llvm::legacy::FunctionPassManager FPM(F.getParent());
  FPM.add(createCFGSimplificationPass());

  FPM.doInitialization();
  FPM.run(F);
  FPM.doFinalization();
}

namespace {
struct TaskSimplify : public FunctionPass {
  static char ID; // Pass identification, replacement for typeid
  TaskSimplify() : FunctionPass(ID) {
    initializeTaskSimplifyPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TaskInfoWrapperPass>();
    AU.addPreserved<GlobalsAAWrapperPass>();
  }
};
}

char TaskSimplify::ID = 0;
INITIALIZE_PASS_BEGIN(TaskSimplify, "task-simplify",
                "Simplify Tapir tasks", false, false)
INITIALIZE_PASS_DEPENDENCY(TaskInfoWrapperPass)
INITIALIZE_PASS_END(TaskSimplify, "task-simplify",
                "Simplify Tapir tasks", false, false)

namespace llvm {
Pass *createTaskSimplifyPass() { return new TaskSimplify(); }
} // end namespace llvm

/// runOnFunction - Run through all tasks in the function and simplify them in
/// post order.
///
bool TaskSimplify::runOnFunction(Function &F) {
  if (skipFunction(F))
    return false;

  TaskInfo &TI = getAnalysis<TaskInfoWrapperPass>().getTaskInfo();
  if (TI.isSerial())
    return false;

  bool Changed = false;
  // Simplify syncs in each task in the function.
  for (Task *T : post_order(TI.getRootTask()))
    Changed |= simplifySyncs(T, TI);

  // Simplify each task in the function.
  for (Task *T : post_order(TI.getRootTask()))
    Changed |= simplifyTask(T, TI);

  if (Changed)
    simplifyCFG(F);

  return Changed;
}

PreservedAnalyses TaskSimplifyPass::run(Function &F,
                                        FunctionAnalysisManager &AM) {
  TaskInfo &TI = AM.getResult<TaskAnalysis>(F);
  if (F.empty() || TI.isSerial())
    return PreservedAnalyses::all();

  bool Changed = false;
  // Simplify syncs in each task in the function.
  for (Task *T : post_order(TI.getRootTask()))
    Changed |= simplifySyncs(T, TI);

  // Simplify each task in the function.
  for (Task *T : post_order(TI.getRootTask()))
    Changed |= simplifyTask(T, TI);

  if (!Changed)
    return PreservedAnalyses::all();
  PreservedAnalyses PA;
  PA.preserve<GlobalsAA>();
  return PA;
}
