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
  Instruction *FirstSyncRegion = nullptr;
  for (Task *SubT : T->subtasks()) {
    UniqueSyncRegs.insert(SubT->getDetach()->getSyncRegion());
    if (!FirstSyncRegion)
      FirstSyncRegion = cast<Instruction>(
          SubT->getDetach()->getSyncRegion());
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
        LLVM_DEBUG(dbgs() << "Replacing " << *SR << " with " << *FirstSyncRegion
                   << "\n");
        Changed = true;
        SR->replaceAllUsesWith(FirstSyncRegion);
        // Ensure that the first sync region is in the entry block of T.
        if (FirstSyncRegion->getParent() != T->getEntry())
          FirstSyncRegion->moveAfter(&*T->getEntry()->getFirstInsertionPt());
      }
    }
  }
  return Changed;
}

bool llvm::simplifySyncs(Task *T, MaybeParallelTasks &MPTasks) {
  bool Changed = false;

  LLVM_DEBUG(dbgs() <<
             "Simplifying syncs in task @ " << T->getEntry()->getName() <<
             "\n");

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

bool llvm::simplifyTask(Task *T) {
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
  LLVM_DEBUG(dbgs() << "TaskSimplify running on function " << F.getName()
             << "\n");

  // Evaluate the tasks that might be in parallel with each spindle, and
  // determine number of discriminating syncs: syncs that sync a subset of the
  // detached tasks, based on sync regions.
  MaybeParallelTasks MPTasks;
  TI.evaluateParallelState<MaybeParallelTasks>(MPTasks);

  // Simplify syncs in each task in the function.
  for (Task *T : post_order(TI.getRootTask()))
    Changed |= simplifySyncs(T, MPTasks);

  // Simplify each task in the function.
  for (Task *T : post_order(TI.getRootTask()))
    Changed |= simplifyTask(T);

  if (Changed)
    simplifyCFG(F);

  return Changed;
}

PreservedAnalyses TaskSimplifyPass::run(Function &F,
                                        FunctionAnalysisManager &AM) {
  if (F.empty())
    return PreservedAnalyses::all();

  TaskInfo &TI = AM.getResult<TaskAnalysis>(F);
  if (TI.isSerial())
    return PreservedAnalyses::all();

  bool Changed = false;
  LLVM_DEBUG(dbgs() << "TaskSimplify running on function " << F.getName()
             << "\n");

  // Evaluate the tasks that might be in parallel with each spindle, and
  // determine number of discriminating syncs: syncs that sync a subset of the
  // detached tasks, based on sync regions.
  MaybeParallelTasks MPTasks;
  TI.evaluateParallelState<MaybeParallelTasks>(MPTasks);

  // Simplify syncs in each task in the function.
  for (Task *T : post_order(TI.getRootTask()))
    Changed |= simplifySyncs(T, MPTasks);

  // Simplify each task in the function.
  for (Task *T : post_order(TI.getRootTask()))
    Changed |= simplifyTask(T);

  if (!Changed)
    return PreservedAnalyses::all();
  PreservedAnalyses PA;
  PA.preserve<GlobalsAA>();
  return PA;
}
