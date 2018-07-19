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
    DEBUG(dbgs() << "markDefiningSpindle @ " << *S << "\n");
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
    DEBUG(dbgs() << "evaluate @ " << *S << "\n");

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
    DEBUG(dbgs() << "markDefiningSpindle @ " << *S << "\n");
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
          DEBUG(dbgs() << "TaskCont spindle " << *S << "\n");
          bool Complete = true;
          for (const Spindle *Pred : predecessors(S)) {
            DEBUG(dbgs() << "pred spindle " << *Pred << "\n");
            if (S->predInDifferentTask(Pred))
              TaskList[S].insert(Pred->getParentTask());
            // If we have a Phi predecessor of this spindle, we'll want to
            // re-evaluate it.
            if (Pred->isPhi())
              Complete = false;
          }
          DEBUG({
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
    DEBUG(dbgs() << "evaluate @ " << *S << "\n");
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
    DEBUG({
        dbgs() << "New MPT list for " << *S << "(Complete? " << Complete << ")\n";
        for (const Task *MP : TaskList[S])
          dbgs() << "\t" << MP->getEntry()->getName() << "\n";
      });
    return Complete;
  }
};

bool AnalyzeTapir::runOnFunction(Function &F) {
  if (skipFunction(F))
    return false;

  TaskInfo &TI = getAnalysis<TaskInfoWrapperPass>().getTaskInfo();
  auto &ORE =
    getAnalysis<OptimizationRemarkEmitterWrapperPass>().getORE();

  if (TI.getRootTask()->isSerial())
    return false;

  // dbgs() << "Analyzing function " << F.getName() << "\n";

  // DEBUG({
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
  DEBUG({
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
