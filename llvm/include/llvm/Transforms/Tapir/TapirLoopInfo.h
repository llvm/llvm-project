//===- TapirLoopInfo.h - Utility functions for Tapir loops -----*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines utility functions for hanling Tapir loops.
//
//===----------------------------------------------------------------------===//

#ifndef TAPIR_UTILS_H_
#define TAPIR_UTILS_H_

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/TapirTaskInfo.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Transforms/Tapir/LoweringUtils.h"
#include "llvm/Transforms/Utils/LoopUtils.h"

namespace llvm {

class AssumptionCache;
class BasicBlock;
class DominatorTree;
class ICmpInst;
class Instruction;
class OptimizationRemarkAnalysis;
class OptimizationRemarkEmitter;
class PHINode;
class PredicatedScalarEvolution;
class ScalarEvolution;
class TargetTransformInfo;

class TapirLoopInfo {
public:
  /// InductionList saves induction variables and maps them to the induction
  /// descriptor.
  using InductionList = MapVector<PHINode *, InductionDescriptor>;

  TapirLoopInfo(Loop *L, Task *T) : TheLoop(L), TheTask(T) {
    // Get the exit block for this loop.
    TerminatorInst *TI = TheLoop->getLoopLatch()->getTerminator();
    ExitBlock = TI->getSuccessor(0);
    if (ExitBlock == TheLoop->getHeader())
      ExitBlock = TI->getSuccessor(1);

    // Get the unwind destination for this loop.
    DetachInst *DI = T->getDetach();
    if (DI->hasUnwindDest())
      UnwindDest = DI->getUnwindDest();
  }

  ~TapirLoopInfo() {
    if (StartIterArg)
      delete StartIterArg;
    if (EndIterArg)
      delete EndIterArg;
    if (GrainsizeArg)
      delete GrainsizeArg;

    DescendantTasks.clear();
    Inductions.clear();
  }
      
  Loop *getLoop() const { return TheLoop; }
  Task *getTask() const { return TheTask; }

  /// Top-level call to prepare a Tapir loop for outlining.
  bool prepareForOutlining(
    DominatorTree &DT, LoopInfo &LI, TaskInfo &TI,
    PredicatedScalarEvolution &PSE, AssumptionCache &AC,
    OptimizationRemarkEmitter &ORE, const TargetTransformInfo &TTI);

  /// Gather all induction variables in this loop that need special handling
  /// during outlining.
  bool collectIVs(PredicatedScalarEvolution &PSE,
                  OptimizationRemarkEmitter &ORE);

  /// Replace all induction variables in this loop that are not primary with
  /// stronger forms.
  void replaceNonPrimaryIVs(PredicatedScalarEvolution &PSE);

  /// Fix up external users of the induction variable.
  void fixupIVUsers(PHINode *OrigPhi, const InductionDescriptor &II,
                    PredicatedScalarEvolution &PSE);

  /// Returns (and creates if needed) the original loop trip count.
  Value *getOrCreateTripCount(PredicatedScalarEvolution &PSE);

  /// Record task T as a descendant task under this loop and not under a
  /// descendant Tapir loop.
  void addDescendantTask(Task *T) { DescendantTasks.push_back(T); }

  /// Adds \p Phi, with induction descriptor ID, to the inductions list.  This
  /// can set \p Phi as the main induction of the loop if \p Phi is a better
  /// choice for the main induction than the existing one.
  void addInductionPhi(PHINode *Phi, const InductionDescriptor &ID);

  /// Returns the original loop trip count, if it has been computed.
  Value *getTripCount() const {
    assert(TripCount.pointsToAliveValue() &&
           "TripCount does not point to alive value.");
    return TripCount;
  }

  /// Returns the original loop condition, if it has been computed.
  ICmpInst *getCondition() const { return Condition; }

  /// Returns true if this loop condition includes the end iteration.
  bool isInclusiveRange() const { return InclusiveRange; }

  /// Returns the widest induction type.
  Type *getWidestInductionType() const { return WidestIndTy; }

  const std::pair<PHINode *, InductionDescriptor> &getPrimaryInduction() const {
    assert(PrimaryInduction && "No primary induction.");
    return *Inductions.find(PrimaryInduction);
  }

  /// Returns the induction variables found in the loop.
  InductionList *getInductionVars() { return &Inductions; }

  /// Get the grainsize associated with this Tapir Loop.  A return value of 0
  /// indicates the absence of a specified grainsize.
  unsigned getGrainsize() const { return Grainsize; }

  /// Get the exit block assoicated with this Tapir loop.
  BasicBlock *getExitBlock() const { return ExitBlock; }

  /// Get the unwind destination for this Tapir loop.
  BasicBlock *getUnwindDest() const { return UnwindDest; }

  /// Get the set of tasks enclosed in this Tapir loop and not a descendant
  /// Tapir loop.
  void getEnclosedTasks(SmallVectorImpl<Task *> &TaskVec) const {
    TaskVec.push_back(TheTask);
    for (Task *T : reverse(DescendantTasks))
      TaskVec.push_back(T);
  }

  /// Update information on this Tapir loop based on its metadata.
  void readTapirLoopMetadata(OptimizationRemarkEmitter &ORE);

  DebugLoc getDebugLoc() const { return TheTask->getDetach()->getDebugLoc(); }

private:
  /// The loop that we evaluate.
  Loop *TheLoop;

  /// The task contained in this loop.
  Task *TheTask;

  /// Descendants of TheTask that are enclosed by this loop and not a descendant
  /// Tapir loop.
  SmallVector<Task *, 4> DescendantTasks;

  /// The single exit block for this Tapir loop.
  BasicBlock *ExitBlock = nullptr;

  /// The unwind destination of this Tapir loop, if it has one.
  BasicBlock *UnwindDest = nullptr;

  /// Holds the primary induction variable. This is the counter of the loop.
  PHINode *PrimaryInduction = nullptr;

  /// Holds all of the induction variables that we found in the loop.  Notice
  /// that inductions don't need to start at zero and that induction variables
  /// can be pointers.
  InductionList Inductions;

  /// Holds the widest induction type encountered.
  Type *WidestIndTy = nullptr;

  /// Trip count of the original loop.
  WeakTrackingVH TripCount;

  /// Latch condition of the original loop.
  ICmpInst *Condition = nullptr;
  bool InclusiveRange = false;

  /// Grainsize value to use for loop.  A value of 0 indicates that a call to
  /// Tapir's grainsize intrinsic should be used.
  unsigned Grainsize = 0;

public:
  /// Placeholder argument values.
  Argument *StartIterArg = nullptr;
  Argument *EndIterArg = nullptr;
  Argument *GrainsizeArg = nullptr;
};

}  // end namepsace llvm

#endif
