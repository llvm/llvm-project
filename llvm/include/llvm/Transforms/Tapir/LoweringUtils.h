//===- LoweringUtils.h - Utility functions for lowering Tapir --*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements several utility functions for lowering Tapir.
//
//===----------------------------------------------------------------------===//

#ifndef LOWERING_UTILS_H_
#define LOWERING_UTILS_H_

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Transforms/Tapir/TapirTargetIDs.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

namespace llvm {

class AssumptionCache;
class BasicBlock;
class DominatorTree;
class Function;
class TapirLoopInfo;
class Task;
class TaskInfo;
class Value;

using ValueSet = SetVector<Value *>;

/// Structure that captures relevant information about an outlined task,
/// including the following:
/// -) A pointer to the outlined function.
/// -) The inputs passed to the call or invoke of that outlined function.
/// -) Pointers to the instructions that replaced the detach in the parent
/// function, ending with the call or invoke instruction to the outlined
/// function.
/// -) The normal and unwind destinations of the call or invoke of the outlined
/// function.
struct TaskOutlineInfo {
  Function *Outline = nullptr;
  SmallVector<Value *, 8> OutlineInputs;
  Instruction *ReplStart = nullptr;
  Instruction *ReplCall = nullptr;
  BasicBlock *ReplRet = nullptr;
  BasicBlock *ReplUnwind = nullptr;

  TaskOutlineInfo() : OutlineInputs(1) {}
  TaskOutlineInfo(Function *Outline,
                  SmallVectorImpl<Value *> &OutlineInputs,
                  Instruction *ReplStart, Instruction *ReplCall,
                  BasicBlock *ReplRet, BasicBlock *ReplUnwind = nullptr)
      : Outline(Outline), OutlineInputs(OutlineInputs.begin(), OutlineInputs.end()),
        ReplStart(ReplStart), ReplCall(ReplCall),
        ReplRet(ReplRet), ReplUnwind(ReplUnwind)
  {}

  // Replaces the stored call or invoke instruction to the outlined function
  // with \p NewReplCall, and updates other information in this TaskOutlineInfo
  // struct appropriately.
  void replaceReplCall(Instruction *NewReplCall) {
    if (ReplStart == ReplCall)
      ReplStart = NewReplCall;
    ReplCall = NewReplCall;
  }

  // Helper routine to remap relevant TaskOutlineInfo values in the event, for
  // instance, that these values are themselves outlined.
  void remapOutlineInfo(ValueToValueMapTy &VMap) {
    ReplStart = cast<Instruction>(VMap[ReplStart]);
    ReplCall = cast<Instruction>(VMap[ReplCall]);
    ReplRet = cast<BasicBlock>(VMap[ReplRet]);
    if (ReplUnwind)
      ReplUnwind = cast<BasicBlock>(VMap[ReplUnwind]);
  }
};

using TaskOutlineMapTy = DenseMap<Task *, TaskOutlineInfo>;

/// Find all inputs to tasks within a function \p F, including nested tasks.
DenseMap<Task *, ValueSet>
findAllTaskInputs(Function &F, DominatorTree &DT, TaskInfo &TI);

/// Create a struct to store the inputs to pass to an outlined function for the
/// task \p T.  Stores into the struct will be inserted \p StorePt, which should
/// precede the detach.  Loads from the struct will be inserted at \p LoadPt,
/// which should be inside \p T.
std::pair<AllocaInst *, Instruction *>
createTaskArgsStruct(ValueSet &Inputs, Task *T, Instruction *StorePt,
                     Instruction *LoadPt);

/// Organize the inputs to task \p T, given in \p TaskInputs, to create an
/// appropriate set of inputs, \p HelperInputs, to pass to the outlined
/// function for \p T.
Instruction *fixupHelperInputs(Function &F, Task *T, ValueSet &TaskInputs,
                               ValueSet &HelperInputs, Instruction *StorePt,
                               Instruction *LoadPt);

/// Returns true if BasicBlock \p B is the immediate successor of a
/// detached-rethrow instruction.
bool isSuccessorOfDetachedRethrow(const BasicBlock *B);

/// Collect the set of blocks in task \p T.  All blocks enclosed by \p T will be
/// pushed onto \p TaskBlocks.  The set of blocks terminated by reattaches from
/// \p T are added to \p ReattachBlocks.  The set of blocks terminated by
/// detached-rethrow instructions are added to \p DetachedRethrowBlocks.  The
/// set of entry points to exception-handling blocks shared by \p T and other
/// tasks in the same function are added to \p SharedEHEntries.
void getTaskBlocks(Task *T, std::vector<BasicBlock *> &TaskBlocks,
                   SmallPtrSetImpl<BasicBlock *> &ReattachBlocks,
                   SmallPtrSetImpl<BasicBlock *> &DetachedRethrowBlocks,
                   SmallPtrSetImpl<BasicBlock *> &SharedEHEntries);

/// Outlines the content of task \p T in function \p F into a new helper
/// function.  The parameter \p Inputs specified the inputs to the helper
/// function.  The map \p VMap is updated with the mapping of instructions in
/// \p T to instructions in the new helper function.
Function *createHelperForTask(
    Function &F, Task *T, ValueSet &Inputs, ValueToValueMapTy &VMap,
    AssumptionCache *AC, DominatorTree *DT);

/// Replaces the detach instruction that spawns task \p T, with associated
/// TaskOutlineInfo \p Out, with a call or invoke to the outlined helper function
/// created for \p T.
Instruction *replaceDetachWithCallToOutline(Task *T, TaskOutlineInfo &Out);

/// Outlines a task \p T into a helper function that accepts the inputs \p
/// Inputs.  The map \p VMap is updated with the mapping of instructions in \p T
/// to instructions in the new helper function.  Information about the helper
/// function is returned as a TaskOutlineInfo structure.
TaskOutlineInfo outlineTask(
    Task *T, ValueSet &Inputs, ValueToValueMapTy &VMap, AssumptionCache *AC,
    DominatorTree *DT);

//----------------------------------------------------------------------------//
// Methods for lowering Tapir loops

/// Given a Tapir loop \p TL and the set of inputs to the task inside that loop,
/// returns the set of inputs for the Tapir loop itself.
ValueSet getTapirLoopInputs(TapirLoopInfo *TL, ValueSet &TaskInputs);


/// Replaces the Tapir loop \p TL, with associated TaskOutlineInfo \p Out, with
/// a call or invoke to the outlined helper function created for \p TL.
Instruction *replaceLoopWithCallToOutline(TapirLoopInfo *TL,
                                          TaskOutlineInfo &Out);

//----------------------------------------------------------------------------//
// Old lowering utils

Function *extractDetachBodyToFunction(DetachInst &Detach,
                                      DominatorTree &DT, AssumptionCache &AC,
                                      Instruction **CallSite = nullptr);

/// Abstract class for a parallel-runtime-system target for Tapir lowering.
class TapirTarget {
protected:
  Module &M;
public:
  TapirTarget(Module &M) : M(M) {}
  virtual ~TapirTarget() {}
  virtual Value *lowerGrainsizeCall(CallInst *GrainsizeCall) = 0;
  virtual void lowerTaskFrameAddrCall(CallInst *TaskFrameAddrCall);
  virtual void lowerSync(SyncInst &inst) = 0;

  virtual bool shouldProcessFunction(const Function &F);
  virtual void preProcessFunction(Function &F) = 0;
  virtual void postProcessFunction(Function &F) = 0;
  virtual void postProcessHelper(Function &F) = 0;

  virtual void processOutlinedTask(Function &F) = 0;
  virtual void processSpawner(Function &F) = 0;
  virtual void processSubTaskCall(TaskOutlineInfo &TOI, DominatorTree &DT) = 0;
};

TapirTarget *getTapirTargetFromID(Module &M, TapirTargetID TargetID);

}  // end namepsace llvm

#endif
