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

  void replaceReplCall(Instruction *NewReplCall) {
    if (ReplStart == ReplCall)
      ReplStart = NewReplCall;
    ReplCall = NewReplCall;
  }

  void remapOutlineInfo(ValueToValueMapTy &VMap) {
    ReplStart = cast<Instruction>(VMap[ReplStart]);
    ReplCall = cast<Instruction>(VMap[ReplCall]);
    ReplRet = cast<BasicBlock>(VMap[ReplRet]);
    if (ReplUnwind)
      ReplUnwind = cast<BasicBlock>(VMap[ReplUnwind]);
  }
};
using TaskOutlineMapTy = DenseMap<Task *, TaskOutlineInfo>;

DenseMap<Task *, ValueSet>
findAllTaskInputs(Function &F, DominatorTree &DT, TaskInfo &TI);

std::pair<AllocaInst *, Instruction *>
createTaskArgsStruct(ValueSet &Inputs, Task *T, Instruction *StorePt,
                     Instruction *LoadPt);

Instruction *fixupHelperInputs(Function &F, Task *T, ValueSet &TaskInputs,
                               ValueSet &HelperInputs, Instruction *StorePt,
                               Instruction *LoadPt);

bool isSuccessorOfDetachedRethrow(const BasicBlock *B);

void getTaskBlocks(Task *T, std::vector<BasicBlock *> &TaskBlocks,
                   SmallPtrSetImpl<BasicBlock *> &ReattachBlocks,
                   SmallPtrSetImpl<BasicBlock *> &DetachedRethrowBlocks,
                   SmallPtrSetImpl<BasicBlock *> &SharedEHEntries);

Function *createHelperForTask(
    Function &F, Task *T, ValueSet &Inputs, ValueToValueMapTy &VMap,
    AssumptionCache *AC, DominatorTree *DT);

Instruction *replaceDetachWithCallToOutline(Task *T, TaskOutlineInfo &Out);

TaskOutlineInfo outlineTask(
    Task *T, ValueSet &Inputs, ValueToValueMapTy &VMap, AssumptionCache *AC,
    DominatorTree *DT);

//----------------------------------------------------------------------------//
// Methods for lowering Tapir loops

ValueSet getTapirLoopInputs(TapirLoopInfo *TL, ValueSet &TaskInputs);

Instruction *replaceLoopWithCallToOutline(TapirLoopInfo *TL,
                                          TaskOutlineInfo &Out);
//----------------------------------------------------------------------------//
// Old lowering utils

Function *extractDetachBodyToFunction(DetachInst &Detach,
                                      DominatorTree &DT, AssumptionCache &AC,
                                      Instruction **CallSite = nullptr);

class TapirTarget {
public:
  virtual ~TapirTarget() {};
  virtual Value *lowerGrainsizeCall(CallInst *GrainsizeCall) = 0;
  virtual void createSync(SyncInst &inst) = 0;
  virtual Function *createDetach(DetachInst &Detach,
                                 DominatorTree &DT, AssumptionCache &AC) = 0;
  virtual bool shouldProcessFunction(const Function &F);
  virtual void preProcessFunction(Function &F) = 0;
  virtual void postProcessFunction(Function &F) = 0;
  virtual void postProcessHelper(Function &F) = 0;

  virtual void processOutlinedTask(Function &F) = 0;
  virtual void processSpawner(Function &F) = 0;
  virtual void processSubTaskCall(TaskOutlineInfo &TOI, DominatorTree &DT) = 0;
};

TapirTarget *getTapirTargetFromID(TapirTargetID TargetID);

}  // end namepsace llvm

#endif
