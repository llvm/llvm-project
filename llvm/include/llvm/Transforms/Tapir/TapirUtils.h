//===- TapirUtils.h - Utility functions for handling Tapir -----*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements several utility functions for operating with Tapir.
//
//===----------------------------------------------------------------------===//

#ifndef TAPIR_UTILS_H_
#define TAPIR_UTILS_H_

#include "llvm/Transforms/Tapir/TapirTypes.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

namespace llvm {

class AssumptionCache;
class BasicBlock;
class CallInst;
class DetachInst;
class DominatorTree;
class Function;
class Instruction;
class SyncInst;
class Value;

bool verifyDetachedCFG(const DetachInst &Detach, DominatorTree &DT,
                       bool error = true);

bool populateDetachedCFG(const DetachInst &Detach, DominatorTree &DT,
                         SmallPtrSetImpl<BasicBlock *> &functionPieces,
                         SmallPtrSetImpl<BasicBlock *> &ExitBlocks);

Function *extractDetachBodyToFunction(DetachInst &Detach,
                                      DominatorTree &DT, AssumptionCache &AC,
                                      Instruction **CallSite = nullptr);

class TapirTarget {
public:
  //! For use in loopspawning grainsize calculation
  virtual Value *GetOrCreateWorker8(Function &F) = 0;
  virtual void createSync(SyncInst &inst,
                          ValueToValueMapTy &DetachCtxToStackFrame) = 0;
  virtual Function *createDetach(DetachInst &Detach,
                                 ValueToValueMapTy &DetachCtxToStackFrame,
                                 DominatorTree &DT, AssumptionCache &AC) = 0;
  virtual bool shouldProcessFunction(const Function &F);
  virtual void preProcessFunction(Function &F) = 0;
  virtual void postProcessFunction(Function &F) = 0;
  virtual void postProcessHelper(Function &F) = 0;
};

TapirTarget *getTapirTargetFromType(TapirTargetType Type);

}  // end namepsace llvm

#endif
