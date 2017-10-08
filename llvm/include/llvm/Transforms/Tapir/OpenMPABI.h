//===- OpenMPABI.h - Interface to the OpenMP runtime ----*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass is a simple pass wrapper around the PromoteMemToReg function call
// exposed by the Utils library.
//
//===----------------------------------------------------------------------===//
#ifndef OMP_ABI_H_
#define OMP_ABI_H_

#include "llvm/Transforms/Scalar.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/TypeBuilder.h"
#include "llvm/IR/ValueSymbolTable.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/UnifyFunctionExitNodes.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include "llvm/Transforms/Tapir/TapirUtils.h"
#include <deque>

namespace llvm {
enum OpenMPRuntimeFunction {
  OMPRTL__kmpc_fork_call,
  OMPRTL__kmpc_for_static_init_4,
  OMPRTL__kmpc_for_static_fini,
  OMPRTL__kmpc_master,
  OMPRTL__kmpc_end_master,
  OMPRTL__kmpc_omp_task_alloc,
  OMPRTL__kmpc_omp_task,
  OMPRTL__kmpc_omp_taskwait,
  OMPRTL__kmpc_global_thread_num,
  OMPRTL__kmpc_barrier,
};

enum OpenMPSchedType {
  OMP_sch_static = 34,
};

namespace tapir {

class OpenMPABI : public TapirTarget {
public:
OpenMPABI();
Value *GetOrCreateWorker8(Function &F) override final;
void createSync(SyncInst &inst, ValueToValueMapTy &DetachCtxToStackFrame) override final;

Function *createDetach(DetachInst &Detach,
                       ValueToValueMapTy &DetachCtxToStackFrame,
                       DominatorTree &DT, AssumptionCache &AC) override final;
void preProcessFunction(Function &F) override final;
void postProcessFunction(Function &F) override final;
void postProcessHelper(Function &F) override final;
};

}  // end of tapir namespace
}  // end of llvm namespace

#endif
