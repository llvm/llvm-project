//===- OpenMPABI.h - Interface to the OpenMP runtime -----------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the OpenMP ABI to converts Tapir instructions to calls
// into the OpenMP runtime system.
//
//===----------------------------------------------------------------------===//
#ifndef OMP_ABI_H_
#define OMP_ABI_H_

#include "llvm/Transforms/Tapir/TapirUtils.h"

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

}  // end of llvm namespace

#endif
