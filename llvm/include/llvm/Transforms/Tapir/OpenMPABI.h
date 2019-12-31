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

#include "llvm/Transforms/Tapir/LoweringUtils.h"

namespace llvm {
class Value;

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
  OMPRTL__kmpc_global_num_threads,
};

enum OpenMPSchedType {
  OMP_sch_static = 34,
};

class OpenMPABI : public TapirTarget {
public:
  OpenMPABI(Module &M);
  Value *lowerGrainsizeCall(CallInst *GrainsizeCall) override final;
  void lowerSync(SyncInst &SI) override final;

  void preProcessFunction(Function &F, TaskInfo &TI,
                          bool OutliningTapirLoops) override final;
  void postProcessFunction(Function &F, bool OutliningTapirLoops)
    override final;
  void postProcessHelper(Function &F) override final;

  void processOutlinedTask(Function &F) override final;
  void processSpawner(Function &F) override final;
  void processSubTaskCall(TaskOutlineInfo &TOI, DominatorTree &DT)
    override final;
};

}  // end of llvm namespace

#endif
