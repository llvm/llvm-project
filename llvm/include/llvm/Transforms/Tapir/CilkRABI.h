//===- CilkRABI.h - Interface to the CilkR runtime system ------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the CilkR ABI to converts Tapir instructions to calls
// into the CilkR runtime system.
//
//===----------------------------------------------------------------------===//
#ifndef CILK_RABI_H_
#define CILK_RABI_H_

#include "llvm/IR/IRBuilder.h"
#include "llvm/Transforms/Tapir/LoweringUtils.h"

namespace llvm {
class Value;

class CilkRABI : public TapirTarget {
  ValueToValueMapTy DetachCtxToStackFrame;

  // Cilk RTS data types
  StructType *StackFrameTy = nullptr;
  enum StackFrameFields
    {
     flags = 0,
     call_parent,
     worker,
     // except_data,
     ctx,
     mxcsr,
     fpcsr,
     reserved,
     magic,
    };
  StructType *WorkerTy = nullptr;
  enum WorkerFields
    {
     tail = 0,
     head,
     exc,
     ltq_limit,
     self,
     g,
     l,
     current_stack_frame,
     // reducer_map,
    };

  // Opaque Cilk RTS functions
  Function *CilkRTSLeaveFrame = nullptr;
  // Function *CilkRTSRethrow = nullptr;
  Function *CilkRTSSync = nullptr;
  Function *CilkRTSGetNworkers = nullptr;
  Function *CilkRTSGetTLSWorker = nullptr;

  // Accessors for Cilk RTS functions
  Function *Get__cilkrts_enter_frame();
  Function *Get__cilkrts_enter_frame_fast();
  Function *Get__cilkrts_leave_frame();
  // Function *Get__cilkrts_rethrow();
  Function *Get__cilkrts_sync();
  Function *Get__cilkrts_detach();
  Function *Get__cilkrts_pop_frame();
  Function *Get__cilkrts_get_nworkers();
  Function *Get__cilkrts_get_tls_worker();

  // Helper functions for implementing the Cilk ABI protocol
  Function *GetCilkSyncFn();
  Function *GetCilkParentEpilogueFn();
  static void EmitSaveFloatingPointState(IRBuilder<> &B, Value *SF);
  AllocaInst *CreateStackFrame(Function &F);
  Value *GetOrInitCilkStackFrame(Function &F, bool Helper);
  CallInst *EmitCilkSetJmp(IRBuilder<> &B, Value *SF);
  bool makeFunctionDetachable(Function &Extracted);

public:
  CilkRABI(Module &M);
  ~CilkRABI() { DetachCtxToStackFrame.clear(); }
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
