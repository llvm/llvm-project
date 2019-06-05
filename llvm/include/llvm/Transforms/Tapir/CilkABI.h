//===- CilkABI.h - Interface to the Intel Cilk Plus runtime ----*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Cilk ABI to converts Tapir instructions to calls
// into the Cilk runtime system.
//
//===----------------------------------------------------------------------===//
#ifndef CILK_ABI_H_
#define CILK_ABI_H_

#include "llvm/IR/IRBuilder.h"
#include "llvm/Transforms/Tapir/LoopSpawning.h"
#include "llvm/Transforms/Tapir/LoweringUtils.h"

namespace llvm {
class Value;

/// CilkABILoopSpawning uses the Cilk Plus ABI to handle Tapir loops.
class CilkABILoopSpawning : public LoopOutline {
public:
  CilkABILoopSpawning(Loop *OrigLoop, unsigned Grainsize, ScalarEvolution &SE,
                      LoopInfo *LI, DominatorTree *DT, AssumptionCache *AC,
                      OptimizationRemarkEmitter &ORE)
      : LoopOutline(OrigLoop, SE, LI, DT, AC, ORE),
        M(*OrigLoop->getHeader()->getModule()),
        SpecifiedGrainsize(Grainsize)
  {}

  bool processLoop();

  virtual ~CilkABILoopSpawning() {}

protected:
  Value *canonicalizeLoopLatch(PHINode *IV, Value *Limit);

  Module &M;
  unsigned SpecifiedGrainsize;

  // Opaque Cilk RTS functions
  Function *CilkRTSCilkFor32 = nullptr;
  Function *CilkRTSCilkFor64 = nullptr;

  Function *Get__cilkrts_cilk_for_32();
  Function *Get__cilkrts_cilk_for_64();
};

class CilkABI : public TapirTarget {
  ValueToValueMapTy DetachCtxToStackFrame;

  // Cilk RTS data types
  StructType *PedigreeTy = nullptr;
  enum PedigreeFields { rank = 0, next };
  StructType *StackFrameTy = nullptr;
  enum StackFrameFields
    {
     flags = 0,
     size,
     call_parent,
     worker,
     except_data,
     ctx,
     mxcsr,
     fpcsr,
     reserved,
     parent_pedigree
    };
  StructType *WorkerTy = nullptr;
  enum WorkerFields
    {
     tail = 0,
     head,
     exc,
     protected_tail,
     ltq_limit,
     self,
     g,
     l,
     reducer_map,
     current_stack_frame,
     saved_protected_tail,
     sysdep,
     pedigree
    };

  // Opaque Cilk RTS functions
  Function *CilkRTSInit = nullptr;
  Function *CilkRTSLeaveFrame = nullptr;
  Function *CilkRTSRethrow = nullptr;
  Function *CilkRTSSync = nullptr;
  Function *CilkRTSGetNworkers = nullptr;
  Function *CilkRTSGetTLSWorker = nullptr;
  Function *CilkRTSGetTLSWorkerFast = nullptr;
  Function *CilkRTSBindThread1 = nullptr;

  // Accessors for Cilk RTS functions
  Function *Get__cilkrts_init();
  Function *Get__cilkrts_enter_frame_1();
  Function *Get__cilkrts_enter_frame_fast_1();
  Function *Get__cilkrts_leave_frame();
  Function *Get__cilkrts_rethrow();
  Function *Get__cilkrts_sync();
  Function *Get__cilkrts_detach();
  Function *Get__cilkrts_pop_frame();
  Function *Get__cilkrts_get_nworkers();
  Function *Get__cilkrts_get_tls_worker();
  Function *Get__cilkrts_get_tls_worker_fast();
  Function *Get__cilkrts_bind_thread_1();

  // Helper functions for implementing the Cilk ABI protocol
  Function *GetCilkSyncFn(bool instrument = false);
  Function *GetCilkSyncNothrowFn(bool instrument = false);
  Function *GetCilkParentEpilogueFn(bool instrument = false);
  static void EmitSaveFloatingPointState(IRBuilder<> &B, Value *SF);
  AllocaInst *CreateStackFrame(Function &F);
  Value *GetOrInitCilkStackFrame(Function &F, bool Helper,
                                 bool instrumet = false);
  CallInst *EmitCilkSetJmp(IRBuilder<> &B, Value *SF);
  bool makeFunctionDetachable(Function &Extracted, bool instrument = false);

public:
  CilkABI(Module &M);
  ~CilkABI() { DetachCtxToStackFrame.clear(); }
  Value *lowerGrainsizeCall(CallInst *GrainsizeCall) override final;
  void lowerSync(SyncInst &inst) override final;

  void preProcessFunction(Function &F) override final;
  void postProcessFunction(Function &F) override final;
  void postProcessHelper(Function &F) override final;

  void processOutlinedTask(Function &F) override final;
  void processSpawner(Function &F) override final;
  void processSubTaskCall(TaskOutlineInfo &TOI, DominatorTree &DT)
    override final;
};
}  // end of llvm namespace

#endif
