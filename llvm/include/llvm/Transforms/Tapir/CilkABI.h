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
#include "llvm/Transforms/Tapir/LoweringUtils.h"
#include "llvm/Transforms/Tapir/TapirLoopInfo.h"

namespace llvm {
class Value;

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
  FunctionCallee CilkRTSInit = nullptr;
  FunctionCallee CilkRTSLeaveFrame = nullptr;
  FunctionCallee CilkRTSRethrow = nullptr;
  FunctionCallee CilkRTSSync = nullptr;
  FunctionCallee CilkRTSGetNworkers = nullptr;
  FunctionCallee CilkRTSGetTLSWorker = nullptr;
  FunctionCallee CilkRTSGetTLSWorkerFast = nullptr;
  FunctionCallee CilkRTSBindThread1 = nullptr;

  // Accessors for opaque Cilk RTS functions
  FunctionCallee Get__cilkrts_init();
  FunctionCallee Get__cilkrts_leave_frame();
  FunctionCallee Get__cilkrts_rethrow();
  FunctionCallee Get__cilkrts_sync();
  FunctionCallee Get__cilkrts_get_nworkers();
  FunctionCallee Get__cilkrts_get_tls_worker();
  FunctionCallee Get__cilkrts_get_tls_worker_fast();
  FunctionCallee Get__cilkrts_bind_thread_1();
  // Accessors for compiler-generated Cilk RTS functions
  Function *Get__cilkrts_enter_frame_1();
  Function *Get__cilkrts_enter_frame_fast_1();
  Function *Get__cilkrts_detach();
  Function *Get__cilkrts_pop_frame();

  // Helper functions for implementing the Cilk ABI protocol
  Function *GetCilkSyncFn(bool instrument = false);
  Function *GetCilkSyncNothrowFn(bool instrument = false);
  Function *GetCilkCatchExceptionFn(Type *ExnTy);
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
  void lowerSync(SyncInst &SI) override final;

  ArgStructMode getArgStructMode() const override final;
  void addHelperAttributes(Function &F) override final;

  void preProcessFunction(Function &F, TaskInfo &TI,
                          bool OutliningTapirLoops) override final;
  void postProcessFunction(Function &F, bool OutliningTapirLoops)
    override final;
  void postProcessHelper(Function &F) override final;

  void processOutlinedTask(Function &F) override final;
  void processSpawner(Function &F) override final;
  void processSubTaskCall(TaskOutlineInfo &TOI, DominatorTree &DT)
    override final;

  LoopOutlineProcessor *getLoopOutlineProcessor(const TapirLoopInfo *TL) const
    override final;
};

/// The RuntimeCilkFor loop-outline processor transforms an outlined Tapir loop
/// to be processed using a call to a runtime method __cilkrts_cilk_for_32 or
/// __cilkrts_cilk_for_64.
class RuntimeCilkFor : public LoopOutlineProcessor {
  FunctionCallee CilkRTSCilkFor32 = nullptr;
  FunctionCallee CilkRTSCilkFor64 = nullptr;
  Type *GrainsizeType = nullptr;

  FunctionCallee Get__cilkrts_cilk_for_32();
  FunctionCallee Get__cilkrts_cilk_for_64();
public:
  RuntimeCilkFor(Module &M) : LoopOutlineProcessor(M) {
    GrainsizeType = Type::getInt32Ty(M.getContext());
  }

  ArgStructMode getArgStructMode() const override final {
    // return ArgStructMode::Dynamic;
    return ArgStructMode::Static;
  }
  void setupLoopOutlineArgs(
      Function &F, ValueSet &HelperArgs, SmallVectorImpl<Value *> &HelperInputs,
      ValueSet &InputSet, const SmallVectorImpl<Value *> &LCArgs,
      const SmallVectorImpl<Value *> &LCInputs,
      const ValueSet &TLInputsFixed)
    override final;
  unsigned getIVArgIndex(const Function &F, const ValueSet &Args) const
    override final;
  void postProcessOutline(TapirLoopInfo &TL, TaskOutlineInfo &Out,
                          ValueToValueMapTy &VMap) override final;
  void processOutlinedLoopCall(TapirLoopInfo &TL, TaskOutlineInfo &TOI,
                               DominatorTree &DT) override final;
};
}  // end of llvm namespace

#endif
