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

#include "llvm/Transforms/Tapir/LoweringUtils.h"

namespace llvm {
class Value;

class CilkRABI : public TapirTarget {
  ValueToValueMapTy DetachCtxToStackFrame;
public:
  CilkRABI(Module &M) : TapirTarget(M) {}
  ~CilkRABI() { DetachCtxToStackFrame.clear(); }
  Value *lowerGrainsizeCall(CallInst *GrainsizeCall) override final;
  void lowerSync(SyncInst &inst) override final;

  void preProcessFunction(Function &F) override final;
  void postProcessFunction(Function &F) override final;
  void postProcessHelper(Function &F) override final;

  void processOutlinedTask(Function &F) override final;
  void processSpawner(Function &F) override final;
  void processSubTaskCall(TaskOutlineInfo &TOI, DominatorTree &DT)
    override final;

  // struct __cilkrts_pedigree {};
  struct __cilkrts_stack_frame {};
  struct __cilkrts_worker {};
};

}  // end of llvm namespace

#endif
