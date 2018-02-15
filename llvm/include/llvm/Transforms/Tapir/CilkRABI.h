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

class CilkRABI : public TapirTarget {
public:
  CilkRABI();
  Value *GetOrCreateWorker8(Function &F) override final;
  void createSync(SyncInst &inst, ValueToValueMapTy &DetachCtxToStackFrame)
    override final;

  Function *createDetach(DetachInst &Detach,
                         ValueToValueMapTy &DetachCtxToStackFrame,
                         DominatorTree &DT, AssumptionCache &AC) override final;
  void preProcessFunction(Function &F) override final;
  void postProcessFunction(Function &F) override final;
  void postProcessHelper(Function &F) override final;

  // struct __cilkrts_pedigree {};
  struct __cilkrts_stack_frame {};
  struct __cilkrts_worker {};
};

}  // end of llvm namespace

#endif
