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
        SpecifiedGrainsize(Grainsize)
  {}

  bool processLoop();

  virtual ~CilkABILoopSpawning() {}

protected:
  Value *canonicalizeLoopLatch(PHINode *IV, Value *Limit);

  unsigned SpecifiedGrainsize;
};

class CilkABI : public TapirTarget {
  ValueToValueMapTy DetachCtxToStackFrame;
public:
  CilkABI(Module &M) : TapirTarget(M) {}
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

  struct __cilkrts_pedigree {};
  struct __cilkrts_stack_frame {};
  struct __cilkrts_worker {};

};
}  // end of llvm namespace

#endif
