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
  // PHINode* canonicalizeIVs(Type *Ty);
  Value *canonicalizeLoopLatch(PHINode *IV, Value *Limit);

  unsigned SpecifiedGrainsize;
// private:
//   /// Report an analysis message to assist the user in diagnosing loops that are
//   /// not transformed.  These are handled as LoopAccessReport rather than
//   /// VectorizationReport because the << operator of LoopSpawningReport returns
//   /// LoopAccessReport.
//   void emitAnalysis(const LoopAccessReport &Message) const {
//     emitAnalysisDiag(OrigLoop, *ORE, Message);
//   }
};

class CilkABI : public TapirTarget {
public:
  CilkABI();
  Value *lowerGrainsizeCall(CallInst *GrainsizeCall) override final;
  void createSync(SyncInst &inst, ValueToValueMapTy &DetachCtxToStackFrame)
    override final;

  Function *createDetach(DetachInst &Detach,
                         ValueToValueMapTy &DetachCtxToStackFrame,
                         DominatorTree &DT, AssumptionCache &AC) override final;
  void preProcessFunction(Function &F) override final;
  void postProcessFunction(Function &F) override final;
  void postProcessHelper(Function &F) override final;

  struct __cilkrts_pedigree {};
  struct __cilkrts_stack_frame {};
  struct __cilkrts_worker {};

};

}  // end of llvm namespace

#endif
