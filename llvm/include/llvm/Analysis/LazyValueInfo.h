//===- LazyValueInfo.h - Value constraint analysis --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interface for lazy computation of value constraint
// information.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_LAZYVALUEINFO_H
#define LLVM_ANALYSIS_LAZYVALUEINFO_H

#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

namespace llvm {
  class AssumptionCache;
  class Constant;
  class ConstantRange;
  class DataLayout;
  class DominatorTree;
  class Instruction;
  class TargetLibraryInfo;
  class Value;

/// This pass computes, caches, and vends lazy value constraint information.
class LazyValueInfo {
  friend class LazyValueInfoWrapperPass;
  AssumptionCache *AC = nullptr;
  class TargetLibraryInfo *TLI = nullptr;
  DominatorTree *DT = nullptr;
  void *PImpl = nullptr;
  LazyValueInfo(const LazyValueInfo&) = delete;
  void operator=(const LazyValueInfo&) = delete;
public:
  ~LazyValueInfo();
  LazyValueInfo() {}
  LazyValueInfo(AssumptionCache *AC_, TargetLibraryInfo *TLI_,
                DominatorTree *DT_)
      : AC(AC_), TLI(TLI_), DT(DT_) {}
  LazyValueInfo(LazyValueInfo &&Arg)
      : AC(Arg.AC), TLI(Arg.TLI), DT(Arg.DT), PImpl(Arg.PImpl) {
    Arg.PImpl = nullptr;
  }
  LazyValueInfo &operator=(LazyValueInfo &&Arg) {
    releaseMemory();
    AC = Arg.AC;
    TLI = Arg.TLI;
    DT = Arg.DT;
    PImpl = Arg.PImpl;
    Arg.PImpl = nullptr;
    return *this;
  }

  /// This is used to return true/false/dunno results.
  enum Tristate {
    Unknown = -1, False = 0, True = 1
  };

  // Public query interface.

  /// Determine whether the specified value comparison with a constant is known
  /// to be true or false on the specified CFG edge.
  /// Pred is a CmpInst predicate.
  Tristate getPredicateOnEdge(unsigned Pred, Value *V, Constant *C,
                              BasicBlock *FromBB, BasicBlock *ToBB,
                              Instruction *CxtI = nullptr);

  /// Determine whether the specified value comparison with a constant is known
  /// to be true or false at the specified instruction
  /// (from an assume intrinsic). Pred is a CmpInst predicate.
  Tristate getPredicateAt(unsigned Pred, Value *V, Constant *C,
                          Instruction *CxtI);

  /// Determine whether the specified value is known to be a
  /// constant at the end of the specified block.  Return null if not.
  Constant *getConstant(Value *V, BasicBlock *BB, Instruction *CxtI = nullptr);

  /// Return the ConstantRange constraint that is known to hold for the
  /// specified value at the end of the specified block. This may only be called
  /// on integer-typed Values.
  ConstantRange getConstantRange(Value *V, BasicBlock *BB, Instruction *CxtI = nullptr);

  /// Determine whether the specified value is known to be a
  /// constant on the specified edge.  Return null if not.
  Constant *getConstantOnEdge(Value *V, BasicBlock *FromBB, BasicBlock *ToBB,
                              Instruction *CxtI = nullptr);

  /// Inform the analysis cache that we have threaded an edge from
  /// PredBB to OldSucc to be from PredBB to NewSucc instead.
  void threadEdge(BasicBlock *PredBB, BasicBlock *OldSucc, BasicBlock *NewSucc);

  /// Inform the analysis cache that we have erased a block.
  void eraseBlock(BasicBlock *BB);

  // For old PM pass. Delete once LazyValueInfoWrapperPass is gone.
  void releaseMemory();

  /// Handle invalidation events in the new pass manager.
  bool invalidate(Function &F, const PreservedAnalyses &PA,
                  FunctionAnalysisManager::Invalidator &Inv);
};

/// \brief Analysis to compute lazy value information.
class LazyValueAnalysis : public AnalysisInfoMixin<LazyValueAnalysis> {
public:
  typedef LazyValueInfo Result;
  Result run(Function &F, FunctionAnalysisManager &FAM);

private:
  static AnalysisKey Key;
  friend struct AnalysisInfoMixin<LazyValueAnalysis>;
};

/// Wrapper around LazyValueInfo.
class LazyValueInfoWrapperPass : public FunctionPass {
  LazyValueInfoWrapperPass(const LazyValueInfoWrapperPass&) = delete;
  void operator=(const LazyValueInfoWrapperPass&) = delete;
public:
  static char ID;
  LazyValueInfoWrapperPass() : FunctionPass(ID) {
    initializeLazyValueInfoWrapperPassPass(*PassRegistry::getPassRegistry());
  }
  ~LazyValueInfoWrapperPass() override {
    assert(!Info.PImpl && "releaseMemory not called");
  }

  LazyValueInfo &getLVI();

  void getAnalysisUsage(AnalysisUsage &AU) const override;
  void releaseMemory() override;
  bool runOnFunction(Function &F) override;
private:
  LazyValueInfo Info;
};

}  // end namespace llvm

#endif

