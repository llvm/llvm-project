//===- LazyValueInfo.h - Value constraint analysis --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
  const DataLayout *DL = nullptr;
  class TargetLibraryInfo *TLI = nullptr;
  void *PImpl = nullptr;
  LazyValueInfo(const LazyValueInfo&) = delete;
  void operator=(const LazyValueInfo&) = delete;
public:
  ~LazyValueInfo();
  LazyValueInfo() = default;
  LazyValueInfo(AssumptionCache *AC_, const DataLayout *DL_,
                TargetLibraryInfo *TLI_)
      : AC(AC_), DL(DL_), TLI(TLI_) {}
  LazyValueInfo(LazyValueInfo &&Arg)
      : AC(Arg.AC), DL(Arg.DL), TLI(Arg.TLI), PImpl(Arg.PImpl) {
    Arg.PImpl = nullptr;
  }
  LazyValueInfo &operator=(LazyValueInfo &&Arg) {
    releaseMemory();
    AC = Arg.AC;
    DL = Arg.DL;
    TLI = Arg.TLI;
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
  /// to be true or false at the specified instruction.
  /// \p Pred is a CmpInst predicate. If \p UseBlockValue is true, the block
  /// value is also taken into account.
  Tristate getPredicateAt(unsigned Pred, Value *V, Constant *C,
                          Instruction *CxtI, bool UseBlockValue);

  /// Determine whether the specified value comparison is known to be true
  /// or false at the specified instruction. While this takes two Value's,
  /// it still requires that one of them is a constant.
  /// \p Pred is a CmpInst predicate.
  /// If \p UseBlockValue is true, the block value is also taken into account.
  Tristate getPredicateAt(unsigned Pred, Value *LHS, Value *RHS,
                          Instruction *CxtI, bool UseBlockValue);

  /// Determine whether the specified value is known to be a constant at the
  /// specified instruction. Return null if not.
  Constant *getConstant(Value *V, Instruction *CxtI);

  /// Return the ConstantRange constraint that is known to hold for the
  /// specified value at the specified instruction. This may only be called
  /// on integer-typed Values.
  ConstantRange getConstantRange(Value *V, Instruction *CxtI,
                                 bool UndefAllowed = true);

  /// Return the ConstantRange constraint that is known to hold for the value
  /// at a specific use-site.
  ConstantRange getConstantRangeAtUse(const Use &U, bool UndefAllowed = true);

  /// Determine whether the specified value is known to be a
  /// constant on the specified edge.  Return null if not.
  Constant *getConstantOnEdge(Value *V, BasicBlock *FromBB, BasicBlock *ToBB,
                              Instruction *CxtI = nullptr);

  /// Return the ConstantRage constraint that is known to hold for the
  /// specified value on the specified edge. This may be only be called
  /// on integer-typed Values.
  ConstantRange getConstantRangeOnEdge(Value *V, BasicBlock *FromBB,
                                       BasicBlock *ToBB,
                                       Instruction *CxtI = nullptr);

  /// Inform the analysis cache that we have threaded an edge from
  /// PredBB to OldSucc to be from PredBB to NewSucc instead.
  void threadEdge(BasicBlock *PredBB, BasicBlock *OldSucc, BasicBlock *NewSucc);

  /// Remove information related to this value from the cache.
  void forgetValue(Value *V);

  /// Inform the analysis cache that we have erased a block.
  void eraseBlock(BasicBlock *BB);

  /// Complete flush all previously computed values
  void clear(const Module *M);

  /// Print the \LazyValueInfo Analysis.
  /// We pass in the DTree that is required for identifying which basic blocks
  /// we can solve/print for, in the LVIPrinter.
  void printLVI(Function &F, DominatorTree &DTree, raw_ostream &OS);

  // For old PM pass. Delete once LazyValueInfoWrapperPass is gone.
  void releaseMemory();

  /// Handle invalidation events in the new pass manager.
  bool invalidate(Function &F, const PreservedAnalyses &PA,
                  FunctionAnalysisManager::Invalidator &Inv);
};

/// Analysis to compute lazy value information.
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
  LazyValueInfoWrapperPass();
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

