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

#include "llvm/Pass.h"

namespace llvm {
  class Constant;
  class DataLayout;
  class TargetLibraryInfo;
  class Value;
  
/// LazyValueInfo - This pass computes, caches, and vends lazy value constraint
/// information.
class LazyValueInfo : public FunctionPass {
  class DataLayout *TD;
  class TargetLibraryInfo *TLI;
  void *PImpl;
  LazyValueInfo(const LazyValueInfo&) LLVM_DELETED_FUNCTION;
  void operator=(const LazyValueInfo&) LLVM_DELETED_FUNCTION;
public:
  static char ID;
  LazyValueInfo() : FunctionPass(ID), PImpl(0) {
    initializeLazyValueInfoPass(*PassRegistry::getPassRegistry());
  }
  ~LazyValueInfo() { assert(PImpl == 0 && "releaseMemory not called"); }

  /// Tristate - This is used to return true/false/dunno results.
  enum Tristate {
    Unknown = -1, False = 0, True = 1
  };
  
  
  // Public query interface.
  
  /// getPredicateOnEdge - Determine whether the specified value comparison
  /// with a constant is known to be true or false on the specified CFG edge.
  /// Pred is a CmpInst predicate.
  Tristate getPredicateOnEdge(unsigned Pred, Value *V, Constant *C,
                              BasicBlock *FromBB, BasicBlock *ToBB);
  
  
  /// getConstant - Determine whether the specified value is known to be a
  /// constant at the end of the specified block.  Return null if not.
  Constant *getConstant(Value *V, BasicBlock *BB);

  /// getConstantOnEdge - Determine whether the specified value is known to be a
  /// constant on the specified edge.  Return null if not.
  Constant *getConstantOnEdge(Value *V, BasicBlock *FromBB, BasicBlock *ToBB);
  
  /// threadEdge - Inform the analysis cache that we have threaded an edge from
  /// PredBB to OldSucc to be from PredBB to NewSucc instead.
  void threadEdge(BasicBlock *PredBB, BasicBlock *OldSucc, BasicBlock *NewSucc);
  
  /// eraseBlock - Inform the analysis cache that we have erased a block.
  void eraseBlock(BasicBlock *BB);
  
  // Implementation boilerplate.
  
  virtual void getAnalysisUsage(AnalysisUsage &AU) const;
  virtual void releaseMemory();
  virtual bool runOnFunction(Function &F);
};

}  // end namespace llvm

#endif

