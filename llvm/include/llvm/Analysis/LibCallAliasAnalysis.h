//===- LibCallAliasAnalysis.h - Implement AliasAnalysis for libcalls ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the LibCallAliasAnalysis class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_LIBCALL_AA_H
#define LLVM_ANALYSIS_LIBCALL_AA_H

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Pass.h"

namespace llvm {
  class LibCallInfo;
  class LibCallFunctionInfo;
  
  /// LibCallAliasAnalysis - Alias analysis driven from LibCallInfo.
  struct LibCallAliasAnalysis : public FunctionPass, AliasAnalysis {
    static char ID; // Class identification
    
    LibCallInfo *LCI;
    
    explicit LibCallAliasAnalysis(LibCallInfo *LC = 0)
      : FunctionPass(&ID), LCI(LC) {
    }
    explicit LibCallAliasAnalysis(const void *ID, LibCallInfo *LC)
      : FunctionPass(ID), LCI(LC) {
    }
    ~LibCallAliasAnalysis();
    
    ModRefResult getModRefInfo(CallSite CS, Value *P, unsigned Size);
    
    ModRefResult getModRefInfo(CallSite CS1, CallSite CS2) {
      // TODO: Could compare two direct calls against each other if we cared to.
      return AliasAnalysis::getModRefInfo(CS1,CS2);
    }
    
    virtual void getAnalysisUsage(AnalysisUsage &AU) const;
    
    virtual bool runOnFunction(Function &F) {
      InitializeAliasAnalysis(this);                 // set up super class
      return false;
    }
    
    /// hasNoModRefInfoForCalls - We can provide mod/ref information against
    /// non-escaping allocations.
    virtual bool hasNoModRefInfoForCalls() const { return false; }
  private:
    ModRefResult AnalyzeLibCallDetails(const LibCallFunctionInfo *FI,
                                       CallSite CS, Value *P, unsigned Size);
  };
}  // End of llvm namespace

#endif
