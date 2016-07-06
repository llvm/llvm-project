//=- CFLAndersAliasAnalysis.h - Unification-based Alias Analysis ---*- C++-*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This is the interface for LLVM's inclusion-based alias analysis
/// implemented with CFL graph reachability.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_CFLANDERSALIASANALYSIS_H
#define LLVM_ANALYSIS_CFLANDERSALIASANALYSIS_H

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/IR/Function.h"
#include "llvm/Pass.h"

namespace llvm {

class CFLAndersAAResult : public AAResultBase<CFLAndersAAResult> {
  friend AAResultBase<CFLAndersAAResult>;

public:
  explicit CFLAndersAAResult();

  AliasResult alias(const MemoryLocation &LocA, const MemoryLocation &LocB) {
    // Dummy implementation
    return AAResultBase::alias(LocA, LocB);
  }
};

/// Analysis pass providing a never-invalidated alias analysis result.
///
/// FIXME: We really should refactor CFL to use the analysis more heavily, and
/// in particular to leverage invalidation to trigger re-computation.
class CFLAndersAA : public AnalysisInfoMixin<CFLAndersAA> {
  friend AnalysisInfoMixin<CFLAndersAA>;
  static char PassID;

public:
  typedef CFLAndersAAResult Result;

  CFLAndersAAResult run(Function &F, AnalysisManager<Function> &AM);
};

/// Legacy wrapper pass to provide the CFLAndersAAResult object.
class CFLAndersAAWrapperPass : public ImmutablePass {
  std::unique_ptr<CFLAndersAAResult> Result;

public:
  static char ID;

  CFLAndersAAWrapperPass();

  CFLAndersAAResult &getResult() { return *Result; }
  const CFLAndersAAResult &getResult() const { return *Result; }

  void initializePass() override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;
};

//===--------------------------------------------------------------------===//
//
// createCFLAndersAAWrapperPass - This pass implements a set-based approach to
// alias analysis.
//
ImmutablePass *createCFLAndersAAWrapperPass();
}

#endif
