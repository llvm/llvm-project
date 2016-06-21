//===- CFLAliasAnalysis.h - CFL-Based Alias Analysis Interface ---*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This is the interface for LLVM's primary stateless and local alias analysis.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_CFLALIASANALYSIS_H
#define LLVM_ANALYSIS_CFLALIASANALYSIS_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Pass.h"
#include <forward_list>

namespace llvm {

class TargetLibraryInfo;

class CFLAAResult : public AAResultBase<CFLAAResult> {
  friend AAResultBase<CFLAAResult>;
  class FunctionInfo;

public:
  explicit CFLAAResult(const TargetLibraryInfo &);
  CFLAAResult(CFLAAResult &&Arg);
  ~CFLAAResult();

  /// Handle invalidation events from the new pass manager.
  ///
  /// By definition, this result is stateless and so remains valid.
  bool invalidate(Function &, const PreservedAnalyses &) { return false; }

  /// \brief Inserts the given Function into the cache.
  void scan(Function *Fn);

  void evict(Function *Fn);

  /// \brief Ensures that the given function is available in the cache.
  /// Returns the appropriate entry from the cache.
  const Optional<FunctionInfo> &ensureCached(Function *Fn);

  AliasResult query(const MemoryLocation &LocA, const MemoryLocation &LocB);

  AliasResult alias(const MemoryLocation &LocA, const MemoryLocation &LocB) {
    if (LocA.Ptr == LocB.Ptr)
      return LocA.Size == LocB.Size ? MustAlias : PartialAlias;

    // Comparisons between global variables and other constants should be
    // handled by BasicAA.
    // TODO: ConstantExpr handling -- CFLAA may report NoAlias when comparing
    // a GlobalValue and ConstantExpr, but every query needs to have at least
    // one Value tied to a Function, and neither GlobalValues nor ConstantExprs
    // are.
    if (isa<Constant>(LocA.Ptr) && isa<Constant>(LocB.Ptr))
      return AAResultBase::alias(LocA, LocB);

    AliasResult QueryResult = query(LocA, LocB);
    if (QueryResult == MayAlias)
      return AAResultBase::alias(LocA, LocB);

    return QueryResult;
  }

private:
  struct FunctionHandle final : public CallbackVH {
    FunctionHandle(Function *Fn, CFLAAResult *Result)
        : CallbackVH(Fn), Result(Result) {
      assert(Fn != nullptr);
      assert(Result != nullptr);
    }

    void deleted() override { removeSelfFromCache(); }
    void allUsesReplacedWith(Value *) override { removeSelfFromCache(); }

  private:
    CFLAAResult *Result;

    void removeSelfFromCache() {
      assert(Result != nullptr);
      auto *Val = getValPtr();
      Result->evict(cast<Function>(Val));
      setValPtr(nullptr);
    }
  };

  const TargetLibraryInfo &TLI;

  /// \brief Cached mapping of Functions to their StratifiedSets.
  /// If a function's sets are currently being built, it is marked
  /// in the cache as an Optional without a value. This way, if we
  /// have any kind of recursion, it is discernable from a function
  /// that simply has empty sets.
  DenseMap<Function *, Optional<FunctionInfo>> Cache;
  std::forward_list<FunctionHandle> Handles;

  FunctionInfo buildSetsFrom(Function *F);
};

/// Analysis pass providing a never-invalidated alias analysis result.
///
/// FIXME: We really should refactor CFL to use the analysis more heavily, and
/// in particular to leverage invalidation to trigger re-computation of sets.
class CFLAA : public AnalysisInfoMixin<CFLAA> {
  friend AnalysisInfoMixin<CFLAA>;
  static char PassID;

public:
  typedef CFLAAResult Result;

  CFLAAResult run(Function &F, AnalysisManager<Function> &AM);
};

/// Legacy wrapper pass to provide the CFLAAResult object.
class CFLAAWrapperPass : public ImmutablePass {
  std::unique_ptr<CFLAAResult> Result;

public:
  static char ID;

  CFLAAWrapperPass();

  CFLAAResult &getResult() { return *Result; }
  const CFLAAResult &getResult() const { return *Result; }

  void initializePass() override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;
};

//===--------------------------------------------------------------------===//
//
// createCFLAAWrapperPass - This pass implements a set-based approach to
// alias analysis.
//
ImmutablePass *createCFLAAWrapperPass();
}

#endif
