//===- BasicAliasAnalysis.h - Stateless, local Alias Analysis ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This is the interface for LLVM's primary stateless and local alias analysis.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_BASICALIASANALYSIS_H
#define LLVM_ANALYSIS_BASICALIASANALYSIS_H

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include <memory>
#include <utility>

namespace llvm {

class AssumptionCache;
class BasicBlock;
class DataLayout;
class DominatorTree;
class Function;
class GEPOperator;
class PHINode;
class SelectInst;
class TargetLibraryInfo;
class PhiValues;
class Value;

/// This is the AA result object for the basic, local, and stateless alias
/// analysis. It implements the AA query interface in an entirely stateless
/// manner. As one consequence, it is never invalidated due to IR changes.
/// While it does retain some storage, that is used as an optimization and not
/// to preserve information from query to query. However it does retain handles
/// to various other analyses and must be recomputed when those analyses are.
class BasicAAResult : public AAResultBase<BasicAAResult> {
  friend AAResultBase<BasicAAResult>;

  const DataLayout &DL;
  const Function &F;
  const TargetLibraryInfo &TLI;
  AssumptionCache &AC;
  DominatorTree *DT;
  PhiValues *PV;

public:
  BasicAAResult(const DataLayout &DL, const Function &F,
                const TargetLibraryInfo &TLI, AssumptionCache &AC,
                DominatorTree *DT = nullptr, PhiValues *PV = nullptr)
      : DL(DL), F(F), TLI(TLI), AC(AC), DT(DT), PV(PV) {}

  BasicAAResult(const BasicAAResult &Arg)
      : AAResultBase(Arg), DL(Arg.DL), F(Arg.F), TLI(Arg.TLI), AC(Arg.AC),
        DT(Arg.DT), PV(Arg.PV) {}
  BasicAAResult(BasicAAResult &&Arg)
      : AAResultBase(std::move(Arg)), DL(Arg.DL), F(Arg.F), TLI(Arg.TLI),
        AC(Arg.AC), DT(Arg.DT), PV(Arg.PV) {}

  /// Handle invalidation events in the new pass manager.
  bool invalidate(Function &Fn, const PreservedAnalyses &PA,
                  FunctionAnalysisManager::Invalidator &Inv);

  AliasResult alias(const MemoryLocation &LocA, const MemoryLocation &LocB,
                    AAQueryInfo &AAQI);

  ModRefInfo getModRefInfo(const CallBase *Call, const MemoryLocation &Loc,
                           AAQueryInfo &AAQI);

  ModRefInfo getModRefInfo(const CallBase *Call1, const CallBase *Call2,
                           AAQueryInfo &AAQI);

  /// Chases pointers until we find a (constant global) or not.
  bool pointsToConstantMemory(const MemoryLocation &Loc, AAQueryInfo &AAQI,
                              bool OrLocal);

  /// Get the location associated with a pointer argument of a callsite.
  ModRefInfo getArgModRefInfo(const CallBase *Call, unsigned ArgIdx);

  /// Returns the behavior when calling the given call site.
  FunctionModRefBehavior getModRefBehavior(const CallBase *Call);

  /// Returns the behavior when calling the given function. For use when the
  /// call site is not known.
  FunctionModRefBehavior getModRefBehavior(const Function *Fn);

private:
  struct DecomposedGEP;

  /// Tracks phi nodes we have visited.
  ///
  /// When interpret "Value" pointer equality as value equality we need to make
  /// sure that the "Value" is not part of a cycle. Otherwise, two uses could
  /// come from different "iterations" of a cycle and see different values for
  /// the same "Value" pointer.
  ///
  /// The following example shows the problem:
  ///   %p = phi(%alloca1, %addr2)
  ///   %l = load %ptr
  ///   %addr1 = gep, %alloca2, 0, %l
  ///   %addr2 = gep  %alloca2, 0, (%l + 1)
  ///      alias(%p, %addr1) -> MayAlias !
  ///   store %l, ...
  SmallPtrSet<const BasicBlock *, 8> VisitedPhiBBs;

  /// Tracks instructions visited by pointsToConstantMemory.
  SmallPtrSet<const Value *, 16> Visited;

  static DecomposedGEP
  DecomposeGEPExpression(const Value *V, const DataLayout &DL,
                         AssumptionCache *AC, DominatorTree *DT);

  /// A Heuristic for aliasGEP that searches for a constant offset
  /// between the variables.
  ///
  /// GetLinearExpression has some limitations, as generally zext(%x + 1)
  /// != zext(%x) + zext(1) if the arithmetic overflows. GetLinearExpression
  /// will therefore conservatively refuse to decompose these expressions.
  /// However, we know that, for all %x, zext(%x) != zext(%x + 1), even if
  /// the addition overflows.
  bool
  constantOffsetHeuristic(const DecomposedGEP &GEP, LocationSize V1Size,
                          LocationSize V2Size, AssumptionCache *AC,
                          DominatorTree *DT);

  bool isValueEqualInPotentialCycles(const Value *V1, const Value *V2);

  void subtractDecomposedGEPs(DecomposedGEP &DestGEP,
                              const DecomposedGEP &SrcGEP);

  AliasResult aliasGEP(const GEPOperator *V1, LocationSize V1Size,
                       const Value *V2, LocationSize V2Size,
                       const Value *UnderlyingV1, const Value *UnderlyingV2,
                       AAQueryInfo &AAQI);

  AliasResult aliasPHI(const PHINode *PN, LocationSize PNSize,
                       const Value *V2, LocationSize V2Size, AAQueryInfo &AAQI);

  AliasResult aliasSelect(const SelectInst *SI, LocationSize SISize,
                          const Value *V2, LocationSize V2Size,
                          AAQueryInfo &AAQI);

  AliasResult aliasCheck(const Value *V1, LocationSize V1Size,
                         const Value *V2, LocationSize V2Size,
                         AAQueryInfo &AAQI);

  AliasResult aliasCheckRecursive(const Value *V1, LocationSize V1Size,
                                  const Value *V2, LocationSize V2Size,
                                  AAQueryInfo &AAQI, const Value *O1,
                                  const Value *O2);
};

/// Analysis pass providing a never-invalidated alias analysis result.
class BasicAA : public AnalysisInfoMixin<BasicAA> {
  friend AnalysisInfoMixin<BasicAA>;

  static AnalysisKey Key;

public:
  using Result = BasicAAResult;

  BasicAAResult run(Function &F, FunctionAnalysisManager &AM);
};

/// Legacy wrapper pass to provide the BasicAAResult object.
class BasicAAWrapperPass : public FunctionPass {
  std::unique_ptr<BasicAAResult> Result;

  virtual void anchor();

public:
  static char ID;

  BasicAAWrapperPass();

  BasicAAResult &getResult() { return *Result; }
  const BasicAAResult &getResult() const { return *Result; }

  bool runOnFunction(Function &F) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;
};

FunctionPass *createBasicAAWrapperPass();

/// A helper for the legacy pass manager to create a \c BasicAAResult object
/// populated to the best of our ability for a particular function when inside
/// of a \c ModulePass or a \c CallGraphSCCPass.
BasicAAResult createLegacyPMBasicAAResult(Pass &P, Function &F);

/// This class is a functor to be used in legacy module or SCC passes for
/// computing AA results for a function. We store the results in fields so that
/// they live long enough to be queried, but we re-use them each time.
class LegacyAARGetter {
  Pass &P;
  Optional<BasicAAResult> BAR;
  Optional<AAResults> AAR;

public:
  LegacyAARGetter(Pass &P) : P(P) {}
  AAResults &operator()(Function &F) {
    BAR.emplace(createLegacyPMBasicAAResult(P, F));
    AAR.emplace(createLegacyPMAAResults(P, F, *BAR));
    return *AAR;
  }
};

} // end namespace llvm

#endif // LLVM_ANALYSIS_BASICALIASANALYSIS_H
