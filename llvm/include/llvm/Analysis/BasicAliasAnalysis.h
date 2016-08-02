//===- BasicAliasAnalysis.h - Stateless, local Alias Analysis ---*- C++ -*-===//
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

#ifndef LLVM_ANALYSIS_BASICALIASANALYSIS_H
#define LLVM_ANALYSIS_BASICALIASANALYSIS_H

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/ErrorHandling.h"

namespace llvm {
class AssumptionCache;
class DominatorTree;
class LoopInfo;

/// This is the AA result object for the basic, local, and stateless alias
/// analysis. It implements the AA query interface in an entirely stateless
/// manner. As one consequence, it is never invalidated. While it does retain
/// some storage, that is used as an optimization and not to preserve
/// information from query to query.
class BasicAAResult : public AAResultBase<BasicAAResult> {
  friend AAResultBase<BasicAAResult>;

  const DataLayout &DL;
  AssumptionCache &AC;
  DominatorTree *DT;
  LoopInfo *LI;

public:
  BasicAAResult(const DataLayout &DL, const TargetLibraryInfo &TLI,
                AssumptionCache &AC, DominatorTree *DT = nullptr,
                LoopInfo *LI = nullptr)
      : AAResultBase(TLI), DL(DL), AC(AC), DT(DT), LI(LI) {}

  BasicAAResult(const BasicAAResult &Arg)
      : AAResultBase(Arg), DL(Arg.DL), AC(Arg.AC), DT(Arg.DT), LI(Arg.LI) {}
  BasicAAResult(BasicAAResult &&Arg)
      : AAResultBase(std::move(Arg)), DL(Arg.DL), AC(Arg.AC), DT(Arg.DT),
        LI(Arg.LI) {}

  /// Handle invalidation events from the new pass manager.
  ///
  /// By definition, this result is stateless and so remains valid.
  bool invalidate(Function &, const PreservedAnalyses &) { return false; }

  AliasResult alias(const MemoryLocation &LocA, const MemoryLocation &LocB);

  ModRefInfo getModRefInfo(ImmutableCallSite CS, const MemoryLocation &Loc);

  ModRefInfo getModRefInfo(ImmutableCallSite CS1, ImmutableCallSite CS2);

  /// Chases pointers until we find a (constant global) or not.
  bool pointsToConstantMemory(const MemoryLocation &Loc, bool OrLocal);

  /// Get the location associated with a pointer argument of a callsite.
  ModRefInfo getArgModRefInfo(ImmutableCallSite CS, unsigned ArgIdx);

  /// Returns the behavior when calling the given call site.
  FunctionModRefBehavior getModRefBehavior(ImmutableCallSite CS);

  /// Returns the behavior when calling the given function. For use when the
  /// call site is not known.
  FunctionModRefBehavior getModRefBehavior(const Function *F);

private:
  // A linear transformation of a Value; this class represents ZExt(SExt(V,
  // SExtBits), ZExtBits) * Scale + Offset.
  struct VariableGEPIndex {

    // An opaque Value - we can't decompose this further.
    const Value *V;

    // We need to track what extensions we've done as we consider the same Value
    // with different extensions as different variables in a GEP's linear
    // expression;
    // e.g.: if V == -1, then sext(x) != zext(x).
    unsigned ZExtBits;
    unsigned SExtBits;

    int64_t Scale;

    bool operator==(const VariableGEPIndex &Other) const {
      return V == Other.V && ZExtBits == Other.ZExtBits &&
             SExtBits == Other.SExtBits && Scale == Other.Scale;
    }

    bool operator!=(const VariableGEPIndex &Other) const {
      return !operator==(Other);
    }
  };

  /// Track alias queries to guard against recursion.
  typedef std::pair<MemoryLocation, MemoryLocation> LocPair;
  typedef SmallDenseMap<LocPair, AliasResult, 8> AliasCacheTy;
  AliasCacheTy AliasCache;

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

  static const Value *
  GetLinearExpression(const Value *V, APInt &Scale, APInt &Offset,
                      unsigned &ZExtBits, unsigned &SExtBits,
                      const DataLayout &DL, unsigned Depth, AssumptionCache *AC,
                      DominatorTree *DT, bool &NSW, bool &NUW);

  static const Value *
  DecomposeGEPExpression(const Value *V, int64_t &BaseOffs,
                         SmallVectorImpl<VariableGEPIndex> &VarIndices,
                         bool &MaxLookupReached, const DataLayout &DL,
                         AssumptionCache *AC, DominatorTree *DT);
  /// \brief A Heuristic for aliasGEP that searches for a constant offset
  /// between the variables.
  ///
  /// GetLinearExpression has some limitations, as generally zext(%x + 1)
  /// != zext(%x) + zext(1) if the arithmetic overflows. GetLinearExpression
  /// will therefore conservatively refuse to decompose these expressions.
  /// However, we know that, for all %x, zext(%x) != zext(%x + 1), even if
  /// the addition overflows.
  bool
  constantOffsetHeuristic(const SmallVectorImpl<VariableGEPIndex> &VarIndices,
                          uint64_t V1Size, uint64_t V2Size, int64_t BaseOffset,
                          AssumptionCache *AC, DominatorTree *DT);

  bool isValueEqualInPotentialCycles(const Value *V1, const Value *V2);

  void GetIndexDifference(SmallVectorImpl<VariableGEPIndex> &Dest,
                          const SmallVectorImpl<VariableGEPIndex> &Src);

  AliasResult aliasGEP(const GEPOperator *V1, uint64_t V1Size,
                       const AAMDNodes &V1AAInfo, const Value *V2,
                       uint64_t V2Size, const AAMDNodes &V2AAInfo,
                       const Value *UnderlyingV1, const Value *UnderlyingV2);

  AliasResult aliasPHI(const PHINode *PN, uint64_t PNSize,
                       const AAMDNodes &PNAAInfo, const Value *V2,
                       uint64_t V2Size, const AAMDNodes &V2AAInfo);

  AliasResult aliasSelect(const SelectInst *SI, uint64_t SISize,
                          const AAMDNodes &SIAAInfo, const Value *V2,
                          uint64_t V2Size, const AAMDNodes &V2AAInfo);

  AliasResult aliasCheck(const Value *V1, uint64_t V1Size, AAMDNodes V1AATag,
                         const Value *V2, uint64_t V2Size, AAMDNodes V2AATag);
};

/// Analysis pass providing a never-invalidated alias analysis result.
class BasicAA {
public:
  typedef BasicAAResult Result;

  /// \brief Opaque, unique identifier for this analysis pass.
  static void *ID() { return (void *)&PassID; }

  BasicAAResult run(Function &F, AnalysisManager<Function> *AM);

  /// \brief Provide access to a name for this pass for debugging purposes.
  static StringRef name() { return "BasicAliasAnalysis"; }

private:
  static char PassID;
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
}

#endif
