//===- EscapeAnalysis.h - Intraprocedural Escape Analysis -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the interface for a simple, conservative intraprocedural
// escape analysis. It is designed as a helper utility for other passes, like
// ThreadSanitizer, to determine if an allocation escapes the context of its
// containing function.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_INSTRUMENTATION_ESCAPEANALYSIS_H
#define LLVM_TRANSFORMS_INSTRUMENTATION_ESCAPEANALYSIS_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/CaptureTracking.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/PassManager.h"

namespace llvm {
/// Find underlying base objects for a pointer possibly produced by loads.
///
/// This routine walks backwards through MemorySSA clobbering definitions of
/// simple loads to find stores that defined the loaded pointer values, and
/// collects their base objects. Additionally, it attempts ValueTracking
/// `getUnderlyingObjects` to peel pointer casts/GEPs/phis where profitable.
///
/// Collected "base" objects are:
///  - `AllocaInst` (stack, base=stack)
///  - `Argument` (function argument, base=arg)
///  - `GlobalVariable` and `GlobalAlias` (base=global|alias)
///  - `ConstantPointerNull` (base=null)
///  - Results of known heap-allocating calls (e.g. `malloc`, `calloc`,
///    `realloc`, `aligned_alloc`, `strdup`, or C\+\+ `new`) when recognized
///    via `TargetLibraryInfo` (base=heap).
///
/// If the walk encounters an unrecognized defining write, a non-simple store,
/// a memintrinsic as a defining write, or the step budget is exceeded, the
/// analysis conservatively treats the current value as a terminal non-base
/// and marks the result as incomplete.
///
/// Contract and guarantees:
///  - If `MSSA` is null, the analysis immediately returns with
///    `*IsComplete == false` (if provided).
///  - If `TLI` is null, heap allocations cannot be recognized; terminals that
///    are calls are treated as non-bases and lead to `*IsComplete == false`.
///  - `Result` is a set of terminal values observed (may include non-bases if
///    the analysis is incomplete). Use `*IsComplete` to know if all are bases.
///  - `MaxSteps` is a per-query safety valve limiting the combined number of
///    processed worklist nodes. When exceeded, the analysis bails out and
///    sets `*IsComplete == false`.
void getUnderlyingObjectsThroughLoads(const Value *Ptr, MemorySSA *MSSA,
                                      SmallPtrSetImpl<const Value *> &Result,
                                      const TargetLibraryInfo *TLI = nullptr,
                                      LoopInfo *LI = nullptr,
                                      bool *IsComplete = nullptr,
                                      unsigned MaxSteps = 10000);

/// Detect heap allocations. Complements isAllocationFn() by checking
/// library functions directly when attributes might be missing.
bool isHeapAllocation(const CallBase *CB, const TargetLibraryInfo &TLI);

/// EscapeAnalysisInfo - This class implements the actual backward dataflow
/// analysis for a function; queries are per allocation site.
///
/// This is a lightweight, intraprocedural and conservative analysis intended
/// to help instrumentation passes (e.g. ThreadSanitizer) skip objects that do
/// not escape the function scope. The main query is \c isEscaping(Value&),
/// which answers whether an allocation site (alloca/malloc-like) may escape
/// the current function. Results are memoized per underlying object.
struct EscapeAnalysisInfo {
  /// Constructs an escape analysis utility for a given function.
  /// Requires a FunctionAnalysisManager to obtain other analyses like AA.
  EscapeAnalysisInfo(Function &F, FunctionAnalysisManager &FAM) : F(F) {
    TLI = &FAM.getResult<TargetLibraryAnalysis>(F);
    MSSA = &FAM.getResult<MemorySSAAnalysis>(F).getMSSA();
    LI = &FAM.getResult<LoopAnalysis>(F);
  };
  ~EscapeAnalysisInfo() = default;

  /// Return true if \p Alloc may escape the function.
  /// \param Alloc - Must be an allocation site (AllocaInst or heap allocation
  ///                call). Passing GEPs/bitcasts is not supported; use the base
  ///                allocation.
  /// \returns true if the allocation escapes or if \p Alloc is not an
  /// allocation site.
  bool isEscaping(const Value &Alloc);

  /// Print escape information for all allocations in the function
  void print(raw_ostream &OS);

  bool invalidate(Function &F, const PreservedAnalyses &PA,
                  FunctionAnalysisManager::Invalidator &Inv);

private:
  Function &F;
  DenseMap<const Value *, bool> Cache;

  TargetLibraryInfo *TLI = nullptr;
  MemorySSA *MSSA = nullptr;
  LoopInfo *LI = nullptr;

  /// Checks whether a base location is externally visible (thus escapes).
  static bool isExternalObject(const Value *Base);

  /// Custom CaptureTracker for escape analysis
  class EscapeCaptureTracker : public CaptureTracker {
  public:
    EscapeCaptureTracker(EscapeAnalysisInfo &EAI,
                         const SmallPtrSet<const Value *, 32> &ProcessingSet)
        : EAI(EAI), ProcessingSet(ProcessingSet) {}

    void tooManyUses() override { Escaped = true; }
    bool shouldExplore(const Use *U) override;
    Action captured(const Use *U, UseCaptureInfo CI) override;
    bool hasEscaped() const { return Escaped; }

  private:
    EscapeAnalysisInfo &EAI;
    SmallPtrSet<const Value *, 32> ProcessingSet;
    bool Escaped = false;

    /// Analyze if storing to destination causes escape
    bool doesStoreDestEscape(const Value *Dest);

    /// Get indices of pointer-typed arguments that are marked 'nocapture'
    SmallVector<unsigned, 8>
    getNoCapturePointerArgIndices(const CallBase *CB) const;

    /// Check if any of the 'nocapture' arguments can reach the query object
    bool canEscapeViaNocaptureArgs(
        const CallBase &CB, ArrayRef<unsigned> NoCapPtrArgs,
        SmallPtrSetImpl<const Value *> &StorePtrOpndBases) const;

    /// Check if the given clobber stems from StartMDef
    bool stemsFromStartStore(MemoryUseOrDef *MUOD, const MemoryDef *StartMDef,
                             MemoryLocation Loc, bool &IsComplete,
                             MemorySSAWalker *Walker) const;

    /// Walk MemorySSA forward from StartStore and:
    ///  - collect pointer-typed Loads that may read bytes written by StartStore
    ///  - detect calls that may export those bytes via nocapture pointer args
    /// Sets ContentMayEscape if any call may export the bytes.
    SmallVector<const LoadInst *, 32>
    findStoreReadersAndExports(const StoreInst *StartStore,
                               bool &ContentMayEscape, bool &IsComplete);

    /// Analyze whether the pointer value stored by `Store` can escape
    bool doesStoredPointerEscapeViaLoads(const StoreInst *Store);
  };

  /// Solve escape for a single allocation site using backward dataflow.
  bool solveEscapeFor(const Value &Ptr,
                      SmallPtrSet<const Value *, 32> &ProcessingSet);

  /// Helper function to detect allocation sites (malloc/new-like)
  /// Returns true if V is an Alloca or a call to a known heap alloc function.
  bool isAllocationSite(const Value *V);
};

/// EscapeAnalysisInfo wrapper for the new pass manager.
class EscapeAnalysis : public AnalysisInfoMixin<EscapeAnalysis> {
  friend AnalysisInfoMixin<EscapeAnalysis>;
  static AnalysisKey Key;

public:
  using Result = EscapeAnalysisInfo;
  static Result run(Function &F, FunctionAnalysisManager &FAM);
};

/// Printer pass for the \c EscapeAnalysis results.
class EscapeAnalysisPrinterPass
    : public PassInfoMixin<EscapeAnalysisPrinterPass> {
  raw_ostream &OS;

public:
  explicit EscapeAnalysisPrinterPass(raw_ostream &OS) : OS(OS) {}
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM) const;
  static bool isRequired() { return true; }
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_INSTRUMENTATION_ESCAPEANALYSIS_H