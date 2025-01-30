//===----- llvm/Analysis/CaptureTracking.h - Pointer capture ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains routines that help determine which pointers are captured.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_CAPTURETRACKING_H
#define LLVM_ANALYSIS_CAPTURETRACKING_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/ModRef.h"

namespace llvm {

  class Value;
  class Use;
  class CaptureInfo;
  class DataLayout;
  class Instruction;
  class DominatorTree;
  class LoopInfo;
  class Function;
  template <typename Fn> class function_ref;

  /// getDefaultMaxUsesToExploreForCaptureTracking - Return default value of
  /// the maximal number of uses to explore before giving up. It is used by
  /// PointerMayBeCaptured family analysis.
  unsigned getDefaultMaxUsesToExploreForCaptureTracking();

  /// PointerMayBeCaptured - Return true if this pointer value may be captured
  /// by the enclosing function (which is required to exist).  This routine can
  /// be expensive, so consider caching the results.  The boolean ReturnCaptures
  /// specifies whether returning the value (or part of it) from the function
  /// counts as capturing it or not.  The boolean StoreCaptures specified
  /// whether storing the value (or part of it) into memory anywhere
  /// automatically counts as capturing it or not.
  /// MaxUsesToExplore specifies how many uses the analysis should explore for
  /// one value before giving up due too "too many uses". If MaxUsesToExplore
  /// is zero, a default value is assumed.
  bool PointerMayBeCaptured(const Value *V, bool ReturnCaptures,
                            bool StoreCaptures, unsigned MaxUsesToExplore = 0);

  /// PointerMayBeCapturedBefore - Return true if this pointer value may be
  /// captured by the enclosing function (which is required to exist). If a
  /// DominatorTree is provided, only captures which happen before the given
  /// instruction are considered. This routine can be expensive, so consider
  /// caching the results.  The boolean ReturnCaptures specifies whether
  /// returning the value (or part of it) from the function counts as capturing
  /// it or not.  The boolean StoreCaptures specified whether storing the value
  /// (or part of it) into memory anywhere automatically counts as capturing it
  /// or not. Captures by the provided instruction are considered if the
  /// final parameter is true.
  /// MaxUsesToExplore specifies how many uses the analysis should explore for
  /// one value before giving up due too "too many uses". If MaxUsesToExplore
  /// is zero, a default value is assumed.
  bool PointerMayBeCapturedBefore(const Value *V, bool ReturnCaptures,
                                  bool StoreCaptures, const Instruction *I,
                                  const DominatorTree *DT,
                                  bool IncludeI = false,
                                  unsigned MaxUsesToExplore = 0,
                                  const LoopInfo *LI = nullptr);

  // Returns the 'earliest' instruction that captures \p V in \F. An instruction
  // A is considered earlier than instruction B, if A dominates B. If 2 escapes
  // do not dominate each other, the terminator of the common dominator is
  // chosen. If not all uses can be analyzed, the earliest escape is set to
  // the first instruction in the function entry block. If \p V does not escape,
  // nullptr is returned. Note that the caller of the function has to ensure
  // that the instruction the result value is compared against is not in a
  // cycle.
  Instruction *FindEarliestCapture(const Value *V, Function &F,
                                   bool ReturnCaptures, bool StoreCaptures,
                                   const DominatorTree &DT,
                                   unsigned MaxUsesToExplore = 0);

  /// This callback is used in conjunction with PointerMayBeCaptured. In
  /// addition to the interface here, you'll need to provide your own getters
  /// to see whether anything was captured.
  struct CaptureTracker {
    virtual ~CaptureTracker();

    /// tooManyUses - The depth of traversal has breached a limit. There may be
    /// capturing instructions that will not be passed into captured().
    virtual void tooManyUses() = 0;

    /// shouldExplore - This is the use of a value derived from the pointer.
    /// To prune the search (ie., assume that none of its users could possibly
    /// capture) return false. To search it, return true.
    ///
    /// U->getUser() is always an Instruction.
    virtual bool shouldExplore(const Use *U);

    /// When returned from captures(), stop the traversal.
    static std::optional<CaptureComponents> stop() { return std::nullopt; }

    /// When returned from captures(), continue traversal, but do not follow
    /// the return value of this user, even if it has additional capture
    /// components. Should only be used if captures() has already taken the
    /// potential return caputres into account.
    static std::optional<CaptureComponents> continueIgnoringReturn() {
      return CaptureComponents::None;
    }

    /// When returned from captures(), continue traversal, and also follow
    /// the return value of this user if it has additional capture components
    /// (that is, capture components in Ret that are not part of Other).
    static std::optional<CaptureComponents> continueDefault(CaptureInfo CI) {
      CaptureComponents RetCC = CI.getRetComponents();
      if (!capturesNothing(RetCC & ~CI.getOtherComponents()))
        return RetCC;
      return CaptureComponents::None;
    }

    /// Use U directly captures CI.getOtherComponents() and additionally
    /// CI.getRetComponents() through the return value of the user of U.
    ///
    /// Return std::nullopt to stop the traversal, or the CaptureComponents to
    /// follow via the return value, which must be a subset of
    /// CI.getRetComponents().
    ///
    /// For convenience, prefer returning one of stop(), continueDefault(CI) or
    /// continueIgnoringReturn().
    virtual std::optional<CaptureComponents> captured(const Use *U,
                                                      CaptureInfo CI) = 0;

    /// isDereferenceableOrNull - Overload to allow clients with additional
    /// knowledge about pointer dereferenceability to provide it and thereby
    /// avoid conservative responses when a pointer is compared to null.
    virtual bool isDereferenceableOrNull(Value *O, const DataLayout &DL);
  };

  /// Determine what kind of capture behaviour \p U may exhibit.
  ///
  /// The Other part of the returned CaptureInfo indicates which component of
  /// the pointer may be captured directly by the use. The Ret part indicates
  /// which components may be captured by following uses of the user of \p U.
  /// The \p IsDereferenceableOrNull callback is used to rule out capturing for
  /// certain comparisons.
  CaptureInfo
  DetermineUseCaptureKind(const Use &U,
                          llvm::function_ref<bool(Value *, const DataLayout &)>
                              IsDereferenceableOrNull);

  /// PointerMayBeCaptured - Visit the value and the values derived from it and
  /// find values which appear to be capturing the pointer value. This feeds
  /// results into and is controlled by the CaptureTracker object.
  /// MaxUsesToExplore specifies how many uses the analysis should explore for
  /// one value before giving up due too "too many uses". If MaxUsesToExplore
  /// is zero, a default value is assumed.
  void PointerMayBeCaptured(const Value *V, CaptureTracker *Tracker,
                            unsigned MaxUsesToExplore = 0);

  /// Returns true if the pointer is to a function-local object that never
  /// escapes from the function.
  bool isNonEscapingLocalObject(
      const Value *V,
      SmallDenseMap<const Value *, bool, 8> *IsCapturedCache = nullptr);
} // end namespace llvm

#endif
