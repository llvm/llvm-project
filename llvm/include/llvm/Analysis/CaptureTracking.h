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
#include "llvm/Support/Compiler.h"
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
  LLVM_ABI unsigned getDefaultMaxUsesToExploreForCaptureTracking();

  /// PointerMayBeCaptured - Return true if this pointer value may be captured
  /// by the enclosing function (which is required to exist).  This routine can
  /// be expensive, so consider caching the results.  The boolean ReturnCaptures
  /// specifies whether returning the value (or part of it) from the function
  /// counts as capturing it or not.
  /// MaxUsesToExplore specifies how many uses the analysis should explore for
  /// one value before giving up due too "too many uses". If MaxUsesToExplore
  /// is zero, a default value is assumed.
  /// This function only considers captures of the passed value via its def-use
  /// chain, without considering captures of values it may be based on, or
  /// implicit captures such as for external globals.
  LLVM_ABI bool PointerMayBeCaptured(const Value *V, bool ReturnCaptures,
                                     unsigned MaxUsesToExplore = 0);

  /// Return which components of the pointer may be captured. Only consider
  /// components that are part of \p Mask. Once \p StopFn on the accumulated
  /// components returns true, the traversal is aborted early. By default, this
  /// happens when *any* of the components in \p Mask are captured.
  /// This function only considers captures of the passed value via its def-use
  /// chain, without considering captures of values it may be based on, or
  /// implicit captures such as for external globals.
  LLVM_ABI CaptureComponents PointerMayBeCaptured(
      const Value *V, bool ReturnCaptures, CaptureComponents Mask,
      function_ref<bool(CaptureComponents)> StopFn = capturesAnything,
      unsigned MaxUsesToExplore = 0);

  /// PointerMayBeCapturedBefore - Return true if this pointer value may be
  /// captured by the enclosing function (which is required to exist). If a
  /// DominatorTree is provided, only captures which happen before the given
  /// instruction are considered. This routine can be expensive, so consider
  /// caching the results.  The boolean ReturnCaptures specifies whether
  /// returning the value (or part of it) from the function counts as capturing
  /// it or not. Captures by the provided instruction are considered if the
  /// final parameter is true.
  /// MaxUsesToExplore specifies how many uses the analysis should explore for
  /// one value before giving up due too "too many uses". If MaxUsesToExplore
  /// is zero, a default value is assumed.
  /// This function only considers captures of the passed value via its def-use
  /// chain, without considering captures of values it may be based on, or
  /// implicit captures such as for external globals.
  LLVM_ABI bool PointerMayBeCapturedBefore(const Value *V, bool ReturnCaptures,
                                           const Instruction *I,
                                           const DominatorTree *DT,
                                           bool IncludeI = false,
                                           unsigned MaxUsesToExplore = 0,
                                           const LoopInfo *LI = nullptr);

  /// Return which components of the pointer may be captured on the path to
  /// \p I. Only consider components that are part of \p Mask. Once \p StopFn
  /// on the accumulated components returns true, the traversal is aborted
  /// early. By default, this happens when *any* of the components in \p Mask
  /// are captured.
  /// This function only considers captures of the passed value via its def-use
  /// chain, without considering captures of values it may be based on, or
  /// implicit captures such as for external globals.
  LLVM_ABI CaptureComponents PointerMayBeCapturedBefore(
      const Value *V, bool ReturnCaptures, const Instruction *I,
      const DominatorTree *DT, bool IncludeI, CaptureComponents Mask,
      function_ref<bool(CaptureComponents)> StopFn = capturesAnything,
      const LoopInfo *LI = nullptr, unsigned MaxUsesToExplore = 0);

  // Returns the 'earliest' instruction that captures \p V in \F, and which
  // components may be captured (by any use, not necessarily the earliest one).
  // An instruction A is considered earlier than instruction B, if A dominates
  // B. If 2 escapes do not dominate each other, the terminator of the common
  // dominator is chosen. If not all uses can be analyzed, the earliest escape
  // is set to the first instruction in the function entry block. If \p V does
  // not escape, nullptr is returned. Note that the caller of the function has
  // to ensure that the instruction the result value is compared against is
  // not in a cycle.
  //
  // Only consider components that are part of \p Mask.
  LLVM_ABI std::pair<Instruction *, CaptureComponents>
  FindEarliestCapture(const Value *V, Function &F, bool ReturnCaptures,
                      const DominatorTree &DT, CaptureComponents Mask,
                      unsigned MaxUsesToExplore = 0);

  /// Capture information for a specific Use.
  struct UseCaptureInfo {
    /// Components captured by this use.
    CaptureComponents UseCC;
    /// Components captured by the return value of the user of this Use.
    CaptureComponents ResultCC;

    UseCaptureInfo(CaptureComponents UseCC,
                   CaptureComponents ResultCC = CaptureComponents::None)
        : UseCC(UseCC), ResultCC(ResultCC) {}

    static UseCaptureInfo passthrough() {
      return UseCaptureInfo(CaptureComponents::None, CaptureComponents::All);
    }

    bool isPassthrough() const {
      return capturesNothing(UseCC) && capturesAnything(ResultCC);
    }

    operator CaptureComponents() const { return UseCC | ResultCC; }
  };

  /// This callback is used in conjunction with PointerMayBeCaptured. In
  /// addition to the interface here, you'll need to provide your own getters
  /// to see whether anything was captured.
  struct LLVM_ABI CaptureTracker {
    /// Action returned from captures().
    enum Action {
      /// Stop the traversal.
      Stop,
      /// Continue traversal, and also follow the return value of the user if
      /// it has additional capture components (that is, if it has capture
      /// components in Ret that are not part of Other).
      Continue,
      /// Continue traversal, but do not follow the return value of the user,
      /// even if it has additional capture components. Should only be used if
      /// captures() has already taken the potential return captures into
      /// account.
      ContinueIgnoringReturn,
    };

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

    /// Use U directly captures CI.UseCC and additionally CI.ResultCC
    /// through the return value of the user of U.
    ///
    /// Return one of Stop, Continue or ContinueIgnoringReturn to control
    /// further traversal.
    virtual Action captured(const Use *U, UseCaptureInfo CI) = 0;
  };

  /// Determine what kind of capture behaviour \p U may exhibit.
  ///
  /// The returned UseCaptureInfo contains the components captured directly
  /// by the use (UseCC) and the components captured through the return value
  /// of the user (ResultCC).
  ///
  /// \p Base is the starting value of the capture analysis, which is
  /// relevant for address_is_null captures.
  LLVM_ABI UseCaptureInfo DetermineUseCaptureKind(const Use &U,
                                                  const Value *Base);

  /// PointerMayBeCaptured - Visit the value and the values derived from it and
  /// find values which appear to be capturing the pointer value. This feeds
  /// results into and is controlled by the CaptureTracker object.
  /// MaxUsesToExplore specifies how many uses the analysis should explore for
  /// one value before giving up due too "too many uses". If MaxUsesToExplore
  /// is zero, a default value is assumed.
  /// This function only considers captures of the passed value via its def-use
  /// chain, without considering captures of values it may be based on, or
  /// implicit captures such as for external globals.
  LLVM_ABI void PointerMayBeCaptured(const Value *V, CaptureTracker *Tracker,
                                     unsigned MaxUsesToExplore = 0);
} // end namespace llvm

#endif
