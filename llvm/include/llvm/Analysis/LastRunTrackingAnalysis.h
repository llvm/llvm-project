//===- LastRunTrackingAnalysis.h - Avoid running redundant pass -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is an analysis pass to track a set of passes that have been run, so that
// we can avoid running a pass again if there is no change since the last run of
// the pass.
//
// In this analysis we track a set of passes S for each function with the
// following transition rules:
//   1. If pass P makes changes, set S = {P}.
//   2. If pass P doesn't make changes, set S = S + {P}.
//
// Before running a pass P which satisfies P(P(x)) == P(x), we check if P is in
// S. If so, we skip this pass since we know that there will be no change.
//
// Notes:
//   1. Some transform passes have parameters that may vary in the optimization
//   pipeline. We should check if parameters in current run is compatible with
//   that in the last run.
//   2. This pass only tracks at the module/function level. Loop passes are not
//   supported for now.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_LASTRUNTRACKINGANALYSIS_H
#define LLVM_ANALYSIS_LASTRUNTRACKINGANALYSIS_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/PassManager.h"
#include <functional>

namespace llvm {

/// This class is used to track the last run of a set of module/function passes.
/// Invalidation are conservatively handled by the pass manager if a pass
/// doesn't explicitly preserve the result.
/// If we want to skip a pass, we should define a unique ID \p PassID to
/// identify the pass, which is usually a pointer to a static member. If a pass
/// has parameters, they should be stored in a struct \p OptionT with a method
/// bool isCompatibleWith(const OptionT& LastOpt) const to check compatibility.
class LastRunTrackingInfo {
public:
  using PassID = const void *;
  using OptionPtr = const void *;
  // CompatibilityCheckFn is a closure that stores the parameters of last run.
  using CompatibilityCheckFn = std::function<bool(OptionPtr)>;

  /// Check if we should skip a pass.
  /// \param ID The unique ID of the pass.
  /// \param Opt The parameters of the pass. If the pass has no parameters, use
  /// shouldSkip(PassID ID) instead.
  /// \return True if we should skip the pass.
  /// \sa shouldSkip(PassID ID)
  template <typename OptionT>
  bool shouldSkip(PassID ID, const OptionT &Opt) const {
    return shouldSkipImpl(ID, &Opt);
  }
  bool shouldSkip(PassID ID) const { return shouldSkipImpl(ID, nullptr); }

  /// Update the tracking info.
  /// \param ID The unique ID of the pass.
  /// \param Changed Whether the pass makes changes.
  /// \param Opt The parameters of the pass. It must have the same type as the
  /// parameters of the last run. If the pass has no parameters, use
  /// update(PassID ID, bool Changed) instead.
  /// \sa update(PassID ID, bool Changed)
  template <typename OptionT>
  void update(PassID ID, bool Changed, const OptionT &Opt) {
    updateImpl(ID, Changed, [Opt](OptionPtr Ptr) {
      return static_cast<const OptionT *>(Ptr)->isCompatibleWith(Opt);
    });
  }
  void update(PassID ID, bool Changed) {
    updateImpl(ID, Changed, CompatibilityCheckFn{});
  }

private:
  bool shouldSkipImpl(PassID ID, OptionPtr Ptr) const;
  void updateImpl(PassID ID, bool Changed, CompatibilityCheckFn CheckFn);

  DenseMap<PassID, CompatibilityCheckFn> TrackedPasses;
};

/// A function/module analysis which provides an empty \c LastRunTrackingInfo.
class LastRunTrackingAnalysis final
    : public AnalysisInfoMixin<LastRunTrackingAnalysis> {
  friend AnalysisInfoMixin<LastRunTrackingAnalysis>;
  static AnalysisKey Key;

public:
  using Result = LastRunTrackingInfo;
  LastRunTrackingInfo run(Function &F, FunctionAnalysisManager &) {
    return LastRunTrackingInfo();
  }
  LastRunTrackingInfo run(Module &M, ModuleAnalysisManager &) {
    return LastRunTrackingInfo();
  }
};

} // namespace llvm

#endif // LLVM_ANALYSIS_LASTRUNTRACKINGANALYSIS_H
