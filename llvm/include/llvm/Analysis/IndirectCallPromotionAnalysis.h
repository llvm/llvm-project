//===- IndirectCallPromotionAnalysis.h - Indirect call analysis -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// Interface to identify indirect call promotion candidates.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_INDIRECTCALLPROMOTIONANALYSIS_H
#define LLVM_ANALYSIS_INDIRECTCALLPROMOTIONANALYSIS_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ProfileData/InstrProf.h"

namespace llvm {

class CallBase;
class Function;
class Instruction;

/// Find all possible function targets of an indirect call whose called operand
/// is formed entirely from selects, phis, and function constants. Returns
/// false if the target set is not exhaustive, exceeds the configured target
/// limit, or requires more than the configured traversal budget.
/// \p Targets is replaced with the discovered targets on success and is empty
/// on failure.
LLVM_ABI bool
getStaticIndirectCallTargets(const CallBase &CB,
                             SmallVectorImpl<Function *> &Targets);

// Class for identifying profitable indirect call promotion candidates when
// the indirect-call value profile metadata is available.
class ICallPromotionAnalysis {
private:
  // Allocate space to read the profile annotation.
  SmallVector<InstrProfValueData, 4> ValueDataArray;

  // Count is the call count for the direct-call target.
  // TotalCount is the total call count for the indirect-call callsite.
  // RemainingCount is the TotalCount minus promoted-direct-call count.
  // Return true we should promote this indirect-call target.
  bool isPromotionProfitable(uint64_t Count, uint64_t TotalCount,
                             uint64_t RemainingCount);

  // Returns the number of profitable candidates to promote for the
  // current ValueDataArray and the given \p Inst.
  uint32_t getProfitablePromotionCandidates(const Instruction *Inst,
                                            uint64_t TotalCount);

  // Noncopyable
  ICallPromotionAnalysis(const ICallPromotionAnalysis &other) = delete;
  ICallPromotionAnalysis &
  operator=(const ICallPromotionAnalysis &other) = delete;

public:
  ICallPromotionAnalysis() = default;

  /// Returns reference to array of InstrProfValueData for the given
  /// instruction \p I.
  ///
  /// The \p TotalCount and \p NumCandidates are set to the the total profile
  /// count of the indirect call \p I and the number of profitable candidates
  /// in the given array (which is sorted in reverse order of profitability).
  /// The value of \p MaxNumValueData can be used to override the max set
  /// from the -icp-max-prom option with a larger value.
  ///
  /// The returned array space is owned by this class, and overwritten on
  /// subsequent calls.
  LLVM_ABI MutableArrayRef<InstrProfValueData>
  getPromotionCandidatesForInstruction(const Instruction *I,
                                       uint64_t &TotalCount,
                                       uint32_t &NumCandidates,
                                       unsigned MaxNumValueData = 0);
};

} // end namespace llvm

#endif
