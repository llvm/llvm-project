//===- VPlanCrossPartCSE.h - Cross-part CSE for VPlan -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares prediction-only profitability analysis for exact load
// overlap across two modeled logical VPlan parts. It does not transform VPlan.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_VECTORIZE_VPLANCROSSPARTCSE_H
#define LLVM_TRANSFORMS_VECTORIZE_VPLANCROSSPARTCSE_H

#include "VPlanHelpers.h"
#include "llvm/Support/InstructionCost.h"
#include "llvm/Support/TypeSize.h"
#include <limits>

namespace llvm {

class Loop;
class PredicatedScalarEvolution;
class VPlan;

/// The interleave count and logical unroll factor modeled by the analysis.
constexpr unsigned CrossPartCSERequiredInterleaveCount = 2;

/// Profitability criteria supplied by the caller.
///
/// Fail-closed defaults require callers to provide both criteria explicitly.
struct CrossPartCSEOptions {
  /// Sentinel used until the caller supplies an explicit policy value.
  static constexpr unsigned Unspecified = std::numeric_limits<unsigned>::max();

  /// Minimum saving; the default rejects analysis until policy supplies it.
  unsigned MinSavingPct = Unspecified;
  /// Minimum opportunities; the default likewise keeps the API fail-closed.
  unsigned MinOpportunities = Unspecified;
};

/// Return whether predicted exact load overlap is profitable for \p Plan and
/// \p VF under \p Options.
///
/// The caller must establish that interleaving \p Plan is legal before using
/// this profitability result to raise its interleave count. The analysis reads
/// \p Plan but takes a non-const reference because the VPlan query APIs it uses
/// are not const-qualified.
///
/// \p RecipeCosts is borrowed for this call and is neither copied nor retained.
bool isCrossPartCSEProfitable(VPlan &Plan, ElementCount VF,
                              InstructionCost LoopCost, const Loop *OrigLoop,
                              PredicatedScalarEvolution &PSE,
                              const VPRecipeCostMap &RecipeCosts,
                              const CrossPartCSEOptions &Options);

} // namespace llvm

#endif // LLVM_TRANSFORMS_VECTORIZE_VPLANCROSSPARTCSE_H
