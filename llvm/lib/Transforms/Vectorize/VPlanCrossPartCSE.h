//===- VPlanCrossPartCSE.h - Cross-part CSE for VPlan -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares prediction-only profitability analysis for exact load
// redundancy across two modeled logical VPlan parts. It does not transform
// VPlan or guarantee that a later pass will eliminate the redundant load.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_VECTORIZE_VPLANCROSSPARTCSE_H
#define LLVM_TRANSFORMS_VECTORIZE_VPLANCROSSPARTCSE_H

#include "llvm/Support/InstructionCost.h"
#include "llvm/Support/TypeSize.h"
#include <limits>

namespace llvm {

class VPlan;
struct VPCostContext;

/// The interleave count and logical unroll factor modeled by the analysis.
constexpr unsigned CrossPartCSERequiredInterleaveCount = 2;

/// Profitability criterion supplied by the caller.
///
/// The fail-closed default requires the caller to provide an explicit value.
struct CrossPartCSEOptions {
  /// Sentinel used until the caller supplies an explicit policy value.
  static constexpr unsigned Unspecified = std::numeric_limits<unsigned>::max();

  /// Minimum saving; the default rejects analysis until policy supplies it.
  unsigned MinSavingPct = Unspecified;
};

/// Return whether exact cross-part load redundancy in \p Plan at \p VF meets
/// \p Options.
///
/// The caller must establish that interleaving \p Plan is legal before using
/// this opportunity estimate to raise its interleave count. A positive result
/// does not guarantee that a later pass will eliminate the redundant load.
/// The analysis reads \p Plan but takes a non-const reference because the VPlan
/// query APIs it uses are not const-qualified.
///
/// \p CostCtx is local to the interleave decision and is not retained.
bool isCrossPartCSEProfitable(VPlan &Plan, ElementCount VF,
                              InstructionCost LoopCost, VPCostContext &CostCtx,
                              const CrossPartCSEOptions &Options);

} // namespace llvm

#endif // LLVM_TRANSFORMS_VECTORIZE_VPLANCROSSPARTCSE_H
