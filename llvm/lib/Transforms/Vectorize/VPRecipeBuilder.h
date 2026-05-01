//===- VPRecipeBuilder.h - Helper class to build recipes --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_VECTORIZE_VPRECIPEBUILDER_H
#define LLVM_TRANSFORMS_VECTORIZE_VPRECIPEBUILDER_H

#include "LoopVectorizationPlanner.h"
#include "VPlan.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"

namespace llvm {

class LoopVectorizationLegality;
class LoopVectorizationCostModel;
class TargetLibraryInfo;
struct HistogramInfo;
struct VFRange;

/// Helper class to create VPRecipies from IR instructions.
class VPRecipeBuilder {
  /// The VPlan new recipes are added to.
  VPlan &Plan;

  /// Target Library Info.
  const TargetLibraryInfo *TLI;

  /// The legality analysis.
  LoopVectorizationLegality *Legal;

  /// The profitablity analysis.
  LoopVectorizationCostModel &CM;

  VPBuilder &Builder;

  /// Check if \p I can be widened at the start of \p Range and possibly
  /// decrease the range such that the returned value holds for the entire \p
  /// Range. The function should not be called for memory instructions or calls.
  bool shouldWiden(Instruction *I, VFRange &Range) const;

  /// Optimize the special case where the operand of \p VPI is a constant
  /// integer induction variable.
  VPWidenIntOrFpInductionRecipe *
  tryToOptimizeInductionTruncate(VPInstruction *VPI, VFRange &Range);

  /// Handle call instructions. If \p VPI can be widened for \p Range.Start,
  /// return a new VPWidenCallRecipe or VPWidenIntrinsicRecipe. Range.End may be
  /// decreased to ensure same decision from \p Range.Start to \p Range.End.
  VPSingleDefRecipe *tryToWidenCall(VPInstruction *VPI, VFRange &Range);

  /// Check if \p VPI has an opcode that can be widened and return a
  /// VPWidenRecipe if it can. The function should only be called if the
  /// cost-model indicates that widening should be performed.
  VPWidenRecipe *tryToWiden(VPInstruction *VPI);

public:
  VPRecipeBuilder(VPlan &Plan, const TargetLibraryInfo *TLI,
                  LoopVectorizationLegality *Legal,
                  LoopVectorizationCostModel &CM, VPBuilder &Builder)
      : Plan(Plan), TLI(TLI), Legal(Legal), CM(CM), Builder(Builder) {}

  /// Create and return a widened recipe for a non-phi recipe \p R if one can be
  /// created within the given VF \p Range.
  VPRecipeBase *tryToCreateWidenNonPhiRecipe(VPSingleDefRecipe *R,
                                             VFRange &Range);

  /// Check if the load or store instruction \p VPI should widened for \p
  /// Range.Start and potentially masked. Such instructions are handled by a
  /// recipe that takes an additional VPInstruction for the mask.
  VPRecipeBase *tryToWidenMemory(VPInstruction *VPI, VFRange &Range);

  /// If \p VPI represents a histogram operation (as determined by
  /// LoopVectorizationLegality) make that safe for vectorization, by emitting a
  /// llvm.experimental.vector.histogram.add intrinsic in place of the Load +
  /// Add|Sub + Store operations that perform the histogram in the original
  /// scalar loop.
  VPHistogramRecipe *widenIfHistogram(VPInstruction *VPI);

  /// If \p VPI is a store of a reduction into an invariant address, delete it.
  /// If it is the final store of a reduction result, a uniform store recipe
  /// will be created for it in the middle block. Returns `true` if replacement
  /// took place. The order of stores must be preserved, hence \p
  /// FinalRedStoresBuidler.
  bool replaceWithFinalIfReductionStore(VPInstruction *VPI,
                                        VPBuilder &FinalRedStoresBuilder);

  /// Build a VPReplicationRecipe for \p VPI. If it is predicated, add the mask
  /// as last operand. Range.End may be decreased to ensure same recipe behavior
  /// from \p Range.Start to \p Range.End.
  VPReplicateRecipe *handleReplication(VPInstruction *VPI, VFRange &Range);
};
} // end namespace llvm

#endif // LLVM_TRANSFORMS_VECTORIZE_VPRECIPEBUILDER_H
