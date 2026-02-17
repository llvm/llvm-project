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
#include "llvm/ADT/DenseMap.h"
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

  // VPlan construction support: Hold a mapping from ingredients to
  // their recipe.
  DenseMap<Instruction *, VPRecipeBase *> Ingredient2Recipe;

  /// Cross-iteration reduction & first-order recurrence phis for which we need
  /// to add the incoming value from the backedge after all recipes have been
  /// created.
  SmallVector<VPHeaderPHIRecipe *, 4> PhisToFix;

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

  VPBuilder &getVPBuilder() const { return Builder; }

  /// Create and return a widened recipe for a non-phi recipe \p R if one can be
  /// created within the given VF \p Range.
  VPRecipeBase *tryToCreateWidenNonPhiRecipe(VPSingleDefRecipe *R,
                                             VFRange &Range);

  /// Check if the load or store instruction \p VPI should widened for \p
  /// Range.Start and potentially masked. Such instructions are handled by a
  /// recipe that takes an additional VPInstruction for the mask.
  VPRecipeBase *tryToWidenMemory(VPInstruction *VPI, VFRange &Range);

  /// Makes Histogram count operations safe for vectorization, by emitting a
  /// llvm.experimental.vector.histogram.add intrinsic in place of the
  /// Load + Add|Sub + Store operations that perform the histogram in the
  /// original scalar loop.
  VPHistogramRecipe *tryToWidenHistogram(const HistogramInfo *HI,
                                         VPInstruction *VPI);

  /// Set the recipe created for given ingredient.
  void setRecipe(Instruction *I, VPRecipeBase *R) {
    assert(!Ingredient2Recipe.contains(I) &&
           "Cannot reset recipe for instruction.");
    Ingredient2Recipe[I] = R;
  }

  /// Return the recipe created for given ingredient.
  VPRecipeBase *getRecipe(Instruction *I) {
    assert(Ingredient2Recipe.count(I) &&
           "Recording this ingredients recipe was not requested");
    assert(Ingredient2Recipe[I] != nullptr &&
           "Ingredient doesn't have a recipe");
    return Ingredient2Recipe[I];
  }

  /// Build a VPReplicationRecipe for \p VPI. If it is predicated, add the mask
  /// as last operand. Range.End may be decreased to ensure same recipe behavior
  /// from \p Range.Start to \p Range.End.
  VPReplicateRecipe *handleReplication(VPInstruction *VPI, VFRange &Range);

  VPValue *getVPValueOrAddLiveIn(Value *V) {
    if (auto *I = dyn_cast<Instruction>(V)) {
      if (auto *R = Ingredient2Recipe.lookup(I))
        return R->getVPSingleValue();
    }
    return Plan.getOrAddLiveIn(V);
  }
};
} // end namespace llvm

#endif // LLVM_TRANSFORMS_VECTORIZE_VPRECIPEBUILDER_H
