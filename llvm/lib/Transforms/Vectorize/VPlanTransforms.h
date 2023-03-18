//===- VPlanTransforms.h - Utility VPlan to VPlan transforms --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file provides utility VPlan to VPlan transformations.
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_VECTORIZE_VPLANTRANSFORMS_H
#define LLVM_TRANSFORMS_VECTORIZE_VPLANTRANSFORMS_H

#include "VPlan.h"
#include "llvm/ADT/STLFunctionalExtras.h"

namespace llvm {

class InductionDescriptor;
class Instruction;
class PHINode;
class ScalarEvolution;
class Loop;
class PredicatedScalarEvolution;
class TargetLibraryInfo;
class VPBuilder;
class VPRecipeBuilder;

struct VPlanTransforms {
  /// Replaces the VPInstructions in \p Plan with corresponding
  /// widen recipes.
  static void
  VPInstructionsToVPRecipes(VPlanPtr &Plan,
                            function_ref<const InductionDescriptor *(PHINode *)>
                                GetIntOrFpInductionDescriptor,
                            SmallPtrSetImpl<Instruction *> &DeadInstructions,
                            ScalarEvolution &SE, const TargetLibraryInfo &TLI);

  /// Wrap predicated VPReplicateRecipes with a mask operand in an if-then
  /// region block and remove the mask operand. Optimize the created regions by
  /// iteratively sinking scalar operands into the region, followed by merging
  /// regions until no improvements are remaining.
  static void createAndOptimizeReplicateRegions(VPlan &Plan);

  /// Remove redundant VPBasicBlocks by merging them into their predecessor if
  /// the predecessor has a single successor.
  static bool mergeBlocksIntoPredecessors(VPlan &Plan);

  /// Remove redundant casts of inductions.
  ///
  /// Such redundant casts are casts of induction variables that can be ignored,
  /// because we already proved that the casted phi is equal to the uncasted phi
  /// in the vectorized loop. There is no need to vectorize the cast - the same
  /// value can be used for both the phi and casts in the vector loop.
  static void removeRedundantInductionCasts(VPlan &Plan);

  /// Try to replace VPWidenCanonicalIVRecipes with a widened canonical IV
  /// recipe, if it exists.
  static void removeRedundantCanonicalIVs(VPlan &Plan);

  static void removeDeadRecipes(VPlan &Plan);

  /// If any user of a VPWidenIntOrFpInductionRecipe needs scalar values,
  /// provide them by building scalar steps off of the canonical scalar IV and
  /// update the original IV's users. This is an optional optimization to reduce
  /// the needs of vector extracts.
  static void optimizeInductions(VPlan &Plan, ScalarEvolution &SE);

  /// Remove redundant EpxandSCEVRecipes in \p Plan's entry block by replacing
  /// them with already existing recipes expanding the same SCEV expression.
  static void removeRedundantExpandSCEVRecipes(VPlan &Plan);

  /// Sink users of fixed-order recurrences after the recipe defining their
  /// previous value. Then introduce FirstOrderRecurrenceSplice VPInstructions
  /// to combine the value from the recurrence phis and previous values. The
  /// current implementation assumes all users can be sunk after the previous
  /// value, which is enforced by earlier legality checks.
  static void adjustFixedOrderRecurrences(VPlan &Plan, VPBuilder &Builder);

  /// Optimize \p Plan based on \p BestVF and \p BestUF. This may restrict the
  /// resulting plan to \p BestVF and \p BestUF.
  static void optimizeForVFAndUF(VPlan &Plan, ElementCount BestVF,
                                 unsigned BestUF,
                                 PredicatedScalarEvolution &PSE);
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_VECTORIZE_VPLANTRANSFORMS_H
