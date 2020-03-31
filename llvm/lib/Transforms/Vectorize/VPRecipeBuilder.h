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
#include "llvm/IR/IRBuilder.h"

namespace llvm {

class LoopVectorizationLegality;
class LoopVectorizationCostModel;
class TargetTransformInfo;
class TargetLibraryInfo;

/// Helper class to create VPRecipies from IR instructions.
class VPRecipeBuilder {
  /// The loop that we evaluate.
  Loop *OrigLoop;

  /// Target Library Info.
  const TargetLibraryInfo *TLI;

  /// The legality analysis.
  LoopVectorizationLegality *Legal;

  /// The profitablity analysis.
  LoopVectorizationCostModel &CM;

  VPBuilder &Builder;

  /// When we if-convert we need to create edge masks. We have to cache values
  /// so that we don't end up with exponential recursion/IR. Note that
  /// if-conversion currently takes place during VPlan-construction, so these
  /// caches are only used at that stage.
  using EdgeMaskCacheTy =
      DenseMap<std::pair<BasicBlock *, BasicBlock *>, VPValue *>;
  using BlockMaskCacheTy = DenseMap<BasicBlock *, VPValue *>;
  EdgeMaskCacheTy EdgeMaskCache;
  BlockMaskCacheTy BlockMaskCache;

  // VPlan-VPlan transformations support: Hold a mapping from ingredients to
  // their recipe. To save on memory, only do so for selected ingredients,
  // marked by having a nullptr entry in this map.
  DenseMap<Instruction *, VPRecipeBase *> Ingredient2Recipe;

  /// Set the recipe created for given ingredient. This operation is a no-op for
  /// ingredients that were not marked using a nullptr entry in the map.
  void setRecipe(Instruction *I, VPRecipeBase *R) {
    if (!Ingredient2Recipe.count(I))
      return;
    assert(Ingredient2Recipe[I] == nullptr &&
           "Recipe already set for ingredient");
    Ingredient2Recipe[I] = R;
  }

public:
  /// A helper function that computes the predicate of the block BB, assuming
  /// that the header block of the loop is set to True. It returns the *entry*
  /// mask for the block BB.
  VPValue *createBlockInMask(BasicBlock *BB, VPlanPtr &Plan);

  /// A helper function that computes the predicate of the edge between SRC
  /// and DST.
  VPValue *createEdgeMask(BasicBlock *Src, BasicBlock *Dst, VPlanPtr &Plan);

  /// Mark given ingredient for recording its recipe once one is created for
  /// it.
  void recordRecipeOf(Instruction *I) {
    assert((!Ingredient2Recipe.count(I) || Ingredient2Recipe[I] == nullptr) &&
           "Recipe already set for ingredient");
    Ingredient2Recipe[I] = nullptr;
  }

  /// Return the recipe created for given ingredient.
  VPRecipeBase *getRecipe(Instruction *I) {
    assert(Ingredient2Recipe.count(I) &&
           "Recording this ingredients recipe was not requested");
    assert(Ingredient2Recipe[I] != nullptr &&
           "Ingredient doesn't have a recipe");
    return Ingredient2Recipe[I];
  }

  /// Check if \I is a memory instruction to be widened for \p Range.Start and
  /// potentially masked. Such instructions are handled by a recipe that takes
  /// an additional VPInstruction for the mask.
  VPWidenMemoryInstructionRecipe *
  tryToWidenMemory(Instruction *I, VFRange &Range, VPlanPtr &Plan);

  /// Check if an induction recipe should be constructed for \I within the given
  /// VF \p Range. If so build and return it. If not, return null. \p Range.End
  /// may be decreased to ensure same decision from \p Range.Start to
  /// \p Range.End.
  VPWidenIntOrFpInductionRecipe *tryToOptimizeInduction(Instruction *I,
                                                        VFRange &Range);

  /// Handle non-loop phi nodes. Currently all such phi nodes are turned into
  /// a sequence of select instructions as the vectorizer currently performs
  /// full if-conversion.
  VPBlendRecipe *tryToBlend(Instruction *I, VPlanPtr &Plan);

  /// Check if \p I can be widened within the given VF \p Range. If \p I can be
  /// widened for \p Range.Start, check if the last recipe of \p VPBB can be
  /// extended to include \p I or else build a new VPWidenRecipe for it and
  /// append it to \p VPBB. Return true if \p I can be widened for Range.Start,
  /// false otherwise. Range.End may be decreased to ensure same decision from
  /// \p Range.Start to \p Range.End.
  bool tryToWiden(Instruction *I, VPBasicBlock *VPBB, VFRange &Range);

  /// Create a replicating region for instruction \p I that requires
  /// predication. \p PredRecipe is a VPReplicateRecipe holding \p I.
  VPRegionBlock *createReplicateRegion(Instruction *I, VPRecipeBase *PredRecipe,
                                       VPlanPtr &Plan);

public:
  VPRecipeBuilder(Loop *OrigLoop, const TargetLibraryInfo *TLI,
                  LoopVectorizationLegality *Legal,
                  LoopVectorizationCostModel &CM, VPBuilder &Builder)
      : OrigLoop(OrigLoop), TLI(TLI), Legal(Legal), CM(CM), Builder(Builder) {}

  /// Check if a recipe can be create for \p I withing the given VF \p Range.
  /// If a recipe can be created, it adds it to \p VPBB.
  bool tryToCreateRecipe(Instruction *Instr, VFRange &Range, VPlanPtr &Plan,
                         VPBasicBlock *VPBB);

  /// Build a VPReplicationRecipe for \p I and enclose it within a Region if it
  /// is predicated. \return \p VPBB augmented with this new recipe if \p I is
  /// not predicated, otherwise \return a new VPBasicBlock that succeeds the new
  /// Region. Update the packing decision of predicated instructions if they
  /// feed \p I. Range.End may be decreased to ensure same recipe behavior from
  /// \p Range.Start to \p Range.End.
  VPBasicBlock *handleReplication(
      Instruction *I, VFRange &Range, VPBasicBlock *VPBB,
      DenseMap<Instruction *, VPReplicateRecipe *> &PredInst2Recipe,
      VPlanPtr &Plan);
};
} // end namespace llvm

#endif // LLVM_TRANSFORMS_VECTORIZE_VPRECIPEBUILDER_H
