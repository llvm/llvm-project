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
#include "llvm/ADT/PointerUnion.h"
#include "llvm/IR/IRBuilder.h"

namespace llvm {

class LoopVectorizationLegality;
class LoopVectorizationCostModel;
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

  PredicatedScalarEvolution &PSE;

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

  /// Cross-iteration reduction & first-order recurrence phis for which we need
  /// to add the incoming value from the backedge after all recipes have been
  /// created.
  SmallVector<VPHeaderPHIRecipe *, 4> PhisToFix;

  /// Check if \p I can be widened at the start of \p Range and possibly
  /// decrease the range such that the returned value holds for the entire \p
  /// Range. The function should not be called for memory instructions or calls.
  bool shouldWiden(Instruction *I, VFRange &Range) const;

  /// Check if the load or store instruction \p I should widened for \p
  /// Range.Start and potentially masked. Such instructions are handled by a
  /// recipe that takes an additional VPInstruction for the mask.
  VPWidenMemoryInstructionRecipe *tryToWidenMemory(Instruction *I,
                                                   ArrayRef<VPValue *> Operands,
                                                   VFRange &Range,
                                                   VPlanPtr &Plan);

  /// Check if an induction recipe should be constructed for \p Phi. If so build
  /// and return it. If not, return null.
  VPHeaderPHIRecipe *tryToOptimizeInductionPHI(PHINode *Phi,
                                               ArrayRef<VPValue *> Operands,
                                               VPlan &Plan, VFRange &Range);

  /// Optimize the special case where the operand of \p I is a constant integer
  /// induction variable.
  VPWidenIntOrFpInductionRecipe *
  tryToOptimizeInductionTruncate(TruncInst *I, ArrayRef<VPValue *> Operands,
                                 VFRange &Range, VPlan &Plan);

  /// Handle non-loop phi nodes. Return a new VPBlendRecipe otherwise. Currently
  /// all such phi nodes are turned into a sequence of select instructions as
  /// the vectorizer currently performs full if-conversion.
  VPBlendRecipe *tryToBlend(PHINode *Phi, ArrayRef<VPValue *> Operands,
                            VPlanPtr &Plan);

  /// Handle call instructions. If \p CI can be widened for \p Range.Start,
  /// return a new VPWidenCallRecipe. Range.End may be decreased to ensure same
  /// decision from \p Range.Start to \p Range.End.
  VPWidenCallRecipe *tryToWidenCall(CallInst *CI, ArrayRef<VPValue *> Operands,
                                    VFRange &Range, VPlanPtr &Plan);

  /// Check if \p I has an opcode that can be widened and return a VPWidenRecipe
  /// if it can. The function should only be called if the cost-model indicates
  /// that widening should be performed.
  VPWidenRecipe *tryToWiden(Instruction *I, ArrayRef<VPValue *> Operands,
                            VPBasicBlock *VPBB, VPlanPtr &Plan);

public:
  VPRecipeBuilder(Loop *OrigLoop, const TargetLibraryInfo *TLI,
                  LoopVectorizationLegality *Legal,
                  LoopVectorizationCostModel &CM,
                  PredicatedScalarEvolution &PSE, VPBuilder &Builder)
      : OrigLoop(OrigLoop), TLI(TLI), Legal(Legal), CM(CM), PSE(PSE),
        Builder(Builder) {}

  /// Create and return a widened recipe for \p I if one can be created within
  /// the given VF \p Range.
  VPRecipeBase *tryToCreateWidenRecipe(Instruction *Instr,
                                       ArrayRef<VPValue *> Operands,
                                       VFRange &Range, VPBasicBlock *VPBB,
                                       VPlanPtr &Plan);

  /// Set the recipe created for given ingredient. This operation is a no-op for
  /// ingredients that were not marked using a nullptr entry in the map.
  void setRecipe(Instruction *I, VPRecipeBase *R) {
    if (!Ingredient2Recipe.count(I))
      return;
    assert(Ingredient2Recipe[I] == nullptr &&
           "Recipe already set for ingredient");
    Ingredient2Recipe[I] = R;
  }

  /// Create the mask for the vector loop header block.
  void createHeaderMask(VPlan &Plan);

  /// A helper function that computes the predicate of the block BB, assuming
  /// that the header block of the loop is set to True or the loop mask when
  /// tail folding.
  void createBlockInMask(BasicBlock *BB, VPlan &Plan);

  /// Returns the *entry* mask for the block \p BB.
  VPValue *getBlockInMask(BasicBlock *BB) const;

  /// A helper function that computes the predicate of the edge between SRC
  /// and DST.
  VPValue *createEdgeMask(BasicBlock *Src, BasicBlock *Dst, VPlan &Plan);

  /// A helper that returns the previously computed predicate of the edge
  /// between SRC and DST.
  VPValue *getEdgeMask(BasicBlock *Src, BasicBlock *Dst) const;

  /// A helper function to create VPWidenCastRecipe of a \p V VPValue to a \p To
  /// type.
  /// FIXME: Remove \p From argument and take it from a \p V value
  static VPWidenCastRecipe *createCast(VPValue *V, Type *From, Type *To);

  /// A helper function which widens \p WIV step, multiplies it by WidenVFxUF
  /// and attaches to loop latch of the \p Plan. Returns multiplication.
  static VPRecipeBase *
  createWidenStep(VPWidenIntOrFpInductionRecipe &WIV, ScalarEvolution &SE,
                  VPlan &Plan,
                  DenseSet<VPRecipeBase *> *CreatedRecipes = nullptr);

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

  /// Build a VPReplicationRecipe for \p I. If it is predicated, add the mask as
  /// last operand. Range.End may be decreased to ensure same recipe behavior
  /// from \p Range.Start to \p Range.End.
  VPReplicateRecipe *handleReplication(Instruction *I, VFRange &Range,
                                       VPlan &Plan);

  /// Add the incoming values from the backedge to reduction & first-order
  /// recurrence cross-iteration phis.
  void fixHeaderPhis(VPlan &Plan);
};
} // end namespace llvm

#endif // LLVM_TRANSFORMS_VECTORIZE_VPRECIPEBUILDER_H
