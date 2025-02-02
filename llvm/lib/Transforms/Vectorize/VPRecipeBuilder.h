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
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/IRBuilder.h"

namespace llvm {

class LoopVectorizationLegality;
class LoopVectorizationCostModel;
class TargetLibraryInfo;
class TargetTransformInfo;
struct HistogramInfo;
struct VFRange;

/// A chain of instructions that form a partial reduction.
/// Designed to match: reduction_bin_op (bin_op (extend (A), (extend (B))),
/// accumulator).
struct PartialReductionChain {
  PartialReductionChain(Instruction *Reduction, Instruction *ExtendA,
                        Instruction *ExtendB, Instruction *BinOp)
      : Reduction(Reduction), ExtendA(ExtendA), ExtendB(ExtendB), BinOp(BinOp) {
  }
  /// The top-level binary operation that forms the reduction to a scalar
  /// after the loop body.
  Instruction *Reduction;
  /// The extension of each of the inner binary operation's operands.
  Instruction *ExtendA;
  Instruction *ExtendB;

  /// The binary operation using the extends that is then reduced.
  Instruction *BinOp;
};

/// Helper class to create VPRecipies from IR instructions.
class VPRecipeBuilder {
  /// The VPlan new recipes are added to.
  VPlan &Plan;

  /// The loop that we evaluate.
  Loop *OrigLoop;

  /// Target Library Info.
  const TargetLibraryInfo *TLI;

  // Target Transform Info.
  const TargetTransformInfo *TTI;

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

  // VPlan construction support: Hold a mapping from ingredients to
  // their recipe.
  DenseMap<Instruction *, VPRecipeBase *> Ingredient2Recipe;

  /// Cross-iteration reduction & first-order recurrence phis for which we need
  /// to add the incoming value from the backedge after all recipes have been
  /// created.
  SmallVector<VPHeaderPHIRecipe *, 4> PhisToFix;

  /// A mapping of partial reduction exit instructions to their scaling factor.
  DenseMap<const Instruction *, unsigned> ScaledReductionMap;

  /// Check if \p I can be widened at the start of \p Range and possibly
  /// decrease the range such that the returned value holds for the entire \p
  /// Range. The function should not be called for memory instructions or calls.
  bool shouldWiden(Instruction *I, VFRange &Range) const;

  /// Check if the load or store instruction \p I should widened for \p
  /// Range.Start and potentially masked. Such instructions are handled by a
  /// recipe that takes an additional VPInstruction for the mask.
  VPWidenMemoryRecipe *tryToWidenMemory(Instruction *I,
                                        ArrayRef<VPValue *> Operands,
                                        VFRange &Range);

  /// Check if an induction recipe should be constructed for \p Phi. If so build
  /// and return it. If not, return null.
  VPHeaderPHIRecipe *tryToOptimizeInductionPHI(PHINode *Phi,
                                               ArrayRef<VPValue *> Operands,
                                               VFRange &Range);

  /// Optimize the special case where the operand of \p I is a constant integer
  /// induction variable.
  VPWidenIntOrFpInductionRecipe *
  tryToOptimizeInductionTruncate(TruncInst *I, ArrayRef<VPValue *> Operands,
                                 VFRange &Range);

  /// Handle non-loop phi nodes. Return a new VPBlendRecipe otherwise. Currently
  /// all such phi nodes are turned into a sequence of select instructions as
  /// the vectorizer currently performs full if-conversion.
  VPBlendRecipe *tryToBlend(PHINode *Phi, ArrayRef<VPValue *> Operands);

  /// Handle call instructions. If \p CI can be widened for \p Range.Start,
  /// return a new VPWidenCallRecipe or VPWidenIntrinsicRecipe. Range.End may be
  /// decreased to ensure same decision from \p Range.Start to \p Range.End.
  VPSingleDefRecipe *tryToWidenCall(CallInst *CI, ArrayRef<VPValue *> Operands,
                                    VFRange &Range);

  /// Check if \p I has an opcode that can be widened and return a VPWidenRecipe
  /// if it can. The function should only be called if the cost-model indicates
  /// that widening should be performed.
  VPWidenRecipe *tryToWiden(Instruction *I, ArrayRef<VPValue *> Operands,
                            VPBasicBlock *VPBB);

  /// Makes Histogram count operations safe for vectorization, by emitting a
  /// llvm.experimental.vector.histogram.add intrinsic in place of the
  /// Load + Add|Sub + Store operations that perform the histogram in the
  /// original scalar loop.
  VPHistogramRecipe *tryToWidenHistogram(const HistogramInfo *HI,
                                         ArrayRef<VPValue *> Operands);

  /// Examines reduction operations to see if the target can use a cheaper
  /// operation with a wider per-iteration input VF and narrower PHI VF.
  /// Each element within Chains is a pair with a struct containing reduction
  /// information and the scaling factor between the number of elements in
  /// the input and output.
  /// Recursively calls itself to identify chained scaled reductions.
  /// Returns true if this invocation added an entry to Chains, otherwise false.
  /// i.e. returns false in the case that a subcall adds an entry to Chains,
  /// but the top-level call does not.
  bool getScaledReductions(
      Instruction *PHI, Instruction *RdxExitInstr, VFRange &Range,
      SmallVectorImpl<std::pair<PartialReductionChain, unsigned>> &Chains);

public:
  VPRecipeBuilder(VPlan &Plan, Loop *OrigLoop, const TargetLibraryInfo *TLI,
                  const TargetTransformInfo *TTI,
                  LoopVectorizationLegality *Legal,
                  LoopVectorizationCostModel &CM,
                  PredicatedScalarEvolution &PSE, VPBuilder &Builder)
      : Plan(Plan), OrigLoop(OrigLoop), TLI(TLI), TTI(TTI), Legal(Legal),
        CM(CM), PSE(PSE), Builder(Builder) {}

  std::optional<unsigned> getScalingForReduction(const Instruction *ExitInst) {
    auto It = ScaledReductionMap.find(ExitInst);
    return It == ScaledReductionMap.end() ? std::nullopt
                                          : std::make_optional(It->second);
  }

  /// Find all possible partial reductions in the loop and track all of those
  /// that are valid so recipes can be formed later.
  void collectScaledReductions(VFRange &Range);

  /// Create and return a widened recipe for \p I if one can be created within
  /// the given VF \p Range.
  VPRecipeBase *tryToCreateWidenRecipe(Instruction *Instr,
                                       ArrayRef<VPValue *> Operands,
                                       VFRange &Range, VPBasicBlock *VPBB);

  /// Create and return a partial reduction recipe for a reduction instruction
  /// along with binary operation and reduction phi operands.
  VPRecipeBase *tryToCreatePartialReduction(Instruction *Reduction,
                                            ArrayRef<VPValue *> Operands);

  /// Set the recipe created for given ingredient.
  void setRecipe(Instruction *I, VPRecipeBase *R) {
    assert(!Ingredient2Recipe.contains(I) &&
           "Cannot reset recipe for instruction.");
    Ingredient2Recipe[I] = R;
  }

  /// Create the mask for the vector loop header block.
  void createHeaderMask();

  /// A helper function that computes the predicate of the block BB, assuming
  /// that the header block of the loop is set to True or the loop mask when
  /// tail folding.
  void createBlockInMask(BasicBlock *BB);

  /// Returns the *entry* mask for the block \p BB.
  VPValue *getBlockInMask(BasicBlock *BB) const;

  /// Create an edge mask for every destination of cases and/or default.
  void createSwitchEdgeMasks(SwitchInst *SI);

  /// A helper function that computes the predicate of the edge between SRC
  /// and DST.
  VPValue *createEdgeMask(BasicBlock *Src, BasicBlock *Dst);

  /// A helper that returns the previously computed predicate of the edge
  /// between SRC and DST.
  VPValue *getEdgeMask(BasicBlock *Src, BasicBlock *Dst) const;

  /// Return the recipe created for given ingredient.
  VPRecipeBase *getRecipe(Instruction *I) {
    assert(Ingredient2Recipe.count(I) &&
           "Recording this ingredients recipe was not requested");
    assert(Ingredient2Recipe[I] != nullptr &&
           "Ingredient doesn't have a recipe");
    return Ingredient2Recipe[I];
  }

  /// Build a VPReplicationRecipe for \p I using \p Operands. If it is
  /// predicated, add the mask as last operand. Range.End may be decreased to
  /// ensure same recipe behavior from \p Range.Start to \p Range.End.
  VPReplicateRecipe *handleReplication(Instruction *I,
                                       ArrayRef<VPValue *> Operands,
                                       VFRange &Range);

  /// Add the incoming values from the backedge to reduction & first-order
  /// recurrence cross-iteration phis.
  void fixHeaderPhis();

  /// Returns a range mapping the values of the range \p Operands to their
  /// corresponding VPValues.
  iterator_range<mapped_iterator<Use *, std::function<VPValue *(Value *)>>>
  mapToVPValues(User::op_range Operands);

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
