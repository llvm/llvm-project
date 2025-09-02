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
#include "VPlanVerifier.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

class InductionDescriptor;
class Instruction;
class PHINode;
class ScalarEvolution;
class PredicatedScalarEvolution;
class TargetLibraryInfo;
class VPBuilder;
class VPRecipeBuilder;
struct VFRange;

extern cl::opt<bool> VerifyEachVPlan;

struct VPlanTransforms {
  /// Helper to run a VPlan transform \p Transform on \p VPlan, forwarding extra
  /// arguments to the transform. Returns the boolean returned by the transform.
  template <typename... ArgsTy>
  static bool runPass(bool (*Transform)(VPlan &, ArgsTy...), VPlan &Plan,
                      typename std::remove_reference<ArgsTy>::type &...Args) {
    bool Res = Transform(Plan, Args...);
    if (VerifyEachVPlan)
      verifyVPlanIsValid(Plan);
    return Res;
  }
  /// Helper to run a VPlan transform \p Transform on \p VPlan, forwarding extra
  /// arguments to the transform.
  template <typename... ArgsTy>
  static void runPass(void (*Fn)(VPlan &, ArgsTy...), VPlan &Plan,
                      typename std::remove_reference<ArgsTy>::type &...Args) {
    Fn(Plan, Args...);
    if (VerifyEachVPlan)
      verifyVPlanIsValid(Plan);
  }

  /// Create a base VPlan0, serving as the common starting point for all later
  /// candidates. It consists of an initial plain CFG loop with loop blocks from
  /// \p TheLoop being directly translated to VPBasicBlocks with VPInstruction
  /// corresponding to the input IR.
  ///
  /// The created loop is wrapped in an initial skeleton to facilitate
  /// vectorization, consisting of a vector pre-header, an exit block for the
  /// main vector loop (middle.block) and a new block as preheader of the scalar
  /// loop (scalar.ph). See below for an illustration. It also adds a canonical
  /// IV and its increment, using \p InductionTy and \p IVDL, and creates a
  /// VPValue expression for the original trip count.
  ///
  ///    [ ] <-- Plan's entry VPIRBasicBlock, wrapping the original loop's
  ///    / \       old preheader. Will contain iteration number check and SCEV
  ///   |   |      expansions.
  ///   |   |
  ///   /   v
  ///  |   [ ] <-- vector loop bypass (may consist of multiple blocks) will be
  ///  |  / |      added later.
  ///  | /  v
  ///  ||  [ ]     <-- vector pre header.
  ///  |/   |
  ///  |    v
  ///  |   [  ] \  <-- plain CFG loop wrapping original loop to be vectorized.
  ///  |   [  ]_|
  ///  |    |
  ///  |    v
  ///  |   [ ]   <--- middle-block with the branch to successors
  ///  |   / |
  ///  |  /  |
  ///  | |   v
  ///  \--->[ ]   <--- scalar preheader (initial a VPBasicBlock, which will be
  ///    |   |        replaced later by a VPIRBasicBlock wrapping the scalar
  ///    |   |         preheader basic block.
  ///    |   |
  ///        v      <-- edge from middle to exit iff epilogue is not required.
  ///    |  [ ] \
  ///    |  [ ]_|   <-- old scalar loop to handle remainder (scalar epilogue,
  ///    |   |          header wrapped in VPIRBasicBlock).
  ///    \   |
  ///     \  v
  ///      >[ ]     <-- original loop exit block(s), wrapped in VPIRBasicBlocks.
  LLVM_ABI_FOR_TEST static std::unique_ptr<VPlan>
  buildVPlan0(Loop *TheLoop, LoopInfo &LI, Type *InductionTy, DebugLoc IVDL,
              PredicatedScalarEvolution &PSE);

  /// Update \p Plan to account for all early exits.
  LLVM_ABI_FOR_TEST static void handleEarlyExits(VPlan &Plan,
                                                 bool HasUncountableExit);

  /// If a check is needed to guard executing the scalar epilogue loop, it will
  /// be added to the middle block.
  LLVM_ABI_FOR_TEST static void addMiddleCheck(VPlan &Plan,
                                               bool RequiresScalarEpilogueCheck,
                                               bool TailFolded);

  // Create a check to \p Plan to see if the vector loop should be executed.
  static void addMinimumIterationCheck(
      VPlan &Plan, ElementCount VF, unsigned UF,
      ElementCount MinProfitableTripCount, bool RequiresScalarEpilogue,
      bool TailFolded, bool CheckNeededWithTailFolding, Loop *OrigLoop,
      const uint32_t *MinItersBypassWeights, DebugLoc DL, ScalarEvolution &SE);

  /// Replace loops in \p Plan's flat CFG with VPRegionBlocks, turning \p Plan's
  /// flat CFG into a hierarchical CFG.
  LLVM_ABI_FOR_TEST static void createLoopRegions(VPlan &Plan);

  /// Wrap runtime check block \p CheckBlock in a VPIRBB and \p Cond in a
  /// VPValue and connect the block to \p Plan, using the VPValue as branch
  /// condition.
  static void attachCheckBlock(VPlan &Plan, Value *Cond, BasicBlock *CheckBlock,
                               bool AddBranchWeights);

  /// Replaces the VPInstructions in \p Plan with corresponding
  /// widen recipes. Returns false if any VPInstructions could not be converted
  /// to a wide recipe if needed.
  LLVM_ABI_FOR_TEST static bool tryToConvertVPInstructionsToVPRecipes(
      VPlanPtr &Plan,
      function_ref<const InductionDescriptor *(PHINode *)>
          GetIntOrFpInductionDescriptor,
      const TargetLibraryInfo &TLI);

  /// Try to have all users of fixed-order recurrences appear after the recipe
  /// defining their previous value, by either sinking users or hoisting recipes
  /// defining their previous value (and its operands). Then introduce
  /// FirstOrderRecurrenceSplice VPInstructions to combine the value from the
  /// recurrence phis and previous values.
  /// \returns true if all users of fixed-order recurrences could be re-arranged
  /// as needed or false if it is not possible. In the latter case, \p Plan is
  /// not valid.
  static bool adjustFixedOrderRecurrences(VPlan &Plan, VPBuilder &Builder);

  /// Check if \p Plan contains any FMaxNum or FMinNum reductions. If they do,
  /// try to update the vector loop to exit early if any input is NaN and resume
  /// executing in the scalar loop to handle the NaNs there. Return false if
  /// this attempt was unsuccessful.
  static bool handleMaxMinNumReductions(VPlan &Plan);

  /// Clear NSW/NUW flags from reduction instructions if necessary.
  static void clearReductionWrapFlags(VPlan &Plan);

  /// Explicitly unroll \p Plan by \p UF.
  static void unrollByUF(VPlan &Plan, unsigned UF);

  /// Replace each VPReplicateRecipe outside on any replicate region in \p Plan
  /// with \p VF single-scalar recipes.
  /// TODO: Also replicate VPReplicateRecipes inside replicate regions, thereby
  /// dissolving the latter.
  static void replicateByVF(VPlan &Plan, ElementCount VF);

  /// Optimize \p Plan based on \p BestVF and \p BestUF. This may restrict the
  /// resulting plan to \p BestVF and \p BestUF.
  static void optimizeForVFAndUF(VPlan &Plan, ElementCount BestVF,
                                 unsigned BestUF,
                                 PredicatedScalarEvolution &PSE);

  /// Apply VPlan-to-VPlan optimizations to \p Plan, including induction recipe
  /// optimizations, dead recipe removal, replicate region optimizations and
  /// block merging.
  static void optimize(VPlan &Plan);

  /// Wrap predicated VPReplicateRecipes with a mask operand in an if-then
  /// region block and remove the mask operand. Optimize the created regions by
  /// iteratively sinking scalar operands into the region, followed by merging
  /// regions until no improvements are remaining.
  static void createAndOptimizeReplicateRegions(VPlan &Plan);

  /// Replace (ICMP_ULE, wide canonical IV, backedge-taken-count) checks with an
  /// (active-lane-mask recipe, wide canonical IV, trip-count). If \p
  /// UseActiveLaneMaskForControlFlow is true, introduce an
  /// VPActiveLaneMaskPHIRecipe. If \p DataAndControlFlowWithoutRuntimeCheck is
  /// true, no minimum-iteration runtime check will be created (during skeleton
  /// creation) and instead it is handled using active-lane-mask. \p
  /// DataAndControlFlowWithoutRuntimeCheck implies \p
  /// UseActiveLaneMaskForControlFlow.
  static void addActiveLaneMask(VPlan &Plan,
                                bool UseActiveLaneMaskForControlFlow,
                                bool DataAndControlFlowWithoutRuntimeCheck);

  /// Insert truncates and extends for any truncated recipe. Redundant casts
  /// will be folded later.
  static void
  truncateToMinimalBitwidths(VPlan &Plan,
                             const MapVector<Instruction *, uint64_t> &MinBWs);

  /// Replace symbolic strides from \p StridesMap in \p Plan with constants when
  /// possible.
  static void
  replaceSymbolicStrides(VPlan &Plan, PredicatedScalarEvolution &PSE,
                         const DenseMap<Value *, const SCEV *> &StridesMap);

  /// Drop poison flags from recipes that may generate a poison value that is
  /// used after vectorization, even when their operands are not poison. Those
  /// recipes meet the following conditions:
  ///  * Contribute to the address computation of a recipe generating a widen
  ///    memory load/store (VPWidenMemoryInstructionRecipe or
  ///    VPInterleaveRecipe).
  ///  * Such a widen memory load/store has at least one underlying Instruction
  ///    that is in a basic block that needs predication and after vectorization
  ///    the generated instruction won't be predicated.
  /// Uses \p BlockNeedsPredication to check if a block needs predicating.
  /// TODO: Replace BlockNeedsPredication callback with retrieving info from
  ///       VPlan directly.
  static void dropPoisonGeneratingRecipes(
      VPlan &Plan,
      const std::function<bool(BasicBlock *)> &BlockNeedsPredication);

  /// Add a VPEVLBasedIVPHIRecipe and related recipes to \p Plan and
  /// replaces all uses except the canonical IV increment of
  /// VPCanonicalIVPHIRecipe with a VPEVLBasedIVPHIRecipe.
  /// VPCanonicalIVPHIRecipe is only used to control the loop after
  /// this transformation.
  static void
  addExplicitVectorLength(VPlan &Plan,
                          const std::optional<unsigned> &MaxEVLSafeElements);

  // For each Interleave Group in \p InterleaveGroups replace the Recipes
  // widening its memory instructions with a single VPInterleaveRecipe at its
  // insertion point.
  static void createInterleaveGroups(
      VPlan &Plan,
      const SmallPtrSetImpl<const InterleaveGroup<Instruction> *>
          &InterleaveGroups,
      VPRecipeBuilder &RecipeBuilder, const bool &ScalarEpilogueAllowed);

  /// Remove dead recipes from \p Plan.
  static void removeDeadRecipes(VPlan &Plan);

  /// Update \p Plan to account for the uncountable early exit from \p
  /// EarlyExitingVPBB to \p EarlyExitVPBB by
  ///  * updating the condition exiting the loop via the latch to include the
  ///    early exit condition,
  ///  * splitting the original middle block to branch to the early exit block
  ///    conditionally - according to the early exit condition.
  static void handleUncountableEarlyExit(VPBasicBlock *EarlyExitingVPBB,
                                         VPBasicBlock *EarlyExitVPBB,
                                         VPlan &Plan, VPBasicBlock *HeaderVPBB,
                                         VPBasicBlock *LatchVPBB);

  /// Replace loop regions with explicit CFG.
  static void dissolveLoopRegions(VPlan &Plan);

  /// Transform EVL loops to use variable-length stepping after region
  /// dissolution.
  ///
  /// Once loop regions are replaced with explicit CFG, EVL loops can step with
  /// variable vector lengths instead of fixed lengths. This transformation:
  ///  * Makes EVL-Phi concrete.
  //   * Removes CanonicalIV and increment.
  ///  * Replaces the exit condition from
  ///      (branch-on-count CanonicalIVInc, VectorTripCount)
  ///    to
  ///      (branch-on-cond eq AVLNext, 0)
  static void canonicalizeEVLLoops(VPlan &Plan);

  /// Lower abstract recipes to concrete ones, that can be codegen'd.
  static void convertToConcreteRecipes(VPlan &Plan);

  /// This function converts initial recipes to the abstract recipes and clamps
  /// \p Range based on cost model for following optimizations and cost
  /// estimations. The converted abstract recipes will lower to concrete
  /// recipes before codegen.
  static void convertToAbstractRecipes(VPlan &Plan, VPCostContext &Ctx,
                                       VFRange &Range);

  /// Perform instcombine-like simplifications on recipes in \p Plan.
  static void simplifyRecipes(VPlan &Plan);

  /// Remove BranchOnCond recipes with true or false conditions together with
  /// removing dead edges to their successors.
  static void removeBranchOnConst(VPlan &Plan);

  /// If there's a single exit block, optimize its phi recipes that use exiting
  /// IV values by feeding them precomputed end values instead, possibly taken
  /// one step backwards.
  static void
  optimizeInductionExitUsers(VPlan &Plan,
                             DenseMap<VPValue *, VPValue *> &EndValues,
                             ScalarEvolution &SE);

  /// Add explicit broadcasts for live-ins and VPValues defined in \p Plan's entry block if they are used as vectors.
  static void materializeBroadcasts(VPlan &Plan);

  // Materialize vector trip counts for constants early if it can simply be
  // computed as (Original TC / VF * UF) * VF * UF.
  static void
  materializeConstantVectorTripCount(VPlan &Plan, ElementCount BestVF,
                                     unsigned BestUF,
                                     PredicatedScalarEvolution &PSE);

  /// Materialize vector trip count computations to a set of VPInstructions.
  static void materializeVectorTripCount(VPlan &Plan,
                                         VPBasicBlock *VectorPHVPBB,
                                         bool TailByMasking,
                                         bool RequiresScalarEpilogue);

  /// Materialize the backedge-taken count to be computed explicitly using
  /// VPInstructions.
  static void materializeBackedgeTakenCount(VPlan &Plan,
                                            VPBasicBlock *VectorPH);

  /// Add explicit Build[Struct]Vector recipes that combine multiple scalar
  /// values into single vectors.
  static void materializeBuildVectors(VPlan &Plan);

  /// Materialize VF and VFxUF to be computed explicitly using VPInstructions.
  static void materializeVFAndVFxUF(VPlan &Plan, VPBasicBlock *VectorPH,
                                    ElementCount VF);

  /// Expand VPExpandSCEVRecipes in \p Plan's entry block. Each
  /// VPExpandSCEVRecipe is replaced with a live-in wrapping the expanded IR
  /// value. A mapping from SCEV expressions to their expanded IR value is
  /// returned.
  static DenseMap<const SCEV *, Value *> expandSCEVs(VPlan &Plan,
                                                     ScalarEvolution &SE);

  /// Try to convert a plan with interleave groups with VF elements to a plan
  /// with the interleave groups replaced by wide loads and stores processing VF
  /// elements, if all transformed interleave groups access the full vector
  /// width (checked via \o VectorRegWidth). This effectively is a very simple
  /// form of loop-aware SLP, where we use interleave groups to identify
  /// candidates.
  static void narrowInterleaveGroups(VPlan &Plan, ElementCount VF,
                                     unsigned VectorRegWidth);

  /// Predicate and linearize the control-flow in the only loop region of
  /// \p Plan. If \p FoldTail is true, create a mask guarding the loop
  /// header, otherwise use all-true for the header mask. Masks for blocks are
  /// added to a block-to-mask map which is returned in order to be used later
  /// for wide recipe construction. This argument is temporary and will be
  /// removed in the future.
  static DenseMap<VPBasicBlock *, VPValue *>
  introduceMasksAndLinearize(VPlan &Plan, bool FoldTail);

  /// Add branch weight metadata, if the \p Plan's middle block is terminated by
  /// a BranchOnCond recipe.
  static void
  addBranchWeightToMiddleTerminator(VPlan &Plan, ElementCount VF,
                                    std::optional<unsigned> VScaleForTuning);
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_VECTORIZE_VPLANTRANSFORMS_H
