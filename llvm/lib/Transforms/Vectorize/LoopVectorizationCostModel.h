//===- LoopVectorizationCostModel.h - Costing for LoopVectorize -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_VECTORIZE_LOOPVECTORIZATIONCOSTMODEL_H
#define LLVM_TRANSFORMS_VECTORIZE_LOOPVECTORIZATIONCOSTMODEL_H

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/SizeOpts.h"
#include "llvm/Transforms/Vectorize/LoopVectorizationLegality.h"

namespace llvm {
extern cl::opt<bool> ForceTargetSupportsScalableVectors;
extern cl::opt<cl::boolOrDefault> ForceSafeDivisor;
extern cl::opt<bool> PreferPredicatedReductionSelect;

/// A class that represents two vectorization factors (initialized with 0 by
/// default). One for fixed-width vectorization and one for scalable
/// vectorization. This can be used by the vectorizer to choose from a range of
/// fixed and/or scalable VFs in order to find the most cost-effective VF to
/// vectorize with.
struct FixedScalableVFPair {
  ElementCount FixedVF;
  ElementCount ScalableVF;

  FixedScalableVFPair()
      : FixedVF(ElementCount::getFixed(0)),
        ScalableVF(ElementCount::getScalable(0)) {}
  FixedScalableVFPair(const ElementCount &Max) : FixedScalableVFPair() {
    *(Max.isScalable() ? &ScalableVF : &FixedVF) = Max;
  }
  FixedScalableVFPair(const ElementCount &FixedVF,
                      const ElementCount &ScalableVF)
      : FixedVF(FixedVF), ScalableVF(ScalableVF) {
    assert(!FixedVF.isScalable() && ScalableVF.isScalable() &&
           "Invalid scalable properties");
  }

  static FixedScalableVFPair getNone() { return FixedScalableVFPair(); }

  /// \return true if either fixed- or scalable VF is non-zero.
  explicit operator bool() const { return FixedVF || ScalableVF; }

  /// \return true if either fixed- or scalable VF is a valid vector VF.
  bool hasVector() const { return FixedVF.isVector() || ScalableVF.isVector(); }
};

// Loop vectorization cost-model hints how the scalar epilogue loop should be
// lowered.
enum ScalarEpilogueLowering {

  // The default: allowing scalar epilogues.
  CM_ScalarEpilogueAllowed,

  // Vectorization with OptForSize: don't allow epilogues.
  CM_ScalarEpilogueNotAllowedOptSize,

  // A special case of vectorisation with OptForSize: loops with a very small
  // trip count are considered for vectorization under OptForSize, thereby
  // making sure the cost of their loop body is dominant, free of runtime
  // guards and scalar iteration overheads.
  CM_ScalarEpilogueNotAllowedLowTripLoop,

  // Loop hint predicate indicating an epilogue is undesired.
  CM_ScalarEpilogueNotNeededUsePredicate,

  // Directive indicating we must either tail fold or not vectorize
  CM_ScalarEpilogueNotAllowedUsePredicate
};

/// LoopVectorizationCostModel - estimates the expected speedups due to
/// vectorization.
/// In many cases vectorization is not profitable. This can happen because of
/// a number of reasons. In this class we mainly attempt to predict the
/// expected speedup/slowdowns due to the supported instruction set. We use the
/// TargetTransformInfo to query the different backends for the cost of
/// different operations.
class LoopVectorizationCostModel {
  friend class LoopVectorizationPlanner;

public:
  LoopVectorizationCostModel(ScalarEpilogueLowering SEL, Loop *L,
                             PredicatedScalarEvolution &PSE, LoopInfo *LI,
                             LoopVectorizationLegality *Legal,
                             const TargetTransformInfo &TTI,
                             const TargetLibraryInfo *TLI, DemandedBits *DB,
                             AssumptionCache *AC,
                             OptimizationRemarkEmitter *ORE, const Function *F,
                             const LoopVectorizeHints *Hints,
                             InterleavedAccessInfo &IAI,
                             ProfileSummaryInfo *PSI, BlockFrequencyInfo *BFI)
      : ScalarEpilogueStatus(SEL), TheLoop(L), PSE(PSE), LI(LI), Legal(Legal),
        TTI(TTI), TLI(TLI), DB(DB), AC(AC), ORE(ORE), TheFunction(F),
        Hints(Hints), InterleaveInfo(IAI) {
    if (TTI.supportsScalableVectors() || ForceTargetSupportsScalableVectors)
      initializeVScaleForTuning();
    CostKind = F->hasMinSize() ? TTI::TCK_CodeSize : TTI::TCK_RecipThroughput;
    // Query this against the original loop and save it here because the profile
    // of the original loop header may change as the transformation happens.
    OptForSize = llvm::shouldOptimizeForSize(L->getHeader(), PSI, BFI,
                                             PGSOQueryType::IRPass);
  }

  /// \return An upper bound for the vectorization factors (both fixed and
  /// scalable). If the factors are 0, vectorization and interleaving should be
  /// avoided up front.
  FixedScalableVFPair computeMaxVF(ElementCount UserVF, unsigned UserIC);

  /// \return True if runtime checks are required for vectorization, and false
  /// otherwise.
  bool runtimeChecksRequired();

  /// Setup cost-based decisions for user vectorization factor.
  /// \return true if the UserVF is a feasible VF to be chosen.
  bool selectUserVectorizationFactor(ElementCount UserVF) {
    collectNonVectorizedAndSetWideningDecisions(UserVF);
    return expectedCost(UserVF).isValid();
  }

  /// \return True if maximizing vector bandwidth is enabled by the target or
  /// user options, for the given register kind.
  bool useMaxBandwidth(TargetTransformInfo::RegisterKind RegKind);

  /// \return True if register pressure should be considered for the given VF.
  bool shouldConsiderRegPressureForVF(ElementCount VF);

  /// \return The size (in bits) of the smallest and widest types in the code
  /// that needs to be vectorized. We ignore values that remain scalar such as
  /// 64 bit loop indices.
  std::pair<unsigned, unsigned> getSmallestAndWidestTypes();

  /// Memory access instruction may be vectorized in more than one way.
  /// Form of instruction after vectorization depends on cost.
  /// This function takes cost-based decisions for Load/Store instructions
  /// and collects them in a map. This decisions map is used for building
  /// the lists of loop-uniform and loop-scalar instructions.
  /// The calculated cost is saved with widening decision in order to
  /// avoid redundant calculations.
  void setCostBasedWideningDecision(ElementCount VF);

  /// A call may be vectorized in different ways depending on whether we have
  /// vectorized variants available and whether the target supports masking.
  /// This function analyzes all calls in the function at the supplied VF,
  /// makes a decision based on the costs of available options, and stores that
  /// decision in a map for use in planning and plan execution.
  void setVectorizedCallDecision(ElementCount VF);

  /// Collect values we want to ignore in the cost model.
  void collectValuesToIgnore();

  /// Collect all element types in the loop for which widening is needed.
  void collectElementTypesForWidening();

  /// Split reductions into those that happen in the loop, and those that happen
  /// outside. In loop reductions are collected into InLoopReductions.
  void collectInLoopReductions();

  /// Returns true if we should use strict in-order reductions for the given
  /// RdxDesc. This is true if the -enable-strict-reductions flag is passed,
  /// the IsOrdered flag of RdxDesc is set and we do not allow reordering
  /// of FP operations.
  bool useOrderedReductions(const RecurrenceDescriptor &RdxDesc) const {
    return !Hints->allowReordering() && RdxDesc.isOrdered();
  }

  /// \returns The smallest bitwidth each instruction can be represented with.
  /// The vector equivalents of these instructions should be truncated to this
  /// type.
  const MapVector<Instruction *, uint64_t> &getMinimalBitwidths() const {
    return MinBWs;
  }

  /// \returns True if it is more profitable to scalarize instruction \p I for
  /// vectorization factor \p VF.
  bool isProfitableToScalarize(Instruction *I, ElementCount VF) const {
    assert(VF.isVector() &&
           "Profitable to scalarize relevant only for VF > 1.");
    assert(
        TheLoop->isInnermost() &&
        "cost-model should not be used for outer loops (in VPlan-native path)");

    auto Scalars = InstsToScalarize.find(VF);
    assert(Scalars != InstsToScalarize.end() &&
           "VF not yet analyzed for scalarization profitability");
    return Scalars->second.contains(I);
  }

  /// Returns true if \p I is known to be uniform after vectorization.
  bool isUniformAfterVectorization(Instruction *I, ElementCount VF) const {
    assert(
        TheLoop->isInnermost() &&
        "cost-model should not be used for outer loops (in VPlan-native path)");
    // Pseudo probe needs to be duplicated for each unrolled iteration and
    // vector lane so that profiled loop trip count can be accurately
    // accumulated instead of being under counted.
    if (isa<PseudoProbeInst>(I))
      return false;

    if (VF.isScalar())
      return true;

    auto UniformsPerVF = Uniforms.find(VF);
    assert(UniformsPerVF != Uniforms.end() &&
           "VF not yet analyzed for uniformity");
    return UniformsPerVF->second.count(I);
  }

  /// Returns true if \p I is known to be scalar after vectorization.
  bool isScalarAfterVectorization(Instruction *I, ElementCount VF) const {
    assert(
        TheLoop->isInnermost() &&
        "cost-model should not be used for outer loops (in VPlan-native path)");
    if (VF.isScalar())
      return true;

    auto ScalarsPerVF = Scalars.find(VF);
    assert(ScalarsPerVF != Scalars.end() &&
           "Scalar values are not calculated for VF");
    return ScalarsPerVF->second.count(I);
  }

  /// \returns True if instruction \p I can be truncated to a smaller bitwidth
  /// for vectorization factor \p VF.
  bool canTruncateToMinimalBitwidth(Instruction *I, ElementCount VF) const {
    return VF.isVector() && MinBWs.contains(I) &&
           !isProfitableToScalarize(I, VF) &&
           !isScalarAfterVectorization(I, VF);
  }

  /// Decision that was taken during cost calculation for memory instruction.
  enum InstWidening {
    CM_Unknown,
    CM_Widen,         // For consecutive accesses with stride +1.
    CM_Widen_Reverse, // For consecutive accesses with stride -1.
    CM_Interleave,
    CM_GatherScatter,
    CM_Scalarize,
    CM_VectorCall,
    CM_IntrinsicCall
  };

  /// Save vectorization decision \p W and \p Cost taken by the cost model for
  /// instruction \p I and vector width \p VF.
  void setWideningDecision(Instruction *I, ElementCount VF, InstWidening W,
                           InstructionCost Cost) {
    assert(VF.isVector() && "Expected VF >=2");
    WideningDecisions[{I, VF}] = {W, Cost};
  }

  /// Save vectorization decision \p W and \p Cost taken by the cost model for
  /// interleaving group \p Grp and vector width \p VF.
  void setWideningDecision(const InterleaveGroup<Instruction> *Grp,
                           ElementCount VF, InstWidening W,
                           InstructionCost Cost) {
    assert(VF.isVector() && "Expected VF >=2");
    /// Broadcast this decicion to all instructions inside the group.
    /// When interleaving, the cost will only be assigned one instruction, the
    /// insert position. For other cases, add the appropriate fraction of the
    /// total cost to each instruction. This ensures accurate costs are used,
    /// even if the insert position instruction is not used.
    InstructionCost InsertPosCost = Cost;
    InstructionCost OtherMemberCost = 0;
    if (W != CM_Interleave)
      OtherMemberCost = InsertPosCost = Cost / Grp->getNumMembers();
    ;
    for (unsigned Idx = 0; Idx < Grp->getFactor(); ++Idx) {
      if (auto *I = Grp->getMember(Idx)) {
        if (Grp->getInsertPos() == I)
          WideningDecisions[{I, VF}] = {W, InsertPosCost};
        else
          WideningDecisions[{I, VF}] = {W, OtherMemberCost};
      }
    }
  }

  /// Return the cost model decision for the given instruction \p I and vector
  /// width \p VF. Return CM_Unknown if this instruction did not pass
  /// through the cost modeling.
  InstWidening getWideningDecision(Instruction *I, ElementCount VF) const {
    assert(VF.isVector() && "Expected VF to be a vector VF");
    assert(
        TheLoop->isInnermost() &&
        "cost-model should not be used for outer loops (in VPlan-native path)");

    std::pair<Instruction *, ElementCount> InstOnVF(I, VF);
    auto Itr = WideningDecisions.find(InstOnVF);
    if (Itr == WideningDecisions.end())
      return CM_Unknown;
    return Itr->second.first;
  }

  /// Return the vectorization cost for the given instruction \p I and vector
  /// width \p VF.
  InstructionCost getWideningCost(Instruction *I, ElementCount VF) {
    assert(VF.isVector() && "Expected VF >=2");
    std::pair<Instruction *, ElementCount> InstOnVF(I, VF);
    assert(WideningDecisions.contains(InstOnVF) &&
           "The cost is not calculated");
    return WideningDecisions[InstOnVF].second;
  }

  struct CallWideningDecision {
    InstWidening Kind;
    Function *Variant;
    Intrinsic::ID IID;
    std::optional<unsigned> MaskPos;
    InstructionCost Cost;
  };

  void setCallWideningDecision(CallInst *CI, ElementCount VF, InstWidening Kind,
                               Function *Variant, Intrinsic::ID IID,
                               std::optional<unsigned> MaskPos,
                               InstructionCost Cost) {
    assert(!VF.isScalar() && "Expected vector VF");
    CallWideningDecisions[{CI, VF}] = {Kind, Variant, IID, MaskPos, Cost};
  }

  CallWideningDecision getCallWideningDecision(CallInst *CI,
                                               ElementCount VF) const {
    assert(!VF.isScalar() && "Expected vector VF");
    auto I = CallWideningDecisions.find({CI, VF});
    if (I == CallWideningDecisions.end())
      return {CM_Unknown, nullptr, Intrinsic::not_intrinsic, std::nullopt, 0};
    return I->second;
  }

  /// Return True if instruction \p I is an optimizable truncate whose operand
  /// is an induction variable. Such a truncate will be removed by adding a new
  /// induction variable with the destination type.
  bool isOptimizableIVTruncate(Instruction *I, ElementCount VF);

  /// Collects the instructions to scalarize for each predicated instruction in
  /// the loop.
  void collectInstsToScalarize(ElementCount VF);

  /// Collect values that will not be widened, including Uniforms, Scalars, and
  /// Instructions to Scalarize for the given \p VF.
  /// The sets depend on CM decision for Load/Store instructions
  /// that may be vectorized as interleave, gather-scatter or scalarized.
  /// Also make a decision on what to do about call instructions in the loop
  /// at that VF -- scalarize, call a known vector routine, or call a
  /// vector intrinsic.
  void collectNonVectorizedAndSetWideningDecisions(ElementCount VF) {
    // Do the analysis once.
    if (VF.isScalar() || Uniforms.contains(VF))
      return;
    setCostBasedWideningDecision(VF);
    collectLoopUniforms(VF);
    setVectorizedCallDecision(VF);
    collectLoopScalars(VF);
    collectInstsToScalarize(VF);
  }

  /// Returns true if the target machine supports masked store operation
  /// for the given \p DataType and kind of access to \p Ptr.
  bool isLegalMaskedStore(Type *DataType, Value *Ptr, Align Alignment,
                          unsigned AddressSpace) const;

  /// Returns true if the target machine supports masked load operation
  /// for the given \p DataType and kind of access to \p Ptr.
  bool isLegalMaskedLoad(Type *DataType, Value *Ptr, Align Alignment,
                         unsigned AddressSpace) const;

  /// Returns true if the target machine can represent \p V as a masked gather
  /// or scatter operation.
  bool isLegalGatherOrScatter(Value *V, ElementCount VF) {
    bool LI = isa<LoadInst>(V);
    bool SI = isa<StoreInst>(V);
    if (!LI && !SI)
      return false;
    auto *Ty = getLoadStoreType(V);
    Align Align = getLoadStoreAlignment(V);
    if (VF.isVector())
      Ty = VectorType::get(Ty, VF);
    return (LI && TTI.isLegalMaskedGather(Ty, Align)) ||
           (SI && TTI.isLegalMaskedScatter(Ty, Align));
  }

  /// Returns true if the target machine supports all of the reduction
  /// variables found for the given VF.
  bool canVectorizeReductions(ElementCount VF) const;

  /// Given costs for both strategies, return true if the scalar predication
  /// lowering should be used for div/rem.  This incorporates an override
  /// option so it is not simply a cost comparison.
  bool isDivRemScalarWithPredication(InstructionCost ScalarCost,
                                     InstructionCost SafeDivisorCost) const {
    switch (ForceSafeDivisor) {
    case cl::BOU_UNSET:
      return ScalarCost < SafeDivisorCost;
    case cl::BOU_TRUE:
      return false;
    case cl::BOU_FALSE:
      return true;
    }
    llvm_unreachable("impossible case value");
  }

  /// Returns true if \p I is an instruction which requires predication and
  /// for which our chosen predication strategy is scalarization (i.e. we
  /// don't have an alternate strategy such as masking available).
  /// \p VF is the vectorization factor that will be used to vectorize \p I.
  bool isScalarWithPredication(Instruction *I, ElementCount VF) const;

  /// Returns true if \p I is an instruction that needs to be predicated
  /// at runtime.  The result is independent of the predication mechanism.
  /// Superset of instructions that return true for isScalarWithPredication.
  bool isPredicatedInst(Instruction *I) const;

  /// Return the costs for our two available strategies for lowering a
  /// div/rem operation which requires speculating at least one lane.
  /// First result is for scalarization (will be invalid for scalable
  /// vectors); second is for the safe-divisor strategy.
  std::pair<InstructionCost, InstructionCost>
  getDivRemSpeculationCost(Instruction *I, ElementCount VF) const;

  /// Returns true if \p I is a memory instruction with consecutive memory
  /// access that can be widened.
  bool memoryInstructionCanBeWidened(Instruction *I, ElementCount VF);

  /// Returns true if \p I is a memory instruction in an interleaved-group
  /// of memory accesses that can be vectorized with wide vector loads/stores
  /// and shuffles.
  bool interleavedAccessCanBeWidened(Instruction *I, ElementCount VF) const;

  /// Check if \p Instr belongs to any interleaved access group.
  bool isAccessInterleaved(Instruction *Instr) const {
    return InterleaveInfo.isInterleaved(Instr);
  }

  /// Get the interleaved access group that \p Instr belongs to.
  const InterleaveGroup<Instruction> *
  getInterleavedAccessGroup(Instruction *Instr) const {
    return InterleaveInfo.getInterleaveGroup(Instr);
  }

  /// Returns true if we're required to use a scalar epilogue for at least
  /// the final iteration of the original loop.
  bool requiresScalarEpilogue(bool IsVectorizing) const;

  /// Returns true if a scalar epilogue is not allowed due to optsize or a
  /// loop hint annotation.
  bool isScalarEpilogueAllowed() const {
    return ScalarEpilogueStatus == CM_ScalarEpilogueAllowed;
  }

  /// Returns the TailFoldingStyle that is best for the current loop.
  TailFoldingStyle getTailFoldingStyle(bool IVUpdateMayOverflow = true) const {
    if (!ChosenTailFoldingStyle)
      return TailFoldingStyle::None;
    return IVUpdateMayOverflow ? ChosenTailFoldingStyle->first
                               : ChosenTailFoldingStyle->second;
  }

  /// Selects and saves TailFoldingStyle for 2 options - if IV update may
  /// overflow or not.
  /// \param IsScalableVF true if scalable vector factors enabled.
  /// \param UserIC User specific interleave count.
  void setTailFoldingStyles(bool IsScalableVF, unsigned UserIC);

  /// Returns true if all loop blocks should be masked to fold tail loop.
  bool foldTailByMasking() const {
    // TODO: check if it is possible to check for None style independent of
    // IVUpdateMayOverflow flag in getTailFoldingStyle.
    return getTailFoldingStyle() != TailFoldingStyle::None;
  }

  /// Return maximum safe number of elements to be processed per vector
  /// iteration, which do not prevent store-load forwarding and are safe with
  /// regard to the memory dependencies. Required for EVL-based VPlans to
  /// correctly calculate AVL (application vector length) as min(remaining AVL,
  /// MaxSafeElements).
  /// TODO: need to consider adjusting cost model to use this value as a
  /// vectorization factor for EVL-based vectorization.
  std::optional<unsigned> getMaxSafeElements() const { return MaxSafeElements; }

  /// Returns true if the instructions in this block requires predication
  /// for any reason, e.g. because tail folding now requires a predicate
  /// or because the block in the original loop was predicated.
  bool blockNeedsPredicationForAnyReason(BasicBlock *BB) const;

  /// Returns true if VP intrinsics with explicit vector length support should
  /// be generated in the tail folded loop.
  bool foldTailWithEVL() const {
    return getTailFoldingStyle() == TailFoldingStyle::DataWithEVL;
  }

  /// Returns true if the Phi is part of an inloop reduction.
  bool isInLoopReduction(PHINode *Phi) const {
    return InLoopReductions.contains(Phi);
  }

  /// Returns true if the predicated reduction select should be used to set the
  /// incoming value for the reduction phi.
  bool usePredicatedReductionSelect() const {
    // Force to use predicated reduction select since the EVL of the
    // second-to-last iteration might not be VF*UF.
    if (foldTailWithEVL())
      return true;
    return PreferPredicatedReductionSelect ||
           TTI.preferPredicatedReductionSelect();
  }

  /// Estimate cost of an intrinsic call instruction CI if it were vectorized
  /// with factor VF.  Return the cost of the instruction, including
  /// scalarization overhead if it's needed.
  InstructionCost getVectorIntrinsicCost(CallInst *CI, ElementCount VF) const;

  /// Estimate cost of a call instruction CI if it were vectorized with factor
  /// VF. Return the cost of the instruction, including scalarization overhead
  /// if it's needed.
  InstructionCost getVectorCallCost(CallInst *CI, ElementCount VF) const;

  /// Invalidates decisions already taken by the cost model.
  void invalidateCostModelingDecisions() {
    WideningDecisions.clear();
    CallWideningDecisions.clear();
    Uniforms.clear();
    Scalars.clear();
  }

  /// Returns the expected execution cost. The unit of the cost does
  /// not matter because we use the 'cost' units to compare different
  /// vector widths. The cost that is returned is *not* normalized by
  /// the factor width.
  InstructionCost expectedCost(ElementCount VF);

  bool hasPredStores() const { return NumPredStores > 0; }

  /// Returns true if epilogue vectorization is considered profitable, and
  /// false otherwise.
  /// \p VF is the vectorization factor chosen for the original loop.
  /// \p Multiplier is an aditional scaling factor applied to VF before
  /// comparing to EpilogueVectorizationMinVF.
  bool isEpilogueVectorizationProfitable(const ElementCount VF,
                                         const unsigned IC) const;

  /// Returns the execution time cost of an instruction for a given vector
  /// width. Vector width of one means scalar.
  InstructionCost getInstructionCost(Instruction *I, ElementCount VF);

  /// Return the cost of instructions in an inloop reduction pattern, if I is
  /// part of that pattern.
  std::optional<InstructionCost> getReductionPatternCost(Instruction *I,
                                                         ElementCount VF,
                                                         Type *VectorTy) const;

  /// Returns true if \p Op should be considered invariant and if it is
  /// trivially hoistable.
  bool shouldConsiderInvariant(Value *Op);

  /// Return the value of vscale used for tuning the cost model.
  std::optional<unsigned> getVScaleForTuning() const { return VScaleForTuning; }

private:
  unsigned NumPredStores = 0;

  /// Used to store the value of vscale used for tuning the cost model. It is
  /// initialized during object construction.
  std::optional<unsigned> VScaleForTuning;

  /// Initializes the value of vscale used for tuning the cost model. If
  /// vscale_range.min == vscale_range.max then return vscale_range.max, else
  /// return the value returned by the corresponding TTI method.
  void initializeVScaleForTuning() {
    const Function *Fn = TheLoop->getHeader()->getParent();
    if (Fn->hasFnAttribute(Attribute::VScaleRange)) {
      auto Attr = Fn->getFnAttribute(Attribute::VScaleRange);
      auto Min = Attr.getVScaleRangeMin();
      auto Max = Attr.getVScaleRangeMax();
      if (Max && Min == Max) {
        VScaleForTuning = Max;
        return;
      }
    }

    VScaleForTuning = TTI.getVScaleForTuning();
  }

  /// \return An upper bound for the vectorization factors for both
  /// fixed and scalable vectorization, where the minimum-known number of
  /// elements is a power-of-2 larger than zero. If scalable vectorization is
  /// disabled or unsupported, then the scalable part will be equal to
  /// ElementCount::getScalable(0).
  FixedScalableVFPair computeFeasibleMaxVF(unsigned MaxTripCount,
                                           ElementCount UserVF,
                                           bool FoldTailByMasking);

  /// If \p VF > MaxTripcount, clamps it to the next lower VF that is <=
  /// MaxTripCount.
  ElementCount clampVFByMaxTripCount(ElementCount VF, unsigned MaxTripCount,
                                     bool FoldTailByMasking) const;

  /// \return the maximized element count based on the targets vector
  /// registers and the loop trip-count, but limited to a maximum safe VF.
  /// This is a helper function of computeFeasibleMaxVF.
  ElementCount getMaximizedVFForTarget(unsigned MaxTripCount,
                                       unsigned SmallestType,
                                       unsigned WidestType,
                                       ElementCount MaxSafeVF,
                                       bool FoldTailByMasking);

  /// Checks if scalable vectorization is supported and enabled. Caches the
  /// result to avoid repeated debug dumps for repeated queries.
  bool isScalableVectorizationAllowed();

  /// \return the maximum legal scalable VF, based on the safe max number
  /// of elements.
  ElementCount getMaxLegalScalableVF(unsigned MaxSafeElements);

  /// Calculate vectorization cost of memory instruction \p I.
  InstructionCost getMemoryInstructionCost(Instruction *I, ElementCount VF);

  /// The cost computation for scalarized memory instruction.
  InstructionCost getMemInstScalarizationCost(Instruction *I, ElementCount VF);

  /// The cost computation for interleaving group of memory instructions.
  InstructionCost getInterleaveGroupCost(Instruction *I, ElementCount VF);

  /// The cost computation for Gather/Scatter instruction.
  InstructionCost getGatherScatterCost(Instruction *I, ElementCount VF);

  /// The cost computation for widening instruction \p I with consecutive
  /// memory access.
  InstructionCost getConsecutiveMemOpCost(Instruction *I, ElementCount VF);

  /// The cost calculation for Load/Store instruction \p I with uniform pointer
  /// - Load: scalar load + broadcast. Store: scalar store + (loop invariant
  /// value stored? 0 : extract of last element)
  InstructionCost getUniformMemOpCost(Instruction *I, ElementCount VF);

  /// Estimate the overhead of scalarizing an instruction. This is a
  /// convenience wrapper for the type-based getScalarizationOverhead API.
  InstructionCost getScalarizationOverhead(Instruction *I,
                                           ElementCount VF) const;

  /// Returns true if an artificially high cost for emulated masked memrefs
  /// should be used.
  bool useEmulatedMaskMemRefHack(Instruction *I, ElementCount VF);

  /// Map of scalar integer values to the smallest bitwidth they can be legally
  /// represented as. The vector equivalents of these values should be truncated
  /// to this type.
  MapVector<Instruction *, uint64_t> MinBWs;

  /// A type representing the costs for instructions if they were to be
  /// scalarized rather than vectorized. The entries are Instruction-Cost
  /// pairs.
  using ScalarCostsTy = MapVector<Instruction *, InstructionCost>;

  /// A set containing all BasicBlocks that are known to present after
  /// vectorization as a predicated block.
  DenseMap<ElementCount, SmallPtrSet<BasicBlock *, 4>>
      PredicatedBBsAfterVectorization;

  /// Records whether it is allowed to have the original scalar loop execute at
  /// least once. This may be needed as a fallback loop in case runtime
  /// aliasing/dependence checks fail, or to handle the tail/remainder
  /// iterations when the trip count is unknown or doesn't divide by the VF,
  /// or as a peel-loop to handle gaps in interleave-groups.
  /// Under optsize and when the trip count is very small we don't allow any
  /// iterations to execute in the scalar loop.
  ScalarEpilogueLowering ScalarEpilogueStatus = CM_ScalarEpilogueAllowed;

  /// Control finally chosen tail folding style. The first element is used if
  /// the IV update may overflow, the second element - if it does not.
  std::optional<std::pair<TailFoldingStyle, TailFoldingStyle>>
      ChosenTailFoldingStyle;

  /// true if scalable vectorization is supported and enabled.
  std::optional<bool> IsScalableVectorizationAllowed;

  /// Maximum safe number of elements to be processed per vector iteration,
  /// which do not prevent store-load forwarding and are safe with regard to the
  /// memory dependencies. Required for EVL-based veectorization, where this
  /// value is used as the upper bound of the safe AVL.
  std::optional<unsigned> MaxSafeElements;

  /// A map holding scalar costs for different vectorization factors. The
  /// presence of a cost for an instruction in the mapping indicates that the
  /// instruction will be scalarized when vectorizing with the associated
  /// vectorization factor. The entries are VF-ScalarCostTy pairs.
  MapVector<ElementCount, ScalarCostsTy> InstsToScalarize;

  /// Holds the instructions known to be uniform after vectorization.
  /// The data is collected per VF.
  DenseMap<ElementCount, SmallPtrSet<Instruction *, 4>> Uniforms;

  /// Holds the instructions known to be scalar after vectorization.
  /// The data is collected per VF.
  DenseMap<ElementCount, SmallPtrSet<Instruction *, 4>> Scalars;

  /// Holds the instructions (address computations) that are forced to be
  /// scalarized.
  DenseMap<ElementCount, SmallPtrSet<Instruction *, 4>> ForcedScalars;

  /// PHINodes of the reductions that should be expanded in-loop.
  SmallPtrSet<PHINode *, 4> InLoopReductions;

  /// A Map of inloop reduction operations and their immediate chain operand.
  /// FIXME: This can be removed once reductions can be costed correctly in
  /// VPlan. This was added to allow quick lookup of the inloop operations.
  DenseMap<Instruction *, Instruction *> InLoopReductionImmediateChains;

  /// Returns the expected difference in cost from scalarizing the expression
  /// feeding a predicated instruction \p PredInst. The instructions to
  /// scalarize and their scalar costs are collected in \p ScalarCosts. A
  /// non-negative return value implies the expression will be scalarized.
  /// Currently, only single-use chains are considered for scalarization.
  InstructionCost computePredInstDiscount(Instruction *PredInst,
                                          ScalarCostsTy &ScalarCosts,
                                          ElementCount VF);

  /// Collect the instructions that are uniform after vectorization. An
  /// instruction is uniform if we represent it with a single scalar value in
  /// the vectorized loop corresponding to each vector iteration. Examples of
  /// uniform instructions include pointer operands of consecutive or
  /// interleaved memory accesses. Note that although uniformity implies an
  /// instruction will be scalar, the reverse is not true. In general, a
  /// scalarized instruction will be represented by VF scalar values in the
  /// vectorized loop, each corresponding to an iteration of the original
  /// scalar loop.
  void collectLoopUniforms(ElementCount VF);

  /// Collect the instructions that are scalar after vectorization. An
  /// instruction is scalar if it is known to be uniform or will be scalarized
  /// during vectorization. collectLoopScalars should only add non-uniform nodes
  /// to the list if they are used by a load/store instruction that is marked as
  /// CM_Scalarize. Non-uniform scalarized instructions will be represented by
  /// VF values in the vectorized loop, each corresponding to an iteration of
  /// the original scalar loop.
  void collectLoopScalars(ElementCount VF);

  /// Keeps cost model vectorization decision and cost for instructions.
  /// Right now it is used for memory instructions only.
  using DecisionList = DenseMap<std::pair<Instruction *, ElementCount>,
                                std::pair<InstWidening, InstructionCost>>;

  DecisionList WideningDecisions;

  using CallDecisionList =
      DenseMap<std::pair<CallInst *, ElementCount>, CallWideningDecision>;

  CallDecisionList CallWideningDecisions;

  /// Returns true if \p V is expected to be vectorized and it needs to be
  /// extracted.
  bool needsExtract(Value *V, ElementCount VF) const {
    Instruction *I = dyn_cast<Instruction>(V);
    if (VF.isScalar() || !I || !TheLoop->contains(I) ||
        TheLoop->isLoopInvariant(I) ||
        getWideningDecision(I, VF) == CM_Scalarize ||
        (isa<CallInst>(I) &&
         getCallWideningDecision(cast<CallInst>(I), VF).Kind == CM_Scalarize))
      return false;

    // Assume we can vectorize V (and hence we need extraction) if the
    // scalars are not computed yet. This can happen, because it is called
    // via getScalarizationOverhead from setCostBasedWideningDecision, before
    // the scalars are collected. That should be a safe assumption in most
    // cases, because we check if the operands have vectorizable types
    // beforehand in LoopVectorizationLegality.
    return !Scalars.contains(VF) || !isScalarAfterVectorization(I, VF);
  };

  /// Returns a range containing only operands needing to be extracted.
  SmallVector<Value *, 4> filterExtractingOperands(Instruction::op_range Ops,
                                                   ElementCount VF) const {

    SmallPtrSet<const Value *, 4> UniqueOperands;
    SmallVector<Value *, 4> Res;
    for (Value *Op : Ops) {
      if (isa<Constant>(Op) || !UniqueOperands.insert(Op).second ||
          !needsExtract(Op, VF))
        continue;
      Res.push_back(Op);
    }
    return Res;
  }

public:
  /// The loop that we evaluate.
  Loop *TheLoop;

  /// Predicated scalar evolution analysis.
  PredicatedScalarEvolution &PSE;

  /// Loop Info analysis.
  LoopInfo *LI;

  /// Vectorization legality.
  LoopVectorizationLegality *Legal;

  /// Vector target information.
  const TargetTransformInfo &TTI;

  /// Target Library Info.
  const TargetLibraryInfo *TLI;

  /// Demanded bits analysis.
  DemandedBits *DB;

  /// Assumption cache.
  AssumptionCache *AC;

  /// Interface to emit optimization remarks.
  OptimizationRemarkEmitter *ORE;

  const Function *TheFunction;

  /// Loop Vectorize Hint.
  const LoopVectorizeHints *Hints;

  /// The interleave access information contains groups of interleaved accesses
  /// with the same stride and close to each other.
  InterleavedAccessInfo &InterleaveInfo;

  /// Values to ignore in the cost model.
  SmallPtrSet<const Value *, 16> ValuesToIgnore;

  /// Values to ignore in the cost model when VF > 1.
  SmallPtrSet<const Value *, 16> VecValuesToIgnore;

  /// All element types found in the loop.
  SmallPtrSet<Type *, 16> ElementTypesInLoop;

  /// The kind of cost that we are calculating
  TTI::TargetCostKind CostKind;

  /// Whether this loop should be optimized for size based on function attribute
  /// or profile information.
  bool OptForSize;

  /// The highest VF possible for this loop, without using MaxBandwidth.
  FixedScalableVFPair MaxPermissibleVFWithoutMaxBW;
};
} // namespace llvm

#endif // LLVM_TRANSFORMS_VECTORIZE_LOOPVECTORIZATIONCOSTMODEL_H
