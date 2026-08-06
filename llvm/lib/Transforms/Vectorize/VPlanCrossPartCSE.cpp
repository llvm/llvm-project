//===- VPlanCrossPartCSE.cpp - Cross-part CSE for VPlan -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements exact load-overlap profitability analysis across two
// logical VPlan parts.
//
//===----------------------------------------------------------------------===//

#include "VPlanCrossPartCSE.h"
#include "VPlan.h"
#include "VPlanPatternMatch.h"
#include "VPlanUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <optional>

using namespace llvm;

#define DEBUG_TYPE "loop-vectorize"

namespace {

/// A simple widened load supported by the prediction-only analysis.
struct CrossPartSupportedLoad {
  /// VPlan recipe whose logical UF=2 instances are modeled.
  VPWidenLoadRecipe *Recipe;
  /// Underlying scalar load used for type, SCEV, and poison identity.
  LoadInst *Load;
};

/// Return an unmasked, non-EVL, consecutive simple widened load.
static std::optional<CrossPartSupportedLoad>
getCrossPartSupportedLoad(VPRecipeBase &R) {
  auto *Widen = dyn_cast<VPWidenLoadRecipe>(&R);
  if (!Widen || Widen->isMasked() || !Widen->isConsecutive())
    return std::nullopt;

  auto *Load = dyn_cast<LoadInst>(&Widen->getIngredient());
  if (!Load || !Load->isSimple())
    return std::nullopt;
  return CrossPartSupportedLoad{Widen, Load};
}

/// Build addresses only for provenance whose physical UF mapping is explicit.
class CrossPartAddressBuilder {
  /// Return the fixed width after validating the address model's precondition.
  static unsigned getFixedVF(ElementCount VF) {
    assert(!VF.isScalable() && "cross-part analysis requires a fixed VF");
    return VF.getFixedValue();
  }

  /// Predicated SCEV state carrying vectorization assumptions.
  PredicatedScalarEvolution &PSE;
  /// ScalarEvolution used for canonical exact identities.
  ScalarEvolution &SE;
  /// Original loop used to interpret loop-varying VPlan values.
  const Loop *OrigLoop;
  /// Exact fixed vector width used for Part * VF.
  const unsigned FixedVF;
  /// Base SCEVs cached by VPlan value for reuse across loads and parts.
  DenseMap<const VPValue *, const SCEV *> BaseSCEVs;

  /// Return the SCEV represented by \p V, caching it after first construction.
  const SCEV *getBaseSCEV(const VPValue *V) {
    auto It = BaseSCEVs.find(V);
    if (It != BaseSCEVs.end())
      return It->second;

    const SCEV *S = vputils::getSCEVExprForVPValue(V, PSE, OrigLoop);
    BaseSCEVs.try_emplace(V, S);
    return S;
  }

public:
  /// Bind the fixed VF, original loop, and predicated SCEV state.
  CrossPartAddressBuilder(ElementCount VF, PredicatedScalarEvolution &PSE,
                          const Loop *OrigLoop)
      : PSE(PSE), SE(*PSE.getSE()), OrigLoop(OrigLoop),
        FixedVF(getFixedVF(VF)) {}

  /// Return the exact address used by \p Load in logical part \p Part.
  const SCEV *getAddress(const CrossPartSupportedLoad &Load, unsigned Part) {
    assert(Part < CrossPartCSERequiredInterleaveCount &&
           "logical part must be zero or one");
    VPValue *Addr = Load.Recipe->getAddr();

    // VPVectorPointerRecipe is explicitly cloned with a Part * VF offset by
    // VPlanUnroll. Reproduce only that recipe-specific physical rewrite.
    auto *VectorPtr = dyn_cast<VPVectorPointerRecipe>(Addr);
    if (!VectorPtr)
      return SE.getCouldNotCompute();

    // TODO: Generalize the part offset to VFxPart * Stride when adding complete
    // supported load shapes. Until then, accept only a literal unit stride so
    // the simplified Part * VF offset remains exact.
    using namespace VPlanPatternMatch;
    if (!match(VectorPtr->getStride(), m_One()))
      return SE.getCouldNotCompute();

    const SCEV *Base = getBaseSCEV(VectorPtr->getOperand(0));
    if (isa<SCEVCouldNotCompute>(Base))
      return SE.getCouldNotCompute();
    if (Part == 0)
      return Base;

    Type *IndexTy = SE.getDataLayout().getIndexType(VectorPtr->getScalarType());
    const SCEV *Offset = SE.getConstant(IndexTy, uint64_t(Part) * FixedVF);
    // Keep the synthetic address conservative: ScalarEvolution imports GEP
    // nowrap facts only after accounting for their poison semantics. AddExpr
    // uniquing still recognizes equal operands without those facts.
    return SE.getGEPExpr(Base, {Offset}, VectorPtr->getSourceElementType());
  }
};

/// Key for exact load equality after annotated loads have been excluded.
struct CrossPartLoadKey {
  /// Canonical SCEV address for this logical load instance.
  const SCEV *Address;
  /// Loaded scalar type required for value compatibility.
  Type *ValueType;
};

/// DenseMap policy for exact canonical load keys.
struct CrossPartLoadKeyInfo {
  /// Hash every property required by exact load equality.
  static unsigned getHashValue(const CrossPartLoadKey &Key) {
    return hash_combine(Key.Address, Key.ValueType);
  }

  /// Compare every property required by exact load equality.
  static bool isEqual(const CrossPartLoadKey &A, const CrossPartLoadKey &B) {
    return A.Address == B.Address && A.ValueType == B.ValueType;
  }
};

/// Return whether \p R may write memory during VPlan execution.
static bool isCrossPartWrite(const VPRecipeBase &R) {
  // VPVectorEndPointerRecipe is pure but inherits the conservative memory
  // default. This local exception prevents its address computation from being
  // mistaken for a write without changing global recipe memory behavior.
  switch (R.getVPRecipeID()) {
  case VPRecipeBase::VPVectorEndPointerSC:
    return false;
  default:
    return R.mayWriteToMemory();
  }
}

/// Return whether \p Plan keeps the canonical IV increment in the symbolic
/// VF * UF form required to model consecutive logical parts.
static bool hasCanonicalIVIncrementForCrossPartCSE(VPlan &Plan) {
  return !Plan.getVFxUF().isMaterialized() &&
         vputils::findCanonicalIVIncrement(Plan);
}

} // namespace

bool llvm::isCrossPartCSEProfitable(VPlan &Plan, ElementCount VF,
                                    InstructionCost LoopCost,
                                    const Loop *OrigLoop,
                                    PredicatedScalarEvolution &PSE,
                                    const VPRecipeCostMap &RecipeCosts,
                                    const CrossPartCSEOptions &Options) {
  // Reject a partially initialized policy before its sentinel values can
  // participate in saturating cost arithmetic.
  if (Options.MinSavingPct == CrossPartCSEOptions::Unspecified ||
      Options.MinOpportunities == CrossPartCSEOptions::Unspecified)
    return false;

  VPRegionBlock *LoopRegion = Plan.getVectorLoopRegion();
  if (!LoopRegion)
    return false;

  // Fail closed for every shape outside the exact fixed-width, single-block
  // UF=2 model.
  // TODO: Expand coverage by accepting additional plan shapes once their
  // cross-part semantics can be modeled exactly.
  if (VF.isScalable() || !VF.isVector() || !LoopCost.isValid() ||
      LoopCost <= 0 || !OrigLoop->isInnermost() ||
      OrigLoop->getNumBlocks() != 1 ||
      LoopRegion->getEntryBasicBlock() != LoopRegion->getExitingBasicBlock() ||
      !Plan.hasUF(CrossPartCSERequiredInterleaveCount) || Plan.isUnrolled() ||
      !hasCanonicalIVIncrementForCrossPartCSE(Plan))
    return false;

  using AvailableLoadMap =
      DenseMap<CrossPartLoadKey, unsigned, CrossPartLoadKeyInfo>;
  AvailableLoadMap AvailableLoadParts;
  CrossPartAddressBuilder Addresses(VF, PSE, OrigLoop);
  unsigned NumOpportunities = 0;
  InstructionCost SavedCost = 0;

  // Match VPlanUnroll's recipe-major UF=2 order. Clearing on every write
  // enforces a strict no-write interval without alias disambiguation.
  for (VPRecipeBase &R : *LoopRegion->getEntryBasicBlock()) {
    if (isCrossPartWrite(R)) {
      AvailableLoadParts.clear();
      continue;
    }

    std::optional<CrossPartSupportedLoad> Load = getCrossPartSupportedLoad(R);
    if (!Load)
      continue;

    // Conservatively reject annotations whose poison behavior requires the
    // downstream realization to combine metadata across the load pair.
    // TODO: Accept compatible annotated pairs once realization supports the
    // required CSE metadata merge.
    if (Load->Load->hasPoisonGeneratingAnnotations())
      continue;

    for (unsigned Part = 0; Part != CrossPartCSERequiredInterleaveCount;
         ++Part) {
      const SCEV *Address = Addresses.getAddress(*Load, Part);
      if (isa<SCEVCouldNotCompute>(Address))
        continue;

      CrossPartLoadKey Key = {Address, Load->Load->getType()};
      // Only reuse between different logical parts can justify raising IC from
      // 1 to 2. A duplicate already seen in the same part also exists at IC=1
      // and therefore provides no interleaving-specific saving.
      unsigned PartBit = 1U << Part;
      unsigned &AvailableParts = AvailableLoadParts[Key];
      if (AvailableParts & PartBit)
        continue;

      bool HasOppositePart = (AvailableParts & ~PartBit) != 0;
      AvailableParts |= PartBit;
      if (!HasOppositePart) {
        // Record the first occurrence in this part without assigning
        // cross-part credit.
        continue;
      }

      auto CostIt = RecipeCosts.find(Load->Recipe);
      // RecipeCosts records direct VPlan-owned costs only. A zero entry may
      // mean that the underlying instruction was charged or ignored elsewhere,
      // so it cannot establish independently removable work.
      // TODO: Before crediting non-load congruence, model legacy-owned costs
      // with explicit per-occurrence ownership and deduplication.
      if (CostIt == RecipeCosts.end() || !CostIt->second.isValid() ||
          CostIt->second <= 0)
        continue;
      SavedCost += CostIt->second;
      ++NumOpportunities;
    }
  }

  bool MeetsOpportunityThreshold =
      NumOpportunities >= Options.MinOpportunities && SavedCost > 0;
  using CostType = InstructionCost::CostType;
  bool Select = false;
  if (MeetsOpportunityThreshold) {
    // Use InstructionCost arithmetic to preserve fractional cost units.
    InstructionCost ScaledSavedCost = SavedCost * CostType(100);
    InstructionCost RequiredCost =
        LoopCost * CostType(CrossPartCSERequiredInterleaveCount);
    RequiredCost *= CostType(Options.MinSavingPct);
    Select = ScaledSavedCost >= RequiredCost;
  }

  LLVM_DEBUG({
    CostType SavingPct = 0;
    if (SavedCost.isValid() && SavedCost > 0 && LoopCost.isValid() &&
        LoopCost > 0)
      SavingPct = ((SavedCost * CostType(100)) /
                   (LoopCost * CostType(CrossPartCSERequiredInterleaveCount)))
                      .getValue();
    dbgs() << "LV: Cross-part load overlap estimate: ops=" << NumOpportunities
           << ", required-ops=" << Options.MinOpportunities
           << ", predicted-saved-cost=" << SavedCost
           << ", loop-cost=" << LoopCost << ", saving=" << SavingPct
           << "%, required=" << Options.MinSavingPct << "%; "
           << (Select ? "selecting IC=2" : "skipping") << ".\n";
  });
  return Select;
}
