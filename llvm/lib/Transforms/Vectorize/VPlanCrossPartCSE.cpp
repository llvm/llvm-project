//===- VPlanCrossPartCSE.cpp - Cross-part CSE for VPlan -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements exact load-redundancy profitability analysis across two
// logical VPlan parts.
//
//===----------------------------------------------------------------------===//

#include "VPlanCrossPartCSE.h"
#include "VPlan.h"
#include "VPlanHelpers.h"
#include "VPlanPatternMatch.h"
#include "VPlanUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "loop-vectorize"

namespace {

/// Return an unmasked, non-EVL, consecutive widened load.
static VPWidenLoadRecipe *getCrossPartSupportedLoad(VPRecipeBase &R) {
  auto *Load = dyn_cast<VPWidenLoadRecipe>(&R);
  if (!Load || Load->isMasked() || !Load->isConsecutive())
    return nullptr;
  return Load;
}

/// Build addresses only for provenance whose physical UF mapping is explicit.
class CrossPartAddressBuilder {
  /// Predicated SCEV state carrying vectorization assumptions.
  PredicatedScalarEvolution &PSE;
  /// ScalarEvolution used for canonical exact identities.
  ScalarEvolution &SE;
  /// Original loop used to interpret loop-varying VPlan values.
  const Loop *OrigLoop;
  /// Vector factor used to model the exact per-part offset.
  const ElementCount VF;
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

  /// Return a conservative GEP expression for \p Base + \p Offset.
  const SCEV *getGEPAddress(const SCEV *Base, const SCEV *Offset,
                            Type *SourceElementTy) {
    // ScalarEvolution imports GEP nowrap facts only after accounting for their
    // poison semantics. AddExpr uniquing still recognizes equal operands
    // without adding those facts to the synthetic expression.
    return SE.getGEPExpr(Base, {Offset}, SourceElementTy);
  }

  /// Return the physical address produced by \p VectorPtr for \p Part.
  const SCEV *getForwardAddress(VPVectorPointerRecipe &VectorPtr,
                                unsigned Part) {
    // VPlanUnroll models a forward part as Base + Part * VF * Stride. Accept
    // only unit stride until the analysis supports the complete expression.
    using namespace VPlanPatternMatch;
    if (!match(VectorPtr.getStride(), m_One()))
      return SE.getCouldNotCompute();

    const SCEV *Base = getBaseSCEV(VectorPtr.getOperand(0));
    if (isa<SCEVCouldNotCompute>(Base))
      return SE.getCouldNotCompute();
    if (Part == 0)
      return Base;

    Type *IndexTy = SE.getDataLayout().getIndexType(VectorPtr.getScalarType());
    const SCEV *Offset = SE.getElementCount(IndexTy, VF * Part);
    return getGEPAddress(Base, Offset, VectorPtr.getSourceElementType());
  }

  /// Return the physical address produced by \p EndPtr for \p Part.
  const SCEV *getReverseAddress(VPVectorEndPointerRecipe &EndPtr,
                                unsigned Part) {
    const SCEV *Base = getBaseSCEV(EndPtr.getPointer());
    if (isa<SCEVCouldNotCompute>(Base))
      return SE.getCouldNotCompute();

    Type *IndexTy = SE.getDataLayout().getIndexType(EndPtr.getScalarType());
    const SCEV *VFExpr = SE.getElementCount(IndexTy, VF);
    const SCEV *Stride =
        SE.getConstant(IndexTy, EndPtr.getStride(), /*isSigned=*/true);

    // Mirror VPVectorEndPointerRecipe::materializeOffset:
    //   Stride * (VF - 1) + Part * Stride * VF.
    const SCEV *Offset0 =
        SE.getMulExpr(SE.getMinusSCEV(VFExpr, SE.getOne(IndexTy)), Stride);
    int64_t PartStride = static_cast<int64_t>(Part) * EndPtr.getStride();
    const SCEV *PartOffset = SE.getMulExpr(
        SE.getConstant(IndexTy, PartStride, /*isSigned=*/true), VFExpr);
    const SCEV *Offset = SE.getAddExpr(Offset0, PartOffset);
    return getGEPAddress(Base, Offset, EndPtr.getSourceElementType());
  }

public:
  /// Bind the VF, original loop, and predicated SCEV state.
  CrossPartAddressBuilder(ElementCount VF, PredicatedScalarEvolution &PSE,
                          const Loop *OrigLoop)
      : PSE(PSE), SE(*PSE.getSE()), OrigLoop(OrigLoop), VF(VF) {}

  /// Return the exact address used by \p Load in logical part \p Part.
  const SCEV *getAddress(VPWidenLoadRecipe &Load, unsigned Part) {
    assert(Part < CrossPartCSERequiredInterleaveCount &&
           "logical part must be zero or one");
    VPValue *Addr = Load.getAddr();

    // Reproduce only the recipe-specific physical rewrites performed by
    // VPlanUnroll for consecutive forward and reverse accesses.
    auto *VectorPtr = dyn_cast<VPVectorPointerRecipe>(Addr);
    if (VectorPtr)
      return getForwardAddress(*VectorPtr, Part);
    auto *EndPtr = dyn_cast<VPVectorEndPointerRecipe>(Addr);
    if (EndPtr)
      return getReverseAddress(*EndPtr, Part);

    return SE.getCouldNotCompute();
  }
};

/// Key for exact value equality of two logical widened-load instances.
/// Metadata and alignment are not part of value identity. A CSE implementation
/// must intersect retained metadata and preserve an alignment sufficient for
/// every replaced load.
/// Poison-generating source metadata is not propagated to widened loads.
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
  // TODO: Classify VPVectorEndPointerRecipe as non-memory in
  // VPRecipeBase::mayReadFromMemory() and mayWriteToMemory(), then remove this
  // exception. The shared fix can expose new VPlan CSE opportunities and needs
  // dedicated code-generation tests.
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
  return vputils::findCanonicalIVIncrement(Plan);
}

} // namespace

bool llvm::isCrossPartCSEProfitable(VPlan &Plan, ElementCount VF,
                                    InstructionCost LoopCost,
                                    VPCostContext &CostCtx,
                                    const CrossPartCSEOptions &Options) {
  assert(VF.isVector() && "cross-part analysis requires a vector VF");
  assert(CostCtx.L && CostCtx.L->isInnermost() &&
         "cross-part analysis requires an innermost loop");
  assert(Plan.hasUF(CrossPartCSERequiredInterleaveCount) &&
         "cross-part analysis requires support for UF=2");
  assert(!Plan.isUnrolled() && "cross-part analysis requires symbolic UF");

  // Narrowed plans replace symbolic VF * UF with a different effective step,
  // so the selected VF no longer describes their physical per-part offset.
  if (Plan.getVFxUF().isMaterialized())
    return false;

  // Reject an unspecified or impossible percentage before cost arithmetic.
  if (Options.MinSavingPct == CrossPartCSEOptions::Unspecified ||
      Options.MinSavingPct > 100)
    return false;

  VPRegionBlock *LoopRegion = Plan.getVectorLoopRegion();
  if (!LoopRegion)
    return false;

  // Fail closed for every shape outside the exact single-block UF=2 model.
  // TODO: Expand coverage by accepting additional plan shapes once their
  // cross-part semantics can be modeled exactly.
  if (!LoopCost.isValid() || LoopCost <= 0 ||
      LoopRegion->getEntryBasicBlock() != LoopRegion->getExitingBasicBlock() ||
      !hasCanonicalIVIncrementForCrossPartCSE(Plan))
    return false;

  using AvailableLoadMap =
      DenseMap<CrossPartLoadKey, unsigned, CrossPartLoadKeyInfo>;
  AvailableLoadMap AvailableLoadParts;
  // A recipe may participate in multiple logical matches as supported shapes
  // expand, but its local saving estimate is computed at most once.
  DenseMap<const VPRecipeBase *, InstructionCost> SavingCosts;
  CrossPartAddressBuilder Addresses(VF, CostCtx.PSE, CostCtx.L);
#ifndef NDEBUG
  // Count redundant-load opportunities only for diagnostics; profitability
  // uses SavedCost.
  unsigned NumOpportunities = 0;
#endif
  InstructionCost SavedCost = 0;

  // Match VPlanUnroll's recipe-major UF=2 order. Clearing on every write
  // enforces a strict no-write interval without alias disambiguation.
  for (VPRecipeBase &R : *LoopRegion->getEntryBasicBlock()) {
    if (isCrossPartWrite(R)) {
      AvailableLoadParts.clear();
      continue;
    }

    VPWidenLoadRecipe *Load = getCrossPartSupportedLoad(R);
    if (!Load)
      continue;

    for (unsigned Part = 0; Part != CrossPartCSERequiredInterleaveCount;
         ++Part) {
      const SCEV *Address = Addresses.getAddress(*Load, Part);
      if (isa<SCEVCouldNotCompute>(Address))
        continue;

      CrossPartLoadKey Key = {Address, Load->getScalarType()};
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

      auto CostIt = SavingCosts.find(Load);
      if (CostIt == SavingCosts.end())
        CostIt = SavingCosts.try_emplace(Load, Load->cost(VF, CostCtx)).first;

      // This estimate intentionally avoids retaining cost state from VF
      // selection. It is exact for the directly costed widened loads supported
      // here, but does not reproduce legacy attribution included in LoopCost.
      // TODO: If measured profitability loses accuracy as supported recipes
      // expand, consider passing cached costs from the selected-VF cost run.
      // That would restore exact attribution at the cost of cross-phase state
      // and recipe-lifetime management.
      if (!CostIt->second.isValid() || CostIt->second <= 0)
        continue;
      SavedCost += CostIt->second;
      LLVM_DEBUG(++NumOpportunities);
    }
  }

  using CostType = InstructionCost::CostType;
  bool Select = false;
  if (SavedCost > 0) {
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
    dbgs() << "LV: Cross-part load redundancy estimate: opportunities="
           << NumOpportunities << ", predicted-saved-cost=" << SavedCost
           << ", loop-cost=" << LoopCost << ", saving=" << SavingPct
           << "%, required=" << Options.MinSavingPct << "%; "
           << (Select ? "selecting IC=2" : "skipping") << ".\n";
  });
  return Select;
}
