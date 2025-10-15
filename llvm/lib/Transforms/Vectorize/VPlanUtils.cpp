//===- VPlanUtils.cpp - VPlan-related utilities ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "VPlanUtils.h"
#include "VPlanCFG.h"
#include "VPlanPatternMatch.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"

using namespace llvm;
using namespace llvm::VPlanPatternMatch;

bool vputils::onlyFirstLaneUsed(const VPValue *Def) {
  return all_of(Def->users(),
                [Def](const VPUser *U) { return U->onlyFirstLaneUsed(Def); });
}

bool vputils::onlyFirstPartUsed(const VPValue *Def) {
  return all_of(Def->users(),
                [Def](const VPUser *U) { return U->onlyFirstPartUsed(Def); });
}

bool vputils::onlyScalarValuesUsed(const VPValue *Def) {
  return all_of(Def->users(),
                [Def](const VPUser *U) { return U->usesScalars(Def); });
}

VPValue *vputils::getOrCreateVPValueForSCEVExpr(VPlan &Plan, const SCEV *Expr) {
  if (auto *Expanded = Plan.getSCEVExpansion(Expr))
    return Expanded;
  VPValue *Expanded = nullptr;
  if (auto *E = dyn_cast<SCEVConstant>(Expr))
    Expanded = Plan.getOrAddLiveIn(E->getValue());
  else {
    auto *U = dyn_cast<SCEVUnknown>(Expr);
    // Skip SCEV expansion if Expr is a SCEVUnknown wrapping a non-instruction
    // value. Otherwise the value may be defined in a loop and using it directly
    // will break LCSSA form. The SCEV expansion takes care of preserving LCSSA
    // form.
    if (U && !isa<Instruction>(U->getValue())) {
      Expanded = Plan.getOrAddLiveIn(U->getValue());
    } else {
      Expanded = new VPExpandSCEVRecipe(Expr);
      Plan.getEntry()->appendRecipe(Expanded->getDefiningRecipe());
    }
  }
  Plan.addSCEVExpansion(Expr, Expanded);
  return Expanded;
}

bool vputils::isHeaderMask(const VPValue *V, VPlan &Plan) {
  if (isa<VPActiveLaneMaskPHIRecipe>(V))
    return true;

  auto IsWideCanonicalIV = [](VPValue *A) {
    return isa<VPWidenCanonicalIVRecipe>(A) ||
           (isa<VPWidenIntOrFpInductionRecipe>(A) &&
            cast<VPWidenIntOrFpInductionRecipe>(A)->isCanonical());
  };

  VPValue *A, *B;

  if (match(V, m_ActiveLaneMask(m_VPValue(A), m_VPValue(B), m_One())))
    return B == Plan.getTripCount() &&
           (match(A,
                  m_ScalarIVSteps(
                      m_Specific(Plan.getVectorLoopRegion()->getCanonicalIV()),
                      m_One(), m_Specific(&Plan.getVF()))) ||
            IsWideCanonicalIV(A));

  return match(V, m_ICmp(m_VPValue(A), m_VPValue(B))) && IsWideCanonicalIV(A) &&
         B == Plan.getOrCreateBackedgeTakenCount();
}

const SCEV *vputils::getSCEVExprForVPValue(VPValue *V, ScalarEvolution &SE) {
  if (V->isLiveIn()) {
    if (Value *LiveIn = V->getLiveInIRValue())
      return SE.getSCEV(LiveIn);
    return SE.getCouldNotCompute();
  }

  // TODO: Support constructing SCEVs for more recipes as needed.
  return TypeSwitch<const VPRecipeBase *, const SCEV *>(V->getDefiningRecipe())
      .Case<VPExpandSCEVRecipe>(
          [](const VPExpandSCEVRecipe *R) { return R->getSCEV(); })
      .Default([&SE](const VPRecipeBase *) { return SE.getCouldNotCompute(); });
}

bool vputils::isUniformAcrossVFsAndUFs(VPValue *V) {
  // Live-ins are uniform.
  if (V->isLiveIn())
    return true;

  VPRecipeBase *R = V->getDefiningRecipe();
  if (R && V->isDefinedOutsideLoopRegions()) {
    if (match(V->getDefiningRecipe(),
              m_VPInstruction<VPInstruction::CanonicalIVIncrementForPart>()))
      return false;
    return all_of(R->operands(), isUniformAcrossVFsAndUFs);
  }

  auto *CanonicalIV =
      R->getParent()->getEnclosingLoopRegion()->getCanonicalIV();
  // Canonical IV chain is uniform.
  if (V == CanonicalIV || V == CanonicalIV->getBackedgeValue())
    return true;

  return TypeSwitch<const VPRecipeBase *, bool>(R)
      .Case<VPDerivedIVRecipe>([](const auto *R) { return true; })
      .Case<VPReplicateRecipe>([](const auto *R) {
        // Loads and stores that are uniform across VF lanes are handled by
        // VPReplicateRecipe.IsUniform. They are also uniform across UF parts if
        // all their operands are invariant.
        // TODO: Further relax the restrictions.
        return R->isSingleScalar() &&
               (isa<LoadInst, StoreInst>(R->getUnderlyingValue())) &&
               all_of(R->operands(), isUniformAcrossVFsAndUFs);
      })
      .Case<VPInstruction>([](const auto *VPI) {
        return VPI->isScalarCast() &&
               isUniformAcrossVFsAndUFs(VPI->getOperand(0));
      })
      .Case<VPWidenCastRecipe>([](const auto *R) {
        // A cast is uniform according to its operand.
        return isUniformAcrossVFsAndUFs(R->getOperand(0));
      })
      .Default([](const VPRecipeBase *) { // A value is considered non-uniform
                                          // unless proven otherwise.
        return false;
      });
}

VPBasicBlock *vputils::getFirstLoopHeader(VPlan &Plan, VPDominatorTree &VPDT) {
  auto DepthFirst = vp_depth_first_shallow(Plan.getEntry());
  auto I = find_if(DepthFirst, [&VPDT](VPBlockBase *VPB) {
    return VPBlockUtils::isHeader(VPB, VPDT);
  });
  return I == DepthFirst.end() ? nullptr : cast<VPBasicBlock>(*I);
}

unsigned vputils::getVFScaleFactor(VPRecipeBase *R) {
  if (!R)
    return 1;
  if (auto *RR = dyn_cast<VPReductionPHIRecipe>(R))
    return RR->getVFScaleFactor();
  if (auto *RR = dyn_cast<VPPartialReductionRecipe>(R))
    return RR->getVFScaleFactor();
  assert(
      (!isa<VPInstruction>(R) || cast<VPInstruction>(R)->getOpcode() !=
                                     VPInstruction::ReductionStartVector) &&
      "getting scaling factor of reduction-start-vector not implemented yet");
  return 1;
}

std::optional<VPValue *>
vputils::getRecipesForUncountableExit(VPlan &Plan,
                                      SmallVectorImpl<VPRecipeBase *> &Recipes,
                                      SmallVectorImpl<VPRecipeBase *> &GEPs) {
  // Given a VPlan like the following (just including the recipes contributing
  // to loop control exiting here, not the actual work), we're looking to match
  // the recipes contributing to the uncountable exit condition comparison
  // (here, vp<%4>) back to either live-ins or the address nodes for the load
  // used as part of the uncountable exit comparison so that we can copy them
  // to a preheader and rotate the address in the loop to the next vector
  // iteration.
  //
  // Currently, the address of the load is restricted to a GEP with 2 operands
  // and a live-in base address. This constraint may be relaxed later.
  //
  // VPlan ' for UF>=1' {
  // Live-in vp<%0> = VF
  // Live-in ir<64> = original trip-count
  //
  // entry:
  // Successor(s): preheader, vector.ph
  //
  // vector.ph:
  // Successor(s): vector loop
  //
  // <x1> vector loop: {
  //   vector.body:
  //     EMIT vp<%2> = CANONICAL-INDUCTION ir<0>
  //     vp<%3> = SCALAR-STEPS vp<%2>, ir<1>, vp<%0>
  //     CLONE ir<%ee.addr> = getelementptr ir<0>, vp<%3>
  //     WIDEN ir<%ee.load> = load ir<%ee.addr>
  //     WIDEN vp<%4> = icmp eq ir<%ee.load>, ir<0>
  //     EMIT vp<%5> = any-of vp<%4>
  //     EMIT vp<%6> = add vp<%2>, vp<%0>
  //     EMIT vp<%7> = icmp eq vp<%6>, ir<64>
  //     EMIT vp<%8> = or vp<%5>, vp<%7>
  //     EMIT branch-on-cond vp<%8>
  //   No successors
  // }
  // Successor(s): middle.block
  //
  // middle.block:
  // Successor(s): preheader
  //
  // preheader:
  // No successors
  // }

  // Find the uncountable loop exit condition.
  auto *Region = Plan.getVectorLoopRegion();
  VPValue *UncountableCondition = nullptr;
  if (!match(Region->getExitingBasicBlock()->getTerminator(),
             m_BranchOnCond(m_OneUse(m_c_BinaryOr(
                 m_AnyOf(m_VPValue(UncountableCondition)), m_VPValue())))))
    return std::nullopt;

  SmallVector<VPValue *, 4> Worklist;
  Worklist.push_back(UncountableCondition);
  while (!Worklist.empty()) {
    VPValue *V = Worklist.pop_back_val();

    // Any value defined outside the loop does not need to be copied.
    if (V->isDefinedOutsideLoopRegions())
      continue;

    // FIXME: Remove the single user restriction; it's here because we're
    //        starting with the simplest set of loops we can, and multiple
    //        users means needing to add PHI nodes in the transform.
    if (V->getNumUsers() > 1)
      return std::nullopt;

    VPValue *Op1, *Op2;
    // Walk back through recipes until we find at least one load from memory.
    if (match(V, m_ICmp(m_VPValue(Op1), m_VPValue(Op2)))) {
      Worklist.push_back(Op1);
      Worklist.push_back(Op2);
      Recipes.push_back(V->getDefiningRecipe());
    } else if (auto *Load = dyn_cast<VPWidenLoadRecipe>(V)) {
      // Reject masked loads for the time being; they make the exit condition
      // more complex.
      if (Load->isMasked())
        return std::nullopt;

      VPValue *GEP = Load->getAddr();
      if (!match(GEP, m_GetElementPtr(m_LiveIn(), m_VPValue())))
        return std::nullopt;

      Recipes.push_back(Load);
      Recipes.push_back(GEP->getDefiningRecipe());
      GEPs.push_back(GEP->getDefiningRecipe());
    } else
      return std::nullopt;
  }

  return UncountableCondition;
}
