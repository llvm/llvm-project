//===- VPlanUtils.cpp - VPlan-related utilities ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "VPlanUtils.h"
#include "VPlanCFG.h"
#include "VPlanDominatorTree.h"
#include "VPlanPatternMatch.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"

using namespace llvm;
using namespace llvm::VPlanPatternMatch;

bool vputils::onlyFirstLaneUsed(const VPValue *Def) {
  return all_of(Def->users(),
                [Def](const VPUser *U) { return U->usesFirstLaneOnly(Def); });
}

bool vputils::onlyFirstPartUsed(const VPValue *Def) {
  return all_of(Def->users(),
                [Def](const VPUser *U) { return U->usesFirstPartOnly(Def); });
}

bool vputils::onlyScalarValuesUsed(const VPValue *Def) {
  return all_of(Def->users(),
                [Def](const VPUser *U) { return U->usesScalars(Def); });
}

VPValue *vputils::getOrCreateVPValueForSCEVExpr(VPlan &Plan, const SCEV *Expr) {
  if (auto *E = dyn_cast<SCEVConstant>(Expr))
    return Plan.getOrAddLiveIn(E->getValue());
  // Skip SCEV expansion if Expr is a SCEVUnknown wrapping a non-instruction
  // value. Otherwise the value may be defined in a loop and using it directly
  // will break LCSSA form. The SCEV expansion takes care of preserving LCSSA
  // form.
  auto *U = dyn_cast<SCEVUnknown>(Expr);
  if (U && !isa<Instruction>(U->getValue()))
    return Plan.getOrAddLiveIn(U->getValue());
  auto *Expanded = new VPExpandSCEVRecipe(Expr);
  Plan.getEntry()->appendRecipe(Expanded);
  return Expanded;
}

bool vputils::isHeaderMask(const VPValue *V, const VPlan &Plan) {
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
         B == Plan.getBackedgeTakenCount();
}

const SCEV *vputils::getSCEVExprForVPValue(const VPValue *V,
                                           ScalarEvolution &SE, const Loop *L) {
  if (V->isLiveIn()) {
    if (Value *LiveIn = V->getLiveInIRValue())
      return SE.getSCEV(LiveIn);
    return SE.getCouldNotCompute();
  }

  // TODO: Support constructing SCEVs for more recipes as needed.
  return TypeSwitch<const VPRecipeBase *, const SCEV *>(V->getDefiningRecipe())
      .Case<VPExpandSCEVRecipe>(
          [](const VPExpandSCEVRecipe *R) { return R->getSCEV(); })
      .Case<VPCanonicalIVPHIRecipe>([&SE, L](const VPCanonicalIVPHIRecipe *R) {
        if (!L)
          return SE.getCouldNotCompute();
        const SCEV *Start = getSCEVExprForVPValue(R->getOperand(0), SE, L);
        return SE.getAddRecExpr(Start, SE.getOne(Start->getType()), L,
                                SCEV::FlagAnyWrap);
      })
      .Case<VPWidenIntOrFpInductionRecipe>(
          [&SE, L](const VPWidenIntOrFpInductionRecipe *R) {
            const SCEV *Step = getSCEVExprForVPValue(R->getStepValue(), SE, L);
            if (!L || isa<SCEVCouldNotCompute>(Step))
              return SE.getCouldNotCompute();
            const SCEV *Start =
                getSCEVExprForVPValue(R->getStartValue(), SE, L);
            return SE.getAddRecExpr(Start, Step, L, SCEV::FlagAnyWrap);
          })
      .Case<VPDerivedIVRecipe>([&SE, L](const VPDerivedIVRecipe *R) {
        const SCEV *Start = getSCEVExprForVPValue(R->getOperand(0), SE, L);
        const SCEV *IV = getSCEVExprForVPValue(R->getOperand(1), SE, L);
        const SCEV *Scale = getSCEVExprForVPValue(R->getOperand(2), SE, L);
        if (any_of(ArrayRef({Start, IV, Scale}), IsaPred<SCEVCouldNotCompute>))
          return SE.getCouldNotCompute();

        return SE.getAddExpr(SE.getTruncateOrSignExtend(Start, IV->getType()),
                             SE.getMulExpr(IV, SE.getTruncateOrSignExtend(
                                                   Scale, IV->getType())));
      })
      .Case<VPScalarIVStepsRecipe>([&SE, L](const VPScalarIVStepsRecipe *R) {
        const SCEV *IV = getSCEVExprForVPValue(R->getOperand(0), SE, L);
        const SCEV *Step = getSCEVExprForVPValue(R->getOperand(1), SE, L);
        if (isa<SCEVCouldNotCompute>(IV) || isa<SCEVCouldNotCompute>(Step) ||
            !Step->isOne())
          return SE.getCouldNotCompute();
        return SE.getMulExpr(SE.getTruncateOrSignExtend(IV, Step->getType()),
                             Step);
      })
      .Case<VPReplicateRecipe>([&SE, L](const VPReplicateRecipe *R) {
        if (R->getOpcode() != Instruction::GetElementPtr)
          return SE.getCouldNotCompute();

        const SCEV *Base = getSCEVExprForVPValue(R->getOperand(0), SE, L);
        if (isa<SCEVCouldNotCompute>(Base))
          return SE.getCouldNotCompute();

        SmallVector<const SCEV *> IndexExprs;
        for (VPValue *Index : drop_begin(R->operands())) {
          const SCEV *IndexExpr = getSCEVExprForVPValue(Index, SE, L);
          if (isa<SCEVCouldNotCompute>(IndexExpr))
            return SE.getCouldNotCompute();
          IndexExprs.push_back(IndexExpr);
        }

        Type *SrcElementTy = cast<GetElementPtrInst>(R->getUnderlyingInstr())
                                 ->getSourceElementType();
        return SE.getGEPExpr(Base, IndexExprs, SrcElementTy);
      })
      .Default([&SE](const VPRecipeBase *) { return SE.getCouldNotCompute(); });
}

bool vputils::isSingleScalar(const VPValue *VPV) {
  auto PreservesUniformity = [](unsigned Opcode) -> bool {
    if (Instruction::isBinaryOp(Opcode) || Instruction::isCast(Opcode))
      return true;
    switch (Opcode) {
    case Instruction::GetElementPtr:
    case Instruction::ICmp:
    case Instruction::FCmp:
    case Instruction::Select:
    case VPInstruction::Not:
    case VPInstruction::Broadcast:
    case VPInstruction::PtrAdd:
      return true;
    default:
      return false;
    }
  };

  // A live-in must be uniform across the scope of VPlan.
  if (VPV->isLiveIn())
    return true;

  if (auto *Rep = dyn_cast<VPReplicateRecipe>(VPV)) {
    const VPRegionBlock *RegionOfR = Rep->getRegion();
    // Don't consider recipes in replicate regions as uniform yet; their first
    // lane cannot be accessed when executing the replicate region for other
    // lanes.
    if (RegionOfR && RegionOfR->isReplicator())
      return false;
    return Rep->isSingleScalar() || (PreservesUniformity(Rep->getOpcode()) &&
                                     all_of(Rep->operands(), isSingleScalar));
  }
  if (isa<VPWidenGEPRecipe, VPDerivedIVRecipe, VPBlendRecipe,
          VPWidenSelectRecipe>(VPV))
    return all_of(VPV->getDefiningRecipe()->operands(), isSingleScalar);
  if (auto *WidenR = dyn_cast<VPWidenRecipe>(VPV)) {
    return PreservesUniformity(WidenR->getOpcode()) &&
           all_of(WidenR->operands(), isSingleScalar);
  }
  if (auto *VPI = dyn_cast<VPInstruction>(VPV))
    return VPI->isSingleScalar() || VPI->isVectorToScalar() ||
           (PreservesUniformity(VPI->getOpcode()) &&
            all_of(VPI->operands(), isSingleScalar));
  if (isa<VPPartialReductionRecipe>(VPV))
    return false;
  if (isa<VPReductionRecipe>(VPV))
    return true;
  if (auto *Expr = dyn_cast<VPExpressionRecipe>(VPV))
    return Expr->isSingleScalar();

  // VPExpandSCEVRecipes must be placed in the entry and are always uniform.
  return isa<VPExpandSCEVRecipe>(VPV);
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
        // Be conservative about side-effects, except for the
        // known-side-effecting assumes and stores, which we know will be
        // uniform.
        return R->isSingleScalar() &&
               (!R->mayHaveSideEffects() ||
                isa<AssumeInst, StoreInst>(R->getUnderlyingInstr())) &&
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
  if (auto *ER = dyn_cast<VPExpressionRecipe>(R))
    return ER->getVFScaleFactor();
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

bool VPBlockUtils::isHeader(const VPBlockBase *VPB,
                            const VPDominatorTree &VPDT) {
  auto *VPBB = dyn_cast<VPBasicBlock>(VPB);
  if (!VPBB)
    return false;

  // If VPBB is in a region R, VPBB is a loop header if R is a loop region with
  // VPBB as its entry, i.e., free of predecessors.
  if (auto *R = VPBB->getParent())
    return !R->isReplicator() && !VPBB->hasPredecessors();

  // A header dominates its second predecessor (the latch), with the other
  // predecessor being the preheader
  return VPB->getPredecessors().size() == 2 &&
         VPDT.dominates(VPB, VPB->getPredecessors()[1]);
}

bool VPBlockUtils::isLatch(const VPBlockBase *VPB,
                           const VPDominatorTree &VPDT) {
  // A latch has a header as its second successor, with its other successor
  // leaving the loop. A preheader OTOH has a header as its first (and only)
  // successor.
  return VPB->getNumSuccessors() == 2 &&
         VPBlockUtils::isHeader(VPB->getSuccessors()[1], VPDT);
}
