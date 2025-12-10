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
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"

using namespace llvm;

bool vputils::onlyFirstLaneUsed(const VPValue *Def) {
  return all_of(Def->users(),
                [Def](const VPUser *U) { return U->onlyFirstLaneUsed(Def); });
}

bool vputils::onlyFirstPartUsed(const VPValue *Def) {
  return all_of(Def->users(),
                [Def](const VPUser *U) { return U->onlyFirstPartUsed(Def); });
}

VPValue *vputils::getOrCreateVPValueForSCEVExpr(VPlan &Plan, const SCEV *Expr,
                                                ScalarEvolution &SE) {
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
      Expanded = new VPExpandSCEVRecipe(Expr, SE);
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
  using namespace VPlanPatternMatch;

  if (match(V, m_ActiveLaneMask(m_VPValue(A), m_VPValue(B))))
    return B == Plan.getTripCount() &&
           (match(A, m_ScalarIVSteps(m_Specific(Plan.getCanonicalIV()),
                                     m_SpecificInt(1),
                                     m_Specific(&Plan.getVF()))) ||
            IsWideCanonicalIV(A));

  return match(V, m_Binary<Instruction::ICmp>(m_VPValue(A), m_VPValue(B))) &&
         IsWideCanonicalIV(A) && B == Plan.getOrCreateBackedgeTakenCount();
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

bool vputils::isUniformAcrossVFsAndUFs(VPValue *V) {
  using namespace VPlanPatternMatch;
  // Live-ins are uniform.
  if (V->isLiveIn())
    return true;

  VPRecipeBase *R = V->getDefiningRecipe();
  if (R && V->isDefinedOutsideLoopRegions()) {
    if (match(V->getDefiningRecipe(),
              m_VPInstruction<VPInstruction::CanonicalIVIncrementForPart>(
                  m_VPValue())))
      return false;
    return all_of(R->operands(), isUniformAcrossVFsAndUFs);
  }

  auto *CanonicalIV = R->getParent()->getPlan()->getCanonicalIV();
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

std::optional<MemoryLocation>
vputils::getMemoryLocation(const VPRecipeBase &R) {
  return TypeSwitch<const VPRecipeBase *, std::optional<MemoryLocation>>(&R)
      .Case<VPWidenMemoryRecipe, VPInterleaveRecipe, VPReplicateRecipe>(
          [](auto *S) {
            MemoryLocation Loc;
            // Populate noalias metadata from VPIRMetadata.
            if (MDNode *NoAliasMD = S->getMetadata(LLVMContext::MD_noalias))
              Loc.AATags.NoAlias = NoAliasMD;
            if (MDNode *AliasScopeMD =
                    S->getMetadata(LLVMContext::MD_alias_scope))
              Loc.AATags.Scope = AliasScopeMD;
            return Loc;
          })
      .Default([](auto *) { return std::nullopt; });
}
