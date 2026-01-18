//===- VPlanUtils.cpp - VPlan-related utilities ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "VPlanUtils.h"
#include "VPlanAnalysis.h"
#include "VPlanCFG.h"
#include "VPlanDominatorTree.h"
#include "VPlanPatternMatch.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/ScalarEvolutionPatternMatch.h"

using namespace llvm;
using namespace llvm::VPlanPatternMatch;
using namespace llvm::SCEVPatternMatch;

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

  auto m_CanonicalScalarIVSteps =
      m_ScalarIVSteps(m_Specific(Plan.getVectorLoopRegion()->getCanonicalIV()),
                      m_One(), m_Specific(&Plan.getVF()));

  if (match(V, m_ActiveLaneMask(m_VPValue(A), m_VPValue(B), m_One())))
    return B == Plan.getTripCount() &&
           (match(A, m_CanonicalScalarIVSteps) || IsWideCanonicalIV(A));

  // For scalar plans, the header mask uses the scalar steps.
  if (match(V, m_ICmp(m_CanonicalScalarIVSteps,
                      m_Specific(Plan.getBackedgeTakenCount())))) {
    assert(Plan.hasScalarVFOnly() &&
           "Non-scalar VF using scalar IV steps for header mask?");
    return true;
  }

  return match(V, m_ICmp(m_VPValue(A), m_VPValue(B))) && IsWideCanonicalIV(A) &&
         B == Plan.getBackedgeTakenCount();
}

/// Returns true if \p R propagates poison from any operand to its result.
static bool propagatesPoisonFromRecipeOp(const VPRecipeBase *R) {
  return TypeSwitch<const VPRecipeBase *, bool>(R)
      .Case<VPWidenGEPRecipe, VPWidenCastRecipe>(
          [](const VPRecipeBase *) { return true; })
      .Case<VPReplicateRecipe>([](const VPReplicateRecipe *Rep) {
        // GEP and casts propagate poison from all operands.
        unsigned Opcode = Rep->getOpcode();
        return Opcode == Instruction::GetElementPtr ||
               Instruction::isCast(Opcode);
      })
      .Default([](const VPRecipeBase *) { return false; });
}

/// Returns true if \p V being poison is guaranteed to trigger UB because it
/// propagates to the address of a memory recipe.
static bool poisonGuaranteesUB(const VPValue *V) {
  SmallPtrSet<const VPValue *, 8> Visited;
  SmallVector<const VPValue *, 16> Worklist;

  Worklist.push_back(V);

  while (!Worklist.empty()) {
    const VPValue *Current = Worklist.pop_back_val();
    if (!Visited.insert(Current).second)
      continue;

    for (VPUser *U : Current->users()) {
      // Check if Current is used as an address operand for load/store.
      if (auto *MemR = dyn_cast<VPWidenMemoryRecipe>(U)) {
        if (MemR->getAddr() == Current)
          return true;
        continue;
      }
      if (auto *Rep = dyn_cast<VPReplicateRecipe>(U)) {
        unsigned Opcode = Rep->getOpcode();
        if ((Opcode == Instruction::Load && Rep->getOperand(0) == Current) ||
            (Opcode == Instruction::Store && Rep->getOperand(1) == Current))
          return true;
      }

      // Check if poison propagates through this recipe to any of its users.
      auto *R = cast<VPRecipeBase>(U);
      for (const VPValue *Op : R->operands()) {
        if (Op == Current && propagatesPoisonFromRecipeOp(R)) {
          Worklist.push_back(R->getVPSingleValue());
          break;
        }
      }
    }
  }

  return false;
}

const SCEV *vputils::getSCEVExprForVPValue(const VPValue *V,
                                           PredicatedScalarEvolution &PSE,
                                           const Loop *L) {
  ScalarEvolution &SE = *PSE.getSE();
  if (isa<VPIRValue, VPSymbolicValue>(V)) {
    Value *LiveIn = V->getUnderlyingValue();
    if (LiveIn && SE.isSCEVable(LiveIn->getType()))
      return SE.getSCEV(LiveIn);
    return SE.getCouldNotCompute();
  }

  // Helper to create SCEVs for binary and unary operations.
  auto CreateSCEV =
      [&](ArrayRef<VPValue *> Ops,
          function_ref<const SCEV *(ArrayRef<const SCEV *>)> CreateFn)
      -> const SCEV * {
    SmallVector<const SCEV *, 2> SCEVOps;
    for (VPValue *Op : Ops) {
      const SCEV *S = getSCEVExprForVPValue(Op, PSE, L);
      if (isa<SCEVCouldNotCompute>(S))
        return SE.getCouldNotCompute();
      SCEVOps.push_back(S);
    }
    return CreateFn(SCEVOps);
  };

  VPValue *LHSVal, *RHSVal;
  if (match(V, m_Add(m_VPValue(LHSVal), m_VPValue(RHSVal))))
    return CreateSCEV({LHSVal, RHSVal}, [&](ArrayRef<const SCEV *> Ops) {
      return SE.getAddExpr(Ops[0], Ops[1], SCEV::FlagAnyWrap, 0);
    });
  if (match(V, m_Sub(m_VPValue(LHSVal), m_VPValue(RHSVal))))
    return CreateSCEV({LHSVal, RHSVal}, [&](ArrayRef<const SCEV *> Ops) {
      return SE.getMinusSCEV(Ops[0], Ops[1], SCEV::FlagAnyWrap, 0);
    });
  if (match(V, m_Not(m_VPValue(LHSVal)))) {
    // not X = xor X, -1 = -1 - X
    return CreateSCEV({LHSVal}, [&](ArrayRef<const SCEV *> Ops) {
      return SE.getMinusSCEV(SE.getMinusOne(Ops[0]->getType()), Ops[0]);
    });
  }
  if (match(V, m_Trunc(m_VPValue(LHSVal)))) {
    const VPlan *Plan = V->getDefiningRecipe()->getParent()->getPlan();
    Type *DestTy = VPTypeAnalysis(*Plan).inferScalarType(V);
    return CreateSCEV({LHSVal}, [&](ArrayRef<const SCEV *> Ops) {
      return SE.getTruncateExpr(Ops[0], DestTy);
    });
  }
  if (match(V, m_ZExt(m_VPValue(LHSVal)))) {
    const VPlan *Plan = V->getDefiningRecipe()->getParent()->getPlan();
    Type *DestTy = VPTypeAnalysis(*Plan).inferScalarType(V);
    return CreateSCEV({LHSVal}, [&](ArrayRef<const SCEV *> Ops) {
      return SE.getZeroExtendExpr(Ops[0], DestTy);
    });
  }
  if (match(V, m_SExt(m_VPValue(LHSVal)))) {
    const VPlan *Plan = V->getDefiningRecipe()->getParent()->getPlan();
    Type *DestTy = VPTypeAnalysis(*Plan).inferScalarType(V);

    // Mirror SCEV's createSCEV handling for sext(sub nsw): push sign extension
    // onto the operands before computing the subtraction.
    VPValue *SubLHS, *SubRHS;
    auto *SubR = dyn_cast<VPRecipeWithIRFlags>(LHSVal);
    if (match(LHSVal, m_Sub(m_VPValue(SubLHS), m_VPValue(SubRHS))) && SubR &&
        SubR->hasNoSignedWrap() && poisonGuaranteesUB(LHSVal)) {
      const SCEV *V1 = getSCEVExprForVPValue(SubLHS, PSE, L);
      const SCEV *V2 = getSCEVExprForVPValue(SubRHS, PSE, L);
      if (!isa<SCEVCouldNotCompute>(V1) && !isa<SCEVCouldNotCompute>(V2))
        return SE.getMinusSCEV(SE.getSignExtendExpr(V1, DestTy),
                               SE.getSignExtendExpr(V2, DestTy), SCEV::FlagNSW);
    }

    return CreateSCEV({LHSVal}, [&](ArrayRef<const SCEV *> Ops) {
      return SE.getSignExtendExpr(Ops[0], DestTy);
    });
  }
  if (match(V,
            m_Intrinsic<Intrinsic::umax>(m_VPValue(LHSVal), m_VPValue(RHSVal))))
    return CreateSCEV({LHSVal, RHSVal}, [&](ArrayRef<const SCEV *> Ops) {
      return SE.getUMaxExpr(Ops[0], Ops[1]);
    });
  if (match(V,
            m_Intrinsic<Intrinsic::smax>(m_VPValue(LHSVal), m_VPValue(RHSVal))))
    return CreateSCEV({LHSVal, RHSVal}, [&](ArrayRef<const SCEV *> Ops) {
      return SE.getSMaxExpr(Ops[0], Ops[1]);
    });
  if (match(V,
            m_Intrinsic<Intrinsic::umin>(m_VPValue(LHSVal), m_VPValue(RHSVal))))
    return CreateSCEV({LHSVal, RHSVal}, [&](ArrayRef<const SCEV *> Ops) {
      return SE.getUMinExpr(Ops[0], Ops[1]);
    });
  if (match(V,
            m_Intrinsic<Intrinsic::smin>(m_VPValue(LHSVal), m_VPValue(RHSVal))))
    return CreateSCEV({LHSVal, RHSVal}, [&](ArrayRef<const SCEV *> Ops) {
      return SE.getSMinExpr(Ops[0], Ops[1]);
    });

  ArrayRef<VPValue *> Ops;
  Type *SourceElementType;
  if (match(V, m_GetElementPtr(SourceElementType, Ops))) {
    const SCEV *GEPExpr = CreateSCEV(Ops, [&](ArrayRef<const SCEV *> Ops) {
      return SE.getGEPExpr(Ops.front(), Ops.drop_front(), SourceElementType);
    });
    return PSE.getPredicatedSCEV(GEPExpr);
  }

  // TODO: Support constructing SCEVs for more recipes as needed.
  const VPRecipeBase *DefR = V->getDefiningRecipe();
  const SCEV *Expr = TypeSwitch<const VPRecipeBase *, const SCEV *>(DefR)
      .Case<VPExpandSCEVRecipe>(
          [](const VPExpandSCEVRecipe *R) { return R->getSCEV(); })
      .Case<VPCanonicalIVPHIRecipe>([&SE, &PSE,
                                     L](const VPCanonicalIVPHIRecipe *R) {
        if (!L)
          return SE.getCouldNotCompute();
        const SCEV *Start = getSCEVExprForVPValue(R->getOperand(0), PSE, L);
        return SE.getAddRecExpr(Start, SE.getOne(Start->getType()), L,
                                SCEV::FlagAnyWrap);
      })
      .Case<VPWidenIntOrFpInductionRecipe>(
          [&SE, &PSE, L](const VPWidenIntOrFpInductionRecipe *R) {
            const SCEV *Step = getSCEVExprForVPValue(R->getStepValue(), PSE, L);
            if (!L || isa<SCEVCouldNotCompute>(Step))
              return SE.getCouldNotCompute();
            const SCEV *Start =
                getSCEVExprForVPValue(R->getStartValue(), PSE, L);
            const SCEV *AddRec =
                SE.getAddRecExpr(Start, Step, L, SCEV::FlagAnyWrap);
            if (R->getTruncInst())
              return SE.getTruncateExpr(AddRec, R->getScalarType());
            return AddRec;
          })
      .Case<VPDerivedIVRecipe>([&SE, &PSE, L](const VPDerivedIVRecipe *R) {
        const SCEV *Start = getSCEVExprForVPValue(R->getOperand(0), PSE, L);
        const SCEV *IV = getSCEVExprForVPValue(R->getOperand(1), PSE, L);
        const SCEV *Scale = getSCEVExprForVPValue(R->getOperand(2), PSE, L);
        if (any_of(ArrayRef({Start, IV, Scale}), IsaPred<SCEVCouldNotCompute>))
          return SE.getCouldNotCompute();

        return SE.getAddExpr(SE.getTruncateOrSignExtend(Start, IV->getType()),
                             SE.getMulExpr(IV, SE.getTruncateOrSignExtend(
                                                   Scale, IV->getType())));
      })
      .Case<VPScalarIVStepsRecipe>([&SE, &PSE,
                                    L](const VPScalarIVStepsRecipe *R) {
        const SCEV *IV = getSCEVExprForVPValue(R->getOperand(0), PSE, L);
        const SCEV *Step = getSCEVExprForVPValue(R->getOperand(1), PSE, L);
        if (isa<SCEVCouldNotCompute>(IV) || !isa<SCEVConstant>(Step))
          return SE.getCouldNotCompute();
        return SE.getTruncateOrSignExtend(IV, Step->getType());
      })
      .Default(
          [&SE](const VPRecipeBase *) { return SE.getCouldNotCompute(); });

  return PSE.getPredicatedSCEV(Expr);
}

bool vputils::isAddressSCEVForCost(const SCEV *Addr, ScalarEvolution &SE,
                                   const Loop *L) {
  // If address is an SCEVAddExpr, we require that all operands must be either
  // be invariant or a (possibly sign-extend) affine AddRec.
  if (auto *PtrAdd = dyn_cast<SCEVAddExpr>(Addr)) {
    return all_of(PtrAdd->operands(), [&SE, L](const SCEV *Op) {
      return SE.isLoopInvariant(Op, L) ||
             match(Op, m_scev_SExt(m_scev_AffineAddRec(m_SCEV(), m_SCEV()))) ||
             match(Op, m_scev_AffineAddRec(m_SCEV(), m_SCEV()));
    });
  }

  // Otherwise, check if address is loop invariant or an affine add recurrence.
  return SE.isLoopInvariant(Addr, L) ||
         match(Addr, m_scev_AffineAddRec(m_SCEV(), m_SCEV()));
}

/// Returns true if \p Opcode preserves uniformity, i.e., if all operands are
/// uniform, the result will also be uniform.
static bool preservesUniformity(unsigned Opcode) {
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
}

bool vputils::isSingleScalar(const VPValue *VPV) {
  // A live-in must be uniform across the scope of VPlan.
  if (isa<VPIRValue, VPSymbolicValue>(VPV))
    return true;

  if (auto *Rep = dyn_cast<VPReplicateRecipe>(VPV)) {
    const VPRegionBlock *RegionOfR = Rep->getRegion();
    // Don't consider recipes in replicate regions as uniform yet; their first
    // lane cannot be accessed when executing the replicate region for other
    // lanes.
    if (RegionOfR && RegionOfR->isReplicator())
      return false;
    return Rep->isSingleScalar() || (preservesUniformity(Rep->getOpcode()) &&
                                     all_of(Rep->operands(), isSingleScalar));
  }
  if (isa<VPWidenGEPRecipe, VPDerivedIVRecipe, VPBlendRecipe>(VPV))
    return all_of(VPV->getDefiningRecipe()->operands(), isSingleScalar);
  if (auto *WidenR = dyn_cast<VPWidenRecipe>(VPV)) {
    return preservesUniformity(WidenR->getOpcode()) &&
           all_of(WidenR->operands(), isSingleScalar);
  }
  if (auto *VPI = dyn_cast<VPInstruction>(VPV))
    return VPI->isSingleScalar() || VPI->isVectorToScalar() ||
           (preservesUniformity(VPI->getOpcode()) &&
            all_of(VPI->operands(), isSingleScalar));
  if (auto *RR = dyn_cast<VPReductionRecipe>(VPV))
    return !RR->isPartialReduction();
  if (isa<VPCanonicalIVPHIRecipe, VPVectorPointerRecipe,
          VPVectorEndPointerRecipe>(VPV))
    return true;
  if (auto *Expr = dyn_cast<VPExpressionRecipe>(VPV))
    return Expr->isSingleScalar();

  // VPExpandSCEVRecipes must be placed in the entry and are always uniform.
  return isa<VPExpandSCEVRecipe>(VPV);
}

bool vputils::isUniformAcrossVFsAndUFs(VPValue *V) {
  // Live-ins are uniform.
  if (isa<VPIRValue, VPSymbolicValue>(V))
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
      .Case<VPWidenRecipe>([](const auto *R) {
        return preservesUniformity(R->getOpcode()) &&
               all_of(R->operands(), isUniformAcrossVFsAndUFs);
      })
      .Case<VPInstruction>([](const auto *VPI) {
        return (VPI->isScalarCast() &&
                isUniformAcrossVFsAndUFs(VPI->getOperand(0))) ||
               (preservesUniformity(VPI->getOpcode()) &&
                all_of(VPI->operands(), isUniformAcrossVFsAndUFs));
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
  if (auto *RR = dyn_cast<VPReductionRecipe>(R))
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
  //     EMIT branch-on-two-conds vp<%5>, vp<%7>
  //   No successors
  // }
  // Successor(s): early.exit, middle.block
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
             m_BranchOnTwoConds(m_AnyOf(m_VPValue(UncountableCondition)),
                                m_VPValue())))
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

std::optional<MemoryLocation>
vputils::getMemoryLocation(const VPRecipeBase &R) {
  auto *M = dyn_cast<VPIRMetadata>(&R);
  if (!M)
    return std::nullopt;
  MemoryLocation Loc;
  // Populate noalias metadata from VPIRMetadata.
  if (MDNode *NoAliasMD = M->getMetadata(LLVMContext::MD_noalias))
    Loc.AATags.NoAlias = NoAliasMD;
  if (MDNode *AliasScopeMD = M->getMetadata(LLVMContext::MD_alias_scope))
    Loc.AATags.Scope = AliasScopeMD;
  return Loc;
}
