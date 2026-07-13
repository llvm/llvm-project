//===- VPlanUtils.cpp - VPlan-related utilities ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "VPlanUtils.h"
#include "LoopVectorizationPlanner.h"
#include "VPlanAnalysis.h"
#include "VPlanCFG.h"
#include "VPlanDominatorTree.h"
#include "VPlanPatternMatch.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/ScalarEvolutionPatternMatch.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"

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
  VPBasicBlock *EntryVPBB = Plan.getEntry();
  auto Iter = EntryVPBB->getFirstNonPhi();
  while (Iter != EntryVPBB->end() && isa<VPIRInstruction>(*Iter))
    ++Iter;
  EntryVPBB->insert(Expanded, Iter);
  return Expanded;
}

/// Returns true if \p R propagates poison from any operand to its result.
static bool propagatesPoisonFromRecipeOp(const VPRecipeBase *R) {
  return TypeSwitch<const VPRecipeBase *, bool>(R)
      .Case<VPWidenGEPRecipe, VPWidenCastRecipe>(
          [](const VPRecipeBase *) { return true; })
      .Case([](const VPReplicateRecipe *Rep) {
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
      if (auto *MemR = dyn_cast<VPWidenMemoryRecipe>(cast<VPRecipeBase>(U))) {
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

GEPNoWrapFlags vputils::getGEPFlagsForPtr(VPValue *Ptr) {
  // Like IR stripPointerCasts, look through GEPs with all-zero indices and
  // casts to find a root GEP VPInstruction.
  while (auto *PtrVPI = dyn_cast<VPInstruction>(Ptr)) {
    unsigned Opcode = PtrVPI->getOpcode();
    if (Opcode == Instruction::GetElementPtr) {
      if (!all_of(drop_begin(PtrVPI->operands()), match_fn(m_ZeroInt())))
        return PtrVPI->getGEPNoWrapFlags();
      Ptr = PtrVPI->getOperand(0);
      continue;
    }
    if (Opcode != Instruction::BitCast && Opcode != Instruction::AddrSpaceCast)
      break;
    Ptr = PtrVPI->getOperand(0);
  }
  return GEPNoWrapFlags::none();
}

const SCEV *vputils::getSCEVExprForVPValue(const VPValue *V,
                                           PredicatedScalarEvolution &PSE,
                                           const Loop *L) {
  ScalarEvolution &SE = *PSE.getSE();
  if (auto *RV = dyn_cast<VPRegionValue>(V)) {
    assert(RV == RV->getDefiningRegion()->getCanonicalIV() &&
           "RegionValue must be canonical IV");
    if (!L)
      return SE.getCouldNotCompute();
    return SE.getAddRecExpr(SE.getZero(RV->getType()), SE.getOne(RV->getType()),
                            L, SCEV::FlagAnyWrap);
  }

  if (isa<VPIRValue, VPSymbolicValue>(V)) {
    Value *LiveIn = V->getUnderlyingValue();
    if (LiveIn && SE.isSCEVable(LiveIn->getType()))
      return SE.getSCEV(LiveIn);
    return SE.getCouldNotCompute();
  }

  // Helper to create SCEVs for binary and unary operations.
  auto CreateSCEV = [&](ArrayRef<VPValue *> Ops,
                        function_ref<const SCEV *(ArrayRef<SCEVUse>)> CreateFn)
      -> const SCEV * {
    SmallVector<SCEVUse, 2> SCEVOps;
    for (VPValue *Op : Ops) {
      const SCEV *S = getSCEVExprForVPValue(Op, PSE, L);
      if (isa<SCEVCouldNotCompute>(S))
        return SE.getCouldNotCompute();
      SCEVOps.push_back(S);
    }
    return PSE.getPredicatedSCEV(CreateFn(SCEVOps));
  };

  VPValue *LHSVal, *RHSVal;
  if (match(V, m_Add(m_VPValue(LHSVal), m_VPValue(RHSVal))))
    return CreateSCEV({LHSVal, RHSVal}, [&](ArrayRef<SCEVUse> Ops) {
      return SE.getAddExpr(Ops[0], Ops[1], SCEV::FlagAnyWrap, 0);
    });
  if (match(V, m_Sub(m_VPValue(LHSVal), m_VPValue(RHSVal))))
    return CreateSCEV({LHSVal, RHSVal}, [&](ArrayRef<SCEVUse> Ops) {
      return SE.getMinusSCEV(Ops[0], Ops[1], SCEV::FlagAnyWrap, 0);
    });
  if (match(V, m_Not(m_VPValue(LHSVal)))) {
    // not X = xor X, -1 = -1 - X
    return CreateSCEV({LHSVal}, [&](ArrayRef<SCEVUse> Ops) {
      return SE.getMinusSCEV(SE.getMinusOne(Ops[0]->getType()), Ops[0]);
    });
  }
  if (match(V, m_Mul(m_VPValue(LHSVal), m_VPValue(RHSVal))))
    return CreateSCEV({LHSVal, RHSVal}, [&](ArrayRef<SCEVUse> Ops) {
      return SE.getMulExpr(Ops[0], Ops[1], SCEV::FlagAnyWrap, 0);
    });
  // Handle shl by constant: x << c is equivalent to x * (1 << c). A shift
  // amount >= the bit width produces poison; do not rewrite it, as
  // getPowerOfTwo requires the power to be in range.
  uint64_t ShiftAmt;
  if (match(V, m_Shl(m_VPValue(LHSVal), m_ConstantInt(ShiftAmt))) &&
      ShiftAmt < LHSVal->getScalarType()->getScalarSizeInBits())
    return CreateSCEV(LHSVal, [&](ArrayRef<SCEVUse> Ops) {
      return SE.getMulExpr(Ops[0],
                           SE.getPowerOfTwo(Ops[0]->getType(), ShiftAmt));
    });
  if (match(V, m_LShr(m_VPValue(LHSVal), m_ConstantInt(ShiftAmt)))) {
    Type *Ty = V->getScalarType();
    if (ShiftAmt < SE.getTypeSizeInBits(Ty))
      return CreateSCEV(LHSVal, [&](ArrayRef<SCEVUse> Ops) {
        return SE.getUDivExpr(Ops[0], SE.getPowerOfTwo(Ty, ShiftAmt));
      });
  }
  if (match(V, m_UDiv(m_VPValue(LHSVal), m_VPValue(RHSVal))))
    return CreateSCEV({LHSVal, RHSVal}, [&](ArrayRef<SCEVUse> Ops) {
      return SE.getUDivExpr(Ops[0], Ops[1]);
    });
  if (match(V, m_URem(m_VPValue(LHSVal), m_VPValue(RHSVal))))
    return CreateSCEV({LHSVal, RHSVal}, [&](ArrayRef<SCEVUse> Ops) {
      return SE.getURemExpr(Ops[0], Ops[1]);
    });
  // A SRem with non-negative operands is equivalent to an URem.
  if (match(V, m_SRem(m_VPValue(LHSVal), m_VPValue(RHSVal)))) {
    return CreateSCEV({LHSVal, RHSVal}, [&](ArrayRef<SCEVUse> Ops) {
      if (!SE.isKnownNonNegative(Ops[0]) || !SE.isKnownNonNegative(Ops[1]))
        return SE.getCouldNotCompute();
      return SE.getURemExpr(Ops[0], Ops[1]);
    });
  }
  // Handle AND with constant mask: x & (2^n - 1) can be represented as x % 2^n.
  const APInt *Mask;
  if (match(V, m_c_BinaryAnd(m_VPValue(LHSVal), m_APInt(Mask))) &&
      (*Mask + 1).isPowerOf2())
    return CreateSCEV({LHSVal}, [&](ArrayRef<SCEVUse> Ops) {
      return SE.getURemExpr(Ops[0], SE.getConstant(*Mask + 1));
    });
  if (match(V, m_Trunc(m_VPValue(LHSVal)))) {
    Type *DestTy = V->getScalarType();
    return CreateSCEV({LHSVal}, [&](ArrayRef<SCEVUse> Ops) {
      return SE.getTruncateExpr(Ops[0], DestTy);
    });
  }
  if (match(V, m_ZExt(m_VPValue(LHSVal)))) {
    Type *DestTy = V->getScalarType();
    return CreateSCEV({LHSVal}, [&](ArrayRef<SCEVUse> Ops) {
      return SE.getZeroExtendExpr(Ops[0], DestTy);
    });
  }
  if (match(V, m_SExt(m_VPValue(LHSVal)))) {
    Type *DestTy = V->getScalarType();

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

    return CreateSCEV({LHSVal}, [&](ArrayRef<SCEVUse> Ops) {
      return SE.getSignExtendExpr(Ops[0], DestTy);
    });
  }
  if (match(V,
            m_Intrinsic<Intrinsic::umax>(m_VPValue(LHSVal), m_VPValue(RHSVal))))
    return CreateSCEV({LHSVal, RHSVal}, [&](ArrayRef<SCEVUse> Ops) {
      return SE.getUMaxExpr(Ops[0], Ops[1]);
    });
  if (match(V,
            m_Intrinsic<Intrinsic::smax>(m_VPValue(LHSVal), m_VPValue(RHSVal))))
    return CreateSCEV({LHSVal, RHSVal}, [&](ArrayRef<SCEVUse> Ops) {
      return SE.getSMaxExpr(Ops[0], Ops[1]);
    });
  if (match(V,
            m_Intrinsic<Intrinsic::umin>(m_VPValue(LHSVal), m_VPValue(RHSVal))))
    return CreateSCEV({LHSVal, RHSVal}, [&](ArrayRef<SCEVUse> Ops) {
      return SE.getUMinExpr(Ops[0], Ops[1]);
    });
  if (match(V,
            m_Intrinsic<Intrinsic::smin>(m_VPValue(LHSVal), m_VPValue(RHSVal))))
    return CreateSCEV({LHSVal, RHSVal}, [&](ArrayRef<SCEVUse> Ops) {
      return SE.getSMinExpr(Ops[0], Ops[1]);
    });
  if (match(V, m_Intrinsic<Intrinsic::abs>(m_VPValue(LHSVal), m_VPValue())))
    return CreateSCEV({LHSVal}, [&](ArrayRef<SCEVUse> Ops) {
      // is_int_min_poison is local to this intrinsic: poison on INT_MIN is
      // not proof that the input is never INT_MIN, nor that poison reaches
      // UB. Do not translate it to SCEV's global IsNSW flag.
      return SE.getAbsExpr(Ops[0], /*IsNSW=*/false);
    });

  ArrayRef<VPValue *> Ops;
  Type *SourceElementType;
  if (match(V, m_GetElementPtr(SourceElementType, Ops))) {
    return CreateSCEV(Ops, [&](ArrayRef<SCEVUse> Ops) {
      return SE.getGEPExpr(Ops.front(), Ops.drop_front(), SourceElementType);
    });
  }

  // TODO: Support constructing SCEVs for more recipes as needed.
  const VPRecipeBase *DefR = V->getDefiningRecipe();
  const SCEV *Expr =
      TypeSwitch<const VPRecipeBase *, const SCEV *>(DefR)
          .Case([](const VPExpandSCEVRecipe *R) { return R->getSCEV(); })
          .Case([&SE, &PSE, L](const VPWidenIntOrFpInductionRecipe *R) {
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
          .Case([&SE, &PSE, L](const VPWidenPointerInductionRecipe *R) {
            const SCEV *Start =
                getSCEVExprForVPValue(R->getStartValue(), PSE, L);
            if (!L || isa<SCEVCouldNotCompute>(Start))
              return SE.getCouldNotCompute();
            const SCEV *Step = getSCEVExprForVPValue(R->getStepValue(), PSE, L);
            if (isa<SCEVCouldNotCompute>(Step))
              return SE.getCouldNotCompute();
            return SE.getAddRecExpr(Start, Step, L, SCEV::FlagAnyWrap);
          })
          .Case([&SE, &PSE, L](const VPDerivedIVRecipe *R) {
            const SCEV *Start = getSCEVExprForVPValue(R->getOperand(0), PSE, L);
            const SCEV *IV = getSCEVExprForVPValue(R->getOperand(1), PSE, L);
            const SCEV *Scale = getSCEVExprForVPValue(R->getOperand(2), PSE, L);
            if (any_of(ArrayRef({Start, IV, Scale}),
                       IsaPred<SCEVCouldNotCompute>))
              return SE.getCouldNotCompute();

            return SE.getAddExpr(
                SE.getTruncateOrSignExtend(Start, IV->getType()),
                SE.getMulExpr(
                    IV, SE.getTruncateOrSignExtend(Scale, IV->getType())));
          })
          .Case([&SE, &PSE, L](const VPScalarIVStepsRecipe *R) {
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
  case Instruction::Freeze:
  case Instruction::GetElementPtr:
  case Instruction::ICmp:
  case Instruction::FCmp:
  case Instruction::Select:
  case VPInstruction::Not:
  case VPInstruction::Broadcast:
  case VPInstruction::MaskedCond:
  case VPInstruction::PtrAdd:
    return true;
  default:
    return false;
  }
}

bool vputils::isElementwise(const VPValue *V) {
  unsigned Opcode = TypeSwitch<const VPValue *, unsigned>(V)
                        .Case<VPInstruction, VPWidenRecipe>(
                            [](auto *R) { return R->getOpcode(); })
                        .Default([](auto *) { return 0; });
  // TODO: Handle more opcodes and recipes.
  return Instruction::isUnaryOp(Opcode) || Instruction::isBinaryOp(Opcode);
}

bool vputils::isSingleScalar(const VPValue *VPV) {
  // Live-in, symbolic and canonical-IV region values are single-scalar.
  if (auto *RV = dyn_cast<VPRegionValue>(VPV))
    return RV == RV->getDefiningRegion()->getCanonicalIV();
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
  if (isa<VPWidenGEPRecipe, VPBlendRecipe>(VPV))
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
  if (isa<VPVectorPointerRecipe, VPVectorEndPointerRecipe, VPDerivedIVRecipe>(
          VPV))
    return true;
  if (auto *Expr = dyn_cast<VPExpressionRecipe>(VPV))
    return Expr->isVectorToScalar();

  // VPExpandSCEVRecipes must be placed in the entry and are always uniform.
  return isa<VPExpandSCEVRecipe>(VPV);
}

bool vputils::isUniformAcrossVFsAndUFs(const VPValue *V) {
  // Live-ins, symbolic and canonical-IV region values are uniform.
  if (auto *RV = dyn_cast<VPRegionValue>(V))
    return RV == RV->getDefiningRegion()->getCanonicalIV();
  if (isa<VPIRValue, VPSymbolicValue>(V))
    return true;

  const VPRecipeBase *R = V->getDefiningRecipe();
  const VPBasicBlock *VPBB = R ? R->getParent() : nullptr;
  const VPlan *Plan = VPBB ? VPBB->getPlan() : nullptr;
  if (VPBB) {
    if ((VPBB == Plan->getVectorPreheader() || VPBB == Plan->getEntry())) {
      if (match(V->getDefiningRecipe(),
                m_VPInstruction<VPInstruction::CanonicalIVIncrementForPart>()))
        return false;
      return all_of(R->operands(), isUniformAcrossVFsAndUFs);
    }
  }

  return TypeSwitch<const VPRecipeBase *, bool>(R)
      .Case([](const VPDerivedIVRecipe *R) { return true; })
      .Case([](const VPReplicateRecipe *R) {
        // Be conservative about side-effects, except for the
        // known-side-effecting assumes and stores, which we know will be
        // uniform.
        return R->isSingleScalar() &&
               (!R->mayHaveSideEffects() ||
                isa<AssumeInst, StoreInst>(R->getUnderlyingInstr())) &&
               all_of(R->operands(), isUniformAcrossVFsAndUFs);
      })
      .Case([](const VPWidenRecipe *R) {
        return preservesUniformity(R->getOpcode()) &&
               all_of(R->operands(), isUniformAcrossVFsAndUFs);
      })
      .Case([](const VPPhi *) {
        // Bail out on VPPhi, as we can end up in infinite cycles.
        return false;
      })
      .Case([](const VPInstruction *VPI) {
        return (VPI->isSingleScalar() || VPI->isVectorToScalar() ||
                preservesUniformity(VPI->getOpcode())) &&
               all_of(VPI->operands(), isUniformAcrossVFsAndUFs);
      })
      .Case([](const VPWidenCastRecipe *R) {
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

bool vputils::cannotHoistOrSinkRecipe(const VPRecipeBase &R, bool Sinking) {
  // Assumes don't alias anything or throw; as long as they're guaranteed to
  // execute, they're safe to hoist. They should however not be sunk, as it
  // would destroy information.
  if (match(&R, m_Intrinsic<Intrinsic::assume>()))
    return Sinking;
  if (R.mayHaveSideEffects() || R.mayReadFromMemory() || R.isPhi())
    return true;
  // Allocas cannot be hoisted.
  auto *RepR = dyn_cast<VPReplicateRecipe>(&R);
  return RepR && RepR->getOpcode() == Instruction::Alloca;
}

SmallVector<VPBasicBlock *>
VPBlockUtils::blocksInSingleSuccessorChainBetween(VPBasicBlock *FirstBB,
                                                  VPBasicBlock *LastBB) {
  assert(FirstBB->getParent() == LastBB->getParent() &&
         "FirstBB and LastBB from different regions");
#ifndef NDEBUG
  bool InSingleSuccChain = false;
  for (VPBlockBase *Succ = FirstBB; Succ; Succ = Succ->getSingleSuccessor())
    InSingleSuccChain |= (Succ == LastBB);
  assert(InSingleSuccChain &&
         "LastBB unreachable from FirstBB in single-successor chain");
#endif
  auto Blocks = to_vector(
      VPBlockUtils::blocksOnly<VPBasicBlock>(vp_depth_first_deep(FirstBB)));
  auto *LastIt = find(Blocks, LastBB);
  assert(LastIt != Blocks.end() &&
         "LastBB unreachable from FirstBB in depth-first traversal");
  Blocks.erase(std::next(LastIt), Blocks.end());
  return Blocks;
}

VPValue *vputils::findIncomingAliasMask(const VPlan &Plan) {
  for (VPRecipeBase &R : *Plan.getVectorPreheader())
    if (match(&R, m_VPInstruction<VPInstruction::IncomingAliasMask>()))
      return cast<VPInstruction>(&R);
  return nullptr;
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
  // A latch has a header as its last successor, with its other successors
  // leaving the loop. A preheader OTOH has a header as its first (and only)
  // successor.
  return VPB->getNumSuccessors() >= 2 &&
         VPBlockUtils::isHeader(VPB->getSuccessors().back(), VPDT);
}

std::pair<VPBasicBlock *, VPBasicBlock *>
VPBlockUtils::getPlainCFGHeaderAndLatch(const VPlan &Plan) {
  auto *Header = cast<VPBasicBlock>(
      Plan.getEntry()->getSuccessors()[1]->getSingleSuccessor());
  auto *Latch = cast<VPBasicBlock>(Header->getPredecessors()[1]);
  return {Header, Latch};
}

VPBasicBlock *VPBlockUtils::getPlainCFGMiddleBlock(const VPlan &Plan) {
  return cast<VPBasicBlock>(Plan.getScalarPreheader()->getPredecessors()[0]);
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

VPInstruction *vputils::findCanonicalIVIncrement(VPlan &Plan) {
  VPRegionBlock *LoopRegion = Plan.getVectorLoopRegion();
  VPRegionValue *CanIV = LoopRegion->getCanonicalIV();
  assert(CanIV && "Expected loop region to have a canonical IV");

  VPSymbolicValue &VFxUF = Plan.getVFxUF();

  // Check if \p Step matches the expected increment step, accounting for
  // materialization of VFxUF and UF.
  auto IsIncrementStep = [&](VPValue *Step) -> bool {
    if (!VFxUF.isMaterialized())
      return Step == &VFxUF;

    VPSymbolicValue &UF = Plan.getUF();
    if (!UF.isMaterialized())
      return Step == &UF ||
             match(Step, m_c_Mul(m_Specific(&Plan.getUF()), m_VScale()));

    // Alias masking: step is number of active lanes of a dependence mask.
    if (match(Step, m_ZExtOrTruncOrSelf(
                        m_VPInstruction<VPInstruction::NumActiveLanes>())))
      return true;

    unsigned ConcreteUF = Plan.getConcreteUF();
    // Fixed VF: step is just the concrete UF.
    if (match(Step, m_SpecificInt(ConcreteUF)))
      return true;

    // Scalable VF: step involves VScale.
    if (ConcreteUF == 1)
      return match(Step, m_VScale());
    if (match(Step, m_c_Mul(m_SpecificInt(ConcreteUF), m_VScale())))
      return true;
    // mul(VScale, ConcreteUF) may have been simplified to
    // shl(VScale, log2(ConcreteUF)) when ConcreteUF is a power of 2.
    return isPowerOf2_32(ConcreteUF) &&
           match(Step, m_Shl(m_VScale(), m_SpecificInt(Log2_32(ConcreteUF))));
  };

  VPInstruction *Increment = nullptr;
  for (VPUser *U : CanIV->users()) {
    VPValue *Step;
    if (isa<VPInstruction>(U) &&
        match(U, m_c_Add(m_Specific(CanIV), m_VPValue(Step))) &&
        IsIncrementStep(Step)) {
      assert(!Increment && "There must be a unique increment");
      Increment = cast<VPInstruction>(U);
    }
  }

  assert((!VFxUF.isMaterialized() || Increment) &&
         "After materializing VFxUF, an increment must exist");
  assert((!Increment ||
          LoopRegion->hasCanonicalIVNUW() == Increment->hasNoUnsignedWrap()) &&
         "NUW flag in region and increment must match");
  return Increment;
}

/// Find the ComputeReductionResult recipe for \p PhiR, looking through selects
/// inserted for predicated reductions or tail folding.
VPInstruction *vputils::findComputeReductionResult(VPReductionPHIRecipe *PhiR) {
  VPValue *BackedgeVal = PhiR->getBackedgeValue();
  if (auto *Res =
          findUserOf<VPInstruction::ComputeReductionResult>(BackedgeVal))
    return Res;

  // Look through selects inserted for tail folding or predicated reductions.
  VPRecipeBase *SelR =
      findUserOf(BackedgeVal, m_Select(m_VPValue(), m_VPValue(), m_VPValue()));
  if (!SelR)
    return nullptr;
  return findUserOf<VPInstruction::ComputeReductionResult>(
      cast<VPSingleDefRecipe>(SelR));
}

bool vputils::isUsedByLoadStoreAddress(const VPValue *V) {
  SmallPtrSet<const VPValue *, 4> Seen;
  SmallVector<const VPValue *> WorkList = {V};

  while (!WorkList.empty()) {
    const VPValue *Cur = WorkList.pop_back_val();
    if (!Seen.insert(Cur).second)
      continue;

    auto *Blend = dyn_cast<VPBlendRecipe>(Cur);
    // Skip blends that use V only through a compare by checking if any incoming
    // value was already visited.
    if (Blend && none_of(seq<unsigned>(0, Blend->getNumIncomingValues()),
                         [&](unsigned I) {
                           return Seen.contains(Blend->getIncomingValue(I));
                         }))
      continue;

    for (VPUser *U : Cur->users()) {
      if (auto *InterleaveR = dyn_cast<VPInterleaveBase>(U))
        if (InterleaveR->getAddr() == Cur)
          return true;
      // Cur is used as the pointer of a (possibly masked) load (operand 0) or
      // store (operand 1).
      if (match(U, m_CombineOr(m_Unary<Instruction::Load>(m_Specific(Cur)),
                               m_Binary<Instruction::Store>(m_VPValue(),
                                                            m_Specific(Cur)))))
        return true;
      if (auto *MemR = dyn_cast<VPWidenMemoryRecipe>(cast<VPRecipeBase>(U))) {
        if (MemR->getAddr() == Cur && MemR->isConsecutive())
          return true;
      }
    }

    // The legacy cost model only supports scalarization loads/stores with phi
    // addresses, if the phi is directly used as load/store address. Don't
    // traverse further for Blends.
    if (Blend)
      continue;

    // Only traverse further through users that also define a value (and can
    // thus have their own users walked). Skip when Cur is only used as mask ,
    // as well as loads: a loaded value does not depend on the load's operand.
    for (VPUser *U : Cur->users()) {
      auto *VPI = dyn_cast<VPInstruction>(U);
      if (VPI && VPI->getMask() == Cur &&
          none_of(VPI->operandsWithoutMask(),
                  [Cur](VPValue *Op) { return Op == Cur; }))
        continue;
      if (match(U, m_VPInstruction<Instruction::Load>()))
        continue;
      if (auto *SDR = dyn_cast<VPSingleDefRecipe>(U))
        WorkList.push_back(SDR);
    }
  }
  return false;
}

/// Try to find a loop-invariant IR value for \p S in the plan's entry block
/// that can be reused. Returns the corresponding live-in VPValue, or nullptr
/// if no reusable IR value is found.
VPValue *VPSCEVExpander::tryToReuseIRValue(const SCEV *S) {
  if (isa<SCEVConstant, SCEVUnknown>(S))
    return nullptr;
  VPlan &Plan = Builder.getPlan();
  BasicBlock *PH = cast<VPIRBasicBlock>(Plan.getEntry())->getIRBasicBlock();
  for (Value *V : SE.getSCEVValues(S)) {
    // Only reuse instructions in the plan's entry block, or, when a
    // DominatorTree is available, any instruction that dominates it.
    // Instructions in sibling branches may not dominate the entry block.
    auto *I = dyn_cast<Instruction>(V);
    if (!I)
      return Plan.getOrAddLiveIn(V);
    if (!SE.DT.dominates(I->getParent(), PH))
      continue;
    SmallVector<Instruction *> DropPoisonGeneratingInsts;
    if (!SE.canReuseInstruction(S, I, DropPoisonGeneratingInsts))
      continue;
    for (Instruction *DropI : DropPoisonGeneratingInsts)
      SCEVExpander::dropPoisonGeneratingAnnotationsAndReinfer(SE, DropI);
    return Plan.getOrAddLiveIn(V);
  }
  return nullptr;
}

VPValue *VPSCEVExpander::tryToExpand(const SCEV *S) {
  if (VPValue *V = tryToReuseIRValue(S))
    return V;

  switch (S->getSCEVType()) {
  case scConstant:
    return Builder.getPlan().getOrAddLiveIn(cast<SCEVConstant>(S)->getValue());
  case scUnknown:
    return Builder.getPlan().getOrAddLiveIn(cast<SCEVUnknown>(S)->getValue());
  case scVScale:
    return Builder.createVScale(S->getType(), DL);
  case scAddExpr:
  case scMulExpr: {
    auto *NAry = cast<SCEVNAryExpr>(S);
    VPIRFlags::WrapFlagsTy WrapFlags(NAry->hasNoUnsignedWrap(),
                                     NAry->hasNoSignedWrap());

    // Expanded poiner SCEVAddExpr as a ptradd of the pointer base and the
    // integer offset, matching SCEVExpander.
    if (S->getType()->isPointerTy()) {
      VPValue *Base = tryToExpand(SE.getPointerBase(S));
      if (!Base)
        return nullptr;
      VPValue *Offset = tryToExpand(SE.removePointerBase(S));
      if (!Offset)
        return nullptr;
      GEPNoWrapFlags GEPFlags = WrapFlags.HasNUW
                                    ? GEPNoWrapFlags::noUnsignedWrap()
                                    : GEPNoWrapFlags::none();
      return Builder.createNoWrapPtrAdd(Base, Offset, GEPFlags, DL);
    }

    unsigned Opcode =
        S->getSCEVType() == scAddExpr ? Instruction::Add : Instruction::Mul;
    // Iterate in reverse so that constants are emitted last.
    SmallVector<VPValue *, 2> Ops;
    for (const SCEVUse &Op : reverse(NAry->operands())) {
      VPValue *OpV = tryToExpand(Op);
      if (!OpV)
        return nullptr;
      Ops.push_back(OpV);
    }
    VPValue *Result = Ops.front();
    for (VPValue *Op : drop_begin(Ops))
      Result = Builder.createOverflowingOp(Opcode, {Result, Op}, WrapFlags, DL);
    return Result;
  }
  case scUDivExpr: {
    auto *UDiv = cast<SCEVUDivExpr>(S);
    VPValue *LHS = tryToExpand(UDiv->getLHS());
    if (!LHS)
      return nullptr;
    VPValue *RHS = tryToExpand(UDiv->getRHS());
    if (!RHS)
      return nullptr;
    return Builder.createNaryOp(Instruction::UDiv, {LHS, RHS},
                                VPIRFlags::getDefaultFlags(Instruction::UDiv),
                                DL);
  }
  case scTruncate:
  case scZeroExtend:
  case scSignExtend: {
    auto *Cast = cast<SCEVCastExpr>(S);
    VPValue *Op = tryToExpand(Cast->getOperand());
    if (!Op)
      return nullptr;
    Instruction::CastOps Opcode;
    switch (S->getSCEVType()) {
    case scTruncate:
      Opcode = Instruction::Trunc;
      break;
    case scZeroExtend:
      Opcode = Instruction::ZExt;
      break;
    case scSignExtend:
      Opcode = Instruction::SExt;
      break;
    default:
      llvm_unreachable("Unhandled cast SCEV");
    }
    return Builder.createScalarCast(Opcode, Op, S->getType(), DL);
  }
  case scUMaxExpr:
  case scSMaxExpr:
  case scUMinExpr:
  case scSMinExpr: {
    auto *MinMax = cast<SCEVMinMaxExpr>(S);
    Intrinsic::ID IntrinsicID;
    switch (S->getSCEVType()) {
    case scUMaxExpr:
      IntrinsicID = Intrinsic::umax;
      break;
    case scSMaxExpr:
      IntrinsicID = Intrinsic::smax;
      break;
    case scUMinExpr:
      IntrinsicID = Intrinsic::umin;
      break;
    case scSMinExpr:
      IntrinsicID = Intrinsic::smin;
      break;
    default:
      llvm_unreachable("Unexpected min/max SCEV type");
    }
    // Chain operands in reverse order matching SCEVExpander's expansion of
    // min/max expressions.
    SmallVector<VPValue *, 2> Ops;
    for (const SCEVUse &Op : reverse(MinMax->operands())) {
      VPValue *OpV = tryToExpand(Op);
      if (!OpV)
        return nullptr;
      Ops.push_back(OpV);
    }
    Type *ResultTy = MinMax->getType();
    VPValue *Result = Ops.front();
    for (VPValue *Op : drop_begin(Ops))
      Result = Builder.createScalarIntrinsic(IntrinsicID, {Result, Op},
                                             ResultTy, DL);
    return Result;
  }
  default:
    return nullptr;
  }
}
