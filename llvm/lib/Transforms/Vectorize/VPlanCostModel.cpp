//===- VPlanCostModel.h - VPlan-based Vectorizer Cost Model ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// VPlan-based cost model
///
//===----------------------------------------------------------------------===//

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Operator.h"
#include "llvm/Support/Debug.h"

#include "VPlan.h"
#include "VPlanCFG.h"
#include "VPlanCostModel.h"
#include "VPlanValue.h"

using namespace llvm;

#define DEBUG_TYPE "vplan-cost-model"

namespace llvm {
InstructionCost VPlanCostModel::expectedCost(const VPlan &Plan, ElementCount VF,
                                             bool &IsVec) {
  InstructionCost VectorIterCost = 0;
  for (const VPBlockBase *Block : vp_depth_first_deep(Plan.getEntry()))
    VectorIterCost += getCost(Block, VF, IsVec);

  return VectorIterCost;
}

InstructionCost VPlanCostModel::getCost(const VPBlockBase *Block,
                                        ElementCount VF, bool &IsVec) {
  return TypeSwitch<const VPBlockBase *, InstructionCost>(Block)
      .Case<VPBasicBlock>([&](const VPBasicBlock *BBlock) {
        InstructionCost Cost = 0;
        for (const VPRecipeBase &Recipe : *BBlock)
          Cost += getCost(&Recipe, VF, IsVec);
        return Cost;
      })
      .Default([&](const VPBlockBase *BBlock) -> InstructionCost { return 0; });
}

InstructionCost VPlanCostModel::getCost(const VPRecipeBase *Recipe,
                                        ElementCount VF, bool &IsVec) {
  auto *ScCondTy = Type::getInt1Ty(Context);
  auto *VecCondTy = VectorType::get(ScCondTy, VF);
  InstructionCost Cost =
      TypeSwitch<const VPRecipeBase *, InstructionCost>(Recipe)
          .Case<VPInstruction>([&](const VPInstruction *VPI)
                                   -> InstructionCost {
            unsigned Opcode = VPI->getOpcode();
            if (Instruction::isBinaryOp(Opcode)) {
              // Operands: A, B
              IsVec |= true;
              Type *VectorTy = VectorType::get(getReturnElementType(VPI), VF);
              return TTI.getArithmeticInstrCost(Opcode, VectorTy, CostKind);
            }
            switch (Opcode) {
            case VPInstruction::Not: {
              // Operands: A
              IsVec |= true;
              Type *VectorTy = VectorType::get(getElementType(VPI, 0), VF);
              return TTI.getArithmeticInstrCost(Instruction::Xor, VectorTy,
                                                CostKind);
            }
            case VPInstruction::ICmpULE: {
              // Operands: IV, TripCount
              IsVec |= true;
              Type *VectorTy = VectorType::get(getElementType(VPI, 0), VF);
              return TTI.getCmpSelInstrCost(Instruction::ICmp, VectorTy,
                                            VecCondTy, CmpInst::ICMP_ULE,
                                            CostKind);
            }
            case Instruction::Select: {
              // Operands: Cond, Op1, Op2
              IsVec |= true;
              Type *VectorTy = VectorType::get(getReturnElementType(VPI), VF);
              return TTI.getCmpSelInstrCost(
                  Instruction::Select, VectorTy, VecCondTy,
                  CmpInst::BAD_ICMP_PREDICATE, CostKind);
            }
            case VPInstruction::ActiveLaneMask: {
              // Operands: IV, TripCount
              IsVec |= true;
              Type *OpTy = Type::getIntNTy(
                  Context, getElementType(VPI, 0)->getScalarSizeInBits());
              IntrinsicCostAttributes ICA(Intrinsic::get_active_lane_mask,
                                          VecCondTy, {OpTy, OpTy});
              return TTI.getIntrinsicInstrCost(ICA, CostKind);
            }
            case VPInstruction::FirstOrderRecurrenceSplice: {
              // Operands: FOR, FOR.backedge
              IsVec |= true;
              Type *VectorTy = VectorType::get(getReturnElementType(VPI), VF);
              SmallVector<int> Mask(VF.getKnownMinValue());
              std::iota(Mask.begin(), Mask.end(), VF.getKnownMinValue() - 1);
              return TTI.getShuffleCost(TargetTransformInfo::SK_Splice,
                                        cast<VectorType>(VectorTy), Mask,
                                        CostKind, VF.getKnownMinValue() - 1);
            }
            case VPInstruction::CalculateTripCountMinusVF: {
              // Operands: TripCount
              Type *ScalarTy = getReturnElementType(VPI);
              return TTI.getArithmeticInstrCost(Instruction::Sub, ScalarTy,
                                                CostKind) +
                     TTI.getCmpSelInstrCost(Instruction::ICmp, ScalarTy,
                                            ScCondTy, CmpInst::ICMP_UGT,
                                            CostKind) +
                     TTI.getCmpSelInstrCost(
                         Instruction::Select, ScalarTy, ScCondTy,
                         CmpInst::BAD_ICMP_PREDICATE, CostKind);
            }
            case VPInstruction::CanonicalIVIncrement:
            case VPInstruction::CanonicalIVIncrementNUW:
              // Operands: IVPhi, CanonicalIVIncrement
            case VPInstruction::CanonicalIVIncrementForPart:
            case VPInstruction::CanonicalIVIncrementForPartNUW: {
              // Operands: StartV
              Type *ScalarTy = getReturnElementType(VPI);
              return TTI.getArithmeticInstrCost(Instruction::Add, ScalarTy,
                                                CostKind);
            }
            case VPInstruction::BranchOnCond:
              // Operands: Cond
            case VPInstruction::BranchOnCount: {
              // Operands: IV, TripCount
              Type *ScalarTy = getElementType(VPI, 0);
              return TTI.getCmpSelInstrCost(Instruction::ICmp, ScalarTy,
                                            ScCondTy, CmpInst::ICMP_EQ,
                                            CostKind) +
                     TTI.getCFInstrCost(Instruction::Br, CostKind);
            }
            default:
              llvm_unreachable("Unsupported opcode for VPInstruction");
            } // end of switch
          })
          .Case<VPWidenMemoryInstructionRecipe>(
              [&](const VPWidenMemoryInstructionRecipe *VPWMIR) {
                IsVec |= true;
                return getMemoryOpCost(VPWMIR, VF);
              })
          .Default([&](const VPRecipeBase *R) -> InstructionCost {
            if (!R->hasUnderlyingInstr()) {
              LLVM_DEBUG(
                  dbgs() << "VPlanCM: unsupported recipe ";
                  VPSlotTracker SlotTracker((Recipe->getParent())
                                                ? Recipe->getParent()->getPlan()
                                                : nullptr);
                  Recipe->print(dbgs(), Twine(), SlotTracker); dbgs() << '\n');
              return 0;
            }
            Instruction *I = const_cast<Instruction *>(R->getUnderlyingInstr());
            return getLegacyInstructionCost(I, VF);
          });

  LLVM_DEBUG(dbgs() << "VPlanCM: cost " << Cost << " for VF " << VF
                    << " for VPInstruction: ";
             VPSlotTracker SlotTracker((Recipe->getParent())
                                           ? Recipe->getParent()->getPlan()
                                           : nullptr);
             Recipe->print(dbgs(), Twine(), SlotTracker); dbgs() << '\n');
  return Cost;
}

InstructionCost VPlanCostModel::getMemoryOpCost(const Instruction *I, Type *Ty,
                                                bool IsConsecutive,
                                                bool IsMasked, bool IsReverse) {
  const Align Alignment = getLoadStoreAlignment(const_cast<Instruction *>(I));
  const Value *Ptr = getLoadStorePointerOperand(I);
  unsigned AS = getLoadStoreAddressSpace(const_cast<Instruction *>(I));
  if (IsConsecutive) {
    InstructionCost Cost = 0;
    if (IsMasked) {
      Cost += TTI.getMaskedMemoryOpCost(I->getOpcode(), Ty, Alignment, AS,
                                        CostKind);
    } else {
      TTI::OperandValueInfo OpInfo = TTI::getOperandInfo(I->getOperand(0));
      Cost += TTI.getMemoryOpCost(I->getOpcode(), Ty, Alignment, AS, CostKind,
                                  OpInfo, I);
    }
    if (IsReverse)
      Cost +=
          TTI.getShuffleCost(TargetTransformInfo::SK_Reverse,
                             cast<VectorType>(Ty), std::nullopt, CostKind, 0);
    return Cost;
  }
  return TTI.getAddressComputationCost(Ty) +
         TTI.getGatherScatterOpCost(I->getOpcode(), Ty, Ptr, IsMasked,
                                    Alignment, CostKind, I);
}

InstructionCost
VPlanCostModel::getMemoryOpCost(const VPWidenMemoryInstructionRecipe *VPWMIR,
                                ElementCount VF) {
  Instruction *I = &VPWMIR->getIngredient();
  const bool IsMasked = VPWMIR->getMask() != nullptr;
  Type *VectorTy = VectorType::get(getReturnElementType(VPWMIR), VF);

  return getMemoryOpCost(I, VectorTy, VPWMIR->isConsecutive(), IsMasked,
                         VPWMIR->isReverse());
}

// Return element type the recipe processes since VF is not carried in VPlan
Type *VPlanCostModel::getElementType(const VPRecipeBase *Recipe,
                                     unsigned N) const {
  auto TruncatedType = [&](Value *V) -> Type * {
    Type *ValTy = V->getType();
    ;
    if (llvm::Instruction *Inst = llvm::dyn_cast<llvm::Instruction>(V))
      ValTy = truncateToMinimalBitwidth(V->getType(), Inst);
    return ValTy;
  };
  Value *V = Recipe->getOperand(N)->getUnderlyingValue();
  if (V)
    return TruncatedType(V);
  assert(Recipe->getOperand(N)->hasDefiningRecipe() &&
         "VPValue has no live-in and defining recipe");
  return getReturnElementType(Recipe->getOperand(N)->getDefiningRecipe());
}

Type *VPlanCostModel::getReturnElementType(const VPRecipeBase *Recipe) const {
  auto *Int1Ty = Type::getInt1Ty(Context);
  Type *ValTy =
      TypeSwitch<const VPRecipeBase *, Type *>(Recipe)
          .Case<VPInstruction>([&](const VPInstruction *VPI) -> Type * {
            unsigned Opcode = VPI->getOpcode();
            if (Instruction::isBinaryOp(Opcode))
              // Operands: A, B
              return getElementType(VPI, 0);
            switch (Opcode) {
            case VPInstruction::Not:
              // Operands: A
            case VPInstruction::ICmpULE:
              // Operands: IV, TripCount
              return Int1Ty;
            case Instruction::Select:
              // Operands: Cond, Op1, Op2
              return getElementType(VPI, 1);
            case VPInstruction::ActiveLaneMask:
              // Operands: IV, TripCount
              return Int1Ty;
            case VPInstruction::FirstOrderRecurrenceSplice:
              // Operands: FOR, FOR.backedge
            case VPInstruction::CalculateTripCountMinusVF:
              // Operands: TripCount
            case VPInstruction::CanonicalIVIncrement:
            case VPInstruction::CanonicalIVIncrementNUW:
              // Operands: IVPhi, CanonicalIVIncrement
            case VPInstruction::CanonicalIVIncrementForPart:
            case VPInstruction::CanonicalIVIncrementForPartNUW:
              // Operands: StartV
              return getElementType(VPI, 0);
            case VPInstruction::BranchOnCond:
              // Operands: Cond
            case VPInstruction::BranchOnCount: {
              // Operands: IV, TripCount
              llvm_unreachable("Operation doesn't have return type");
            }
            default:
              llvm_unreachable("Unsupported opcode for VPInstruction");
            }
          })
          .Case<VPWidenMemoryInstructionRecipe>(
              [&](const VPWidenMemoryInstructionRecipe *VPWMIR) -> Type * {
                Instruction *I = &VPWMIR->getIngredient();
                Type *ValTy = truncateToMinimalBitwidth(getLoadStoreType(I), I);
                return ValTy;
              })
          .Default([&](const VPRecipeBase *R) -> Type * {
            llvm_unreachable("Unsupported VPRecipe");
          });
  return ValTy;
}

} // namespace llvm
