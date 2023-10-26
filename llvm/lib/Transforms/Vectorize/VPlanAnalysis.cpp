//===- VPlanAnalysis.cpp - Various Analyses working on VPlan ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "VPlanAnalysis.h"
#include "VPlan.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace llvm;

#define DEBUG_TYPE "vplan"

Type *VPTypeAnalysis::inferScalarType(const VPBlendRecipe *R) {
  Type *ResTy = inferScalarType(R->getIncomingValue(0));
  for (unsigned I = 1, E = R->getNumIncomingValues(); I != E; ++I) {
    VPValue *Inc = R->getIncomingValue(I);
    assert(inferScalarType(Inc) == ResTy &&
           "different types inferred for different incoming values");
    CachedTypes[Inc] = ResTy;
  }
  return ResTy;
}

Type *VPTypeAnalysis::inferScalarType(const VPInstruction *R) {
  switch (R->getOpcode()) {
  case Instruction::Select: {
    Type *ResTy = inferScalarType(R->getOperand(1));
    VPValue *OtherV = R->getOperand(2);
    assert(inferScalarType(OtherV) == ResTy &&
           "different types inferred for different operands");
    CachedTypes[OtherV] = ResTy;
    return ResTy;
  }
  case VPInstruction::FirstOrderRecurrenceSplice:
    return inferScalarType(R->getOperand(0));
  default:
    break;
  }
  llvm_unreachable("Unhandled opcode!");
}

Type *VPTypeAnalysis::inferScalarType(const VPWidenRecipe *R) {
  unsigned Opcode = R->getOpcode();
  switch (Opcode) {
  case Instruction::ICmp:
  case Instruction::FCmp:
    return IntegerType::get(Ctx, 1);
  case Instruction::UDiv:
  case Instruction::SDiv:
  case Instruction::SRem:
  case Instruction::URem:
  case Instruction::Add:
  case Instruction::FAdd:
  case Instruction::Sub:
  case Instruction::FSub:
  case Instruction::Mul:
  case Instruction::FMul:
  case Instruction::FDiv:
  case Instruction::FRem:
  case Instruction::Shl:
  case Instruction::LShr:
  case Instruction::AShr:
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor: {
    Type *ResTy = inferScalarType(R->getOperand(0));
    assert(ResTy == inferScalarType(R->getOperand(1)) &&
           "types for both operands must match for binary op");
    CachedTypes[R->getOperand(1)] = ResTy;
    return ResTy;
  }
  case Instruction::FNeg:
  case Instruction::Freeze:
    return inferScalarType(R->getOperand(0));
  default:
    break;
  }

  // Type inferrence not implemented for opcode.
  LLVM_DEBUG(dbgs() << "LV: Found unhandled opcode: "
                    << Instruction::getOpcodeName(Opcode));
  llvm_unreachable("Unhandled opcode!");
}

Type *VPTypeAnalysis::inferScalarType(const VPWidenCallRecipe *R) {
  auto &CI = *cast<CallInst>(R->getUnderlyingInstr());
  return CI.getType();
}

Type *VPTypeAnalysis::inferScalarType(const VPWidenMemoryInstructionRecipe *R) {
  assert(!R->isStore() && "Store recipes should not define any values");
  return cast<LoadInst>(&R->getIngredient())->getType();
}

Type *VPTypeAnalysis::inferScalarType(const VPWidenSelectRecipe *R) {
  Type *ResTy = inferScalarType(R->getOperand(1));
  VPValue *OtherV = R->getOperand(2);
  assert(inferScalarType(OtherV) == ResTy &&
         "different types inferred for different operands");
  CachedTypes[OtherV] = ResTy;
  return ResTy;
}

Type *VPTypeAnalysis::inferScalarType(const VPReplicateRecipe *R) {
  switch (R->getUnderlyingInstr()->getOpcode()) {
  case Instruction::Call: {
    unsigned CallIdx = R->getNumOperands() - (R->isPredicated() ? 2 : 1);
    return cast<Function>(R->getOperand(CallIdx)->getLiveInIRValue())
        ->getReturnType();
  }
  case Instruction::UDiv:
  case Instruction::SDiv:
  case Instruction::SRem:
  case Instruction::URem:
  case Instruction::Add:
  case Instruction::FAdd:
  case Instruction::Sub:
  case Instruction::FSub:
  case Instruction::Mul:
  case Instruction::FMul:
  case Instruction::FDiv:
  case Instruction::FRem:
  case Instruction::Shl:
  case Instruction::LShr:
  case Instruction::AShr:
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor: {
    Type *ResTy = inferScalarType(R->getOperand(0));
    assert(ResTy == inferScalarType(R->getOperand(1)) &&
           "inferred types for operands of binary op don't match");
    CachedTypes[R->getOperand(1)] = ResTy;
    return ResTy;
  }
  case Instruction::Select: {
    Type *ResTy = inferScalarType(R->getOperand(1));
    assert(ResTy == inferScalarType(R->getOperand(2)) &&
           "inferred types for operands of select op don't match");
    CachedTypes[R->getOperand(2)] = ResTy;
    return ResTy;
  }
  case Instruction::ICmp:
  case Instruction::FCmp:
    return IntegerType::get(Ctx, 1);
  case Instruction::Alloca:
  case Instruction::BitCast:
  case Instruction::Trunc:
  case Instruction::SExt:
  case Instruction::ZExt:
  case Instruction::FPExt:
  case Instruction::FPTrunc:
  case Instruction::ExtractValue:
  case Instruction::SIToFP:
  case Instruction::UIToFP:
  case Instruction::FPToSI:
  case Instruction::FPToUI:
    return R->getUnderlyingInstr()->getType();
  case Instruction::Freeze:
  case Instruction::FNeg:
  case Instruction::GetElementPtr:
    return inferScalarType(R->getOperand(0));
  case Instruction::Load:
    return cast<LoadInst>(R->getUnderlyingInstr())->getType();
  default:
    break;
  }

  llvm_unreachable("Unhandled instruction");
}

Type *VPTypeAnalysis::inferScalarType(const VPValue *V) {
  if (Type *CachedTy = CachedTypes.lookup(V))
    return CachedTy;

  if (V->isLiveIn())
    return V->getLiveInIRValue()->getType();

  Type *ResultTy =
      TypeSwitch<const VPRecipeBase *, Type *>(V->getDefiningRecipe())
          .Case<VPCanonicalIVPHIRecipe, VPFirstOrderRecurrencePHIRecipe,
                VPReductionPHIRecipe, VPWidenPointerInductionRecipe>(
              [this](const auto *R) {
                // Handle header phi recipes, except VPWienIntOrFpInduction
                // which needs special handling due it being possibly truncated.
                return inferScalarType(R->getStartValue());
              })
          .Case<VPWidenIntOrFpInductionRecipe>(
              [](const VPWidenIntOrFpInductionRecipe *R) {
                return R->getScalarType();
              })
          .Case<VPDerivedIVRecipe>([this](const VPDerivedIVRecipe *R) {
            // VPDerivedIV may truncate the IV to a specified scalar type or use
            // the
            // type of the first operand (the step).
            Type *T = R->getScalarType();
            return T ? T : inferScalarType(R->getOperand(0));
          })
          .Case<VPPredInstPHIRecipe, VPWidenPHIRecipe, VPScalarIVStepsRecipe,
                VPWidenGEPRecipe>([this](const VPRecipeBase *R) {
            return inferScalarType(R->getOperand(0));
          })
          .Case<VPBlendRecipe, VPInstruction, VPWidenRecipe, VPReplicateRecipe,
                VPWidenCallRecipe, VPWidenMemoryInstructionRecipe,
                VPWidenSelectRecipe>(
              [this](const auto *R) { return inferScalarType(R); })
          .Case<VPInterleaveRecipe>([V](const VPInterleaveRecipe *R) {
            // TODO: Use info from interleave group.
            return V->getUnderlyingValue()->getType();
          })
          .Case<VPWidenCastRecipe>(
              [](const VPWidenCastRecipe *R) { return R->getResultType(); });
  assert(ResultTy && "could not infer type for the given VPValue");
  CachedTypes[V] = ResultTy;
  return ResultTy;
}
