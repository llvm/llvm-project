//===- VPlanAnalysis.cpp - Various Analyses working on VPlan ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "VPlanAnalysis.h"
#include "VPlan.h"

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
  return inferScalarType(R->getIncomingValue(0));
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
    llvm_unreachable("Unhandled instruction!");
  }
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
  if (R->isStore())
    return cast<StoreInst>(&R->getIngredient())->getValueOperand()->getType();

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
  case Instruction::Xor:
  case Instruction::ICmp:
  case Instruction::FCmp: {
    Type *ResTy = inferScalarType(R->getOperand(0));
    assert(ResTy == inferScalarType(R->getOperand(1)));
    CachedTypes[R->getOperand(1)] = ResTy;
    return ResTy;
  }
  case Instruction::Trunc:
  case Instruction::SExt:
  case Instruction::ZExt:
  case Instruction::FPExt:
  case Instruction::FPTrunc:
    return R->getUnderlyingInstr()->getType();
  case Instruction::ExtractValue: {
    return R->getUnderlyingInstr()->getType();
  }
  case Instruction::Freeze:
  case Instruction::FNeg:
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

  Type *ResultTy = nullptr;
  if (V->isLiveIn())
    return V->getLiveInIRValue()->getType();

    const VPRecipeBase *Def = V->getDefiningRecipe();
    switch (Def->getVPDefID()) {
    case VPDef::VPCanonicalIVPHISC:
    case VPDef::VPFirstOrderRecurrencePHISC:
    case VPDef::VPReductionPHISC:
    case VPDef::VPWidenPointerInductionSC:
    // Handle header phi recipes, except VPWienIntOrFpInduction which needs
    // special handling due it being possibly truncated.
    ResultTy = cast<VPHeaderPHIRecipe>(Def)
                   ->getStartValue()
                   ->getLiveInIRValue()
                   ->getType();
    break;
    case VPDef::VPWidenIntOrFpInductionSC:
    ResultTy = cast<VPWidenIntOrFpInductionRecipe>(Def)->getScalarType();
    break;
    case VPDef::VPPredInstPHISC:
    case VPDef::VPScalarIVStepsSC:
    case VPDef::VPWidenPHISC:
    ResultTy = inferScalarType(Def->getOperand(0));
    break;
    case VPDef::VPBlendSC:
    ResultTy = inferScalarType(cast<VPBlendRecipe>(Def));
    break;
    case VPDef::VPInstructionSC:
    ResultTy = inferScalarType(cast<VPInstruction>(Def));
    break;
    case VPDef::VPInterleaveSC:
    // TODO: Use info from interleave group.
    ResultTy = V->getUnderlyingValue()->getType();
    break;
    case VPDef::VPReplicateSC:
    ResultTy = inferScalarType(cast<VPReplicateRecipe>(Def));
    break;
    case VPDef::VPWidenSC:
    ResultTy = inferScalarType(cast<VPWidenRecipe>(Def));
    break;
    case VPDef::VPWidenCallSC:
    ResultTy = inferScalarType(cast<VPWidenCallRecipe>(Def));
    break;
    case VPDef::VPWidenCastSC:
      ResultTy = cast<VPWidenCastRecipe>(Def)->getResultType();
      break;
    case VPDef::VPWidenGEPSC:
      ResultTy = PointerType::get(Ctx, 0);
      break;
    case VPDef::VPWidenMemoryInstructionSC:
      ResultTy = inferScalarType(cast<VPWidenMemoryInstructionRecipe>(Def));
      break;
    case VPDef::VPWidenSelectSC:
      ResultTy = inferScalarType(cast<VPWidenSelectRecipe>(Def));
      break;
    }
    assert(ResultTy && "could not infer type for the given VPValue");
    CachedTypes[V] = ResultTy;
    return ResultTy;
}
