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

Type *VPTypeAnalysis::inferType(const VPBlendRecipe *R) {
  return inferType(R->getIncomingValue(0));
}

Type *VPTypeAnalysis::inferType(const VPInstruction *R) {
  switch (R->getOpcode()) {
  case Instruction::Select:
    return inferType(R->getOperand(1));
  case VPInstruction::FirstOrderRecurrenceSplice:
    return inferType(R->getOperand(0));
  default:
    llvm_unreachable("Unhandled instruction!");
  }
}

Type *VPTypeAnalysis::inferType(const VPInterleaveRecipe *R) { return nullptr; }

Type *VPTypeAnalysis::inferType(const VPReductionPHIRecipe *R) {
  return R->getOperand(0)->getLiveInIRValue()->getType();
}

Type *VPTypeAnalysis::inferType(const VPWidenRecipe *R) {
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
  case Instruction::FNeg:
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
    Type *ResTy = inferType(R->getOperand(0));
    if (Opcode != Instruction::FNeg) {
      assert(ResTy == inferType(R->getOperand(1)));
      CachedTypes[R->getOperand(1)] = ResTy;
    }
    return ResTy;
  }
  case Instruction::Freeze:
    return inferType(R->getOperand(0));
  default:
    // This instruction is not vectorized by simple widening.
    //    LLVM_DEBUG(dbgs() << "LV: Found an unhandled instruction: " << I);
    llvm_unreachable("Unhandled instruction!");
  }

  return nullptr;
}

Type *VPTypeAnalysis::inferType(const VPWidenCallRecipe *R) {
  auto &CI = *cast<CallInst>(R->getUnderlyingInstr());
  return CI.getType();
}

Type *VPTypeAnalysis::inferType(const VPWidenIntOrFpInductionRecipe *R) {
  return R->getScalarType();
}

Type *VPTypeAnalysis::inferType(const VPWidenMemoryInstructionRecipe *R) {
  if (R->isStore())
    return cast<StoreInst>(&R->getIngredient())->getValueOperand()->getType();

  return cast<LoadInst>(&R->getIngredient())->getType();
}

Type *VPTypeAnalysis::inferType(const VPWidenSelectRecipe *R) {
  return inferType(R->getOperand(1));
}

Type *VPTypeAnalysis::inferType(const VPReplicateRecipe *R) {
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
  case Instruction::FNeg:
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
    Type *ResTy = inferType(R->getOperand(0));
    assert(ResTy == inferType(R->getOperand(1)));
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
    return R->getUnderlyingValue()->getType();
  }
  case Instruction::Freeze:
    return inferType(R->getOperand(0));
  case Instruction::Load:
    return cast<LoadInst>(R->getUnderlyingInstr())->getType();
  default:
    llvm_unreachable("Unhandled instruction");
  }

  return nullptr;
}

Type *VPTypeAnalysis::inferType(const VPValue *V) {
  auto Iter = CachedTypes.find(V);
  if (Iter != CachedTypes.end())
    return Iter->second;

  Type *ResultTy = nullptr;
  if (V->isLiveIn())
    ResultTy = V->getLiveInIRValue()->getType();
  else {
    const VPRecipeBase *Def = V->getDefiningRecipe();
    switch (Def->getVPDefID()) {
    case VPDef::VPBlendSC:
      ResultTy = inferType(cast<VPBlendRecipe>(Def));
      break;
    case VPDef::VPCanonicalIVPHISC:
      ResultTy = cast<VPCanonicalIVPHIRecipe>(Def)->getScalarType();
      break;
    case VPDef::VPFirstOrderRecurrencePHISC:
      ResultTy = Def->getOperand(0)->getLiveInIRValue()->getType();
      break;
    case VPDef::VPInstructionSC:
      ResultTy = inferType(cast<VPInstruction>(Def));
      break;
    case VPDef::VPInterleaveSC:
      ResultTy = V->getUnderlyingValue()
                     ->getType(); // inferType(cast<VPInterleaveRecipe>(Def));
      break;
    case VPDef::VPPredInstPHISC:
      ResultTy = inferType(Def->getOperand(0));
      break;
    case VPDef::VPReductionPHISC:
      ResultTy = inferType(cast<VPReductionPHIRecipe>(Def));
      break;
    case VPDef::VPReplicateSC:
      ResultTy = inferType(cast<VPReplicateRecipe>(Def));
      break;
    case VPDef::VPScalarIVStepsSC:
      return inferType(Def->getOperand(0));
      break;
    case VPDef::VPWidenSC:
      ResultTy = inferType(cast<VPWidenRecipe>(Def));
      break;
    case VPDef::VPWidenPHISC:
      return inferType(Def->getOperand(0));
    case VPDef::VPWidenPointerInductionSC:
      return inferType(Def->getOperand(0));
    case VPDef::VPWidenCallSC:
      ResultTy = inferType(cast<VPWidenCallRecipe>(Def));
      break;
    case VPDef::VPWidenCastSC:
      ResultTy = cast<VPWidenCastRecipe>(Def)->getResultType();
      break;
    case VPDef::VPWidenGEPSC:
      ResultTy = PointerType::get(Ctx, 0);
      break;
    case VPDef::VPWidenIntOrFpInductionSC:
      ResultTy = inferType(cast<VPWidenIntOrFpInductionRecipe>(Def));
      break;
    case VPDef::VPWidenMemoryInstructionSC:
      ResultTy = inferType(cast<VPWidenMemoryInstructionRecipe>(Def));
      break;
    case VPDef::VPWidenSelectSC:
      ResultTy = inferType(cast<VPWidenSelectRecipe>(Def));
      break;
    }
  }
  CachedTypes[V] = ResultTy;
  return ResultTy;
}
