//===-- NanoMipsTargetTransformInfo.cpp - nanoMIPS specific TTI -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NanoMipsTargetTransformInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/BasicTTIImpl.h"
#include "llvm/CodeGen/TargetLowering.h"

using namespace llvm;

#define DEBUG_TYPE "nanomipstti"

InstructionCost NanoMipsTTIImpl::getIntImmCost(const APInt &Imm, Type *Ty,
                                               TTI::TargetCostKind CostKind) {
  if (Ty->getIntegerBitWidth() > 64) {
    return 4 * TTI::TCC_Basic;
  }

  uint64_t ImmZExt = Imm.getZExtValue();
  int64_t ImmSExt = Imm.getSExtValue();

  if (CostKind == TTI::TCK_CodeSize) {

    /* Any large constant. */
    if (Ty->getIntegerBitWidth() > 32)
      return 4 * TTI::TCC_Basic;

    /* 16-bit LI */
    if (ImmSExt >= -1 && ImmSExt <= 126)
      return TTI::TCC_Basic;

    /* 32-bit LI with 16-bit immediate */
    if (isUInt<16>(ImmSExt))
      return 2 * TTI::TCC_Basic;

    // This can be loaded with LUI.
    if ((ImmZExt & 0xfff) == 0)
      return 2 * TTI::TCC_Basic;

    // ADDIU can handle 12-bit negative immediates.
    if (isUInt<12>(-ImmSExt))
      return 2 * TTI::TCC_Basic;

    // 48-bit LI covers all 32-bit constants
    if (isUInt<32>(ImmZExt))
      return 3 * TTI::TCC_Basic;

  } else {
    // Throughput-related costs

    if (isUInt<16>(ImmZExt))
      return TTI::TCC_Basic;

    if (isUInt<32>(ImmZExt))
      return TTI::TCC_Basic * 1.5;
  }

  return 2 * TTI::TCC_Basic;
}

InstructionCost NanoMipsTTIImpl::getIntImmCostInst(unsigned Opcode,
                                                   unsigned Idx,
                                                   const APInt &Imm, Type *Ty,
                                                   TTI::TargetCostKind CostKind,
                                                   Instruction *Inst) {
  switch (Opcode) {
  case Instruction::PHI:
    /* Phi doesn't have any other instruction that an immediate can be absorbed into. */
    return getIntImmCost(Imm, Ty, CostKind);
  default:
    /* Most instructions will be able to use zero register */
    if (Imm == 0)
      return TTI::TCC_Free;

    if (Ty->getIntegerBitWidth() > 64)
      return getIntImmCost(Imm, Ty, CostKind);

    uint64_t ImmZExt = Imm.getZExtValue();

    // Most instructions can handle 12-bit unsigned immediates.
    if (isUInt<12>(ImmZExt))
      return TTI::TCC_Free;

  }

  return getIntImmCost(Imm, Ty, CostKind);
}

void NanoMipsTTIImpl::getUnrollingPreferences(Loop *L, ScalarEvolution &SE,
                                              TTI::UnrollingPreferences &UP) {
  BaseT::getUnrollingPreferences(L, SE, UP);
  UP.Threshold = 60;
  UP.OptSizeThreshold = 0;
}


static InstructionCost selectCost(const Value *Cond,
                                  const Value *A, const Value *B) {
  const ConstantInt *AI = dyn_cast<ConstantInt>(A),
    *BI = dyn_cast<ConstantInt>(B);

  if (AI && BI) {
    uint64_t AV = AI->getZExtValue(), BV = BI->getZExtValue();
    if ((AV == 0 && BV == 1) || (AV == 1 && BV == 0)) {
      return TTI::TCC_Free;
    }
  }
  return TTI::TCC_Basic;
}

/// Cost for compare and select. When selecting between constant 0 and
/// 1 values, this can be implemented as just a comparison, making the
/// selection free.
InstructionCost NanoMipsTTIImpl::getCmpSelInstrCost (
    unsigned Opcode, Type *ValTy, Type *CondTy, CmpInst::Predicate VecPred,
    TTI::TargetCostKind CostKind,
    const Instruction *I,
    ArrayRef<const Value *> Operands) const {

  if (I != nullptr) {
    // Decode compare and select
    if (I->getOpcode() == Instruction::Select) {
      return selectCost(I->getOperand(0), I->getOperand(1), I->getOperand(2));
    }
  } else if (Opcode == Instruction::Select && Operands.size() != 0) {
    assert(Operands.size() == 3);
    return selectCost(Operands[0], Operands[1], Operands[2]);
  }

  return TTI::TCC_Basic;
}
