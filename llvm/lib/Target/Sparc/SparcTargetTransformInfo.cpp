//===-- SparcTargetTransformInfo.cpp - SPARC specific TTI -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SparcTargetTransformInfo.h"
#include "llvm/Support/MathExtras.h"

using namespace llvm;

#define DEBUG_TYPE "sparctti"

InstructionCost
SparcTTIImpl::getIntImmCost(const APInt &Imm, Type *Ty,
                            TTI::TargetCostKind CostKind) const {
  assert(Ty->isIntegerTy());
  unsigned BitSize = Ty->getPrimitiveSizeInBits();

  if (BitSize <= 64) {
    // Small constants can be folded into instruction's immediate field.
    if (isInt<13>(Imm.getSExtValue()))
      return TTI::TCC_Free;

    // Medium constants loaded via set.
    if (isUInt<32>(Imm.getZExtValue()))
      return TTI::TCC_Basic;

    // Large constants loaded via setx on 64-bit targets.
    if (ST->is64Bit())
      return 3 * TTI::TCC_Basic;
  }

  // Very large constants load from constant pool.
  return TTI::TCC_Expensive;
}

InstructionCost SparcTTIImpl::getIntImmCostInst(unsigned Opcode, unsigned Idx,
                                                const APInt &Imm, Type *Ty,
                                                TTI::TargetCostKind CostKind,
                                                Instruction *Inst) const {
  assert(Ty->isIntegerTy());

  switch (Opcode) {
  case Instruction::Shl:
  case Instruction::LShr:
  case Instruction::AShr:
    // Always return TCC_Free for the shift value of a shift instruction.
    return (Idx == 1) ? TTI::TCC_Free : TTI::TCC_Basic;
  }

  // All other instructions have the same 13-bit immediate field so we can just
  // pass it here.
  return getIntImmCost(Imm, Ty, CostKind);
}

TargetTransformInfo::PopcntSupportKind
SparcTTIImpl::getPopcntSupport(unsigned TyWidth) const {
  assert(isPowerOf2_32(TyWidth) && "Type width must be power of 2");
  if (ST->usePopc())
    return TTI::PSK_FastHardware;
  return TTI::PSK_Software;
}

unsigned SparcTTIImpl::getRegisterClassForType(bool Vector, Type *Ty) const {
  if (Vector)
    return FPRRC;
  if (Ty &&
      (Ty->getScalarType()->isFloatTy() || Ty->getScalarType()->isDoubleTy()))
    return FPRRC;
  if (Ty && (Ty->getScalarType()->isFP128Ty()))
    return FP128RRC;
  return GPRRC;
}

unsigned SparcTTIImpl::getNumberOfRegisters(unsigned ClassID) const {
  switch (ClassID) {
  case GPRRC:
    // %g0, %g6, %g7, %o6, %i6, and %i7 are used for special purposes so we
    // discount them here.
    return 26;
  case FPRRC:
    return 32;
  case FP128RRC:
    return 16;
  }

  llvm_unreachable("Unsupported register class");
}

TypeSize
SparcTTIImpl::getRegisterBitWidth(TargetTransformInfo::RegisterKind K) const {
  switch (K) {
  case TargetTransformInfo::RGK_Scalar:
    // TODO When targeting V8+ ABI, G and O registers are 64-bit.
    return TypeSize::getFixed(ST->is64Bit() ? 64 : 32);
  case TargetTransformInfo::RGK_FixedWidthVector:
    // TODO We have vector capabilities as part of the VIS extensions, but the
    // codegen doesn't currently use it. Revisit this when vector codegen is
    // ready.
    return TypeSize::getFixed(0);
  case TargetTransformInfo::RGK_ScalableVector:
    return TypeSize::getScalable(0);
  }

  llvm_unreachable("Unsupported register kind");
}
