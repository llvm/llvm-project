//===-- WebAssemblyTargetTransformInfo.cpp - WebAssembly-specific TTI -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the WebAssembly-specific TargetTransformInfo
/// implementation.
///
//===----------------------------------------------------------------------===//

#include "WebAssemblyTargetTransformInfo.h"
#include "llvm/CodeGen/CostTable.h"
#include "llvm/Support/Debug.h"
using namespace llvm;

#define DEBUG_TYPE "wasmtti"

TargetTransformInfo::PopcntSupportKind
WebAssemblyTTIImpl::getPopcntSupport(unsigned TyWidth) const {
  assert(isPowerOf2_32(TyWidth) && "Ty width must be power of 2");
  return TargetTransformInfo::PSK_FastHardware;
}

unsigned WebAssemblyTTIImpl::getNumberOfRegisters(unsigned ClassID) const {
  unsigned Result = BaseT::getNumberOfRegisters(ClassID);

  // For SIMD, use at least 16 registers, as a rough guess.
  bool Vector = (ClassID == 1);
  if (Vector)
    Result = std::max(Result, 16u);

  return Result;
}

unsigned WebAssemblyTTIImpl::getRegisterBitWidth(bool Vector) const {
  if (Vector && getST()->hasSIMD128())
    return 128;

  return 64;
}

unsigned WebAssemblyTTIImpl::getArithmeticInstrCost(
    unsigned Opcode, Type *Ty, TTI::OperandValueKind Opd1Info,
    TTI::OperandValueKind Opd2Info, TTI::OperandValueProperties Opd1PropInfo,
    TTI::OperandValueProperties Opd2PropInfo, ArrayRef<const Value *> Args,
    const Instruction *CxtI) {

  unsigned Cost = BasicTTIImplBase<WebAssemblyTTIImpl>::getArithmeticInstrCost(
      Opcode, Ty, Opd1Info, Opd2Info, Opd1PropInfo, Opd2PropInfo);

  if (auto *VTy = dyn_cast<VectorType>(Ty)) {
    switch (Opcode) {
    case Instruction::LShr:
    case Instruction::AShr:
    case Instruction::Shl:
      // SIMD128's shifts currently only accept a scalar shift count. For each
      // element, we'll need to extract, op, insert. The following is a rough
      // approxmation.
      if (Opd2Info != TTI::OK_UniformValue &&
          Opd2Info != TTI::OK_UniformConstantValue)
        Cost = VTy->getNumElements() *
               (TargetTransformInfo::TCC_Basic +
                getArithmeticInstrCost(Opcode, VTy->getElementType()) +
                TargetTransformInfo::TCC_Basic);
      break;
    }
  }
  return Cost;
}

unsigned WebAssemblyTTIImpl::getVectorInstrCost(unsigned Opcode, Type *Val,
                                                unsigned Index) {
  unsigned Cost = BasicTTIImplBase::getVectorInstrCost(Opcode, Val, Index);

  // SIMD128's insert/extract currently only take constant indices.
  if (Index == -1u)
    return Cost + 25 * TargetTransformInfo::TCC_Expensive;

  return Cost;
}
