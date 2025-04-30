//===-- Next32TargetTransformInfo.cpp - Next32 specific TTI ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements a TargetTransformInfo analysis pass specific to the
/// Next32 target machine. It uses the target's detailed information to provide
/// more precise answers to certain TTI queries, while letting the target
/// independent and default TTI implementations handle the rest.
///
//===----------------------------------------------------------------------===//

#include "Next32TargetTransformInfo.h"
#include "TargetInfo/Next32BaseInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/BasicTTIImpl.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "next32tti"

static cl::opt<bool>
    Next32Vectorization("next32-vectorization", cl::init(true), cl::Hidden,
                        cl::desc("Enable vectorization for Next32"));

bool Next32TTIImpl::enableVectorization() const {
  return ST->hasVectorInst() && Next32Vectorization;
}

bool Next32TTIImpl::isValidVectorLoadStore(FixedVectorType *VTy) const {
  Type *ElemTy = VTy->getElementType();

  // Avoid types like <2 x i32*>.
  // TODO: Maybe use TLI->getValueType(DL, Src) instead of getEVT?
  // It can handle pointer types, but the question is whether we want
  // to handle vector of pointers.
  if (!ElemTy->isFloatingPointTy() && !ElemTy->isIntegerTy())
    return false;

  return Next32Helpers::IsValidVectorTy(EVT::getEVT(VTy));
}

unsigned Next32TTIImpl::getNumberOfRegisters(unsigned ClassID) const {
  bool Vector = (ClassID == 1);
  if (Vector)
    return enableVectorization() ? 32 : 0;
  return 32;
}

TypeSize
Next32TTIImpl::getRegisterBitWidth(TargetTransformInfo::RegisterKind K) const {
  switch (K) {
  case TargetTransformInfo::RGK_Scalar:
    return TypeSize::getFixed(32);
  case TargetTransformInfo::RGK_FixedWidthVector:
    return TypeSize::getFixed(enableVectorization() ? 512 : 0);
  case TargetTransformInfo::RGK_ScalableVector:
    return TypeSize::getScalable(0);
  }

  llvm_unreachable("Unsupported register kind");
}

bool Next32TTIImpl::enableInterleavedAccessVectorization() { return true; }

bool Next32TTIImpl::shouldForceVectorizeInst(Instruction *I, Type *Ty) const {
  if (I->getOpcode() != Instruction::Load &&
      I->getOpcode() != Instruction::Store)
    return false;

  FixedVectorType *VTy = dyn_cast<FixedVectorType>(Ty);
  if (!VTy)
    return false;

  return isValidVectorLoadStore(VTy);
}

InstructionCost Next32TTIImpl::getMemoryOpCost(unsigned Opcode, Type *Src,
                                               MaybeAlign Alignment,
                                               unsigned AddressSpace,
                                               TTI::TargetCostKind CostKind,
                                               TTI::OperandValueInfo OpInfo,
                                               const Instruction *I) {
  assert((Opcode == Instruction::Load || Opcode == Instruction::Store) &&
         "Invalid Opcode");

  // TODO: Handle other cost kinds.
  if (CostKind != TTI::TCK_RecipThroughput)
    return BaseT::getMemoryOpCost(Opcode, Src, Alignment, AddressSpace,
                                  CostKind, OpInfo, I);

  // Type legalization can't handle structs
  if (TLI->getValueType(DL, Src, true) == MVT::Other)
    return BaseT::getMemoryOpCost(Opcode, Src, Alignment, AddressSpace,
                                  CostKind, OpInfo, I);

  if (FixedVectorType *VTy = dyn_cast<FixedVectorType>(Src)) {
    if (isValidVectorLoadStore(VTy))
      return 1;

    // Set higher cost for vector types that we don't support.
    return VTy->getNumElements() * 16;
  }

  return 1;
}

InstructionCost Next32TTIImpl::getInterleavedMemoryOpCost(
    unsigned Opcode, Type *VecTy, unsigned Factor, ArrayRef<unsigned> Indices,
    Align Alignment, unsigned AddressSpace, TTI::TargetCostKind CostKind,
    bool UseMaskForCond, bool UseMaskForGaps) {
  auto *VecVTy = cast<FixedVectorType>(VecTy);

  if (!UseMaskForCond && !UseMaskForGaps && isValidVectorLoadStore(VecVTy))
    return 1;

  return BaseT::getInterleavedMemoryOpCost(Opcode, VecTy, Factor, Indices,
                                           Alignment, AddressSpace, CostKind,
                                           UseMaskForCond, UseMaskForGaps);
}

InstructionCost Next32TTIImpl::getShuffleCost(TTI::ShuffleKind Kind,
                                              VectorType *BaseTp,
                                              ArrayRef<int> Mask,
                                              TTI::TargetCostKind CostKind,
                                              int Index, VectorType *SubTp,
                                              ArrayRef<const Value *> Args,
                                              const Instruction *CxtI) {
  return 0;
}

InstructionCost Next32TTIImpl::getVectorInstrCost(unsigned Opcode, Type *Val,
                                                  TTI::TargetCostKind CostKind,
                                                  unsigned Index, Value *Op0,
                                                  Value *Op1) {
  if (Opcode == Instruction::InsertElement ||
      Opcode == Instruction::ExtractElement)
    return 0;
  return BaseT::getVectorInstrCost(Opcode, Val, CostKind, Index, Op0, Op1);
}

bool Next32TTIImpl::hasDivRemOp(Type *DataType, bool IsSigned) const {
  // Can't call isOperationLegal because I64 are not legal in next32.
  // Falling back to only check isOperationCustom.
  EVT VT = TLI->getValueType(DL, DataType);
  return TLI->isOperationCustom(IsSigned ? ISD::SDIVREM : ISD::UDIVREM, VT);
}
