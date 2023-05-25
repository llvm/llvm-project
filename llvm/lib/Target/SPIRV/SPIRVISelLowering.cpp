//===- SPIRVISelLowering.cpp - SPIR-V DAG Lowering Impl ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the SPIRVTargetLowering class.
//
//===----------------------------------------------------------------------===//

#include "SPIRVISelLowering.h"
#include "SPIRV.h"
#include "llvm/IR/IntrinsicsSPIRV.h"

#define DEBUG_TYPE "spirv-lower"

using namespace llvm;

unsigned SPIRVTargetLowering::getNumRegistersForCallingConv(
    LLVMContext &Context, CallingConv::ID CC, EVT VT) const {
  // This code avoids CallLowering fail inside getVectorTypeBreakdown
  // on v3i1 arguments. Maybe we need to return 1 for all types.
  // TODO: remove it once this case is supported by the default implementation.
  if (VT.isVector() && VT.getVectorNumElements() == 3 &&
      (VT.getVectorElementType() == MVT::i1 ||
       VT.getVectorElementType() == MVT::i8))
    return 1;
  if (!VT.isVector() && VT.isInteger() && VT.getSizeInBits() <= 64)
    return 1;
  return getNumRegisters(Context, VT);
}

MVT SPIRVTargetLowering::getRegisterTypeForCallingConv(LLVMContext &Context,
                                                       CallingConv::ID CC,
                                                       EVT VT) const {
  // This code avoids CallLowering fail inside getVectorTypeBreakdown
  // on v3i1 arguments. Maybe we need to return i32 for all types.
  // TODO: remove it once this case is supported by the default implementation.
  if (VT.isVector() && VT.getVectorNumElements() == 3) {
    if (VT.getVectorElementType() == MVT::i1)
      return MVT::v4i1;
    else if (VT.getVectorElementType() == MVT::i8)
      return MVT::v4i8;
  }
  return getRegisterType(Context, VT);
}

bool SPIRVTargetLowering::getTgtMemIntrinsic(IntrinsicInfo &Info,
                                             const CallInst &I,
                                             MachineFunction &MF,
                                             unsigned Intrinsic) const {
  unsigned AlignIdx = 3;
  switch (Intrinsic) {
  case Intrinsic::spv_load:
    AlignIdx = 2;
    [[fallthrough]];
  case Intrinsic::spv_store: {
    if (I.getNumOperands() >= AlignIdx + 1) {
      auto *AlignOp = cast<ConstantInt>(I.getOperand(AlignIdx));
      Info.align = Align(AlignOp->getZExtValue());
    }
    Info.flags = static_cast<MachineMemOperand::Flags>(
        cast<ConstantInt>(I.getOperand(AlignIdx - 1))->getZExtValue());
    Info.memVT = MVT::i64;
    // TODO: take into account opaque pointers (don't use getElementType).
    // MVT::getVT(PtrTy->getElementType());
    return true;
    break;
  }
  default:
    break;
  }
  return false;
}
