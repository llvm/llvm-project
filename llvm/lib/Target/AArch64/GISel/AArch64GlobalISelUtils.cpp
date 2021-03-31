//===- AArch64GlobalISelUtils.cpp --------------------------------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file Implementations of AArch64-specific helper functions used in the
/// GlobalISel pipeline.
//===----------------------------------------------------------------------===//
#include "AArch64GlobalISelUtils.h"
#include "AArch64InstrInfo.h"
#include "llvm/CodeGen/GlobalISel/Utils.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

Optional<RegOrConstant>
AArch64GISelUtils::getAArch64VectorSplat(const MachineInstr &MI,
                                         const MachineRegisterInfo &MRI) {
  if (auto Splat = getVectorSplat(MI, MRI))
    return Splat;
  if (MI.getOpcode() != AArch64::G_DUP)
    return None;
  Register Src = MI.getOperand(1).getReg();
  if (auto ValAndVReg =
          getConstantVRegValWithLookThrough(MI.getOperand(1).getReg(), MRI))
    return RegOrConstant(ValAndVReg->Value.getSExtValue());
  return RegOrConstant(Src);
}

Optional<int64_t> AArch64GISelUtils::getAArch64VectorSplatScalar(
    const MachineInstr &MI, const MachineRegisterInfo &MRI) {
  auto Splat = getAArch64VectorSplat(MI, MRI);
  if (!Splat || Splat->isReg())
    return None;
  return Splat->getCst();
}
