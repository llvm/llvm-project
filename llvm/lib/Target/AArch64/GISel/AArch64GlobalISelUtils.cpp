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
#include "llvm/CodeGen/TargetLowering.h"
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

Optional<int64_t>
AArch64GISelUtils::getAArch64VectorSplatScalar(const MachineInstr &MI,
                                               const MachineRegisterInfo &MRI) {
  auto Splat = getAArch64VectorSplat(MI, MRI);
  if (!Splat || Splat->isReg())
    return None;
  return Splat->getCst();
}

bool AArch64GISelUtils::isCMN(const MachineInstr *MaybeSub,
                              const CmpInst::Predicate &Pred,
                              const MachineRegisterInfo &MRI) {
  // Match:
  //
  // %sub = G_SUB 0, %y
  // %cmp = G_ICMP eq/ne, %sub, %z
  //
  // Or
  //
  // %sub = G_SUB 0, %y
  // %cmp = G_ICMP eq/ne, %z, %sub
  if (!MaybeSub || MaybeSub->getOpcode() != TargetOpcode::G_SUB ||
      !CmpInst::isEquality(Pred))
    return false;
  auto MaybeZero =
      getConstantVRegValWithLookThrough(MaybeSub->getOperand(1).getReg(), MRI);
  return MaybeZero && MaybeZero->Value.getZExtValue() == 0;
}

bool AArch64GISelUtils::tryEmitBZero(MachineInstr &MI,
                                     MachineIRBuilder &MIRBuilder,
                                     bool MinSize) {
  assert(MI.getOpcode() == TargetOpcode::G_MEMSET);
  MachineRegisterInfo &MRI = *MIRBuilder.getMRI();
  auto &TLI = *MIRBuilder.getMF().getSubtarget().getTargetLowering();
  if (!TLI.getLibcallName(RTLIB::BZERO))
    return false;
  auto Zero = getConstantVRegValWithLookThrough(MI.getOperand(1).getReg(), MRI);
  if (!Zero || Zero->Value.getSExtValue() != 0)
    return false;

  // It's not faster to use bzero rather than memset for sizes <= 256.
  // However, it *does* save us a mov from wzr, so if we're going for
  // minsize, use bzero even if it's slower.
  if (!MinSize) {
    // If the size is known, check it. If it is not known, assume using bzero is
    // better.
    if (auto Size =
            getConstantVRegValWithLookThrough(MI.getOperand(2).getReg(), MRI)) {
      if (Size->Value.getSExtValue() <= 256)
        return false;
    }
  }

  MIRBuilder.setInstrAndDebugLoc(MI);
  MIRBuilder
      .buildInstr(TargetOpcode::G_BZERO, {},
                  {MI.getOperand(0), MI.getOperand(2)})
      .addImm(MI.getOperand(3).getImm())
      .addMemOperand(*MI.memoperands_begin());
  MI.eraseFromParent();
  return true;
}
