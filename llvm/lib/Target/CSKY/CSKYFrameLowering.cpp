//===-- CSKYFrameLowering.cpp - CSKY Frame Information ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the CSKY implementation of TargetFrameLowering class.
//
//===----------------------------------------------------------------------===//

#include "CSKYFrameLowering.h"
#include "CSKYSubtarget.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/MC/MCDwarf.h"

using namespace llvm;

#define DEBUG_TYPE "csky-frame-lowering"

// Returns the register used to hold the frame pointer.
static Register getFPReg(const CSKYSubtarget &STI) { return CSKY::R8; }

// To avoid the BP value clobbered by a function call, we need to choose a
// callee saved register to save the value.
static Register getBPReg(const CSKYSubtarget &STI) { return CSKY::R7; }

bool CSKYFrameLowering::hasFP(const MachineFunction &MF) const {
  const TargetRegisterInfo *RegInfo = MF.getSubtarget().getRegisterInfo();

  const MachineFrameInfo &MFI = MF.getFrameInfo();
  return MF.getTarget().Options.DisableFramePointerElim(MF) ||
         RegInfo->hasStackRealignment(MF) || MFI.hasVarSizedObjects() ||
         MFI.isFrameAddressTaken();
}

bool CSKYFrameLowering::hasBP(const MachineFunction &MF) const {
  const MachineFrameInfo &MFI = MF.getFrameInfo();

  return MFI.hasVarSizedObjects();
}

void CSKYFrameLowering::emitPrologue(MachineFunction &MF,
                                     MachineBasicBlock &MBB) const {
  // FIXME: Implement this when we have function calls
}

void CSKYFrameLowering::emitEpilogue(MachineFunction &MF,
                                     MachineBasicBlock &MBB) const {
  // FIXME: Implement this when we have function calls
}
