//===-- ConnexFrameLowering.cpp - Connex Frame Information ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the Connex implementation of TargetFrameLowering class.
//
//===----------------------------------------------------------------------===//

#include "ConnexFrameLowering.h"
#include "ConnexInstrInfo.h"
#include "ConnexSubtarget.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"

using namespace llvm;

bool ConnexFrameLowering::hasFPImpl(const MachineFunction &MF) const {
  return true;
}

void ConnexFrameLowering::emitPrologue(MachineFunction &MF,
                                       MachineBasicBlock &MBB) const {}

void ConnexFrameLowering::emitEpilogue(MachineFunction &MF,
                                       MachineBasicBlock &MBB) const {}

void ConnexFrameLowering::determineCalleeSaves(MachineFunction &MF,
                                               BitVector &SavedRegs,
                                               RegScavenger *RS) const {
  TargetFrameLowering::determineCalleeSaves(MF, SavedRegs, RS);
  SavedRegs.reset(Connex::R6);
  SavedRegs.reset(Connex::R7);
  SavedRegs.reset(Connex::R8);
  SavedRegs.reset(Connex::R9);
}
