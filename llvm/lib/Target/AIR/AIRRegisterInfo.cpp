//===-- AIRRegisterInfo.cpp - RegisterInfo for AIR -*- C++ ---------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the AIR specific subclass of TargetRegisterInfo.
//
//===----------------------------------------------------------------------===//

#include "AIRRegisterInfo.h"
#include "MCTargetDesc/AIRMCTargetDesc.h"
#include "AIRFrameLowering.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"

#define GET_REGINFO_TARGET_DESC
#include "AIRGenRegisterInfo.inc"

using namespace llvm;

AIRRegisterInfo::~AIRRegisterInfo() {}

const MCPhysReg *
AIRRegisterInfo::getCalleeSavedRegs(const MachineFunction *MF) const {
  return nullptr;
}

BitVector AIRRegisterInfo::getReservedRegs(const MachineFunction &MF) const {
  return BitVector(getNumRegs());
}

bool AIRRegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator II,
                                            int SPAdj, unsigned FIOperandNum,
                                            RegScavenger *RS) const {
  return false;
}

Register AIRRegisterInfo::getFrameRegister(const MachineFunction &MF) const {
  return Register();
}
