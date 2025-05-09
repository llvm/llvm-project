//===-- DirectXRegisterInfo.cpp - RegisterInfo for DirectX -*- C++ ------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the DirectX specific subclass of TargetRegisterInfo.
//
//===----------------------------------------------------------------------===//

#include "DirectXRegisterInfo.h"
#include "DirectXFrameLowering.h"
#include "MCTargetDesc/DirectXMCTargetDesc.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"

#define GET_REGINFO_TARGET_DESC
#include "DirectXGenRegisterInfo.inc"

using namespace llvm;

DirectXRegisterInfo::~DirectXRegisterInfo() {}

const MCPhysReg *
DirectXRegisterInfo::getCalleeSavedRegs(const MachineFunction *MF) const {
  return nullptr;
}
BitVector
DirectXRegisterInfo::getReservedRegs(const MachineFunction &MF) const {
  return BitVector(getNumRegs());
}

bool DirectXRegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator II,
                                              int SPAdj, unsigned FIOperandNum,
                                              RegScavenger *RS) const {
  return false;
}

// Debug information queries.
Register
DirectXRegisterInfo::getFrameRegister(const MachineFunction &MF) const {
  return Register();
}
