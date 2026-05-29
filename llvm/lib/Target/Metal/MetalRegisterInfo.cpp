//===-- MetalRegisterInfo.cpp - RegisterInfo for Metal -*- C++ ---------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the Metal specific subclass of TargetRegisterInfo.
//
//===----------------------------------------------------------------------===//

#include "MetalRegisterInfo.h"
#include "MCTargetDesc/MetalMCTargetDesc.h"
#include "MetalFrameLowering.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"

#define GET_REGINFO_TARGET_DESC
#include "MetalGenRegisterInfo.inc"

using namespace llvm;

MetalRegisterInfo::~MetalRegisterInfo() {}

const MCPhysReg *
MetalRegisterInfo::getCalleeSavedRegs(const MachineFunction *MF) const {
  return nullptr;
}

BitVector MetalRegisterInfo::getReservedRegs(const MachineFunction &MF) const {
  return BitVector(getNumRegs());
}

bool MetalRegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator II,
                                            int SPAdj, unsigned FIOperandNum,
                                            RegScavenger *RS) const {
  return false;
}

Register MetalRegisterInfo::getFrameRegister(const MachineFunction &MF) const {
  return Register();
}
