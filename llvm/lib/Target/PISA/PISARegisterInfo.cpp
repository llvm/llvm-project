//===-- PISARegisterInfo.cpp - PISA Register Information ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PISARegisterInfo.h"
#include "PISA.h"
#include "PISASubtarget.h"
#include "llvm/CodeGen/MachineFunction.h"

#define GET_REGINFO_TARGET_DESC
#include "PISAGenRegisterInfo.inc"

using namespace llvm;

PISARegisterInfo::PISARegisterInfo() : PISAGenRegisterInfo(PISA::DummyReg) {}

const MCPhysReg *
PISARegisterInfo::getCalleeSavedRegs(const MachineFunction *MF) const {
  static const MCPhysReg CalleeSavedRegs[] = {0};
  return CalleeSavedRegs;
}

BitVector PISARegisterInfo::getReservedRegs(const MachineFunction &MF) const {
  return BitVector(getNumRegs());
}
