//===-- SPIRVTypeInst.cpp - SPIR-V Type Instruction -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation associated to SPIRVTypeInst.h.
//
//===----------------------------------------------------------------------===//

#include "SPIRVTypeInst.h"
#include "SPIRVInstrInfo.h"

namespace llvm {
static bool definesATypeRegister(const MachineInstr &MI) {
  const MachineRegisterInfo &MRI = MI.getMF()->getRegInfo();
  return MRI.getRegClass(MI.getOperand(0).getReg()) == &SPIRV::TYPERegClass;
}

SPIRVTypeInst::SPIRVTypeInst(const MachineInstr *MI) : MI(MI) {
  // A SPIRV Type whose result is not a type is invalid.
  assert(!MI || definesATypeRegister(*MI));
}
} // namespace llvm
