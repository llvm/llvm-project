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
#include "MCTargetDesc/SPIRVMCTargetDesc.h"
#include "SPIRVInstrInfo.h"

#include "SPIRV.h"

namespace llvm {
[[maybe_unused]] static bool definesATypeRegister(const MachineInstr &MI) {
  const MachineRegisterInfo &MRI = MI.getMF()->getRegInfo();
  return MRI.getRegClass(MI.getOperand(0).getReg()) == &SPIRV::TYPERegClass;
}

SPIRVTypeInst::SPIRVTypeInst(const MachineInstr *MI) : MI(MI) {
  // A SPIRV Type whose result is not a type is invalid.
  assert(!MI || definesATypeRegister(*MI));
}

bool SPIRVTypeInst::isTypeIntN(unsigned N) const {
  if (MI->getOpcode() != SPIRV::OpTypeInt)
    return false;
  if (N)
    return MI->getOperand(1).getImm() == N;
  return true;
}

bool SPIRVTypeInst::isAnyTypeFloat() const {
  return MI->getOpcode() == SPIRV::OpTypeFloat;
}
} // namespace llvm
