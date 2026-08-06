//===-- PISARegisterBankInfo.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PISARegisterBankInfo.h"
#include "PISARegisterInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterBank.h"

#define GET_REGINFO_ENUM
#include "PISAGenRegisterInfo.inc"

#define GET_TARGET_REGBANK_IMPL
#include "PISAGenRegisterBank.inc"

using namespace llvm;

const RegisterBankInfo::InstructionMapping &
PISARegisterBankInfo::getInstrMapping(const MachineInstr &MI) const {
  const RegisterBankInfo::InstructionMapping &Mapping = getInstrMappingImpl(MI);

  if (Mapping.isValid())
    return Mapping;

  const MachineFunction &MF = *MI.getParent()->getParent();
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  const TargetRegisterInfo *TRI = MRI.getTargetRegisterInfo();

  SmallVector<const ValueMapping *, 8> OpdsMapping(MI.getNumOperands());

  for (unsigned Idx = 0; Idx < MI.getNumOperands(); ++Idx) {
    auto &MO = MI.getOperand(Idx);

    if (MO.isReg() && MO.getReg().isValid()) {
      unsigned Size = getSizeInBits(MO.getReg(), MRI, *TRI);
      OpdsMapping[Idx] = &getValueMapping(0, Size, PISA::RegistersRegBank);
    }
  }

  return getInstructionMapping(DefaultMappingID, 1,
                               getOperandsMapping(OpdsMapping),
                               MI.getNumOperands());
}

const RegisterBank &PISARegisterBankInfo::getRegBankFromRegClass(
    const TargetRegisterClass & /* RC */, LLT /* Ty */) const {
  return PISA::RegistersRegBank;
}
