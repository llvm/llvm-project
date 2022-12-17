//===- PPCRegisterBankInfo.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements the targeting of the RegisterBankInfo class for
/// PowerPC.
//===----------------------------------------------------------------------===//

#include "PPCRegisterBankInfo.h"
#include "PPCRegisterInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "ppc-reg-bank-info"

#define GET_TARGET_REGBANK_IMPL
#include "PPCGenRegisterBank.inc"

// This file will be TableGen'ed at some point.
#include "PPCGenRegisterBankInfo.def"

using namespace llvm;

PPCRegisterBankInfo::PPCRegisterBankInfo(const TargetRegisterInfo &TRI) {}

const RegisterBank &
PPCRegisterBankInfo::getRegBankFromRegClass(const TargetRegisterClass &RC,
                                            LLT Ty) const {
  switch (RC.getID()) {
  case PPC::G8RCRegClassID:
  case PPC::G8RC_NOX0RegClassID:
  case PPC::G8RC_and_G8RC_NOX0RegClassID:
    return getRegBank(PPC::GPRRegBankID);
  case PPC::VSFRCRegClassID:
  case PPC::SPILLTOVSRRC_and_VSFRCRegClassID:
  case PPC::SPILLTOVSRRC_and_VFRCRegClassID:
  case PPC::SPILLTOVSRRC_and_F4RCRegClassID:
  case PPC::F8RCRegClassID:
  case PPC::VFRCRegClassID:
  case PPC::VSSRCRegClassID:
  case PPC::F4RCRegClassID:
    return getRegBank(PPC::FPRRegBankID);
  default:
    llvm_unreachable("Unexpected register class");
  }
}

const RegisterBankInfo::InstructionMapping &
PPCRegisterBankInfo::getInstrMapping(const MachineInstr &MI) const {
  const unsigned Opc = MI.getOpcode();

  // Try the default logic for non-generic instructions that are either copies
  // or already have some operands assigned to banks.
  if (!isPreISelGenericOpcode(Opc) || Opc == TargetOpcode::G_PHI) {
    const RegisterBankInfo::InstructionMapping &Mapping =
        getInstrMappingImpl(MI);
    if (Mapping.isValid())
      return Mapping;
  }

  const MachineFunction &MF = *MI.getParent()->getParent();
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  const TargetSubtargetInfo &STI = MF.getSubtarget();
  const TargetRegisterInfo &TRI = *STI.getRegisterInfo();

  unsigned NumOperands = MI.getNumOperands();
  const ValueMapping *OperandsMapping = nullptr;
  unsigned Cost = 1;
  unsigned MappingID = DefaultMappingID;

  switch (Opc) {
    // Arithmetic ops.
  case TargetOpcode::G_ADD:
  case TargetOpcode::G_SUB:
    // Bitwise ops.
  case TargetOpcode::G_AND:
  case TargetOpcode::G_OR:
  case TargetOpcode::G_XOR:
    assert(NumOperands <= 3 &&
           "This code is for instructions with 3 or less operands");
    OperandsMapping = getValueMapping(PMI_GPR64);
    break;
  case TargetOpcode::G_FADD:
  case TargetOpcode::G_FSUB:
  case TargetOpcode::G_FMUL:
  case TargetOpcode::G_FDIV: {
    Register SrcReg = MI.getOperand(1).getReg();
    unsigned Size = getSizeInBits(SrcReg, MRI, TRI);

    assert((Size == 32 || Size == 64) && "Unsupported floating point types!\n");
    OperandsMapping = getValueMapping(Size == 32 ? PMI_FPR32 : PMI_FPR64);
    break;
  }
  case TargetOpcode::G_CONSTANT:
    OperandsMapping = getOperandsMapping({getValueMapping(PMI_GPR64), nullptr});
    break;
  case TargetOpcode::G_FPTOUI:
  case TargetOpcode::G_FPTOSI: {
    Register SrcReg = MI.getOperand(1).getReg();
    unsigned Size = getSizeInBits(SrcReg, MRI, TRI);

    OperandsMapping = getOperandsMapping(
        {getValueMapping(PMI_GPR64),
         getValueMapping(Size == 32 ? PMI_FPR32 : PMI_FPR64)});
    break;
  }
  case TargetOpcode::G_UITOFP:
  case TargetOpcode::G_SITOFP: {
    Register SrcReg = MI.getOperand(0).getReg();
    unsigned Size = getSizeInBits(SrcReg, MRI, TRI);

    OperandsMapping =
        getOperandsMapping({getValueMapping(Size == 32 ? PMI_FPR32 : PMI_FPR64),
                            getValueMapping(PMI_GPR64)});
    break;
  }
  default:
    return getInvalidInstructionMapping();
  }

  return getInstructionMapping(MappingID, Cost, OperandsMapping, NumOperands);
}

RegisterBankInfo::InstructionMappings
PPCRegisterBankInfo::getInstrAlternativeMappings(const MachineInstr &MI) const {
  // TODO Implement.
  return RegisterBankInfo::getInstrAlternativeMappings(MI);
}
