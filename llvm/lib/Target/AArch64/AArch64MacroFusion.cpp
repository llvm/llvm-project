//===- AArch64MacroFusion.cpp - AArch64 Macro Fusion ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file This file contains the AArch64 implementation of the DAG scheduling
///  mutation to pair instructions back to back.
//
//===----------------------------------------------------------------------===//

#include "AArch64Subtarget.h"
#include "llvm/CodeGen/MacroFusion.h"
#include "llvm/CodeGen/TargetInstrInfo.h"

using namespace llvm;

namespace {

/// CMN, CMP, TST followed by Bcc
static bool isArithmeticBccPair(unsigned FirstOpcode, unsigned SecondOpcode,
                                const MachineInstr *FirstMI) {
  if (SecondOpcode != AArch64::Bcc)
    return false;

  switch (FirstOpcode) {
  case AArch64::INSTRUCTION_LIST_END:
    return true;
  case AArch64::ADDSWri:
  case AArch64::ADDSWrr:
  case AArch64::ADDSXri:
  case AArch64::ADDSXrr:
  case AArch64::ANDSWri:
  case AArch64::ANDSWrr:
  case AArch64::ANDSXri:
  case AArch64::ANDSXrr:
  case AArch64::SUBSWri:
  case AArch64::SUBSWrr:
  case AArch64::SUBSXri:
  case AArch64::SUBSXrr:
  case AArch64::BICSWrr:
  case AArch64::BICSXrr:
    return true;
  case AArch64::ADDSWrs:
  case AArch64::ADDSXrs:
  case AArch64::ANDSWrs:
  case AArch64::ANDSXrs:
  case AArch64::SUBSWrs:
  case AArch64::SUBSXrs:
  case AArch64::BICSWrs:
  case AArch64::BICSXrs:
    // Shift value can be 0 making these behave like the "rr" variant...
    return !AArch64InstrInfo::hasShiftedReg(*FirstMI);
  }
  return false;
}

/// ALU operations followed by CBZ/CBNZ.
static bool isArithmeticCbzPair(unsigned FirstOpcode, unsigned SecondOpcode,
                                const MachineInstr *FirstMI) {
  if (SecondOpcode != AArch64::CBNZW &&
      SecondOpcode != AArch64::CBNZX &&
      SecondOpcode != AArch64::CBZW &&
      SecondOpcode != AArch64::CBZX)
    return false;

  switch (FirstOpcode) {
  case AArch64::INSTRUCTION_LIST_END:
    return true;
  case AArch64::ADDWri:
  case AArch64::ADDWrr:
  case AArch64::ADDXri:
  case AArch64::ADDXrr:
  case AArch64::ANDWri:
  case AArch64::ANDWrr:
  case AArch64::ANDXri:
  case AArch64::ANDXrr:
  case AArch64::EORWri:
  case AArch64::EORWrr:
  case AArch64::EORXri:
  case AArch64::EORXrr:
  case AArch64::ORRWri:
  case AArch64::ORRWrr:
  case AArch64::ORRXri:
  case AArch64::ORRXrr:
  case AArch64::SUBWri:
  case AArch64::SUBWrr:
  case AArch64::SUBXri:
  case AArch64::SUBXrr:
    return true;
  case AArch64::ADDWrs:
  case AArch64::ADDXrs:
  case AArch64::ANDWrs:
  case AArch64::ANDXrs:
  case AArch64::SUBWrs:
  case AArch64::SUBXrs:
  case AArch64::BICWrs:
  case AArch64::BICXrs:
    // Shift value can be 0 making these behave like the "rr" variant...
    return !AArch64InstrInfo::hasShiftedReg(*FirstMI);
  }
  return false;
}

/// AES crypto encoding or decoding.
static bool isAESPair(unsigned FirstOpcode, unsigned SecondOpcode) {
  // AES encode.
  if ((FirstOpcode == AArch64::INSTRUCTION_LIST_END ||
       FirstOpcode == AArch64::AESErr) &&
      (SecondOpcode == AArch64::AESMCrr ||
       SecondOpcode == AArch64::AESMCrrTied))
    return true;
  // AES decode.
  else if ((FirstOpcode == AArch64::INSTRUCTION_LIST_END ||
            FirstOpcode == AArch64::AESDrr) &&
           (SecondOpcode == AArch64::AESIMCrr ||
            SecondOpcode == AArch64::AESIMCrrTied))
    return true;

  return false;
}

/// Literal generation.
static bool isLiteralsPair(unsigned FirstOpcode, unsigned SecondOpcode,
                           const MachineInstr *FirstMI,
                           const MachineInstr &SecondMI) {
  // PC relative address.
  if ((FirstOpcode == AArch64::INSTRUCTION_LIST_END ||
       FirstOpcode == AArch64::ADRP) &&
      SecondOpcode == AArch64::ADDXri)
    return true;
  // 32 bit immediate.
  else if ((FirstOpcode == AArch64::INSTRUCTION_LIST_END ||
            FirstOpcode == AArch64::MOVZWi) &&
           (SecondOpcode == AArch64::MOVKWi &&
            SecondMI.getOperand(3).getImm() == 16))
    return true;
  // Lower half of 64 bit immediate.
  else if((FirstOpcode == AArch64::INSTRUCTION_LIST_END ||
           FirstOpcode == AArch64::MOVZXi) &&
          (SecondOpcode == AArch64::MOVKXi &&
           SecondMI.getOperand(3).getImm() == 16))
    return true;
  // Upper half of 64 bit immediate.
  else if ((FirstOpcode == AArch64::INSTRUCTION_LIST_END ||
            (FirstOpcode == AArch64::MOVKXi &&
             FirstMI->getOperand(3).getImm() == 32)) &&
           (SecondOpcode == AArch64::MOVKXi &&
            SecondMI.getOperand(3).getImm() == 48))
    return true;

  return false;
}

// Fuse address generation and loads or stores.
static bool isAddressLdStPair(unsigned FirstOpcode, unsigned SecondOpcode,
                              const MachineInstr &SecondMI) {
  switch (SecondOpcode) {
  case AArch64::STRBBui:
  case AArch64::STRBui:
  case AArch64::STRDui:
  case AArch64::STRHHui:
  case AArch64::STRHui:
  case AArch64::STRQui:
  case AArch64::STRSui:
  case AArch64::STRWui:
  case AArch64::STRXui:
  case AArch64::LDRBBui:
  case AArch64::LDRBui:
  case AArch64::LDRDui:
  case AArch64::LDRHHui:
  case AArch64::LDRHui:
  case AArch64::LDRQui:
  case AArch64::LDRSui:
  case AArch64::LDRWui:
  case AArch64::LDRXui:
  case AArch64::LDRSBWui:
  case AArch64::LDRSBXui:
  case AArch64::LDRSHWui:
  case AArch64::LDRSHXui:
  case AArch64::LDRSWui:
    switch (FirstOpcode) {
    case AArch64::INSTRUCTION_LIST_END:
      return true;
    case AArch64::ADR:
      return SecondMI.getOperand(2).getImm() == 0;
    case AArch64::ADRP:
      return true;
    }
  }
  return false;
}

// Compare and conditional select.
static bool isCCSelectPair(unsigned FirstOpcode, unsigned SecondOpcode,
                           const MachineInstr *FirstMI) {
  // 32 bits
  if (SecondOpcode == AArch64::CSELWr) {
    // Assume the 1st instr to be a wildcard if it is unspecified.
    if (FirstOpcode == AArch64::INSTRUCTION_LIST_END)
      return true;

    if (FirstMI->definesRegister(AArch64::WZR))
      switch (FirstOpcode) {
      case AArch64::SUBSWrs:
        return !AArch64InstrInfo::hasShiftedReg(*FirstMI);
      case AArch64::SUBSWrx:
        return !AArch64InstrInfo::hasExtendedReg(*FirstMI);
      case AArch64::SUBSWrr:
      case AArch64::SUBSWri:
        return true;
      }
  }
  // 64 bits
  else if (SecondOpcode == AArch64::CSELXr) {
    // Assume the 1st instr to be a wildcard if it is unspecified.
    if (FirstOpcode == AArch64::INSTRUCTION_LIST_END)
      return true;

    if (FirstMI->definesRegister(AArch64::XZR))
      switch (FirstOpcode) {
      case AArch64::SUBSXrs:
        return !AArch64InstrInfo::hasShiftedReg(*FirstMI);
      case AArch64::SUBSXrx:
      case AArch64::SUBSXrx64:
        return !AArch64InstrInfo::hasExtendedReg(*FirstMI);
      case AArch64::SUBSXrr:
      case AArch64::SUBSXri:
        return true;
      }
  }
  return false;
}

/// Check if the instr pair, FirstMI and SecondMI, should be fused
/// together. Given SecondMI, when FirstMI is unspecified, then check if
/// SecondMI may be part of a fused pair at all.
static bool shouldScheduleAdjacent(const TargetInstrInfo &TII,
                                   const TargetSubtargetInfo &TSI,
                                   const MachineInstr *FirstMI,
                                   const MachineInstr &SecondMI) {
  const AArch64Subtarget &ST = static_cast<const AArch64Subtarget&>(TSI);

  // Assume the 1st instr to be a wildcard if it is unspecified.
  unsigned FirstOpc =
      FirstMI ? FirstMI->getOpcode()
              : static_cast<unsigned>(AArch64::INSTRUCTION_LIST_END);
  unsigned SecondOpc = SecondMI.getOpcode();

  if (ST.hasArithmeticBccFusion() &&
      isArithmeticBccPair(FirstOpc, SecondOpc, FirstMI))
    return true;
  if (ST.hasArithmeticCbzFusion() &&
      isArithmeticCbzPair(FirstOpc, SecondOpc, FirstMI))
    return true;
  if (ST.hasFuseAES() && isAESPair(FirstOpc, SecondOpc))
    return true;
  if (ST.hasFuseLiterals() &&
      isLiteralsPair(FirstOpc, SecondOpc, FirstMI, SecondMI))
    return true;
  if (ST.hasFuseAddress() && isAddressLdStPair(FirstOpc, SecondOpc, SecondMI))
    return true;
  if (ST.hasFuseCCSelect() && isCCSelectPair(FirstOpc, SecondOpc, FirstMI))
    return true;

  return false;
}

} // end namespace


namespace llvm {

std::unique_ptr<ScheduleDAGMutation> createAArch64MacroFusionDAGMutation () {
  return createMacroFusionDAGMutation(shouldScheduleAdjacent);
}

} // end namespace llvm
