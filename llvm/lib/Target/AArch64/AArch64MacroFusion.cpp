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

/// \brief Check if the instr pair, FirstMI and SecondMI, should be fused
/// together. Given SecondMI, when FirstMI is unspecified, then check if
/// SecondMI may be part of a fused pair at all.
static bool shouldScheduleAdjacent(const TargetInstrInfo &TII,
                                   const TargetSubtargetInfo &TSI,
                                   const MachineInstr *FirstMI,
                                   const MachineInstr &SecondMI) {
  const AArch64InstrInfo &II = static_cast<const AArch64InstrInfo&>(TII);
  const AArch64Subtarget &ST = static_cast<const AArch64Subtarget&>(TSI);

  // Assume wildcards for unspecified instrs.
  unsigned FirstOpcode =
      FirstMI ? FirstMI->getOpcode()
              : static_cast<unsigned>(AArch64::INSTRUCTION_LIST_END);
  unsigned SecondOpcode = SecondMI.getOpcode();

  if (ST.hasArithmeticBccFusion())
    // Fuse CMN, CMP, TST followed by Bcc.
    if (SecondOpcode == AArch64::Bcc)
      switch (FirstOpcode) {
      default:
        return false;
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
        return !II.hasShiftedReg(*FirstMI);
      case AArch64::INSTRUCTION_LIST_END:
        return true;
      }

  if (ST.hasArithmeticCbzFusion())
    // Fuse ALU operations followed by CBZ/CBNZ.
    if (SecondOpcode == AArch64::CBNZW || SecondOpcode == AArch64::CBNZX ||
        SecondOpcode == AArch64::CBZW || SecondOpcode == AArch64::CBZX)
      switch (FirstOpcode) {
      default:
        return false;
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
        return !II.hasShiftedReg(*FirstMI);
      case AArch64::INSTRUCTION_LIST_END:
        return true;
      }

  if (ST.hasFuseAES())
    // Fuse AES crypto operations.
    switch(SecondOpcode) {
    // AES encode.
    case AArch64::AESMCrr:
    case AArch64::AESMCrrTied:
      return FirstOpcode == AArch64::AESErr ||
             FirstOpcode == AArch64::INSTRUCTION_LIST_END;
    // AES decode.
    case AArch64::AESIMCrr:
    case AArch64::AESIMCrrTied:
      return FirstOpcode == AArch64::AESDrr ||
             FirstOpcode == AArch64::INSTRUCTION_LIST_END;
    }

  if (ST.hasFuseLiterals())
    // Fuse literal generation operations.
    switch (SecondOpcode) {
    // PC relative address.
    case AArch64::ADDXri:
      return FirstOpcode == AArch64::ADRP ||
             FirstOpcode == AArch64::INSTRUCTION_LIST_END;
    // 32 bit immediate.
    case AArch64::MOVKWi:
      return (FirstOpcode == AArch64::MOVZWi &&
              SecondMI.getOperand(3).getImm() == 16) ||
             FirstOpcode == AArch64::INSTRUCTION_LIST_END;
    // Lower and upper half of 64 bit immediate.
    case AArch64::MOVKXi:
      return FirstOpcode == AArch64::INSTRUCTION_LIST_END ||
             (FirstOpcode == AArch64::MOVZXi &&
              SecondMI.getOperand(3).getImm() == 16) ||
             (FirstOpcode == AArch64::MOVKXi &&
              FirstMI->getOperand(3).getImm() == 32 &&
              SecondMI.getOperand(3).getImm() == 48);
    }

  return false;
}

} // end namespace


namespace llvm {

std::unique_ptr<ScheduleDAGMutation> createAArch64MacroFusionDAGMutation () {
  return createMacroFusionDAGMutation(shouldScheduleAdjacent);
}

} // end namespace llvm
