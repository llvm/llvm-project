//===- ARMMacroFusion.cpp - ARM Macro Fusion ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file This file contains the ARM implementation of the DAG scheduling
///  mutation to pair instructions back to back.
//
//===----------------------------------------------------------------------===//

#include "ARMMacroFusion.h"
#include "ARMSubtarget.h"
#include "llvm/CodeGen/MacroFusion.h"
#include "llvm/Target/TargetInstrInfo.h"

namespace llvm {

/// \brief Check if the instr pair, FirstMI and SecondMI, should be fused
/// together. Given SecondMI, when FirstMI is unspecified, then check if
/// SecondMI may be part of a fused pair at all.
static bool shouldScheduleAdjacent(const TargetInstrInfo &TII,
                                   const TargetSubtargetInfo &TSI,
                                   const MachineInstr *FirstMI,
                                   const MachineInstr &SecondMI) {
  const ARMSubtarget &ST = static_cast<const ARMSubtarget&>(TSI);

  // Assume wildcards for unspecified instrs.
  unsigned FirstOpcode =
    FirstMI ? FirstMI->getOpcode()
	    : static_cast<unsigned>(ARM::INSTRUCTION_LIST_END);
  unsigned SecondOpcode = SecondMI.getOpcode();

  if (ST.hasFuseAES())
    // Fuse AES crypto operations.
    switch(SecondOpcode) {
    // AES encode.
    case ARM::AESMC :
      return FirstOpcode == ARM::AESE ||
             FirstOpcode == ARM::INSTRUCTION_LIST_END;
    // AES decode.
    case ARM::AESIMC:
      return FirstOpcode == ARM::AESD ||
             FirstOpcode == ARM::INSTRUCTION_LIST_END;
    }

  return false;
}

std::unique_ptr<ScheduleDAGMutation> createARMMacroFusionDAGMutation () {
  return createMacroFusionDAGMutation(shouldScheduleAdjacent);
}

} // end namespace llvm
