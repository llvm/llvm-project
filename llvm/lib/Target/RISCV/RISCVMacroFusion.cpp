//===- RISCVMacroFusion.cpp - RISC-V Macro Fusion -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file This file contains the RISC-V implementation of the DAG scheduling
/// mutation to pair instructions back to back.
//
//===----------------------------------------------------------------------===//
//
#include "RISCVMacroFusion.h"
#include "RISCVSubtarget.h"
#include "llvm/CodeGen/MacroFusion.h"
#include "llvm/CodeGen/TargetInstrInfo.h"

using namespace llvm;

// Fuse LUI followed by ADDI or ADDIW.
// rd = imm[31:0] which decomposes to
// lui rd, imm[31:12]
// addi(w) rd, rd, imm[11:0]
static bool isLUIADDI(const MachineInstr *FirstMI,
                      const MachineInstr &SecondMI) {
  if (SecondMI.getOpcode() != RISCV::ADDI &&
      SecondMI.getOpcode() != RISCV::ADDIW)
    return false;

  // Assume the 1st instr to be a wildcard if it is unspecified.
  if (!FirstMI)
    return true;

  if (FirstMI->getOpcode() != RISCV::LUI)
    return false;

  Register FirstDest = FirstMI->getOperand(0).getReg();

  // Destination of LUI should be the ADDI(W) source register.
  if (SecondMI.getOperand(1).getReg() != FirstDest)
    return false;

  // If the input is virtual make sure this is the only user.
  if (FirstDest.isVirtual()) {
    auto &MRI = SecondMI.getMF()->getRegInfo();
    return MRI.hasOneNonDBGUse(FirstDest);
  }

  // If the FirstMI destination is non-virtual, it should match the SecondMI
  // destination.
  return SecondMI.getOperand(0).getReg() == FirstDest;
}

static bool shouldScheduleAdjacent(const TargetInstrInfo &TII,
                                   const TargetSubtargetInfo &TSI,
                                   const MachineInstr *FirstMI,
                                   const MachineInstr &SecondMI) {
  const RISCVSubtarget &ST = static_cast<const RISCVSubtarget &>(TSI);

  if (ST.hasLUIADDIFusion() && isLUIADDI(FirstMI, SecondMI))
    return true;

  return false;
}

std::unique_ptr<ScheduleDAGMutation> llvm::createRISCVMacroFusionDAGMutation() {
  return createMacroFusionDAGMutation(shouldScheduleAdjacent);
}
