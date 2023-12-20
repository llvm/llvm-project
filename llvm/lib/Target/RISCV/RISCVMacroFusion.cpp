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

static bool checkRegisters(Register FirstDest, const MachineInstr &SecondMI) {
  if (!SecondMI.getOperand(1).isReg())
    return false;

  if (SecondMI.getOperand(1).getReg() != FirstDest)
    return false;

  // If the input is virtual make sure this is the only user.
  if (FirstDest.isVirtual()) {
    auto &MRI = SecondMI.getMF()->getRegInfo();
    return MRI.hasOneNonDBGUse(FirstDest);
  }

  return SecondMI.getOperand(0).getReg() == FirstDest;
}

// Fuse load with add:
// add rd, rs1, rs2
// ld rd, 0(rd)
static bool isLDADD(const MachineInstr *FirstMI, const MachineInstr &SecondMI) {
  if (SecondMI.getOpcode() != RISCV::LD)
    return false;

  if (!SecondMI.getOperand(2).isImm())
    return false;

  if (SecondMI.getOperand(2).getImm() != 0)
    return false;

  // Given SecondMI, when FirstMI is unspecified, we must return
  // if SecondMI may be part of a fused pair at all.
  if (!FirstMI)
    return true;

  if (FirstMI->getOpcode() != RISCV::ADD)
    return true;

  return checkRegisters(FirstMI->getOperand(0).getReg(), SecondMI);
}

// Fuse these patterns:
//
// slli rd, rs1, 32
// srli rd, rd, x
// where 0 <= x <= 32
//
// and
//
// slli rd, rs1, 48
// srli rd, rd, x
static bool isShiftedZExt(const MachineInstr *FirstMI,
                          const MachineInstr &SecondMI) {
  if (SecondMI.getOpcode() != RISCV::SRLI)
    return false;

  if (!SecondMI.getOperand(2).isImm())
    return false;

  unsigned SRLIImm = SecondMI.getOperand(2).getImm();
  bool IsShiftBy48 = SRLIImm == 48;
  if (SRLIImm > 32 && !IsShiftBy48)
    return false;

  // Given SecondMI, when FirstMI is unspecified, we must return
  // if SecondMI may be part of a fused pair at all.
  if (!FirstMI)
    return true;

  if (FirstMI->getOpcode() != RISCV::SLLI)
    return false;

  unsigned SLLIImm = FirstMI->getOperand(2).getImm();
  if (IsShiftBy48 ? (SLLIImm != 48) : (SLLIImm != 32))
    return false;

  return checkRegisters(FirstMI->getOperand(0).getReg(), SecondMI);
}

// Fuse AUIPC followed by ADDI
// auipc rd, imm20
// addi rd, rd, imm12
static bool isAUIPCADDI(const MachineInstr *FirstMI,
                        const MachineInstr &SecondMI) {
  if (SecondMI.getOpcode() != RISCV::ADDI)
    return false;
  // Assume the 1st instr to be a wildcard if it is unspecified.
  if (!FirstMI)
    return true;

  if (FirstMI->getOpcode() != RISCV::AUIPC)
    return false;

  return checkRegisters(FirstMI->getOperand(0).getReg(), SecondMI);
}

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

  return checkRegisters(FirstMI->getOperand(0).getReg(), SecondMI);
}

static bool shouldScheduleAdjacent(const TargetInstrInfo &TII,
                                   const TargetSubtargetInfo &TSI,
                                   const MachineInstr *FirstMI,
                                   const MachineInstr &SecondMI) {
  const RISCVSubtarget &ST = static_cast<const RISCVSubtarget &>(TSI);

  if (ST.hasLUIADDIFusion() && isLUIADDI(FirstMI, SecondMI))
    return true;

  if (ST.hasAUIPCADDIFusion() && isAUIPCADDI(FirstMI, SecondMI))
    return true;

  if (ST.hasShiftedZExtFusion() && isShiftedZExt(FirstMI, SecondMI))
    return true;

  if (ST.hasLDADDFusion() && isLDADD(FirstMI, SecondMI))
    return true;

  return false;
}

std::unique_ptr<ScheduleDAGMutation> llvm::createRISCVMacroFusionDAGMutation() {
  return createMacroFusionDAGMutation(shouldScheduleAdjacent);
}
