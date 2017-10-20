//===-- RISCV.h - Top-level interface for RISCV -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the entry points for global functions defined in the LLVM
// RISC-V back-end.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_RISCV_RISCV_H
#define LLVM_LIB_TARGET_RISCV_RISCV_H

#include "MCTargetDesc/RISCVMCTargetDesc.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {
class RISCVTargetMachine;
class MCInst;
class MachineInstr;

void LowerRISCVMachineInstrToMCInst(const MachineInstr *MI, MCInst &OutMI);

FunctionPass *createRISCVISelDag(RISCVTargetMachine &TM);
}

#endif
