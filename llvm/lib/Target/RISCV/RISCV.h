//===-- RISCV.h - Top-level interface for RISCV -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the entry points for global functions defined in the LLVM
// RISC-V back-end.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_RISCV_RISCV_H
#define LLVM_LIB_TARGET_RISCV_RISCV_H

#include "Utils/RISCVBaseInfo.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {
class RISCVRegisterBankInfo;
class RISCVSubtarget;
class RISCVTargetMachine;
class AsmPrinter;
class FunctionPass;
class InstructionSelector;
class MCInst;
class MCOperand;
class MachineInstr;
class MachineOperand;
class PassRegistry;

void LowerRISCVMachineInstrToMCInst(const MachineInstr *MI, MCInst &OutMI,
                                    const AsmPrinter &AP);
bool LowerRISCVMachineOperandToMCOperand(const MachineOperand &MO,
                                         MCOperand &MCOp, const AsmPrinter &AP);

FunctionPass *createRISCVISelDag(RISCVTargetMachine &TM);

FunctionPass *createRISCVMergeBaseOffsetOptPass();
void initializeRISCVMergeBaseOffsetOptPass(PassRegistry &);

FunctionPass *createRISCVExpandPseudoPass();
void initializeRISCVExpandPseudoPass(PassRegistry &);

FunctionPass *createRISCVExpandAtomicPseudoPass();
void initializeRISCVExpandAtomicPseudoPass(PassRegistry &);

FunctionPass *createRISCVCleanupVSETVLIPass();
void initializeRISCVCleanupVSETVLIPass(PassRegistry &);

InstructionSelector *createRISCVInstructionSelector(const RISCVTargetMachine &,
                                                    RISCVSubtarget &,
                                                    RISCVRegisterBankInfo &);
}

#endif
