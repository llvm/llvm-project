//===-- Nios2.h - Top-level interface for Nios2 representation --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the entry points for global functions defined in
// the LLVM Nios2 back-end.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_NIOS2_NIOS2_H
#define LLVM_LIB_TARGET_NIOS2_NIOS2_H

#include "MCTargetDesc/Nios2MCTargetDesc.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {
class FunctionPass;
class formatted_raw_ostream;
class Nios2TargetMachine;
class AsmPrinter;
class MachineInstr;
class MCInst;

FunctionPass *createNios2ISelDag(Nios2TargetMachine &TM,
                                 CodeGenOpt::Level OptLevel);
void LowerNios2MachineInstToMCInst(const MachineInstr *MI, MCInst &OutMI,
                                   AsmPrinter &AP);
} // namespace llvm

#endif
