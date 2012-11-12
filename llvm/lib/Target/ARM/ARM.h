//===-- ARM.h - Top-level interface for ARM representation ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the entry points for global functions defined in the LLVM
// ARM back-end.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_ARM_H
#define TARGET_ARM_H

#include "MCTargetDesc/ARMBaseInfo.h"
#include "MCTargetDesc/ARMMCTargetDesc.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {

class ARMAsmPrinter;
class ARMBaseTargetMachine;
class FunctionPass;
class JITCodeEmitter;
class MachineInstr;
class MCInst;

FunctionPass *createARMISelDag(ARMBaseTargetMachine &TM,
                               CodeGenOpt::Level OptLevel);

FunctionPass *createARMJITCodeEmitterPass(ARMBaseTargetMachine &TM,
                                          JITCodeEmitter &JCE);

FunctionPass *createARMLoadStoreOptimizationPass(bool PreAlloc = false);
FunctionPass *createARMExpandPseudoPass();
FunctionPass *createARMGlobalBaseRegPass();
FunctionPass *createARMGlobalMergePass(const TargetLowering* tli);
FunctionPass *createARMConstantIslandPass();
FunctionPass *createMLxExpansionPass();
FunctionPass *createThumb2ITBlockPass();
FunctionPass *createThumb2SizeReductionPass();

void LowerARMMachineInstrToMCInst(const MachineInstr *MI, MCInst &OutMI,
                                  ARMAsmPrinter &AP);

} // end namespace llvm;

#endif
