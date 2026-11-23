//===-- LoongArch.h - Top-level interface for LoongArch ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the entry points for global functions defined in the LLVM
// LoongArch back-end.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_LOONGARCH_LOONGARCH_H
#define LLVM_LIB_TARGET_LOONGARCH_LOONGARCH_H

#include "MCTargetDesc/LoongArchBaseInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionAnalysisManager.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/IR/Analysis.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {
class AsmPrinter;
class FunctionPass;
class LoongArchTargetMachine;
class MCInst;
class MCOperand;
class MachineInstr;
class MachineOperand;
class PassRegistry;

bool lowerLoongArchMachineInstrToMCInst(const MachineInstr *MI, MCInst &OutMI,
                                        AsmPrinter &AP);
bool lowerLoongArchMachineOperandToMCOperand(const MachineOperand &MO,
                                             MCOperand &MCOp,
                                             const AsmPrinter &AP);

FunctionPass *createLoongArchDeadRegisterDefinitionsPass();
FunctionPass *createLoongArchExpandAtomicPseudoPass();

class LoongArchISelDAGToDAGPass : public SelectionDAGISelPass {
public:
  LoongArchISelDAGToDAGPass(LoongArchTargetMachine &TM,
                            CodeGenOptLevel OptLevel);
};

FunctionPass *createLoongArchISelDagLegacyPass(LoongArchTargetMachine &TM,
                                               CodeGenOptLevel OptLevel);
FunctionPass *createLoongArchMemoryBarrierOptPass();
FunctionPass *createLoongArchMergeBaseOffsetOptPass();

class LoongArchOptWInstrsPass
    : public OptionalPassInfoMixin<LoongArchOptWInstrsPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

FunctionPass *createLoongArchOptWInstrsLegacyPass();
FunctionPass *createLoongArchPreRAExpandPseudoPass();
FunctionPass *createLoongArchExpandPseudoPass();
FunctionPass *createLoongArchLateBranchOptPass();
void initializeLoongArchAsmPrinterPass(PassRegistry &);
void initializeLoongArchDAGToDAGISelLegacyPass(PassRegistry &);
void initializeLoongArchDeadRegisterDefinitionsPass(PassRegistry &);
void initializeLoongArchExpandAtomicPseudoPass(PassRegistry &);
void initializeLoongArchMemoryBarrierOptPass(PassRegistry &);
void initializeLoongArchMergeBaseOffsetOptPass(PassRegistry &);
void initializeLoongArchOptWInstrsLegacyPass(PassRegistry &);
void initializeLoongArchPreRAExpandPseudoPass(PassRegistry &);
void initializeLoongArchExpandPseudoPass(PassRegistry &);
void initializeLoongArchLateBranchOptPass(PassRegistry &);
} // end namespace llvm

#endif // LLVM_LIB_TARGET_LOONGARCH_LOONGARCH_H
