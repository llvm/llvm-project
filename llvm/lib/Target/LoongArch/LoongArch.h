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

class LoongArchDeadRegisterDefinitionsPass
    : public OptionalPassInfoMixin<LoongArchDeadRegisterDefinitionsPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

FunctionPass *createLoongArchDeadRegisterDefinitionsLegacyPass();

class LoongArchExpandAtomicPseudoPass
    : public RequiredPassInfoMixin<LoongArchExpandAtomicPseudoPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

FunctionPass *createLoongArchExpandAtomicPseudoLegacyPass();

class LoongArchISelDAGToDAGPass : public SelectionDAGISelPass {
public:
  LoongArchISelDAGToDAGPass(LoongArchTargetMachine &TM,
                            CodeGenOptLevel OptLevel);
};

FunctionPass *createLoongArchISelDagLegacyPass(LoongArchTargetMachine &TM,
                                               CodeGenOptLevel OptLevel);

class LoongArchMemoryBarrierOptPass
    : public OptionalPassInfoMixin<LoongArchMemoryBarrierOptPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

FunctionPass *createLoongArchMemoryBarrierOptLegacyPass();

class LoongArchMergeBaseOffsetPass
    : public OptionalPassInfoMixin<LoongArchMergeBaseOffsetPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);

  MachineFunctionProperties getRequiredProperties() const {
    return MachineFunctionProperties().setIsSSA();
  }
};

FunctionPass *createLoongArchMergeBaseOffsetLegacyPass();

class LoongArchOptWInstrsPass
    : public OptionalPassInfoMixin<LoongArchOptWInstrsPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

FunctionPass *createLoongArchOptWInstrsLegacyPass();

class LoongArchPreRAExpandPseudoPass
    : public RequiredPassInfoMixin<LoongArchPreRAExpandPseudoPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

FunctionPass *createLoongArchPreRAExpandPseudoLegacyPass();

class LoongArchExpandPseudoPass
    : public RequiredPassInfoMixin<LoongArchExpandPseudoPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

FunctionPass *createLoongArchExpandPseudoLegacyPass();

class LoongArchLateBranchOptPass
    : public OptionalPassInfoMixin<LoongArchLateBranchOptPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

FunctionPass *createLoongArchLateBranchOptLegacyPass();
void initializeLoongArchAsmPrinterPass(PassRegistry &);
void initializeLoongArchDAGToDAGISelLegacyPass(PassRegistry &);
void initializeLoongArchDeadRegisterDefinitionsLegacyPass(PassRegistry &);
void initializeLoongArchExpandAtomicPseudoLegacyPass(PassRegistry &);
void initializeLoongArchMemoryBarrierOptLegacyPass(PassRegistry &);
void initializeLoongArchMergeBaseOffsetLegacyPass(PassRegistry &);
void initializeLoongArchOptWInstrsLegacyPass(PassRegistry &);
void initializeLoongArchPreRAExpandPseudoLegacyPass(PassRegistry &);
void initializeLoongArchExpandPseudoLegacyPass(PassRegistry &);
void initializeLoongArchLateBranchOptLegacyPass(PassRegistry &);
} // end namespace llvm

#endif // LLVM_LIB_TARGET_LOONGARCH_LOONGARCH_H
