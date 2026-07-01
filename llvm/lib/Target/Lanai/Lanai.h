//===-- Lanai.h - Top-level interface for Lanai representation --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the entry points for global functions defined in the LLVM
// Lanai back-end.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_LANAI_LANAI_H
#define LLVM_LIB_TARGET_LANAI_LANAI_H

#include "llvm/CodeGen/MachineFunctionAnalysisManager.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/IR/Analysis.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

namespace llvm {
class FunctionPass;
class LanaiTargetMachine;
class PassRegistry;

// createLanaiISelDagPass - This pass converts a legalized DAG into a
// Lanai-specific DAG, ready for instruction scheduling.
class LanaiISelDAGToDAGPass : public SelectionDAGISelPass {
public:
  LanaiISelDAGToDAGPass(LanaiTargetMachine &TM);
};

FunctionPass *createLanaiISelDagLegacyPass(LanaiTargetMachine &TM);

// createLanaiDelaySlotFillerPass - This pass fills delay slots
// with useful instructions or nop's
class LanaiDelaySlotFillerPass
    : public PassInfoMixin<LanaiDelaySlotFillerPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

FunctionPass *
createLanaiDelaySlotFillerLegacyPass(const LanaiTargetMachine &TM);

// createLanaiMemAluCombinerPass - This pass combines loads/stores and
// arithmetic operations.
class LanaiMemAluCombinerPass : public PassInfoMixin<LanaiMemAluCombinerPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

FunctionPass *createLanaiMemAluCombinerLegacyPass();

void initializeLanaiAsmPrinterPass(PassRegistry &);
void initializeLanaiDAGToDAGISelLegacyPass(PassRegistry &);
void initializeLanaiMemAluCombinerLegacyPass(PassRegistry &);

} // namespace llvm

#endif // LLVM_LIB_TARGET_LANAI_LANAI_H
