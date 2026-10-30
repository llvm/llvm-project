//===-- Mips.h - Top-level interface for Mips representation ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the entry points for global functions defined in
// the LLVM Mips back-end.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_MIPS_MIPS_H
#define LLVM_LIB_TARGET_MIPS_MIPS_H

#include "MCTargetDesc/MipsMCTargetDesc.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionAnalysisManager.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/IR/Analysis.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Target/TargetMachine.h"

#define IsMFLOMFHI(instr)                                                      \
  (instr == Mips::MFLO || instr == Mips::MFLO64 || instr == Mips::MFHI ||      \
   instr == Mips::MFHI64)
#define IsDIVMULT(instr)                                                       \
  (instr == Mips::SDIV || instr == Mips::PseudoSDIV || instr == Mips::DSDIV || \
   instr == Mips::PseudoDSDIV || instr == Mips::UDIV ||                        \
   instr == Mips::PseudoUDIV || instr == Mips::DUDIV ||                        \
   instr == Mips::PseudoDUDIV || instr == Mips::MULT ||                        \
   instr == Mips::PseudoMULT || instr == Mips::DMULT ||                        \
   instr == Mips::PseudoDMULT)

namespace llvm {
class FunctionPass;
class InstructionSelector;
class MipsRegisterBankInfo;
class MipsSubtarget;
class MipsTargetMachine;
class MipsTargetMachine;
class ModulePass;
class PassRegistry;

class Mips16ISelDAGPass : public SelectionDAGISelPass {
public:
  Mips16ISelDAGPass(MipsTargetMachine &TM, CodeGenOptLevel OptLevel);
  static StringRef name() { return "Mips16ISelDAGPass"; }
  void printPipeline(raw_ostream &OS,
                     function_ref<StringRef(StringRef)> MapClassName2PassName) {
    StringRef ClassName = name();
    auto PassName = MapClassName2PassName(ClassName);
    OS << PassName;
  }
};

FunctionPass *createMips16ISelDAGLegacyPass(MipsTargetMachine &TM,
                                            CodeGenOptLevel OptLevel);

class MipsOs16Pass : public RequiredPassInfoMixin<MipsOs16Pass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
};

ModulePass *createMipsOs16LegacyPass();

class Mips16HardFloatPass : public RequiredPassInfoMixin<Mips16HardFloatPass> {
  MipsTargetMachine &TM;

public:
  Mips16HardFloatPass(MipsTargetMachine &TM) : TM(TM) {}
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
};

ModulePass *createMips16HardFloatLegacyPass();

class MipsSEISelDAGPass : public SelectionDAGISelPass {
public:
  MipsSEISelDAGPass(MipsTargetMachine &TM, CodeGenOptLevel OptLevel);
  static StringRef name() { return "MipsSEISelDAGPass"; }
  void printPipeline(raw_ostream &OS,
                     function_ref<StringRef(StringRef)> MapClassName2PassName) {
    StringRef ClassName = name();
    auto PassName = MapClassName2PassName(ClassName);
    OS << PassName;
  }
};

FunctionPass *createMipsSEISelDAGLegacyPass(MipsTargetMachine &TM,
                                            CodeGenOptLevel OptLevel);

class MipsOptimizePICCallPass
    : public RequiredPassInfoMixin<MipsOptimizePICCallPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

FunctionPass *createMipsOptimizePICCallLegacyPass();
FunctionPass *createMipsDelaySlotFillerPass();
FunctionPass *createMipsBranchExpansion();
FunctionPass *createMipsConstantIslandPass();

class MipsMicroSizeReducePass
    : public RequiredPassInfoMixin<MipsMicroSizeReducePass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

FunctionPass *createMipsMicroSizeReduceLegacyPass();

class MipsExpandPseudoPass
    : public RequiredPassInfoMixin<MipsExpandPseudoPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

FunctionPass *createMipsExpandPseudoLegacyPass();
FunctionPass *createMipsPreLegalizeCombiner();
FunctionPass *createMipsPostLegalizeCombiner(bool IsOptNone);
FunctionPass *createMipsMulMulBugPass();

InstructionSelector *
createMipsInstructionSelector(const MipsTargetMachine &, const MipsSubtarget &,
                              const MipsRegisterBankInfo &);

void initializeMipsMicroSizeReduceLegacyPass(PassRegistry &);
void initializeMipsAsmPrinterPass(PassRegistry &);
void initializeMipsBranchExpansionPass(PassRegistry &);
void initializeMipsDAGToDAGISelLegacyPass(PassRegistry &);
void initializeMipsDelaySlotFillerPass(PassRegistry &);
void initializeMipsMulMulBugFixPass(PassRegistry &);
void initializeMipsPostLegalizerCombinerPass(PassRegistry &);
void initializeMipsPreLegalizerCombinerPass(PassRegistry &);
} // namespace llvm

#endif
