//===-- DirectXAsmPrinter.cpp - DirectX assembly writer --------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains AsmPrinters for the DirectX backend.
//
//===----------------------------------------------------------------------===//

#include "DirectXAsmPrinter.h"
#include "TargetInfo/DirectXTargetInfo.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/AsmPrinterAnalysis.h"
#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/SectionKind.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Target/TargetLoweringObjectFile.h"

using namespace llvm;

#define DEBUG_TYPE "asm-printer"

namespace {

// The DXILAsmPrinter is mostly a stub because DXIL is just LLVM bitcode which
// gets embedded into a DXContainer file.
class DXILAsmPrinter : public AsmPrinter {
public:
  explicit DXILAsmPrinter(TargetMachine &TM,
                          std::unique_ptr<MCStreamer> Streamer)
      : AsmPrinter(TM, std::move(Streamer)) {}

  StringRef getPassName() const override { return "DXIL Assembly Printer"; }
  void emitGlobalVariable(const GlobalVariable *GV) override;
  bool runOnMachineFunction(MachineFunction &MF) override { return false; }
};
} // namespace

void DXILAsmPrinter::emitGlobalVariable(const GlobalVariable *GV) {
  // If there is no initializer, or no explicit section do nothing
  if (!GV->hasInitializer() || GV->hasImplicitSection() || !GV->hasSection())
    return;
  // Skip the LLVM metadata
  if (GV->getSection() == "llvm.metadata")
    return;
  SectionKind GVKind = TargetLoweringObjectFile::getKindForGlobal(GV, TM);
  MCSection *TheSection = getObjFileLowering().SectionForGlobal(GV, GVKind, TM);
  OutStreamer->switchSection(TheSection);
  emitGlobalConstant(GV->getDataLayout(), GV->getInitializer());
}

extern "C" LLVM_ABI LLVM_EXTERNAL_VISIBILITY void
LLVMInitializeDirectXAsmPrinter() {
  RegisterAsmPrinter<DXILAsmPrinter> X(getTheDirectXTarget());
}

PreservedAnalyses DXILAsmPrinterBeginPass::run(Module &M,
                                               ModuleAnalysisManager &MAM) {
  DXILAsmPrinter &AsmPrinter = static_cast<DXILAsmPrinter &>(
      MAM.getResult<AsmPrinterAnalysis>(M).getPrinter());
  setupModuleAsmPrinter(M, MAM, AsmPrinter);
  AsmPrinter.doInitialization(M);
  return PreservedAnalyses::all();
}

PreservedAnalyses
DXILAsmPrinterPass::run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM) {
  DXILAsmPrinter &AsmPrinter = static_cast<DXILAsmPrinter &>(
      MFAM.getResult<ModuleAnalysisManagerMachineFunctionProxy>(MF)
          .getCachedResult<AsmPrinterAnalysis>(*MF.getFunction().getParent())
          ->getPrinter());
  setupMachineFunctionAsmPrinter(MFAM, MF, AsmPrinter);
  AsmPrinter.runOnMachineFunction(MF);
  return PreservedAnalyses::all();
}

PreservedAnalyses DXILAsmPrinterEndPass::run(Module &M,
                                             ModuleAnalysisManager &MAM) {
  DXILAsmPrinter &AsmPrinter = static_cast<DXILAsmPrinter &>(
      MAM.getResult<AsmPrinterAnalysis>(M).getPrinter());
  setupModuleAsmPrinter(M, MAM, AsmPrinter);
  AsmPrinter.doFinalization(M);
  return PreservedAnalyses::all();
}
