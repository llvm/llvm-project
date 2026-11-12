//===- MIRPrintingPass.cpp - Pass that prints out using the MIR format ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass that prints out the LLVM module using the MIR
// serialization format.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MIRPrinter.h"
#include "llvm/CodeGen/MachineFunctionAnalysis.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineModuleSlotTracker.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"

using namespace llvm;

void PrintMIRPass::printMachineFunction(MachineFunction &MF,
                                        FunctionAnalysisManager &FAM,
                                        MachineFunctionAnalysisManager &MFAM) {
  const VirtRegMap *VRM = MFAM.getCachedResult<VirtRegMapAnalysis>(MF);
  MachineModuleSlotTracker MST(
      [&](const Function &F) {
        return &FAM.getResult<MachineFunctionAnalysis>(
                       const_cast<Function &>(F))
                    .getMF();
      },
      &MF);
  MST.renumberMetadataForAssembly();
  printMIR(OS, FAM, MF, VRM);
}

PreservedAnalyses PrintMIRPass::run(Module &M, ModuleAnalysisManager &MAM) {
  M.renumberMetadataForAssembly();
  printMIR(OS, M);
  auto &FAM = MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  for (Function &F : M) {
    if (F.isDeclaration() || F.hasAvailableExternallyLinkage())
      continue;
    MachineFunction &MF = FAM.getResult<MachineFunctionAnalysis>(F).getMF();
    auto &MFAM = FAM.getResult<MachineFunctionAnalysisManagerFunctionProxy>(F)
                     .getManager();
    printMachineFunction(MF, FAM, MFAM);
  }
  return PreservedAnalyses::all();
}

namespace {

/// This pass prints out the LLVM IR to an output stream using the MIR
/// serialization format.
struct MIRPrintingPass : public MachineFunctionPass {
  static char ID;
  raw_ostream &OS;
  std::string MachineFunctions;

  MIRPrintingPass() : MachineFunctionPass(ID), OS(dbgs()) {}
  MIRPrintingPass(raw_ostream &OS) : MachineFunctionPass(ID), OS(OS) {}

  StringRef getPassName() const override { return "MIR Printing Pass"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    AU.addUsedIfAvailable<VirtRegMapWrapperLegacy>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override {
    std::string Str;
    raw_string_ostream StrOS(Str);

    MachineModuleInfo *MMI =
        &getAnalysis<MachineModuleInfoWrapperPass>().getMMI();

    const VirtRegMap *VRM = nullptr;
    if (auto *W = getAnalysisIfAvailable<VirtRegMapWrapperLegacy>())
      VRM = &W->getVRM();

    MachineModuleSlotTracker MST(
        [&](const Function &F) { return MMI->getMachineFunction(F); }, &MF);
    MST.renumberMetadataForAssembly();
    printMIR(StrOS, *MMI, MF, VRM);
    MachineFunctions.append(Str);
    return false;
  }

  bool doFinalization(Module &M) override {
    M.renumberMetadataForAssembly();
    printMIR(OS, M);
    OS << MachineFunctions;
    return false;
  }
};

char MIRPrintingPass::ID = 0;

} // end anonymous namespace

MachineFunctionPass *llvm::createPrintMIRPass(raw_ostream &OS) {
  return new MIRPrintingPass(OS);
}
