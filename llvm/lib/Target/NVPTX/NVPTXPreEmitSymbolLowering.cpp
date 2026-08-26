//===-- NVPTXPreEmitSymbolLowering.cpp - Lower symbols before emission ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NVPTX.h"
#include "NVPTXMachineFunctionInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

namespace {

class NVPTXPreEmitSymbolLoweringLegacy : public MachineFunctionPass {
  const TargetMachine *TM = nullptr;

public:
  static char ID;

  NVPTXPreEmitSymbolLoweringLegacy() : MachineFunctionPass(ID) {}
  NVPTXPreEmitSymbolLoweringLegacy(const TargetMachine &TM)
      : MachineFunctionPass(ID), TM(&TM) {}

  bool doInitialization(Module &M) override {
    assert(TM && "TargetMachine must be set");
    for (const GlobalValue &GV : M.global_values())
      TM->getSymbol(&GV);
    return false;
  }

  bool runOnMachineFunction(MachineFunction &MF) override {
    auto *MFI = MF.getInfo<NVPTXMachineFunctionInfo>();
    if (MFI->getCallPrototypes().empty())
      return false;

    for (const auto &[Id, Prototype] : MFI->getCallPrototypes()) {
      if (!Prototype.Symbol) {
        MCSymbol *Symbol = MF.getContext().createTempSymbol(
            "prototype_" + Twine(Id), /*AlwaysAddSuffix=*/false);
        MFI->setCallPrototypeSymbol(Id, Symbol);
      }
    }

    bool Changed = false;
    for (MachineBasicBlock &MBB : MF) {
      for (MachineInstr &MI : MBB) {
        if (MI.getOpcode() != NVPTX::CALL && MI.getOpcode() != NVPTX::CALL_conv)
          continue;

        MachineOperand &Proto = MI.getOperand(3);
        if (Proto.isImm()) {
          Proto.ChangeToMCSymbol(MFI->getCallPrototypeSymbol(Proto.getImm()));
          Changed = true;
        }
        assert(Proto.isMCSymbol() &&
               "call prototype operand must be rewritten to a symbol");
      }
    }

    return Changed;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // namespace

char NVPTXPreEmitSymbolLoweringLegacy::ID = 0;

INITIALIZE_PASS(NVPTXPreEmitSymbolLoweringLegacy,
                "nvptx-pre-emit-symbol-lowering",
                "NVPTX Pre-Emit Symbol Lowering", false, false)

MachineFunctionPass *
llvm::createNVPTXPreEmitSymbolLoweringLegacyPass(const TargetMachine &TM) {
  return new NVPTXPreEmitSymbolLoweringLegacy(TM);
}
