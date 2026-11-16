//===------------ RISCVLandingPadSetup.cpp ---------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a RISC-V pass to setup landing pad labels for indirect jumps.
// Currently this pass only supports fixed labels.
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVInstrInfo.h"
#include "RISCVMachineFunctionInfo.h"
#include "RISCVSubtarget.h"
#include "llvm/CodeGen/MachineFunctionAnalysisManager.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/IR/Analysis.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-lpad-setup"
#define PASS_NAME "RISC-V Landing Pad Setup"

namespace {

class RISCVLandingPadSetupLegacy : public MachineFunctionPass {
public:
  static char ID;

  RISCVLandingPadSetupLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &F) override;

  StringRef getPassName() const override { return PASS_NAME; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // end anonymous namespace

static bool runLandingPadSetup(MachineFunction &MF) {
  const auto &STI = MF.getSubtarget<RISCVSubtarget>();
  const RISCVInstrInfo &TII = *STI.getInstrInfo();

  if (!MF.getInfo<RISCVMachineFunctionInfo>()->hasCFProtectionBranch())
    return false;

  uint32_t Label = 0;
  if (std::optional<uint32_t> Preferred = STI.getCLOpts().landing_pad_label) {
    if (!isUInt<20>(*Preferred))
      report_fatal_error("riscv-landing-pad-label=<val>, <val> needs to fit in "
                         "unsigned 20-bits");
    Label = *Preferred;
  }

  // Zicfilp does not check X7 if landing pad label is zero.
  if (Label == 0)
    return false;

  bool Changed = false;
  for (MachineBasicBlock &MBB : MF)
    for (MachineInstr &MI : llvm::make_early_inc_range(MBB)) {
      if (MI.getOpcode() != RISCV::PseudoBRIND &&
          MI.getOpcode() != RISCV::PseudoCALLIndirect &&
          MI.getOpcode() != RISCV::PseudoTAILIndirect)
        continue;
      BuildMI(MBB, MI, MI.getDebugLoc(), TII.get(RISCV::LUI), RISCV::X7)
          .addImm(Label);
      MachineInstrBuilder(MF, &MI).addUse(RISCV::X7, RegState::ImplicitKill);
      Changed = true;
    }

  return Changed;
}

INITIALIZE_PASS(RISCVLandingPadSetupLegacy, DEBUG_TYPE, PASS_NAME, false, false)

char RISCVLandingPadSetupLegacy::ID = 0;

FunctionPass *llvm::createRISCVLandingPadSetupLegacyPass() {
  return new RISCVLandingPadSetupLegacy();
}

bool RISCVLandingPadSetupLegacy::runOnMachineFunction(MachineFunction &MF) {
  return runLandingPadSetup(MF);
}

PreservedAnalyses
RISCVLandingPadSetupPass::run(MachineFunction &MF,
                              MachineFunctionAnalysisManager &MFAM) {
  return runLandingPadSetup(MF) ? getMachineFunctionPassPreservedAnalyses()
                                      .preserveSet<CFGAnalyses>()
                                : PreservedAnalyses::all();
}
