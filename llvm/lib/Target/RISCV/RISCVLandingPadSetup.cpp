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
#include "RISCVSubtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-lpad-setup"
#define PASS_NAME "RISC-V Landing Pad Setup"

extern cl::opt<uint32_t> PreferredLandingPadLabel;

namespace {

class RISCVLandingPadSetup : public MachineFunctionPass {
public:
  static char ID;

  RISCVLandingPadSetup() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &F) override;

  StringRef getPassName() const override { return PASS_NAME; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // end anonymous namespace

bool RISCVLandingPadSetup::runOnMachineFunction(MachineFunction &MF) {
  const auto &STI = MF.getSubtarget<RISCVSubtarget>();
  const RISCVInstrInfo &TII = *STI.getInstrInfo();

  if (!STI.hasStdExtZicfilp())
    return false;

  uint32_t Label = 0;
  if (PreferredLandingPadLabel.getNumOccurrences() > 0) {
    if (!isUInt<20>(PreferredLandingPadLabel))
      report_fatal_error("riscv-landing-pad-label=<val>, <val> needs to fit in "
                         "unsigned 20-bits");
    Label = PreferredLandingPadLabel;
  }

  // Zicfilp does not check X7 if landing pad label is zero.
  if (Label == 0)
    return false;

  bool Changed = false;
  for (MachineBasicBlock &MBB : MF)
    for (MachineInstr &MI : llvm::make_early_inc_range(MBB)) {
      if (MI.getOpcode() != RISCV::PseudoBRINDNonX7 &&
          MI.getOpcode() != RISCV::PseudoCALLIndirectNonX7 &&
          MI.getOpcode() != RISCV::PseudoTAILIndirectNonX7)
        continue;
      BuildMI(MBB, MI, MI.getDebugLoc(), TII.get(RISCV::LUI), RISCV::X7)
          .addImm(Label);
      MachineInstrBuilder(MF, &MI).addUse(RISCV::X7, RegState::ImplicitKill);
      Changed = true;
    }

  return Changed;
}

INITIALIZE_PASS(RISCVLandingPadSetup, DEBUG_TYPE, PASS_NAME, false, false)

char RISCVLandingPadSetup::ID = 0;

FunctionPass *llvm::createRISCVLandingPadSetupPass() {
  return new RISCVLandingPadSetup();
}
