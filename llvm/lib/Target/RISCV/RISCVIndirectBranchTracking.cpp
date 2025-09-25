//===------ RISCVIndirectBranchTracking.cpp - Enables lpad mechanism ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The pass adds LPAD (AUIPC with rd = X0) machine instructions at the
// beginning of each basic block or function that is referenced by an indirect
// jump/call instruction.
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVInstrInfo.h"
#include "RISCVSubtarget.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"

#define DEBUG_TYPE "riscv-indrect-branch-tracking"
#define PASS_NAME "RISC-V Indirect Branch Tracking"

using namespace llvm;

cl::opt<uint32_t> PreferredLandingPadLabel(
    "riscv-landing-pad-label", cl::ReallyHidden,
    cl::desc("Use preferred fixed label for all labels"));

namespace {
class RISCVIndirectBranchTracking : public MachineFunctionPass {
public:
  static char ID;
  RISCVIndirectBranchTracking() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override { return PASS_NAME; }

  bool runOnMachineFunction(MachineFunction &MF) override;

private:
  const Align LpadAlign = Align(4);
};

} // end anonymous namespace

INITIALIZE_PASS(RISCVIndirectBranchTracking, DEBUG_TYPE, PASS_NAME, false,
                false)

char RISCVIndirectBranchTracking::ID = 0;

FunctionPass *llvm::createRISCVIndirectBranchTrackingPass() {
  return new RISCVIndirectBranchTracking();
}

static void emitLpad(MachineBasicBlock &MBB, const RISCVInstrInfo *TII,
                     uint32_t Label) {
  auto I = MBB.begin();
  BuildMI(MBB, I, MBB.findDebugLoc(I), TII->get(RISCV::AUIPC), RISCV::X0)
      .addImm(Label);
}

bool RISCVIndirectBranchTracking::runOnMachineFunction(MachineFunction &MF) {
  const auto &Subtarget = MF.getSubtarget<RISCVSubtarget>();
  const RISCVInstrInfo *TII = Subtarget.getInstrInfo();
  if (!Subtarget.hasStdExtZicfilp())
    return false;

  uint32_t FixedLabel = 0;
  if (PreferredLandingPadLabel.getNumOccurrences() > 0) {
    if (!isUInt<20>(PreferredLandingPadLabel))
      report_fatal_error("riscv-landing-pad-label=<val>, <val> needs to fit in "
                         "unsigned 20-bits");
    FixedLabel = PreferredLandingPadLabel;
  }

  bool Changed = false;
  for (MachineBasicBlock &MBB : MF) {
    if (&MBB == &MF.front()) {
      Function &F = MF.getFunction();
      // When trap is taken, landing pad is not needed.
      if (F.hasFnAttribute("interrupt"))
        continue;

      if (F.hasAddressTaken() || !F.hasLocalLinkage()) {
        emitLpad(MBB, TII, FixedLabel);
        if (MF.getAlignment() < LpadAlign)
          MF.setAlignment(LpadAlign);
        Changed = true;
      }
      continue;
    }

    if (MBB.hasAddressTaken()) {
      emitLpad(MBB, TII, FixedLabel);
      if (MBB.getAlignment() < LpadAlign)
        MBB.setAlignment(LpadAlign);
      Changed = true;
    }
  }

  return Changed;
}
