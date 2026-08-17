//======-- RISCVStateCheck.cpp - Helper for checking RISC-V attributes -======//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVStateAttributes.h"
#include "llvm/CodeGen/MachineFunctionPass.h"

using namespace llvm;

#define RISCV_STATE_CHECK_NAME "RISC-V architecture state check"

namespace {
class RISCVStateCheck : public MachineFunctionPass {
public:
  static char ID;

  RISCVStateCheck() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override { return RISCV_STATE_CHECK_NAME; }
};
} // namespace

char RISCVStateCheck::ID = 0;

INITIALIZE_PASS(RISCVStateCheck, "riscv-state-check", RISCV_STATE_CHECK_NAME,
                false, true)

static const MachineOperand *getCalleeSymbol(const MachineInstr &MI) {
  for (const MachineOperand &MO : MI.operands())
    if (MO.isGlobal() || MO.isSymbol())
      return &MO;
  return nullptr;
}

bool RISCVStateCheck::runOnMachineFunction(MachineFunction &MF) {
  const Function &F = MF.getFunction();
  if (!RISCVState::hasAttribute(F))
    return false;

  for (const MachineBasicBlock &MBB : MF) {
    for (const MachineInstr &MI : MBB) {
      if (!MI.isCall())
        continue;

      // There might be save/restore libcalls generated during frame lowering
      // that only touch GPRs, in that case we can just skip it.
      if (MI.getFlag(MachineInstr::FrameSetup) ||
          MI.getFlag(MachineInstr::FrameDestroy))
        continue;

      const MachineOperand *Callee = getCalleeSymbol(MI);
      if (!Callee)
        continue;

      std::string Name;
      if (Callee->isSymbol()) {
        Name = Callee->getSymbolName();
      } else {
        const GlobalValue *GV = Callee->getGlobal();
        const auto *CalleeFn = dyn_cast<Function>(GV);
        // Skip if this function is attributed which is already checked at
        // frontend.
        if (CalleeFn && RISCVState::hasAttribute(*CalleeFn))
          continue;
        Name = GV->getName().str();
      }

      std::string Message = "cannot emit call to '" + Name +
                            "' from an RISC-V attributed function.";
      reportFatalUsageError(MF.getName() + ": " + Message);
    }
  }

  return false;
}

FunctionPass *llvm::createRISCVStateCheckPass() {
  return new RISCVStateCheck();
}
