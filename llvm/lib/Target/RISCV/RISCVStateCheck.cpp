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

  // Callees that are allowed to be called from a RISC-V state-attributed
  // function. These are known not to read or write any architecture state, so
  // calling them is always safe.
  static constexpr StringLiteral AllowedCallees[] = {
      "__riscv_save_0",     "__riscv_save_1",    "__riscv_save_2",
      "__riscv_save_3",     "__riscv_save_4",    "__riscv_save_5",
      "__riscv_save_6",     "__riscv_save_7",    "__riscv_save_8",
      "__riscv_save_9",     "__riscv_save_10",   "__riscv_save_11",
      "__riscv_save_12",    "__riscv_restore_0", "__riscv_restore_1",
      "__riscv_restore_2",  "__riscv_restore_3", "__riscv_restore_4",
      "__riscv_restore_5",  "__riscv_restore_6", "__riscv_restore_7",
      "__riscv_restore_8",  "__riscv_restore_9", "__riscv_restore_10",
      "__riscv_restore_11", "__riscv_restore_12"};

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

      if (is_contained(AllowedCallees, Name))
        continue;

      std::string Message =
          (MF.getName() + ": cannot emit call to '" + Name +
           "' from an RISC-V attributed function. Only the following "
           "functions are allowed to be called: " +
           join(std::begin(AllowedCallees), std::end(AllowedCallees), ", ") +
           ".")
              .str();
      F.getContext().diagnose(DiagnosticInfoGeneric(Message));
    }
  }

  return false;
}

FunctionPass *llvm::createRISCVStateCheckPass() {
  return new RISCVStateCheck();
}
