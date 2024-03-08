//===------- SystemZFinalizeRegMem.cpp - Finalize FP reg/mem folding ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass converts any remaining reg/reg pseudos into the real target
// instruction in cases where the peephole optimizer did not fold a load into
// a reg/mem instruction.
//
//===----------------------------------------------------------------------===//

#include "SystemZMachineFunctionInfo.h"
#include "SystemZTargetMachine.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

namespace {

class SystemZFinalizeRegMem : public MachineFunctionPass {
public:
  static char ID;
  SystemZFinalizeRegMem()
    : MachineFunctionPass(ID), TII(nullptr), MRI(nullptr) {
    initializeSystemZFinalizeRegMemPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;

private:

  bool visitMBB(MachineBasicBlock &MBB);

  const SystemZInstrInfo *TII;
  MachineRegisterInfo *MRI;
};

char SystemZFinalizeRegMem::ID = 0;

} // end anonymous namespace

INITIALIZE_PASS(SystemZFinalizeRegMem, "systemz-finalize-regmem",
                "SystemZ Finalize RegMem", false, false)

FunctionPass *llvm::
createSystemZFinalizeRegMemPass(SystemZTargetMachine &TM) {
  return new SystemZFinalizeRegMem();
}

void SystemZFinalizeRegMem::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  MachineFunctionPass::getAnalysisUsage(AU);
}

bool SystemZFinalizeRegMem::visitMBB(MachineBasicBlock &MBB) {
  bool Changed = false;
  for (MachineInstr &MI : MBB) {
    unsigned PseudoOpcode = MI.getOpcode();
    unsigned TargetOpcode =
      PseudoOpcode == SystemZ::WFADB_CCPseudo    ? SystemZ::WFADB
      : PseudoOpcode == SystemZ::WFASB_CCPseudo  ? SystemZ::WFASB
      : PseudoOpcode == SystemZ::WFSDB_CCPseudo  ? SystemZ::WFSDB
      : PseudoOpcode == SystemZ::WFSSB_CCPseudo  ? SystemZ::WFSSB
      : PseudoOpcode == SystemZ::WFMADB_CCPseudo  ? SystemZ::WFMADB
      : PseudoOpcode == SystemZ::WFMASB_CCPseudo  ? SystemZ::WFMASB
      : 0;
    if (TargetOpcode) {
        MI.setDesc(TII->get(TargetOpcode));
        int CCIdx = MI.findRegisterDefOperandIdx(SystemZ::CC);
        MI.removeOperand(CCIdx);
        Changed = true;
    }
  }
  return Changed;
}

bool SystemZFinalizeRegMem::runOnMachineFunction(MachineFunction &F) {
  TII = F.getSubtarget<SystemZSubtarget>().getInstrInfo();
  MRI = &F.getRegInfo();

  bool Modified = false;
  for (auto &MBB : F)
    Modified |= visitMBB(MBB);

  return Modified;
}
