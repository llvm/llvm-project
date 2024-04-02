//===---- SystemZFinalizeReassociation.cpp - Finalize FP reassociation ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass is the last step of the process of enabling reassociation with
// the MachineCombiner. These are the steps involved:
//
// 1. Instruction selection: Disable reg/mem folding for any operations that
//    are reassociable since MachineCombiner will not succeed otherwise.
//    Select a reg/reg pseudo that pretends to clobber CC since the reg/mem
//    opcode clobbers it.
//
// 2. MachineCombiner: Performs reassociation with the reg/reg instructions.
//
// 3. PeepholeOptimizer: Fold loads into reg/mem instructions.
//
// 4. This pass: Convert any remaining reg/reg pseudos.
//
//===----------------------------------------------------------------------===//

#include "SystemZTargetMachine.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/TargetInstrInfo.h"

using namespace llvm;

namespace {

class SystemZFinalizeReassociation : public MachineFunctionPass {
public:
  static char ID;
  SystemZFinalizeReassociation()
    : MachineFunctionPass(ID), TII(nullptr) {
    initializeSystemZFinalizeReassociationPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;

private:

  bool visitMBB(MachineBasicBlock &MBB);

  const SystemZInstrInfo *TII;
};

char SystemZFinalizeReassociation::ID = 0;

} // end anonymous namespace

INITIALIZE_PASS(SystemZFinalizeReassociation, "systemz-finalize-reassoc",
                "SystemZ Finalize Reassociation", false, false)

FunctionPass *llvm::
createSystemZFinalizeReassociationPass(SystemZTargetMachine &TM) {
  return new SystemZFinalizeReassociation();
}

void SystemZFinalizeReassociation::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  MachineFunctionPass::getAnalysisUsage(AU);
}

bool SystemZFinalizeReassociation::visitMBB(MachineBasicBlock &MBB) {
  bool Changed = false;
  for (MachineInstr &MI : MBB) {
    unsigned PseudoOpcode = MI.getOpcode();
    unsigned TargetOpcode =
      PseudoOpcode == SystemZ::WFADB_CCPseudo    ? SystemZ::WFADB
      : PseudoOpcode == SystemZ::WFASB_CCPseudo  ? SystemZ::WFASB
      : PseudoOpcode == SystemZ::WFSDB_CCPseudo  ? SystemZ::WFSDB
      : PseudoOpcode == SystemZ::WFSSB_CCPseudo  ? SystemZ::WFSSB
      : PseudoOpcode == SystemZ::WFMADB_CCPseudo ? SystemZ::WFMADB
      : PseudoOpcode == SystemZ::WFMASB_CCPseudo ? SystemZ::WFMASB
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

bool SystemZFinalizeReassociation::runOnMachineFunction(MachineFunction &F) {
  TII = F.getSubtarget<SystemZSubtarget>().getInstrInfo();

  bool Modified = false;
  for (auto &MBB : F)
    Modified |= visitMBB(MBB);

  return Modified;
}
