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
//    are reassociable since MachineCombiner will not succeed
//    otherwise. Instead select a reg/reg pseudo that pretends to clobber CC.
//
// 2. MachineCombiner: Performs reassociation with the reg/reg instructions.
//
// 3. PeepholeOptimizer: fold loads into reg/mem instructions after
//    reassociation. The reg/mem opcode sets CC which is why the special
//    reg/reg pseudo is needed.
//
// 4. Convert any remaining pseudos into the target opcodes that do not
//    clobber CC (this pass).
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

class SystemZFinalizeReassociation : public MachineFunctionPass {
public:
  static char ID;
  SystemZFinalizeReassociation()
    : MachineFunctionPass(ID), TII(nullptr), MRI(nullptr) {
    initializeSystemZFinalizeReassociationPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;

private:

  bool visitMBB(MachineBasicBlock &MBB);

  const SystemZInstrInfo *TII;
  MachineRegisterInfo *MRI;
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
  for (MachineInstr &MI : llvm::make_early_inc_range(MBB)) {
    unsigned PseudoOpcode = MI.getOpcode();
    unsigned TargetOpcode =
      PseudoOpcode == SystemZ::WFADB_CCPseudo    ? SystemZ::WFADB
      : PseudoOpcode == SystemZ::WFASB_CCPseudo  ? SystemZ::WFASB
      : PseudoOpcode == SystemZ::WFSDB_CCPseudo  ? SystemZ::WFSDB
      : PseudoOpcode == SystemZ::WFSSB_CCPseudo  ? SystemZ::WFSSB
      : PseudoOpcode == SystemZ::WFMDB_CCPseudo  ? SystemZ::WFMDB
      : PseudoOpcode == SystemZ::WFMSB_CCPseudo  ? SystemZ::WFMSB
      : PseudoOpcode == SystemZ::WFMADB_CCPseudo ? SystemZ::WFMADB
      : PseudoOpcode == SystemZ::WFMASB_CCPseudo ? SystemZ::WFMASB
      : 0;
    if (TargetOpcode) {
      // PeepholeOptimizer will not fold any loads across basic blocks, which
      // however seems beneficial, so do it here:
      bool Folded = false;
      for (unsigned Op = 1; Op <= 2; ++Op) {
        Register Reg = MI.getOperand(Op).getReg();
        if (MachineInstr *DefMI = MRI->getVRegDef(Reg))
          if (TII->optimizeLoadInstr(MI, MRI, Reg, DefMI)) {
            MI.eraseFromParent();
            DefMI->eraseFromParent();
            MRI->markUsesInDebugValueAsUndef(Reg);
            Folded = true;
            break;
          }
      }

      if (!Folded) {
        MI.setDesc(TII->get(TargetOpcode));
        int CCIdx = MI.findRegisterDefOperandIdx(SystemZ::CC);
        MI.removeOperand(CCIdx);
      }
      Changed = true;
    }
  }
  return Changed;
}

bool SystemZFinalizeReassociation::runOnMachineFunction(MachineFunction &F) {
  TII = F.getSubtarget<SystemZSubtarget>().getInstrInfo();
  MRI = &F.getRegInfo();

  bool Modified = false;
  for (auto &MBB : F)
    Modified |= visitMBB(MBB);

  return Modified;
}
