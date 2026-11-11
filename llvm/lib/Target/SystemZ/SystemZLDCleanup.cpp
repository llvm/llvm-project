//===-- SystemZLDCleanup.cpp - Clean up local-dynamic TLS accesses --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass combines multiple accesses to local-dynamic TLS variables so that
// the TLS base address for the module is only fetched once per execution path
// through the function.
//
//===----------------------------------------------------------------------===//

#include "SystemZ.h"
#include "SystemZMachineFunctionInfo.h"
#include "SystemZTargetMachine.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionAnalysisManager.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/IR/Analysis.h"

using namespace llvm;

namespace {

class SystemZLDCleanupLegacy : public MachineFunctionPass {
public:
  static char ID;
  SystemZLDCleanupLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;
};

char SystemZLDCleanupLegacy::ID = 0;

} // end anonymous namespace

INITIALIZE_PASS(SystemZLDCleanupLegacy, "systemz-ld-cleanup",
                "SystemZ Local Dynamic TLS Access Clean-up", false, false)

FunctionPass *llvm::createSystemZLDCleanupLegacyPass(SystemZTargetMachine &TM) {
  return new SystemZLDCleanupLegacy();
}

void SystemZLDCleanupLegacy::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  AU.addRequired<MachineDominatorTreeWrapperPass>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

// Replace the TLS_LDCALL instruction I with a copy from TLSBaseAddrReg,
// returning the new instruction.
static MachineInstr *replaceTLSCall(MachineInstr *I, unsigned TLSBaseAddrReg,
                                    const TargetInstrInfo *TII) {
  // Insert a Copy from TLSBaseAddrReg to R2.
  MachineInstr *Copy = BuildMI(*I->getParent(), I, I->getDebugLoc(),
                               TII->get(TargetOpcode::COPY), SystemZ::R2D)
                           .addReg(TLSBaseAddrReg);

  // Erase the TLS_LDCALL instruction.
  I->eraseFromParent();

  return Copy;
}

// Create a virtual register in *TLSBaseAddrReg, and populate it by
// inserting a copy instruction after I. Returns the new instruction.
static MachineInstr *setRegister(MachineFunction &MF, MachineInstr *I,
                                 unsigned *TLSBaseAddrReg,
                                 const TargetInstrInfo *TII) {
  // Create a virtual register for the TLS base address.
  MachineRegisterInfo &RegInfo = MF.getRegInfo();
  *TLSBaseAddrReg = RegInfo.createVirtualRegister(&SystemZ::GR64BitRegClass);

  // Insert a copy from R2 to TLSBaseAddrReg.
  MachineInstr *Next = I->getNextNode();
  MachineInstr *Copy = BuildMI(*I->getParent(), Next, I->getDebugLoc(),
                               TII->get(TargetOpcode::COPY), *TLSBaseAddrReg)
                           .addReg(SystemZ::R2D);

  return Copy;
}

// Visit the dominator subtree rooted at Node in pre-order.
// If TLSBaseAddrReg is non-null, then use that to replace any
// TLS_LDCALL instructions. Otherwise, create the register
// when the first such instruction is seen, and then use it
// as we encounter more instructions.
static bool visitNode(MachineFunction &MF, MachineDomTreeNode *Node,
                      unsigned TLSBaseAddrReg, const TargetInstrInfo *TII) {
  MachineBasicBlock *BB = Node->getBlock();
  bool Changed = false;

  // Traverse the current block.
  for (auto I = BB->begin(), E = BB->end(); I != E; ++I) {
    switch (I->getOpcode()) {
      case SystemZ::TLS_LDCALL:
        if (TLSBaseAddrReg)
          I = replaceTLSCall(&*I, TLSBaseAddrReg, TII);
        else
          I = setRegister(MF, &*I, &TLSBaseAddrReg, TII);
        Changed = true;
        break;
      default:
        break;
    }
  }

  // Visit the children of this block in the dominator tree.
  for (MachineDomTreeNode *Child : *Node)
    Changed |= visitNode(MF, Child, TLSBaseAddrReg, TII);

  return Changed;
}

static bool runLDCleanupOnMF(MachineFunction &MF,
                             function_ref<MachineDominatorTree *()> GetMDT) {
  const TargetInstrInfo *TII =
      MF.getSubtarget<SystemZSubtarget>().getInstrInfo();

  SystemZMachineFunctionInfo *MFI = MF.getInfo<SystemZMachineFunctionInfo>();
  if (MFI->getNumLocalDynamicTLSAccesses() < 2) {
    // No point folding accesses if there isn't at least two.
    return false;
  }

  MachineDominatorTree *DT = GetMDT();
  return visitNode(MF, DT->getRootNode(), 0, TII);
}

bool SystemZLDCleanupLegacy::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  return runLDCleanupOnMF(MF, [&]() {
    return &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  });
}

PreservedAnalyses
SystemZLDCleanupPass::run(MachineFunction &MF,
                          MachineFunctionAnalysisManager &MFAM) {
  bool Changed = runLDCleanupOnMF(
      MF, [&]() { return &MFAM.getResult<MachineDominatorTreeAnalysis>(MF); });
  if (!Changed)
    return PreservedAnalyses::all();
  return getMachineFunctionPassPreservedAnalyses().preserveSet<CFGAnalyses>();
}
