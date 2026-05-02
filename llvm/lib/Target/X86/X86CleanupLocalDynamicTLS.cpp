//===- X86CleanupLocalDynamicTLS.cpp - Cleanup local dynamic TLS access ---===//
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

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86MachineFunctionInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "x86-cleanup-local-dynamic-tls"

namespace {
class X86CleanupLocalDynamicTLSLegacy : public MachineFunctionPass {
public:
  static char ID;

  X86CleanupLocalDynamicTLSLegacy() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override {
    return "Local Dynamic TLS Access Clean-up";
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<MachineDominatorTreeWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};
} // end anonymous namespace

char X86CleanupLocalDynamicTLSLegacy::ID = 0;

FunctionPass *llvm::createCleanupLocalDynamicTLSLegacyPass() {
  return new X86CleanupLocalDynamicTLSLegacy();
}

// Replace the TLS_base_addr instruction I with a copy from
// TLSBaseAddrReg, returning the new instruction.
static MachineInstr *ReplaceTLSBaseAddrCall(MachineInstr &I,
                                            Register TLSBaseAddrReg) {
  MachineFunction *MF = I.getParent()->getParent();
  const X86Subtarget &STI = MF->getSubtarget<X86Subtarget>();
  const bool is64Bit = STI.is64Bit();
  const X86InstrInfo *TII = STI.getInstrInfo();

  // Insert a Copy from TLSBaseAddrReg to RAX/EAX.
  MachineInstr *Copy =
      BuildMI(*I.getParent(), I, I.getDebugLoc(), TII->get(TargetOpcode::COPY),
              is64Bit ? X86::RAX : X86::EAX)
          .addReg(TLSBaseAddrReg);

  // Erase the TLS_base_addr instruction.
  I.eraseFromParent();

  return Copy;
}

// Create a virtual register in *TLSBaseAddrReg, and populate it by
// inserting a copy instruction after I. Returns the new instruction.
static MachineInstr *SetRegister(MachineInstr &I, Register *TLSBaseAddrReg) {
  MachineFunction *MF = I.getParent()->getParent();
  const X86Subtarget &STI = MF->getSubtarget<X86Subtarget>();
  const bool is64Bit = STI.is64Bit();
  const X86InstrInfo *TII = STI.getInstrInfo();

  // Create a virtual register for the TLS base address.
  MachineRegisterInfo &RegInfo = MF->getRegInfo();
  *TLSBaseAddrReg = RegInfo.createVirtualRegister(is64Bit ? &X86::GR64RegClass
                                                          : &X86::GR32RegClass);

  // Insert a copy from RAX/EAX to TLSBaseAddrReg.
  MachineInstr *Next = I.getNextNode();
  MachineInstr *Copy = BuildMI(*I.getParent(), Next, I.getDebugLoc(),
                               TII->get(TargetOpcode::COPY), *TLSBaseAddrReg)
                           .addReg(is64Bit ? X86::RAX : X86::EAX);

  return Copy;
}

// Visit the dominator subtree rooted at Node in pre-order.
// If TLSBaseAddrReg is non-null, then use that to replace any
// TLS_base_addr instructions. Otherwise, create the register
// when the first such instruction is seen, and then use it
// as we encounter more instructions.
static bool VisitNode(MachineDomTreeNode *Node, Register TLSBaseAddrReg) {
  MachineBasicBlock *BB = Node->getBlock();
  bool Changed = false;

  // Traverse the current block.
  for (MachineBasicBlock::iterator I = BB->begin(), E = BB->end(); I != E;
       ++I) {
    switch (I->getOpcode()) {
    case X86::TLS_base_addr32:
    case X86::TLS_base_addr64:
      if (TLSBaseAddrReg)
        I = ReplaceTLSBaseAddrCall(*I, TLSBaseAddrReg);
      else
        I = SetRegister(*I, &TLSBaseAddrReg);
      Changed = true;
      break;
    default:
      break;
    }
  }

  // Visit the children of this block in the dominator tree.
  for (MachineDomTreeNode *I : Node->children())
    Changed |= VisitNode(I, TLSBaseAddrReg);

  return Changed;
}

static bool cleanupLocalDynamicTLS(MachineDominatorTree &DT) {
  return VisitNode(DT.getRootNode(), Register());
}

static bool shouldSkipLocalDynamicTLS(MachineFunction &MF) {
  X86MachineFunctionInfo *MFI = MF.getInfo<X86MachineFunctionInfo>();
  if (MFI->getNumLocalDynamicTLSAccesses() < 2) {
    // No point folding accesses if there isn't at least two.
    return true;
  }
  return false;
}

bool X86CleanupLocalDynamicTLSLegacy::runOnMachineFunction(
    MachineFunction &MF) {
  if (skipFunction(MF.getFunction()) || shouldSkipLocalDynamicTLS(MF))
    return false;

  MachineDominatorTree &DT =
      getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  return cleanupLocalDynamicTLS(DT);
}

PreservedAnalyses
X86CleanupLocalDynamicTLSPass::run(MachineFunction &MF,
                                   MachineFunctionAnalysisManager &MFAM) {
  if (shouldSkipLocalDynamicTLS(MF))
    return PreservedAnalyses::all();

  MachineDominatorTree &DT = MFAM.getResult<MachineDominatorTreeAnalysis>(MF);
  return cleanupLocalDynamicTLS(DT) ? getMachineFunctionPassPreservedAnalyses()
                                          .preserveSet<CFGAnalyses>()
                                    : PreservedAnalyses::all();
}
