//===----------- CFIInfoVerifier.cpp - CFI Information Verifier -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass verifies incoming and outgoing CFI information of basic blocks. CFI
// information is information about offset and register set by CFI directives,
// valid at the start and end of a basic block. This pass checks that outgoing
// information of predecessors matches incoming information of their successors.
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Target/TargetMachine.h"
using namespace llvm;

namespace {
class CFIInfoVerifier : public MachineFunctionPass {
 public:
  static char ID;

  CFIInfoVerifier() : MachineFunctionPass(ID) {
    initializeCFIInfoVerifierPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override {
    bool NeedsDwarfCFI = (MF.getMMI().hasDebugInfo() ||
                          MF.getFunction()->needsUnwindTableEntry()) &&
                         (!MF.getTarget().getTargetTriple().isOSDarwin() &&
                          !MF.getTarget().getTargetTriple().isOSWindows());
    if (!NeedsDwarfCFI) return false;
    verify(MF);
    return false;
  }

 private:
  // Go through each MBB in a function and check that outgoing offset and
  // register of its predecessors match incoming offset and register of that
  // MBB, as well as that incoming offset and register of its successors match
  // outgoing offset and register of the MBB.
  void verify(MachineFunction &MF);
  void report(const char *msg, MachineBasicBlock &MBB);
};
}

char CFIInfoVerifier::ID = 0;
INITIALIZE_PASS(CFIInfoVerifier, "cfiinfoverifier",
                "Verify that corresponding in/out CFI info matches", false,
                false)
FunctionPass *llvm::createCFIInfoVerifier() { return new CFIInfoVerifier(); }

void CFIInfoVerifier::verify(MachineFunction &MF) {
  for (auto &CurrMBB : MF) {
    for (auto Pred : CurrMBB.predecessors()) {
      // Check that outgoing offset values of predecessors match the incoming
      // offset value of CurrMBB
      if (Pred->getOutgoingCFAOffset() != CurrMBB.getIncomingCFAOffset()) {
        report("The outgoing offset of a predecessor is inconsistent.",
               CurrMBB);
        errs() << "Predecessor BB#" << Pred->getNumber()
               << " has outgoing offset (" << Pred->getOutgoingCFAOffset()
               << "), while BB#" << CurrMBB.getNumber()
               << " has incoming offset (" << CurrMBB.getIncomingCFAOffset()
               << ").\n";
      }
      // Check that outgoing register values of predecessors match the incoming
      // register value of CurrMBB
      if (Pred->getOutgoingCFARegister() != CurrMBB.getIncomingCFARegister()) {
        report("The outgoing register of a predecessor is inconsistent.",
               CurrMBB);
        errs() << "Predecessor BB#" << Pred->getNumber()
               << " has outgoing register (" << Pred->getOutgoingCFARegister()
               << "), while BB#" << CurrMBB.getNumber()
               << " has incoming register (" << CurrMBB.getIncomingCFARegister()
               << ").\n";
      }
    }

    for (auto Succ : CurrMBB.successors()) {
      // Check that incoming offset values of successors match the outgoing
      // offset value of CurrMBB
      if (Succ->getIncomingCFAOffset() != CurrMBB.getOutgoingCFAOffset()) {
        report("The incoming offset of a successor is inconsistent.", CurrMBB);
        errs() << "Successor BB#" << Succ->getNumber()
               << " has incoming offset (" << Succ->getIncomingCFAOffset()
               << "), while BB#" << CurrMBB.getNumber()
               << " has outgoing offset (" << CurrMBB.getOutgoingCFAOffset()
               << ").\n";
      }
      // Check that incoming register values of successors match the outgoing
      // register value of CurrMBB
      if (Succ->getIncomingCFARegister() != CurrMBB.getOutgoingCFARegister()) {
        report("The incoming register of a successor is inconsistent.",
               CurrMBB);
        errs() << "Successor BB#" << Succ->getNumber()
               << " has incoming register (" << Succ->getIncomingCFARegister()
               << "), while BB#" << CurrMBB.getNumber()
               << " has outgoing register (" << CurrMBB.getOutgoingCFARegister()
               << ").\n";
      }
    }
  }
}

void CFIInfoVerifier::report(const char *msg, MachineBasicBlock &MBB) {
  assert(&MBB);
  errs() << '\n';
  errs() << "*** " << msg << " ***\n"
         << "- function:    " << MBB.getParent()->getName() << "\n";
  errs() << "- basic block: BB#" << MBB.getNumber() << ' ' << MBB.getName()
         << " (" << (const void *)&MBB << ')';
  errs() << '\n';
}
