//===-- RISCVLatePeephole.cpp - Late stage peephole optimization ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file provides RISC-V specific target descriptions.
///
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/RISCVMCTargetDesc.h"
#include "RISCV.h"
#include "RISCVInstrInfo.h"
#include "RISCVSubtarget.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineBranchProbabilityInfo.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-late-peephole"
#define RISCV_LATE_PEEPHOLE_NAME "RISC-V Late Stage Peephole"

namespace {

struct RISCVLatePeepholeOpt : public MachineFunctionPass {
  static char ID;

  RISCVLatePeepholeOpt() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override { return RISCV_LATE_PEEPHOLE_NAME; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MachineDominatorTreeWrapperPass>();
    AU.addPreserved<MachineDominatorTreeWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &Fn) override;

private:
  void removeBlock(MachineBasicBlock *B, MachineBasicBlock *NewB);
  bool removeSingleBranchBlock(MachineBasicBlock *B);

  const RISCVInstrInfo *RII = nullptr;
  MachineFunction *MFN = nullptr;
  MachineDominatorTree *MDT = nullptr;
};
} // namespace

char RISCVLatePeepholeOpt::ID = 0;
INITIALIZE_PASS(RISCVLatePeepholeOpt, "riscv-late-peephole",
                RISCV_LATE_PEEPHOLE_NAME, false, false)

void RISCVLatePeepholeOpt::removeBlock(MachineBasicBlock *B, MachineBasicBlock *NewB) {
  LLVM_DEBUG(dbgs() << "Removing block '#'" << B->getNumber() << "\n");

  // Transfer the immediate dominator information from B to its descendants.
  MachineDomTreeNode *N = MDT->getNode(B);
  MachineDomTreeNode *IDN = N->getIDom();
  if (IDN) {
    MachineBasicBlock *IDB = IDN->getBlock();

    using GTN = GraphTraits<MachineDomTreeNode *>;
    using DTNodeVectType = SmallVector<MachineDomTreeNode *, 4>;

    DTNodeVectType Cn(GTN::child_begin(N), GTN::child_end(N));
    for (auto &I : Cn) {
      MachineBasicBlock *SB = I->getBlock();
      MDT->changeImmediateDominator(SB, IDB);
    }
  }

  while (!B->succ_empty())
    B->removeSuccessor(B->succ_begin());

  for (MachineBasicBlock *Pred : B->predecessors()) {
    Pred->removeSuccessor(B, true);
    // TODO: how do I calculate the branch probability here?
    if (Pred != NewB)
      Pred->addSuccessor(NewB);
  }

  MDT->eraseNode(B);
  MFN->erase(B->getIterator());
}

bool RISCVLatePeepholeOpt::removeSingleBranchBlock(MachineBasicBlock *B) {
  LLVM_DEBUG(dbgs() << "Checking flow pattern at " << printMBBReference(*B)
                    << "\n");

  if (B->size() != 1)
    return false;

  MachineBasicBlock::const_iterator T1I = B->getFirstTerminator();
  if (T1I == B->end())
    return false;
  unsigned Opc = T1I->getOpcode();
  if (Opc != RISCV::BEQ && Opc != RISCV::BNE)
    return false;

  Register DstReg = T1I->getOperand(0).getReg();
  Register SrcReg = T1I->getOperand(1).getReg();
  if (DstReg != SrcReg)
    return false;

  // Get the layout successor, or 0 if B does not have one.
  MachineFunction::iterator NextBI = std::next(MachineFunction::iterator(B));
  MachineBasicBlock *NextB = (NextBI != MFN->end()) ? &*NextBI : nullptr;

  MachineBasicBlock *T1B = T1I->getOperand(2).getMBB();
  assert(std::next(T1I) == B->end());

  MachineBasicBlock *T2B = NextB;

  MachineBasicBlock *PredB = B;
  MachineBasicBlock *SuccB = Opc == RISCV::BEQ ? T1B : T2B;
  MachineBasicBlock *DiscB = Opc == RISCV::BEQ ? T2B : T1B;

  LLVM_DEBUG(dbgs() << "Merging blocks '#'" << PredB->getNumber() << " and '#'"
                    << SuccB->getNumber() << "\n");

  RII->removeBranch(*PredB);
  PredB->removeSuccessor(DiscB);
  PredB->splice(PredB->end(), SuccB, SuccB->begin(), SuccB->end());
  removeBlock(SuccB, PredB);
  return true;
}

bool RISCVLatePeepholeOpt::runOnMachineFunction(MachineFunction &Fn) {
  if (skipFunction(Fn.getFunction()))
    return false;

  auto &ST = Fn.getSubtarget<RISCVSubtarget>();
  RII = ST.getInstrInfo();
  MFN = &Fn;
  MDT = &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();

  bool Changed = false;

  for (MachineBasicBlock &MBB : Fn)
    Changed |= removeSingleBranchBlock(&MBB);

  return Changed;
}

/// Returns an instance of the Make Compressible Optimization pass.
FunctionPass *llvm::createRISCVLatePeepholeOptPass() {
  return new RISCVLatePeepholeOpt();
}
