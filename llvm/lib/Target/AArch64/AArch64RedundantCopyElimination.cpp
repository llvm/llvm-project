//=- AArch64RedundantCopyElimination.cpp - Remove useless copy for AArch64 -=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
// This pass removes unnecessary zero copies in BBs that are targets of
// cbz/cbnz instructions. For instance, the copy instruction in the code below
// can be removed because the CBZW jumps to BB#2 when W0 is zero.
//  BB#1:
//    CBZW %W0, <BB#2>
//  BB#2:
//    %W0 = COPY %WZR
// This pass should be run after register allocation.
//
// FIXME: This should be extended to handle any constant other than zero. E.g.,
//   cmp w0, #1
//     b.eq .BB1
//   BB1:
//     mov w0, #1
//
// FIXME: This could also be extended to check the whole dominance subtree below
// the comparison if the compile time regression is acceptable.
//
//===----------------------------------------------------------------------===//

#include "AArch64.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "aarch64-copyelim"

STATISTIC(NumCopiesRemoved, "Number of copies removed.");

namespace llvm {
void initializeAArch64RedundantCopyEliminationPass(PassRegistry &);
}

namespace {
class AArch64RedundantCopyElimination : public MachineFunctionPass {
  const MachineRegisterInfo *MRI;
  const TargetRegisterInfo *TRI;

public:
  static char ID;
  AArch64RedundantCopyElimination() : MachineFunctionPass(ID) {}
  bool optimizeCopy(MachineBasicBlock *MBB);
  bool runOnMachineFunction(MachineFunction &MF) override;
  const char *getPassName() const override {
    return "AArch64 Redundant Copy Elimination";
  }
};
char AArch64RedundantCopyElimination::ID = 0;
}

INITIALIZE_PASS(AArch64RedundantCopyElimination, "aarch64-copyelim",
                "AArch64 redundant copy elimination pass", false, false)

bool AArch64RedundantCopyElimination::optimizeCopy(MachineBasicBlock *MBB) {
  // Check if the current basic block has a single predecessor.
  if (MBB->pred_size() != 1)
    return false;

  MachineBasicBlock *PredMBB = *MBB->pred_begin();
  MachineBasicBlock::iterator CompBr = PredMBB->getLastNonDebugInstr();
  if (CompBr == PredMBB->end() || PredMBB->succ_size() != 2)
    return false;

  unsigned LastOpc = CompBr->getOpcode();
  // Check if the current basic block is the target block to which the cbz/cbnz
  // instruction jumps when its Wt/Xt is zero.
  if (LastOpc == AArch64::CBZW || LastOpc == AArch64::CBZX) {
    if (MBB != CompBr->getOperand(1).getMBB())
      return false;
  } else if (LastOpc == AArch64::CBNZW || LastOpc == AArch64::CBNZX) {
    if (MBB == CompBr->getOperand(1).getMBB())
      return false;
  } else {
    return false;
  }

  unsigned TargetReg = CompBr->getOperand(0).getReg();
  if (!TargetReg)
    return false;
  assert(TargetRegisterInfo::isPhysicalRegister(TargetReg) &&
         "Expect physical register");

  // Remember all registers aliasing with TargetReg.
  SmallSetVector<unsigned, 8> TargetRegs;
  for (MCRegAliasIterator AI(TargetReg, TRI, true); AI.isValid(); ++AI)
    TargetRegs.insert(*AI);

  bool Changed = false;
  // Remove redundant Copy instructions unless TargetReg is modified.
  for (MachineBasicBlock::iterator I = MBB->begin(), E = MBB->end(); I != E;) {
    MachineInstr *MI = &*I;
    ++I;
    if (MI->isCopy() && MI->getOperand(0).isReg() &&
        MI->getOperand(1).isReg()) {

      unsigned DefReg = MI->getOperand(0).getReg();
      unsigned SrcReg = MI->getOperand(1).getReg();

      if ((SrcReg == AArch64::XZR || SrcReg == AArch64::WZR) &&
          !MRI->isReserved(DefReg) &&
          (TargetReg == DefReg || TRI->isSuperRegister(DefReg, TargetReg))) {

        CompBr->clearRegisterKills(DefReg, TRI);
        if (MBB->isLiveIn(DefReg))
          // Clear any kills of TargetReg between CompBr and MI.
          for (MachineInstr &MMI :
               make_range(MBB->begin()->getIterator(), MI->getIterator()))
            MMI.clearRegisterKills(DefReg, TRI);
        else
          MBB->addLiveIn(DefReg);

        DEBUG(dbgs() << "Remove redundant Copy : ");
        DEBUG((MI)->print(dbgs()));

        MI->eraseFromParent();
        Changed = true;
        NumCopiesRemoved++;
        continue;
      }
    }

    for (const MachineOperand &MO : MI->operands()) {
      // FIXME: It is possible to use the register mask to check if all
      // registers in TargetRegs are not clobbered. For now, we treat it like
      // a basic block boundary.
      if (MO.isRegMask())
        return Changed;
      if (!MO.isReg())
        continue;
      unsigned Reg = MO.getReg();

      if (!Reg)
        continue;

      assert(TargetRegisterInfo::isPhysicalRegister(Reg) &&
             "Expect physical register");

      // Stop if the TargetReg is modified.
      if (MO.isDef() && TargetRegs.count(Reg))
        return Changed;
    }
  }
  return Changed;
}

bool AArch64RedundantCopyElimination::runOnMachineFunction(
    MachineFunction &MF) {
  TRI = MF.getSubtarget().getRegisterInfo();
  MRI = &MF.getRegInfo();
  bool Changed = false;
  for (MachineBasicBlock &MBB : MF)
    Changed |= optimizeCopy(&MBB);
  return Changed;
}

FunctionPass *llvm::createAArch64RedundantCopyEliminationPass() {
  return new AArch64RedundantCopyElimination();
}
