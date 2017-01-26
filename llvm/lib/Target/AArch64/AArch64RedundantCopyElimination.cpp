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

namespace {
class AArch64RedundantCopyElimination : public MachineFunctionPass {
  const MachineRegisterInfo *MRI;
  const TargetRegisterInfo *TRI;

public:
  static char ID;
  AArch64RedundantCopyElimination() : MachineFunctionPass(ID) {
    initializeAArch64RedundantCopyEliminationPass(
        *PassRegistry::getPassRegistry());
  }
  bool optimizeCopy(MachineBasicBlock *MBB);
  bool runOnMachineFunction(MachineFunction &MF) override;
  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::NoVRegs);
  }
  StringRef getPassName() const override {
    return "AArch64 Redundant Copy Elimination";
  }
};
char AArch64RedundantCopyElimination::ID = 0;
}

INITIALIZE_PASS(AArch64RedundantCopyElimination, "aarch64-copyelim",
                "AArch64 redundant copy elimination pass", false, false)

static bool guaranteesZeroRegInBlock(MachineInstr &MI, MachineBasicBlock *MBB) {
  unsigned Opc = MI.getOpcode();
  // Check if the current basic block is the target block to which the
  // CBZ/CBNZ instruction jumps when its Wt/Xt is zero.
  return ((Opc == AArch64::CBZW || Opc == AArch64::CBZX) &&
          MBB == MI.getOperand(1).getMBB()) ||
         ((Opc == AArch64::CBNZW || Opc == AArch64::CBNZX) &&
          MBB != MI.getOperand(1).getMBB());
}

bool AArch64RedundantCopyElimination::optimizeCopy(MachineBasicBlock *MBB) {
  // Check if the current basic block has a single predecessor.
  if (MBB->pred_size() != 1)
    return false;

  // Check if the predecessor has two successors, implying the block ends in a
  // conditional branch.
  MachineBasicBlock *PredMBB = *MBB->pred_begin();
  if (PredMBB->succ_size() != 2)
    return false;

  MachineBasicBlock::iterator CompBr = PredMBB->getLastNonDebugInstr();
  if (CompBr == PredMBB->end())
    return false;

  ++CompBr;
  do {
    --CompBr;
    if (guaranteesZeroRegInBlock(*CompBr, MBB))
      break;
  } while (CompBr != PredMBB->begin() && CompBr->isTerminator());

  // We've not found a CBZ/CBNZ, time to bail out.
  if (!guaranteesZeroRegInBlock(*CompBr, MBB))
    return false;

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
  MachineBasicBlock::iterator LastChange = MBB->begin();
  unsigned SmallestDef = TargetReg;
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
        DEBUG(dbgs() << "Remove redundant Copy : ");
        DEBUG((MI)->print(dbgs()));

        MI->eraseFromParent();
        Changed = true;
        LastChange = I;
        NumCopiesRemoved++;
        SmallestDef =
            TRI->isSubRegister(SmallestDef, DefReg) ? DefReg : SmallestDef;
        continue;
      }
    }

    if (MI->modifiesRegister(TargetReg, TRI))
      break;
  }

  if (!Changed)
    return false;

  // Otherwise, we have to fixup the use-def chain, starting with the
  // CBZ/CBNZ. Conservatively mark as much as we can live.
  CompBr->clearRegisterKills(SmallestDef, TRI);

  if (none_of(TargetRegs, [&](unsigned Reg) { return MBB->isLiveIn(Reg); }))
    MBB->addLiveIn(TargetReg);

  // Clear any kills of TargetReg between CompBr and the last removed COPY.
  for (MachineInstr &MMI : make_range(MBB->begin(), LastChange))
    MMI.clearRegisterKills(SmallestDef, TRI);

  return true;
}

bool AArch64RedundantCopyElimination::runOnMachineFunction(
    MachineFunction &MF) {
  if (skipFunction(*MF.getFunction()))
    return false;
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
