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
  BitVector ClobberedRegs;

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

/// Remember what registers the specified instruction modifies.
static void trackRegDefs(const MachineInstr &MI, BitVector &ClobberedRegs,
                         const TargetRegisterInfo *TRI) {
  for (const MachineOperand &MO : MI.operands()) {
    if (MO.isRegMask()) {
      ClobberedRegs.setBitsNotInMask(MO.getRegMask());
      continue;
    }

    if (!MO.isReg())
      continue;
    unsigned Reg = MO.getReg();
    if (!Reg)
      continue;
    if (!MO.isDef())
      continue;

    for (MCRegAliasIterator AI(Reg, TRI, true); AI.isValid(); ++AI)
      ClobberedRegs.set(*AI);
  }
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

  // Keep track of the earliest point in the PredMBB block where kill markers
  // need to be removed if a COPY is removed.
  MachineBasicBlock::iterator FirstUse;
  // Registers that are known to contain zeros at the start of MBB.
  SmallVector<MCPhysReg, 4> KnownZeroRegs;
  // Registers clobbered in PredMBB between CompBr instruction and current
  // instruction being checked in loop.
  ClobberedRegs.reset();
  ++CompBr;
  do {
    --CompBr;
    if (!guaranteesZeroRegInBlock(*CompBr, MBB))
      continue;

    KnownZeroRegs.push_back(CompBr->getOperand(0).getReg());
    FirstUse = CompBr;
    // Look backward in PredMBB for COPYs from the known zero reg to
    // find other registers that are known to be zero.
    for (auto PredI = CompBr;; --PredI) {
      if (PredI->isCopy()) {
        MCPhysReg CopyDstReg = PredI->getOperand(0).getReg();
        MCPhysReg CopySrcReg = PredI->getOperand(1).getReg();
        for (MCPhysReg KnownZeroReg : KnownZeroRegs) {
          if (ClobberedRegs[KnownZeroReg])
            continue;
          // If we have X = COPY Y, and Y is known to be zero, then now X is
          // known to be zero.
          if (CopySrcReg == KnownZeroReg && !ClobberedRegs[CopyDstReg]) {
            KnownZeroRegs.push_back(CopyDstReg);
            FirstUse = PredI;
            break;
          }
          // If we have X = COPY Y, and X is known to be zero, then now Y is
          // known to be zero.
          if (CopyDstReg == KnownZeroReg && !ClobberedRegs[CopySrcReg]) {
            KnownZeroRegs.push_back(CopySrcReg);
            FirstUse = PredI;
            break;
          }
        }
      }

      // Stop if we get to the beginning of PredMBB.
      if (PredI == PredMBB->begin())
        break;

      trackRegDefs(*PredI, ClobberedRegs, TRI);
      // Stop if all of the known-zero regs have been clobbered.
      if (all_of(KnownZeroRegs, [&](MCPhysReg KnownZeroReg) {
            return ClobberedRegs[KnownZeroReg];
          }))
        break;
    }
    break;

  } while (CompBr != PredMBB->begin() && CompBr->isTerminator());

  // We've not found a known zero register, time to bail out.
  if (KnownZeroRegs.empty())
    return false;

  bool Changed = false;
  // UsedKnownZeroRegs is the set of KnownZeroRegs that have had uses added to MBB.
  SmallSetVector<unsigned, 4> UsedKnownZeroRegs;
  MachineBasicBlock::iterator LastChange = MBB->begin();
  // Remove redundant Copy instructions unless KnownZeroReg is modified.
  for (MachineBasicBlock::iterator I = MBB->begin(), E = MBB->end(); I != E;) {
    MachineInstr *MI = &*I;
    ++I;
    bool RemovedCopy = false;
    if (MI->isCopy()) {
      MCPhysReg DefReg = MI->getOperand(0).getReg();
      MCPhysReg SrcReg = MI->getOperand(1).getReg();

      if ((SrcReg == AArch64::XZR || SrcReg == AArch64::WZR) &&
          !MRI->isReserved(DefReg)) {
        for (MCPhysReg KnownZeroReg : KnownZeroRegs) {
          if (KnownZeroReg == DefReg ||
              TRI->isSuperRegister(DefReg, KnownZeroReg)) {
            DEBUG(dbgs() << "Remove redundant Copy : " << *MI);

            MI->eraseFromParent();
            Changed = true;
            LastChange = I;
            NumCopiesRemoved++;
            UsedKnownZeroRegs.insert(KnownZeroReg);
            RemovedCopy = true;
            break;
          }
        }
      }
    }

    // Skip to the next instruction if we removed the COPY from WZR/XZR.
    if (RemovedCopy)
      continue;

    // Remove any regs the MI clobbers from the KnownZeroRegs set.
    for (unsigned RI = 0; RI < KnownZeroRegs.size();)
      if (MI->modifiesRegister(KnownZeroRegs[RI], TRI)) {
        std::swap(KnownZeroRegs[RI], KnownZeroRegs[KnownZeroRegs.size() - 1]);
        KnownZeroRegs.pop_back();
        // Don't increment RI since we need to now check the swapped-in
        // KnownZeroRegs[RI].
      } else {
        ++RI;
      }

    // Continue until the KnownZeroRegs set is empty.
    if (KnownZeroRegs.empty())
      break;
  }

  if (!Changed)
    return false;

  // Add newly used regs to the block's live-in list if they aren't there
  // already.
  for (MCPhysReg KnownZeroReg : UsedKnownZeroRegs)
    if (!MBB->isLiveIn(KnownZeroReg))
      MBB->addLiveIn(KnownZeroReg);

  // Clear kills in the range where changes were made.  This is conservative,
  // but should be okay since kill markers are being phased out.
  for (MachineInstr &MMI : make_range(FirstUse, PredMBB->end()))
    MMI.clearKillInfo();
  for (MachineInstr &MMI : make_range(MBB->begin(), LastChange))
    MMI.clearKillInfo();

  return true;
}

bool AArch64RedundantCopyElimination::runOnMachineFunction(
    MachineFunction &MF) {
  if (skipFunction(*MF.getFunction()))
    return false;
  TRI = MF.getSubtarget().getRegisterInfo();
  MRI = &MF.getRegInfo();
  ClobberedRegs.resize(TRI->getNumRegs());
  bool Changed = false;
  for (MachineBasicBlock &MBB : MF)
    Changed |= optimizeCopy(&MBB);
  return Changed;
}

FunctionPass *llvm::createAArch64RedundantCopyEliminationPass() {
  return new AArch64RedundantCopyElimination();
}
