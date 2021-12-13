//===- NanoMipsMoveOptimizer.cpp - nanoMIPS redundant copy elimination ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Adapted from Aarch64's AArch64RedundantCopyElimination pass, which is
// mostly architecture-neutral so ideally could be refactored into a
// mostly-common pass.
//
// The pass uses dominating control-flow information and conditions to
// find cases where a constant value is moved into a register which is
// known to already hold that value. There are several instances of
// this pattern in CoreMark in particular. NanoMips's use of general
// purpose registers for conditions simplifies this process:
//
//    bnezc  a0, LB12
//    move   a0, zero // redundant as dominating condition ensures a0 == 0
//
// Or:
//    beqc   a1, 7, LB7
//    ...
//    LB7:
//    move   a1, 7
//
// Redundancy like this is usually removed by GVN etc in the
// middle-end, but can be created after lowering to MIR by transforms
// like tail duplication that modify control flow.
//
//===----------------------------------------------------------------------===//

#include "Mips.h"
#include "MipsSubtarget.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include <algorithm>
#include <cmath>


#define DEBUG_TYPE "redundantcopyelim"

using namespace llvm;

namespace {
class RedundantCopyElimination : public MachineFunctionPass {
  const MachineRegisterInfo *MRI;
  const TargetRegisterInfo *TRI;

  // OptBBClobberedRegs is used when optimizing away redundant copies/moves.
  LiveRegUnits OptBBClobberedRegs, OptBBUsedRegs;

public:
  static char ID;
  RedundantCopyElimination() : MachineFunctionPass(ID) {
    initializeRedundantCopyEliminationPass(
        *PassRegistry::getPassRegistry());
  }

  struct RegImm {
    MCPhysReg Reg;
    int32_t Imm;
    RegImm(MCPhysReg Reg, int32_t Imm) : Reg(Reg), Imm(Imm) {}
  };

  bool knownRegValInBlock(MachineInstr &CondBr, MachineBasicBlock *MBB,
                          SmallVectorImpl<RegImm> &KnownRegs,
                          MachineBasicBlock::iterator &FirstUse);
  bool optimizeBlock(MachineBasicBlock *MBB);
  bool runOnMachineFunction(MachineFunction &MF) override;
  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::NoVRegs);
  }
  StringRef getPassName() const override {
    return "Redundant Copy Elimination";
  }

};

} // namespace

char RedundantCopyElimination::ID = 0;


INITIALIZE_PASS(RedundantCopyElimination, DEBUG_TYPE,
                "Redundant copy elimination pass", false, false)


/// It's possible to determine the value of a register based on a dominating
/// condition.  To do so, this function checks to see if the basic block \p MBB
/// is the target of a conditional branch \p CondBr with an equality comparison.
/// If the branch is a CBZ/CBNZ, we know the value of its source operand is zero
/// in \p MBB for some cases.  Otherwise, we find and inspect the NZCV setting
/// instruction (e.g., SUBS, ADDS).  If this instruction defines a register
/// other than WZR/XZR, we know the value of the destination register is zero in
/// \p MMB for some cases.  In addition, if the NZCV setting instruction is
/// comparing against a constant we know the other source register is equal to
/// the constant in \p MBB for some cases.  If we find any constant values, push
/// a physical register and constant value pair onto the KnownRegs vector and
/// return true.  Otherwise, return false if no known values were found.
bool RedundantCopyElimination::knownRegValInBlock(
    MachineInstr &CondBr, MachineBasicBlock *MBB,
    SmallVectorImpl<RegImm> &KnownRegs, MachineBasicBlock::iterator &FirstUse) {
  unsigned Opc = CondBr.getOpcode();

  // Check if the current basic block is the target block to which the
  // BEQZC/BNEZC instruction jumps when its operand is zero.
  if ((Opc == Mips::BEQZC_NM &&
       MBB == CondBr.getOperand(1).getMBB()) ||
      (Opc == Mips::BNEZC_NM &&
       MBB != CondBr.getOperand(1).getMBB())) {
    FirstUse = CondBr;
    KnownRegs.push_back(RegImm(CondBr.getOperand(0).getReg(), 0));
    return true;
  }

  // Otherwise, must be a conditional branch
  if (!CondBr.isConditionalBranch())
    return false;

  // Compare with immediate
  if ((Opc == Mips::BEQIC_NM &&
       MBB == CondBr.getOperand(2).getMBB()) ||
      (Opc == Mips::BNEIC_NM &&
       MBB != CondBr.getOperand(2).getMBB())) {
    FirstUse = CondBr;
    KnownRegs.push_back(RegImm(CondBr.getOperand(0).getReg(), CondBr.getOperand(1).getImm()));
    return true;
  }
  
  // Must be an equality check (i.e., == or !=).
  if (Opc != Mips::BEQC_NM && Opc != Mips::BNEC_NM)
    return false;

  MachineBasicBlock *BrTarget = CondBr.getOperand(2).getMBB();
  if ((Opc == Mips::BEQC_NM && BrTarget != MBB) ||
      (Opc == Mips::BNEC_NM && BrTarget == MBB))
    return false;

  return false;

  // Stop if we get to the beginning of PredMBB.
  MachineBasicBlock *PredMBB = *MBB->pred_begin();
  assert(PredMBB == CondBr.getParent() &&
         "Conditional branch not in predecessor block!");
  if (CondBr == PredMBB->begin())
    return false;

  return false;
}

bool RedundantCopyElimination::optimizeBlock(MachineBasicBlock *MBB) {
  // Check if the current basic block has a single predecessor.
  if (MBB->pred_size() != 1)
    return false;

  // Check if the predecessor has two successors, implying the block ends in a
  // conditional branch.
  MachineBasicBlock *PredMBB = *MBB->pred_begin();
  if (PredMBB->succ_size() != 2)
    return false;

  MachineBasicBlock::iterator CondBr = PredMBB->getLastNonDebugInstr();
  if (CondBr == PredMBB->end())
    return false;

  // Keep track of the earliest point in the PredMBB block where kill markers
  // need to be removed if a COPY is removed.
  MachineBasicBlock::iterator FirstUse;
  // After calling knownRegValInBlock, FirstUse will either point to a CBZ/CBNZ
  // or a compare (i.e., SUBS).  In the latter case, we must take care when
  // updating FirstUse when scanning for COPY instructions.  In particular, if
  // there's a COPY in between the compare and branch the COPY should not
  // update FirstUse.
  bool SeenFirstUse = false;
  // Registers that contain a known value at the start of MBB.
  SmallVector<RegImm, 4> KnownRegs;

  MachineBasicBlock::iterator Itr = std::next(CondBr);
  do {
    --Itr;
    if (!knownRegValInBlock(*Itr, MBB, KnownRegs, FirstUse))
      continue;

    // Look backward in PredMBB for COPYs from the known reg to find other
    // registers that are known to be a constant value.
    for (auto PredI = Itr;; --PredI) {
      if (FirstUse == PredI)
        SeenFirstUse = true;

      if (PredI->isCopy()) {
        MCPhysReg CopyDstReg = PredI->getOperand(0).getReg();
        MCPhysReg CopySrcReg = PredI->getOperand(1).getReg();
        for (auto &KnownReg : KnownRegs) {
          if (!OptBBClobberedRegs.available(KnownReg.Reg))
            continue;
          // If we have X = COPY Y, and Y is known to be zero, then now X is
          // known to be zero.
          if (CopySrcReg == KnownReg.Reg &&
              OptBBClobberedRegs.available(CopyDstReg)) {
            KnownRegs.push_back(RegImm(CopyDstReg, KnownReg.Imm));
            if (SeenFirstUse)
              FirstUse = PredI;
            break;
          }
          // If we have X = COPY Y, and X is known to be zero, then now Y is
          // known to be zero.
          if (CopyDstReg == KnownReg.Reg &&
              OptBBClobberedRegs.available(CopySrcReg)) {
            KnownRegs.push_back(RegImm(CopySrcReg, KnownReg.Imm));
            if (SeenFirstUse)
              FirstUse = PredI;
            break;
          }
        }
      }

      // Stop if we get to the beginning of PredMBB.
      if (PredI == PredMBB->begin())
        break;

      LiveRegUnits::accumulateUsedDefed(*PredI, OptBBClobberedRegs,
                                        OptBBUsedRegs, TRI);
      // Stop if all of the known-zero regs have been clobbered.
      if (all_of(KnownRegs, [&](RegImm KnownReg) {
            return !OptBBClobberedRegs.available(KnownReg.Reg);
          }))
        break;
    }
    break;

  } while (Itr != PredMBB->begin() && Itr->isTerminator());

  // We've not found a registers with a known value, time to bail out.
  if (KnownRegs.empty())
    return false;

  bool Changed = false;
  // UsedKnownRegs is the set of KnownRegs that have had uses added to MBB.
  SmallSetVector<unsigned, 4> UsedKnownRegs;
  MachineBasicBlock::iterator LastChange = MBB->begin();
  // Remove redundant copy/move instructions unless KnownReg is modified.
  for (MachineBasicBlock::iterator I = MBB->begin(), E = MBB->end(); I != E;) {
    MachineInstr *MI = &*I;
    ++I;
    bool RemovedMI = false;
    bool IsCopy = MI->isCopy();
    bool IsMoveImm = MI->isMoveImmediate();
    if (IsCopy || IsMoveImm) {
      Register DefReg = MI->getOperand(0).getReg();
      Register SrcReg = IsCopy ? MI->getOperand(1).getReg() : Register();
      int64_t SrcImm = IsMoveImm ? MI->getOperand(1).getImm() : 0;
      if (!MRI->isReserved(DefReg) &&
          ((IsCopy && SrcReg == Mips::ZERO_NM) ||
           IsMoveImm)) {
        for (RegImm &KnownReg : KnownRegs) {
          if (KnownReg.Reg != DefReg &&
              !TRI->isSuperRegister(DefReg, KnownReg.Reg))
            continue;

          // For a copy, the known value must be a zero.
          if (IsCopy && KnownReg.Imm != 0)
            continue;

          if (IsMoveImm) {
            // For a move immediate, the known immediate must match the source
            // immediate.
            if (KnownReg.Imm != SrcImm)
              continue;

            // Don't remove a move immediate that implicitly defines the upper
            // bits when only the lower 32 bits are known.
            MCPhysReg CmpReg = KnownReg.Reg;
            if (any_of(MI->implicit_operands(), [CmpReg](MachineOperand &O) {
                  return !O.isDead() && O.isReg() && O.isDef() &&
                         O.getReg() != CmpReg;
                }))
              continue;

            // Don't remove a move immediate that implicitly defines the upper
            // bits as different.
            if (TRI->isSuperRegister(DefReg, KnownReg.Reg) && KnownReg.Imm < 0)
              continue;
          }

          if (IsCopy)
            LLVM_DEBUG(dbgs() << "Remove redundant Copy : " << *MI);
          else
            LLVM_DEBUG(dbgs() << "Remove redundant Move : " << *MI);

          MI->eraseFromParent();
          Changed = true;
          LastChange = I;
          UsedKnownRegs.insert(KnownReg.Reg);
          RemovedMI = true;
          break;
        }
      }
    }

    // Skip to the next instruction if we removed the COPY/MovImm.
    if (RemovedMI)
      continue;

    // Remove any regs the MI clobbers from the KnownConstRegs set.
    for (unsigned RI = 0; RI < KnownRegs.size();)
      if (MI->modifiesRegister(KnownRegs[RI].Reg, TRI)) {
        std::swap(KnownRegs[RI], KnownRegs[KnownRegs.size() - 1]);
        KnownRegs.pop_back();
        // Don't increment RI since we need to now check the swapped-in
        // KnownRegs[RI].
      } else {
        ++RI;
      }

    // Continue until the KnownRegs set is empty.
    if (KnownRegs.empty())
      break;
  }

  if (!Changed)
    return false;

  // Add newly used regs to the block's live-in list if they aren't there
  // already.
  for (MCPhysReg KnownReg : UsedKnownRegs)
    if (!MBB->isLiveIn(KnownReg))
      MBB->addLiveIn(KnownReg);

  // Clear kills in the range where changes were made.  This is conservative,
  // but should be okay since kill markers are being phased out.
  LLVM_DEBUG(dbgs() << "Clearing kill flags.\n\tFirstUse: " << *FirstUse);
  if (LastChange != MBB->end())
    LLVM_DEBUG(dbgs() << "\tLastChange: " << *LastChange);

  for (MachineInstr &MMI : make_range(FirstUse, PredMBB->end()))
    MMI.clearKillInfo();
  for (MachineInstr &MMI : make_range(MBB->begin(), LastChange))
    MMI.clearKillInfo();

  return true;
}

bool RedundantCopyElimination::runOnMachineFunction(
    MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;
  LLVM_DEBUG(dbgs() << "Running redundant copy elimination on " << MF.getName() << "\n");
  TRI = MF.getSubtarget().getRegisterInfo();
  MRI = &MF.getRegInfo();

  // Resize the clobbered and used register unit trackers.  We do this once per
  // function.
  OptBBClobberedRegs.init(*TRI);
  OptBBUsedRegs.init(*TRI);

  bool Changed = false;
  for (MachineBasicBlock &MBB : MF)
    Changed |= optimizeBlock(&MBB);
  return Changed;
}

namespace llvm {
FunctionPass *createRedundantCopyEliminationPass() { return new RedundantCopyElimination(); }
} // namespace llvm
