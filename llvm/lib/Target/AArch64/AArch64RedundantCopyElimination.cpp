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
// Similarly, this pass also handles non-zero copies.
//  BB#0:
//    cmp x0, #1
//    b.eq .LBB0_1
//  .LBB0_1:
//    orr x0, xzr, #0x1
//
// This pass should be run after register allocation.
//
// FIXME: This could also be extended to check the whole dominance subtree below
// the comparison if the compile time regression is acceptable.
//
//===----------------------------------------------------------------------===//

#include "AArch64.h"
#include "llvm/ADT/Optional.h"
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

  struct RegImm {
    MCPhysReg Reg;
    int32_t Imm;
    RegImm(MCPhysReg Reg, int32_t Imm) : Reg(Reg), Imm(Imm) {}
  };

  Optional<RegImm> knownRegValInBlock(MachineInstr &CondBr,
                                      MachineBasicBlock *MBB,
                                      MachineBasicBlock::iterator &FirstUse);
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

/// It's possible to determine the value of a register based on a dominating
/// condition.  To do so, this function checks to see if the basic block \p MBB
/// is the target to which a conditional branch \p CondBr jumps and whose
/// equality comparison is against a constant.  If so, return a known physical
/// register and constant value pair.  Otherwise, return None.
Optional<AArch64RedundantCopyElimination::RegImm>
AArch64RedundantCopyElimination::knownRegValInBlock(
    MachineInstr &CondBr, MachineBasicBlock *MBB,
    MachineBasicBlock::iterator &FirstUse) {
  unsigned Opc = CondBr.getOpcode();

  // Check if the current basic block is the target block to which the
  // CBZ/CBNZ instruction jumps when its Wt/Xt is zero.
  if (((Opc == AArch64::CBZW || Opc == AArch64::CBZX) &&
       MBB == CondBr.getOperand(1).getMBB()) ||
      ((Opc == AArch64::CBNZW || Opc == AArch64::CBNZX) &&
       MBB != CondBr.getOperand(1).getMBB())) {
    FirstUse = CondBr;
    return RegImm(CondBr.getOperand(0).getReg(), 0);
  }

  // Otherwise, must be a conditional branch.
  if (Opc != AArch64::Bcc)
    return None;

  // Must be an equality check (i.e., == or !=).
  AArch64CC::CondCode CC = (AArch64CC::CondCode)CondBr.getOperand(0).getImm();
  if (CC != AArch64CC::EQ && CC != AArch64CC::NE)
    return None;

  MachineBasicBlock *BrTarget = CondBr.getOperand(1).getMBB();
  if ((CC == AArch64CC::EQ && BrTarget != MBB) ||
      (CC == AArch64CC::NE && BrTarget == MBB))
    return None;

  // Stop if we get to the beginning of PredMBB.
  MachineBasicBlock *PredMBB = *MBB->pred_begin();
  assert(PredMBB == CondBr.getParent() &&
         "Conditional branch not in predecessor block!");
  if (CondBr == PredMBB->begin())
    return None;

  // Registers clobbered in PredMBB between CondBr instruction and current
  // instruction being checked in loop.
  ClobberedRegs.reset();

  // Find compare instruction that sets NZCV used by CondBr.
  MachineBasicBlock::reverse_iterator RIt = CondBr.getReverseIterator();
  for (MachineInstr &PredI : make_range(std::next(RIt), PredMBB->rend())) {

    // Track clobbered registers.
    trackRegDefs(PredI, ClobberedRegs, TRI);

    bool IsCMN = false;
    switch (PredI.getOpcode()) {
    default:
      break;

    // CMN is an alias for ADDS with a dead destination register.
    case AArch64::ADDSWri:
    case AArch64::ADDSXri:
      IsCMN = true;
      LLVM_FALLTHROUGH;
    // CMP is an alias for SUBS with a dead destination register.
    case AArch64::SUBSWri:
    case AArch64::SUBSXri: {
      // Sometimes the first operand is a FrameIndex. Bail if tht happens.
      if (!PredI.getOperand(1).isReg())
        return None;
      MCPhysReg SrcReg = PredI.getOperand(1).getReg();

      // Must not be a symbolic immediate.
      if (!PredI.getOperand(2).isImm())
        return None;

      // The src register must not be modified between the cmp and conditional
      // branch.  This includes a self-clobbering compare.
      if (ClobberedRegs[SrcReg])
        return None;

      // We've found the Cmp that sets NZCV.
      int32_t KnownImm = PredI.getOperand(2).getImm();
      int32_t Shift = PredI.getOperand(3).getImm();
      KnownImm <<= Shift;
      if (IsCMN)
        KnownImm = -KnownImm;
      FirstUse = PredI;
      return RegImm(SrcReg, KnownImm);
    }
    }

    // Bail if we see an instruction that defines NZCV that we don't handle.
    if (PredI.definesRegister(AArch64::NZCV))
      return None;
  }
  return None;
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

    Optional<RegImm> KnownRegImm = knownRegValInBlock(*Itr, MBB, FirstUse);
    if (KnownRegImm == None)
      continue;

    KnownRegs.push_back(*KnownRegImm);

    // Reset the clobber list, which is used by knownRegValInBlock.
    ClobberedRegs.reset();

    // Look backward in PredMBB for COPYs from the known reg to find other
    // registers that are known to be a constant value.
    for (auto PredI = Itr;; --PredI) {
      if (FirstUse == PredI)
        SeenFirstUse = true;

      if (PredI->isCopy()) {
        MCPhysReg CopyDstReg = PredI->getOperand(0).getReg();
        MCPhysReg CopySrcReg = PredI->getOperand(1).getReg();
        for (auto &KnownReg : KnownRegs) {
          if (ClobberedRegs[KnownReg.Reg])
            continue;
          // If we have X = COPY Y, and Y is known to be zero, then now X is
          // known to be zero.
          if (CopySrcReg == KnownReg.Reg && !ClobberedRegs[CopyDstReg]) {
            KnownRegs.push_back(RegImm(CopyDstReg, KnownReg.Imm));
            if (SeenFirstUse)
              FirstUse = PredI;
            break;
          }
          // If we have X = COPY Y, and X is known to be zero, then now Y is
          // known to be zero.
          if (CopyDstReg == KnownReg.Reg && !ClobberedRegs[CopySrcReg]) {
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

      trackRegDefs(*PredI, ClobberedRegs, TRI);
      // Stop if all of the known-zero regs have been clobbered.
      if (all_of(KnownRegs, [&](RegImm KnownReg) {
            return ClobberedRegs[KnownReg.Reg];
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
  // Remove redundant Copy instructions unless KnownReg is modified.
  for (MachineBasicBlock::iterator I = MBB->begin(), E = MBB->end(); I != E;) {
    MachineInstr *MI = &*I;
    ++I;
    bool RemovedMI = false;
    bool IsCopy = MI->isCopy();
    bool IsMoveImm = MI->isMoveImmediate();
    if (IsCopy || IsMoveImm) {
      MCPhysReg DefReg = MI->getOperand(0).getReg();
      MCPhysReg SrcReg = IsCopy ? MI->getOperand(1).getReg() : 0;
      int64_t SrcImm = IsMoveImm ? MI->getOperand(1).getImm() : 0;
      if (!MRI->isReserved(DefReg) &&
          ((IsCopy && (SrcReg == AArch64::XZR || SrcReg == AArch64::WZR)) ||
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
          }

          if (IsCopy)
            DEBUG(dbgs() << "Remove redundant Copy : " << *MI);
          else
            DEBUG(dbgs() << "Remove redundant Move : " << *MI);

          MI->eraseFromParent();
          Changed = true;
          LastChange = I;
          NumCopiesRemoved++;
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
  DEBUG(dbgs() << "Clearing kill flags.\n\tFirstUse: " << *FirstUse
               << "\tLastChange: " << *LastChange);
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

  // Resize the clobber register bitfield tracker.  We do this once per
  // function and then clear the bitfield each time we optimize a copy.
  ClobberedRegs.resize(TRI->getNumRegs());

  bool Changed = false;
  for (MachineBasicBlock &MBB : MF)
    Changed |= optimizeCopy(&MBB);
  return Changed;
}

FunctionPass *llvm::createAArch64RedundantCopyEliminationPass() {
  return new AArch64RedundantCopyElimination();
}
