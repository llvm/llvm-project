//=- AArch64RedundantCopyElimination.cpp - Remove useless copy for AArch64 -=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This pass removes unnecessary copies/moves in BBs based on a dominating
// condition.
//
// We handle four cases:
// 1. For BBs that are targets of CBZ/CBNZ instructions, we know the value of
//    the CBZ/CBNZ source register is zero on the taken/not-taken path. For
//    instance, the copy instruction in the code below can be removed because
//    the CBZW jumps to %bb.2 when w0 is zero.
//
//  %bb.1:
//    cbz w0, .LBB0_2
//  .LBB0_2:
//    mov w0, wzr  ; <-- redundant
//
// 2. If the flag setting instruction defines a register other than WZR/XZR, we
//    can remove a zero copy in some cases.
//
//  %bb.0:
//    subs w0, w1, w2
//    str w0, [x1]
//    b.ne .LBB0_2
//  %bb.1:
//    mov w0, wzr  ; <-- redundant
//    str w0, [x2]
//  .LBB0_2
//
// 3. Finally, if the flag setting instruction is a comparison against a
//    constant (i.e., ADDS[W|X]ri, SUBS[W|X]ri), we can remove a mov immediate
//    in some cases.
//
//  %bb.0:
//    subs xzr, x0, #1
//    b.eq .LBB0_1
//  .LBB0_1:
//    orr x0, xzr, #0x1  ; <-- redundant
//
// 4. If the flag setting instruction is a register comparison (i.e.,
// SUBS[W|X]rr
//    to WZR/XZR), we can remove a redundant copy between the compared registers
//    on the EQ path.
//
//  %bb.0:
//    subs xzr, x0, x1
//    b.eq .LBB0_1
//  .LBB0_1:
//    mov x0, x1  ; <-- redundant
//
// This pass should be run after register allocation.
//
// FIXME: This could also be extended to check the whole dominance subtree below
// the comparison if the compile time regression is acceptable.
//
// FIXME: Add support for handling CCMP instructions.
// FIXME: If the known register value is zero, we should be able to rewrite uses
//        to use WZR/XZR directly in some cases.
//===----------------------------------------------------------------------===//
#include "AArch64.h"
#include "AArch64InstrInfo.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/CodeGen/LiveRegUnits.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "aarch64-copyelim"

STATISTIC(NumCopiesRemoved, "Number of copies removed.");

namespace {
struct RegImm {
  MCPhysReg Reg;
  int32_t Imm;
  RegImm(MCPhysReg Reg, int32_t Imm) : Reg(Reg), Imm(Imm) {}
};

struct RegEqual {
  MCPhysReg Reg1;
  MCPhysReg Reg2;
  RegEqual(MCPhysReg Reg1, MCPhysReg Reg2) : Reg1(Reg1), Reg2(Reg2) {}
};

class AArch64RedundantCopyEliminationImpl {
public:
  bool run(MachineFunction &MF);

private:
  const MachineRegisterInfo *MRI;
  const TargetRegisterInfo *TRI;

  // DomBBClobberedRegs is used when computing known values in the dominating
  // BB.
  LiveRegUnits DomBBClobberedRegs, DomBBUsedRegs;

  // OptBBClobberedRegs is used when optimizing away redundant copies/moves.
  LiveRegUnits OptBBClobberedRegs, OptBBUsedRegs;

  bool knownRegValInBlock(MachineInstr &CondBr, MachineBasicBlock *MBB,
                          SmallVectorImpl<RegImm> &KnownRegs,
                          SmallVectorImpl<RegEqual> &KnownEqualRegs,
                          MachineBasicBlock::iterator &FirstUse);
  bool optimizeBlock(MachineBasicBlock *MBB);
};

class AArch64RedundantCopyEliminationLegacy : public MachineFunctionPass {
public:
  static char ID;
  AArch64RedundantCopyEliminationLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().setNoVRegs();
  }
  StringRef getPassName() const override {
    return "AArch64 Redundant Copy Elimination";
  }
};
char AArch64RedundantCopyEliminationLegacy::ID = 0;
} // end anonymous namespace

INITIALIZE_PASS(AArch64RedundantCopyEliminationLegacy, "aarch64-copyelim",
                "AArch64 redundant copy elimination pass", false, false)

static bool isCompareReg(const MachineInstr &MI, MCPhysReg &Rn, MCPhysReg &Rm) {
  MCPhysReg DstReg;
  switch (MI.getOpcode()) {
  case AArch64::SUBSWrr:
  case AArch64::SUBSXrr:
    DstReg = MI.getOperand(0).getReg();
    if (DstReg != AArch64::WZR && DstReg != AArch64::XZR)
      return false;
    Rn = MI.getOperand(1).getReg();
    Rm = MI.getOperand(2).getReg();
    return true;
  default:
    return false;
  }
}

static bool isKnownZeroSource(Register SrcReg, ArrayRef<RegImm> KnownRegs) {
  for (const RegImm &KnownReg : KnownRegs) {
    if (KnownReg.Imm == 0 && KnownReg.Reg == SrcReg)
      return true;
  }
  return false;
}

static bool isRedundantEqualRegAssign(const MachineInstr &MI,
                                      ArrayRef<RegEqual> KnownEqualRegs) {
  if (MI.getNumOperands() < 2 || !MI.getOperand(0).isReg() ||
      !MI.getOperand(1).isReg())
    return false;
  MCPhysReg DefReg = MI.getOperand(0).getReg();
  MCPhysReg SrcReg = MI.getOperand(1).getReg();
  for (const RegEqual &Eq : KnownEqualRegs) {
    if ((DefReg == Eq.Reg1 && SrcReg == Eq.Reg2) ||
        (DefReg == Eq.Reg2 && SrcReg == Eq.Reg1))
      return true;
  }
  return false;
}

static void propagateEqualRegThroughCopy(MCPhysReg CopyDstReg,
                                         MCPhysReg CopySrcReg,
                                         ArrayRef<RegEqual> KnownEqualRegs,
                                         SmallVectorImpl<RegEqual> &Out) {
  Out.push_back(RegEqual(CopyDstReg, CopySrcReg));
  for (const RegEqual &Eq : KnownEqualRegs) {
    if (CopySrcReg == Eq.Reg1)
      Out.push_back(RegEqual(CopyDstReg, Eq.Reg2));
    else if (CopySrcReg == Eq.Reg2)
      Out.push_back(RegEqual(CopyDstReg, Eq.Reg1));
    else if (CopyDstReg == Eq.Reg1)
      Out.push_back(RegEqual(CopySrcReg, Eq.Reg2));
    else if (CopyDstReg == Eq.Reg2)
      Out.push_back(RegEqual(CopySrcReg, Eq.Reg1));
  }
}

static bool shouldStopPredScan(ArrayRef<RegImm> KnownRegs,
                               ArrayRef<RegEqual> KnownEqualRegs,
                               const LiveRegUnits &Clobbered) {
  bool AnyKnownReg = any_of(
      KnownRegs, [&](const RegImm &K) { return Clobbered.available(K.Reg); });
  bool AnyKnownEqual = any_of(KnownEqualRegs, [&](const RegEqual &Eq) {
    return Clobbered.available(Eq.Reg1) && Clobbered.available(Eq.Reg2);
  });
  return !AnyKnownReg && !AnyKnownEqual;
}

/// It's possible to determine the value of a register based on a dominating
/// condition.  To do so, this function checks to see if the basic block \p MBB
/// is the target of a conditional branch \p CondBr with an equality comparison.
/// If the branch is a CBZ/CBNZ, we know the value of its source operand is zero
/// in \p MBB for some cases.  Otherwise, we find and inspect the NZCV setting
/// instruction (e.g., SUBS, ADDS).  If this instruction defines a register
/// other than WZR/XZR, we know the value of the destination register is zero in
/// \p MMB for some cases.  In addition, if the NZCV setting instruction is
/// comparing against a constant we know the other source register is equal to
/// the constant in \p MBB for some cases.  If the NZCV setting instruction is a
/// register comparison, we know the compared registers are equal in \p MBB for
/// some cases.  If we find any known values, push them onto the KnownRegs or
/// KnownEqualRegs vectors and return true.  Otherwise, return false if no known
/// values were found.
bool AArch64RedundantCopyEliminationImpl::knownRegValInBlock(
    MachineInstr &CondBr, MachineBasicBlock *MBB,
    SmallVectorImpl<RegImm> &KnownRegs,
    SmallVectorImpl<RegEqual> &KnownEqualRegs,
    MachineBasicBlock::iterator &FirstUse) {
  unsigned Opc = CondBr.getOpcode();

  // Check if the current basic block is the target block to which the
  // CBZ/CBNZ instruction jumps when its Wt/Xt is zero.
  if (((Opc == AArch64::CBZW || Opc == AArch64::CBZX) &&
       MBB == CondBr.getOperand(1).getMBB()) ||
      ((Opc == AArch64::CBNZW || Opc == AArch64::CBNZX) &&
       MBB != CondBr.getOperand(1).getMBB())) {
    FirstUse = CondBr;
    KnownRegs.push_back(RegImm(CondBr.getOperand(0).getReg(), 0));
    return true;
  }

  // Otherwise, must be a conditional branch.
  if (Opc != AArch64::Bcc)
    return false;

  // Must be an equality check (i.e., == or !=).
  AArch64CC::CondCode CC = (AArch64CC::CondCode)CondBr.getOperand(0).getImm();
  if (CC != AArch64CC::EQ && CC != AArch64CC::NE)
    return false;

  MachineBasicBlock *BrTarget = CondBr.getOperand(1).getMBB();
  if ((CC == AArch64CC::EQ && BrTarget != MBB) ||
      (CC == AArch64CC::NE && BrTarget == MBB))
    return false;

  // Stop if we get to the beginning of PredMBB.
  MachineBasicBlock *PredMBB = *MBB->pred_begin();
  assert(PredMBB == CondBr.getParent() &&
         "Conditional branch not in predecessor block!");
  if (CondBr == PredMBB->begin())
    return false;

  // Registers clobbered in PredMBB between CondBr instruction and current
  // instruction being checked in loop.
  DomBBClobberedRegs.clear();
  DomBBUsedRegs.clear();

  // Find compare instruction that sets NZCV used by CondBr.
  MachineBasicBlock::reverse_iterator RIt = CondBr.getReverseIterator();
  for (MachineInstr &PredI : make_range(std::next(RIt), PredMBB->rend())) {
    MCPhysReg Rn;
    MCPhysReg Rm;
    if (isCompareReg(PredI, Rn, Rm)) {
      // If both source registers of the compare are not modified between the
      // compare and conditional branch we know they are equal.
      if (DomBBClobberedRegs.available(Rn) &&
          DomBBClobberedRegs.available(Rm)) {
        FirstUse = PredI;
        KnownEqualRegs.push_back(RegEqual(Rn, Rm));
        return true;
      }
      return false;
    }

    bool IsCMN = false;
    switch (PredI.getOpcode()) {
    default:
      break;

    // CMN is an alias for ADDS with a dead destination register.
    case AArch64::ADDSWri:
    case AArch64::ADDSXri:
      IsCMN = true;
      [[fallthrough]];
    // CMP is an alias for SUBS with a dead destination register.
    case AArch64::SUBSWri:
    case AArch64::SUBSXri: {
      // Sometimes the first operand is a FrameIndex. Bail if tht happens.
      if (!PredI.getOperand(1).isReg())
        return false;
      MCPhysReg DstReg = PredI.getOperand(0).getReg();
      MCPhysReg SrcReg = PredI.getOperand(1).getReg();

      bool Res = false;
      // If we're comparing against a non-symbolic immediate and the source
      // register of the compare is not modified (including a self-clobbering
      // compare) between the compare and conditional branch we known the value
      // of the 1st source operand.
      if (PredI.getOperand(2).isImm() && DomBBClobberedRegs.available(SrcReg) &&
          SrcReg != DstReg) {
        // We've found the instruction that sets NZCV.
        int32_t KnownImm = PredI.getOperand(2).getImm();
        int32_t Shift = PredI.getOperand(3).getImm();
        KnownImm <<= Shift;
        if (IsCMN)
          KnownImm = -KnownImm;
        FirstUse = PredI;
        KnownRegs.push_back(RegImm(SrcReg, KnownImm));
        Res = true;
      }

      // If this instructions defines something other than WZR/XZR, we know it's
      // result is zero in some cases.
      if (DstReg == AArch64::WZR || DstReg == AArch64::XZR)
        return Res;

      // The destination register must not be modified between the NZCV setting
      // instruction and the conditional branch.
      if (!DomBBClobberedRegs.available(DstReg))
        return Res;

      FirstUse = PredI;
      KnownRegs.push_back(RegImm(DstReg, 0));
      return true;
    }

    // Look for NZCV setting instructions that define something other than
    // WZR/XZR.
    case AArch64::ADCSWr:
    case AArch64::ADCSXr:
    case AArch64::ADDSWrr:
    case AArch64::ADDSWrs:
    case AArch64::ADDSWrx:
    case AArch64::ADDSXrr:
    case AArch64::ADDSXrs:
    case AArch64::ADDSXrx:
    case AArch64::ADDSXrx64:
    case AArch64::ANDSWri:
    case AArch64::ANDSWrr:
    case AArch64::ANDSWrs:
    case AArch64::ANDSXri:
    case AArch64::ANDSXrr:
    case AArch64::ANDSXrs:
    case AArch64::BICSWrr:
    case AArch64::BICSWrs:
    case AArch64::BICSXrs:
    case AArch64::BICSXrr:
    case AArch64::SBCSWr:
    case AArch64::SBCSXr:
    case AArch64::SUBSWrr:
    case AArch64::SUBSWrs:
    case AArch64::SUBSWrx:
    case AArch64::SUBSXrr:
    case AArch64::SUBSXrs:
    case AArch64::SUBSXrx:
    case AArch64::SUBSXrx64: {
      MCPhysReg DstReg = PredI.getOperand(0).getReg();
      if (DstReg == AArch64::WZR || DstReg == AArch64::XZR)
        return false;

      // The destination register of the NZCV setting instruction must not be
      // modified before the conditional branch.
      if (!DomBBClobberedRegs.available(DstReg))
        return false;

      // We've found the instruction that sets NZCV whose DstReg == 0.
      FirstUse = PredI;
      KnownRegs.push_back(RegImm(DstReg, 0));
      return true;
    }
    }

    // Bail if we see an instruction that defines NZCV that we don't handle.
    if (PredI.definesRegister(AArch64::NZCV, /*TRI=*/nullptr))
      return false;

    // Track clobbered and used registers.
    LiveRegUnits::accumulateUsedDefed(PredI, DomBBClobberedRegs, DomBBUsedRegs,
                                      TRI);
  }
  return false;
}

bool AArch64RedundantCopyEliminationImpl::optimizeBlock(
    MachineBasicBlock *MBB) {
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
  SmallVector<RegEqual, 2> KnownEqualRegs;

  MachineBasicBlock::iterator Itr = std::next(CondBr);
  do {
    --Itr;

    if (!knownRegValInBlock(*Itr, MBB, KnownRegs, KnownEqualRegs, FirstUse))
      continue;

    // Reset the clobbered and used register units.
    OptBBClobberedRegs.clear();
    OptBBUsedRegs.clear();

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
        SmallVector<RegEqual, 2> NewEqual;
        if (OptBBClobberedRegs.available(CopyDstReg) &&
            OptBBClobberedRegs.available(CopySrcReg)) {
          propagateEqualRegThroughCopy(CopyDstReg, CopySrcReg, KnownEqualRegs,
                                       NewEqual);
        }
        if (!NewEqual.empty()) {
          KnownEqualRegs.append(NewEqual.begin(), NewEqual.end());
          if (SeenFirstUse)
            FirstUse = PredI;
        }
      }

      // Stop if we get to the beginning of PredMBB.
      if (PredI == PredMBB->begin())
        break;

      LiveRegUnits::accumulateUsedDefed(*PredI, OptBBClobberedRegs,
                                        OptBBUsedRegs, TRI);
      // Stop if all of the known regs have been clobbered.
      if (shouldStopPredScan(KnownRegs, KnownEqualRegs, OptBBClobberedRegs))
        break;
    }
    break;

  } while (Itr != PredMBB->begin() && Itr->isTerminator());

  // We've not found registers with a known value, time to bail out.
  if (KnownRegs.empty() && KnownEqualRegs.empty())
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
    Register SrcReg = IsCopy ? MI->getOperand(1).getReg() : Register();
    int64_t SrcImm = IsMoveImm ? MI->getOperand(1).getImm() : 0;
    if (IsCopy || IsMoveImm) {
      Register DefReg = MI->getOperand(0).getReg();
      if (!MRI->isReserved(DefReg)) {
        if (IsCopy && !KnownEqualRegs.empty() &&
            isRedundantEqualRegAssign(*MI, KnownEqualRegs)) {
          // Don't remove an instruction that has other live defs.
          if (any_of(MI->operands(), [DefReg](MachineOperand &O) {
                return O.isReg() && O.isDef() && !O.isDead() &&
                       O.getReg() != DefReg;
              }))
            continue;

          for (const RegEqual &Eq : KnownEqualRegs) {
            if (MI->getOperand(0).getReg() == Eq.Reg1 ||
                MI->getOperand(0).getReg() == Eq.Reg2 ||
                MI->getOperand(1).getReg() == Eq.Reg1 ||
                MI->getOperand(1).getReg() == Eq.Reg2) {
              UsedKnownRegs.insert(Eq.Reg1);
              UsedKnownRegs.insert(Eq.Reg2);
            }
          }
          LLVM_DEBUG(dbgs() << "Remove redundant equal-reg copy: " << *MI);
          MI->eraseFromParent();
          Changed = true;
          LastChange = I;
          NumCopiesRemoved++;
          RemovedMI = true;
        }

        if (!RemovedMI && IsCopy) {
          bool SrcIsZero = (SrcReg == AArch64::XZR || SrcReg == AArch64::WZR ||
                            isKnownZeroSource(SrcReg, KnownRegs));
          if (SrcIsZero) {
            for (RegImm &KnownReg : KnownRegs) {
              if (KnownReg.Imm != 0)
                continue;
              if (KnownReg.Reg != DefReg &&
                  !TRI->isSuperRegister(DefReg, KnownReg.Reg))
                continue;
              LLVM_DEBUG(dbgs() << "Remove redundant Copy : " << *MI);
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

        if (!RemovedMI && IsMoveImm) {
          for (RegImm &KnownReg : KnownRegs) {
            if (KnownReg.Reg != DefReg &&
                !TRI->isSuperRegister(DefReg, KnownReg.Reg))
              continue;

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

            LLVM_DEBUG(dbgs() << "Remove redundant Move : " << *MI);

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
    }

    // Skip to the next instruction if we removed the COPY/MovImm.
    if (RemovedMI)
      continue;

    // Remove any regs the MI clobbers from the KnownRegs set.
    for (unsigned RI = 0; RI < KnownRegs.size();)
      if (MI->modifiesRegister(KnownRegs[RI].Reg, TRI)) {
        std::swap(KnownRegs[RI], KnownRegs[KnownRegs.size() - 1]);
        KnownRegs.pop_back();
        // Don't increment RI since we need to now check the swapped-in
        // KnownRegs[RI].
      } else {
        ++RI;
      }

    // Remove any equal-reg pairs the MI clobbers from the KnownEqualRegs set.
    for (unsigned RI = 0; RI < KnownEqualRegs.size();) {
      if (MI->modifiesRegister(KnownEqualRegs[RI].Reg1, TRI) ||
          MI->modifiesRegister(KnownEqualRegs[RI].Reg2, TRI)) {
        std::swap(KnownEqualRegs[RI],
                  KnownEqualRegs[KnownEqualRegs.size() - 1]);
        KnownEqualRegs.pop_back();
      } else {
        ++RI;
      }
    }

    // Continue until the KnownRegs and KnownEqualRegs sets are empty.
    if (KnownRegs.empty() && KnownEqualRegs.empty())
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
  LLVM_DEBUG(dbgs() << "Clearing kill flags.\n\tFirstUse: " << *FirstUse
                    << "\tLastChange: ";
             if (LastChange == MBB->end()) dbgs() << "<end>\n";
             else dbgs() << *LastChange);
  for (MachineInstr &MMI : make_range(FirstUse, PredMBB->end()))
    MMI.clearKillInfo();
  for (MachineInstr &MMI : make_range(MBB->begin(), LastChange))
    MMI.clearKillInfo();

  return true;
}

bool AArch64RedundantCopyEliminationImpl::run(MachineFunction &MF) {
  TRI = MF.getSubtarget().getRegisterInfo();
  MRI = &MF.getRegInfo();
  const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();

  // Resize the clobbered and used register unit trackers.  We do this once per
  // function.
  DomBBClobberedRegs.init(*TRI);
  DomBBUsedRegs.init(*TRI);
  OptBBClobberedRegs.init(*TRI);
  OptBBUsedRegs.init(*TRI);

  bool Changed = false;
  for (MachineBasicBlock &MBB : MF) {
    Changed |= optimizeTerminators(&MBB, TII);
    Changed |= optimizeBlock(&MBB);
  }
  return Changed;
}

bool AArch64RedundantCopyEliminationLegacy::runOnMachineFunction(
    MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;
  return AArch64RedundantCopyEliminationImpl().run(MF);
}

PreservedAnalyses
AArch64RedundantCopyEliminationPass::run(MachineFunction &MF,
                                         MachineFunctionAnalysisManager &MFAM) {
  const bool Changed = AArch64RedundantCopyEliminationImpl().run(MF);
  if (!Changed)
    return PreservedAnalyses::all();
  PreservedAnalyses PA = getMachineFunctionPassPreservedAnalyses();
  PA.preserveSet<CFGAnalyses>();
  return PA;
}

FunctionPass *llvm::createAArch64RedundantCopyEliminationPass() {
  return new AArch64RedundantCopyEliminationLegacy();
}
