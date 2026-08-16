//=- ARMRedundantCopyElimination.cpp - Remove useless copy for ARM -=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This pass removes unnecessary copies/moves in BBs based on a dominating
// condition.
//
// We handle four cases:
// 1. For BBs that are targets of tCBZ/tCBNZ instructions, we know the value of
//    the source register is zero on the taken/not-taken path. For instance, the
//    move instruction in the code below can be removed because the tCBZ jumps
//    to %bb.2 when r0 is zero.
//
//  %bb.1:
//    tCBZ $r0, %bb.2
//  %bb.2:
//    $r0 = t2MOVi 0  ; <-- redundant
//
// 2. If the CPSR-setting instruction defines a GPR, we can remove a redundant
//    zero move in some cases.
//
//  %bb.0:
//    $r0 = t2SUBrr $r1, $r2, def $cpsr
//    t2Bcc %bb.2, ne, killed $cpsr
//  %bb.1:
//    $r0 = t2MOVi 0  ; <-- redundant
//  %bb.2:
//
// 3. If the CPSR-setting instruction is a comparison against a constant, we
//    can remove a redundant move immediate in some cases.
//
//  %bb.0:
//    t2CMPri $r0, 1, implicit-def $cpsr
//    t2Bcc %bb.1, eq, killed $cpsr
//  %bb.1:
//    $r0 = t2MOVi 1  ; <-- redundant
//
// 4. If the CPSR-setting instruction is a register comparison, we can remove a
//    redundant copy/move between the compared registers on the EQ path.
//
//  %bb.0:
//    t2CMPrr $r0, $r1, implicit-def $cpsr
//    t2Bcc %bb.1, eq, killed $cpsr
//  %bb.1:
//    $r0 = t2MOVr $r1  ; <-- redundant
//
// This pass should be run after register allocation.
//
// FIXME: This could also be extended to check the whole dominance subtree below
// the comparison if the compile time regression is acceptable.
//
//===----------------------------------------------------------------------===//
#include "ARM.h"
#include "ARMBaseInstrInfo.h"
#include "Utils/ARMBaseInfo.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/CodeGen/LiveRegUnits.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "arm-copyelim"

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

class ARMRedundantCopyEliminationImpl {
public:
  bool run(MachineFunction &MF);

private:
  const MachineRegisterInfo *MRI;
  const TargetRegisterInfo *TRI;
  const TargetInstrInfo *TII;

  // DomBBClobberedRegs is used when computing known values in the dominating
  // BB.
  LiveRegUnits DomBBClobberedRegs, DomBBUsedRegs;

  // OptBBClobberedRegs is used when optimizing away redundant copies/moves.
  LiveRegUnits OptBBClobberedRegs, OptBBUsedRegs;

  bool getCBZKnownRegs(MachineInstr &CondBr, MachineBasicBlock *MBB,
                       SmallVectorImpl<RegImm> &KnownRegs,
                       MachineBasicBlock::iterator &FirstUse);
  bool knownRegValInBlock(MachineInstr &CondBr, MachineBasicBlock *MBB,
                          SmallVectorImpl<RegImm> &KnownRegs,
                          SmallVectorImpl<RegEqual> &KnownEqualRegs,
                          MachineBasicBlock::iterator &FirstUse);
  bool optimizeBlock(MachineBasicBlock *MBB);
};

class ARMRedundantCopyElimination : public MachineFunctionPass {
public:
  static char ID;
  ARMRedundantCopyElimination() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;
  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().setNoVRegs();
  }
  StringRef getPassName() const override {
    return "ARM Redundant Copy Elimination";
  }
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

char ARMRedundantCopyElimination::ID = 0;

} // end anonymous namespace

INITIALIZE_PASS(ARMRedundantCopyElimination, "arm-copyelim",
                "ARM redundant copy elimination pass", false, false)

static bool isCompareImm(const MachineInstr &MI, MCPhysReg &SrcReg,
                         int32_t &Imm) {
  switch (MI.getOpcode()) {
  case ARM::CMPri:
  case ARM::t2CMPri:
  case ARM::tCMPi8:
    SrcReg = MI.getOperand(0).getReg();
    Imm = MI.getOperand(1).getImm();
    return true;
  case ARM::CMNri:
  case ARM::t2CMNri:
    SrcReg = MI.getOperand(0).getReg();
    Imm = -MI.getOperand(1).getImm();
    return true;
  default:
    return false;
  }
}

static bool isCompareReg(const MachineInstr &MI, MCPhysReg &Rn, MCPhysReg &Rm) {
  switch (MI.getOpcode()) {
  case ARM::CMPrr:
  case ARM::t2CMPrr:
  case ARM::tCMPr:
  case ARM::tCMPhir:
    Rn = MI.getOperand(0).getReg();
    Rm = MI.getOperand(1).getReg();
    return true;
  default:
    return false;
  }
}

static bool isFlagSettingAluWithDest(const MachineInstr &MI) {
  if (!MI.definesRegister(ARM::CPSR, /*TRI=*/nullptr))
    return false;
  if (MI.getNumOperands() == 0 || !MI.getOperand(0).isReg())
    return false;
  return MI.getOperand(0).isDef();
}

static bool isKnownZeroSource(const MachineInstr &MI,
                              ArrayRef<RegImm> KnownRegs) {
  if (MI.getNumOperands() < 2 || !MI.getOperand(1).isReg())
    return false;
  MCPhysReg SrcReg = MI.getOperand(1).getReg();
  for (const RegImm &KnownReg : KnownRegs) {
    if (KnownReg.Imm == 0 && KnownReg.Reg == SrcReg)
      return true;
  }
  return false;
}

static bool getMoveImmediateValue(const MachineInstr &MI, int64_t &ImmVal) {
  for (const MachineOperand &MO : MI.operands()) {
    if (MO.isImm()) {
      ImmVal = MO.getImm();
      return true;
    }
  }
  return false;
}

static bool isRedundantZeroAssign(const MachineInstr &MI, MCPhysReg DefReg,
                                  ArrayRef<RegImm> KnownRegs) {
  for (const RegImm &KnownReg : KnownRegs) {
    if (KnownReg.Imm != 0)
      continue;
    if (KnownReg.Reg != DefReg)
      continue;
    int64_t ImmVal;
    if (MI.isMoveImmediate() && getMoveImmediateValue(MI, ImmVal) &&
        ImmVal == 0)
      return true;
    if (MI.isCopy() && isKnownZeroSource(MI, KnownRegs))
      return true;
    switch (MI.getOpcode()) {
    case ARM::MOVr:
    case ARM::t2MOVr:
    case ARM::tMOVr:
      if (isKnownZeroSource(MI, KnownRegs))
        return true;
      break;
    default:
      break;
    }
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

/// It's possible to determine the value of a register based on a dominating
/// condition.  To do so, this function checks to see if the basic block \p MBB
/// is the target of a conditional branch \p CondBr with an equality comparison.
/// If the branch is a tCBZ/tCBNZ, we know the value of its source operand is
/// zero in \p MBB for some cases.
bool ARMRedundantCopyEliminationImpl::getCBZKnownRegs(
    MachineInstr &CondBr, MachineBasicBlock *MBB,
    SmallVectorImpl<RegImm> &KnownRegs, MachineBasicBlock::iterator &FirstUse) {
  unsigned Opc = CondBr.getOpcode();
  if (Opc != ARM::tCBZ && Opc != ARM::tCBNZ)
    return false;

  // Check if the current basic block is the target block to which the
  // tCBZ/tCBNZ instruction jumps when its register is zero.
  MCPhysReg Reg = CondBr.getOperand(0).getReg();
  MachineBasicBlock *Target = CondBr.getOperand(1).getMBB();
  if ((Opc == ARM::tCBZ && Target == MBB) ||
      (Opc == ARM::tCBNZ && Target != MBB)) {
    FirstUse = CondBr;
    KnownRegs.push_back(RegImm(Reg, 0));
    return true;
  }
  return false;
}

/// It's possible to determine the value of a register based on a dominating
/// condition.  To do so, this function checks to see if the basic block \p MBB
/// is the target of a conditional branch \p CondBr with an equality comparison.
/// If the branch is a tCBZ/tCBNZ, getCBZKnownRegs handles that case. Otherwise,
/// we find and inspect the CPSR-setting instruction (e.g., CMP, CMN, SUBS).  If
/// this instruction defines a GPR, we know the value of the destination
/// register is zero in \p MBB for some cases.  In addition, if the CPSR-setting
/// instruction is comparing against a constant we know the source register is
/// equal to the constant in \p MBB for some cases.  If the CPSR-setting
/// instruction is a register comparison, we know the compared registers are
/// equal in \p MBB for some cases.
///
/// If we find any known constant values, push a physical register and constant
/// value pair onto the KnownRegs vector. If we find any known equal registers,
/// push them onto KnownEqualRegs. Return true if we find any known values;
/// otherwise, return false.
bool ARMRedundantCopyEliminationImpl::knownRegValInBlock(
    MachineInstr &CondBr, MachineBasicBlock *MBB,
    SmallVectorImpl<RegImm> &KnownRegs,
    SmallVectorImpl<RegEqual> &KnownEqualRegs,
    MachineBasicBlock::iterator &FirstUse) {
  // Otherwise, must be a conditional branch.
  if (!isCondBranchOpcode(CondBr.getOpcode()))
    return false;

  // Must be an equality check (i.e., == or !=).
  ARMCC::CondCodes CC = (ARMCC::CondCodes)CondBr.getOperand(1).getImm();
  if (CC != ARMCC::EQ && CC != ARMCC::NE)
    return false;

  MachineBasicBlock *BrTarget = CondBr.getOperand(0).getMBB();
  if ((CC == ARMCC::EQ && BrTarget != MBB) ||
      (CC == ARMCC::NE && BrTarget == MBB))
    return false;

  // Stop if we get to the beginning of PredMBB.
  MachineBasicBlock *PredMBB = *MBB->pred_begin();
  assert(PredMBB == CondBr.getParent() &&
         "Conditional branch not in predecessor block!");
  // Stop if we get to the beginning of PredMBB.
  if (CondBr == PredMBB->begin())
    return false;

  // Registers clobbered in PredMBB between CondBr instruction and current
  // instruction being checked in loop.
  DomBBClobberedRegs.clear();
  DomBBUsedRegs.clear();

  // Find compare instruction that sets CPSR used by CondBr.
  MachineBasicBlock::reverse_iterator RIt = CondBr.getReverseIterator();
  for (MachineInstr &PredI : make_range(std::next(RIt), PredMBB->rend())) {
    MCPhysReg SrcReg;
    int32_t KnownImm;
    if (isCompareImm(PredI, SrcReg, KnownImm)) {
      Register PredReg;
      if (getInstrPredicate(PredI, PredReg) != ARMCC::AL)
        return false;
      bool Res = false;
      // If we're comparing against a non-symbolic immediate and the source
      // register of the compare is not modified (including a self-clobbering
      // compare) between the compare and conditional branch we know the value
      // of the 1st source operand.
      if (DomBBClobberedRegs.available(SrcReg)) {
        FirstUse = PredI;
        KnownRegs.push_back(RegImm(SrcReg, KnownImm));
        Res = true;
      }
      return Res;
    }

    MCPhysReg Rn;
    MCPhysReg Rm;
    if (isCompareReg(PredI, Rn, Rm)) {
      Register PredReg;
      if (getInstrPredicate(PredI, PredReg) != ARMCC::AL)
        return false;
      // If we're comparing two registers and neither are modified between the
      // compare and the conditional branch we know they are equal on the EQ
      // path.
      if (DomBBClobberedRegs.available(Rn) &&
          DomBBClobberedRegs.available(Rm)) {
        FirstUse = PredI;
        KnownEqualRegs.push_back(RegEqual(Rn, Rm));
        return true;
      }
      return false;
    }

    // Look for CPSR-setting instructions that define a GPR.
    if (isFlagSettingAluWithDest(PredI)) {
      Register PredReg;
      if (getInstrPredicate(PredI, PredReg) != ARMCC::AL)
        return false;
      MCPhysReg DstReg = PredI.getOperand(0).getReg();

      // The destination register must not be modified between the CPSR setting
      // instruction and the conditional branch.
      if (!DomBBClobberedRegs.available(DstReg))
        return false;

      // We've found the instruction that sets CPSR whose DstReg == 0.
      FirstUse = PredI;
      KnownRegs.push_back(RegImm(DstReg, 0));
      return true;
    }

    // Bail if we see an instruction that defines CPSR that we don't handle.
    if (PredI.definesRegister(ARM::CPSR, /*TRI=*/nullptr))
      return false;

    // Track clobbered and used registers.
    LiveRegUnits::accumulateUsedDefed(PredI, DomBBClobberedRegs, DomBBUsedRegs,
                                      TRI);
  }
  return false;
}

bool ARMRedundantCopyEliminationImpl::optimizeBlock(MachineBasicBlock *MBB) {
  // Check if the current basic block has a single predecessor.
  if (MBB->pred_size() != 1)
    return false;

  // Check if the predecessor has two successors, implying the block ends in a
  // conditional branch.
  MachineBasicBlock *PredMBB = *MBB->pred_begin();
  if (PredMBB->succ_size() != 2)
    return false;

  // Keep track of the earliest point in the PredMBB block where kill markers
  // need to be removed if a COPY is removed.
  MachineBasicBlock::iterator FirstUse;
  // After calling getCBZKnownRegs or knownRegValInBlock, FirstUse will either
  // point to a tCBZ/tCBNZ or a compare (e.g., CMP).  In the latter case, we
  // must take care when updating FirstUse when scanning for COPY instructions.
  // In particular, if there's a COPY in between the compare and branch the COPY
  // should not update FirstUse.
  bool SeenFirstUse = false;
  // Registers that contain a known value at the start of MBB.
  SmallVector<RegImm, 4> KnownRegs;
  SmallVector<RegEqual, 2> KnownEqualRegs;
  MachineBasicBlock::iterator CondBrIt = PredMBB->end();
  for (MachineInstr &Term : PredMBB->terminators()) {
    if (Term.getOpcode() == ARM::tCBZ || Term.getOpcode() == ARM::tCBNZ) {
      CondBrIt = Term.getIterator();
      if (getCBZKnownRegs(Term, MBB, KnownRegs, FirstUse))
        break;
      KnownRegs.clear();
      CondBrIt = PredMBB->end();
    }
  }

  if (KnownRegs.empty() && KnownEqualRegs.empty()) {
    MachineBasicBlock *TBB = nullptr, *FBB = nullptr;
    SmallVector<MachineOperand, 4> Cond;
    if (TII->analyzeBranch(*PredMBB, TBB, FBB, Cond, /*AllowModify*/ false) ||
        Cond.size() != 2)
      return false;

    // Must be a conditional branch.
    for (MachineInstr &MI : reverse(PredMBB->instrs())) {
      if (!isCondBranchOpcode(MI.getOpcode()))
        continue;
      CondBrIt = MI.getIterator();
      if (!knownRegValInBlock(MI, MBB, KnownRegs, KnownEqualRegs, FirstUse))
        return false;
      break;
    }
    if (CondBrIt == PredMBB->end())
      return false;
  }

  // We've not found registers with a known value, time to bail out.
  if (KnownRegs.empty() && KnownEqualRegs.empty())
    return false;

  // Reset the clobbered and used register units.
  OptBBClobberedRegs.clear();
  OptBBUsedRegs.clear();

  // Look backward in PredMBB for COPYs from the known reg to find other
  // registers that are known to be a constant value.
  for (auto PredI = CondBrIt;; --PredI) {
    if (FirstUse == PredI)
      SeenFirstUse = true;

    if (PredI->isCopy()) {
      MCPhysReg CopyDstReg = PredI->getOperand(0).getReg();
      MCPhysReg CopySrcReg = PredI->getOperand(1).getReg();
      for (const RegImm &KnownReg : KnownRegs) {
        if (!OptBBClobberedRegs.available(KnownReg.Reg))
          continue;
        // If a known register is copied, the copy destination also has the
        // known value.
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

    LiveRegUnits::accumulateUsedDefed(*PredI, OptBBClobberedRegs, OptBBUsedRegs,
                                      TRI);
    // Stop if all of the known regs have been clobbered.
    if (shouldStopPredScan(KnownRegs, KnownEqualRegs, OptBBClobberedRegs))
      break;
  }

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
    bool IsRegMov = false;
    switch (MI->getOpcode()) {
    case ARM::MOVr:
    case ARM::t2MOVr:
    case ARM::tMOVr:
      IsRegMov = true;
      break;
    }
    if (IsCopy || IsMoveImm || IsRegMov) {
      Register DefReg = MI->getOperand(0).getReg();
      if (!MRI->isReserved(DefReg)) {
        if ((IsCopy || IsRegMov) && !KnownEqualRegs.empty() &&
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
          LLVM_DEBUG(dbgs() << "Remove redundant equal-reg move: " << *MI);
          MI->eraseFromParent();
          Changed = true;
          LastChange = I;
          ++NumCopiesRemoved;
          RemovedMI = true;
        }

        for (RegImm &KnownReg : KnownRegs) {
          if (RemovedMI)
            break;
          if (KnownReg.Reg != DefReg)
            continue;

          bool ShouldRemove = false;
          if (KnownReg.Imm == 0 &&
              isRedundantZeroAssign(*MI, DefReg, KnownRegs))
            ShouldRemove = true;
          else if (IsMoveImm) {
            // For a move immediate, the known immediate must match the source
            // immediate.
            int64_t ImmVal;
            if (getMoveImmediateValue(*MI, ImmVal) && KnownReg.Imm == ImmVal)
              ShouldRemove = true;
          }

          if (!ShouldRemove)
            continue;

          // Don't remove an instruction that has other live defs.
          MCPhysReg CmpReg = KnownReg.Reg;
          if (any_of(MI->operands(), [CmpReg](MachineOperand &O) {
                return O.isReg() && O.isDef() && !O.isDead() &&
                       O.getReg() != CmpReg;
              }))
            continue;

          if (IsCopy)
            LLVM_DEBUG(dbgs() << "Remove redundant Copy : " << *MI);
          else
            LLVM_DEBUG(dbgs() << "Remove redundant Move : " << *MI);

          MI->eraseFromParent();
          Changed = true;
          LastChange = I;
          ++NumCopiesRemoved;
          UsedKnownRegs.insert(KnownReg.Reg);
          RemovedMI = true;
          break;
        }
      }
    }

    // Skip to the next instruction if we removed the COPY/MovImm.
    if (RemovedMI)
      continue;

    // Remove any regs the MI clobbers from the KnownRegs set.
    for (unsigned RI = 0; RI < KnownRegs.size();) {
      if (MI->modifiesRegister(KnownRegs[RI].Reg, TRI)) {
        std::swap(KnownRegs[RI], KnownRegs[KnownRegs.size() - 1]);
        KnownRegs.pop_back();
        // Don't increment RI since we need to now check the swapped-in
        // KnownRegs[RI].
      } else {
        ++RI;
      }
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

    // Continue until both known sets are empty.
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

  LLVM_DEBUG(dbgs() << "Clearing kill flags.\n\tFirstUse: " << *FirstUse
                    << "\tLastChange: ");
  if (LastChange == MBB->end())
    LLVM_DEBUG(dbgs() << "<end>\n");
  else
    LLVM_DEBUG(dbgs() << *LastChange);
  // Clear kills in the range where changes were made.  This is conservative,
  // but should be okay since kill markers are being phased out.
  for (MachineInstr &MMI : make_range(FirstUse, PredMBB->end()))
    MMI.clearKillInfo();
  for (MachineInstr &MMI : make_range(MBB->begin(), LastChange))
    MMI.clearKillInfo();

  return true;
}

bool ARMRedundantCopyEliminationImpl::run(MachineFunction &MF) {
  if (MF.getFunction().hasOptNone())
    return false;
  if (!MF.getProperties().hasNoVRegs())
    return false;

  TII = MF.getSubtarget().getInstrInfo();
  TRI = MF.getSubtarget().getRegisterInfo();
  MRI = &MF.getRegInfo();

  // Resize the clobbered and used register unit trackers.  We do this once per
  // function.
  DomBBClobberedRegs.init(*TRI);
  DomBBUsedRegs.init(*TRI);
  OptBBClobberedRegs.init(*TRI);
  OptBBUsedRegs.init(*TRI);

  bool Changed = false;
  for (MachineBasicBlock &MBB : MF)
    Changed |= optimizeBlock(&MBB);
  return Changed;
}

bool ARMRedundantCopyElimination::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;
  return ARMRedundantCopyEliminationImpl().run(MF);
}

PreservedAnalyses
ARMRedundantCopyEliminationPass::run(MachineFunction &MF,
                                     MachineFunctionAnalysisManager &MFAM) {
  if (MF.getFunction().hasOptNone())
    return PreservedAnalyses::all();
  if (!MF.getProperties().hasNoVRegs())
    return PreservedAnalyses::all();

  const bool Changed = ARMRedundantCopyEliminationImpl().run(MF);
  if (!Changed)
    return PreservedAnalyses::all();
  PreservedAnalyses PA = getMachineFunctionPassPreservedAnalyses();
  PA.preserveSet<CFGAnalyses>();
  return PA;
}

FunctionPass *llvm::createARMRedundantCopyEliminationPass() {
  return new ARMRedundantCopyElimination();
}
