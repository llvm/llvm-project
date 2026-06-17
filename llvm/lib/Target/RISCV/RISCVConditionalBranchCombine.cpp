//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file implements a pass that detects consecutive conditional branches
/// and uses logical operations to reduce branch density. On out-of-order
/// machines with high branch costs, developers can set the maximum allowed
/// consecutive conditional branches using -riscv-bbc-max-branch=N. This pass
/// aims to break sequences of up to N consecutive branches as much as possible,
/// thereby reducing front-end stalls.
///
/// For example, -riscv-bbc-max-branch=4
///
/// beqz a1, .L1
/// beqz t1, .L1
/// beqz a5, .L1
/// beqz a6, .L1
/// beqz t0, .L1
///
/// is transformed to
///
/// beqz a1, .L1
/// beqz t1, .L1
/// beqz a5, .L1
/// seqz t2, a6
/// seqz t3, t0
/// or t2, t2, t3
/// bne t2, zero, .L1
///
/// This pass should be run before register allocation.
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVInstrInfo.h"
#include "RISCVSubtarget.h"
#include "RISCVTargetTransformInfo.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/Support/InstructionCost.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-branch-cond-combine"

// N = 0 disables this pass.
static cl::opt<unsigned> MaxConsecutiveCondBranches(
    "riscv-cbc-max-branch", cl::Hidden,
    cl::desc("A maximum of N consecutive conditional "
             "branches is allowed."),
    cl::init(4));

namespace {

class RISCVConditionalBranchCombine : public MachineFunctionPass {
public:
  static char ID;

  const RISCVSubtarget *STI;
  const TargetInstrInfo *TII;
  MachineRegisterInfo *MRI;
  InstructionCost BranchMispredictPenalty = 0;

  RISCVConditionalBranchCombine() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;
  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::IsSSA);
  }

  StringRef getPassName() const override {
    return "RISC-V Conditional Branch Combine";
  }

private:
  bool isProfitableToCombine();
  void findBranchChains(MachineFunction &MF,
                        SmallVector<SmallVector<MachineInstr *, 4>, 8> &Chains);
  bool isSafeToCombine(const MachineInstr &MI1, const MachineInstr &MI2) const;
  bool checkOperand(const MachineInstr &MI) const;
  void combineTwoBranches(MachineInstr *Br1, MachineInstr *Br2);
  Register buildBranchCondValue(MachineInstr *Br, MachineBasicBlock &MBB,
                                MachineInstr *InsertPt);
};

} // end anonymous namespace

char RISCVConditionalBranchCombine::ID = 0;

INITIALIZE_PASS(RISCVConditionalBranchCombine, DEBUG_TYPE,
                "RISC-V Conditional Branch Combine", false, false)

/// Determine whether the basic block contains only one branch.
bool isSingleCondBranchBlock(const MachineBasicBlock &MBB) {
  SmallVector<const MachineInstr *, 4> Terms;

  for (const MachineInstr &MI : MBB) {
    if (MI.isDebugInstr())
      continue;

    if (!MI.isTerminator())
      return false;

    Terms.push_back(&MI);
  }

  if (Terms.size() != 2)
    return false;

  return Terms[0]->isConditionalBranch() && Terms[1]->isUnconditionalBranch();
}

/// Detect chains of consecutive N+1 branch instructions.
void RISCVConditionalBranchCombine::findBranchChains(
    MachineFunction &MF,
    SmallVector<SmallVector<MachineInstr *, 4>, 8> &Chains) {

  SmallPtrSet<MachineBasicBlock *, 16> Visited;

  for (auto &MBB : MF) {

    if (Visited.count(&MBB))
      continue;

    // The first branch instruction must be the last instruction of the bb or
    // the only instruction in the bb.
    auto It = MBB.getFirstTerminator();
    if (It == MBB.end())
      continue;
    MachineInstr *LastInstr = &*It;

    if (!LastInstr || !LastInstr->isConditionalBranch())
      continue;

    SmallVector<MachineInstr *, 4> Chain;
    MachineBasicBlock *CurBB = &MBB;

    while (true) {

      auto It = CurBB->getFirstTerminator();
      if (It == CurBB->end())
        break;
      MachineInstr *Br = &*It;

      if (!Br || !Br->isConditionalBranch())
        break;

      Chain.push_back(Br);
      Visited.insert(CurBB);

      if (Chain.size() >= MaxConsecutiveCondBranches + 1)
        break;

      MachineBasicBlock *Next = nullptr;
      if (CurBB->canFallThrough())
        Next = CurBB->getFallThrough();
      else
        break;

      // The inner bb contains only a conditional branch (ignore pseudoBR).
      if (!isSingleCondBranchBlock(*Next))
        break;

      CurBB = Next;
    }

    if (Chain.size() >= MaxConsecutiveCondBranches)
      Chains.push_back(std::move(Chain));
  }
}

/// Check if operands are virtual registers (not memory).
bool RISCVConditionalBranchCombine::checkOperand(const MachineInstr &MI) const {

  // Check first operand.
  if (!MI.getOperand(0).isReg())
    return false;

  Register Reg1 = MI.getOperand(0).getReg();
  if (Reg1.isVirtual()) {
    MachineInstr *Def1 = MRI->getVRegDef(Reg1);
    if (!Def1)
      return false;
  }

  // Check second operand.
  if (MI.getNumOperands() >= 3 && !MI.getOperand(1).isReg())
    return false;

  if (MI.getNumOperands() >= 3 && MI.getOperand(1).isReg()) {
    Register Reg2 = MI.getOperand(1).getReg();
    if (Reg2.isVirtual()) {
      MachineInstr *Def1 = MRI->getVRegDef(Reg2);
      if (!Def1)
        return false;
    }
  }

  return true;
}

/// Check if combining these two branch instructions is safe.
bool RISCVConditionalBranchCombine::isSafeToCombine(
    const MachineInstr &Br1, const MachineInstr &Br2) const {
  const MachineBasicBlock *MBB1 = Br1.getParent();
  const MachineBasicBlock *MBB2 = Br2.getParent();

  // If the basic block has other predecessor blocks, then this branch should
  // not be combined.
  if (!MBB1->getSinglePredecessor() || !MBB2->getSinglePredecessor())
    return false;

  if (Br1.getNumOperands() < 2 || Br2.getNumOperands() < 2)
    return false;

  // Check if both branches have the same direct target.
  MachineBasicBlock *TargetMBB1 =
      Br1.getOperand(Br1.getNumOperands() - 1).getMBB();
  MachineBasicBlock *TargetMBB2 =
      Br2.getOperand(Br2.getNumOperands() - 1).getMBB();

  if (TargetMBB1 != TargetMBB2) {
    // Should we canonicalize branch conditions using reverseBranchCondition()?
    // For example, convert inequality conditions into equality form.
    return false;
  }

  // Check if branches are register comparisons (not memory accesses or calls).
  if (!checkOperand(Br1) || !checkOperand(Br2))
    return false;

  return true;
}

/// Booleanize the branch condition based on the comparison predicate.
Register RISCVConditionalBranchCombine::buildBranchCondValue(
    MachineInstr *Br, MachineBasicBlock &MBB, MachineInstr *InsertPt) {
  DebugLoc DL = Br->getDebugLoc();

  Register rs1 = Br->getOperand(0).getReg();
  Register rs2 = Br->getOperand(1).getReg();
  unsigned Opc = Br->getOpcode();

  Register Dst = MRI->createVirtualRegister(&RISCV::GPRRegClass);

  auto buildXorSeqz = [&](bool IsEq) -> Register {
    unsigned Opc2 = IsEq ? RISCV::SLTIU : RISCV::SLTU;
    if (rs2 == RISCV::X0) {

      if (Opc == RISCV::BEQ)
        BuildMI(MBB, InsertPt, DL, TII->get(Opc2), Dst).addUse(rs1).addImm(1);
      else
        BuildMI(MBB, InsertPt, DL, TII->get(Opc2), Dst)
            .addUse(RISCV::X0)
            .addUse(rs1);
      return Dst;
    }

    Register X = MRI->createVirtualRegister(&RISCV::GPRRegClass);
    BuildMI(MBB, InsertPt, DL, TII->get(RISCV::XOR), X).addReg(rs1).addReg(rs2);

    if (Opc == RISCV::BEQ)
      BuildMI(MBB, InsertPt, DL, TII->get(Opc2), Dst).addReg(X).addImm(1);
    else
      BuildMI(MBB, InsertPt, DL, TII->get(Opc2), Dst)
          .addReg(RISCV::X0)
          .addReg(X);

    return Dst;
  };

  switch (Opc) {
  case RISCV::BEQ:
    return buildXorSeqz(true);

  case RISCV::BNE:
    return buildXorSeqz(false);

  case RISCV::BLT:
    BuildMI(MBB, InsertPt, DL, TII->get(RISCV::SLT), Dst)
        .addReg(rs1)
        .addReg(rs2);
    return Dst;

  case RISCV::BLTU:
    BuildMI(MBB, InsertPt, DL, TII->get(RISCV::SLTU), Dst)
        .addReg(rs1)
        .addReg(rs2);
    return Dst;

  case RISCV::BGE: {
    Register Tmp = MRI->createVirtualRegister(&RISCV::GPRRegClass);
    BuildMI(MBB, InsertPt, DL, TII->get(RISCV::SLT), Tmp)
        .addReg(rs1)
        .addReg(rs2);

    BuildMI(MBB, InsertPt, DL, TII->get(RISCV::SLTIU), Dst)
        .addReg(Tmp)
        .addImm(1);
    return Dst;
  }

  case RISCV::BGEU: {
    Register Tmp = MRI->createVirtualRegister(&RISCV::GPRRegClass);
    BuildMI(MBB, InsertPt, DL, TII->get(RISCV::SLTU), Tmp)
        .addReg(rs1)
        .addReg(rs2);

    BuildMI(MBB, InsertPt, DL, TII->get(RISCV::SLTIU), Dst)
        .addReg(Tmp)
        .addImm(1);
    return Dst;
  }

  default:
    llvm_unreachable("unsupported branch");
  }
}

/// Combine the last two branches in a branch chain using logical operations.
void RISCVConditionalBranchCombine::combineTwoBranches(MachineInstr *Br1,
                                                       MachineInstr *Br2) {

  MachineBasicBlock &MBB1 = *Br1->getParent();
  MachineBasicBlock &MBB2 = *Br2->getParent();
  DebugLoc DL = Br2->getDebugLoc();
  MachineBasicBlock *TargetBB = Br1->getOperand(2).getMBB();

  // Convert the branch condition into a boolean value.
  Register Cond1 = buildBranchCondValue(Br1, MBB2, Br2);
  Register Cond2 = buildBranchCondValue(Br2, MBB2, Br2);

  // Compute the final branch condition using logical operations.
  Register Dst = MRI->createVirtualRegister(&RISCV::GPRRegClass);
  BuildMI(MBB2, Br2, DL, TII->get(RISCV::OR), Dst).addReg(Cond1).addReg(Cond2);

  // Jump to TargetBB if any branch condition is true.
  BuildMI(MBB2, Br2, DL, TII->get(RISCV::BNE))
      .addReg(Dst)
      .addReg(RISCV::X0)
      .addMBB(TargetBB);

  Br1->eraseFromParent();
  Br2->eraseFromParent();

  // MBB1 only has a PseudoBR left; remove it.
  MachineBasicBlock *MBB1Pre = MBB1.getSinglePredecessor();

  // Update PHI nodes of MBB1's successors.
  for (MachineBasicBlock *MBB1Succ : MBB1.successors())
    MBB1Succ->removePHIsIncomingValuesForPredecessor(MBB1);

  // Update the cfg .
  for (MachineInstr &MI : MBB1Pre->terminators()) {
    if (MI.getOpcode() == RISCV::PseudoBR) {
      MI.getOperand(0).setMBB(&MBB2);
    }
  }

  MBB1Pre->replaceSuccessor(&MBB1, &MBB2);
  MBB1.removeSuccessor(TargetBB);
  MBB1.removeSuccessor(&MBB2);
  MBB1.eraseFromParent();
}

bool RISCVConditionalBranchCombine::isProfitableToCombine() {

  return BranchMispredictPenalty >= 10;
}

bool RISCVConditionalBranchCombine::runOnMachineFunction(MachineFunction &MF) {

  if (skipFunction(MF.getFunction()))
    return false;

  STI = &MF.getSubtarget<RISCVSubtarget>();
  TII = STI->getInstrInfo();
  MRI = &MF.getRegInfo();
  const auto &SchedModel = STI->getSchedModel();
  BranchMispredictPenalty = SchedModel.MispredictPenalty;

  if (!isProfitableToCombine())
    return false;

  if (MaxConsecutiveCondBranches == 0)
    return false;

  bool Changed = false;
  SmallVector<SmallVector<MachineInstr *, 4>, 8> BrChains;

  // Detect chains of up to N+1 consecutive branch instructions.
  findBranchChains(MF, BrChains);

  for (auto &Chain : BrChains) {
    if (Chain.size() < 2)
      continue;

    MachineInstr *Br1 = Chain[Chain.size() - 2];
    MachineInstr *Br2 = Chain[Chain.size() - 1];

    // Combine the last two branches in a chain to reduce branch density and
    // avoid long sequences of consecutive branches (up to N).
    if (isSafeToCombine(*Br1, *Br2)) {
      combineTwoBranches(&*Br1, &*Br2);
      Changed = true;
    }
  }

  return Changed;
}

/// createRISCVConditionalBranchCombinePass() returns an instance of the branch
/// combine pass.
FunctionPass *llvm::createRISCVConditionalBranchCombinePass() {
  return new RISCVConditionalBranchCombine();
}
