//===- NanoMipsMoveOptimizer.cpp - nanoMIPS move opt. pass ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file This file contains a pass that performs move-related peephole
/// optimizations.
//
//===----------------------------------------------------------------------===//

#include "Mips.h"
#include "MipsSubtarget.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include <algorithm>
#include <cmath>

using namespace llvm;

#define NM_MOVE_OPT_NAME "nanoMIPS move optimization pass"

namespace {
struct NMMoveOpt : public MachineFunctionPass {
  using MBBIter = MachineBasicBlock::iterator;
  SmallVector<unsigned> GPR2REG1 = {Mips::A0_NM, Mips::A1_NM, Mips::A2_NM,
                                    Mips::A3_NM};
  SmallVector<unsigned> GPR2REG2 = {Mips::A1_NM, Mips::A2_NM, Mips::A3_NM,
                                    Mips::A4_NM};
  SmallVector<unsigned> GPR4 = {
      Mips::A0_NM, Mips::A1_NM, Mips::A2_NM, Mips::A3_NM,
      Mips::A4_NM, Mips::A5_NM, Mips::A6_NM, Mips::A7_NM,
      Mips::S0_NM, Mips::S1_NM, Mips::S2_NM, Mips::S3_NM,
      Mips::S4_NM, Mips::S5_NM, Mips::S6_NM, Mips::S7_NM};
  SmallVector<unsigned> GPR4ZERO = {
      Mips::A0_NM, Mips::A1_NM, Mips::A2_NM, Mips::A3_NM,
      Mips::A4_NM, Mips::A5_NM, Mips::A6_NM, Mips::ZERO_NM,
      Mips::S0_NM, Mips::S1_NM, Mips::S2_NM, Mips::S3_NM,
      Mips::S4_NM, Mips::S5_NM, Mips::S6_NM, Mips::S7_NM};
  static char ID;
  const MipsSubtarget *STI;
  const TargetInstrInfo *TII;
  NMMoveOpt() : MachineFunctionPass(ID) {}
  StringRef getPassName() const override { return NM_MOVE_OPT_NAME; }
  bool runOnMachineFunction(MachineFunction &) override;
  bool generateMoveP(MachineBasicBlock &);
  bool areMovePCompatibleMoves(MachineInstr *, MachineInstr *, bool &);
  bool areMovePRevCompatibleMoves(MachineInstr *, MachineInstr *, bool &);
};
} // namespace

char NMMoveOpt::ID = 0;

bool NMMoveOpt::runOnMachineFunction(MachineFunction &Fn) {
  STI = &static_cast<const MipsSubtarget &>(Fn.getSubtarget());
  TII = STI->getInstrInfo();
  bool Modified = false;
  for (MachineFunction::iterator MFI = Fn.begin(), E = Fn.end(); MFI != E;
       ++MFI) {
    MachineBasicBlock &MBB = *MFI;
    Modified |= generateMoveP(MBB);
  }

  return Modified;
}

static inline bool isInSet(SmallVector<unsigned> Set, unsigned Reg) {
  return std::find(Set.begin(), Set.end(), Reg) != Set.end();
}

bool NMMoveOpt::areMovePCompatibleMoves(MachineInstr *Move1,
                                        MachineInstr *Move2, bool &Swap) {
  Register Src1 = Move1->getOperand(1).getReg();
  Register Src2 = Move2->getOperand(1).getReg();
  Register Dst1 = Move1->getOperand(0).getReg();
  Register Dst2 = Move2->getOperand(0).getReg();
  if (Dst1 == Src1 || Dst1 == Src2 || Dst2 == Src1 || Dst2 == Src2)
    return false;

  if (!isInSet(GPR4ZERO, Src1) || !isInSet(GPR4ZERO, Src2))
    return false;

  auto *Dst1Iter = std::find(GPR2REG1.begin(), GPR2REG1.end(), Dst1);
  auto *Dst2Iter = std::find(GPR2REG2.begin(), GPR2REG2.end(), Dst2);
  if (Dst1Iter != GPR2REG1.end() && Dst2Iter != GPR2REG2.end()) {
    int Dst1Index = std::distance(GPR2REG1.begin(), Dst1Iter);
    int Dst2Index = std::distance(GPR2REG2.begin(), Dst2Iter);
    // Dst1 and Dst2 need to have same indices in their register sets.
    if (Dst1Index == Dst2Index) {
      Swap = false;
      return true;
    }
  }

  // Try to swap moves and see if they are compatible.
  Dst1Iter = std::find(GPR2REG2.begin(), GPR2REG2.end(), Dst1);
  Dst2Iter = std::find(GPR2REG1.begin(), GPR2REG1.end(), Dst2);
  if (Dst1Iter != GPR2REG2.end() && Dst2Iter != GPR2REG1.end()) {
    int Dst1Index = std::distance(GPR2REG2.begin(), Dst1Iter);
    int Dst2Index = std::distance(GPR2REG1.begin(), Dst2Iter);
    // Dst1 and Dst2 need to have same indices in their register sets.
    if (Dst1Index == Dst2Index) {
      Swap = true;
      return true;
    }
  }

  return false;
}

bool NMMoveOpt::areMovePRevCompatibleMoves(MachineInstr *Move1,
                                           MachineInstr *Move2, bool &Swap) {
  Register Src1 = Move1->getOperand(1).getReg();
  Register Src2 = Move2->getOperand(1).getReg();
  Register Dst1 = Move1->getOperand(0).getReg();
  Register Dst2 = Move2->getOperand(0).getReg();
  if (Dst1 == Src1 || Dst1 == Src2 || Dst2 == Src1 || Dst2 == Src2)
    return false;

  if (!isInSet(GPR4, Dst1) || !isInSet(GPR4, Dst2))
    return false;

  auto *Src1Iter = std::find(GPR2REG1.begin(), GPR2REG1.end(), Src1);
  auto *Src2Iter = std::find(GPR2REG2.begin(), GPR2REG2.end(), Src2);
  if (Src1Iter != GPR2REG1.end() && Src2Iter != GPR2REG2.end()) {
    int Src1Index = std::distance(GPR2REG1.begin(), Src1Iter);
    int Src2Index = std::distance(GPR2REG2.begin(), Src2Iter);
    // Src1 and Src2 need to have same indices in their register sets.
    if (Src1Index == Src2Index) {
      Swap = false;
      return true;
    }
  }

  // Try to swap moves and see if they are compatible.
  Src1Iter = std::find(GPR2REG2.begin(), GPR2REG2.end(), Src1);
  Src2Iter = std::find(GPR2REG1.begin(), GPR2REG1.end(), Src2);
  if (Src1Iter != GPR2REG2.end() && Src2Iter != GPR2REG1.end()) {
    int Src1Index = std::distance(GPR2REG2.begin(), Src1Iter);
    int Src2Index = std::distance(GPR2REG1.begin(), Src2Iter);
    // Src1 and Src2 need to have same indices in their register sets.
    if (Src1Index == Src2Index) {
      Swap = true;
      return true;
    }
  }

  return false;
}

//
// Generate MOVEP instruction from 2 MOVE instructions.
//
// There are 2 variants of MOVEP instruction (MOVEP and MOVEP[REV]). First has
// strict register rules for destination operands, while the second one has
// strict rules for source operands.
//
// The strict rule is: Supported registers are only $a0-$a4 and $dst2 has to be
// $(dst1 + 1). In REV variant this applies to source registers.
//
// move $a0, $s1  ->  movep $a0, $a1, $s1, $s7
// move $a1, $s7
//
// move $a1, $s1  ->  movep $a0, $a1, $s7, $s1
// move $a0, $s7
//
// move $s6, $a2  ->  movep $s6, $a4, $a2, $a3
// move $a4, $a3
//
// move $s6, $a2  ->  movep $a4, $s6, $a1, $a2
// move $a4, $a1
//
bool NMMoveOpt::generateMoveP(MachineBasicBlock &MBB) {
  SmallVector<std::pair<MachineInstr *, MachineInstr *>> MovePairs;
  MachineInstr *PrevMove = nullptr;

  auto IsMovePCandidate = [this](MachineInstr *MI) -> bool {
    Register Dst = MI->getOperand(0).getReg();
    Register Src = MI->getOperand(1).getReg();
    // Check if it's a candidate for MOVEP.
    if ((isInSet(GPR2REG1, Dst) || isInSet(GPR2REG2, Dst)) &&
        isInSet(GPR4ZERO, Src))
      return true;
    // Check if it's a candidate for MOVEP[REV].
    if (isInSet(GPR4, Dst) &&
        (isInSet(GPR2REG1, Src) || isInSet(GPR2REG2, Src)))
      return true;
    return false;
  };

  for (auto &MI : MBB) {
    if (MI.getOpcode() == Mips::MOVE_NM) {
      if (PrevMove) {
        bool Swap;
        if (areMovePRevCompatibleMoves(PrevMove, &MI, Swap) ||
            areMovePCompatibleMoves(PrevMove, &MI, Swap)) {
          if (Swap)
            MovePairs.push_back({&MI, PrevMove});
          else
            MovePairs.push_back({PrevMove, &MI});
          PrevMove = nullptr;
          continue;
        }
      }
      if (IsMovePCandidate(&MI)) {
        PrevMove = &MI;
        continue;
      }
    }
    PrevMove = nullptr;
  }

  for (const auto &Pair : MovePairs) {
    auto InsertBefore = std::next(MBBIter(Pair.first));
    BuildMI(MBB, InsertBefore, Pair.first->getDebugLoc(),
            TII->get(Mips::MOVEP_NM))
        .addReg(Pair.first->getOperand(0).getReg(), RegState::Define)
        .addReg(Pair.second->getOperand(0).getReg(), RegState::Define)
        .addReg(Pair.first->getOperand(1).getReg())
        .addReg(Pair.second->getOperand(1).getReg());
    MBB.erase(Pair.first);
    MBB.erase(Pair.second);
  }

  return false;
}

namespace llvm {
FunctionPass *createNanoMipsMoveOptimizerPass() { return new NMMoveOpt(); }
} // namespace llvm
