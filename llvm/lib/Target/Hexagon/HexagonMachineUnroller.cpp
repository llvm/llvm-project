//===- HexagonMachineUnroller.cpp - Hexagon machine unroller --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Hexagon-specific implementation of machine loop unrolling.
//
//===----------------------------------------------------------------------===//

#include "HexagonMachineUnroller.h"
#include "HexagonInstrInfo.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/MachineUnroller.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"

using namespace llvm;

#define DEBUG_TYPE "hexagon-machine-unroller"

static bool executesAtMostOnce(MachineInstr *MI) {
  if (MI->getOpcode() != Hexagon::A2_andir)
    return false;
  if (MI->getOperand(2).getImm() == 1)
    return true;
  return false;
}

unsigned HexagonMachineUnroller::getLoopCount(MachineBasicBlock &LoopBB) const {
  // We expect a hardware loop currently. This means that IndVar is set
  // to null, and the compare is the ENDLOOP instruction.
  MachineBasicBlock::iterator I = LoopBB.getFirstTerminator();
  assert(I != LoopBB.end() && HII->isEndLoopN(I->getOpcode()) &&
         "Expecting a hardware loop");
  DebugLoc DL = I->getDebugLoc();
  SmallPtrSet<MachineBasicBlock *, 8> VisitedBBs;
  MachineInstr *Loop = HII->findLoopInstr(
      &LoopBB, I->getOpcode(), I->getOperand(0).getMBB(), VisitedBBs);
  if (!Loop)
    return 0;
  // The loop trip count is a compile-time value.
  if (Loop->getOpcode() == Hexagon::J2_loop0i ||
      Loop->getOpcode() == Hexagon::J2_loop1i) {
    LLVM_DEBUG(
        dbgs() << "HexagonMachineUnroller: Found compile-time loop count: "
               << Loop->getOperand(1).getImm() << "\n");
    return Loop->getOperand(1).getImm();
  }

  // The loop trip count is a run-time value.
  assert(Loop->getOpcode() == Hexagon::J2_loop0r && "Unexpected instruction");
  LLVM_DEBUG(
      dbgs() << "HexagonMachineUnroller: Found run-time loop count in reg "
             << Loop->getOperand(1).getReg() << "\n");
  return Loop->getOperand(1).getReg();
}

unsigned HexagonMachineUnroller::addUnrolledLoopCountMI(
    MachineBasicBlock &MBB, unsigned LC, unsigned UnrollFactor) const {
  assert(isPowerOf2_32(UnrollFactor) && "UnrollFactor must be a power of 2");
  MachineFunction *MF = MBB.getParent();
  unsigned ShiftBy = Log2_32(UnrollFactor);
  unsigned NewUnrolledLC = HII->createVR(MF, MVT::i32);
  BuildMI(MBB, MBB.instr_end(), DebugLoc(), HII->get(Hexagon::S2_lsr_i_r),
          NewUnrolledLC)
      .addReg(LC)
      .addImm(ShiftBy);
  return NewUnrolledLC;
}

unsigned
HexagonMachineUnroller::addRemLoopCountMI(MachineBasicBlock &MBB, unsigned LC,
                                          unsigned UnrollFactor) const {
  assert(isPowerOf2_32(UnrollFactor) && "UnrollFactor must be a power of 2");
  MachineFunction *MF = MBB.getParent();
  unsigned RemLC = HII->createVR(MF, MVT::i32);
  BuildMI(MBB, MBB.instr_end(), DebugLoc(), HII->get(Hexagon::A2_andir), RemLC)
      .addReg(LC)
      .addImm(UnrollFactor - 1);
  return RemLC;
}

void HexagonMachineUnroller::changeLoopCount(
    MachineBasicBlock &BB, MachineBasicBlock &Preheader,
    MachineBasicBlock &Header, MachineBasicBlock &LoopBB, unsigned LC,
    SmallVectorImpl<MachineOperand> &Cond) const {

  // We expect a hardware loop currently. This means that IndVar is set
  // to null, and the compare is the ENDLOOP instruction.
  MachineBasicBlock::iterator I = LoopBB.getFirstTerminator();
  assert(I != LoopBB.end() && HII->isEndLoopN(I->getOpcode()) &&
         "Expecting a hardware loop");
  DebugLoc DL = I->getDebugLoc();
  SmallPtrSet<MachineBasicBlock *, 8> VisitedBBs;
  MachineInstr *Loop = HII->findLoopInstr(
      &Header, I->getOpcode(), I->getOperand(0).getMBB(), VisitedBBs);
  if (!Loop) {
    LLVM_DEBUG(
        dbgs() << "HexagonMachineUnroller: Loop instruction not found\n");
    return;
  }
  // The loop trip count is a run-time value.
  if (Loop->getOpcode() != Hexagon::J2_loop0r) {
    LLVM_DEBUG(dbgs() << "HexagonMachineUnroller: Unexpected loop opcode: "
                      << Loop->getOpcode() << "\n");
    return;
  }
  MachineRegisterInfo &MRI = I->getParent()->getParent()->getRegInfo();
  MachineInstr *LCDefMI = MRI.getVRegDef(LC);
  MachineInstr *NewCmp;
  if (executesAtMostOnce(LCDefMI)) {
    // The loop executes at most once. Therefore, it must be unrolled
    // by removing loop setup, endloop and back-edge (jump) instruction to avoid
    // stalls due to front-end mispredictions.
    // FYI: the front end predicts endloop is taken twice and then waits to see
    // which way it goes when it encounters it a third time. Since loop[01] is
    // resolved by the back-end and it takes at least 10 cycles from fetch to
    // commit, for the very small loops that execute only once, it can result
    // into a lot of stalled cycles.
    unsigned LoopEnd = HII->createVR(MF, MVT::i1);
    NewCmp = BuildMI(&BB, DL, HII->get(Hexagon::C2_cmpgtui), LoopEnd)
                 .addReg(LC)
                 .addImm(0);
    I->eraseFromParent();
    Header.removeSuccessor(&Header);
  } else {
    unsigned LoopEnd = HII->createVR(MF, MVT::i1);
    NewCmp = BuildMI(&BB, DL, HII->get(Hexagon::C2_cmpgtui), LoopEnd)
                 .addReg(LC)
                 .addImm(0);
    BuildMI(&Preheader, DL, HII->get(Hexagon::J2_loop0r))
        .addMBB(Loop->getOperand(0).getMBB())
        .addReg(LC);
  }
  // Delete the old loop instruction.
  Loop->eraseFromParent();
  Cond.push_back(MachineOperand::CreateImm(Hexagon::J2_jumpf));
  Cond.push_back(NewCmp->getOperand(0));
  LLVM_DEBUG(dbgs() << "HexagonMachineUnroller: Updated loop count mechanism "
                       "for unrolled loop.\n");
}
