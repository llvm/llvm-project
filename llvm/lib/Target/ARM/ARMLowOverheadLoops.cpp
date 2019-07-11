//===-- ARMLowOverheadLoops.cpp - CodeGen Low-overhead Loops ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// Finalize v8.1-m low-overhead loops by converting the associated pseudo
/// instructions into machine operations.
/// The expectation is that the loop contains three pseudo instructions:
/// - t2*LoopStart - placed in the preheader or pre-preheader. The do-loop
///   form should be in the preheader, whereas the while form should be in the
///   preheaders only predecessor. TODO: Could DoLoopStart get moved into the
///   pre-preheader?
/// - t2LoopDec - placed within in the loop body.
/// - t2LoopEnd - the loop latch terminator.
///
//===----------------------------------------------------------------------===//

#include "ARM.h"
#include "ARMBaseInstrInfo.h"
#include "ARMBaseRegisterInfo.h"
#include "ARMBasicBlockInfo.h"
#include "ARMSubtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"

using namespace llvm;

#define DEBUG_TYPE "arm-low-overhead-loops"
#define ARM_LOW_OVERHEAD_LOOPS_NAME "ARM Low Overhead Loops pass"

namespace {

  class ARMLowOverheadLoops : public MachineFunctionPass {
    const ARMBaseInstrInfo    *TII = nullptr;
    MachineRegisterInfo       *MRI = nullptr;
    std::unique_ptr<ARMBasicBlockUtils> BBUtils = nullptr;

  public:
    static char ID;

    ARMLowOverheadLoops() : MachineFunctionPass(ID) { }

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.setPreservesCFG();
      AU.addRequired<MachineLoopInfo>();
      MachineFunctionPass::getAnalysisUsage(AU);
    }

    bool runOnMachineFunction(MachineFunction &MF) override;

    bool ProcessLoop(MachineLoop *ML);

    void RevertWhile(MachineInstr *MI) const;

    void RevertLoopDec(MachineInstr *MI) const;

    void RevertLoopEnd(MachineInstr *MI) const;

    void Expand(MachineLoop *ML, MachineInstr *Start,
                MachineInstr *Dec, MachineInstr *End, bool Revert);

    MachineFunctionProperties getRequiredProperties() const override {
      return MachineFunctionProperties().set(
          MachineFunctionProperties::Property::NoVRegs);
    }

    StringRef getPassName() const override {
      return ARM_LOW_OVERHEAD_LOOPS_NAME;
    }
  };
}
  
char ARMLowOverheadLoops::ID = 0;

INITIALIZE_PASS(ARMLowOverheadLoops, DEBUG_TYPE, ARM_LOW_OVERHEAD_LOOPS_NAME,
                false, false)

bool ARMLowOverheadLoops::runOnMachineFunction(MachineFunction &MF) {
  if (!static_cast<const ARMSubtarget&>(MF.getSubtarget()).hasLOB())
    return false;

  LLVM_DEBUG(dbgs() << "ARM Loops on " << MF.getName() << " ------------- \n");

  auto &MLI = getAnalysis<MachineLoopInfo>();
  MRI = &MF.getRegInfo();
  TII = static_cast<const ARMBaseInstrInfo*>(
    MF.getSubtarget().getInstrInfo());
  BBUtils = std::unique_ptr<ARMBasicBlockUtils>(new ARMBasicBlockUtils(MF));
  BBUtils->computeAllBlockSizes();
  BBUtils->adjustBBOffsetsAfter(&MF.front());

  bool Changed = false;
  for (auto ML : MLI) {
    if (!ML->getParentLoop())
      Changed |= ProcessLoop(ML);
  }
  return Changed;
}

bool ARMLowOverheadLoops::ProcessLoop(MachineLoop *ML) {

  bool Changed = false;

  // Process inner loops first.
  for (auto I = ML->begin(), E = ML->end(); I != E; ++I)
    Changed |= ProcessLoop(*I);

  LLVM_DEBUG(dbgs() << "ARM Loops: Processing " << *ML);

  auto IsLoopStart = [](MachineInstr &MI) {
    return MI.getOpcode() == ARM::t2DoLoopStart ||
           MI.getOpcode() == ARM::t2WhileLoopStart;
  };

  // Search the given block for a loop start instruction. If one isn't found,
  // and there's only one predecessor block, search that one too.
  std::function<MachineInstr*(MachineBasicBlock*)> SearchForStart =
    [&IsLoopStart, &SearchForStart](MachineBasicBlock *MBB) -> MachineInstr* {
    for (auto &MI : *MBB) {
      if (IsLoopStart(MI))
        return &MI;
    }
    if (MBB->pred_size() == 1)
      return SearchForStart(*MBB->pred_begin());
    return nullptr;
  };

  MachineInstr *Start = nullptr;
  MachineInstr *Dec = nullptr;
  MachineInstr *End = nullptr;
  bool Revert = false;

  // Search the preheader for the start intrinsic, or look through the
  // predecessors of the header to find exactly one set.iterations intrinsic.
  // FIXME: I don't see why we shouldn't be supporting multiple predecessors
  // with potentially multiple set.loop.iterations, so we need to enable this.
  if (auto *Preheader = ML->getLoopPreheader()) {
    Start = SearchForStart(Preheader);
  } else {
    LLVM_DEBUG(dbgs() << "ARM Loops: Failed to find loop preheader!\n"
               << " - Performing manual predecessor search.\n");
    MachineBasicBlock *Pred = nullptr;
    for (auto *MBB : ML->getHeader()->predecessors()) {
      if (!ML->contains(MBB)) {
        if (Pred) {
          LLVM_DEBUG(dbgs() << " - Found multiple out-of-loop preds.\n");
          Start = nullptr;
          break;
        }
        Pred = MBB;
        Start = SearchForStart(MBB);
      }
    }
  }

  // Find the low-overhead loop components and decide whether or not to fall
  // back to a normal loop.
  for (auto *MBB : reverse(ML->getBlocks())) {
    for (auto &MI : *MBB) {
      if (MI.getOpcode() == ARM::t2LoopDec)
        Dec = &MI;
      else if (MI.getOpcode() == ARM::t2LoopEnd)
        End = &MI;
      else if (MI.getDesc().isCall())
        // TODO: Though the call will require LE to execute again, does this
        // mean we should revert? Always executing LE hopefully should be
        // faster than performing a sub,cmp,br or even subs,br.
        Revert = true;

      if (!Dec)
        continue;

      // If we find that we load/store LR between LoopDec and LoopEnd, expect
      // that the decremented value has been spilled to the stack. Because
      // this value isn't actually going to be produced until the latch, by LE,
      // we would need to generate a real sub. The value is also likely to be
      // reloaded for use of LoopEnd - in which in case we'd need to perform
      // an add because it gets negated again by LE! The other option is to
      // then generate the other form of LE which doesn't perform the sub.
      if (MI.mayLoad() || MI.mayStore())
        Revert =
          MI.getOperand(0).isReg() && MI.getOperand(0).getReg() == ARM::LR;
    }

    if (Dec && End && Revert)
      break;
  }

  if (!Start && !Dec && !End) {
    LLVM_DEBUG(dbgs() << "ARM Loops: Not a low-overhead loop.\n");
    return Changed;
  } if (!(Start && Dec && End)) {
    report_fatal_error("Failed to find all loop components");
  }

  if (!End->getOperand(1).isMBB() ||
      End->getOperand(1).getMBB() != ML->getHeader())
    report_fatal_error("Expected LoopEnd to target Loop Header");

  // The WLS and LE instructions have 12-bits for the label offset. WLS
  // requires a positive offset, while LE uses negative.
  if (BBUtils->getOffsetOf(End) < BBUtils->getOffsetOf(ML->getHeader()) ||
      !BBUtils->isBBInRange(End, ML->getHeader(), 4094)) {
    LLVM_DEBUG(dbgs() << "ARM Loops: LE offset is out-of-range\n");
    Revert = true;
  }
  if (Start->getOpcode() == ARM::t2WhileLoopStart &&
      (BBUtils->getOffsetOf(Start) >
       BBUtils->getOffsetOf(Start->getOperand(1).getMBB()) ||
       !BBUtils->isBBInRange(Start, Start->getOperand(1).getMBB(), 4094))) {
    LLVM_DEBUG(dbgs() << "ARM Loops: WLS offset is out-of-range!\n");
    Revert = true;
  }

  LLVM_DEBUG(dbgs() << "ARM Loops:\n - Found Loop Start: " << *Start
                    << " - Found Loop Dec: " << *Dec
                    << " - Found Loop End: " << *End);

  Expand(ML, Start, Dec, End, Revert);
  return true;
}

// WhileLoopStart holds the exit block, so produce a cmp lr, 0 and then a
// beq that branches to the exit branch.
// FIXME: Need to check that we're not trashing the CPSR when generating the
// cmp. We could also try to generate a cbz if the value in LR is also in
// another low register.
void ARMLowOverheadLoops::RevertWhile(MachineInstr *MI) const {
  LLVM_DEBUG(dbgs() << "ARM Loops: Reverting to cmp: " << *MI);
  MachineBasicBlock *MBB = MI->getParent();
  MachineInstrBuilder MIB = BuildMI(*MBB, MI, MI->getDebugLoc(),
                                    TII->get(ARM::t2CMPri));
  MIB.addReg(ARM::LR);
  MIB.addImm(0);
  MIB.addImm(ARMCC::AL);
  MIB.addReg(ARM::CPSR);

  // TODO: Try to use tBcc instead
  MIB = BuildMI(*MBB, MI, MI->getDebugLoc(), TII->get(ARM::t2Bcc));
  MIB.add(MI->getOperand(1));   // branch target
  MIB.addImm(ARMCC::EQ);        // condition code
  MIB.addReg(ARM::CPSR);
  MI->eraseFromParent();
}

// TODO: Check flags so that we can possibly generate a tSubs or tSub.
void ARMLowOverheadLoops::RevertLoopDec(MachineInstr *MI) const {
  LLVM_DEBUG(dbgs() << "ARM Loops: Reverting to sub: " << *MI);
  MachineBasicBlock *MBB = MI->getParent();
  MachineInstrBuilder MIB = BuildMI(*MBB, MI, MI->getDebugLoc(),
                                    TII->get(ARM::t2SUBri));
  MIB.addDef(ARM::LR);
  MIB.add(MI->getOperand(1));
  MIB.add(MI->getOperand(2));
  MIB.addImm(ARMCC::AL);
  MIB.addReg(0);
  MIB.addReg(0);
  MI->eraseFromParent();
}

// Generate a subs, or sub and cmp, and a branch instead of an LE.
// FIXME: Need to check that we're not trashing the CPSR when generating
// the cmp.
void ARMLowOverheadLoops::RevertLoopEnd(MachineInstr *MI) const {
  LLVM_DEBUG(dbgs() << "ARM Loops: Reverting to cmp, br: " << *MI);

  // Create cmp
  MachineBasicBlock *MBB = MI->getParent();
  MachineInstrBuilder MIB = BuildMI(*MBB, MI, MI->getDebugLoc(),
                                    TII->get(ARM::t2CMPri));
  MIB.addReg(ARM::LR);
  MIB.addImm(0);
  MIB.addImm(ARMCC::AL);
  MIB.addReg(ARM::CPSR);

  // TODO Try to use tBcc instead.
  // Create bne
  MIB = BuildMI(*MBB, MI, MI->getDebugLoc(), TII->get(ARM::t2Bcc));
  MIB.add(MI->getOperand(1));   // branch target
  MIB.addImm(ARMCC::NE);        // condition code
  MIB.addReg(ARM::CPSR);
  MI->eraseFromParent();
}

void ARMLowOverheadLoops::Expand(MachineLoop *ML, MachineInstr *Start,
                                 MachineInstr *Dec, MachineInstr *End,
                                 bool Revert) {

  auto ExpandLoopStart = [this](MachineLoop *ML, MachineInstr *Start) {
    // The trip count should already been held in LR since the instructions
    // within the loop can only read and write to LR. So, there should be a
    // mov to setup the count. WLS/DLS perform this move, so find the original
    // and delete it - inserting WLS/DLS in its place.
    MachineBasicBlock *MBB = Start->getParent();
    MachineInstr *InsertPt = Start;
    for (auto &I : MRI->def_instructions(ARM::LR)) {
      if (I.getParent() != MBB)
        continue;

      // Always execute.
      if (!I.getOperand(2).isImm() || I.getOperand(2).getImm() != ARMCC::AL)
        continue;

      // Only handle move reg, if the trip count it will need moving into a reg
      // before the setup instruction anyway.
      if (!I.getDesc().isMoveReg() ||
          !I.getOperand(1).isIdenticalTo(Start->getOperand(0)))
        continue;
      InsertPt = &I;
      break;
    }

    unsigned Opc = Start->getOpcode() == ARM::t2DoLoopStart ?
      ARM::t2DLS : ARM::t2WLS;
    MachineInstrBuilder MIB =
      BuildMI(*MBB, InsertPt, InsertPt->getDebugLoc(), TII->get(Opc));

    MIB.addDef(ARM::LR);
    MIB.add(Start->getOperand(0));
    if (Opc == ARM::t2WLS)
      MIB.add(Start->getOperand(1));

    if (InsertPt != Start)
      InsertPt->eraseFromParent();
    Start->eraseFromParent();
    LLVM_DEBUG(dbgs() << "ARM Loops: Inserted start: " << *MIB);
    return &*MIB;
  };

  // Combine the LoopDec and LoopEnd instructions into LE(TP).
  auto ExpandLoopEnd = [this](MachineLoop *ML, MachineInstr *Dec,
                              MachineInstr *End) {
    MachineBasicBlock *MBB = End->getParent();
    MachineInstrBuilder MIB = BuildMI(*MBB, End, End->getDebugLoc(),
                                      TII->get(ARM::t2LEUpdate));
    MIB.addDef(ARM::LR);
    MIB.add(End->getOperand(0));
    MIB.add(End->getOperand(1));
    LLVM_DEBUG(dbgs() << "ARM Loops: Inserted LE: " << *MIB);

    End->eraseFromParent();
    Dec->eraseFromParent();
    return &*MIB;
  };

  // TODO: We should be able to automatically remove these branches before we
  // get here - probably by teaching analyzeBranch about the pseudo
  // instructions.
  // If there is an unconditional branch, after I, that just branches to the
  // next block, remove it.
  auto RemoveDeadBranch = [](MachineInstr *I) {
    MachineBasicBlock *BB = I->getParent();
    MachineInstr *Terminator = &BB->instr_back();
    if (Terminator->isUnconditionalBranch() && I != Terminator) {
      MachineBasicBlock *Succ = Terminator->getOperand(0).getMBB();
      if (BB->isLayoutSuccessor(Succ)) {
        LLVM_DEBUG(dbgs() << "ARM Loops: Removing branch: " << *Terminator);
        Terminator->eraseFromParent();
      }
    }
  };

  if (Revert) {
    if (Start->getOpcode() == ARM::t2WhileLoopStart)
      RevertWhile(Start);
    else
      Start->eraseFromParent();
    RevertLoopDec(Dec);
    RevertLoopEnd(End);
  } else {
    Start = ExpandLoopStart(ML, Start);
    RemoveDeadBranch(Start);
    End = ExpandLoopEnd(ML, Dec, End);
    RemoveDeadBranch(End);
  }
}

FunctionPass *llvm::createARMLowOverheadLoopsPass() {
  return new ARMLowOverheadLoops();
}
