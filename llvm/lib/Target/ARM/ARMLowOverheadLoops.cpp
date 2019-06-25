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
    return MI.getOpcode() == ARM::t2DoLoopStart;
  };

  auto SearchForStart =
    [&IsLoopStart](MachineBasicBlock *MBB) -> MachineInstr* {
    for (auto &MI : *MBB) {
      if (IsLoopStart(MI))
        return &MI;
    }
    return nullptr;
  };

  MachineInstr *Start = nullptr;
  MachineInstr *Dec = nullptr;
  MachineInstr *End = nullptr;
  bool Revert = false;

  if (auto *Preheader = ML->getLoopPreheader())
    Start = SearchForStart(Preheader);

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

  if (Start || Dec || End) {
    if (!Start || !Dec || !End)
      report_fatal_error("Failed to find all loop components");
  } else {
    LLVM_DEBUG(dbgs() << "ARM Loops: Not a low-overhead loop.\n");
    return Changed;
  }

  if (!End->getOperand(1).isMBB() ||
      End->getOperand(1).getMBB() != ML->getHeader())
    report_fatal_error("Expected LoopEnd to target Loop Header");

  // The LE instructions has 12-bits for the label offset.
  if (!BBUtils->isBBInRange(End, ML->getHeader(), 4096)) {
    LLVM_DEBUG(dbgs() << "ARM Loops: Too large for a low-overhead loop!\n");
    Revert = true;
  }

  LLVM_DEBUG(dbgs() << "ARM Loops:\n - Found Loop Start: " << *Start
                    << " - Found Loop Dec: " << *Dec
                    << " - Found Loop End: " << *End);

  Expand(ML, Start, Dec, End, Revert);
  return true;
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

    MachineInstrBuilder MIB =
      BuildMI(*MBB, InsertPt, InsertPt->getDebugLoc(), TII->get(ARM::t2DLS));
    if (InsertPt != Start)
      InsertPt->eraseFromParent();

    MIB.addDef(ARM::LR);
    MIB.add(Start->getOperand(0));
    LLVM_DEBUG(dbgs() << "ARM Loops: Inserted DLS: " << *MIB);
    Start->eraseFromParent();
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

    // If there is a branch after loop end, which branches to the fallthrough
    // block, remove the branch.
    MachineBasicBlock *Latch = End->getParent();
    MachineInstr *Terminator = &Latch->instr_back();
    if (End != Terminator) {
      MachineBasicBlock *Exit = ML->getExitBlock();
      if (Latch->isLayoutSuccessor(Exit)) {
        LLVM_DEBUG(dbgs() << "ARM Loops: Removing loop exit branch: "
                   << *Terminator);
        Terminator->eraseFromParent();
      }
    }
    End->eraseFromParent();
    Dec->eraseFromParent();
  };

  // Generate a subs, or sub and cmp, and a branch instead of an LE.
  // TODO: Check flags so that we can possibly generate a subs.
  auto ExpandBranch = [this](MachineInstr *Dec, MachineInstr *End) {
    LLVM_DEBUG(dbgs() << "ARM Loops: Reverting to sub, cmp, br.\n");
    // Create sub
    MachineBasicBlock *MBB = Dec->getParent();
    MachineInstrBuilder MIB = BuildMI(*MBB, Dec, Dec->getDebugLoc(),
                                      TII->get(ARM::t2SUBri));
    MIB.addDef(ARM::LR);
    MIB.add(Dec->getOperand(1));
    MIB.add(Dec->getOperand(2));
    MIB.addImm(ARMCC::AL);
    MIB.addReg(0);
    MIB.addReg(0);

    // Create cmp
    MBB = End->getParent();
    MIB = BuildMI(*MBB, End, End->getDebugLoc(), TII->get(ARM::t2CMPri));
    MIB.addReg(ARM::LR);
    MIB.addImm(0);
    MIB.addImm(ARMCC::AL);
    MIB.addReg(ARM::CPSR);

    // Create bne
    MIB = BuildMI(*MBB, End, End->getDebugLoc(), TII->get(ARM::t2Bcc));
    MIB.add(End->getOperand(1));  // branch target
    MIB.addImm(ARMCC::NE);        // condition code
    MIB.addReg(ARM::CPSR);
    End->eraseFromParent();
    Dec->eraseFromParent();
  };

  if (Revert) {
    Start->eraseFromParent();
    ExpandBranch(Dec, End);
  } else {
    ExpandLoopStart(ML, Start);
    ExpandLoopEnd(ML, Dec, End);
  }
}

FunctionPass *llvm::createARMLowOverheadLoopsPass() {
  return new ARMLowOverheadLoops();
}
