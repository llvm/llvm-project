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
///   preheaders only predecessor.
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
    MachineFunction           *MF = nullptr;
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

    MachineFunctionProperties getRequiredProperties() const override {
      return MachineFunctionProperties().set(
          MachineFunctionProperties::Property::NoVRegs);
    }

    StringRef getPassName() const override {
      return ARM_LOW_OVERHEAD_LOOPS_NAME;
    }

  private:
    bool ProcessLoop(MachineLoop *ML);

    MachineInstr * IsSafeToDefineLR(MachineInstr *MI);

    bool RevertNonLoops();

    void RevertWhile(MachineInstr *MI) const;

    bool RevertLoopDec(MachineInstr *MI, bool AllowFlags = false) const;

    void RevertLoopEnd(MachineInstr *MI, bool SkipCmp = false) const;

    void Expand(MachineLoop *ML, MachineInstr *Start,
                MachineInstr *InsertPt, MachineInstr *Dec,
                MachineInstr *End, bool Revert);

  };
}

char ARMLowOverheadLoops::ID = 0;

INITIALIZE_PASS(ARMLowOverheadLoops, DEBUG_TYPE, ARM_LOW_OVERHEAD_LOOPS_NAME,
                false, false)

bool ARMLowOverheadLoops::runOnMachineFunction(MachineFunction &mf) {
  const ARMSubtarget &ST = static_cast<const ARMSubtarget&>(mf.getSubtarget());
  if (!ST.hasLOB())
    return false;

  MF = &mf;
  LLVM_DEBUG(dbgs() << "ARM Loops on " << MF->getName() << " ------------- \n");

  auto &MLI = getAnalysis<MachineLoopInfo>();
  MF->getProperties().set(MachineFunctionProperties::Property::TracksLiveness);
  MRI = &MF->getRegInfo();
  TII = static_cast<const ARMBaseInstrInfo*>(ST.getInstrInfo());
  BBUtils = std::unique_ptr<ARMBasicBlockUtils>(new ARMBasicBlockUtils(*MF));
  BBUtils->computeAllBlockSizes();
  BBUtils->adjustBBOffsetsAfter(&MF->front());

  bool Changed = false;
  for (auto ML : MLI) {
    if (!ML->getParentLoop())
      Changed |= ProcessLoop(ML);
  }
  Changed |= RevertNonLoops();
  return Changed;
}

static bool IsLoopStart(MachineInstr &MI) {
  return MI.getOpcode() == ARM::t2DoLoopStart ||
         MI.getOpcode() == ARM::t2WhileLoopStart;
}

template<typename T>
static MachineInstr* SearchForDef(MachineInstr *Begin, T End, unsigned Reg) {
  for(auto &MI : make_range(T(Begin), End)) {
    for (auto &MO : MI.operands()) {
      if (!MO.isReg() || !MO.isDef() || MO.getReg() != Reg)
        continue;
      return &MI;
    }
  }
  return nullptr;
}

static MachineInstr* SearchForUse(MachineInstr *Begin,
                                  MachineBasicBlock::iterator End,
                                  unsigned Reg) {
  for(auto &MI : make_range(MachineBasicBlock::iterator(Begin), End)) {
    for (auto &MO : MI.operands()) {
      if (!MO.isReg() || !MO.isUse() || MO.getReg() != Reg)
        continue;
      return &MI;
    }
  }
  return nullptr;
}

// Is it safe to define LR with DLS/WLS?
// LR can defined if it is the operand to start, because it's the same value,
// or if it's going to be equivalent to the operand to Start.
MachineInstr *ARMLowOverheadLoops::IsSafeToDefineLR(MachineInstr *Start) {

  auto IsMoveLR = [](MachineInstr *MI, unsigned Reg) {
    return MI->getOpcode() == ARM::tMOVr &&
           MI->getOperand(0).getReg() == ARM::LR &&
           MI->getOperand(1).getReg() == Reg &&
           MI->getOperand(2).getImm() == ARMCC::AL;
   };

  MachineBasicBlock *MBB = Start->getParent();
  unsigned CountReg = Start->getOperand(0).getReg();
  // Walk forward and backward in the block to find the closest instructions
  // that define LR. Then also filter them out if they're not a mov lr.
  MachineInstr *PredLRDef = SearchForDef(Start, MBB->rend(), ARM::LR);
  if (PredLRDef && !IsMoveLR(PredLRDef, CountReg))
    PredLRDef = nullptr;

  MachineInstr *SuccLRDef = SearchForDef(Start, MBB->end(), ARM::LR);
  if (SuccLRDef && !IsMoveLR(SuccLRDef, CountReg))
    SuccLRDef = nullptr;

  // We've either found one, two or none mov lr instructions... Now figure out
  // if they are performing the equilvant mov that the Start instruction will.
  // Do this by scanning forward and backward to see if there's a def of the
  // register holding the count value. If we find a suitable def, return it as
  // the insert point. Later, if InsertPt != Start, then we can remove the
  // redundant instruction.
  if (SuccLRDef) {
    MachineBasicBlock::iterator End(SuccLRDef);
    if (!SearchForDef(Start, End, CountReg)) {
      return SuccLRDef;
    } else
      SuccLRDef = nullptr;
  }
  if (PredLRDef) {
    MachineBasicBlock::reverse_iterator End(PredLRDef);
    if (!SearchForDef(Start, End, CountReg)) {
      return PredLRDef;
    } else
      PredLRDef = nullptr;
  }

  // We can define LR because LR already contains the same value.
  if (Start->getOperand(0).getReg() == ARM::LR)
    return Start;

  // We've found no suitable LR def and Start doesn't use LR directly. Can we
  // just define LR anyway? 
  const TargetRegisterInfo *TRI = MF->getSubtarget().getRegisterInfo();
  LivePhysRegs LiveRegs(*TRI);
  LiveRegs.addLiveOuts(*MBB);

  // Not if we've haven't found a suitable mov and LR is live out.
  if (LiveRegs.contains(ARM::LR))
    return nullptr;

  // If LR is not live out, we can insert the instruction if nothing else
  // uses LR after it.
  if (!SearchForUse(Start, MBB->end(), ARM::LR))
    return Start;

  LLVM_DEBUG(dbgs() << "ARM Loops: Failed to find suitable insertion point for"
             << " LR\n");
  return nullptr;
}

bool ARMLowOverheadLoops::ProcessLoop(MachineLoop *ML) {

  bool Changed = false;

  // Process inner loops first.
  for (auto I = ML->begin(), E = ML->end(); I != E; ++I)
    Changed |= ProcessLoop(*I);

  LLVM_DEBUG(dbgs() << "ARM Loops: Processing " << *ML);

  // Search the given block for a loop start instruction. If one isn't found,
  // and there's only one predecessor block, search that one too.
  std::function<MachineInstr*(MachineBasicBlock*)> SearchForStart =
    [&SearchForStart](MachineBasicBlock *MBB) -> MachineInstr* {
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
      else if (IsLoopStart(MI))
        Start = &MI;
      else if (MI.getDesc().isCall()) {
        // TODO: Though the call will require LE to execute again, does this
        // mean we should revert? Always executing LE hopefully should be
        // faster than performing a sub,cmp,br or even subs,br.
        Revert = true;
        LLVM_DEBUG(dbgs() << "ARM Loops: Found call.\n");
      }

      if (!Dec || End)
        continue;

      // If we find that LR has been written or read between LoopDec and
      // LoopEnd, expect that the decremented value is being used else where.
      // Because this value isn't actually going to be produced until the
      // latch, by LE, we would need to generate a real sub. The value is also
      // likely to be copied/reloaded for use of LoopEnd - in which in case
      // we'd need to perform an add because it gets subtracted again by LE!
      // The other option is to then generate the other form of LE which doesn't
      // perform the sub.
      for (auto &MO : MI.operands()) {
        if (MI.getOpcode() != ARM::t2LoopDec && MO.isReg() &&
            MO.getReg() == ARM::LR) {
          LLVM_DEBUG(dbgs() << "ARM Loops: Found LR Use/Def: " << MI);
          Revert = true;
          break;
        }
      }
    }

    if (Dec && End && Revert)
      break;
  }

  LLVM_DEBUG(if (Start) dbgs() << "ARM Loops: Found Loop Start: " << *Start;
             if (Dec) dbgs() << "ARM Loops: Found Loop Dec: " << *Dec;
             if (End) dbgs() << "ARM Loops: Found Loop End: " << *End;);

  if (!Start && !Dec && !End) {
    LLVM_DEBUG(dbgs() << "ARM Loops: Not a low-overhead loop.\n");
    return Changed;
  } else if (!(Start && Dec && End)) {
    LLVM_DEBUG(dbgs() << "ARM Loops: Failed to find all loop components.\n");
    return false;
  }

  if (!End->getOperand(1).isMBB())
    report_fatal_error("Expected LoopEnd to target basic block");

  // TODO Maybe there's cases where the target doesn't have to be the header,
  // but for now be safe and revert.
  if (End->getOperand(1).getMBB() != ML->getHeader()) {
    LLVM_DEBUG(dbgs() << "ARM Loops: LoopEnd is not targetting header.\n");
    Revert = true;
  }

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

  MachineInstr *InsertPt = Revert ? nullptr : IsSafeToDefineLR(Start);
  if (!InsertPt) {
    LLVM_DEBUG(dbgs() << "ARM Loops: Unable to find safe insertion point.\n");
    Revert = true;
  } else
    LLVM_DEBUG(dbgs() << "ARM Loops: Start insertion point: " << *InsertPt);

  Expand(ML, Start, InsertPt, Dec, End, Revert);
  return true;
}

// WhileLoopStart holds the exit block, so produce a cmp lr, 0 and then a
// beq that branches to the exit branch.
// TODO: We could also try to generate a cbz if the value in LR is also in
// another low register.
void ARMLowOverheadLoops::RevertWhile(MachineInstr *MI) const {
  LLVM_DEBUG(dbgs() << "ARM Loops: Reverting to cmp: " << *MI);
  MachineBasicBlock *MBB = MI->getParent();
  MachineInstrBuilder MIB = BuildMI(*MBB, MI, MI->getDebugLoc(),
                                    TII->get(ARM::t2CMPri));
  MIB.add(MI->getOperand(0));
  MIB.addImm(0);
  MIB.addImm(ARMCC::AL);
  MIB.addReg(ARM::NoRegister);
  
  MachineBasicBlock *DestBB = MI->getOperand(1).getMBB();
  unsigned BrOpc = BBUtils->isBBInRange(MI, DestBB, 254) ?
    ARM::tBcc : ARM::t2Bcc;

  MIB = BuildMI(*MBB, MI, MI->getDebugLoc(), TII->get(BrOpc));
  MIB.add(MI->getOperand(1));   // branch target
  MIB.addImm(ARMCC::EQ);        // condition code
  MIB.addReg(ARM::CPSR);
  MI->eraseFromParent();
}

bool ARMLowOverheadLoops::RevertLoopDec(MachineInstr *MI,
                                        bool AllowFlags) const {
  LLVM_DEBUG(dbgs() << "ARM Loops: Reverting to sub: " << *MI);
  MachineBasicBlock *MBB = MI->getParent();

  // If nothing uses or defines CPSR between LoopDec and LoopEnd, use a t2SUBS.
  bool SetFlags = false;
  if (AllowFlags) {
    if (auto *Def = SearchForDef(MI, MBB->end(), ARM::CPSR)) {
      if (!SearchForUse(MI, MBB->end(), ARM::CPSR) &&
          Def->getOpcode() == ARM::t2LoopEnd)
        SetFlags = true;
    }
  }

  MachineInstrBuilder MIB = BuildMI(*MBB, MI, MI->getDebugLoc(),
                                    TII->get(ARM::t2SUBri));
  MIB.addDef(ARM::LR);
  MIB.add(MI->getOperand(1));
  MIB.add(MI->getOperand(2));
  MIB.addImm(ARMCC::AL);
  MIB.addReg(0);

  if (SetFlags) {
    MIB.addReg(ARM::CPSR);
    MIB->getOperand(5).setIsDef(true);
  } else
    MIB.addReg(0);

  MI->eraseFromParent();
  return SetFlags;
}

// Generate a subs, or sub and cmp, and a branch instead of an LE.
void ARMLowOverheadLoops::RevertLoopEnd(MachineInstr *MI, bool SkipCmp) const {
  LLVM_DEBUG(dbgs() << "ARM Loops: Reverting to cmp, br: " << *MI);

  MachineBasicBlock *MBB = MI->getParent();
  // Create cmp
  if (!SkipCmp) {
    MachineInstrBuilder MIB = BuildMI(*MBB, MI, MI->getDebugLoc(),
                                      TII->get(ARM::t2CMPri));
    MIB.addReg(ARM::LR);
    MIB.addImm(0);
    MIB.addImm(ARMCC::AL);
    MIB.addReg(ARM::NoRegister);
  }

  MachineBasicBlock *DestBB = MI->getOperand(1).getMBB();
  unsigned BrOpc = BBUtils->isBBInRange(MI, DestBB, 254) ?
    ARM::tBcc : ARM::t2Bcc;

  // Create bne
  MachineInstrBuilder MIB =
    BuildMI(*MBB, MI, MI->getDebugLoc(), TII->get(BrOpc));
  MIB.add(MI->getOperand(1));   // branch target
  MIB.addImm(ARMCC::NE);        // condition code
  MIB.addReg(ARM::CPSR);
  MI->eraseFromParent();
}

void ARMLowOverheadLoops::Expand(MachineLoop *ML, MachineInstr *Start,
                                 MachineInstr *InsertPt,
                                 MachineInstr *Dec, MachineInstr *End,
                                 bool Revert) {

  auto ExpandLoopStart = [this](MachineLoop *ML, MachineInstr *Start,
                                MachineInstr *InsertPt) {
    MachineBasicBlock *MBB = InsertPt->getParent();
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
    bool FlagsAlreadySet = RevertLoopDec(Dec, true);
    RevertLoopEnd(End, FlagsAlreadySet);
  } else {
    Start = ExpandLoopStart(ML, Start, InsertPt);
    RemoveDeadBranch(Start);
    End = ExpandLoopEnd(ML, Dec, End);
    RemoveDeadBranch(End);
  }
}

bool ARMLowOverheadLoops::RevertNonLoops() {
  LLVM_DEBUG(dbgs() << "ARM Loops: Reverting any remaining pseudos...\n");
  bool Changed = false;

  for (auto &MBB : *MF) {
    SmallVector<MachineInstr*, 4> Starts;
    SmallVector<MachineInstr*, 4> Decs;
    SmallVector<MachineInstr*, 4> Ends;

    for (auto &I : MBB) {
      if (IsLoopStart(I))
        Starts.push_back(&I);
      else if (I.getOpcode() == ARM::t2LoopDec)
        Decs.push_back(&I);
      else if (I.getOpcode() == ARM::t2LoopEnd)
        Ends.push_back(&I);
    }

    if (Starts.empty() && Decs.empty() && Ends.empty())
      continue;

    Changed = true;

    for (auto *Start : Starts) {
      if (Start->getOpcode() == ARM::t2WhileLoopStart)
        RevertWhile(Start);
      else
        Start->eraseFromParent();
    }
    for (auto *Dec : Decs)
      RevertLoopDec(Dec);

    for (auto *End : Ends)
      RevertLoopEnd(End);
  }
  return Changed;
}

FunctionPass *llvm::createARMLowOverheadLoopsPass() {
  return new ARMLowOverheadLoops();
}
