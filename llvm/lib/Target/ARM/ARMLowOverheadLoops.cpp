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
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/ReachingDefAnalysis.h"
#include "llvm/MC/MCInstrDesc.h"

using namespace llvm;

#define DEBUG_TYPE "arm-low-overhead-loops"
#define ARM_LOW_OVERHEAD_LOOPS_NAME "ARM Low Overhead Loops pass"

namespace {

  struct LowOverheadLoop {

    MachineLoop *ML = nullptr;
    MachineFunction *MF = nullptr;
    MachineInstr *InsertPt = nullptr;
    MachineInstr *Start = nullptr;
    MachineInstr *Dec = nullptr;
    MachineInstr *End = nullptr;
    MachineInstr *VCTP = nullptr;
    SmallVector<MachineInstr*, 4> VPTUsers;
    bool Revert = false;
    bool FoundOneVCTP = false;
    bool CannotTailPredicate = false;

    LowOverheadLoop(MachineLoop *ML) : ML(ML) {
      MF = ML->getHeader()->getParent();
    }

    // For now, only support one vctp instruction. If we find multiple then
    // we shouldn't perform tail predication.
    void addVCTP(MachineInstr *MI) {
      if (!VCTP) {
        VCTP = MI;
        FoundOneVCTP = true;
      } else
        FoundOneVCTP = false;
    }

    // Check that nothing else is writing to VPR and record any insts
    // reading the VPR.
    void ScanForVPR(MachineInstr *MI) {
      for (auto &MO : MI->operands()) {
        if (!MO.isReg() || MO.getReg() != ARM::VPR)
          continue;
        if (MO.isUse())
          VPTUsers.push_back(MI);
        if (MO.isDef()) {
          CannotTailPredicate = true;
          break;
        }
      }
    }

    // If this is an MVE instruction, check that we know how to use tail
    // predication with it.
    void CheckTPValidity(MachineInstr *MI) {
      if (CannotTailPredicate)
        return;

      const MCInstrDesc &MCID = MI->getDesc();
      uint64_t Flags = MCID.TSFlags;
      if ((Flags & ARMII::DomainMask) != ARMII::DomainMVE)
        return;

      if ((Flags & ARMII::ValidForTailPredication) == 0) {
        LLVM_DEBUG(dbgs() << "ARM Loops: Can't tail predicate: " << *MI);
        CannotTailPredicate = true;
      }
    }

    bool IsTailPredicationLegal() const {
      // For now, let's keep things really simple and only support a single
      // block for tail predication.
      return !Revert && FoundAllComponents() && FoundOneVCTP &&
             !CannotTailPredicate && ML->getNumBlocks() == 1;
    }

    // Is it safe to define LR with DLS/WLS?
    // LR can be defined if it is the operand to start, because it's the same
    // value, or if it's going to be equivalent to the operand to Start.
    MachineInstr *IsSafeToDefineLR(ReachingDefAnalysis *RDA);

    // Check the branch targets are within range and we satisfy our
    // restrictions.
    void CheckLegality(ARMBasicBlockUtils *BBUtils, ReachingDefAnalysis *RDA);

    bool FoundAllComponents() const {
      return Start && Dec && End;
    }

    void dump() const {
      if (Start) dbgs() << "ARM Loops: Found Loop Start: " << *Start;
      if (Dec) dbgs() << "ARM Loops: Found Loop Dec: " << *Dec;
      if (End) dbgs() << "ARM Loops: Found Loop End: " << *End;
      if (VCTP) dbgs() << "ARM Loops: Found VCTP: " << *VCTP;
      if (!FoundAllComponents())
        dbgs() << "ARM Loops: Not a low-overhead loop.\n";
      else if (!(Start && Dec && End))
        dbgs() << "ARM Loops: Failed to find all loop components.\n";
    }
  };

  class ARMLowOverheadLoops : public MachineFunctionPass {
    MachineFunction           *MF = nullptr;
    ReachingDefAnalysis       *RDA = nullptr;
    const ARMBaseInstrInfo    *TII = nullptr;
    MachineRegisterInfo       *MRI = nullptr;
    std::unique_ptr<ARMBasicBlockUtils> BBUtils = nullptr;

  public:
    static char ID;

    ARMLowOverheadLoops() : MachineFunctionPass(ID) { }

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.setPreservesCFG();
      AU.addRequired<MachineLoopInfo>();
      AU.addRequired<ReachingDefAnalysis>();
      MachineFunctionPass::getAnalysisUsage(AU);
    }

    bool runOnMachineFunction(MachineFunction &MF) override;

    MachineFunctionProperties getRequiredProperties() const override {
      return MachineFunctionProperties().set(
          MachineFunctionProperties::Property::NoVRegs).set(
          MachineFunctionProperties::Property::TracksLiveness);
    }

    StringRef getPassName() const override {
      return ARM_LOW_OVERHEAD_LOOPS_NAME;
    }

  private:
    bool ProcessLoop(MachineLoop *ML);

    bool RevertNonLoops();

    void RevertWhile(MachineInstr *MI) const;

    bool RevertLoopDec(MachineInstr *MI, bool AllowFlags = false) const;

    void RevertLoopEnd(MachineInstr *MI, bool SkipCmp = false) const;

    void RemoveVPTBlocks(LowOverheadLoop &LoLoop);

    MachineInstr *ExpandLoopStart(LowOverheadLoop &LoLoop);

    void Expand(LowOverheadLoop &LoLoop);

  };
}

char ARMLowOverheadLoops::ID = 0;

INITIALIZE_PASS(ARMLowOverheadLoops, DEBUG_TYPE, ARM_LOW_OVERHEAD_LOOPS_NAME,
                false, false)

static bool IsLoopStart(MachineInstr &MI) {
  return MI.getOpcode() == ARM::t2DoLoopStart ||
         MI.getOpcode() == ARM::t2WhileLoopStart;
}

static bool IsVCTP(MachineInstr *MI) {
  switch (MI->getOpcode()) {
  default:
    break;
  case ARM::MVE_VCTP8:
  case ARM::MVE_VCTP16:
  case ARM::MVE_VCTP32:
  case ARM::MVE_VCTP64:
    return true;
  }
  return false;
}

MachineInstr *LowOverheadLoop::IsSafeToDefineLR(ReachingDefAnalysis *RDA) {
  // We can define LR because LR already contains the same value.
  if (Start->getOperand(0).getReg() == ARM::LR)
    return Start;

  unsigned CountReg = Start->getOperand(0).getReg();
  auto IsMoveLR = [&CountReg](MachineInstr *MI) {
    return MI->getOpcode() == ARM::tMOVr &&
           MI->getOperand(0).getReg() == ARM::LR &&
           MI->getOperand(1).getReg() == CountReg &&
           MI->getOperand(2).getImm() == ARMCC::AL;
   };

  MachineBasicBlock *MBB = Start->getParent();

  // Find an insertion point:
  // - Is there a (mov lr, Count) before Start? If so, and nothing else writes
  //   to Count before Start, we can insert at that mov.
  // - Is there a (mov lr, Count) after Start? If so, and nothing else writes
  //   to Count after Start, we can insert at that mov.
  if (auto *LRDef = RDA->getReachingMIDef(&MBB->back(), ARM::LR)) {
    if (IsMoveLR(LRDef) && RDA->hasSameReachingDef(Start, LRDef, CountReg))
      return LRDef;
  }

  // We've found no suitable LR def and Start doesn't use LR directly. Can we
  // just define LR anyway?
  if (!RDA->isRegUsedAfter(Start, ARM::LR))
    return Start;

  return nullptr;
}

void LowOverheadLoop::CheckLegality(ARMBasicBlockUtils *BBUtils,
                                    ReachingDefAnalysis *RDA) {
  if (Revert)
    return;

  if (!End->getOperand(1).isMBB())
    report_fatal_error("Expected LoopEnd to target basic block");

  // TODO Maybe there's cases where the target doesn't have to be the header,
  // but for now be safe and revert.
  if (End->getOperand(1).getMBB() != ML->getHeader()) {
    LLVM_DEBUG(dbgs() << "ARM Loops: LoopEnd is not targetting header.\n");
    Revert = true;
    return;
  }

  // The WLS and LE instructions have 12-bits for the label offset. WLS
  // requires a positive offset, while LE uses negative.
  if (BBUtils->getOffsetOf(End) < BBUtils->getOffsetOf(ML->getHeader()) ||
      !BBUtils->isBBInRange(End, ML->getHeader(), 4094)) {
    LLVM_DEBUG(dbgs() << "ARM Loops: LE offset is out-of-range\n");
    Revert = true;
    return;
  }

  if (Start->getOpcode() == ARM::t2WhileLoopStart &&
      (BBUtils->getOffsetOf(Start) >
       BBUtils->getOffsetOf(Start->getOperand(1).getMBB()) ||
       !BBUtils->isBBInRange(Start, Start->getOperand(1).getMBB(), 4094))) {
    LLVM_DEBUG(dbgs() << "ARM Loops: WLS offset is out-of-range!\n");
    Revert = true;
    return;
  }

  InsertPt = Revert ? nullptr : IsSafeToDefineLR(RDA);
  if (!InsertPt) {
    LLVM_DEBUG(dbgs() << "ARM Loops: Unable to find safe insertion point.\n");
    Revert = true;
  } else
    LLVM_DEBUG(dbgs() << "ARM Loops: Start insertion point: " << *InsertPt);

  LLVM_DEBUG(if (IsTailPredicationLegal()) {
               dbgs() << "ARM Loops: Will use tail predication to convert:\n";
               for (auto *MI : VPTUsers)
                 dbgs() << " - " << *MI;
             });
}

bool ARMLowOverheadLoops::runOnMachineFunction(MachineFunction &mf) {
  const ARMSubtarget &ST = static_cast<const ARMSubtarget&>(mf.getSubtarget());
  if (!ST.hasLOB())
    return false;

  MF = &mf;
  LLVM_DEBUG(dbgs() << "ARM Loops on " << MF->getName() << " ------------- \n");

  auto &MLI = getAnalysis<MachineLoopInfo>();
  RDA = &getAnalysis<ReachingDefAnalysis>();
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

  LowOverheadLoop LoLoop(ML);
  // Search the preheader for the start intrinsic, or look through the
  // predecessors of the header to find exactly one set.iterations intrinsic.
  // FIXME: I don't see why we shouldn't be supporting multiple predecessors
  // with potentially multiple set.loop.iterations, so we need to enable this.
  if (auto *Preheader = ML->getLoopPreheader())
    LoLoop.Start = SearchForStart(Preheader);
  else {
    LLVM_DEBUG(dbgs() << "ARM Loops: Failed to find loop preheader!\n"
               << " - Performing manual predecessor search.\n");
    MachineBasicBlock *Pred = nullptr;
    for (auto *MBB : ML->getHeader()->predecessors()) {
      if (!ML->contains(MBB)) {
        if (Pred) {
          LLVM_DEBUG(dbgs() << " - Found multiple out-of-loop preds.\n");
          LoLoop.Start = nullptr;
          break;
        }
        Pred = MBB;
        LoLoop.Start = SearchForStart(MBB);
      }
    }
  }

  // Find the low-overhead loop components and decide whether or not to fall
  // back to a normal loop. Also look for a vctp instructions and decide
  // whether we can convert that predicate using tail predication.
  for (auto *MBB : reverse(ML->getBlocks())) {
    for (auto &MI : *MBB) {
      if (MI.getOpcode() == ARM::t2LoopDec)
        LoLoop.Dec = &MI;
      else if (MI.getOpcode() == ARM::t2LoopEnd)
        LoLoop.End = &MI;
      else if (IsLoopStart(MI))
        LoLoop.Start = &MI;
      else if (IsVCTP(&MI))
        LoLoop.addVCTP(&MI);
      else if (MI.getDesc().isCall()) {
        // TODO: Though the call will require LE to execute again, does this
        // mean we should revert? Always executing LE hopefully should be
        // faster than performing a sub,cmp,br or even subs,br.
        LoLoop.Revert = true;
        LLVM_DEBUG(dbgs() << "ARM Loops: Found call.\n");
      } else {
        // Once we've found a vctp, record the users of vpr and check there's
        // no more vpr defs.
        if (LoLoop.FoundOneVCTP)
          LoLoop.ScanForVPR(&MI);
        // Check we know how to tail predicate any mve instructions.
        LoLoop.CheckTPValidity(&MI);
      }

      // We need to ensure that LR is not used or defined inbetween LoopDec and
      // LoopEnd.
      if (!LoLoop.Dec || LoLoop.End || LoLoop.Revert)
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
          LoLoop.Revert = true;
          break;
        }
      }
    }
  }

  LLVM_DEBUG(LoLoop.dump());
  if (!LoLoop.FoundAllComponents())
    return false;

  LoLoop.CheckLegality(BBUtils.get(), RDA);
  Expand(LoLoop);
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
                                        bool SetFlags) const {
  LLVM_DEBUG(dbgs() << "ARM Loops: Reverting to sub: " << *MI);
  MachineBasicBlock *MBB = MI->getParent();

  // If nothing defines CPSR between LoopDec and LoopEnd, use a t2SUBS.
  if (SetFlags &&
      (RDA->isRegUsedAfter(MI, ARM::CPSR) ||
       !RDA->hasSameReachingDef(MI, &MBB->back(), ARM::CPSR)))
      SetFlags = false;

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

MachineInstr* ARMLowOverheadLoops::ExpandLoopStart(LowOverheadLoop &LoLoop) {
  MachineInstr *InsertPt = LoLoop.InsertPt;
  MachineInstr *Start = LoLoop.Start;
  MachineBasicBlock *MBB = InsertPt->getParent();
  bool IsDo = Start->getOpcode() == ARM::t2DoLoopStart;
  unsigned Opc = 0;

  if (!LoLoop.IsTailPredicationLegal())
    Opc = IsDo ? ARM::t2DLS : ARM::t2WLS;
  else {
    switch (LoLoop.VCTP->getOpcode()) {
    case ARM::MVE_VCTP8:
      Opc = IsDo ? ARM::MVE_DLSTP_8 : ARM::MVE_WLSTP_8;
      break;
    case ARM::MVE_VCTP16:
      Opc = IsDo ? ARM::MVE_DLSTP_16 : ARM::MVE_WLSTP_16;
      break;
    case ARM::MVE_VCTP32:
      Opc = IsDo ? ARM::MVE_DLSTP_32 : ARM::MVE_WLSTP_32;
      break;
    case ARM::MVE_VCTP64:
      Opc = IsDo ? ARM::MVE_DLSTP_64 : ARM::MVE_WLSTP_64;
      break;
    }
  }

  MachineInstrBuilder MIB =
    BuildMI(*MBB, InsertPt, InsertPt->getDebugLoc(), TII->get(Opc));

  MIB.addDef(ARM::LR);
  MIB.add(Start->getOperand(0));
  if (!IsDo)
    MIB.add(Start->getOperand(1));

  if (InsertPt != Start)
    InsertPt->eraseFromParent();
  Start->eraseFromParent();
  LLVM_DEBUG(dbgs() << "ARM Loops: Inserted start: " << *MIB);
  return &*MIB;
}

void ARMLowOverheadLoops::RemoveVPTBlocks(LowOverheadLoop &LoLoop) {
  LLVM_DEBUG(dbgs() << "ARM Loops: Removing VCTP: " << *LoLoop.VCTP);
  LoLoop.VCTP->eraseFromParent();

  for (auto *MI : LoLoop.VPTUsers) {
    if (MI->getOpcode() == ARM::MVE_VPST) {
      LLVM_DEBUG(dbgs() << "ARM Loops: Removing VPST: " << *MI);
      MI->eraseFromParent();
    } else {
      unsigned OpNum = MI->getNumOperands() - 1;
      assert((MI->getOperand(OpNum).isReg() &&
              MI->getOperand(OpNum).getReg() == ARM::VPR) &&
             "Expected VPR");
      assert((MI->getOperand(OpNum-1).isImm() &&
              MI->getOperand(OpNum-1).getImm() == ARMVCC::Then) &&
             "Expected Then predicate");
      MI->getOperand(OpNum-1).setImm(ARMVCC::None);
      MI->getOperand(OpNum).setReg(0);
      LLVM_DEBUG(dbgs() << "ARM Loops: Removed predicate from: " << *MI);
    }
  }
}

void ARMLowOverheadLoops::Expand(LowOverheadLoop &LoLoop) {

  // Combine the LoopDec and LoopEnd instructions into LE(TP).
  auto ExpandLoopEnd = [this](LowOverheadLoop &LoLoop) {
    MachineInstr *End = LoLoop.End;
    MachineBasicBlock *MBB = End->getParent();
    unsigned Opc = LoLoop.IsTailPredicationLegal() ?
      ARM::MVE_LETP : ARM::t2LEUpdate;
    MachineInstrBuilder MIB = BuildMI(*MBB, End, End->getDebugLoc(),
                                      TII->get(Opc));
    MIB.addDef(ARM::LR);
    MIB.add(End->getOperand(0));
    MIB.add(End->getOperand(1));
    LLVM_DEBUG(dbgs() << "ARM Loops: Inserted LE: " << *MIB);

    LoLoop.End->eraseFromParent();
    LoLoop.Dec->eraseFromParent();
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

  if (LoLoop.Revert) {
    if (LoLoop.Start->getOpcode() == ARM::t2WhileLoopStart)
      RevertWhile(LoLoop.Start);
    else
      LoLoop.Start->eraseFromParent();
    bool FlagsAlreadySet = RevertLoopDec(LoLoop.Dec, true);
    RevertLoopEnd(LoLoop.End, FlagsAlreadySet);
  } else {
    LoLoop.Start = ExpandLoopStart(LoLoop);
    RemoveDeadBranch(LoLoop.Start);
    LoLoop.End = ExpandLoopEnd(LoLoop);
    RemoveDeadBranch(LoLoop.End);
    if (LoLoop.IsTailPredicationLegal())
      RemoveVPTBlocks(LoLoop);
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
