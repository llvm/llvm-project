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
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineLoopUtils.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/ReachingDefAnalysis.h"
#include "llvm/MC/MCInstrDesc.h"

using namespace llvm;

#define DEBUG_TYPE "arm-low-overhead-loops"
#define ARM_LOW_OVERHEAD_LOOPS_NAME "ARM Low Overhead Loops pass"

namespace {

  struct PredicatedMI {
    MachineInstr *MI = nullptr;
    SetVector<MachineInstr*> Predicates;

  public:
    PredicatedMI(MachineInstr *I, SetVector<MachineInstr*> &Preds) :
    MI(I) {
      Predicates.insert(Preds.begin(), Preds.end());
    }
  };

  // Represent a VPT block, a list of instructions that begins with a VPST and
  // has a maximum of four proceeding instructions. All instructions within the
  // block are predicated upon the vpr and we allow instructions to define the
  // vpr within in the block too.
  class VPTBlock {
    std::unique_ptr<PredicatedMI> VPST;
    PredicatedMI *Divergent = nullptr;
    SmallVector<PredicatedMI, 4> Insts;

  public:
    VPTBlock(MachineInstr *MI, SetVector<MachineInstr*> &Preds) {
      VPST = std::make_unique<PredicatedMI>(MI, Preds);
    }

    void addInst(MachineInstr *MI, SetVector<MachineInstr*> &Preds) {
      LLVM_DEBUG(dbgs() << "ARM Loops: Adding predicated MI: " << *MI);
      if (!Divergent && !set_difference(Preds, VPST->Predicates).empty()) {
        Divergent = &Insts.back();
        LLVM_DEBUG(dbgs() << " - has divergent predicate: " << *Divergent->MI);
      }
      Insts.emplace_back(MI, Preds);
      assert(Insts.size() <= 4 && "Too many instructions in VPT block!");
    }

    // Have we found an instruction within the block which defines the vpr? If
    // so, not all the instructions in the block will have the same predicate.
    bool HasNonUniformPredicate() const {
      return Divergent != nullptr;
    }

    // Is the given instruction part of the predicate set controlling the entry
    // to the block.
    bool IsPredicatedOn(MachineInstr *MI) const {
      return VPST->Predicates.count(MI);
    }

    // Is the given instruction the only predicate which controls the entry to
    // the block.
    bool IsOnlyPredicatedOn(MachineInstr *MI) const {
      return IsPredicatedOn(MI) && VPST->Predicates.size() == 1;
    }

    unsigned size() const { return Insts.size(); }
    SmallVectorImpl<PredicatedMI> &getInsts() { return Insts; }
    MachineInstr *getVPST() const { return VPST->MI; }
    PredicatedMI *getDivergent() const { return Divergent; }
  };

  struct LowOverheadLoop {

    MachineLoop *ML = nullptr;
    MachineFunction *MF = nullptr;
    MachineInstr *InsertPt = nullptr;
    MachineInstr *Start = nullptr;
    MachineInstr *Dec = nullptr;
    MachineInstr *End = nullptr;
    MachineInstr *VCTP = nullptr;
    VPTBlock *CurrentBlock = nullptr;
    SetVector<MachineInstr*> CurrentPredicate;
    SmallVector<VPTBlock, 4> VPTBlocks;
    bool Revert = false;
    bool CannotTailPredicate = false;

    LowOverheadLoop(MachineLoop *ML) : ML(ML) {
      MF = ML->getHeader()->getParent();
    }

    bool RecordVPTBlocks(MachineInstr *MI);

    // If this is an MVE instruction, check that we know how to use tail
    // predication with it.
    void CheckTPValidity(MachineInstr *MI) {
      if (CannotTailPredicate)
        return;

      if (!RecordVPTBlocks(MI)) {
        CannotTailPredicate = true;
        return;
      }

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
      return !Revert && FoundAllComponents() && VCTP &&
             !CannotTailPredicate && ML->getNumBlocks() == 1;
    }

    // Is it safe to define LR with DLS/WLS?
    // LR can be defined if it is the operand to start, because it's the same
    // value, or if it's going to be equivalent to the operand to Start.
    MachineInstr *IsSafeToDefineLR(ReachingDefAnalysis *RDA);

    // Check the branch targets are within range and we satisfy our
    // restrictions.
    void CheckLegality(ARMBasicBlockUtils *BBUtils, ReachingDefAnalysis *RDA,
                       MachineLoopInfo *MLI);

    bool FoundAllComponents() const {
      return Start && Dec && End;
    }

    SmallVectorImpl<VPTBlock> &getVPTBlocks() { return VPTBlocks; }

    // Return the loop iteration count, or the number of elements if we're tail
    // predicating.
    MachineOperand &getCount() {
      return IsTailPredicationLegal() ?
        VCTP->getOperand(1) : Start->getOperand(0);
    }

    unsigned getStartOpcode() const {
      bool IsDo = Start->getOpcode() == ARM::t2DoLoopStart;
      if (!IsTailPredicationLegal())
        return IsDo ? ARM::t2DLS : ARM::t2WLS;

      return VCTPOpcodeToLSTP(VCTP->getOpcode(), IsDo);
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
    MachineLoopInfo           *MLI = nullptr;
    ReachingDefAnalysis       *RDA = nullptr;
    const ARMBaseInstrInfo    *TII = nullptr;
    MachineRegisterInfo       *MRI = nullptr;
    const TargetRegisterInfo  *TRI = nullptr;
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

    void RemoveLoopUpdate(LowOverheadLoop &LoLoop);

    void ConvertVPTBlocks(LowOverheadLoop &LoLoop);

    MachineInstr *ExpandLoopStart(LowOverheadLoop &LoLoop);

    void Expand(LowOverheadLoop &LoLoop);

  };
}

char ARMLowOverheadLoops::ID = 0;

INITIALIZE_PASS(ARMLowOverheadLoops, DEBUG_TYPE, ARM_LOW_OVERHEAD_LOOPS_NAME,
                false, false)

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
  if (auto *LRDef = RDA->getReachingMIDef(Start, ARM::LR))
    if (IsMoveLR(LRDef) && RDA->hasSameReachingDef(Start, LRDef, CountReg))
      return LRDef;

  // - Is there a (mov lr, Count) after Start? If so, and nothing else writes
  //   to Count after Start, we can insert at that mov.
  if (auto *LRDef = RDA->getLocalLiveOutMIDef(MBB, ARM::LR))
    if (IsMoveLR(LRDef) && RDA->hasSameReachingDef(Start, LRDef, CountReg))
      return LRDef;

  // We've found no suitable LR def and Start doesn't use LR directly. Can we
  // just define LR anyway?
  if (!RDA->isRegUsedAfter(Start, ARM::LR))
    return Start;

  return nullptr;
}

// Can we safely move 'From' to just before 'To'? To satisfy this, 'From' must
// not define a register that is used by any instructions, after and including,
// 'To'. These instructions also must not redefine any of Froms operands.
template<typename Iterator>
static bool IsSafeToMove(MachineInstr *From, MachineInstr *To, ReachingDefAnalysis *RDA) {
  SmallSet<int, 2> Defs;
  // First check that From would compute the same value if moved.
  for (auto &MO : From->operands()) {
    if (!MO.isReg() || MO.isUndef() || !MO.getReg())
      continue;
    if (MO.isDef())
      Defs.insert(MO.getReg());
    else if (!RDA->hasSameReachingDef(From, To, MO.getReg()))
      return false;
  }

  // Now walk checking that the rest of the instructions will compute the same
  // value.
  for (auto I = ++Iterator(From), E = Iterator(To); I != E; ++I) {
    for (auto &MO : I->operands())
      if (MO.isReg() && MO.getReg() && MO.isUse() && Defs.count(MO.getReg()))
        return false;
  }
  return true;
}

void LowOverheadLoop::CheckLegality(ARMBasicBlockUtils *BBUtils,
                                    ReachingDefAnalysis *RDA,
                                    MachineLoopInfo *MLI) {
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
    return;
  } else
    LLVM_DEBUG(dbgs() << "ARM Loops: Start insertion point: " << *InsertPt);

  if (!IsTailPredicationLegal()) {
    LLVM_DEBUG(dbgs() << "ARM Loops: Tail-predication is not valid.\n");
    return;
  }

  // All predication within the loop should be based on vctp. If the block
  // isn't predicated on entry, check whether the vctp is within the block
  // and that all other instructions are then predicated on it.
  for (auto &Block : VPTBlocks) {
    if (Block.IsPredicatedOn(VCTP))
      continue;
    if (!Block.HasNonUniformPredicate() || !isVCTP(Block.getDivergent()->MI)) {
      CannotTailPredicate = true;
      return;
    }
    SmallVectorImpl<PredicatedMI> &Insts = Block.getInsts();
    for (auto &PredMI : Insts) {
      if (PredMI.Predicates.count(VCTP) || isVCTP(PredMI.MI))
        continue;
      LLVM_DEBUG(dbgs() << "ARM Loops: Can't convert: " << *PredMI.MI
                        << " - which is predicated on:\n";
                        for (auto *MI : PredMI.Predicates)
                          dbgs() << "   - " << *MI;
                 );
      CannotTailPredicate = true;
      return;
    }
  }

  // For tail predication, we need to provide the number of elements, instead
  // of the iteration count, to the loop start instruction. The number of
  // elements is provided to the vctp instruction, so we need to check that
  // we can use this register at InsertPt.
  Register NumElements = VCTP->getOperand(1).getReg();

  // If the register is defined within loop, then we can't perform TP.
  // TODO: Check whether this is just a mov of a register that would be
  // available.
  if (RDA->getReachingDef(VCTP, NumElements) >= 0) {
    CannotTailPredicate = true;
    return;
  }

  // The element count register maybe defined after InsertPt, in which case we
  // need to try to move either InsertPt or the def so that the [w|d]lstp can
  // use the value.
  MachineBasicBlock *InsertBB = InsertPt->getParent();
  if (!RDA->isReachingDefLiveOut(InsertPt, NumElements)) {
    if (auto *ElemDef = RDA->getLocalLiveOutMIDef(InsertBB, NumElements)) {
      if (IsSafeToMove<MachineBasicBlock::reverse_iterator>(ElemDef, InsertPt, RDA)) {
        ElemDef->removeFromParent();
        InsertBB->insert(MachineBasicBlock::iterator(InsertPt), ElemDef);
        LLVM_DEBUG(dbgs() << "ARM Loops: Moved element count def: "
                   << *ElemDef);
      } else if (IsSafeToMove<MachineBasicBlock::iterator>(InsertPt, ElemDef, RDA)) {
        InsertPt->removeFromParent();
        InsertBB->insertAfter(MachineBasicBlock::iterator(ElemDef), InsertPt);
        LLVM_DEBUG(dbgs() << "ARM Loops: Moved start past: " << *ElemDef);
      } else {
        CannotTailPredicate = true;
        return;
      }
    }
  }

  // Especially in the case of while loops, InsertBB may not be the
  // preheader, so we need to check that the register isn't redefined
  // before entering the loop.
  auto CannotProvideElements = [&RDA](MachineBasicBlock *MBB,
                                      Register NumElements) {
    // NumElements is redefined in this block.
    if (RDA->getReachingDef(&MBB->back(), NumElements) >= 0)
      return true;

    // Don't continue searching up through multiple predecessors.
    if (MBB->pred_size() > 1)
      return true;

    return false;
  };

  // First, find the block that looks like the preheader.
  MachineBasicBlock *MBB = MLI->findLoopPreheader(ML, true);
  if (!MBB) {
    CannotTailPredicate = true;
    return;
  }

  // Then search backwards for a def, until we get to InsertBB.
  while (MBB != InsertBB) {
    CannotTailPredicate = CannotProvideElements(MBB, NumElements);
    if (CannotTailPredicate)
      return;
    MBB = *MBB->pred_begin();
  }

  LLVM_DEBUG(dbgs() << "ARM Loops: Will use tail predication.\n");
}

bool LowOverheadLoop::RecordVPTBlocks(MachineInstr* MI) {
  // Only support a single vctp.
  if (isVCTP(MI) && VCTP)
    return false;

  // Start a new vpt block when we discover a vpt.
  if (MI->getOpcode() == ARM::MVE_VPST) {
    VPTBlocks.emplace_back(MI, CurrentPredicate);
    CurrentBlock = &VPTBlocks.back();
    return true;
  }

  if (isVCTP(MI))
    VCTP = MI;

  unsigned VPROpNum = MI->getNumOperands() - 1;
  bool IsUse = false;
  if (MI->getOperand(VPROpNum).isReg() &&
      MI->getOperand(VPROpNum).getReg() == ARM::VPR &&
      MI->getOperand(VPROpNum).isUse()) {
    // If this instruction is predicated by VPR, it will be its last
    // operand.  Also check that it's only 'Then' predicated.
    if (!MI->getOperand(VPROpNum-1).isImm() ||
        MI->getOperand(VPROpNum-1).getImm() != ARMVCC::Then) {
      LLVM_DEBUG(dbgs() << "ARM Loops: Found unhandled predicate on: "
                 << *MI);
      return false;
    }
    CurrentBlock->addInst(MI, CurrentPredicate);
    IsUse = true;
  }

  bool IsDef = false;
  for (unsigned i = 0; i < MI->getNumOperands() - 1; ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg() || MO.getReg() != ARM::VPR)
      continue;

    if (MO.isDef()) {
      CurrentPredicate.insert(MI);
      IsDef = true;
    } else {
      LLVM_DEBUG(dbgs() << "ARM Loops: Found instruction using vpr: " << *MI);
      return false;
    }
  }

  // If we find a vpr def that is not already predicated on the vctp, we've
  // got disjoint predicates that may not be equivalent when we do the
  // conversion.
  if (IsDef && !IsUse && VCTP && !isVCTP(MI)) {
    LLVM_DEBUG(dbgs() << "ARM Loops: Found disjoint vpr def: " << *MI);
    return false;
  }

  return true;
}

bool ARMLowOverheadLoops::runOnMachineFunction(MachineFunction &mf) {
  const ARMSubtarget &ST = static_cast<const ARMSubtarget&>(mf.getSubtarget());
  if (!ST.hasLOB())
    return false;

  MF = &mf;
  LLVM_DEBUG(dbgs() << "ARM Loops on " << MF->getName() << " ------------- \n");

  MLI = &getAnalysis<MachineLoopInfo>();
  RDA = &getAnalysis<ReachingDefAnalysis>();
  MF->getProperties().set(MachineFunctionProperties::Property::TracksLiveness);
  MRI = &MF->getRegInfo();
  TII = static_cast<const ARMBaseInstrInfo*>(ST.getInstrInfo());
  TRI = ST.getRegisterInfo();
  BBUtils = std::unique_ptr<ARMBasicBlockUtils>(new ARMBasicBlockUtils(*MF));
  BBUtils->computeAllBlockSizes();
  BBUtils->adjustBBOffsetsAfter(&MF->front());

  bool Changed = false;
  for (auto ML : *MLI) {
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

  LLVM_DEBUG(dbgs() << "ARM Loops: Processing loop containing:\n";
             if (auto *Preheader = ML->getLoopPreheader())
               dbgs() << " - " << Preheader->getName() << "\n";
             else if (auto *Preheader = MLI->findLoopPreheader(ML))
               dbgs() << " - " << Preheader->getName() << "\n";
             for (auto *MBB : ML->getBlocks())
               dbgs() << " - " << MBB->getName() << "\n";
            );

  // Search the given block for a loop start instruction. If one isn't found,
  // and there's only one predecessor block, search that one too.
  std::function<MachineInstr*(MachineBasicBlock*)> SearchForStart =
    [&SearchForStart](MachineBasicBlock *MBB) -> MachineInstr* {
    for (auto &MI : *MBB) {
      if (isLoopStart(MI))
        return &MI;
    }
    if (MBB->pred_size() == 1)
      return SearchForStart(*MBB->pred_begin());
    return nullptr;
  };

  LowOverheadLoop LoLoop(ML);
  // Search the preheader for the start intrinsic.
  // FIXME: I don't see why we shouldn't be supporting multiple predecessors
  // with potentially multiple set.loop.iterations, so we need to enable this.
  if (auto *Preheader = ML->getLoopPreheader())
    LoLoop.Start = SearchForStart(Preheader);
  else if (auto *Preheader = MLI->findLoopPreheader(ML, true))
    LoLoop.Start = SearchForStart(Preheader);
  else
    return false;

  // Find the low-overhead loop components and decide whether or not to fall
  // back to a normal loop. Also look for a vctp instructions and decide
  // whether we can convert that predicate using tail predication.
  for (auto *MBB : reverse(ML->getBlocks())) {
    for (auto &MI : *MBB) {
      if (MI.getOpcode() == ARM::t2LoopDec)
        LoLoop.Dec = &MI;
      else if (MI.getOpcode() == ARM::t2LoopEnd)
        LoLoop.End = &MI;
      else if (isLoopStart(MI))
        LoLoop.Start = &MI;
      else if (MI.getDesc().isCall()) {
        // TODO: Though the call will require LE to execute again, does this
        // mean we should revert? Always executing LE hopefully should be
        // faster than performing a sub,cmp,br or even subs,br.
        LoLoop.Revert = true;
        LLVM_DEBUG(dbgs() << "ARM Loops: Found call.\n");
      } else {
        // Record VPR defs and build up their corresponding vpt blocks.
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

  LoLoop.CheckLegality(BBUtils.get(), RDA, MLI);
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
  unsigned Opc = LoLoop.getStartOpcode();
  MachineOperand &Count = LoLoop.getCount();

  MachineInstrBuilder MIB =
    BuildMI(*MBB, InsertPt, InsertPt->getDebugLoc(), TII->get(Opc));

  MIB.addDef(ARM::LR);
  MIB.add(Count);
  if (!IsDo)
    MIB.add(Start->getOperand(1));

  // When using tail-predication, try to delete the dead code that was used to
  // calculate the number of loop iterations.
  if (LoLoop.IsTailPredicationLegal()) {
    SmallVector<MachineInstr*, 4> Killed;
    SmallVector<MachineInstr*, 4> Dead;
    if (auto *Def = RDA->getReachingMIDef(Start,
                                          Start->getOperand(0).getReg())) {
      Killed.push_back(Def);

      while (!Killed.empty()) {
        MachineInstr *Def = Killed.back();
        Killed.pop_back();
        Dead.push_back(Def);
        for (auto &MO : Def->operands()) {
          if (!MO.isReg() || !MO.isKill())
            continue;

          MachineInstr *Kill = RDA->getReachingMIDef(Def, MO.getReg());
          if (Kill && RDA->getNumUses(Kill, MO.getReg()) == 1)
            Killed.push_back(Kill);
        }
      }
      for (auto *MI : Dead)
        MI->eraseFromParent();
    }
  }

  // If we're inserting at a mov lr, then remove it as it's redundant.
  if (InsertPt != Start)
    InsertPt->eraseFromParent();
  Start->eraseFromParent();
  LLVM_DEBUG(dbgs() << "ARM Loops: Inserted start: " << *MIB);
  return &*MIB;
}

// Goal is to optimise and clean-up these loops:
//
//   vector.body:
//     renamable $vpr = MVE_VCTP32 renamable $r3, 0, $noreg
//     renamable $r3, dead $cpsr = tSUBi8 killed renamable $r3(tied-def 0), 4
//     ..
//     $lr = MVE_DLSTP_32 renamable $r3
//
// The SUB is the old update of the loop iteration count expression, which
// is no longer needed. This sub is removed when the element count, which is in
// r3 in this example, is defined by an instruction in the loop, and it has
// no uses.
//
void ARMLowOverheadLoops::RemoveLoopUpdate(LowOverheadLoop &LoLoop) {
  Register ElemCount = LoLoop.VCTP->getOperand(1).getReg();
  MachineInstr *LastInstrInBlock = &LoLoop.VCTP->getParent()->back();

  LLVM_DEBUG(dbgs() << "ARM Loops: Trying to remove loop update stmt\n");

  if (LoLoop.ML->getNumBlocks() != 1) {
    LLVM_DEBUG(dbgs() << "ARM Loops: single block loop expected\n");
    return;
  }

  LLVM_DEBUG(dbgs() << "ARM Loops: Analyzing MO: ";
             LoLoop.VCTP->getOperand(1).dump());

  // Find the definition we are interested in removing, if there is one.
  MachineInstr *Def = RDA->getReachingMIDef(LastInstrInBlock, ElemCount);
  if (!Def)
    return;

  // Bail if we define CPSR and it is not dead
  if (!Def->registerDefIsDead(ARM::CPSR, TRI)) {
    LLVM_DEBUG(dbgs() << "ARM Loops: CPSR is not dead\n");
    return;
  }

  // Bail if elemcount is used in exit blocks, i.e. if it is live-in.
  if (isRegLiveInExitBlocks(LoLoop.ML, ElemCount)) {
    LLVM_DEBUG(dbgs() << "ARM Loops: Elemcount is live-out, can't remove stmt\n");
    return;
  }

  // Bail if there are uses after this Def in the block.
  SmallVector<MachineInstr*, 4> Uses;
  RDA->getReachingLocalUses(Def, ElemCount, Uses);
  if (Uses.size()) {
    LLVM_DEBUG(dbgs() << "ARM Loops: Local uses in block, can't remove stmt\n");
    return;
  }

  Uses.clear();
  RDA->getAllInstWithUseBefore(Def, ElemCount, Uses);

  // Remove Def if there are no uses, or if the only use is the VCTP
  // instruction.
  if (!Uses.size() || (Uses.size() == 1 && Uses[0] == LoLoop.VCTP)) {
    LLVM_DEBUG(dbgs() << "ARM Loops: Removing loop update instruction: ";
               Def->dump());
    Def->eraseFromParent();
  }
}

void ARMLowOverheadLoops::ConvertVPTBlocks(LowOverheadLoop &LoLoop) {
  auto RemovePredicate = [](MachineInstr *MI) {
    LLVM_DEBUG(dbgs() << "ARM Loops: Removing predicate from: " << *MI);
    unsigned OpNum = MI->getNumOperands() - 1;
    assert(MI->getOperand(OpNum-1).getImm() == ARMVCC::Then &&
           "Expected Then predicate!");
    MI->getOperand(OpNum-1).setImm(ARMVCC::None);
    MI->getOperand(OpNum).setReg(0);
  };

  // There are a few scenarios which we have to fix up:
  // 1) A VPT block with is only predicated by the vctp and has no internal vpr
  //    defs.
  // 2) A VPT block which is only predicated by the vctp but has an internal
  //    vpr def.
  // 3) A VPT block which is predicated upon the vctp as well as another vpr
  //    def.
  // 4) A VPT block which is not predicated upon a vctp, but contains it and
  //    all instructions within the block are predicated upon in.

  for (auto &Block : LoLoop.getVPTBlocks()) {
    SmallVectorImpl<PredicatedMI> &Insts = Block.getInsts();
    if (Block.HasNonUniformPredicate()) {
      PredicatedMI *Divergent = Block.getDivergent();
      if (isVCTP(Divergent->MI)) {
        // The vctp will be removed, so the size of the vpt block needs to be
        // modified.
        uint64_t Size = getARMVPTBlockMask(Block.size() - 1);
        Block.getVPST()->getOperand(0).setImm(Size);
        LLVM_DEBUG(dbgs() << "ARM Loops: Modified VPT block mask.\n");
      } else if (Block.IsOnlyPredicatedOn(LoLoop.VCTP)) {
        // The VPT block has a non-uniform predicate but it's entry is guarded
        // only by a vctp, which means we:
        // - Need to remove the original vpst.
        // - Then need to unpredicate any following instructions, until
        //   we come across the divergent vpr def.
        // - Insert a new vpst to predicate the instruction(s) that following
        //   the divergent vpr def.
        // TODO: We could be producing more VPT blocks than necessary and could
        // fold the newly created one into a proceeding one.
        for (auto I = ++MachineBasicBlock::iterator(Block.getVPST()),
             E = ++MachineBasicBlock::iterator(Divergent->MI); I != E; ++I)
          RemovePredicate(&*I);

        unsigned Size = 0;
        auto E = MachineBasicBlock::reverse_iterator(Divergent->MI);
        auto I = MachineBasicBlock::reverse_iterator(Insts.back().MI);
        MachineInstr *InsertAt = nullptr;
        while (I != E) {
          InsertAt = &*I;
          ++Size;
          ++I;
        }
        MachineInstrBuilder MIB = BuildMI(*InsertAt->getParent(), InsertAt,
                                          InsertAt->getDebugLoc(),
                                          TII->get(ARM::MVE_VPST));
        MIB.addImm(getARMVPTBlockMask(Size));
        LLVM_DEBUG(dbgs() << "ARM Loops: Removing VPST: " << *Block.getVPST());
        LLVM_DEBUG(dbgs() << "ARM Loops: Created VPST: " << *MIB);
        Block.getVPST()->eraseFromParent();
      }
    } else if (Block.IsOnlyPredicatedOn(LoLoop.VCTP)) {
      // A vpt block which is only predicated upon vctp and has no internal vpr
      // defs:
      // - Remove vpst.
      // - Unpredicate the remaining instructions.
      LLVM_DEBUG(dbgs() << "ARM Loops: Removing VPST: " << *Block.getVPST());
      Block.getVPST()->eraseFromParent();
      for (auto &PredMI : Insts)
        RemovePredicate(PredMI.MI);
    }
  }

  LLVM_DEBUG(dbgs() << "ARM Loops: Removing VCTP: " << *LoLoop.VCTP);
  LoLoop.VCTP->eraseFromParent();
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
    if (LoLoop.IsTailPredicationLegal()) {
      RemoveLoopUpdate(LoLoop);
      ConvertVPTBlocks(LoLoop);
    }
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
      if (isLoopStart(I))
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
