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
/// In addition to this, we also look for the presence of the VCTP instruction,
/// which determines whether we can generated the tail-predicated low-overhead
/// loop form.
///
/// Assumptions and Dependencies:
/// Low-overhead loops are constructed and executed using a setup instruction:
/// DLS, WLS, DLSTP or WLSTP and an instruction that loops back: LE or LETP.
/// WLS(TP) and LE(TP) are branching instructions with a (large) limited range
/// but fixed polarity: WLS can only branch forwards and LE can only branch
/// backwards. These restrictions mean that this pass is dependent upon block
/// layout and block sizes, which is why it's the last pass to run. The same is
/// true for ConstantIslands, but this pass does not increase the size of the
/// basic blocks, nor does it change the CFG. Instructions are mainly removed
/// during the transform and pseudo instructions are replaced by real ones. In
/// some cases, when we have to revert to a 'normal' loop, we have to introduce
/// multiple instructions for a single pseudo (see RevertWhile and
/// RevertLoopEnd). To handle this situation, t2WhileLoopStart and t2LoopEnd
/// are defined to be as large as this maximum sequence of replacement
/// instructions.
///
//===----------------------------------------------------------------------===//

#include "ARM.h"
#include "ARMBaseInstrInfo.h"
#include "ARMBaseRegisterInfo.h"
#include "ARMBasicBlockInfo.h"
#include "ARMSubtarget.h"
#include "Thumb2InstrInfo.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/CodeGen/LivePhysRegs.h"
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

  class PostOrderLoopTraversal {
    MachineLoop &ML;
    MachineLoopInfo &MLI;
    SmallPtrSet<MachineBasicBlock*, 4> Visited;
    SmallVector<MachineBasicBlock*, 4> Order;

  public:
    PostOrderLoopTraversal(MachineLoop &ML, MachineLoopInfo &MLI)
      : ML(ML), MLI(MLI) { }

    const SmallVectorImpl<MachineBasicBlock*> &getOrder() const {
      return Order;
    }

    // Visit all the blocks within the loop, as well as exit blocks and any
    // blocks properly dominating the header.
    void ProcessLoop() {
      std::function<void(MachineBasicBlock*)> Search = [this, &Search]
        (MachineBasicBlock *MBB) -> void {
        if (Visited.count(MBB))
          return;

        Visited.insert(MBB);
        for (auto *Succ : MBB->successors()) {
          if (!ML.contains(Succ))
            continue;
          Search(Succ);
        }
        Order.push_back(MBB);
      };

      // Insert exit blocks.
      SmallVector<MachineBasicBlock*, 2> ExitBlocks;
      ML.getExitBlocks(ExitBlocks);
      for (auto *MBB : ExitBlocks)
        Order.push_back(MBB);

      // Then add the loop body.
      Search(ML.getHeader());

      // Then try the preheader and its predecessors.
      std::function<void(MachineBasicBlock*)> GetPredecessor =
        [this, &GetPredecessor] (MachineBasicBlock *MBB) -> void {
        Order.push_back(MBB);
        if (MBB->pred_size() == 1)
          GetPredecessor(*MBB->pred_begin());
      };

      if (auto *Preheader = ML.getLoopPreheader())
        GetPredecessor(Preheader);
      else if (auto *Preheader = MLI.findLoopPreheader(&ML, true))
        GetPredecessor(Preheader);
    }
  };

  struct PredicatedMI {
    MachineInstr *MI = nullptr;
    SetVector<MachineInstr*> Predicates;

  public:
    PredicatedMI(MachineInstr *I, SetVector<MachineInstr*> &Preds) :
      MI(I) { Predicates.insert(Preds.begin(), Preds.end()); }
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

    MachineLoop &ML;
    MachineLoopInfo &MLI;
    ReachingDefAnalysis &RDA;
    const TargetRegisterInfo &TRI;
    MachineFunction *MF = nullptr;
    MachineInstr *InsertPt = nullptr;
    MachineInstr *Start = nullptr;
    MachineInstr *Dec = nullptr;
    MachineInstr *End = nullptr;
    MachineInstr *VCTP = nullptr;
    VPTBlock *CurrentBlock = nullptr;
    SetVector<MachineInstr*> CurrentPredicate;
    SmallVector<VPTBlock, 4> VPTBlocks;
    SmallPtrSet<MachineInstr*, 4> ToRemove;
    bool Revert = false;
    bool CannotTailPredicate = false;

    LowOverheadLoop(MachineLoop &ML, MachineLoopInfo &MLI,
                    ReachingDefAnalysis &RDA, const TargetRegisterInfo &TRI)
      : ML(ML), MLI(MLI), RDA(RDA), TRI(TRI) {
      MF = ML.getHeader()->getParent();
    }

    // If this is an MVE instruction, check that we know how to use tail
    // predication with it. Record VPT blocks and return whether the
    // instruction is valid for tail predication.
    bool ValidateMVEInst(MachineInstr *MI);

    void AnalyseMVEInst(MachineInstr *MI) {
      CannotTailPredicate = !ValidateMVEInst(MI);
    }

    bool IsTailPredicationLegal() const {
      // For now, let's keep things really simple and only support a single
      // block for tail predication.
      return !Revert && FoundAllComponents() && VCTP &&
             !CannotTailPredicate && ML.getNumBlocks() == 1;
    }

    // Check that the predication in the loop will be equivalent once we
    // perform the conversion. Also ensure that we can provide the number
    // of elements to the loop start instruction.
    bool ValidateTailPredicate(MachineInstr *StartInsertPt);

    // Check that any values available outside of the loop will be the same
    // after tail predication conversion.
    bool ValidateLiveOuts() const;

    // Is it safe to define LR with DLS/WLS?
    // LR can be defined if it is the operand to start, because it's the same
    // value, or if it's going to be equivalent to the operand to Start.
    MachineInstr *isSafeToDefineLR();

    // Check the branch targets are within range and we satisfy our
    // restrictions.
    void CheckLegality(ARMBasicBlockUtils *BBUtils);

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

    bool RevertLoopDec(MachineInstr *MI) const;

    void RevertLoopEnd(MachineInstr *MI, bool SkipCmp = false) const;

    void ConvertVPTBlocks(LowOverheadLoop &LoLoop);

    MachineInstr *ExpandLoopStart(LowOverheadLoop &LoLoop);

    void Expand(LowOverheadLoop &LoLoop);

    void IterationCountDCE(LowOverheadLoop &LoLoop);
  };
}

char ARMLowOverheadLoops::ID = 0;

INITIALIZE_PASS(ARMLowOverheadLoops, DEBUG_TYPE, ARM_LOW_OVERHEAD_LOOPS_NAME,
                false, false)

MachineInstr *LowOverheadLoop::isSafeToDefineLR() {
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
  if (auto *LRDef = RDA.getUniqueReachingMIDef(Start, ARM::LR))
    if (IsMoveLR(LRDef) && RDA.hasSameReachingDef(Start, LRDef, CountReg))
      return LRDef;

  // - Is there a (mov lr, Count) after Start? If so, and nothing else writes
  //   to Count after Start, we can insert at that mov.
  if (auto *LRDef = RDA.getLocalLiveOutMIDef(MBB, ARM::LR))
    if (IsMoveLR(LRDef) && RDA.hasSameReachingDef(Start, LRDef, CountReg))
      return LRDef;

  // We've found no suitable LR def and Start doesn't use LR directly. Can we
  // just define LR anyway?
  return RDA.isSafeToDefRegAt(Start, ARM::LR) ? Start : nullptr;
}

bool LowOverheadLoop::ValidateTailPredicate(MachineInstr *StartInsertPt) {
  assert(VCTP && "VCTP instruction expected but is not set");
  // All predication within the loop should be based on vctp. If the block
  // isn't predicated on entry, check whether the vctp is within the block
  // and that all other instructions are then predicated on it.
  for (auto &Block : VPTBlocks) {
    if (Block.IsPredicatedOn(VCTP))
      continue;
    if (!Block.HasNonUniformPredicate() || !isVCTP(Block.getDivergent()->MI)) {
      LLVM_DEBUG(dbgs() << "ARM Loops: Found unsupported diverging predicate: "
                 << *Block.getDivergent()->MI);
      return false;
    }
    SmallVectorImpl<PredicatedMI> &Insts = Block.getInsts();
    for (auto &PredMI : Insts) {
      if (PredMI.Predicates.count(VCTP) || isVCTP(PredMI.MI))
        continue;
      LLVM_DEBUG(dbgs() << "ARM Loops: Can't convert: " << *PredMI.MI
                 << " - which is predicated on:\n";
                 for (auto *MI : PredMI.Predicates)
                   dbgs() << "   - " << *MI);
      return false;
    }
  }

  if (!ValidateLiveOuts())
    return false;

  // For tail predication, we need to provide the number of elements, instead
  // of the iteration count, to the loop start instruction. The number of
  // elements is provided to the vctp instruction, so we need to check that
  // we can use this register at InsertPt.
  Register NumElements = VCTP->getOperand(1).getReg();

  // If the register is defined within loop, then we can't perform TP.
  // TODO: Check whether this is just a mov of a register that would be
  // available.
  if (RDA.hasLocalDefBefore(VCTP, NumElements)) {
    LLVM_DEBUG(dbgs() << "ARM Loops: VCTP operand is defined in the loop.\n");
    return false;
  }

  // The element count register maybe defined after InsertPt, in which case we
  // need to try to move either InsertPt or the def so that the [w|d]lstp can
  // use the value.
  // TODO: On failing to move an instruction, check if the count is provided by
  // a mov and whether we can use the mov operand directly.
  MachineBasicBlock *InsertBB = StartInsertPt->getParent();
  if (!RDA.isReachingDefLiveOut(StartInsertPt, NumElements)) {
    if (auto *ElemDef = RDA.getLocalLiveOutMIDef(InsertBB, NumElements)) {
      if (RDA.isSafeToMoveForwards(ElemDef, StartInsertPt)) {
        ElemDef->removeFromParent();
        InsertBB->insert(MachineBasicBlock::iterator(StartInsertPt), ElemDef);
        LLVM_DEBUG(dbgs() << "ARM Loops: Moved element count def: "
                   << *ElemDef);
      } else if (RDA.isSafeToMoveBackwards(StartInsertPt, ElemDef)) {
        StartInsertPt->removeFromParent();
        InsertBB->insertAfter(MachineBasicBlock::iterator(ElemDef),
                              StartInsertPt);
        LLVM_DEBUG(dbgs() << "ARM Loops: Moved start past: " << *ElemDef);
      } else {
        LLVM_DEBUG(dbgs() << "ARM Loops: Unable to move element count to loop "
                   << "start instruction.\n");
        return false;
      }
    }
  }

  // Especially in the case of while loops, InsertBB may not be the
  // preheader, so we need to check that the register isn't redefined
  // before entering the loop.
  auto CannotProvideElements = [this](MachineBasicBlock *MBB,
                                      Register NumElements) {
    // NumElements is redefined in this block.
    if (RDA.hasLocalDefBefore(&MBB->back(), NumElements))
      return true;

    // Don't continue searching up through multiple predecessors.
    if (MBB->pred_size() > 1)
      return true;

    return false;
  };

  // First, find the block that looks like the preheader.
  MachineBasicBlock *MBB = MLI.findLoopPreheader(&ML, true);
  if (!MBB) {
    LLVM_DEBUG(dbgs() << "ARM Loops: Didn't find preheader.\n");
    return false;
  }

  // Then search backwards for a def, until we get to InsertBB.
  while (MBB != InsertBB) {
    if (CannotProvideElements(MBB, NumElements)) {
      LLVM_DEBUG(dbgs() << "ARM Loops: Unable to provide element count.\n");
      return false;
    }
    MBB = *MBB->pred_begin();
  }

  // Check that the value change of the element count is what we expect and
  // that the predication will be equivalent. For this we need:
  // NumElements = NumElements - VectorWidth. The sub will be a sub immediate
  // and we can also allow register copies within the chain too.
  auto IsValidSub = [](MachineInstr *MI, unsigned ExpectedVecWidth) {
    unsigned ImmOpIdx = 0;
    switch (MI->getOpcode()) {
    default:
      llvm_unreachable("unhandled sub opcode");
    case ARM::tSUBi3:
    case ARM::tSUBi8:
      ImmOpIdx = 3;
      break;
    case ARM::t2SUBri:
    case ARM::t2SUBri12:
      ImmOpIdx = 2;
      break;
    }
    return MI->getOperand(ImmOpIdx).getImm() == ExpectedVecWidth;
  };

  MBB = VCTP->getParent();
  if (auto *Def = RDA.getUniqueReachingMIDef(&MBB->back(), NumElements)) {
    SmallPtrSet<MachineInstr*, 2> ElementChain;
    SmallPtrSet<MachineInstr*, 2> Ignore = { VCTP };
    unsigned ExpectedVectorWidth = getTailPredVectorWidth(VCTP->getOpcode());

    if (RDA.isSafeToRemove(Def, ElementChain, Ignore)) {
      bool FoundSub = false;

      for (auto *MI : ElementChain) {
        if (isMovRegOpcode(MI->getOpcode()))
          continue;

        if (isSubImmOpcode(MI->getOpcode())) {
          if (FoundSub || !IsValidSub(MI, ExpectedVectorWidth))
            return false;
          FoundSub = true;
        } else
          return false;
      }

      LLVM_DEBUG(dbgs() << "ARM Loops: Will remove element count chain:\n";
                 for (auto *MI : ElementChain)
                   dbgs() << " - " << *MI);
      ToRemove.insert(ElementChain.begin(), ElementChain.end());
    }
  }
  return true;
}

bool LowOverheadLoop::ValidateLiveOuts() const {
  // Collect Q-regs that are live in the exit blocks. We don't collect scalars
  // because they won't be affected by lane predication.
  const TargetRegisterClass *QPRs = TRI.getRegClass(ARM::MQPRRegClassID);
  SmallSet<Register, 2> LiveOuts;
  SmallVector<MachineBasicBlock*, 2> ExitBlocks;
  ML.getExitBlocks(ExitBlocks);
  for (auto *MBB : ExitBlocks)
    for (const MachineBasicBlock::RegisterMaskPair &RegMask : MBB->liveins())
      if (QPRs->contains(RegMask.PhysReg))
        LiveOuts.insert(RegMask.PhysReg);

  // Collect the instructions in the loop body that define the live-out values.
  SmallPtrSet<MachineInstr*, 2> LiveMIs;
  MachineBasicBlock *MBB = ML.getHeader();
  for (auto Reg : LiveOuts)
    if (auto *MI = RDA.getLocalLiveOutMIDef(MBB, Reg))
      LiveMIs.insert(MI);

  LLVM_DEBUG(dbgs() << "ARM Loops: Found loop live-outs:\n";
             for (auto *MI : LiveMIs)
               dbgs() << " - " << *MI);
  // We've already validated that any VPT predication within the loop will be
  // equivalent when we perform the predication transformation; so we know that
  // any VPT predicated instruction is predicated upon VCTP. Any live-out
  // instruction needs to be predicated, so check this here.
  for (auto *MI : LiveMIs) {
    int PIdx = llvm::findFirstVPTPredOperandIdx(*MI);
    if (PIdx == -1 || MI->getOperand(PIdx+1).getReg() != ARM::VPR)
      return false;
  }

  return true;
}

void LowOverheadLoop::CheckLegality(ARMBasicBlockUtils *BBUtils) {
  if (Revert)
    return;

  if (!End->getOperand(1).isMBB())
    report_fatal_error("Expected LoopEnd to target basic block");

  // TODO Maybe there's cases where the target doesn't have to be the header,
  // but for now be safe and revert.
  if (End->getOperand(1).getMBB() != ML.getHeader()) {
    LLVM_DEBUG(dbgs() << "ARM Loops: LoopEnd is not targetting header.\n");
    Revert = true;
    return;
  }

  // The WLS and LE instructions have 12-bits for the label offset. WLS
  // requires a positive offset, while LE uses negative.
  if (BBUtils->getOffsetOf(End) < BBUtils->getOffsetOf(ML.getHeader()) ||
      !BBUtils->isBBInRange(End, ML.getHeader(), 4094)) {
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

  InsertPt = Revert ? nullptr : isSafeToDefineLR();
  if (!InsertPt) {
    LLVM_DEBUG(dbgs() << "ARM Loops: Unable to find safe insertion point.\n");
    Revert = true;
    return;
  } else
    LLVM_DEBUG(dbgs() << "ARM Loops: Start insertion point: " << *InsertPt);

  if (!IsTailPredicationLegal()) {
    LLVM_DEBUG(if (!VCTP)
                 dbgs() << "ARM Loops: Didn't find a VCTP instruction.\n";
               dbgs() << "ARM Loops: Tail-predication is not valid.\n");
    return;
  }

  assert(ML.getBlocks().size() == 1 &&
         "Shouldn't be processing a loop with more than one block");
  CannotTailPredicate = !ValidateTailPredicate(InsertPt);
  LLVM_DEBUG(if (CannotTailPredicate)
             dbgs() << "ARM Loops: Couldn't validate tail predicate.\n");
}

bool LowOverheadLoop::ValidateMVEInst(MachineInstr* MI) {
  if (CannotTailPredicate)
    return false;

  // Only support a single vctp.
  if (isVCTP(MI) && VCTP)
    return false;

  // Start a new vpt block when we discover a vpt.
  if (MI->getOpcode() == ARM::MVE_VPST) {
    VPTBlocks.emplace_back(MI, CurrentPredicate);
    CurrentBlock = &VPTBlocks.back();
    return true;
  } else if (isVCTP(MI))
    VCTP = MI;
  else if (MI->getOpcode() == ARM::MVE_VPSEL ||
           MI->getOpcode() == ARM::MVE_VPNOT)
    return false;

  // TODO: Allow VPSEL and VPNOT, we currently cannot because:
  // 1) It will use the VPR as a predicate operand, but doesn't have to be
  //    instead a VPT block, which means we can assert while building up
  //    the VPT block because we don't find another VPST to being a new
  //    one.
  // 2) VPSEL still requires a VPR operand even after tail predicating,
  //    which means we can't remove it unless there is another
  //    instruction, such as vcmp, that can provide the VPR def.

  bool IsUse = false;
  bool IsDef = false;
  const MCInstrDesc &MCID = MI->getDesc();
  for (int i = MI->getNumOperands() - 1; i >= 0; --i) {
    const MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg() || MO.getReg() != ARM::VPR)
      continue;

    if (MO.isDef()) {
      CurrentPredicate.insert(MI);
      IsDef = true;
    } else if (ARM::isVpred(MCID.OpInfo[i].OperandType)) {
      CurrentBlock->addInst(MI, CurrentPredicate);
      IsUse = true;
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

  uint64_t Flags = MCID.TSFlags;
  if ((Flags & ARMII::DomainMask) != ARMII::DomainMVE)
    return true;

  // If we find an instruction that has been marked as not valid for tail
  // predication, only allow the instruction if it's contained within a valid
  // VPT block.
  if ((Flags & ARMII::ValidForTailPredication) == 0 && !IsUse) {
    LLVM_DEBUG(dbgs() << "ARM Loops: Can't tail predicate: " << *MI);
    return false;
  }

  // If the instruction is already explicitly predicated, then the conversion
  // will be fine, but ensure that all memory operations are predicated.
  return !IsUse && MI->mayLoadOrStore() ? false : true;
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
             else if (auto *Preheader = MLI->findLoopPreheader(ML, true))
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

  LowOverheadLoop LoLoop(*ML, *MLI, *RDA, *TRI);
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
      if (MI.isDebugValue())
        continue;
      else if (MI.getOpcode() == ARM::t2LoopDec)
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
        LoLoop.AnalyseMVEInst(&MI);
      }
    }
  }

  LLVM_DEBUG(LoLoop.dump());
  if (!LoLoop.FoundAllComponents()) {
    LLVM_DEBUG(dbgs() << "ARM Loops: Didn't find loop start, update, end\n");
    return false;
  }

  // Check that the only instruction using LoopDec is LoopEnd.
  // TODO: Check for copy chains that really have no effect.
  SmallPtrSet<MachineInstr*, 2> Uses;
  RDA->getReachingLocalUses(LoLoop.Dec, ARM::LR, Uses);
  if (Uses.size() > 1 || !Uses.count(LoLoop.End)) {
    LLVM_DEBUG(dbgs() << "ARM Loops: Unable to remove LoopDec.\n");
    LoLoop.Revert = true;
  }
  LoLoop.CheckLegality(BBUtils.get());
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

bool ARMLowOverheadLoops::RevertLoopDec(MachineInstr *MI) const {
  LLVM_DEBUG(dbgs() << "ARM Loops: Reverting to sub: " << *MI);
  MachineBasicBlock *MBB = MI->getParent();
  SmallPtrSet<MachineInstr*, 1> Ignore;
  for (auto I = MachineBasicBlock::iterator(MI), E = MBB->end(); I != E; ++I) {
    if (I->getOpcode() == ARM::t2LoopEnd) {
      Ignore.insert(&*I);
      break;
    }
  }

  // If nothing defines CPSR between LoopDec and LoopEnd, use a t2SUBS.
  bool SetFlags = RDA->isSafeToDefRegAt(MI, ARM::CPSR, Ignore);

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

// Perform dead code elimation on the loop iteration count setup expression.
// If we are tail-predicating, the number of elements to be processed is the
// operand of the VCTP instruction in the vector body, see getCount(), which is
// register $r3 in this example:
//
//   $lr = big-itercount-expression
//   ..
//   t2DoLoopStart renamable $lr
//   vector.body:
//     ..
//     $vpr = MVE_VCTP32 renamable $r3
//     renamable $lr = t2LoopDec killed renamable $lr, 1
//     t2LoopEnd renamable $lr, %vector.body
//     tB %end
//
// What we would like achieve here is to replace the do-loop start pseudo
// instruction t2DoLoopStart with:
//
//    $lr = MVE_DLSTP_32 killed renamable $r3
//
// Thus, $r3 which defines the number of elements, is written to $lr,
// and then we want to delete the whole chain that used to define $lr,
// see the comment below how this chain could look like.
//
void ARMLowOverheadLoops::IterationCountDCE(LowOverheadLoop &LoLoop) {
  if (!LoLoop.IsTailPredicationLegal())
    return;

  MachineInstr *Def = RDA->getMIOperand(LoLoop.Start, 0);
  if (!Def)
    return;

  // Collect IT blocks.
  std::map<MachineInstr *, SmallPtrSet<MachineInstr *, 2>> ITBlocks;
  std::map<MachineInstr *, MachineInstr *> Predicates;
  MachineInstr *IT = nullptr;
  for (auto &MI : *Def->getParent()) {
    if (MI.getOpcode() == ARM::t2IT)
      IT = &MI;
    else if (TII->getPredicate(MI) != ARMCC::AL) {
      ITBlocks[IT].insert(&MI);
      Predicates[&MI] = IT;
    }
  }

  // If we're removing all of the instructions within an IT block, then
  // also remove the IT instruction.
  SmallPtrSet<MachineInstr*, 2> ModifiedITs;
  SmallPtrSet<MachineInstr*, 2> DeadITs;
  SmallPtrSet<MachineInstr*, 4> Killed;
  RDA->collectLocalKilledOperands(Def, Killed);
  for (auto *MI : Killed) {
    if (!Predicates.count(MI))
      continue;

    MachineInstr *IT = Predicates[MI];
    auto &CurrentBlock = ITBlocks[IT];
    CurrentBlock.erase(MI);
    ModifiedITs.insert(IT);
    if (CurrentBlock.empty()) {
      DeadITs.insert(IT);
      ModifiedITs.erase(IT);
    }
  }

  // Delete the killed instructions only if we don't have any IT blocks that
  // need to be modified because we need to fixup the mask.
  // TODO: Handle cases where IT blocks are modified.
  if (ModifiedITs.empty()) {
    LLVM_DEBUG(dbgs() << "ARM Loops: Will remove iteration count:\n";
               for (auto *MI : Killed)
                 dbgs() << " - " << *MI);
    LoLoop.ToRemove.insert(Killed.begin(), Killed.end());
  }

  // Collect and remove the users of iteration count.
  SmallPtrSet<MachineInstr*, 4> Ignore = { LoLoop.Start, LoLoop.Dec,
                                           LoLoop.End, LoLoop.InsertPt };
  SmallPtrSet<MachineInstr*, 2> Remove;
  if (RDA->isSafeToRemove(Def, Remove, Ignore))
    LoLoop.ToRemove.insert(Remove.begin(), Remove.end());
}

MachineInstr* ARMLowOverheadLoops::ExpandLoopStart(LowOverheadLoop &LoLoop) {
  LLVM_DEBUG(dbgs() << "ARM Loops: Expanding LoopStart.\n");
  // When using tail-predication, try to delete the dead code that was used to
  // calculate the number of loop iterations.
  IterationCountDCE(LoLoop);

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

  // If we're inserting at a mov lr, then remove it as it's redundant.
  if (InsertPt != Start)
    LoLoop.ToRemove.insert(InsertPt);
  LoLoop.ToRemove.insert(Start);
  LLVM_DEBUG(dbgs() << "ARM Loops: Inserted start: " << *MIB);
  return &*MIB;
}

void ARMLowOverheadLoops::ConvertVPTBlocks(LowOverheadLoop &LoLoop) {
  auto RemovePredicate = [](MachineInstr *MI) {
    LLVM_DEBUG(dbgs() << "ARM Loops: Removing predicate from: " << *MI);
    if (int PIdx = llvm::findFirstVPTPredOperandIdx(*MI)) {
      assert(MI->getOperand(PIdx).getImm() == ARMVCC::Then &&
             "Expected Then predicate!");
      MI->getOperand(PIdx).setImm(ARMVCC::None);
      MI->getOperand(PIdx+1).setReg(0);
    } else
      llvm_unreachable("trying to unpredicate a non-predicated instruction");
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
        LoLoop.ToRemove.insert(Block.getVPST());
      }
    } else if (Block.IsOnlyPredicatedOn(LoLoop.VCTP)) {
      // A vpt block which is only predicated upon vctp and has no internal vpr
      // defs:
      // - Remove vpst.
      // - Unpredicate the remaining instructions.
      LLVM_DEBUG(dbgs() << "ARM Loops: Removing VPST: " << *Block.getVPST());
      LoLoop.ToRemove.insert(Block.getVPST());
      for (auto &PredMI : Insts)
        RemovePredicate(PredMI.MI);
    }
  }
  LLVM_DEBUG(dbgs() << "ARM Loops: Removing VCTP: " << *LoLoop.VCTP);
  LoLoop.ToRemove.insert(LoLoop.VCTP);
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
    LoLoop.Dec->eraseFromParent();
    End->eraseFromParent();
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
    bool FlagsAlreadySet = RevertLoopDec(LoLoop.Dec);
    RevertLoopEnd(LoLoop.End, FlagsAlreadySet);
  } else {
    LoLoop.Start = ExpandLoopStart(LoLoop);
    RemoveDeadBranch(LoLoop.Start);
    LoLoop.End = ExpandLoopEnd(LoLoop);
    RemoveDeadBranch(LoLoop.End);
    if (LoLoop.IsTailPredicationLegal())
      ConvertVPTBlocks(LoLoop);
    for (auto *I : LoLoop.ToRemove) {
      LLVM_DEBUG(dbgs() << "ARM Loops: Erasing " << *I);
      I->eraseFromParent();
    }
  }

  PostOrderLoopTraversal DFS(LoLoop.ML, *MLI);
  DFS.ProcessLoop();
  const SmallVectorImpl<MachineBasicBlock*> &PostOrder = DFS.getOrder();
  for (auto *MBB : PostOrder) {
    recomputeLiveIns(*MBB);
    // FIXME: For some reason, the live-in print order is non-deterministic for
    // our tests and I can't out why... So just sort them.
    MBB->sortUniqueLiveIns();
  }

  for (auto *MBB : reverse(PostOrder))
    recomputeLivenessFlags(*MBB);

  // We've moved, removed and inserted new instructions, so update RDA.
  RDA->reset();
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
