//===- RISCVInsertVSETVLI.cpp - Insert VSETVLI instructions ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a function pass that inserts VSETVLI instructions where
// needed and expands the vl outputs of VLEFF/VLSEGFF to PseudoReadVL
// instructions.
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVSubtarget.h"
#include "RISCVVectorConfigAnalysis.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/LiveDebugVariables.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LiveStacks.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include <queue>
using namespace llvm;

#define DEBUG_TYPE "riscv-insert-vsetvli"
#define RISCV_INSERT_VSETVLI_NAME "RISC-V Insert VSETVLI pass"

STATISTIC(NumInsertedVSETVL, "Number of VSETVL inst inserted");
STATISTIC(NumCoalescedVSETVL, "Number of VSETVL inst coalesced");

static cl::opt<bool> EnsureWholeVectorRegisterMoveValidVTYPE(
    DEBUG_TYPE "-whole-vector-register-move-valid-vtype", cl::Hidden,
    cl::desc("Insert vsetvlis before vmvNr.vs to ensure vtype is valid and "
             "vill is cleared"),
    cl::init(true));

namespace {

static unsigned getVLOpNum(const MachineInstr &MI) {
  return RISCVII::getVLOpNum(MI.getDesc());
}

class RISCVInsertVSETVLI : public MachineFunctionPass {
  RISCVVectorConfigInfo *VConfig;
  const RISCVSubtarget *ST;
  const TargetInstrInfo *TII;
  MachineRegisterInfo *MRI;
  // Possibly null!
  LiveIntervals *LIS;

  std::queue<const MachineBasicBlock *> WorkList;

public:
  static char ID;

  RISCVInsertVSETVLI() : MachineFunctionPass(ID) {}
  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();

    AU.addUsedIfAvailable<LiveIntervalsWrapperPass>();
    AU.addPreserved<LiveIntervalsWrapperPass>();
    AU.addPreserved<SlotIndexesWrapperPass>();
    AU.addPreserved<LiveDebugVariablesWrapperLegacy>();
    AU.addPreserved<LiveStacksWrapperLegacy>();
    AU.addRequired<RISCVVectorConfigWrapperPass>();

    MachineFunctionPass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override { return RISCV_INSERT_VSETVLI_NAME; }

private:
  bool needVSETVLIPHI(const VSETVLIInfo &Require,
                      const MachineBasicBlock &MBB) const;
  void insertVSETVLI(MachineBasicBlock &MBB,
                     MachineBasicBlock::iterator InsertPt, DebugLoc DL,
                     const VSETVLIInfo &Info, const VSETVLIInfo &PrevInfo);

  void emitVSETVLIs(MachineBasicBlock &MBB);
  void doPRE(MachineBasicBlock &MBB);
  void insertReadVL(MachineBasicBlock &MBB);

  void coalesceVSETVLIs(MachineBasicBlock &MBB) const;
};

} // end anonymous namespace

char RISCVInsertVSETVLI::ID = 0;
char &llvm::RISCVInsertVSETVLIID = RISCVInsertVSETVLI::ID;

INITIALIZE_PASS_BEGIN(RISCVInsertVSETVLI, DEBUG_TYPE, RISCV_INSERT_VSETVLI_NAME,
                      false, false)
INITIALIZE_PASS_DEPENDENCY(RISCVVectorConfigWrapperPass)
INITIALIZE_PASS_END(RISCVInsertVSETVLI, DEBUG_TYPE, RISCV_INSERT_VSETVLI_NAME,
                    false, false)

// If the AVL is defined by a vsetvli's output vl with the same VLMAX, we can
// replace the AVL operand with the AVL of the defining vsetvli. E.g.
//
// %vl = PseudoVSETVLI %avl:gpr, SEW=32, LMUL=M1
// $x0 = PseudoVSETVLI %vl:gpr, SEW=32, LMUL=M1
// ->
// %vl = PseudoVSETVLI %avl:gpr, SEW=32, LMUL=M1
// $x0 = PseudoVSETVLI %avl:gpr, SEW=32, LMUL=M1
void RISCVInsertVSETVLI::insertVSETVLI(MachineBasicBlock &MBB,
                     MachineBasicBlock::iterator InsertPt, DebugLoc DL,
                     const VSETVLIInfo &Info, const VSETVLIInfo &PrevInfo) {

  ++NumInsertedVSETVL;
  if (PrevInfo.isValid() && !PrevInfo.isUnknown()) {
    // Use X0, X0 form if the AVL is the same and the SEW+LMUL gives the same
    // VLMAX.
    if (Info.hasSameAVL(PrevInfo) && Info.hasSameVLMAX(PrevInfo)) {
      auto MI = BuildMI(MBB, InsertPt, DL, TII->get(RISCV::PseudoVSETVLIX0X0))
                    .addReg(RISCV::X0, RegState::Define | RegState::Dead)
                    .addReg(RISCV::X0, RegState::Kill)
                    .addImm(Info.encodeVTYPE())
                    .addReg(RISCV::VL, RegState::Implicit);
      if (LIS)
        LIS->InsertMachineInstrInMaps(*MI);
      return;
    }

    // If our AVL is a virtual register, it might be defined by a VSET(I)VLI. If
    // it has the same VLMAX we want and the last VL/VTYPE we observed is the
    // same, we can use the X0, X0 form.
    if (Info.hasSameVLMAX(PrevInfo) && Info.hasAVLReg()) {
      if (const MachineInstr *DefMI = Info.getAVLDefMI(LIS);
          DefMI && RISCVInstrInfo::isVectorConfigInstr(*DefMI)) {
        VSETVLIInfo DefInfo = VConfig->getInfoForVSETVLI(*DefMI);
        if (DefInfo.hasSameAVL(PrevInfo) && DefInfo.hasSameVLMAX(PrevInfo)) {
          auto MI =
              BuildMI(MBB, InsertPt, DL, TII->get(RISCV::PseudoVSETVLIX0X0))
                  .addReg(RISCV::X0, RegState::Define | RegState::Dead)
                  .addReg(RISCV::X0, RegState::Kill)
                  .addImm(Info.encodeVTYPE())
                  .addReg(RISCV::VL, RegState::Implicit);
          if (LIS)
            LIS->InsertMachineInstrInMaps(*MI);
          return;
        }
      }
    }
  }

  if (Info.hasAVLImm()) {
    auto MI = BuildMI(MBB, InsertPt, DL, TII->get(RISCV::PseudoVSETIVLI))
                  .addReg(RISCV::X0, RegState::Define | RegState::Dead)
                  .addImm(Info.getAVLImm())
                  .addImm(Info.encodeVTYPE());
    if (LIS)
      LIS->InsertMachineInstrInMaps(*MI);
    return;
  }

  if (Info.hasAVLVLMAX()) {
    Register DestReg = MRI->createVirtualRegister(&RISCV::GPRNoX0RegClass);
    auto MI = BuildMI(MBB, InsertPt, DL, TII->get(RISCV::PseudoVSETVLIX0))
                  .addReg(DestReg, RegState::Define | RegState::Dead)
                  .addReg(RISCV::X0, RegState::Kill)
                  .addImm(Info.encodeVTYPE());
    if (LIS) {
      LIS->InsertMachineInstrInMaps(*MI);
      LIS->createAndComputeVirtRegInterval(DestReg);
    }
    return;
  }

  Register AVLReg = Info.getAVLReg();
  MRI->constrainRegClass(AVLReg, &RISCV::GPRNoX0RegClass);
  auto MI = BuildMI(MBB, InsertPt, DL, TII->get(RISCV::PseudoVSETVLI))
                .addReg(RISCV::X0, RegState::Define | RegState::Dead)
                .addReg(AVLReg)
                .addImm(Info.encodeVTYPE());
  if (LIS) {
    LIS->InsertMachineInstrInMaps(*MI);
    LiveInterval &LI = LIS->getInterval(AVLReg);
    SlotIndex SI = LIS->getInstructionIndex(*MI).getRegSlot();
    const VNInfo *CurVNI = Info.getAVLVNInfo();
    // If the AVL value isn't live at MI, do a quick check to see if it's easily
    // extendable. Otherwise, we need to copy it.
    if (LI.getVNInfoBefore(SI) != CurVNI) {
      if (!LI.liveAt(SI) && LI.containsOneValue())
        LIS->extendToIndices(LI, SI);
      else {
        Register AVLCopyReg =
            MRI->createVirtualRegister(&RISCV::GPRNoX0RegClass);
        MachineBasicBlock *MBB = LIS->getMBBFromIndex(CurVNI->def);
        MachineBasicBlock::iterator II;
        if (CurVNI->isPHIDef())
          II = MBB->getFirstNonPHI();
        else {
          II = LIS->getInstructionFromIndex(CurVNI->def);
          II = std::next(II);
        }
        assert(II.isValid());
        auto AVLCopy = BuildMI(*MBB, II, DL, TII->get(RISCV::COPY), AVLCopyReg)
                           .addReg(AVLReg);
        LIS->InsertMachineInstrInMaps(*AVLCopy);
        MI->getOperand(1).setReg(AVLCopyReg);
        LIS->createAndComputeVirtRegInterval(AVLCopyReg);
      }
    }
  }
}

// If we weren't able to prove a vsetvli was directly unneeded, it might still
// be unneeded if the AVL was a phi node where all incoming values are VL
// outputs from the last VSETVLI in their respective basic blocks.
bool RISCVInsertVSETVLI::needVSETVLIPHI(const VSETVLIInfo &Require,
                                        const MachineBasicBlock &MBB) const {
  if (!Require.hasAVLReg())
    return true;

  if (!LIS)
    return true;

  // We need the AVL to have been produced by a PHI node in this basic block.
  const VNInfo *Valno = Require.getAVLVNInfo();
  if (!Valno->isPHIDef() || LIS->getMBBFromIndex(Valno->def) != &MBB)
    return true;

  const LiveRange &LR = LIS->getInterval(Require.getAVLReg());

  for (auto *PBB : MBB.predecessors()) {
    const VSETVLIInfo &PBBExit = VConfig->getInfo()[PBB->getNumber()].Exit;

    // We need the PHI input to the be the output of a VSET(I)VLI.
    const VNInfo *Value = LR.getVNInfoBefore(LIS->getMBBEndIdx(PBB));
    if (!Value)
      return true;
    MachineInstr *DefMI = LIS->getInstructionFromIndex(Value->def);
    if (!DefMI || !RISCVInstrInfo::isVectorConfigInstr(*DefMI))
      return true;

    // We found a VSET(I)VLI make sure it matches the output of the
    // predecessor block.
    VSETVLIInfo DefInfo = VConfig->getInfoForVSETVLI(*DefMI);
    if (DefInfo != PBBExit)
      return true;

    // Require has the same VL as PBBExit, so if the exit from the
    // predecessor has the VTYPE we are looking for we might be able
    // to avoid a VSETVLI.
    if (PBBExit.isUnknown() || !PBBExit.hasSameVTYPE(Require))
      return true;
  }

  // If all the incoming values to the PHI checked out, we don't need
  // to insert a VSETVLI.
  return false;
}

void RISCVInsertVSETVLI::emitVSETVLIs(MachineBasicBlock &MBB) {
  VSETVLIInfo CurInfo = VConfig->getInfo()[MBB.getNumber()].Pred;
  // Track whether the prefix of the block we've scanned is transparent
  // (meaning has not yet changed the abstract state).
  bool PrefixTransparent = true;
  for (MachineInstr &MI : MBB) {
    const VSETVLIInfo PrevInfo = CurInfo;
    VConfig->transferBefore(CurInfo, MI);

    // If this is an explicit VSETVLI or VSETIVLI, update our state.
    if (RISCVInstrInfo::isVectorConfigInstr(MI)) {
      // Conservatively, mark the VL and VTYPE as live.
      assert(MI.getOperand(3).getReg() == RISCV::VL &&
             MI.getOperand(4).getReg() == RISCV::VTYPE &&
             "Unexpected operands where VL and VTYPE should be");
      MI.getOperand(3).setIsDead(false);
      MI.getOperand(4).setIsDead(false);
      PrefixTransparent = false;
    }

    if (EnsureWholeVectorRegisterMoveValidVTYPE &&
        RISCVInstrInfo::isVectorCopy(ST->getRegisterInfo(), MI)) {
      if (!PrevInfo.isCompatible(DemandedFields::all(), CurInfo, LIS)) {
        insertVSETVLI(MBB, MI, MI.getDebugLoc(), CurInfo, PrevInfo);
        PrefixTransparent = false;
      }
      MI.addOperand(MachineOperand::CreateReg(RISCV::VTYPE, /*isDef*/ false,
                                              /*isImp*/ true));
    }

    uint64_t TSFlags = MI.getDesc().TSFlags;
    if (RISCVII::hasSEWOp(TSFlags)) {
      if (!PrevInfo.isCompatible(DemandedFields::all(), CurInfo, LIS)) {
        // If this is the first implicit state change, and the state change
        // requested can be proven to produce the same register contents, we
        // can skip emitting the actual state change and continue as if we
        // had since we know the GPR result of the implicit state change
        // wouldn't be used and VL/VTYPE registers are correct.  Note that
        // we *do* need to model the state as if it changed as while the
        // register contents are unchanged, the abstract model can change.
        if (!PrefixTransparent || needVSETVLIPHI(CurInfo, MBB))
          insertVSETVLI(MBB, MI, MI.getDebugLoc(), CurInfo, PrevInfo);
        PrefixTransparent = false;
      }

      if (RISCVII::hasVLOp(TSFlags)) {
        MachineOperand &VLOp = MI.getOperand(getVLOpNum(MI));
        if (VLOp.isReg()) {
          Register Reg = VLOp.getReg();

          // Erase the AVL operand from the instruction.
          VLOp.setReg(RISCV::NoRegister);
          VLOp.setIsKill(false);
          if (LIS) {
            LiveInterval &LI = LIS->getInterval(Reg);
            SmallVector<MachineInstr *> DeadMIs;
            LIS->shrinkToUses(&LI, &DeadMIs);
            // We might have separate components that need split due to
            // needVSETVLIPHI causing us to skip inserting a new VL def.
            SmallVector<LiveInterval *> SplitLIs;
            LIS->splitSeparateComponents(LI, SplitLIs);

            // If the AVL was an immediate > 31, then it would have been emitted
            // as an ADDI. However, the ADDI might not have been used in the
            // vsetvli, or a vsetvli might not have been emitted, so it may be
            // dead now.
            for (MachineInstr *DeadMI : DeadMIs) {
              if (!TII->isAddImmediate(*DeadMI, Reg))
                continue;
              LIS->RemoveMachineInstrFromMaps(*DeadMI);
              DeadMI->eraseFromParent();
            }
          }
        }
        MI.addOperand(MachineOperand::CreateReg(RISCV::VL, /*isDef*/ false,
                                                /*isImp*/ true));
      }
      MI.addOperand(MachineOperand::CreateReg(RISCV::VTYPE, /*isDef*/ false,
                                              /*isImp*/ true));
    }

    if (MI.isInlineAsm()) {
      MI.addOperand(MachineOperand::CreateReg(RISCV::VL, /*isDef*/ true,
                                              /*isImp*/ true));
      MI.addOperand(MachineOperand::CreateReg(RISCV::VTYPE, /*isDef*/ true,
                                              /*isImp*/ true));
    }

    if (MI.isCall() || MI.isInlineAsm() ||
        MI.modifiesRegister(RISCV::VL, /*TRI=*/nullptr) ||
        MI.modifiesRegister(RISCV::VTYPE, /*TRI=*/nullptr))
      PrefixTransparent = false;

    VConfig->transferAfter(CurInfo, MI);
  }

  const auto &Info = VConfig->getInfo()[MBB.getNumber()];
  if (CurInfo != Info.Exit) {
    LLVM_DEBUG(dbgs() << "in block " << printMBBReference(MBB) << "\n");
    LLVM_DEBUG(dbgs() << "  begin        state: " << Info.Pred << "\n");
    LLVM_DEBUG(dbgs() << "  expected end state: " << Info.Exit << "\n");
    LLVM_DEBUG(dbgs() << "  actual   end state: " << CurInfo << "\n");
  }
  assert(CurInfo == Info.Exit && "InsertVSETVLI dataflow invariant violated");
}

/// Perform simple partial redundancy elimination of the VSETVLI instructions
/// we're about to insert by looking for cases where we can PRE from the
/// beginning of one block to the end of one of its predecessors.  Specifically,
/// this is geared to catch the common case of a fixed length vsetvl in a single
/// block loop when it could execute once in the preheader instead.
void RISCVInsertVSETVLI::doPRE(MachineBasicBlock &MBB) {
  if (!VConfig->getInfo()[MBB.getNumber()].Pred.isUnknown())
    return;

  MachineBasicBlock *UnavailablePred = nullptr;
  VSETVLIInfo AvailableInfo;
  for (MachineBasicBlock *P : MBB.predecessors()) {
    const VSETVLIInfo &PredInfo = VConfig->getInfo()[P->getNumber()].Exit;
    if (PredInfo.isUnknown()) {
      if (UnavailablePred)
        return;
      UnavailablePred = P;
    } else if (!AvailableInfo.isValid()) {
      AvailableInfo = PredInfo;
    } else if (AvailableInfo != PredInfo) {
      return;
    }
  }

  // Unreachable, single pred, or full redundancy. Note that FRE is handled by
  // phase 3.
  if (!UnavailablePred || !AvailableInfo.isValid())
    return;

  if (!LIS)
    return;

  // If we don't know the exact VTYPE, we can't copy the vsetvli to the exit of
  // the unavailable pred.
  if (AvailableInfo.hasSEWLMULRatioOnly())
    return;

  // Critical edge - TODO: consider splitting?
  if (UnavailablePred->succ_size() != 1)
    return;

  // If the AVL value is a register (other than our VLMAX sentinel),
  // we need to prove the value is available at the point we're going
  // to insert the vsetvli at.
  if (AvailableInfo.hasAVLReg()) {
    SlotIndex SI = AvailableInfo.getAVLVNInfo()->def;
    // This is an inline dominance check which covers the case of
    // UnavailablePred being the preheader of a loop.
    if (LIS->getMBBFromIndex(SI) != UnavailablePred)
      return;
    if (!UnavailablePred->terminators().empty() &&
        SI >= LIS->getInstructionIndex(*UnavailablePred->getFirstTerminator()))
      return;
  }

  // Model the effect of changing the input state of the block MBB to
  // AvailableInfo.  We're looking for two issues here; one legality,
  // one profitability.
  // 1) If the block doesn't use some of the fields from VL or VTYPE, we
  //    may hit the end of the block with a different end state.  We can
  //    not make this change without reflowing later blocks as well.
  // 2) If we don't actually remove a transition, inserting a vsetvli
  //    into the predecessor block would be correct, but unprofitable.
  VSETVLIInfo OldInfo = VConfig->getInfo()[MBB.getNumber()].Pred;
  VSETVLIInfo CurInfo = AvailableInfo;
  int TransitionsRemoved = 0;
  for (const MachineInstr &MI : MBB) {
    const VSETVLIInfo LastInfo = CurInfo;
    const VSETVLIInfo LastOldInfo = OldInfo;
    VConfig->transferBefore(CurInfo, MI);
    VConfig->transferBefore(OldInfo, MI);
    if (CurInfo == LastInfo)
      TransitionsRemoved++;
    if (LastOldInfo == OldInfo)
      TransitionsRemoved--;
    VConfig->transferAfter(CurInfo, MI);
    VConfig->transferAfter(OldInfo, MI);
    if (CurInfo == OldInfo)
      // Convergence.  All transitions after this must match by construction.
      break;
  }
  if (CurInfo != OldInfo || TransitionsRemoved <= 0)
    // Issues 1 and 2 above
    return;

  // Finally, update both data flow state and insert the actual vsetvli.
  // Doing both keeps the code in sync with the dataflow results, which
  // is critical for correctness of phase 3.
  auto OldExit = VConfig->getInfo()[UnavailablePred->getNumber()].Exit;
  LLVM_DEBUG(dbgs() << "PRE VSETVLI from " << MBB.getName() << " to "
                    << UnavailablePred->getName() << " with state "
                    << AvailableInfo << "\n");
  VConfig->getInfo()[UnavailablePred->getNumber()].Exit = AvailableInfo;
  VConfig->getInfo()[MBB.getNumber()].Pred = AvailableInfo;

  // Note there's an implicit assumption here that terminators never use
  // or modify VL or VTYPE.  Also, fallthrough will return end().
  auto InsertPt = UnavailablePred->getFirstInstrTerminator();
  insertVSETVLI(*UnavailablePred, InsertPt,
                UnavailablePred->findDebugLoc(InsertPt),
                AvailableInfo, OldExit);
}

void RISCVInsertVSETVLI::coalesceVSETVLIs(MachineBasicBlock &MBB) const {
  MachineInstr *NextMI = nullptr;
  // We can have arbitrary code in successors, so VL and VTYPE
  // must be considered demanded.
  DemandedFields Used;
  Used.demandVL();
  Used.demandVTYPE();
  SmallVector<MachineInstr*> ToDelete;

  auto dropAVLUse = [&](MachineOperand &MO) {
    if (!MO.isReg() || !MO.getReg().isVirtual())
      return;
    Register OldVLReg = MO.getReg();
    MO.setReg(RISCV::NoRegister);

    if (LIS)
      LIS->shrinkToUses(&LIS->getInterval(OldVLReg));

    MachineInstr *VLOpDef = MRI->getUniqueVRegDef(OldVLReg);
    if (VLOpDef && TII->isAddImmediate(*VLOpDef, OldVLReg) &&
        MRI->use_nodbg_empty(OldVLReg))
      ToDelete.push_back(VLOpDef);
  };

  for (MachineInstr &MI : make_early_inc_range(reverse(MBB))) {

    if (!RISCVInstrInfo::isVectorConfigInstr(MI)) {
      Used.doUnion(RISCVVectorConfigInfo::getDemanded(MI, ST));
      if (MI.isCall() || MI.isInlineAsm() ||
          MI.modifiesRegister(RISCV::VL, /*TRI=*/nullptr) ||
          MI.modifiesRegister(RISCV::VTYPE, /*TRI=*/nullptr))
        NextMI = nullptr;
      continue;
    }

    if (!MI.getOperand(0).isDead())
      Used.demandVL();

    if (NextMI) {
      if (!Used.usedVL() && !Used.usedVTYPE()) {
        dropAVLUse(MI.getOperand(1));
        if (LIS)
          LIS->RemoveMachineInstrFromMaps(MI);
        MI.eraseFromParent();
        NumCoalescedVSETVL++;
        // Leave NextMI unchanged
        continue;
      }

      if (VConfig->canMutatePriorConfig(MI, *NextMI, Used)) {
        if (!RISCVInstrInfo::isVLPreservingConfig(*NextMI)) {
          Register DefReg = NextMI->getOperand(0).getReg();

          MI.getOperand(0).setReg(DefReg);
          MI.getOperand(0).setIsDead(false);

          // Move the AVL from NextMI to MI
          dropAVLUse(MI.getOperand(1));
          if (NextMI->getOperand(1).isImm())
            MI.getOperand(1).ChangeToImmediate(NextMI->getOperand(1).getImm());
          else
            MI.getOperand(1).ChangeToRegister(NextMI->getOperand(1).getReg(),
                                              false);
          dropAVLUse(NextMI->getOperand(1));

          // The def of DefReg moved to MI, so extend the LiveInterval up to
          // it.
          if (DefReg.isVirtual() && LIS) {
            LiveInterval &DefLI = LIS->getInterval(DefReg);
            SlotIndex MISlot = LIS->getInstructionIndex(MI).getRegSlot();
            SlotIndex NextMISlot =
                LIS->getInstructionIndex(*NextMI).getRegSlot();
            VNInfo *DefVNI = DefLI.getVNInfoAt(NextMISlot);
            LiveInterval::Segment S(MISlot, NextMISlot, DefVNI);
            DefLI.addSegment(S);
            DefVNI->def = MISlot;
            // Mark DefLI as spillable if it was previously unspillable
            DefLI.setWeight(0);

            // DefReg may have had no uses, in which case we need to shrink
            // the LiveInterval up to MI.
            LIS->shrinkToUses(&DefLI);
          }

          MI.setDesc(NextMI->getDesc());
        }
        MI.getOperand(2).setImm(NextMI->getOperand(2).getImm());

        dropAVLUse(NextMI->getOperand(1));
        if (LIS)
          LIS->RemoveMachineInstrFromMaps(*NextMI);
        NextMI->eraseFromParent();
        NumCoalescedVSETVL++;
        // fallthrough
      }
    }
    NextMI = &MI;
    Used = RISCVVectorConfigInfo::getDemanded(MI, ST);
  }

  // Loop over the dead AVL values, and delete them now.  This has
  // to be outside the above loop to avoid invalidating iterators.
  for (auto *MI : ToDelete) {
    if (LIS) {
      LIS->removeInterval(MI->getOperand(0).getReg());
      LIS->RemoveMachineInstrFromMaps(*MI);
    }
    MI->eraseFromParent();
  }
}

void RISCVInsertVSETVLI::insertReadVL(MachineBasicBlock &MBB) {
  for (auto I = MBB.begin(), E = MBB.end(); I != E;) {
    MachineInstr &MI = *I++;
    if (RISCVInstrInfo::isFaultOnlyFirstLoad(MI)) {
      Register VLOutput = MI.getOperand(1).getReg();
      assert(VLOutput.isVirtual());
      if (!MI.getOperand(1).isDead()) {
        auto ReadVLMI = BuildMI(MBB, I, MI.getDebugLoc(),
                                TII->get(RISCV::PseudoReadVL), VLOutput);
        // Move the LiveInterval's definition down to PseudoReadVL.
        if (LIS) {
          SlotIndex NewDefSI =
              LIS->InsertMachineInstrInMaps(*ReadVLMI).getRegSlot();
          LiveInterval &DefLI = LIS->getInterval(VLOutput);
          LiveRange::Segment *DefSeg = DefLI.getSegmentContaining(NewDefSI);
          VNInfo *DefVNI = DefLI.getVNInfoAt(DefSeg->start);
          DefLI.removeSegment(DefSeg->start, NewDefSI);
          DefVNI->def = NewDefSI;
        }
      }
      // We don't use the vl output of the VLEFF/VLSEGFF anymore.
      MI.getOperand(1).setReg(RISCV::X0);
      MI.addRegisterDefined(RISCV::VL, MRI->getTargetRegisterInfo());
    }
  }
}

bool RISCVInsertVSETVLI::runOnMachineFunction(MachineFunction &MF) {
  // Skip if the vector extension is not enabled.
  ST = &MF.getSubtarget<RISCVSubtarget>();
  if (!ST->hasVInstructions())
    return false;

  LLVM_DEBUG(dbgs() << "Entering InsertVSETVLI for " << MF.getName() << "\n");

  TII = ST->getInstrInfo();
  MRI = &MF.getRegInfo();
  auto *LISWrapper = getAnalysisIfAvailable<LiveIntervalsWrapperPass>();
  LIS = LISWrapper ? &LISWrapper->getLIS() : nullptr;
  VConfig = &getAnalysis<RISCVVectorConfigWrapperPass>().getResult();

  if (!VConfig->haveVectorOp())
    return false;
  // Perform partial redundancy elimination of vsetvli transitions.
  for (MachineBasicBlock &MBB : MF)
    doPRE(MBB);

  for (MachineBasicBlock &MBB : MF)
    emitVSETVLIs(MBB);

  // Now that all vsetvlis are explicit, go through and do block local
  // DSE and peephole based demanded fields based transforms.  Note that
  // this *must* be done outside the main dataflow so long as we allow
  // any cross block analysis within the dataflow.  We can't have both
  // demanded fields based mutation and non-local analysis in the
  // dataflow at the same time without introducing inconsistencies.
  // We're visiting blocks from the bottom up because a VSETVLI in the
  // earlier block might become dead when its uses in later blocks are
  // optimized away.
  for (MachineBasicBlock *MBB : post_order(&MF))
    coalesceVSETVLIs(*MBB);

  // Insert PseudoReadVL after VLEFF/VLSEGFF and replace it with the vl output
  // of VLEFF/VLSEGFF.
  for (MachineBasicBlock &MBB : MF)
    insertReadVL(MBB);

  return true;
}

/// Returns an instance of the Insert VSETVLI pass.
FunctionPass *llvm::createRISCVInsertVSETVLIPass() {
  return new RISCVInsertVSETVLI();
}
