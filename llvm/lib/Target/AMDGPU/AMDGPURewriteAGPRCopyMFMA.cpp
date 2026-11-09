//===-- AMDGPURewriteAGPRCopyMFMA.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file \brief Try to replace MFMA instructions using VGPRs with MFMA
/// instructions using AGPRs. We expect MFMAs to be selected using VGPRs, and
/// only use AGPRs if it helps avoid spilling. In this case, the MFMA will have
/// copies between AGPRs and VGPRs and the AGPR variant of an MFMA pseudo. This
/// pass will attempt to delete the cross register bank copy and replace the
/// MFMA opcode.
///
/// After a rewrite, copies between any two registers assigned to AGPRs (e.g.
/// the loop-carried copy from PHI elimination, which the coalescer could not
/// join across the VGPR/AGPR classes) are joined where possible, so they do
/// not become AGPR-to-AGPR copies.
///
/// TODO:
///  - Update LiveIntervals incrementally instead of recomputing from scratch
///
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNRegPressure.h"
#include "GCNSubtarget.h"
#include "SIMachineFunctionInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/LiveDebugVariables.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LiveRegMatrix.h"
#include "llvm/CodeGen/LiveStacks.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/RegisterClassInfo.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/DebugCounter.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-rewrite-agpr-copy-mfma"

DEBUG_COUNTER(RewriteAGPRCopyMFMACounter, DEBUG_TYPE,
              "Controls which MFMA chains are rewritten to AGPR form");

namespace {

STATISTIC(NumMFMAsRewrittenToAGPR,
          "Number of MFMA instructions rewritten to use AGPR form");
STATISTIC(NumCopyRelatedRegsJoined,
          "Number of copy-related registers joined after an AGPR rewrite");

/// Map from spill slot frame index to list of instructions which reference it.
using SpillReferenceMap = DenseMap<int, SmallVector<MachineInstr *, 4>>;

class AMDGPURewriteAGPRCopyMFMAImpl {
  MachineFunction &MF;
  const GCNSubtarget &ST;
  const SIInstrInfo &TII;
  const SIRegisterInfo &TRI;
  MachineRegisterInfo &MRI;
  VirtRegMap &VRM;
  LiveRegMatrix &LRM;
  LiveIntervals &LIS;
  LiveStacks &LSS;
  const RegisterClassInfo &RegClassInfo;
  MachineDominatorTree &MDT;
  /// Null if the analysis is not available.
  LiveDebugVariables *LDV;

  bool attemptReassignmentsToAGPR(SmallSetVector<Register, 4> &InterferingRegs,
                                  MCPhysReg PrefPhysReg) const;

public:
  AMDGPURewriteAGPRCopyMFMAImpl(MachineFunction &MF, VirtRegMap &VRM,
                                LiveRegMatrix &LRM, LiveIntervals &LIS,
                                LiveStacks &LSS,
                                const RegisterClassInfo &RegClassInfo,
                                MachineDominatorTree &MDT,
                                LiveDebugVariables *LDV)
      : MF(MF), ST(MF.getSubtarget<GCNSubtarget>()), TII(*ST.getInstrInfo()),
        TRI(*ST.getRegisterInfo()), MRI(MF.getRegInfo()), VRM(VRM), LRM(LRM),
        LIS(LIS), LSS(LSS), RegClassInfo(RegClassInfo), MDT(MDT), LDV(LDV) {}

  bool isRewriteCandidate(const MachineInstr &MI) const {
    return TII.isMAI(MI) && AMDGPU::getAGPRFormOp(MI.getOpcode()) != -1;
  }

  /// Find AV_* registers assigned to AGPRs (or virtual registers which were
  /// already required to be AGPR).
  ///
  /// \return the assigned physical register that \p VReg is assigned to if it
  /// is an AGPR, otherwise MCRegister().
  MCRegister getAssignedAGPR(Register VReg) const {
    MCRegister PhysReg = VRM.getPhys(VReg);
    if (!PhysReg)
      return MCRegister();

    // If this is an AV register, we have to check if the actual assignment is
    // to an AGPR
    const TargetRegisterClass *AssignedRC = TRI.getPhysRegBaseClass(PhysReg);
    return TRI.isAGPRClass(AssignedRC) ? PhysReg : MCRegister();
  }

  bool tryReassigningMFMAChain(MachineInstr &MFMA, Register MFMAHintReg,
                               MCPhysReg PhysRegHint) const;

  /// Compute the register class constraints based on the uses of \p Reg,
  /// excluding MFMA uses from which can be rewritten to change the register
  /// class constraint. MFMA scale operands need to be constraint checked.
  /// This should be nearly identical to MachineRegisterInfo::recomputeRegClass.

  /// \p RewriteCandidates will collect the set of MFMA instructions that need
  /// to have the opcode mutated to perform the replacement.
  ///
  /// \p RewriteRegs will accumulate the set of register used by those MFMAs
  /// that need to have the register classes adjusted.
  bool recomputeRegClassExceptRewritable(
      Register Reg, SmallVectorImpl<MachineInstr *> &RewriteCandidates,
      SmallSetVector<Register, 4> &RewriteRegs) const;

  /// \returns the slot index at which the def operand \p MO defines its
  /// register.
  SlotIndex getDefIndex(const MachineOperand &MO) const {
    return LIS.getInstructionIndex(*MO.getParent())
        .getRegSlot(MO.isEarlyClobber());
  }

  /// Collect in \p Copies the copies of \p Src that define \p Dst. \returns
  /// true if there are any and removing them preserves every value: \p Dst
  /// and \p Src only overlap where \p Dst holds the value it copied.
  bool collectJoinableCopies(Register Dst, Register Src,
                             SmallVectorImpl<MachineInstr *> &Copies) const;

  /// Clear the dead flag of every def of \p Reg that is not dead in \p LI.
  void clearStaleDeadFlags(Register Reg, const LiveInterval &LI) const;

  /// Erase \p Copy and remove it from the slot indexes.
  void eraseCopy(MachineInstr &Copy) const;

  /// Join \p Dst into \p Src if \p Dst is defined by copies of \p Src and the
  /// two only overlap where they hold the same value. \returns true on
  /// success; \p Dst then has no defs or uses left.
  bool tryJoinCopyPair(Register Dst, Register Src) const;

  /// Join \p Reg with one of its AGPR copy partners. \returns true on success.
  bool joinWithCopyPartner(Register Reg) const;

  /// Join the copy-related registers assigned to AGPRs until nothing more
  /// joins.
  void joinAllCopyPartners() const;

  bool tryFoldCopiesToAGPR(Register VReg, MCRegister AssignedAGPR) const;
  bool tryFoldCopiesFromAGPR(Register VReg, MCRegister AssignedAGPR) const;

  /// Replace spill instruction \p SpillMI which loads/stores from/to \p SpillFI
  /// with a COPY to the replacement register value \p VReg.
  void replaceSpillWithCopyToVReg(MachineInstr &SpillMI, int SpillFI,
                                  Register VReg) const;

  /// Create a map from frame index to use instructions for spills. If a use of
  /// the frame index does not consist only of spill instructions, it will not
  /// be included in the map.
  void collectSpillIndexUses(ArrayRef<LiveInterval *> StackIntervals,
                             SpillReferenceMap &Map) const;

  /// Return true if the reload \p LoadMI of the stack slot with live interval
  /// \p SlotLI is jointly dominated by the slot's spill stores, i.e. every path
  /// from the entry block to the load passes through a store to the slot before
  /// the load. \p StoreFreeReachable is the set of blocks reachable from the
  /// entry block without passing through any store block for the slot.
  bool isLoadJointlyDominatedByStores(
      const MachineInstr &LoadMI, const LiveInterval &SlotLI,
      const SmallPtrSetImpl<MachineBasicBlock *> &StoreFreeReachable) const;

  /// Attempt to unspill VGPRs by finding a free register and replacing the
  /// spill instructions with copies.
  void eliminateSpillsOfReassignedVGPRs() const;

  bool run(MachineFunction &MF) const;
};

bool AMDGPURewriteAGPRCopyMFMAImpl::recomputeRegClassExceptRewritable(
    Register StartReg, SmallVectorImpl<MachineInstr *> &RewriteCandidates,
    SmallSetVector<Register, 4> &RewriteRegs) const {
  SmallVector<Register, 8> Worklist = {StartReg};

  // Recursively visit all transitive MFMA users
  while (!Worklist.empty()) {
    Register Reg = Worklist.pop_back_val();
    const TargetRegisterClass *OldRC = MRI.getRegClass(Reg);

    // Inflate to the equivalent AV_* class.
    const TargetRegisterClass *NewRC = TRI.getLargestLegalSuperClass(OldRC, MF);
    if (OldRC == NewRC)
      return false;

    // Accumulate constraints from all uses.
    for (MachineOperand &MO : MRI.reg_nodbg_operands(Reg)) {
      // Apply the effect of the given operand to NewRC.
      MachineInstr *MI = MO.getParent();

      // We can swap the classes of dst + src2 as a pair to AGPR, so ignore the
      // effects of rewrite candidates. It just so happens that we can use
      // either AGPR or VGPR in src0/src1. We still need to check constraint
      // effects for scale variant, which does not allow AGPR.
      if (isRewriteCandidate(*MI)) {
        int AGPROp = AMDGPU::getAGPRFormOp(MI->getOpcode());
        const MCInstrDesc &AGPRDesc = TII.get(AGPROp);
        const TargetRegisterClass *NewRC =
            TII.getRegClass(AGPRDesc, MO.getOperandNo());
        if (!TRI.hasAGPRs(NewRC))
          return false;

        const MachineOperand *VDst =
            TII.getNamedOperand(*MI, AMDGPU::OpName::vdst);
        const MachineOperand *Src2 =
            TII.getNamedOperand(*MI, AMDGPU::OpName::src2);
        for (const MachineOperand *Op : {VDst, Src2}) {
          if (!Op->isReg())
            continue;

          Register OtherReg = Op->getReg();
          if (OtherReg.isPhysical())
            return false;

          if (OtherReg != Reg && RewriteRegs.insert(OtherReg))
            Worklist.push_back(OtherReg);
        }

        if (!is_contained(RewriteCandidates, MI)) {
          LLVM_DEBUG({
            Register VDstPhysReg = VRM.getPhys(VDst->getReg());
            dbgs() << "Attempting to replace VGPR MFMA with AGPR version:"
                   << " Dst=[" << printReg(VDst->getReg()) << " => "
                   << printReg(VDstPhysReg, &TRI);

            if (Src2->isReg()) {
              Register Src2PhysReg = VRM.getPhys(Src2->getReg());
              dbgs() << "], Src2=[" << printReg(Src2->getReg(), &TRI) << " => "
                     << printReg(Src2PhysReg, &TRI);
            }

            dbgs() << "]: " << MI;
          });

          RewriteCandidates.push_back(MI);
        }

        continue;
      }

      unsigned OpNo = &MO - &MI->getOperand(0);
      NewRC = MI->getRegClassConstraintEffect(OpNo, NewRC, &TII, &TRI);
      if (!NewRC || NewRC == OldRC) {
        LLVM_DEBUG(dbgs() << "User of " << printReg(Reg, &TRI)
                          << " cannot be reassigned to "
                          << (NewRC ? TRI.getRegClassName(NewRC) : "NULL")
                          << ": " << *MI);
        return false;
      }
    }
  }

  return true;
}

bool AMDGPURewriteAGPRCopyMFMAImpl::tryReassigningMFMAChain(
    MachineInstr &MFMA, Register MFMAHintReg, MCPhysReg PhysRegHint) const {
  // src2 and dst have the same physical class constraint; try to preserve
  // the original src2 subclass if one were to exist.
  SmallVector<MachineInstr *, 4> RewriteCandidates = {&MFMA};
  SmallSetVector<Register, 4> RewriteRegs;

  // Make sure we reassign the MFMA we found the copy from first. We want
  // to ensure dst ends up in the physreg we were originally copying to.
  RewriteRegs.insert(MFMAHintReg);

  // We've found av = COPY (MFMA) (or MFMA (v = COPY av)) and need to verify
  // that we can trivially rewrite src2 to use the new AGPR. If we can't
  // trivially replace it, we're going to induce as many copies as we would have
  // emitted in the first place, as well as need to assign another register, and
  // need to figure out where to put them. The live range splitting is smarter
  // than anything we're doing here, so trust it did something reasonable.
  //
  // Note recomputeRegClassExceptRewritable will consider the constraints of
  // this MFMA's src2 as well as the src2/dst of any transitive MFMA users.
  if (!recomputeRegClassExceptRewritable(MFMAHintReg, RewriteCandidates,
                                         RewriteRegs)) {
    LLVM_DEBUG(dbgs() << "Could not recompute the regclass of dst reg "
                      << printReg(MFMAHintReg, &TRI) << '\n');
    return false;
  }

  // If src2 and dst are different registers, we need to also reassign the
  // input to an available AGPR if it is compatible with all other uses.
  //
  // If we can't reassign it, we'd need to introduce a different copy
  // which is likely worse than the copy we'd be saving.
  //
  // It's likely that the MFMA is used in sequence with other MFMAs; if we
  // cannot migrate the full use/def chain of MFMAs, we would need to
  // introduce intermediate copies somewhere. So we only make the
  // transform if all the interfering MFMAs can also be migrated. Collect
  // the set of rewritable MFMAs and check if we can assign an AGPR at
  // that point.
  //
  // If any of the MFMAs aren't reassignable, we give up and rollback to
  // the original register assignments.

  using RecoloringStack =
      SmallVector<std::pair<const LiveInterval *, MCRegister>, 8>;
  RecoloringStack TentativeReassignments;

  for (Register RewriteReg : RewriteRegs) {
    LiveInterval &LI = LIS.getInterval(RewriteReg);
    TentativeReassignments.push_back({&LI, VRM.getPhys(RewriteReg)});
    LRM.unassign(LI);
  }

  if (!DebugCounter::shouldExecute(RewriteAGPRCopyMFMACounter) ||
      !attemptReassignmentsToAGPR(RewriteRegs, PhysRegHint)) {
    // Roll back the register assignments to the original state.
    for (auto [LI, OldAssign] : TentativeReassignments) {
      if (VRM.hasPhys(LI->reg()))
        LRM.unassign(*LI);
      LRM.assign(*LI, OldAssign);
    }

    return false;
  }

  // Fixup the register classes of the virtual registers now that we've
  // committed to the reassignments.
  for (Register InterferingReg : RewriteRegs) {
    const TargetRegisterClass *EquivalentAGPRRegClass =
        TRI.getEquivalentAGPRClass(MRI.getRegClass(InterferingReg));
    MRI.setRegClass(InterferingReg, EquivalentAGPRRegClass);
  }

  for (MachineInstr *RewriteCandidate : RewriteCandidates) {
    int NewMFMAOp = AMDGPU::getAGPRFormOp(RewriteCandidate->getOpcode());
    RewriteCandidate->setDesc(TII.get(NewMFMAOp));
    ++NumMFMAsRewrittenToAGPR;
  }

  return true;
}

bool AMDGPURewriteAGPRCopyMFMAImpl::collectJoinableCopies(
    Register Dst, Register Src, SmallVectorImpl<MachineInstr *> &Copies) const {
  const LiveInterval &DstLI = LIS.getInterval(Dst);
  const LiveInterval &SrcLI = LIS.getInterval(Src);

  // Dst must be (partly) defined by copies of Src with matching subregisters.
  // Every other def of Dst must happen where Src is dead, and every def of Src
  // where Dst is dead. Then wherever both are live, Dst holds the value it
  // copied from Src and neither has been redefined, so renaming Dst to Src and
  // deleting the copies preserves every value.
  for (MachineOperand &MO : MRI.def_operands(Dst)) {
    MachineInstr &MI = *MO.getParent();
    bool IsCopyOfSrc =
        MI.isCopy() && MI.getOperand(1).getReg() == Src &&
        MI.getOperand(0).getSubReg() == MI.getOperand(1).getSubReg();
    if (!IsCopyOfSrc) {
      if (SrcLI.liveAt(getDefIndex(MO)))
        return false;
      continue;
    }

    // The copy defines every lane it writes in Dst, also if the lane is
    // undefined in Src (or the whole read is undef). After the join, a use of
    // such a lane would have no reaching def of Src. Do not join then.
    LaneBitmask ReadLanes =
        TRI.getSubRegIndexLaneMask(MI.getOperand(1).getSubReg()) &
        MRI.getMaxLaneMaskForVReg(Src);
    if (getLiveLaneMask(SrcLI, LIS.getInstructionIndex(MI), MRI, ReadLanes) !=
        ReadLanes)
      return false;
    Copies.push_back(&MI);
  }
  if (Copies.empty())
    return false;

  return none_of(MRI.def_operands(Src), [&](const MachineOperand &MO) {
    return DstLI.liveAt(getDefIndex(MO));
  });
}

void AMDGPURewriteAGPRCopyMFMAImpl::clearStaleDeadFlags(
    Register Reg, const LiveInterval &LI) const {
  for (MachineOperand &MO : MRI.def_operands(Reg)) {
    if (!MO.isDead())
      continue;
    // The def is read if a lane it writes is live after the instruction.
    LaneBitmask Lanes = TRI.getSubRegIndexLaneMask(MO.getSubReg());
    if (getLiveLaneMask(LI, getDefIndex(MO).getDeadSlot(), MRI, Lanes).any())
      MO.setIsDead(false);
  }
}

void AMDGPURewriteAGPRCopyMFMAImpl::eraseCopy(MachineInstr &Copy) const {
  // SplitKit bundles the copies of a register that no single subregister index
  // covers. Remove only the copy, not the bundle.
  if (Copy.isBundled()) {
    LIS.getSlotIndexes()->removeSingleMachineInstrFromMaps(Copy);
    Copy.eraseFromBundle();
    return;
  }
  LIS.RemoveMachineInstrFromMaps(Copy);
  Copy.eraseFromParent();
}

bool AMDGPURewriteAGPRCopyMFMAImpl::tryJoinCopyPair(Register Dst,
                                                    Register Src) const {
  // Only join registers that ended up in AGPRs; VGPR copies are the
  // coalescer's business.
  MCRegister DstPhys = getAssignedAGPR(Dst), SrcPhys = getAssignedAGPR(Src);
  if (!DstPhys || !SrcPhys || DstPhys == SrcPhys)
    return false;

  if (VRM.getStackSlot(Dst) != VirtRegMap::NO_STACK_SLOT ||
      VRM.getStackSlot(Src) != VirtRegMap::NO_STACK_SLOT)
    return false;

  const TargetRegisterClass *SrcRC = MRI.getRegClass(Src);
  const TargetRegisterClass *JoinedRC =
      TRI.getCommonSubClass(MRI.getRegClass(Dst), SrcRC);
  if (!JoinedRC)
    return false;

  SmallVector<MachineInstr *, 4> Copies;
  if (!collectJoinableCopies(Dst, Src, Copies))
    return false;

  // Take Dst and Src out of the matrix, so that their own registers count as
  // free.
  LiveInterval &DstLI = LIS.getInterval(Dst);
  LRM.unassign(DstLI);
  LRM.unassign(LIS.getInterval(Src));

  // The joined live range can be larger than the two old ranges together: a
  // read of a lane that is undefined in one register can see a def of that
  // lane in the other register. Rename tentatively, with the copies detached
  // from Src, and check the interference of the range that will be assigned.
  SmallVector<MachineOperand *, 16> DstOperands;
  for (MachineOperand &MO : MRI.reg_operands(Dst)) {
    if (!is_contained(Copies, MO.getParent()))
      DstOperands.push_back(&MO);
  }
  // Set the operands of Dst outside the copies to NewReg, the source of the
  // copies to CopySrc and the class of Src to RC, then recompute the interval
  // of Src.
  auto RenameAndRecomputeSrc =
      [&](Register NewReg, Register CopySrc,
          const TargetRegisterClass *RC) -> LiveInterval & {
    for (MachineOperand *MO : DstOperands)
      MO->setReg(NewReg);
    for (MachineInstr *Copy : Copies)
      Copy->getOperand(1).setReg(CopySrc);
    MRI.setRegClass(Src, RC);
    LIS.removeInterval(Src);
    LiveInterval &LI = LIS.createAndComputeVirtRegInterval(Src);
    // The interference queries cached for Src are for its old live range.
    LRM.invalidateVirtRegs();
    return LI;
  };

  LiveInterval &Joined = RenameAndRecomputeSrc(Src, Dst, JoinedRC);
  // Prefer Src's register, which the copies read from, then Dst's. Else take
  // any free register of the class, but not a callee-saved one the function
  // does not use yet, which would need a save and restore.
  auto IsFree = [&](MCRegister Cand) {
    return JoinedRC->contains(Cand) &&
           LRM.checkInterference(Joined, Cand) == LiveRegMatrix::IK_Free;
  };
  MCRegister Target;
  if (IsFree(SrcPhys)) {
    Target = SrcPhys;
  } else if (IsFree(DstPhys)) {
    Target = DstPhys;
  } else {
    for (MCRegister Cand : RegClassInfo.getOrder(JoinedRC)) {
      if ((!RegClassInfo.getLastCalleeSavedAlias(Cand) ||
           LRM.isPhysRegUsed(Cand)) &&
          IsFree(Cand)) {
        Target = Cand;
        break;
      }
    }
  }
  if (!Target) {
    LLVM_DEBUG(dbgs() << "Not joining " << printReg(Dst, &TRI) << " into "
                      << printReg(Src, &TRI)
                      << ": the joined live range interferes\n");
    LRM.assign(DstLI, DstPhys);
    LRM.assign(RenameAndRecomputeSrc(Dst, Src, SrcRC), SrcPhys);
    return false;
  }

  LLVM_DEBUG(dbgs() << "Joining " << printReg(Dst, &TRI) << " into "
                    << printReg(Src, &TRI) << " in " << printReg(Target, &TRI)
                    << ", removing " << Copies.size() << " copies\n");

  for (MachineInstr *Copy : Copies)
    eraseCopy(*Copy);
  LIS.removeInterval(Dst);
  // Src's range now extends over Dst's; a kill flag on Src may be stale.
  MRI.clearKillFlags(Src);
  // A def that was dead can be read now, by a use of a lane that was undefined
  // in the other register.
  clearStaleDeadFlags(Src, Joined);
  LRM.assign(Joined, Target);
  if (LDV)
    LDV->splitRegister(Dst, {Src}, LIS);
  ++NumCopyRelatedRegsJoined;
  return true;
}

bool AMDGPURewriteAGPRCopyMFMAImpl::joinWithCopyPartner(Register Reg) const {
  // Collect the pairs first: trying a join rewrites the use lists.
  SmallSetVector<std::pair<Register, Register>, 4> Pairs;
  for (MachineInstr &MI : MRI.reg_nodbg_instructions(Reg)) {
    if (MI.isCopy() && MI.getOperand(0).getReg().isVirtual() &&
        MI.getOperand(1).getReg().isVirtual())
      Pairs.insert({MI.getOperand(0).getReg(), MI.getOperand(1).getReg()});
  }

  return any_of(Pairs, [&](std::pair<Register, Register> Pair) {
    return tryJoinCopyPair(Pair.first, Pair.second);
  });
}

void AMDGPURewriteAGPRCopyMFMAImpl::joinAllCopyPartners() const {
  // A join fails if no AGPR is free for the joined live range. Another join
  // can free one, so retry until nothing more joins.
  bool Joined = true;
  while (Joined) {
    Joined = false;
    for (unsigned I = 0, E = MRI.getNumVirtRegs(); I != E; ++I) {
      Register VReg = Register::index2VirtReg(I);
      if (getAssignedAGPR(VReg))
        Joined |= joinWithCopyPartner(VReg);
    }
  }
}

/// Attempt to reassign the registers in \p InterferingRegs to be AGPRs, with a
/// preference to use \p PhysReg first. Returns false if the reassignments
/// cannot be trivially performed.
bool AMDGPURewriteAGPRCopyMFMAImpl::attemptReassignmentsToAGPR(
    SmallSetVector<Register, 4> &InterferingRegs, MCPhysReg PrefPhysReg) const {
  // FIXME: The ordering may matter here, but we're just taking uselistorder
  // with the special case of ensuring to process the starting instruction
  // first. We probably should extract the priority advisor out of greedy and
  // use that ordering.
  for (Register InterferingReg : InterferingRegs) {
    LiveInterval &ReassignLI = LIS.getInterval(InterferingReg);
    const TargetRegisterClass *EquivalentAGPRRegClass =
        TRI.getEquivalentAGPRClass(MRI.getRegClass(InterferingReg));

    MCPhysReg Assignable = AMDGPU::NoRegister;
    if (EquivalentAGPRRegClass->contains(PrefPhysReg) &&
        LRM.checkInterference(ReassignLI, PrefPhysReg) ==
            LiveRegMatrix::IK_Free) {
      // First try to assign to the AGPR we were already copying to. This
      // should be the first assignment we attempt. We have to guard
      // against the use being a subregister (which doesn't have an exact
      // class match).

      // TODO: If this does happen to be a subregister use, we should
      // still try to assign to a subregister of the original copy result.
      Assignable = PrefPhysReg;
    } else {
      ArrayRef<MCPhysReg> AllocOrder =
          RegClassInfo.getOrder(EquivalentAGPRRegClass);
      for (MCPhysReg Reg : AllocOrder) {
        if (LRM.checkInterference(ReassignLI, Reg) == LiveRegMatrix::IK_Free) {
          Assignable = Reg;
          break;
        }
      }
    }

    if (!Assignable) {
      LLVM_DEBUG(dbgs() << "Unable to reassign VGPR "
                        << printReg(InterferingReg, &TRI)
                        << " to a free AGPR\n");
      return false;
    }

    LLVM_DEBUG(dbgs() << "Reassigning VGPR " << printReg(InterferingReg, &TRI)
                      << " to " << printReg(Assignable, &TRI) << '\n');
    LRM.assign(ReassignLI, Assignable);
  }

  return true;
}

/// Identify copies that look like:
/// %vdst:vgpr = V_MFMA_.. %src0:av, %src1:av, %src2:vgpr
/// %agpr = COPY %vgpr
///
/// Then try to replace the transitive uses of %src2 and %vdst with the AGPR
/// versions of the MFMA. This should cover the common case.
bool AMDGPURewriteAGPRCopyMFMAImpl::tryFoldCopiesToAGPR(
    Register VReg, MCRegister AssignedAGPR) const {
  bool MadeChange = false;
  for (MachineInstr &UseMI : MRI.def_instructions(VReg)) {
    if (!UseMI.isCopy())
      continue;

    Register CopySrcReg = UseMI.getOperand(1).getReg();
    if (!CopySrcReg.isVirtual())
      continue;

    // Loop phis copied to AGPR, e.g.
    //
    // loop:
    //   %phi:vgpr = COPY %mfma:vgpr
    //   %mfma:vgpr = V_MFMA_xxx_vgprcd_e64 %a, %b, %phi
    //   s_cbranch_vccnz loop
    //
    // endloop:
    //   %agpr = mfma
    //
    // are rewritten like any other chain; joinAllCopyPartners then joins %phi
    // with %mfma so the loop-carried copy is not an AGPR-to-AGPR copy in the
    // loop.

    for (MachineInstr &CopySrcDefMI : MRI.def_instructions(CopySrcReg)) {
      if (isRewriteCandidate(CopySrcDefMI) &&
          tryReassigningMFMAChain(
              CopySrcDefMI, CopySrcDefMI.getOperand(0).getReg(), AssignedAGPR))
        MadeChange = true;
    }
  }

  return MadeChange;
}

/// Identify copies that look like:
/// %src:vgpr = COPY %src:agpr
/// %vdst:vgpr = V_MFMA_... %src0:av, %src1:av, %src:vgpr
///
/// Then try to replace the transitive uses of %src2 and %vdst with the AGPR
/// versions of the MFMA. This should cover rarer cases, and will generally be
/// redundant with tryFoldCopiesToAGPR.
bool AMDGPURewriteAGPRCopyMFMAImpl::tryFoldCopiesFromAGPR(
    Register VReg, MCRegister AssignedAGPR) const {
  bool MadeChange = false;
  for (MachineInstr &UseMI : MRI.use_instructions(VReg)) {
    if (!UseMI.isCopy())
      continue;

    Register CopyDstReg = UseMI.getOperand(0).getReg();
    if (!CopyDstReg.isVirtual())
      continue;
    for (MachineOperand &CopyUseMO : MRI.reg_nodbg_operands(CopyDstReg)) {
      if (!CopyUseMO.readsReg())
        continue;

      MachineInstr &CopyUseMI = *CopyUseMO.getParent();
      if (isRewriteCandidate(CopyUseMI)) {
        if (tryReassigningMFMAChain(CopyUseMI, CopyDstReg,
                                    VRM.getPhys(CopyDstReg)))
          MadeChange = true;
      }
    }
  }

  return MadeChange;
}

void AMDGPURewriteAGPRCopyMFMAImpl::replaceSpillWithCopyToVReg(
    MachineInstr &SpillMI, int SpillFI, Register VReg) const {
  const DebugLoc &DL = SpillMI.getDebugLoc();
  MachineBasicBlock &MBB = *SpillMI.getParent();
  MachineInstr *NewCopy;
  if (SpillMI.mayStore()) {
    NewCopy = BuildMI(MBB, SpillMI, DL, TII.get(TargetOpcode::COPY), VReg)
                  .add(SpillMI.getOperand(0));
  } else {
    NewCopy = BuildMI(MBB, SpillMI, DL, TII.get(TargetOpcode::COPY))
                  .add(SpillMI.getOperand(0))
                  .addReg(VReg);
  }

  LIS.ReplaceMachineInstrInMaps(SpillMI, *NewCopy);
  SpillMI.eraseFromParent();
}

void AMDGPURewriteAGPRCopyMFMAImpl::collectSpillIndexUses(
    ArrayRef<LiveInterval *> StackIntervals, SpillReferenceMap &Map) const {

  SmallSet<int, 4> NeededFrameIndexes;
  for (const LiveInterval *LI : StackIntervals)
    NeededFrameIndexes.insert(LI->reg().stackSlotIndex());

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      for (MachineOperand &MO : MI.operands()) {
        if (!MO.isFI() || !NeededFrameIndexes.count(MO.getIndex()))
          continue;

        if (TII.isVGPRSpill(MI)) {
          SmallVector<MachineInstr *, 4> &References = Map[MO.getIndex()];
          References.push_back(&MI);
          break;
        }

        // Verify this was really a spill instruction, if it's not just ignore
        // all uses.

        // TODO: This should probably be verifier enforced.
        NeededFrameIndexes.erase(MO.getIndex());
        Map.erase(MO.getIndex());
      }
    }
  }
}

bool AMDGPURewriteAGPRCopyMFMAImpl::isLoadJointlyDominatedByStores(
    const MachineInstr &LoadMI, const LiveInterval &SlotLI,
    const SmallPtrSetImpl<MachineBasicBlock *> &StoreFreeReachable) const {
  const MachineBasicBlock *LoadMBB = LoadMI.getParent();
  if (!MDT.isReachableFromEntry(LoadMBB))
    return true;

  // Check if every path passed through a store block.
  if (!StoreFreeReachable.contains(LoadMBB))
    return true;

  // Otherwise, there exists a path to this block that has not seen any store
  // yet. We must ensure that within this block there is a store to this slot
  // before the load. Consult the slot's LiveStacks interval: a store to the
  // slot before the load means the slot is not live into this block but is
  // live at the load. If the load reads an undef value, the slot is not live
  // at the load, failing the joint-dominance check.
  SlotIndex LoadIdx = LIS.getInstructionIndex(LoadMI);
  return SlotLI.liveAt(LoadIdx) && !LIS.isLiveInToMBB(SlotLI, LoadMBB);
}

void AMDGPURewriteAGPRCopyMFMAImpl::eliminateSpillsOfReassignedVGPRs() const {
  unsigned NumSlots = LSS.getNumIntervals();
  if (NumSlots == 0)
    return;

  MachineFrameInfo &MFI = MF.getFrameInfo();

  SmallVector<LiveInterval *, 32> StackIntervals;
  StackIntervals.reserve(NumSlots);

  for (auto &[Slot, LI] : LSS) {
    if (!MFI.isSpillSlotObjectIndex(Slot) || MFI.isDeadObjectIndex(Slot))
      continue;

    const TargetRegisterClass *RC = LSS.getIntervalRegClass(Slot);
    if (TRI.hasVGPRs(RC))
      StackIntervals.push_back(&LI);
  }

  sort(StackIntervals, [](const LiveInterval *A, const LiveInterval *B) {
    // The ordering has to be strictly weak.
    /// Sort heaviest intervals first to prioritize their unspilling
    if (A->weight() != B->weight())
      return A->weight() > B->weight();

    if (A->getSize() != B->getSize())
      return A->getSize() > B->getSize();

    // Tie breaker by number to avoid need for stable sort
    return A->reg().stackSlotIndex() < B->reg().stackSlotIndex();
  });

  // FIXME: The APIs for dealing with the LiveInterval of a frame index are
  // cumbersome. LiveStacks owns its LiveIntervals which refer to stack
  // slots. We cannot use the usual LiveRegMatrix::assign and unassign on these,
  // and must create a substitute virtual register to do so. This makes
  // incremental updating here difficult; we need to actually perform the IR
  // mutation to get the new vreg references in place to compute the register
  // LiveInterval to perform an assignment to track the new interference
  // correctly, and we can't simply migrate the LiveInterval we already have.
  //
  // To avoid walking through the entire function for each index, pre-collect
  // all the instructions slot referencess.

  DenseMap<int, SmallVector<MachineInstr *, 4>> SpillSlotReferences;
  collectSpillIndexUses(StackIntervals, SpillSlotReferences);

  for (LiveInterval *LI : StackIntervals) {
    int Slot = LI->reg().stackSlotIndex();
    auto SpillReferences = SpillSlotReferences.find(Slot);
    if (SpillReferences == SpillSlotReferences.end())
      continue;

    // For each spill reload, every path from entry to the reload must pass
    // through at least one spill store to the same stack slot.
    SmallPtrSet<MachineBasicBlock *, 4> StoreBlocks;
    for (MachineInstr *MI : SpillReferences->second) {
      if (MI->mayStore() && MDT.isReachableFromEntry(MI->getParent()))
        StoreBlocks.insert(MI->getParent());
    }

    if (StoreBlocks.empty()) {
      LLVM_DEBUG(dbgs() << "Skipping " << printReg(LI->reg(), &TRI)
                        << ": no reachable stores\n");
      continue;
    }

    // Compute blocks reachable from entry without passing through a store
    // block.
    MachineBasicBlock &EntryMBB = MF.front();
    SmallPtrSet<MachineBasicBlock *, 16> StoreFreeReachable = {&EntryMBB};
    SmallVector<MachineBasicBlock *, 16> Worklist = {&EntryMBB};

    while (!Worklist.empty()) {
      MachineBasicBlock *MBB = Worklist.pop_back_val();
      if (StoreBlocks.contains(MBB))
        continue;

      for (MachineBasicBlock *Succ : MBB->successors()) {
        if (StoreFreeReachable.insert(Succ).second)
          Worklist.push_back(Succ);
      }
    }

    // Every reachable reload must be jointly dominated by the slot's stores.
    if (!llvm::all_of(SpillReferences->second, [&](const MachineInstr *MI) {
          return !MI->mayLoad() ||
                 isLoadJointlyDominatedByStores(*MI, *LI, StoreFreeReachable);
        })) {
      LLVM_DEBUG(
          dbgs() << "Skipping " << printReg(LI->reg(), &TRI)
                 << ": some reachable load not jointly dominated by stores\n");
      continue;
    }

    const TargetRegisterClass *RC = LSS.getIntervalRegClass(Slot);

    LLVM_DEBUG(dbgs() << "Trying to eliminate " << printReg(LI->reg(), &TRI)
                      << " by reassigning\n");

    ArrayRef<MCPhysReg> AllocOrder = RegClassInfo.getOrder(RC);

    // The stack slot's LiveInterval may be discontiguous: a slot can be live
    // in memory around a spill store and around a much later reload. Once
    // we unspill the slot into a register, however, the value must reside in
    // that register continuously from its first reference to its last (modulo
    // the live range splitting that happens later below). Checking
    // interference against the slot's discontiguous interval could let us pick
    // a PhysReg that is busy inside a gap, corrupting it. Instead, check
    // interference over the range the replacement register will occupy.
    for (MCPhysReg PhysReg : AllocOrder) {
      if (LRM.checkInterference(LI->beginIndex(), LI->endIndex(), PhysReg))
        continue;

      LLVM_DEBUG(dbgs() << "Reassigning " << *LI << " to "
                        << printReg(PhysReg, &TRI) << '\n');

      const TargetRegisterClass *RC = LSS.getIntervalRegClass(Slot);
      Register NewVReg = MRI.createVirtualRegister(RC);

      for (MachineInstr *SpillMI : SpillReferences->second)
        replaceSpillWithCopyToVReg(*SpillMI, Slot, NewVReg);

      // TODO: Transferring the stack slot's LiveInterval instead of recomputing
      // would not be a straight copy: its segments would have to be widened to
      // cover each store's reaching reloads.
      LiveInterval &NewLI = LIS.createAndComputeVirtRegInterval(NewVReg);
      VRM.grow();

      // A spill slot can be stored to multiple times, so the replacement
      // vreg may have multiple disconnected live range components. Split
      // them into separate vregs to maintain the single-component invariant.
      SmallVector<LiveInterval *, 4> SplitLIs;
      LIS.splitSeparateComponents(NewLI, SplitLIs);

      LLVM_DEBUG({
        if (!SplitLIs.empty()) {
          dbgs() << "Split unspilled interval into " << (SplitLIs.size() + 1)
                 << " components\n";
        }
      });

      LRM.assign(NewLI, PhysReg);
      for (LiveInterval *SplitLI : SplitLIs) {
        VRM.grow();
        LRM.assign(*SplitLI, PhysReg);
      }

      MFI.RemoveStackObject(Slot);
      break;
    }
  }
}

bool AMDGPURewriteAGPRCopyMFMAImpl::run(MachineFunction &MF) const {
  assert(RegClassInfo.hasCurrentReservedRegs() &&
         "RegisterClassInfo misses a change to the reserved registers");

  // This only applies on subtargets that have a configurable AGPR vs. VGPR
  // allocation.
  if (!ST.hasGFX90AInsts())
    return false;

  // Early exit if no AGPRs were assigned.
  if (!LRM.isPhysRegUsed(AMDGPU::AGPR0)) {
    LLVM_DEBUG(dbgs() << "skipping function that did not allocate AGPRs\n");
    return false;
  }

  bool MadeChange = false;

  for (unsigned I = 0, E = MRI.getNumVirtRegs(); I != E; ++I) {
    Register VReg = Register::index2VirtReg(I);
    MCRegister AssignedAGPR = getAssignedAGPR(VReg);
    if (!AssignedAGPR)
      continue;

    if (tryFoldCopiesToAGPR(VReg, AssignedAGPR))
      MadeChange = true;
    if (tryFoldCopiesFromAGPR(VReg, AssignedAGPR))
      MadeChange = true;
  }

  // Join after all rewrites. A join takes an AGPR that a later rewrite could
  // need, and a join may pick any free AGPR only when no rewrite follows.
  if (MadeChange)
    joinAllCopyPartners();

  // If we've successfully rewritten some MFMAs, we've alleviated some VGPR
  // pressure. See if we can eliminate some spills now that those registers are
  // more available.
  if (MadeChange)
    eliminateSpillsOfReassignedVGPRs();

  return MadeChange;
}

class AMDGPURewriteAGPRCopyMFMALegacy : public MachineFunctionPass {
public:
  static char ID;

  AMDGPURewriteAGPRCopyMFMALegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "AMDGPU Rewrite AGPR-Copy-MFMA";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LiveIntervalsWrapperPass>();
    AU.addRequired<VirtRegMapWrapperLegacy>();
    AU.addRequired<LiveRegMatrixWrapperLegacy>();
    AU.addRequired<LiveStacksWrapperLegacy>();
    AU.addRequired<MachineRegisterClassInfoWrapperPass>();
    AU.addRequired<MachineDominatorTreeWrapperPass>();

    AU.addPreserved<LiveIntervalsWrapperPass>();
    AU.addPreserved<VirtRegMapWrapperLegacy>();
    AU.addPreserved<LiveRegMatrixWrapperLegacy>();
    AU.addPreserved<LiveStacksWrapperLegacy>();
    AU.addPreserved<MachineRegisterClassInfoWrapperPass>();
    AU.addPreserved<MachineDominatorTreeWrapperPass>();

    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // End anonymous namespace.

INITIALIZE_PASS_BEGIN(AMDGPURewriteAGPRCopyMFMALegacy, DEBUG_TYPE,
                      "AMDGPU Rewrite AGPR-Copy-MFMA", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(VirtRegMapWrapperLegacy)
INITIALIZE_PASS_DEPENDENCY(LiveRegMatrixWrapperLegacy)
INITIALIZE_PASS_DEPENDENCY(LiveStacksWrapperLegacy)
INITIALIZE_PASS_DEPENDENCY(MachineRegisterClassInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_END(AMDGPURewriteAGPRCopyMFMALegacy, DEBUG_TYPE,
                    "AMDGPU Rewrite AGPR-Copy-MFMA", false, false)

char AMDGPURewriteAGPRCopyMFMALegacy::ID = 0;

char &llvm::AMDGPURewriteAGPRCopyMFMALegacyID =
    AMDGPURewriteAGPRCopyMFMALegacy::ID;

bool AMDGPURewriteAGPRCopyMFMALegacy::runOnMachineFunction(
    MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  auto &VRM = getAnalysis<VirtRegMapWrapperLegacy>().getVRM();
  auto &LRM = getAnalysis<LiveRegMatrixWrapperLegacy>().getLRM();
  auto &LIS = getAnalysis<LiveIntervalsWrapperPass>().getLIS();
  auto &LSS = getAnalysis<LiveStacksWrapperLegacy>().getLS();
  auto &RCI = getAnalysis<MachineRegisterClassInfoWrapperPass>().getRCI();
  auto &MDT = getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  auto *LDVWrapper = getAnalysisIfAvailable<LiveDebugVariablesWrapperLegacy>();
  AMDGPURewriteAGPRCopyMFMAImpl Impl(MF, VRM, LRM, LIS, LSS, RCI, MDT,
                                     LDVWrapper ? &LDVWrapper->getLDV()
                                                : nullptr);
  return Impl.run(MF);
}

PreservedAnalyses
AMDGPURewriteAGPRCopyMFMAPass::run(MachineFunction &MF,
                                   MachineFunctionAnalysisManager &MFAM) {
  VirtRegMap &VRM = MFAM.getResult<VirtRegMapAnalysis>(MF);
  LiveRegMatrix &LRM = MFAM.getResult<LiveRegMatrixAnalysis>(MF);
  LiveIntervals &LIS = MFAM.getResult<LiveIntervalsAnalysis>(MF);
  LiveStacks &LSS = MFAM.getResult<LiveStacksAnalysis>(MF);
  RegisterClassInfo &RCI = MFAM.getResult<MachineRegisterClassAnalysis>(MF);
  MachineDominatorTree &MDT = MFAM.getResult<MachineDominatorTreeAnalysis>(MF);
  LiveDebugVariables *LDV =
      MFAM.getCachedResult<LiveDebugVariablesAnalysis>(MF);

  AMDGPURewriteAGPRCopyMFMAImpl Impl(MF, VRM, LRM, LIS, LSS, RCI, MDT, LDV);
  if (!Impl.run(MF))
    return PreservedAnalyses::all();
  auto PA = getMachineFunctionPassPreservedAnalyses();
  PA.preserveSet<CFGAnalyses>()
      .preserve<LiveStacksAnalysis>()
      .preserve<VirtRegMapAnalysis>()
      .preserve<SlotIndexesAnalysis>()
      .preserve<LiveIntervalsAnalysis>()
      .preserve<LiveRegMatrixAnalysis>()
      .preserve<MachineRegisterClassAnalysis>()
      .preserve<LiveDebugVariablesAnalysis>();
  return PA;
}
