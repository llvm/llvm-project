//===-- GCNRewriteMFMAForm.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Implements RewriteMFMAFormStage. See GCNRewriteMFMAForm.h.
//
//===----------------------------------------------------------------------===//

#include "GCNRewriteMFMAForm.h"
#include "GCNRegPressure.h"
#include "SIInstrInfo.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "machine-scheduler"

using namespace llvm;

void RewriteMFMAFormStage::findReachingDefs(
    MachineOperand &UseMO, LiveIntervals *LIS,
    SmallVectorImpl<SlotIndex> &DefIdxs) {
  MachineInstr *UseMI = UseMO.getParent();
  LiveInterval &UseLI = LIS->getInterval(UseMO.getReg());
  VNInfo *VNI = UseLI.getVNInfoAt(LIS->getInstructionIndex(*UseMI));

  // If the def is not a PHI, then it must be the only reaching def.
  if (!VNI->isPHIDef()) {
    DefIdxs.push_back(VNI->def);
    return;
  }

  SmallPtrSet<MachineBasicBlock *, 8> Visited = {UseMI->getParent()};
  SmallVector<MachineBasicBlock *, 8> Worklist;

  // Mark the predecessor blocks for traversal
  for (MachineBasicBlock *PredMBB : UseMI->getParent()->predecessors()) {
    Worklist.push_back(PredMBB);
    Visited.insert(PredMBB);
  }

  while (!Worklist.empty()) {
    MachineBasicBlock *CurrMBB = Worklist.pop_back_val();

    SlotIndex CurrMBBEnd = LIS->getMBBEndIdx(CurrMBB);
    VNInfo *VNI = UseLI.getVNInfoAt(CurrMBBEnd.getPrevSlot());

    MachineBasicBlock *DefMBB = LIS->getMBBFromIndex(VNI->def);

    // If there is a def in this block, then add it to the list. This is the
    // reaching def of this path.
    if (!VNI->isPHIDef()) {
      DefIdxs.push_back(VNI->def);
      continue;
    }

    for (MachineBasicBlock *PredMBB : DefMBB->predecessors()) {
      if (Visited.insert(PredMBB).second)
        Worklist.push_back(PredMBB);
    }
  }
}

void RewriteMFMAFormStage::findReachingUses(
    const MachineInstr *DefMI, LiveIntervals *LIS,
    SmallVectorImpl<MachineOperand *> &ReachingUses) {
  SlotIndex DefIdx = LIS->getInstructionIndex(*DefMI);
  for (MachineOperand &UseMO :
       DAG.MRI.use_nodbg_operands(DefMI->getOperand(0).getReg())) {
    SmallVector<SlotIndex, 8> ReachingDefIndexes;
    findReachingDefs(UseMO, LIS, ReachingDefIndexes);

    // If we find a use that contains this DefMI in its reachingDefs, then it is
    // a reaching use.
    if (any_of(ReachingDefIndexes, [DefIdx](SlotIndex RDIdx) {
          return SlotIndex::isSameInstr(RDIdx, DefIdx);
        }))
      ReachingUses.push_back(&UseMO);
  }
}

bool RewriteMFMAFormStage::initGCNSchedStage() {
  // We only need to run this pass if the architecture supports AGPRs.
  // Additionally, we don't use AGPRs at occupancy levels above 1 so there
  // is no need for this pass in that case, either.
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  if (!ST.hasGFX90AInsts() || MFI.getMinWavesPerEU() > 1)
    return false;

  RegionsWithExcessArchVGPR.resize(DAG.Regions.size());
  RegionsWithExcessArchVGPR.reset();
  for (unsigned Region = 0; Region < DAG.Regions.size(); Region++) {
    GCNRegPressure PressureBefore = DAG.Pressure[Region];
    if (PressureBefore.getArchVGPRNum() > ST.getAddressableNumArchVGPRs())
      RegionsWithExcessArchVGPR[Region] = true;
  }

  if (RegionsWithExcessArchVGPR.none())
    return false;

  TII = ST.getInstrInfo();
  SRI = ST.getRegisterInfo();

  std::vector<std::pair<MachineInstr *, unsigned>> RewriteCands;
  DenseMap<MachineBasicBlock *, std::set<Register>> CopyForUse;
  SmallPtrSet<MachineInstr *, 8> CopyForDef;

  if (!initHeuristics(RewriteCands, CopyForUse, CopyForDef))
    return false;

  int64_t Cost = getRewriteCost(RewriteCands, CopyForUse, CopyForDef);

  // If we haven't found the beneficial conditions, prefer the VGPR form which
  // may result in less cross RC copies.
  if (Cost > 0)
    return false;

  return rewrite(RewriteCands);
}

/// Returns true if reaching def \p RD will be in AGPR form after the rewrite
/// and so needs no bridge copy: a candidate MFMA in \p RewriteSet, an
/// AV_MOV_*_IMM_PSEUDO, or a copy from a candidate src2 reg in \p CandSrc2Regs.
/// A non-candidate MFMA stays in VGPR form and still needs a bridge.
static bool isReachingDefAGPRForm(
    MachineInstr *RD, const SmallPtrSetImpl<MachineInstr *> &RewriteSet,
    const DenseSet<Register> &CandSrc2Regs, const SIInstrInfo &TII) {
  if (TII.isMAI(*RD))
    return RewriteSet.contains(RD);
  if (RD->getOpcode() == AMDGPU::AV_MOV_B32_IMM_PSEUDO ||
      RD->getOpcode() == AMDGPU::AV_MOV_B64_IMM_PSEUDO)
    return true;
  if (RD->isCopy() && CandSrc2Regs.contains(RD->getOperand(1).getReg()))
    return true;
  return false;
}

bool RewriteMFMAFormStage::hasUseRequiringVGPR(
    ArrayRef<SlotIndex> Src2ReachingDefs,
    const SmallPtrSetImpl<MachineInstr *> &RewriteSet) {
  for (SlotIndex RDIdx : Src2ReachingDefs) {
    const MachineInstr *RD = DAG.LIS->getInstructionFromIndex(RDIdx);
    SmallVector<MachineOperand *, 8> ReachingUses;
    findReachingUses(RD, DAG.LIS, ReachingUses);
    for (const MachineOperand *UseMO : ReachingUses) {
      const MachineInstr *UseMI = UseMO->getParent();
      if (UseMI->isCopy())
        continue;
      if (TII->isMAI(*UseMI) && RewriteSet.contains(UseMI))
        continue;
      return true;
    }
  }
  return false;
}

void RewriteMFMAFormStage::resetRewriteCandsToVGPR(
    ArrayRef<std::pair<MachineInstr *, unsigned>> RewriteCands) {
  for (auto [MI, OriginalOpcode] : RewriteCands) {
    assert(TII->isMAI(*MI));
    const TargetRegisterClass *ADefRC =
        DAG.MRI.getRegClass(MI->getOperand(0).getReg());
    const TargetRegisterClass *VDefRC = SRI->getEquivalentVGPRClass(ADefRC);
    DAG.MRI.setRegClass(MI->getOperand(0).getReg(), VDefRC);
    MI->setDesc(TII->get(OriginalOpcode));

    MachineOperand *Src2 = TII->getNamedOperand(*MI, AMDGPU::OpName::src2);
    if (!Src2->isReg())
      continue;

    // Have to get src types separately since subregs may cause C and D
    // registers to be different types even though the actual operand is
    // the same size.
    const TargetRegisterClass *AUseRC = DAG.MRI.getRegClass(Src2->getReg());
    const TargetRegisterClass *VUseRC = SRI->getEquivalentVGPRClass(AUseRC);
    DAG.MRI.setRegClass(Src2->getReg(), VUseRC);
  }
}

bool RewriteMFMAFormStage::isRewriteCandidate(MachineInstr *MI) const {
  if (!static_cast<const SIInstrInfo *>(DAG.TII)->isMAI(*MI))
    return false;
  if (AMDGPU::getAGPRFormOp(MI->getOpcode()) == -1)
    return false;
  // Reject candidates whose users force an unavoidable bridge copy.
  Register DstReg = MI->getOperand(0).getReg();
  for (const MachineInstr &UseMI : DAG.MRI.use_nodbg_instructions(DstReg)) {
    if (!TII->isMAI(UseMI) && !UseMI.isCopy())
      return false;
  }
  return true;
}

bool RewriteMFMAFormStage::initHeuristics(
    std::vector<std::pair<MachineInstr *, unsigned>> &RewriteCands,
    DenseMap<MachineBasicBlock *, std::set<Register>> &CopyForUse,
    SmallPtrSetImpl<MachineInstr *> &CopyForDef) {
  bool Changed = false;

  // Collect the candidate group, its members share AGPR-form operands
  // post-rewrite, so reaching defs feeding any member don't need bridge copy.
  SmallPtrSet<MachineInstr *, 16> RewriteSet;
  DenseSet<Register> CandSrc2Regs;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (!isRewriteCandidate(&MI))
        continue;
      RewriteSet.insert(&MI);
      MachineOperand *Src2 = TII->getNamedOperand(MI, AMDGPU::OpName::src2);
      if (Src2 && Src2->isReg())
        CandSrc2Regs.insert(Src2->getReg());
    }
  }

  // Prepare for the heuristics
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (!isRewriteCandidate(&MI))
        continue;

      int ReplacementOp = AMDGPU::getAGPRFormOp(MI.getOpcode());
      assert(ReplacementOp != -1);

      RewriteCands.push_back({&MI, MI.getOpcode()});
      MI.setDesc(TII->get(ReplacementOp));

      MachineOperand *Src2 = TII->getNamedOperand(MI, AMDGPU::OpName::src2);
      if (Src2->isReg()) {
        SmallVector<SlotIndex, 8> Src2ReachingDefs;
        findReachingDefs(*Src2, DAG.LIS, Src2ReachingDefs);

        // If src2 has a use that must remain VGPR, it cannot be reclassified to
        // AGPR.
        bool Src2NeedsVGPR = hasUseRequiringVGPR(Src2ReachingDefs, RewriteSet);
        Src2NeedsVGPRCache[&MI] = Src2NeedsVGPR;

        for (SlotIndex RDIdx : Src2ReachingDefs) {
          MachineInstr *RD = DAG.LIS->getInstructionFromIndex(RDIdx);
          if (!Src2NeedsVGPR &&
              isReachingDefAGPRForm(RD, RewriteSet, CandSrc2Regs, *TII))
            continue;
          CopyForDef.insert(RD);
        }
      }

      MachineOperand &Dst = MI.getOperand(0);
      SmallVector<MachineOperand *, 8> DstReachingUses;

      findReachingUses(&MI, DAG.LIS, DstReachingUses);

      for (MachineOperand *RUOp : DstReachingUses) {
        MachineInstr *UserMI = RUOp->getParent();
        // Group members read the AGPR result directly.
        if (TII->isMAI(*UserMI) && RewriteSet.contains(UserMI))
          continue;

        // For any user of the result of the MFMA which is not an MFMA, we
        // insert a copy. For a given register, we will only insert one copy
        // per user block.
        CopyForUse[UserMI->getParent()].insert(RUOp->getReg());

        if (TII->isMAI(*UserMI))
          continue;

        SmallVector<SlotIndex, 8> DstUsesReachingDefs;
        findReachingDefs(*RUOp, DAG.LIS, DstUsesReachingDefs);

        for (SlotIndex RDIndex : DstUsesReachingDefs) {
          MachineInstr *RD = DAG.LIS->getInstructionFromIndex(RDIndex);
          if (TII->isMAI(*RD))
            continue;

          // For any definition of the user of the MFMA which is not an MFMA,
          // we insert a copy. We do this to transform all the reaching defs
          // of this use to AGPR. By doing this, we can insert a copy from
          // AGPR to VGPR at the user rather than after the MFMA.
          CopyForDef.insert(RD);
        }
      }

      // Do the rewrite to allow for updated RP calculation.
      const TargetRegisterClass *VDefRC = DAG.MRI.getRegClass(Dst.getReg());
      const TargetRegisterClass *ADefRC = SRI->getEquivalentAGPRClass(VDefRC);
      DAG.MRI.setRegClass(Dst.getReg(), ADefRC);
      if (Src2->isReg()) {
        // Have to get src types separately since subregs may cause C and D
        // registers to be different types even though the actual operand is
        // the same size.
        const TargetRegisterClass *VUseRC = DAG.MRI.getRegClass(Src2->getReg());
        const TargetRegisterClass *AUseRC = SRI->getEquivalentAGPRClass(VUseRC);
        DAG.MRI.setRegClass(Src2->getReg(), AUseRC);
      }
      Changed = true;
    }
  }

  return Changed;
}

int64_t RewriteMFMAFormStage::getRewriteCost(
    ArrayRef<std::pair<MachineInstr *, unsigned>> RewriteCands,
    const DenseMap<MachineBasicBlock *, std::set<Register>> &CopyForUse,
    const SmallPtrSetImpl<MachineInstr *> &CopyForDef) {
  MachineBlockFrequencyInfo *MBFI = DAG.MBFI;

  int64_t BestSpillCost = 0;
  int64_t Cost = 0;
  uint64_t EntryFreq = MBFI->getEntryFreq().getFrequency();

  std::pair<unsigned, unsigned> MaxVectorRegs =
      ST.getMaxNumVectorRegs(MF.getFunction());
  unsigned ArchVGPRThreshold = MaxVectorRegs.first;
  unsigned AGPRThreshold = MaxVectorRegs.second;
  unsigned CombinedThreshold = ST.getMaxNumVGPRs(MF);

  for (unsigned Region = 0; Region < DAG.Regions.size(); Region++) {
    if (!RegionsWithExcessArchVGPR[Region])
      continue;

    GCNRegPressure &PressureBefore = DAG.Pressure[Region];
    unsigned SpillCostBefore = PressureBefore.getVGPRSpills(
        MF, ArchVGPRThreshold, AGPRThreshold, CombinedThreshold);

    // For the cases we care about (i.e. ArchVGPR usage is greater than the
    // addressable limit), rewriting alone should bring pressure to manageable
    // level. If we find any such region, then the rewrite is potentially
    // beneficial.
    GCNRegPressure PressureAfter = DAG.getRealRegPressure(Region);
    unsigned SpillCostAfter = PressureAfter.getVGPRSpills(
        MF, ArchVGPRThreshold, AGPRThreshold, CombinedThreshold);

    uint64_t BlockFreq =
        MBFI->getBlockFreq(DAG.Regions[Region].first->getParent())
            .getFrequency();

    bool RelativeFreqIsDenom = EntryFreq > BlockFreq;
    uint64_t RelativeFreq = EntryFreq && BlockFreq
                                ? (RelativeFreqIsDenom ? EntryFreq / BlockFreq
                                                       : BlockFreq / EntryFreq)
                                : 1;

    // This assumes perfect spilling / splitting -- using one spill / copy
    // instruction and one restoreFrom / copy for each excess register,
    int64_t SpillCost = ((int)SpillCostAfter - (int)SpillCostBefore) * 2;

    // Also account for the block frequency.
    if (RelativeFreqIsDenom)
      SpillCost /= (int64_t)RelativeFreq;
    else
      SpillCost *= (int64_t)RelativeFreq;

    // If we have increased spilling in any block, just bail.
    if (SpillCost > 0) {
      resetRewriteCandsToVGPR(RewriteCands);
      return SpillCost;
    }

    if (SpillCost < BestSpillCost)
      BestSpillCost = SpillCost;
  }

  // Set the cost to the largest decrease in spill cost in order to not double
  // count spill reductions.
  Cost = BestSpillCost;
  assert(Cost <= 0);

  unsigned CopyCost = 0;

  // For each CopyForDef, increase the cost by the register size while
  // accounting for block frequency.
  for (MachineInstr *DefMI : CopyForDef) {
    Register DefReg = DefMI->getOperand(0).getReg();
    uint64_t DefFreq =
        EntryFreq
            ? MBFI->getBlockFreq(DefMI->getParent()).getFrequency() / EntryFreq
            : 1;

    const TargetRegisterClass *RC = DAG.MRI.getRegClass(DefReg);
    CopyCost += RC->getCopyCost() * DefFreq;
  }

  // Account for CopyForUse copies in each block that the register is used.
  for (auto &[UseBlock, UseRegs] : CopyForUse) {
    uint64_t UseFreq =
        EntryFreq ? MBFI->getBlockFreq(UseBlock).getFrequency() / EntryFreq : 1;

    for (Register UseReg : UseRegs) {
      const TargetRegisterClass *RC = DAG.MRI.getRegClass(UseReg);
      CopyCost += RC->getCopyCost() * UseFreq;
    }
  }

  // Reset the classes that were changed to AGPR for better register bank
  // analysis. We must do rewriting after copy-insertion, as some defs of the
  // register may require VGPR.  Additionally, if we bail out and don't perform
  // the rewrite then these need to be restored anyway.
  resetRewriteCandsToVGPR(RewriteCands);

  return Cost + CopyCost;
}

bool RewriteMFMAFormStage::rewrite(
    ArrayRef<std::pair<MachineInstr *, unsigned>> RewriteCands) {
  DenseMap<MachineInstr *, unsigned> FirstMIToRegion;
  DenseMap<MachineInstr *, unsigned> LastMIToRegion;

  for (unsigned Region = 0; Region < DAG.Regions.size(); Region++) {
    RegionBoundaries Entry = DAG.Regions[Region];
    if (Entry.first == Entry.second)
      continue;

    FirstMIToRegion[&*Entry.first] = Region;
    if (Entry.second != Entry.first->getParent()->end())
      LastMIToRegion[&*Entry.second] = Region;
  }

  // Rewrite the MFMAs to AGPR, and insert any copies as needed.
  // The general assumption of the algorithm (and the previous cost calculation)
  // is that it is better to insert the copies in the MBB of the def of the src2
  // operands, and in the MBB of the user of the dest operands. This is based on
  // the assumption that the MFMAs are likely to appear in loop bodies, while
  // the src2 and dest operands are live-in / live-out of the loop. Due to this
  // design, the algorithm for finding copy insertion points is more
  // complicated.
  //
  // There are three main cases to handle: 1. the reaching defs of the src2
  // operands, 2. the reaching uses of the dst operands, and 3. the reaching
  // defs of the reaching uses of the dst operand.
  //
  // In the first case, we simply insert copies after each of the reaching
  // definitions. In the second case, we collect all the uses of a given dest
  // and organize them by MBB. Then, we insert 1 copy for each MBB before the
  // earliest use. Since the use may have multiple reaching defs, and since we
  // want to replace the register it is using with the result of the copy, we
  // must handle case 3. In the third case, we simply insert a copy after each
  // of the reaching defs to connect to the copy of the reaching uses of the dst
  // reg. This allows us to avoid inserting copies next to the MFMAs.
  //
  // While inserting the copies, we maintain a map of operands which will use
  // different regs (i.e. the result of the copies). For example, a case 1 src2
  // operand will use the register result of the copies after the reaching defs,
  // as opposed to the original register. Now that we have completed our copy
  // analysis and placement, we can bulk update the registers. We do this
  // separately as to avoid complicating the reachingDef and reachingUse
  // queries.
  //
  // While inserting the copies, we also maintain a list or registers which we
  // will want to reclassify as AGPR. After doing the copy insertion and the
  // register replacement, we can finally do the reclassification. This uses the
  // redef map, as the registers we are interested in reclassifying may be
  // replaced by the result of a copy. We must do this after the copy analysis
  // and placement as we must have an accurate redef map -- otherwise we may end
  // up creating illegal instructions.

  // The original registers of the MFMA that need to be reclassified as AGPR.
  DenseSet<Register> RewriteRegs;
  // The map of an original register in the MFMA to a new register (result of a
  // copy) that it should be replaced with.
  DenseMap<Register, Register> RedefMap;
  // The map of the original MFMA registers to the relevant MFMA operands.
  DenseMap<Register, DenseSet<MachineOperand *>> ReplaceMap;
  // The map of reaching defs for a given register -- to avoid duplicate copies.
  DenseMap<Register, SmallPtrSet<MachineInstr *, 8>> ReachingDefCopyMap;
  // The map of reaching uses for a given register by basic block -- to avoid
  // duplicate copies and to calculate per MBB insert pts.
  DenseMap<unsigned, DenseMap<Register, SmallPtrSet<MachineOperand *, 8>>>
      ReachingUseTracker;

  // Collect the candidate group; its members share AGPR-form operands
  // post-rewrite, so reaching defs feeding any member need no bridge copy.
  SmallPtrSet<MachineInstr *, 16> RewriteCandsSet;
  DenseSet<Register> RewriteSrc2Regs;
  for (auto &[MI, OriginalOpcode] : RewriteCands) {
    RewriteCandsSet.insert(MI);
    MachineOperand *Src2 = TII->getNamedOperand(*MI, AMDGPU::OpName::src2);
    if (Src2 && Src2->isReg())
      RewriteSrc2Regs.insert(Src2->getReg());
  }

  for (auto &[MI, OriginalOpcode] : RewriteCands) {
    int ReplacementOp = AMDGPU::getAGPRFormOp(MI->getOpcode());
    if (ReplacementOp == -1)
      continue;
    MI->setDesc(TII->get(ReplacementOp));

    // Case 1: insert copies for the reaching defs of the Src2Reg.
    MachineOperand *Src2 = TII->getNamedOperand(*MI, AMDGPU::OpName::src2);
    if (Src2->isReg()) {
      Register Src2Reg = Src2->getReg();
      if (!Src2Reg.isVirtual())
        return false;

      Register MappedReg = Src2->getReg();
      SmallVector<SlotIndex, 8> Src2ReachingDefs;
      findReachingDefs(*Src2, DAG.LIS, Src2ReachingDefs);
      SmallSetVector<MachineInstr *, 8> Src2DefsReplace;

      // If src2 has a use that must remain VGPR, it cannot be reclassified to
      // AGPR.
      bool Src2NeedsVGPR = Src2NeedsVGPRCache.lookup(MI);

      for (SlotIndex RDIndex : Src2ReachingDefs) {
        MachineInstr *RD = DAG.LIS->getInstructionFromIndex(RDIndex);
        if (!Src2NeedsVGPR &&
            isReachingDefAGPRForm(RD, RewriteCandsSet, RewriteSrc2Regs, *TII))
          continue;

        Src2DefsReplace.insert(RD);
      }

      if (!Src2DefsReplace.empty()) {
        auto RI = RedefMap.find(Src2Reg);
        if (RI != RedefMap.end()) {
          MappedReg = RI->second;
        } else {
          assert(!ReachingDefCopyMap.contains(Src2Reg));
          const TargetRegisterClass *Src2RC = DAG.MRI.getRegClass(Src2Reg);
          const TargetRegisterClass *VGPRRC =
              SRI->getEquivalentVGPRClass(Src2RC);

          // Track the mapping of the original register to the new register.
          MappedReg = DAG.MRI.createVirtualRegister(VGPRRC);
          RedefMap[Src2Reg] = MappedReg;
        }

        // If none exists, create a copy from this reaching def.
        // We may have inserted a copy already in an earlier iteration.
        for (MachineInstr *RD : Src2DefsReplace) {
          // Do not create redundant copies.
          if (ReachingDefCopyMap[Src2Reg].insert(RD).second) {
            MachineInstrBuilder VGPRCopy =
                BuildMI(*RD->getParent(), std::next(RD->getIterator()),
                        RD->getDebugLoc(), TII->get(TargetOpcode::COPY))
                    .addDef(MappedReg, {}, 0)
                    .addUse(Src2Reg, {}, 0);
            DAG.LIS->InsertMachineInstrInMaps(*VGPRCopy);

            // If this reaching def was the last MI in the region, update the
            // region boundaries.
            if (LastMIToRegion.contains(RD)) {
              unsigned UpdateRegion = LastMIToRegion[RD];
              DAG.Regions[UpdateRegion].second = VGPRCopy;
              LastMIToRegion.erase(RD);
            }
          }
        }
      }

      // Track the register for reclassification
      RewriteRegs.insert(Src2Reg);

      // Always insert the operand for replacement. If this corresponds with a
      // chain of tied-def we may not see the VGPR requirement until later.
      ReplaceMap[Src2Reg].insert(Src2);
    }

    // Case 2 and Case 3: insert copies before the reaching uses of the dsts,
    // and after the reaching defs of the reaching uses of the dsts.

    MachineOperand *Dst = &MI->getOperand(0);
    Register DstReg = Dst->getReg();
    if (!DstReg.isVirtual())
      return false;

    Register MappedReg = DstReg;
    SmallVector<MachineOperand *, 8> DstReachingUses;

    SmallVector<MachineOperand *, 8> DstReachingUseCopies;
    SmallVector<MachineInstr *, 8> DstUseDefsReplace;

    findReachingUses(MI, DAG.LIS, DstReachingUses);

    for (MachineOperand *RUOp : DstReachingUses) {
      MachineInstr *UserMI = RUOp->getParent();
      // Group members read the AGPR result directly.
      if (TII->isMAI(*UserMI) && RewriteCandsSet.contains(UserMI))
        continue;

      // If there is a non mai reaching use, then we need a copy.
      if (find(DstReachingUseCopies, RUOp) == DstReachingUseCopies.end())
        DstReachingUseCopies.push_back(RUOp);

      // Non-rewritten MAI: its defs aren't being reclassified.
      if (TII->isMAI(*UserMI))
        continue;

      SmallVector<SlotIndex, 8> DstUsesReachingDefs;
      findReachingDefs(*RUOp, DAG.LIS, DstUsesReachingDefs);

      for (SlotIndex RDIndex : DstUsesReachingDefs) {
        MachineInstr *RD = DAG.LIS->getInstructionFromIndex(RDIndex);
        if (TII->isMAI(*RD))
          continue;

        // If there is a non mai reaching def of this reaching use, then we will
        // need a copy.
        if (find(DstUseDefsReplace, RD) == DstUseDefsReplace.end())
          DstUseDefsReplace.push_back(RD);
      }
    }

    if (!DstUseDefsReplace.empty()) {
      auto RI = RedefMap.find(DstReg);
      if (RI != RedefMap.end()) {
        MappedReg = RI->second;
      } else {
        assert(!ReachingDefCopyMap.contains(DstReg));
        const TargetRegisterClass *DstRC = DAG.MRI.getRegClass(DstReg);
        const TargetRegisterClass *VGPRRC = SRI->getEquivalentVGPRClass(DstRC);

        // Track the mapping of the original register to the new register.
        MappedReg = DAG.MRI.createVirtualRegister(VGPRRC);
        RedefMap[DstReg] = MappedReg;
      }

      // If none exists, create a copy from this reaching def.
      // We may have inserted a copy already in an earlier iteration.
      for (MachineInstr *RD : DstUseDefsReplace) {
        // Do not create reundant copies.
        if (ReachingDefCopyMap[DstReg].insert(RD).second) {
          MachineInstrBuilder VGPRCopy =
              BuildMI(*RD->getParent(), std::next(RD->getIterator()),
                      RD->getDebugLoc(), TII->get(TargetOpcode::COPY))
                  .addDef(MappedReg, {}, 0)
                  .addUse(DstReg, {}, 0);
          DAG.LIS->InsertMachineInstrInMaps(*VGPRCopy);

          // If this reaching def was the last MI in the region, update the
          // region boundaries.
          auto LMI = LastMIToRegion.find(RD);
          if (LMI != LastMIToRegion.end()) {
            unsigned UpdateRegion = LMI->second;
            DAG.Regions[UpdateRegion].second = VGPRCopy;
            LastMIToRegion.erase(RD);
          }
        }
      }
    }

    DenseSet<MachineOperand *> &DstRegSet = ReplaceMap[DstReg];
    // One AGPR→VGPR copy per dst register, shared by all same-block uses.
    Register SameBlockCopyReg;
    MachineInstr *EarliestSameBlockUse = nullptr;
    for (MachineOperand *RU : DstReachingUseCopies) {
      MachineBasicBlock *RUBlock = RU->getParent()->getParent();
      // Just keep track of the reaching use of this register by block. After we
      // have scanned all the MFMAs we can find optimal insert pts.
      if (RUBlock != MI->getParent()) {
        ReachingUseTracker[RUBlock->getNumber()][DstReg].insert(RU);
        continue;
      }

      // Lazily create the copy register on first same-block use.
      if (!SameBlockCopyReg.isValid()) {
        const TargetRegisterClass *DstRC = DAG.MRI.getRegClass(DstReg);
        const TargetRegisterClass *VGPRRC = SRI->getEquivalentVGPRClass(DstRC);
        SameBlockCopyReg = DAG.MRI.createVirtualRegister(VGPRRC);
      }

      // Track the earliest use for copy insertion point.
      MachineInstr *UseInst = RU->getParent();
      if (!EarliestSameBlockUse ||
          SlotIndex::isEarlierInstr(
              DAG.LIS->getInstructionIndex(*UseInst),
              DAG.LIS->getInstructionIndex(*EarliestSameBlockUse)))
        EarliestSameBlockUse = UseInst;
      RU->setReg(SameBlockCopyReg);
    }

    // Insert the copy before the earliest same-block use.
    if (SameBlockCopyReg.isValid()) {
      MachineInstrBuilder VGPRCopy =
          BuildMI(*EarliestSameBlockUse->getParent(),
                  EarliestSameBlockUse->getIterator(), DebugLoc(),
                  TII->get(TargetOpcode::COPY), SameBlockCopyReg)
              .addUse(DstReg, {}, 0);
      DAG.LIS->InsertMachineInstrInMaps(*VGPRCopy);
      DstRegSet.insert(&VGPRCopy->getOperand(1));
    }

    // Track the register for reclassification
    RewriteRegs.insert(DstReg);

    // Insert the dst operand for replacement. If this dst is in a chain of
    // tied-def MFMAs, and the first src2 needs to be replaced with a new reg,
    // all the correspond operands need to be replaced.
    DstRegSet.insert(Dst);
  }

  // Handle the copies for dst uses.
  using RUBType =
      std::pair<unsigned, DenseMap<Register, SmallPtrSet<MachineOperand *, 8>>>;
  for (RUBType RUBlockEntry : ReachingUseTracker) {
    using RUDType = std::pair<Register, SmallPtrSet<MachineOperand *, 8>>;
    for (RUDType RUDst : RUBlockEntry.second) {
      MachineOperand *OpBegin = *RUDst.second.begin();
      SlotIndex InstPt = DAG.LIS->getInstructionIndex(*OpBegin->getParent());

      // Find the earliest use in this block.
      for (MachineOperand *User : RUDst.second) {
        SlotIndex NewInstPt = DAG.LIS->getInstructionIndex(*User->getParent());
        if (SlotIndex::isEarlierInstr(NewInstPt, InstPt))
          InstPt = NewInstPt;
      }

      const TargetRegisterClass *DstRC = DAG.MRI.getRegClass(RUDst.first);
      const TargetRegisterClass *VGPRRC = SRI->getEquivalentVGPRClass(DstRC);
      Register NewUseReg = DAG.MRI.createVirtualRegister(VGPRRC);
      MachineInstr *UseInst = DAG.LIS->getInstructionFromIndex(InstPt);

      MachineInstrBuilder VGPRCopy =
          BuildMI(*UseInst->getParent(), UseInst->getIterator(),
                  UseInst->getDebugLoc(), TII->get(TargetOpcode::COPY))
              .addDef(NewUseReg, {}, 0)
              .addUse(RUDst.first, {}, 0);
      DAG.LIS->InsertMachineInstrInMaps(*VGPRCopy);

      // If this UseInst was the first MI in the region, update the region
      // boundaries.
      auto FI = FirstMIToRegion.find(UseInst);
      if (FI != FirstMIToRegion.end()) {
        unsigned UpdateRegion = FI->second;
        DAG.Regions[UpdateRegion].first = VGPRCopy;
        FirstMIToRegion.erase(UseInst);
      }

      // Replace the operand for all users.
      for (MachineOperand *User : RUDst.second) {
        User->setReg(NewUseReg);
      }

      // Track the copy source operand for replacement.
      ReplaceMap[RUDst.first].insert(&VGPRCopy->getOperand(1));
    }
  }

  // We may have needed to insert copies after the reaching defs of the MFMAs.
  // Replace the original register with the result of the copy for all relevant
  // operands.
  for (std::pair<Register, Register> NewDef : RedefMap) {
    Register OldReg = NewDef.first;
    Register NewReg = NewDef.second;

    // Replace the register for any associated operand in the MFMA chain.
    for (MachineOperand *ReplaceOp : ReplaceMap[OldReg])
      ReplaceOp->setReg(NewReg);
  }

  // Finally, do the reclassification of the MFMA registers.
  for (Register RewriteReg : RewriteRegs) {
    Register RegToRewrite = RewriteReg;

    // Be sure to update the replacement register and not the original.
    auto RI = RedefMap.find(RewriteReg);
    if (RI != RedefMap.end())
      RegToRewrite = RI->second;

    const TargetRegisterClass *CurrRC = DAG.MRI.getRegClass(RegToRewrite);
    const TargetRegisterClass *AGPRRC = SRI->getEquivalentAGPRClass(CurrRC);

    DAG.MRI.setRegClass(RegToRewrite, AGPRRC);
  }

  // Bulk update the LIS.
  DAG.LIS->reanalyze(DAG.MF);
  // Liveins may have been modified for cross RC copies
  RegionPressureMap LiveInUpdater(&DAG, false);
  LiveInUpdater.buildLiveRegMap();

  for (unsigned Region = 0; Region < DAG.Regions.size(); Region++)
    DAG.LiveIns[Region] = LiveInUpdater.getLiveRegsForRegionIdx(Region);

  DAG.Pressure[RegionIdx] = DAG.getRealRegPressure(RegionIdx);

  return true;
}

