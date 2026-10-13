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
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "machine-scheduler"

using namespace llvm;

static cl::opt<bool> DisableRewriteMFMAFormSchedStage(
    "amdgpu-disable-rewrite-mfma-form-sched-stage", cl::Hidden,
    cl::desc("Disable rewrite mfma rewrite scheduling stage"), cl::init(true));

/// What RewriteMFMAFormStage charges for one register it expects to spill: a
/// scratch store plus a reload. Priced against a register-to-register copy,
/// which is one VALU op, so that the cost model does not trade a spill for a
/// handful of copies.
static constexpr int64_t SpillCostPerReg = 20;

bool RewriteMFMAFormStage::isEnabled() {
  return !DisableRewriteMFMAFormSchedStage;
}

void RewriteMFMAFormStage::findReachingDefs(
    const MachineOperand &UseMO, LiveIntervals *LIS,
    SmallVectorImpl<MachineInstr *> &Defs) {
  const LiveInterval &LI = LIS->getInterval(UseMO.getReg());
  SlotIndex UseIdx = LIS->getInstructionIndex(*UseMO.getParent());

  SmallPtrSet<const VNInfo *, 8> Seen;
  SmallVector<const VNInfo *, 8> Worklist;
  if (const VNInfo *VNI = LI.getVNInfoAt(UseIdx))
    Worklist.push_back(VNI);

  while (!Worklist.empty()) {
    const VNInfo *VNI = Worklist.pop_back_val();
    if (!Seen.insert(VNI).second)
      continue;

    // A value that is not a PHI has exactly one def, and it reaches here.
    if (!VNI->isPHIDef()) {
      if (MachineInstr *DefMI = LIS->getInstructionFromIndex(VNI->def))
        if (!is_contained(Defs, DefMI))
          Defs.push_back(DefMI);
      continue;
    }

    MachineBasicBlock *PHIMBB = LIS->getMBBFromIndex(VNI->def);
    for (MachineBasicBlock *Pred : PHIMBB->predecessors())
      if (const VNInfo *PredVNI = LI.getVNInfoBefore(LIS->getMBBEndIdx(Pred)))
        Worklist.push_back(PredVNI);
  }
}

void RewriteMFMAFormStage::findReachingUses(
    const MachineInstr *DefMI, LiveIntervals *LIS,
    SmallVectorImpl<MachineOperand *> &ReachingUses) {
  for (MachineOperand &UseMO :
       DAG.MRI.use_nodbg_operands(DefMI->getOperand(0).getReg())) {
    SmallVector<MachineInstr *, 8> ReachingDefs;
    findReachingDefs(UseMO, LIS, ReachingDefs);

    // If we find a use that contains this DefMI in its reachingDefs, then it is
    // a reaching use.
    if (is_contained(ReachingDefs, DefMI))
      ReachingUses.push_back(&UseMO);
  }
}

/// \returns true if the dst or src2 of \p MI is a physical register. apply()
/// bridges an accumulator operand by creating a new virtual register for it,
/// which it cannot do for a physical one.
static bool hasPhysicalAccOperand(const MachineInstr &MI,
                                  const SIInstrInfo *TII) {
  const MachineOperand *Src2 = TII->getNamedOperand(MI, AMDGPU::OpName::src2);
  return !MI.getOperand(0).getReg().isVirtual() ||
         (Src2 && Src2->isReg() && !Src2->getReg().isVirtual());
}

SmallVector<MachineInstr *, 32> RewriteMFMAFormStage::selectChainPrefix(
    ArrayRef<SmallVector<MachineInstr *, 8>> Chains, unsigned N) const {
  SmallVector<MachineInstr *, 32> Cands;
  for (ArrayRef<MachineInstr *> Chain : Chains.take_front(N))
    Cands.append(Chain.begin(), Chain.end());

  // Chains are sorted by length, so put the selection back in program order.
  sort(Cands, [this](const MachineInstr *A, const MachineInstr *B) {
    return DAG.LIS->getInstructionIndex(*A) < DAG.LIS->getInstructionIndex(*B);
  });
  return Cands;
}

bool RewriteMFMAFormStage::MFMARewriteCost::isBetterThan(
    const MFMARewriteCost &O) const {
  int64_t NegUtil = -(int64_t)AGPRUtil, ONegUtil = -(int64_t)O.AGPRUtil;
  return std::tie(Cost, NegUtil) < std::tie(O.Cost, ONegUtil);
}

unsigned RewriteMFMAFormStage::findBestChainCount(
    ArrayRef<SmallVector<MachineInstr *, 8>> Chains) {
  auto Probe = [&](unsigned N) {
    MFMARewritePlan Plan;
    MFMARewriteCost Result;
    if (buildPlan(Plan, selectChainPrefix(Chains, N)))
      Result = cost(Plan, measure(Plan));
    MFMAToAGPR->setActiveSet(nullptr);
    LLVM_DEBUG(dbgs() << "RewriteMFMAForm: probe " << N << '/' << Chains.size()
                      << " chains, cost " << Result.Cost << ", AGPR util "
                      << Result.AGPRUtil << '\n');
    return Result;
  };

  MFMARewriteCost Best;
  unsigned BestN = 0;

  // The costs of the chains are not additive: each one both relieves ArchVGPR
  // pressure and adds AGPR pressure, so there is a point past which converting
  // more only moves the spills. The cost is not monotonic either, so scan every
  // prefix rather than binary searching for the crossover.
  //
  // Cheapest prefix wins; equal-cost prefixes are broken toward the one holding
  // the most in AGPRs. See MFMARewriteCost::isBetterThan().
  for (unsigned N = 1; N <= Chains.size(); ++N) {
    MFMARewriteCost Result = Probe(N);
    // A rewrite that does not pay for itself is not worth doing at all, however
    // much cheaper it is than the alternatives.
    if (Result.Cost > 0)
      continue;
    if (BestN == 0 || Result.isBetterThan(Best)) {
      Best = Result;
      BestN = N;
    }
  }

  LLVM_DEBUG(dbgs() << "RewriteMFMAForm: best " << BestN << '/' << Chains.size()
                    << " chains, cost " << Best.Cost << '\n');
  return BestN;
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

  LLVM_DEBUG(dbgs() << "RewriteMFMAForm: " << RegionsWithExcessArchVGPR.count()
                    << '/' << DAG.Regions.size()
                    << " regions over the ArchVGPR limit "
                    << ST.getAddressableNumArchVGPRs() << '\n');

  TII = ST.getInstrInfo();
  SRI = ST.getRegisterInfo();
  MFMAToAGPR.emplace(MF, *TII, *SRI, DAG.MRI);

  // Convert whole accumulator chains at a time. A chain needs at most one
  // bridge copy at each end however long it is, so converting only part of one
  // pays the same copies for less pressure relief.
  SmallVector<SmallVector<MachineInstr *, 8>, 16> Chains =
      MFMAToAGPR->partitionChains();

  // Drop a chain with a physical accumulator operand whole, rather than leave
  // the one MFMA behind to pin the rest of its chain to VGPRs.
  erase_if(Chains, [this](ArrayRef<MachineInstr *> Chain) {
    return any_of(Chain, [this](const MachineInstr *MI) {
      if (!hasPhysicalAccOperand(*MI, TII))
        return false;
      LLVM_DEBUG(dbgs() << "Physical accumulator operand in: " << *MI);
      return true;
    });
  });

  if (Chains.empty()) {
    LLVM_DEBUG(dbgs() << "RewriteMFMAForm: nothing to rewrite\n");
    return false;
  }

  // Longest first: a long chain relieves more pressure for the same copies.
  stable_sort(Chains,
              [](ArrayRef<MachineInstr *> A, ArrayRef<MachineInstr *> B) {
                return A.size() > B.size();
              });

  unsigned BestN = findBestChainCount(Chains);
  if (BestN == 0)
    return false;

  MFMARewritePlan Plan;
  if (!buildPlan(Plan, selectChainPrefix(Chains, BestN)))
    return false;

  apply(Plan);
  return true;
}

bool RewriteMFMAFormStage::buildPlan(MFMARewritePlan &Plan,
                                     ArrayRef<MachineInstr *> Cands) {
  if (Cands.empty())
    return false;

  Plan.MFMAs.assign(Cands.begin(), Cands.end());

  // The candidates share AGPR-form operands post-rewrite, so neither their
  // operands nor the defs feeding them need a bridge copy. Registering them as
  // the active set is what tells the shared analysis to ignore them.
  Plan.MFMASet.insert_range(Plan.MFMAs);
  MFMAToAGPR->setActiveSet(&Plan.MFMASet);

  // Registers apply() will replace with a fresh one, mirrored here so that the
  // analysis sees the same state apply() will. A replaced register keeps its
  // VGPR class, so it can never be recolored.
  DenseSet<Register> RedefRegs;

  for (MachineInstr *MI : Plan.MFMAs) {
    MFMARewriteEntry Entry;
    Entry.MFMA = MI;

    Register DstReg = MI->getOperand(0).getReg();
    Plan.RecolorRegs.insert(DstReg);

    MachineOperand *Src2 = TII->getNamedOperand(*MI, AMDGPU::OpName::src2);
    if (Src2->isReg()) {
      Register Src2Reg = Src2->getReg();
      Plan.RecolorRegs.insert(Src2Reg);

      // If src2 can be recolored to AGPR, its reaching defs are reclassified
      // along with it and no bridge copy is needed. Otherwise every reaching
      // def outside the group has to be bridged; a group member already
      // produces its result in an AGPR.
      if (RedefRegs.contains(Src2Reg) ||
          !MFMAToAGPR->canRecolorToAGPR(Src2Reg)) {
        SmallVector<MachineInstr *, 8> ReachingDefs;
        findReachingDefs(*Src2, DAG.LIS, ReachingDefs);
        for (MachineInstr *RD : ReachingDefs) {
          if (Plan.MFMASet.contains(RD))
            continue;
          if (!is_contained(Entry.Src2Bridges, RD))
            Entry.Src2Bridges.push_back(RD);
        }
      }

      if (!Entry.Src2Bridges.empty()) {
        RedefRegs.insert(Src2Reg);
        for (MachineInstr *RD : Entry.Src2Bridges)
          Plan.DefBridges.insert({RD, Src2Reg});
      }
    }

    SmallVector<MachineOperand *, 8> DstUses;
    findReachingUses(MI, DAG.LIS, DstUses);

    // The reaching uses either read the AGPR form directly or get a bridge
    // copy, so they do not constrain the recolor.
    Entry.DstRecolorSafe = !RedefRegs.contains(DstReg) &&
                           MFMAToAGPR->canRecolorToAGPR(DstReg, DstUses);
    const TargetRegisterClass *DstAGPRClass =
        SRI->getEquivalentAGPRClass(DAG.MRI.getRegClass(DstReg));

    // A tied use shares its register with the def it is tied to, so pointing it
    // at a bridge copy would rewrite one half of the tie and leave the other.
    // A group-member MFMA reads the AGPR form directly and is never redirected,
    // so its tied src2 is not a problem.
    bool HasTiedUse = any_of(DstUses, [&](const MachineOperand *MO) {
      return MO->isTied() && !Plan.MFMASet.contains(MO->getParent());
    });

    // Nothing to redirect, or nothing we are allowed to redirect: copy the
    // AGPR-form result back into the original dst instead and leave every use
    // reading it. apply() emits one copy, so charge for one.
    Entry.DstNeedsRestore =
        !Entry.DstRecolorSafe && (DstUses.empty() || HasTiedUse);
    if (Entry.DstNeedsRestore) {
      Plan.UseBridges.insert({MI->getParent(), DstReg});
      Plan.Entries.push_back(std::move(Entry));
      RedefRegs.insert(DstReg);
      continue;
    }

    for (MachineOperand *RUOp : DstUses) {
      MachineInstr *UserMI = RUOp->getParent();

      // Decide whether this reaching use can read the dst's AGPR form directly
      // or needs an AGPR->VGPR bridge copy. It has to accept an AGPR either
      // way. Beyond that, a group member reads the AGPR result wherever
      // apply() leaves it, while any other user reads the original register,
      // which stays VGPR unless the dst is recolored.
      bool CanReadAGPR =
          (Plan.MFMASet.contains(UserMI) || Entry.DstRecolorSafe) &&
          MFMAToAGPR->operandAcceptsClass(*RUOp, DstAGPRClass);
      if (!CanReadAGPR && !is_contained(Entry.DstUseBridges, RUOp)) {
        Entry.DstUseBridges.push_back(RUOp);
        // One copy per block serves every use of this register in it.
        Plan.UseBridges.insert({UserMI->getParent(), DstReg});
      }

      // If the dst is wholly recolored to AGPR, its reaching defs are
      // reclassified along with it, so none of them need a bridge copy.
      if (Entry.DstRecolorSafe)
        continue;

      SmallVector<MachineInstr *, 8> UseReachingDefs;
      findReachingDefs(*RUOp, DAG.LIS, UseReachingDefs);

      for (MachineInstr *RD : UseReachingDefs) {
        if (Plan.MFMASet.contains(RD))
          continue;

        // Bridge every reaching def of this use into one register, so the
        // AGPR->VGPR copy can sit at the user rather than after the MFMA.
        if (!is_contained(Entry.DstUseDefBridges, RD))
          Entry.DstUseDefBridges.push_back(RD);
      }
    }

    for (MachineInstr *RD : Entry.DstUseDefBridges)
      Plan.DefBridges.insert({RD, DstReg});

    // apply() gives an unsafe dst a fresh register whether or not it ends up
    // needing a bridge copy, so that reclassification recolors that register
    // instead of illegally recoloring the original VGPR dst.
    if (!Entry.DstRecolorSafe)
      RedefRegs.insert(DstReg);

    Plan.Entries.push_back(std::move(Entry));
  }

  return true;
}

SmallVector<GCNRegPressure, 8>
RewriteMFMAFormStage::measure(const MFMARewritePlan &Plan) {
  // Put the candidates in AGPR form just long enough to measure the pressure
  // they produce. The scope restores the VGPR form on every exit path: the real
  // rewrite has to happen after copy insertion, since some defs of a register
  // may still require a VGPR.
  AMDGPU::MFMAToAGPR::FormScope Scope(*MFMAToAGPR, Plan.MFMAs,
                                      Plan.RecolorRegs.getArrayRef());

  SmallVector<GCNRegPressure, 8> Pressures;
  for (unsigned Region = 0; Region < DAG.Regions.size(); Region++)
    if (RegionsWithExcessArchVGPR[Region])
      Pressures.push_back(DAG.getRealRegPressure(Region));
  return Pressures;
}

RewriteMFMAFormStage::MFMARewriteCost
RewriteMFMAFormStage::cost(const MFMARewritePlan &Plan,
                           ArrayRef<GCNRegPressure> PressuresAfter) {
  MachineBlockFrequencyInfo *MBFI = DAG.MBFI;

  int64_t BestSpillCost = 0;
  int64_t Cost = 0;
  unsigned MaxAGPRAfter = 0;
  uint64_t EntryFreq = MBFI->getEntryFreq().getFrequency();

  std::pair<unsigned, unsigned> MaxVectorRegs =
      ST.getMaxNumVectorRegs(MF.getFunction());
  unsigned ArchVGPRThreshold = MaxVectorRegs.first;
  unsigned AGPRThreshold = MaxVectorRegs.second;
  unsigned CombinedThreshold = ST.getMaxNumVGPRs(MF);

  unsigned MeasuredIdx = 0;
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
    GCNRegPressure PressureAfter = PressuresAfter[MeasuredIdx++];
    unsigned SpillCostAfter = PressureAfter.getVGPRSpills(
        MF, ArchVGPRThreshold, AGPRThreshold, CombinedThreshold);
    MaxAGPRAfter = std::max(MaxAGPRAfter, PressureAfter.getAGPRNum());

    LLVM_DEBUG(dbgs() << "  region " << Region << ": ArchVGPR "
                      << PressureBefore.getArchVGPRNum() << " -> "
                      << PressureAfter.getArchVGPRNum() << ", AGPR "
                      << PressureBefore.getAGPRNum() << " -> "
                      << PressureAfter.getAGPRNum() << ", spills "
                      << SpillCostBefore << " -> " << SpillCostAfter << '\n');

    uint64_t BlockFreq =
        MBFI->getBlockFreq(DAG.Regions[Region].first->getParent())
            .getFrequency();

    bool RelativeFreqIsDenom = EntryFreq > BlockFreq;
    uint64_t RelativeFreq = EntryFreq && BlockFreq
                                ? (RelativeFreqIsDenom ? EntryFreq / BlockFreq
                                                       : BlockFreq / EntryFreq)
                                : 1;

    // This assumes perfect spilling / splitting one store and one reload for
    // each excess register. Both go to scratch, so a spilled register is worth
    // far more than the register-to-register copies charged below, which are a
    // single VALU op each.
    int64_t SpillCost =
        ((int)SpillCostAfter - (int)SpillCostBefore) * SpillCostPerReg;

    // Also account for the block frequency.
    if (RelativeFreqIsDenom)
      SpillCost /= (int64_t)RelativeFreq;
    else
      SpillCost *= (int64_t)RelativeFreq;

    // If we have increased spilling in any block, just bail.
    if (SpillCost > 0)
      return {SpillCost, /*AGPRUtil=*/0};

    if (SpillCost < BestSpillCost)
      BestSpillCost = SpillCost;
  }

  // Set the cost to the largest decrease in spill cost in order to not double
  // count spill reductions.
  Cost = BestSpillCost;
  assert(Cost <= 0);

  // Charge for every bridge copy apply() is going to emit, weighted by the
  // frequency of the block it lands in. A bridge crosses the register banks, so
  // it is priced as an AGPR copy even though the register is still VGPR here.
  unsigned CopyCost = 0;
  auto ChargeCopy = [&](const MachineBasicBlock *MBB, Register Reg) {
    uint64_t BlockFreq = MBFI->getBlockFreq(MBB).getFrequency();

    // Scale by the block frequency the same way the spill cost above does, so
    // that a copy in a block colder than entry is discounted rather than
    // truncated to nothing by integer division.
    bool RelativeFreqIsDenom = EntryFreq > BlockFreq;
    uint64_t RelativeFreq = EntryFreq && BlockFreq
                                ? (RelativeFreqIsDenom ? EntryFreq / BlockFreq
                                                       : BlockFreq / EntryFreq)
                                : 1;

    const TargetRegisterClass *RC = DAG.MRI.getRegClass(Reg);
    if (const TargetRegisterClass *AGPRRC = SRI->getEquivalentAGPRClass(RC))
      RC = AGPRRC;

    int64_t Copy = RC->getCopyCost();
    if (RelativeFreqIsDenom)
      Copy /= (int64_t)RelativeFreq;
    else
      Copy *= (int64_t)RelativeFreq;
    CopyCost += Copy;
  };

  for (auto [DefMI, Reg] : Plan.DefBridges)
    ChargeCopy(DefMI->getParent(), Reg);

  for (auto [UseBlock, Reg] : Plan.UseBridges)
    ChargeCopy(UseBlock, Reg);

  // Report the AGPR utilization only while there is headroom left. Past this
  // soft limit, using more AGPRs is a risk rather than a bonus: they spill too.
  unsigned SoftAGPRLimit = AGPRThreshold * 7 / 8;
  unsigned AGPRUtil = MaxAGPRAfter <= SoftAGPRLimit ? MaxAGPRAfter : 0;

  LLVM_DEBUG(dbgs() << "  spill cost " << Cost << ", copy cost " << CopyCost
                    << ", AGPR " << MaxAGPRAfter << '/' << AGPRThreshold
                    << '\n');

  return {Cost + CopyCost, AGPRUtil};
}

void RewriteMFMAFormStage::apply(const MFMARewritePlan &Plan) {
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
  // and placement as we must have an accurate redef map otherwise we may end
  // up creating illegal instructions.

  // The original registers of the MFMA that need to be reclassified as AGPR.
  DenseSet<Register> RewriteRegs;
  // The map of an original register in the MFMA to a new register (result of a
  // copy) that it should be replaced with.
  DenseMap<Register, Register> RedefMap;
  // The map of the original MFMA registers to the relevant MFMA operands.
  DenseMap<Register, DenseSet<MachineOperand *>> ReplaceMap;
  // The map of reaching defs for a given register to avoid duplicate copies.
  DenseMap<Register, SmallPtrSet<MachineInstr *, 8>> ReachingDefCopyMap;
  // The map of reaching uses for a given register by basic block to avoid
  // duplicate copies and to calculate per MBB insert pts.
  DenseMap<unsigned, DenseMap<Register, SmallPtrSet<MachineOperand *, 8>>>
      ReachingUseTracker;

  // \returns the register carrying the VGPR-form value of \p Reg, creating it
  // on first request.
  auto getMappedReg = [&](Register Reg) {
    auto [It, Inserted] = RedefMap.try_emplace(Reg);
    if (Inserted)
      It->second = DAG.MRI.createVirtualRegister(
          SRI->getEquivalentVGPRClass(DAG.MRI.getRegClass(Reg)));
    return It->second;
  };

  // Insert "MappedReg = COPY Reg" after the reaching def \p RD. An earlier
  // iteration may have inserted this copy already.
  auto bridgeAfterDef = [&](MachineInstr *RD, Register Reg) {
    if (!ReachingDefCopyMap[Reg].insert(RD).second)
      return;

    MachineInstrBuilder VGPRCopy =
        BuildMI(*RD->getParent(), std::next(RD->getIterator()),
                RD->getDebugLoc(), TII->get(TargetOpcode::COPY))
            .addDef(getMappedReg(Reg), {}, 0)
            .addUse(Reg, {}, 0);
    DAG.LIS->InsertMachineInstrInMaps(*VGPRCopy);

    // If this reaching def was the last MI in the region, update the region
    // boundaries.
    auto LMI = LastMIToRegion.find(RD);
    if (LMI != LastMIToRegion.end()) {
      DAG.Regions[LMI->second].second = VGPRCopy;
      LastMIToRegion.erase(LMI);
    }
  };

  for (const MFMARewriteEntry &Entry : Plan.Entries) {
    MachineInstr *MI = Entry.MFMA;
    int ReplacementOp = AMDGPU::getAGPRFormOp(MI->getOpcode());
    assert(ReplacementOp != -1);
    MI->setDesc(TII->get(ReplacementOp));

    // Case 1: insert copies for the reaching defs of the Src2Reg.
    MachineOperand *Src2 = TII->getNamedOperand(*MI, AMDGPU::OpName::src2);
    if (Src2->isReg()) {
      Register Src2Reg = Src2->getReg();
      for (MachineInstr *RD : Entry.Src2Bridges)
        bridgeAfterDef(RD, Src2Reg);

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

    // The dst cannot be recolored and its uses cannot be redirected: create a
    // fresh reg to carry the AGPR-form value and record the mapping, leaving
    // the original dst reg in VGPR form.
    if (Entry.DstNeedsRestore && !RedefMap.contains(DstReg)) {
      // The MFMA's def operand is redirected to the mapped reg (which is
      // reclassified to AGPR form), so the original dst reg loses its full-lane
      // def. Bridge the AGPR-form result back into the original VGPR reg right
      // after the MFMA so any later partial redef and downstream use still see
      // a dominating full-lane def.
      MachineInstrBuilder Bridge =
          BuildMI(*MI->getParent(), std::next(MI->getIterator()),
                  MI->getDebugLoc(), TII->get(TargetOpcode::COPY))
              .addDef(DstReg, {}, 0)
              .addUse(getMappedReg(DstReg), {}, 0);
      DAG.LIS->InsertMachineInstrInMaps(*Bridge);
    }

    // An unsafe dst must be mapped even when it has no def to bridge, so that
    // reclassification recolors the mapped reg instead of illegally recoloring
    // the original VGPR dst to AGPR.
    if (!Entry.DstRecolorSafe)
      getMappedReg(DstReg);

    for (MachineInstr *RD : Entry.DstUseDefBridges)
      bridgeAfterDef(RD, DstReg);

    DenseSet<MachineOperand *> &DstRegSet = ReplaceMap[DstReg];
    // One AGPR→VGPR copy per dst register, shared by all same-block uses.
    Register SameBlockCopyReg;
    MachineInstr *EarliestSameBlockUse = nullptr;
    for (MachineOperand *RU : Entry.DstUseBridges) {
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

  // Redirecting the uses of a register to a bridge copy can leave its defs
  // without a common reaching use, which the verifier rejects. The mapped
  // register can end up equally disconnected: it carries the AGPR-form result
  // of a dst that needs restoring and, further on, the AGPR-form value a src2
  // bridge copies back in. Give every disconnected piece its own register.
  for (auto [OldReg, NewReg] : RedefMap) {
    SmallVector<LiveInterval *, 4> SplitLIs;
    if (DAG.LIS->hasInterval(OldReg))
      DAG.LIS->splitSeparateComponents(DAG.LIS->getInterval(OldReg), SplitLIs);
    if (DAG.LIS->hasInterval(NewReg))
      DAG.LIS->splitSeparateComponents(DAG.LIS->getInterval(NewReg), SplitLIs);
  }

  // Liveins may have been modified for cross RC copies
  RegionPressureMap LiveInUpdater(&DAG, false);
  LiveInUpdater.buildLiveRegMap();

  for (unsigned Region = 0; Region < DAG.Regions.size(); Region++)
    DAG.LiveIns[Region] = LiveInUpdater.getLiveRegsForRegionIdx(Region);

  DAG.Pressure[RegionIdx] = DAG.getRealRegPressure(RegionIdx);
}
