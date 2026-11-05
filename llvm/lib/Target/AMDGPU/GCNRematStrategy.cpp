//===-- GCNRematStrategy.cpp - Rematerialization strategy -*- C++ -*-------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Implements the rematerialization strategy.
//
//===----------------------------------------------------------------------===//

#include "GCNRematStrategy.h"
#include "GCNRegPressure.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineBlockFrequencyInfo.h"
#include "llvm/CodeGen/RegisterPressure.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/MC/LaneBitmask.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "gcn-rematerialization-strategy"

using namespace llvm;
using RegSubRegPair = TargetInstrInfo::RegSubRegPair;
using RegisterIdx = Rematerializer::RegisterIdx;

GCNRematStrategy::RematRegions::RematRegions(
    ArrayRef<RegionBoundaries> Regions,
    MutableArrayRef<GCNRPTracker::LiveRegSet> LiveIns,
    MutableArrayRef<GCNRPTracker::LiveRegSet> LiveOuts, LiveIntervals &LIS,
    const MachineBlockFrequencyInfo &MBFI, const MachineRegisterInfo &MRI)
    : Bounds(Regions), LiveIns(LiveIns), LiveOuts(LiveOuts),
      Frequencies(Regions.size(), 0), LIS(LIS), MRI(MRI) {
  assert(LiveIns.size() == Regions.size());
  assert(LiveOuts.size() == Regions.size());

  // Derive frequency of each region. Frequencies are scaled down if they are
  // high to avoid overflow/underflow when combining them.
  MinFreq = MBFI.getEntryFreq().getFrequency();
  MaxFreq = 0;
  for (unsigned RegionIdx : seq(Regions.size())) {
    MachineBasicBlock *MBB = Regions[RegionIdx].first->getParent();
    uint64_t BlockFreq = MBFI.getBlockFreq(MBB).getFrequency();
    Frequencies[RegionIdx] = BlockFreq;
    if (BlockFreq && BlockFreq < MinFreq)
      MinFreq = BlockFreq;
    else if (BlockFreq > MaxFreq)
      MaxFreq = BlockFreq;
  }
  if (!MinFreq)
    return;

  // Scale everything down if frequencies are high.
  if (MinFreq >= FreqScaleFactor * FreqScaleFactor) {
    for (uint64_t &Freq : Frequencies)
      Freq /= FreqScaleFactor;
    MinFreq /= FreqScaleFactor;
    MaxFreq /= FreqScaleFactor;
  }
}

GCNRegPressure GCNRematStrategy::RematRegions::computeMaxRegPressure(
    unsigned RegionIdx) const {
  assert(RegionIdx < Bounds.size() && "out of bounds");
  const auto &[Begin, End] = Bounds[RegionIdx];
  if (Begin == End)
    return llvm::getRegPressure(MRI, LiveIns[RegionIdx]);
  GCNDownwardRPTracker RPTracker(LIS);
  RPTracker.advance(Begin, End, &LiveIns[RegionIdx]);
  return RPTracker.moveMaxPressure();
}

void GCNRematStrategy::RematRegions::removeFromLiveMaps(
    Register Reg, const BitVector &LiveIn, const BitVector &LiveOut) {
  for (unsigned I : LiveIn.set_bits())
    LiveIns[I].erase(Reg);
  for (unsigned I : LiveOut.set_bits())
    LiveOuts[I].erase(Reg);
}

void GCNRematStrategy::RematRegions::addToLiveMaps(Register Reg,
                                                   LaneBitmask Mask,
                                                   const BitVector &LiveIn,
                                                   const BitVector &LiveOut) {
  std::pair<Register, LaneBitmask> LiveReg(Reg, Mask);
  for (unsigned I : LiveIn.set_bits())
    LiveIns[I].insert(LiveReg);
  for (unsigned I : LiveOut.set_bits())
    LiveOuts[I].insert(LiveReg);
}

GCNRematStrategy::RematTargets::RematTargets(
    unsigned MaxSGPRs, unsigned MaxVGPRs, bool SpillReduction,
    const MachineFunction &MF, const RematRegions &Regions,
    ArrayRef<GCNRegPressure> *MaxPressure)
    : ExcessRP(Regions.size()), Regions(Regions),
      SpillReduction(SpillReduction) {
  RPTargets.reserve(Regions.size());
  for (unsigned RegionIdx : seq(Regions.size())) {
    if (MaxPressure) {
      RPTargets.emplace_back(MaxSGPRs, MaxVGPRs, MF, (*MaxPressure)[RegionIdx]);
    } else {
      RPTargets.emplace_back(MaxSGPRs, MaxVGPRs, MF,
                             Regions.computeMaxRegPressure(RegionIdx));
    }
    if (!RPTargets.back().satisfied())
      ExcessRP.set(RegionIdx);
  }
}

void GCNRematStrategy::RematTargets::updateRPTargets(
    const BitVector &UpdateRegions, const GCNRegPressure &RPSave) {
  for (unsigned I : UpdateRegions.set_bits()) {
    RPTargets[I].saveRP(RPSave);
    if (ExcessRP[I] && RPTargets[I].satisfied()) {
      LLVM_DEBUG(dbgs() << "  [" << I << "] Target reached!\n");
      ExcessRP.reset(I);
    }
  }
}

bool GCNRematStrategy::RematTargets::updateAndVerifyRPTargets(
    const BitVector &UpdateRegions) {
  bool TooOptimistic = false;
  for (unsigned RegionIdx : UpdateRegions.set_bits()) {
    GCNRPTarget &Target = RPTargets[RegionIdx];
    Target.setRP(Regions.computeMaxRegPressure(RegionIdx));

    // Since we were optimistic in assessing RP decreases in these regions, we
    // may need to remark the target as a target region if RP didn't decrease
    // as expected.
    if (!ExcessRP.test(RegionIdx) && !Target.satisfied()) {
      LLVM_DEBUG(dbgs() << "  [" << RegionIdx << "] Incorrect RP estimation\n");
      TooOptimistic = true;
      ExcessRP.set(RegionIdx);
    }
  }
  return TooOptimistic;
}

void GCNRematStrategy::RematCandidate::init(const RematRegions &RF,
                                            const Rematerializer &Remater,
                                            const MachineRegisterInfo &MRI) {
  const Rematerializer::Reg &Reg = Remater.getReg(RegIdx);
  Register DefReg = Reg.getDefReg();

  Live |= LiveIn;
  Live |= LiveOut;

  for (unsigned I : Live.set_bits()) {
    // If the register is both unused and live-through in the region, the
    // latter's RP is guaranteed to decrease.
    if (!LiveIn[I] || !LiveOut[I] || Reg.Uses.contains(I))
      UnpredictableRPSave.set(I);
  }
  RPSave.inc(DefReg, LaneBitmask::getNone(), Reg.Mask, MRI);

  // Estimate the difference in instruction execution frequency that will be
  // induced by the rematerialization. A rematerialization from the least
  // frequent region to the most frequent region should yield the greatest
  // frequency penalty (most negative frequency difference). When the register
  // has users in its defining region the original instruction(s) won't be
  // deleted so there is no frequency gain in the defining region.
  int64_t DefOrMin =
      Reg.hasUsersInDefRegion() ? 0 : RF.getFrequencyOrMin(Reg.DefRegion);
  int64_t UseOrMax = 0;
  for (const auto &[UseRegion, _] : Reg.Uses) {
    if (Reg.DefRegion == UseRegion)
      continue;
    UseOrMax += RF.getFrequencyOrMax(UseRegion);
  }
  FreqDiff = DefOrMin - UseOrMax;
}

void GCNRematStrategy::RematCandidate::update(const RematTargets &Targets) {
  MaxFreq = RegionImpact = 0;
  for (unsigned RegionIdx : Targets.ExcessRP.set_bits()) {
    if (!Live[RegionIdx])
      continue;

    // The rematerialization must contribute positively in at least one
    // register class with usage above the RP target for this region to
    // contribute to the score.
    const GCNRPTarget &RegionTarget = Targets.RPTargets[RegionIdx];
    const unsigned NumRegsBenefit = RegionTarget.getNumRegsBenefit(RPSave);
    if (!NumRegsBenefit)
      continue;

    // Regions in which RP is guaranteed to decrease have more weight.
    RegionImpact += (UnpredictableRPSave[RegionIdx] ? 1 : 2) * NumRegsBenefit;

    if (Targets.SpillReduction) {
      uint64_t Freq = Targets.Regions.getFrequency(RegionIdx);
      if (UnpredictableRPSave[RegionIdx]) {
        // Apply a frequency penalty in regions in which we are not sure that RP
        // will decrease.
        Freq /= 2;
      }
      MaxFreq = std::max(MaxFreq, Freq);
    }
  }
}

bool GCNRematStrategy::RematCandidate::maybeBeneficial(
    const RematTargets &Targets) const {
  for (unsigned I : Targets.ExcessRP.set_bits()) {
    if (Live[I] && Targets.RPTargets[I].isSaveBeneficial(RPSave))
      return true;
  }
  return false;
}

void GCNRematStrategy::RematCandidate::rematerialize(
    Rematerializer &Remater) const {
  const Rematerializer::Reg &Reg = Remater.getReg(RegIdx);

  // All rematerializable dependencies are re-used for all rematerializations.
  Rematerializer::DependencyReuseInfo DRI;
  for (RegisterIdx DepRegIdx : Reg.Dependencies)
    DRI.reuse(DepRegIdx);

  // Rematerialize once per using region, except in the defining region.
  SmallVector<unsigned, 8> RematRegions;
  RematRegions.reserve(Reg.Uses.size());
  for (const auto &[UseRegion, _] : Reg.Uses) {
    if (Reg.DefRegion != UseRegion)
      RematRegions.push_back(UseRegion);
  }
  for (const unsigned UseRegion : RematRegions) {
    Rematerializer::DependencyReuseInfo RegionDRI(DRI);
    Remater.rematerializeToRegion(RegIdx, UseRegion, RegionDRI);
  }
}

void GCNRematStrategy::rematerializeFor(
    unsigned MaxSGPRs, unsigned MaxVGPRs, bool SpillReduction,
    BitVector &AffectedRegions, MutableArrayRef<GCNRegPressure> *MaxPressure) {
  assert(AffectedRegions.size() == Regions.size());
  assert(!MaxPressure || MaxPressure->size() == Regions.size());

  RematTargets Targets(MaxSGPRs, MaxVGPRs, SpillReduction, MF, Regions,
                       MaxPressure);
  if (Targets.allAchieved())
    return;
  if (!CandidatesCreated) {
    Remater.analyze();
    createCandidates();
  }

  LLVM_DEBUG({
    dbgs() << "Rematerializing across regions to reduce RP below target in "
              "function ";
    MF.getFunction().printAsOperand(dbgs(), /*PrintType=*/false);
    dbgs() << '\n'
           << Candidates.size()
           << " rematerialization candidates were identified in the function\n"
           << Targets.print() << '\n';
    MF.print(dbgs(), LIS.getSlotIndexes());
  });

  // Keep track of all useful candidates.
  SmallVector<unsigned, 8> CandidateOrder;
  for (auto [CandIdx, Cand] : enumerate(Candidates)) {
    Cand.update(Targets);
    if (!Cand.hasNullScore())
      CandidateOrder.push_back(CandIdx);
  }

  // Rematerialize registers in successive rounds until all RP targets are
  // satisifed or until we run out of rematerialization candidates.
  BitVector RecomputeRP(Regions.size());
  for (;;) {
    RecomputeRP.reset();

    // Sort candidates in increasing score order.
    sort(CandidateOrder, [&](unsigned LHSIndex, unsigned RHSIndex) {
      return Candidates[LHSIndex] < Candidates[RHSIndex];
    });

    LLVM_DEBUG({
      dbgs() << "==== NEW REMAT ROUND ====\n"
             << "Candidates with non-null score, in rematerialization order:\n";
      for (const RematCandidate &Cand : reverse(Candidates)) {
        dbgs() << "  " << Cand.print() << " | "
               << Remater.printRematReg(Cand.RegIdx) << '\n';
      }
      dbgs() << Targets.print();
    });

    // Rematerialize registers in decreasing score order until we estimate
    // that all RP targets are satisfied or until rematerialization candidates
    // are no longer useful to decrease RP.
    while (!CandidateOrder.empty()) {
      const RematCandidate &Cand = Candidates[CandidateOrder.back()];
      const Rematerializer::Reg &Reg = Remater.getReg(Cand.RegIdx);

      // When previous rematerializations in this round have already satisfied
      // RP targets in all regions this rematerialization can impact, we have a
      // good indication that our scores have diverged significantly from
      // reality, in which case we interrupt this round and re-score. This also
      // ensures that every rematerialization we perform is possibly impactful
      // in at least one target region.
      if (!Cand.maybeBeneficial(Targets)) {
        LLVM_DEBUG(dbgs() << "Interrupt round on stale score for "
                          << Cand.print() << " | "
                          << Remater.printRematReg(Cand.RegIdx));
        break;
      }
      CandidateOrder.pop_back();

#ifdef EXPENSIVE_CHECKS
      // All uses outside the defining region are known to be available / live
      // at the remat points. Thus, the uses should already be live in to
      // those using regions.
      const TargetRegisterInfo &TRI = *MRI.getTargetRegisterInfo();
      for (const MachineInstr *DefMI : Reg.Defs) {
        for (const MachineOperand &MO : DefMI->operands()) {
          // Exclude the defined register. We are rematerializing all
          // instructions defining it so we don't care that its value is
          // available at the remat point.
          if (!MO.isReg() || !MO.getReg() || !MO.readsReg() || MO.isDef())
            continue;

          Register UseReg = MO.getReg();
          if (!UseReg.isVirtual())
            continue;

          LiveInterval &LI = LIS.getInterval(UseReg);
          LaneBitmask LM = MRI.getMaxLaneMaskForVReg(MO.getReg());
          if (LI.hasSubRanges() && MO.getSubReg())
            LM = TRI.getSubRegIndexLaneMask(MO.getSubReg());

          for (const auto &[UseRegion, _] : Reg.Uses) {
            if (UseRegion == Reg.DefRegion)
              continue;
            LaneBitmask LiveInMask = getLiveIns(UseRegion).at(UseReg);
            LaneBitmask UncoveredLanes = LM & ~(LiveInMask & LM);
            // If this register has lanes not covered by the LiveIns, be sure
            // they do not map to any subrange. ref:
            // machine-scheduler-sink-trivial-remats.mir::omitted_subrange
            if (UncoveredLanes.any()) {
              assert(LI.hasSubRanges());
              for (LiveInterval::SubRange &SR : LI.subranges())
                assert((SR.LaneMask & UncoveredLanes).none());
            }
          }
        }
      }
#endif

      // Remove the register from all regions where it is a live-in or live-out,
      // then rematerialize the register.
      LLVM_DEBUG(dbgs() << "** REMAT " << Remater.printRematReg(Cand.RegIdx)
                        << '\n');
      Regions.removeFromLiveMaps(Reg.getDefReg(), Cand.LiveIn, Cand.LiveOut);
      if (Recorder) {
        Recorder->LiveMapUpdates.emplace_back(Cand.RegIdx, Cand.LiveIn,
                                              Cand.LiveOut);
      }
      Cand.rematerialize(Remater);

      // Adjust RP targets. The save is guaranteed in regions in which the
      // register is live-through and unused but optimistic in all other regions
      // where the register is live.
      Targets.updateRPTargets(Cand.Live, Cand.RPSave);
      RecomputeRP |= Cand.UnpredictableRPSave;
      AffectedRegions |= Cand.Live;
      if (Targets.allAchieved()) {
        LLVM_DEBUG(dbgs() << "All targets cleared, verifying...\n");
        break;
      }
    }

    if (!Targets.updateAndVerifyRPTargets(RecomputeRP) &&
        Targets.allAchieved()) {
      LLVM_DEBUG(dbgs() << "Objectives achieved!\n");
      break;
    }

    // Update the score of remaining candidates and filter out those that have
    // become useless from the vector. Candidates never become useful after
    // having been useless for a round, so we can freely drop them without
    // losing any future rematerialization opportunity.
    unsigned NumUsefulCandidates = 0;
    for (unsigned CandIdx : CandidateOrder) {
      RematCandidate &Candidate = Candidates[CandIdx];
      Candidate.update(Targets);
      if (!Candidate.hasNullScore())
        CandidateOrder[NumUsefulCandidates++] = CandIdx;
    }
    if (NumUsefulCandidates == 0) {
      LLVM_DEBUG(dbgs() << "Stop on exhausted rematerialization candidates\n");
      break;
    }
    CandidateOrder.truncate(NumUsefulCandidates);
  }

  if (!MaxPressure)
    return;
  for (unsigned RegionIdx : AffectedRegions.set_bits())
    (*MaxPressure)[RegionIdx] = Targets.RPTargets[RegionIdx].getCurrentRP();
}

void GCNRematStrategy::createCandidates() {
  CandidatesCreated = true;

  // Set of registers already marked for potential remterialization; used to
  // avoid rematerialization chains.
  SmallSet<Register, 4> MarkedRegs;

  // Map registers to candidate indices. Use ~0u as null value since 0 is a
  // valid index.
  IndexedMap<unsigned, VirtReg2IndexFunctor> DefRegToCandIdx(~0u);
  DefRegToCandIdx.resize(MRI.getNumVirtRegs());
  const unsigned NumRegions = Regions.size();

  for (unsigned RegIdx = 0, E = Remater.getNumRegs(); RegIdx < E; ++RegIdx) {
    const Rematerializer::Reg &CandReg = Remater.getReg(RegIdx);
    if (!CandReg.hasUsersOutsideDefRegion())
      continue;

    // Rematerialization moves the defining instruction into the region of its
    // uses, which may sit under different control dependencies (e.g., across a
    // change of EXEC). Convergent operations must not be made control-dependent
    // on additional values, so they cannot be safely relocated this way. This
    // mirrors the check MachineSink performs before sinking an instruction.
    if (any_of(CandReg.Defs,
               [](const MachineInstr *DefMI) { return DefMI->isConvergent(); }))
      continue;

    // Rematerializing a load into multiple using regions duplicates the
    // memory access itself. Conservatively restrict rematerialization of loads
    // to a single region.
    if (CandReg.Uses.size() > 1 &&
        any_of(CandReg.Defs,
               [](const MachineInstr *DefMI) { return DefMI->mayLoad(); }))
      continue;

    // We further filter the registers that we can rematerialize based on our
    // current tracking capabilities in the strategy. This ensures that we are
    // not extending any live range while rematerializing, and that
    // rematerializing any register candidate will not affect other candidates.

    // All register dependencies should not already be considered
    // rematerializable.
    if (llvm::any_of(CandReg.Dependencies, [&](RegisterIdx DepRegIdx) {
          return MarkedRegs.contains(Remater.getReg(DepRegIdx).getDefReg());
        }))
      continue;

    if (!candidateHasValidUsers(RegIdx, MarkedRegs))
      continue;

    Register DefReg = CandReg.getDefReg();
    MarkedRegs.insert(DefReg);
    DefRegToCandIdx[DefReg] = static_cast<unsigned>(Candidates.size());
    Candidates.emplace_back(RegIdx, NumRegions);
  }

  // Initialize the LiveIn and LiveOut sets of all candidates. This is more
  // efficient than querying those structures for each candidate separately.
  for (unsigned RegionIdx = 0; RegionIdx < NumRegions; ++RegionIdx) {
    for (const auto &[Reg, Mask] : Regions.getLiveIns(RegionIdx)) {
      if (Register::isPhysicalRegister(Reg))
        continue;
      unsigned CandIdx = DefRegToCandIdx[Reg];
      if (CandIdx != ~0u)
        Candidates[CandIdx].LiveIn.set(RegionIdx);
    }
    for (const auto &[Reg, Mask] : Regions.getLiveOuts(RegionIdx)) {
      if (Register::isPhysicalRegister(Reg))
        continue;
      unsigned CandIdx = DefRegToCandIdx[Reg];
      if (CandIdx != ~0u)
        Candidates[CandIdx].LiveOut.set(RegionIdx);
    }
  }

  for (RematCandidate &Cand : Candidates)
    Cand.init(Regions, Remater, MRI);
}

bool GCNRematStrategy::candidateHasValidUsers(
    RegisterIdx CandIdx, const SmallSet<Register, 4> &MarkedRegs) const {
  const Rematerializer::Reg &CandReg = Remater.getReg(CandIdx);
  SlotIndex RefIdx =
      LIS.getInstructionIndex(*CandReg.getLastDef()).getRegSlot(true);

  for (const auto &[UseRegion, Users] : CandReg.Uses) {
    // Users cannot themselves be rematerializable.
    if (llvm::any_of(Users, [&](const MachineInstr *UserMI) {
          assert(UserMI->getNumOperands() > 0 &&
                 "user must have at least one operand");
          // A user is rematerializable if its first register operand (its
          // register def) was marked already.
          const MachineOperand &DefMO = UserMI->getOperand(0);
          return DefMO.isReg() && MarkedRegs.contains(DefMO.getReg());
        }))
      return false;

    MachineInstr *FirstUseMI = CandReg.getRegionUseBounds(UseRegion, LIS).first;
    assert(FirstUseMI && "there must be a user in the region");
    SlotIndex FirstUseIdx =
        LIS.getInstructionIndex(*FirstUseMI).getRegSlot(true);

    // All dependencies must be available at the first use in the region.
    if (llvm::any_of(CandReg.Dependencies, [&](RegisterIdx DepRegIdx) {
          const Rematerializer::Reg &DepReg = Remater.getReg(DepRegIdx);
          Register DepDefReg = DepReg.getDefReg();
          return !Remater.isRegIdenticalAtUses(DepDefReg, DepReg.Mask, RefIdx,
                                               {FirstUseIdx});
        }))
      return false;
    if (llvm::any_of(Remater.getUnrematableDeps(CandIdx),
                     [&](const std::pair<Register, LaneBitmask> &RegAndMask) {
                       const auto &[Reg, Mask] = RegAndMask;
                       return !Remater.isRegIdenticalAtUses(Reg, Mask, RefIdx,
                                                            {FirstUseIdx});
                     }))
      return false;
  }
  return true;
}

void GCNRematStrategy::rollbackRecordedRemats() {
  // Roll back changes made by the strategy.
  assert(Recorder && "recorder should be defined");
  Recorder->Listener.rollback(Remater);
  for (const auto &[RegIdx, LiveIn, LiveOut] : Recorder->LiveMapUpdates) {
    const Rematerializer::Reg &Reg = Remater.getReg(RegIdx);
    Regions.addToLiveMaps(Reg.getDefReg(), Reg.Mask, LiveIn, LiveOut);
  }
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
Printable GCNRematStrategy::RematCandidate::print() const {
  return Printable([&](raw_ostream &OS) {
    OS << '(' << MaxFreq << ", " << FreqDiff << ", " << RegionImpact << ')';
  });
}

Printable GCNRematStrategy::RematTargets::print() const {
  return Printable([&](raw_ostream &OS) {
    if (allAchieved()) {
      OS << "No target regions\n";
      return;
    }
    OS << "Target regions:\n";
    for (unsigned I : ExcessRP.set_bits())
      OS << "  [" << I << "] " << RPTargets[I] << '\n';
  });
}
#endif
