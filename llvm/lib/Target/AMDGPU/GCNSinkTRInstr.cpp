//===-- GCNSinkTRInstr.cpp ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
///
//===----------------------------------------------------------------------===//

#include "GCNSinkTRInstr.h"
#include "GCNSubtarget.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"

#define DEBUG_TYPE "sink-trivials"

using namespace llvm;

SinkTrivallyRematInstr::SinkTrivallyRematInstr(
    MachineFunction &MF_, LiveIntervals *LIS_,
    const DenseMap<MachineBasicBlock *, GCNRegPressure> &MBBPressure_)
    : MF(MF_), ST(MF.getSubtarget<GCNSubtarget>()), MRI(MF.getRegInfo()),
      TII(*MF.getSubtarget().getInstrInfo()),
      TRI(*MF.getSubtarget().getRegisterInfo()), LIS(*LIS_),
      MBBPressure(MBBPressure_) {}

void SinkTrivallyRematInstr::init() const {
  if (!MBBFirstInstSlot.empty())
    return;
  auto &SII = *LIS.getSlotIndexes();
  for (auto &MBB : MF) {
    auto FirstMII = skipDebugInstructionsForward(MBB.begin(), MBB.end());
    auto SI = SII.getInstructionIndex(*FirstMII);
    MBBFirstInstSlot.push_back(SI);
  }
  sort(MBBFirstInstSlot); // For binary searches.
}

void SinkTrivallyRematInstr::forEveryMBBRegIsLiveIn(
    Register Reg, std::function<void(MachineBasicBlock *MBB)> Callback) const {
  init();
  auto &LI = LIS.getInterval(Reg);
  SmallVector<SlotIndex, 16> LiveSI;
  if (!LI.findIndexesLiveAt(MBBFirstInstSlot, std::back_inserter(LiveSI)))
    return;
  auto &SII = *LIS.getSlotIndexes();
  for (auto SI : LiveSI) {
    auto *MBB = SII.getInstructionFromIndex(SI)->getParent();
    Callback(MBB);
  }
}

void SinkTrivallyRematInstr::cacheLiveOuts(
    const DenseMap<MachineBasicBlock *, SmallVector<Register>> &MBBs) const {
  SmallVector<MachineInstr *, 16> LastMBBMI;
  for (auto &P : MBBs) {
    MachineBasicBlock *MBB = P.first;
    if (LiveOuts.count(MBB))
      continue;
    auto LastI = skipDebugInstructionsForward(MBB->rbegin(), MBB->rend());
    LastMBBMI.push_back(&*LastI);
  }
  if (!LastMBBMI.empty()) {
    for (auto &P : getLiveRegMap(LastMBBMI, true /*After*/, LIS))
      LiveOuts[P.first->getParent()] = P.second;
  }
}

const GCNRegPressure &
SinkTrivallyRematInstr::getPressure(MachineBasicBlock *MBB) const {
  auto I = MBBPressure.find(MBB);
  if (I == MBBPressure.end()) {
    GCNDownwardRPTracker RPTracker(LIS);
    RPTracker.advance(MBB->begin(), MBB->end());
    I = MBBPressure.try_emplace(MBB, RPTracker.moveMaxPressure()).first;
  }
  return I->second;
}

MachineInstr *SinkTrivallyRematInstr::getDefInstr(Register Reg) const {
  assert(MRI.hasOneDef(Reg) && MRI.hasOneNonDBGUse(Reg));
  return MRI.getOneDef(Reg)->getParent();
}

MachineInstr *SinkTrivallyRematInstr::getUserInstr(Register Reg) const {
  assert(MRI.hasOneDef(Reg) && MRI.hasOneNonDBGUse(Reg));
  return &*MRI.use_instr_nodbg_begin(Reg);
}

bool SinkTrivallyRematInstr::fromHighToLowRP(
    DenseMap<MachineBasicBlock *, SmallVector<Register>> &MBBs,
    unsigned Occupancy,
    std::function<bool(MachineBasicBlock *MBB)> Callback) const {

  SmallVector<MachineBasicBlock *, 8> ProcessOrder;
  ProcessOrder.reserve(MBBs.size());
  for (auto &P : MBBs)
    ProcessOrder.push_back(P.first);

  // Move higher RP first.
  sort(ProcessOrder, [&](MachineBasicBlock *A, MachineBasicBlock *B) {
    return getPressure(B).less(ST, getPressure(A), Occupancy) ||
           A->getNumber() < B->getNumber(); // Stabilize output.
  });

  // Select regs for theirs sink target regions.
  for (auto *MBB : ProcessOrder) {
    if (!Callback(MBB))
      return false;
  }
  return true;
}

// Helper class to maintain a pool of available registers for sinking on
// RP tracking.
class RegisterPool {
  SmallVector<Register> Regs;
  BitVector AvailRegs;
  DenseMap<MachineInstr *, SmallVector<unsigned>> MIToAvailRegsMap;
  GCNRegPressure::ExcessMask NoAvailEMask = 0;

public:
  // Select maps registers to instructions: a reg becomes availiable when update
  // method meets an instruction. It returns a pair where:
  //   first:bool - add/skip a reg to the pool.
  //   second:MachineInst* - instruction that makes a reg available on update,
  //     nullptr if the reg is available from the start, like live-through.
  RegisterPool(
      const SmallVectorImpl<Register> &Regs_,
      std::function<std::pair<bool, MachineInstr *>(Register)> Select) {
    Regs.reserve(Regs_.size());
    AvailRegs.resize(Regs_.size());
    for (Register Reg : Regs_) {
      auto R = Select(Reg);
      if (R.first) {
        unsigned Idx = Regs.size();
        Regs.push_back(Reg);
        if (R.second)
          MIToAvailRegsMap[R.second].push_back(Idx);
        else
          AvailRegs.set(Idx);
      }
    }
  }

  // Pop availiable reg satisfying ExcessMask. Registers are selected in the
  // order they had upon creation of this object. Return empty reg on fail.
  Register pop(GCNRegPressure::ExcessMask ExcessMask,
               const MachineRegisterInfo &MRI) {
    // This check prevents redundant bit scans for a sequence of pops when no
    // new available registers added.
    if (0 == (ExcessMask & ~NoAvailEMask))
      return Register();

    for (unsigned IdxReg : AvailRegs.set_bits()) {
      if (ExcessMask & GCNRegPressure::getRegExcessMask(Regs[IdxReg], MRI)) {
        AvailRegs.reset(IdxReg);
        return Regs[IdxReg];
      }
    }
    // If nothing found ExcessMask indicate register kinds that are not avail.
    NoAvailEMask |= ExcessMask;
    return Register();
  }

  // Update availiable regs when MI was met during RP tracking.
  void update(const MachineInstr &MI) {
    auto I = MIToAvailRegsMap.find(&MI);
    if (I != MIToAvailRegsMap.end()) {
      for (auto RI : I->second)
        AvailRegs.set(RI);
      NoAvailEMask = 0;
    }
  }
};

bool SinkTrivallyRematInstr::selectRegsFromSinkSourceMBB(
    MachineBasicBlock *MBB, DenseSet<Register> &SelectedRegs,
    const SmallVectorImpl<Register> &Regs,
    const GCNRegPressure &TargetRP) const {
  LLVM_DEBUG(dbgs() << "Sink source "
                    << printMBBHeader(MBB, Regs, SelectedRegs, TargetRP));
  // TODO: precalculate region live-ins if the reset is slow.
  // Note that a block can be tested several times for different occupancies.
  GCNDownwardRPTracker RPT(LIS);
  RPT.reset(*MBB->begin()); // RPT is in the state before *I in program order.

  // Since a destination register of a sinkable instruction should have one
  // definition it cannot be alive on the entrance of the MBB therefore a reg
  // should either originate from this MBB or be live-through but not both. [1]
  assert(false == any_of(Regs, [&, MBB](Register Reg) {
           return !((MBB == getSinkSourceMBB(Reg)) ^
                    RPT.getLiveMask(Reg).any());
         }));

  // Remove SelectedRegs from live-ins: this can be a live-through reg
  // originating from other source MBB.
  for (Register Reg : SelectedRegs)
    RPT.decIfAlive(Reg);
  LLVM_DEBUG(dbgs() << "Presel RP: " << print(RPT.getPressure(), &ST) << '\n');

  RegisterPool AvailRegs(Regs, [&](Register Reg) {
    return SelectedRegs.contains(Reg)
               ? std::make_pair(false, (MachineInstr *)nullptr)
               : std::make_pair(true, getDefInstr(Reg)); // See [1].
  });
  bool FailedToReachTarget = false;
  while (RPT.advanceBeforeNext()) {
    RPT.advanceToNext(); // RPT is in the state at the MI.
    AvailRegs.update(*RPT.getLastTrackedMI());

    if (auto ExcessMask = RPT.getPressure().exceed(TargetRP)) {
      LLVM_DEBUG(dbgs() << "\tAt " << *RPT.getLastTrackedMI() << "\tRP: "
                        << print(RPT.getPressure(), &ST) << "\tSel: ");
      do {
        Register Reg = AvailRegs.pop(ExcessMask, MRI);
        if (!Reg)
          break;
        LLVM_DEBUG(dbgs() << printReg(Reg) << ", ");
        SelectedRegs.insert(Reg);
        RPT.dec(Reg);
      } while ((ExcessMask = RPT.getPressure().exceed(TargetRP)));

      LLVM_DEBUG(dbgs() << (ExcessMask ? "failed to reach target RP" : "")
                        << "\n\n");

      // Don't stop if we failed to reach TargetRP in a hope that this MBB will
      // be processed as sink target later.
      FailedToReachTarget |= ExcessMask != 0;
    }
  }
  return !FailedToReachTarget;
}

// Select sinkable Regs for a given MBB so that its RP wouldn't exceed TargetRP.
// Regs include live-through ones. Registers are selected in the order of Regs
// so the caller could prioritize them. SelectedRegs is used as in-out argument,
// it may contain registers already selected for sinking earlier.
// Return true if the resulting pressure wouldn't exceed TargetRP.
bool SinkTrivallyRematInstr::selectRegsFromSinkTargetMBB(
    MachineBasicBlock *MBB, DenseSet<Register> &SelectedRegs,
    const SmallVectorImpl<Register> &Regs,
    const GCNRegPressure &TargetRP) const {
  LLVM_DEBUG(dbgs() << "Sink target "
                    << printMBBHeader(MBB, Regs, SelectedRegs, TargetRP));

  auto IsASelectedSinkSourceInstr = [&](MachineInstr &MI) {
    if (MI.getNumOperands() == 0)
      return false;
    MachineOperand &Op0 = MI.getOperand(0);
    return Op0.isReg() && Op0.isDef() && SelectedRegs.contains(Op0.getReg());
  };

  // Track pressure upwards so we can start selecting registers with longer live
  // ranges to minimize the number of registers needed to fit TargetRP.
  auto E = MBB->rend();
  auto I = skipDebugInstructionsForward(MBB->rbegin(), E);
  if (I == E)
    return false;

  GCNUpwardRPTracker RPT(LIS);
  RPT.reset(MRI, LiveOuts[MBB]); // RPT's state is after last instr in the MBB.
  RPT.clearMaxPressure();

  // Regs can target this MBB, be live-through or both because a reg can be used
  // in a loop. [2]
  assert(false == any_of(Regs, [&, MBB](Register Reg) {
           return MBB != getSinkTargetMBB(Reg) && RPT.getLiveMask(Reg).none();
         }));

  // Remove preselected registers from the live-outs if any.
  DenseMap<MachineInstr *, SmallVector<Register>> PreSelRegsAt;
  GCNRegPressure MaxRP = getPressure(MBB);
  for (Register Reg : SelectedRegs) {
    LaneBitmask Mask = RPT.decIfAlive(Reg);
    if (Mask.any()) {
      // Check if this is a selected reg from source instruction in this MBB.
      if (getDefInstr(Reg)->getParent() != MBB)
        // This is a live-through register, decrease max pressure for this MBB.
        MaxRP.dec(Reg, Mask, MRI);
    } else {
      MachineInstr *UserMI = getUserInstr(Reg);
      // Save preselected registers used by a target instruction in this MBB to
      // update pressure when we meet the instruction, see [3].
      if (UserMI->getParent() == MBB)
        PreSelRegsAt[UserMI].push_back(Reg);
    }
  }
  if (!MaxRP.exceed(TargetRP))
    return true;

  RegisterPool AvailRegs(Regs, [&](Register Reg) {
    if (SelectedRegs.contains(Reg))
      return std::make_pair(false, (MachineInstr *)nullptr);
    // A reg can be live-out if the MBB is inside a loop, see [2].
    return std::make_pair(true, RPT.getLiveMask(Reg).none() ? getUserInstr(Reg)
                                                            : nullptr);
  });
  do {
    MachineInstr &MI = *I;
    I = skipDebugInstructionsForward(++I, E);

    // If a sink source instruction was selected from this MBB on the previous
    // stage - skip it.
    if (IsASelectedSinkSourceInstr(MI))
      continue;

    RPT.recede(MI); // RPT is in the state before MI in program order.
    auto RPAtMI = RPT.getMaxPressureAndReset(); // Pressure at the MI.
    if (auto ExcessMask = RPAtMI.exceed(TargetRP)) {
      LLVM_DEBUG(dbgs() << "\tAt " << MI << "\tRP: " << print(RPAtMI, &ST)
                        << "\tSel: ");
      do {
        Register Reg = AvailRegs.pop(ExcessMask, MRI);
        if (!Reg)
          break;
        LLVM_DEBUG(dbgs() << printReg(Reg) << ", ");
        SelectedRegs.insert(Reg);
        LaneBitmask Mask = RPT.dec(Reg);
        RPAtMI.dec(Reg, Mask, MRI);
      } while ((ExcessMask = RPAtMI.exceed(TargetRP)));

      LLVM_DEBUG(dbgs() << (ExcessMask ? "failed to reach target RP" : "")
                        << "\n\n");
      if (ExcessMask)
        return false; // That was the last hope, exiting.
    }

    // Update registers available for sinking after we pass the MI.
    AvailRegs.update(MI);

    // Decrease the pressure by preselected registers as if the instructions
    // defining them were already placed in front of MI. [3]
    auto PSI = PreSelRegsAt.find(&MI);
    if (PSI != PreSelRegsAt.end())
      for_each(PSI->second, [&](Register Reg) { RPT.dec(Reg); });

  } while (I != E);
  return true;
}

// Select registers from Regs so that MBBs where they're live-through doesn't
// excess RP for the Occupancy. SelectedRegs is used as in-out argument, it may
// contain registers already selected for sinking earlier.
// IsMBBProcessed predicate excludes blocks from processing.
void SinkTrivallyRematInstr::selectLiveThroughRegs(
    DenseSet<Register> &SelectedRegs, const SmallVectorImpl<Register> &Regs,
    const GCNRegPressure &TargetRP,
    std::function<bool(MachineBasicBlock *MBB)> IsMBBProcessed) const {

  LLVM_DEBUG(dbgs() << "Selecting live-through regs:\nTarget RP: "
                    << print(TargetRP, &ST));
  // Decrease per MBB pressure by preselected live-through registers.
  DenseMap<MachineBasicBlock *, GCNRegPressure> RPMap;
  for (Register Reg : SelectedRegs) {
    auto *MySinkTargetMBB = getSinkTargetMBB(Reg);
    forEveryMBBRegIsLiveIn(Reg, [&](MachineBasicBlock *MBB) {
      if (IsMBBProcessed(MBB) || MBB == MySinkTargetMBB)
        return;
      auto &RP = RPMap.try_emplace(MBB, getPressure(MBB)).first->second;
      RP.dec(Reg, MRI);
    });
  }

  SmallVector<GCNRegPressure *, 8> RPToUpdate;
  for (Register Reg : Regs) {
    assert(!SelectedRegs.contains(Reg));
    RPToUpdate.clear();
    auto *MySinkTargetMBB = getSinkTargetMBB(Reg);
    bool Selected = false;
    forEveryMBBRegIsLiveIn(Reg, [&, Reg](MachineBasicBlock *MBB) {
      if (IsMBBProcessed(MBB) || MBB == MySinkTargetMBB)
        return;
      auto &RP = RPMap.try_emplace(MBB, getPressure(MBB)).first->second;
      if (!Selected) {
        auto EMask = RP.exceed(TargetRP);
        if (EMask && (EMask & GCNRegPressure::getRegExcessMask(Reg, MRI))) {
          Selected = true;
          LLVM_DEBUG(dbgs() << printReg(Reg) << "@bb." << MBB->getNumber()
                            << " when RP: " << print(RP, &ST));
        } else {
          RPToUpdate.push_back(&RP);
          return;
        }
      }
      RP.dec(Reg, MRI);
    });
    if (Selected) {
      SelectedRegs.insert(Reg);
      for (auto *RP : RPToUpdate)
        RP->dec(Reg, MRI);
    }
  }
  LLVM_DEBUG(dbgs() << '\n');
}

// Return true if the defining instruction for the Reg can be sinked.
bool SinkTrivallyRematInstr::isSinkableReg(Register Reg) const {
  if (!MRI.hasOneDef(Reg) || !MRI.hasOneNonDBGUse(Reg))
    return false;

  MachineOperand *Op = MRI.getOneDef(Reg);
  if (Op->getSubReg())
    return false;

  MachineInstr *DefI = Op->getParent();
  MachineInstr *UseI = &*MRI.use_instr_nodbg_begin(Reg);
  if (DefI->getParent() == UseI->getParent())
    return false; // Not within a single MBB.

  if (!TII.isTriviallyReMaterializable(*DefI))
    return false;

  for (const MachineOperand &MO : DefI->operands())
    if (MO.isReg() && MO.isUse() && MO.getReg().isVirtual())
      return false;

  return true;
}

void SinkTrivallyRematInstr::findExcessiveSinkSourceMBBs(
    const DenseMap<Register, unsigned> &Regs, const GCNRegPressure &TargetRP,
    DenseMap<MachineBasicBlock *, SmallVector<Register>> &SinkSrc) const {
  // Collect sink source MBBs (and theirs sinkables) that have RP excess
  // for the current occupancy.
  SinkSrc.clear();
  for (auto &P : Regs) {
    Register Reg = P.first;
    auto *MBB = getSinkSourceMBB(Reg);
    if (getPressure(MBB).exceed(TargetRP))
      SinkSrc[MBB].push_back(Reg);
  }

  // Add live-through registers.
  for (auto &P : Regs) {
    Register Reg = P.first;
    forEveryMBBRegIsLiveIn(Reg, [&](MachineBasicBlock *MBB) {
      auto I = SinkSrc.find(MBB);
      if (I != SinkSrc.end())
        I->second.push_back(Reg);
    });
  }
}

void SinkTrivallyRematInstr::findExcessiveSinkTargetMBBs(
    const DenseMap<Register, unsigned> &Regs, // Reg -> NumRgnsLiveIn.
    const GCNRegPressure &TargetRP,
    DenseMap<MachineBasicBlock *, SmallVector<Register>> &SinkTgt) const {
  // Collect sink target MBBs (and theirs sinkables) that have RP excess
  // for the current occupancy.
  SinkTgt.clear();
  for (auto &P : Regs) {
    Register Reg = P.first;
    auto *MBB = getSinkTargetMBB(Reg);
    if (getPressure(MBB).exceed(TargetRP))
      SinkTgt[MBB].push_back(Reg);
  }
  if (SinkTgt.empty())
    return;

  // Add live-through registers.
  for (auto &P : Regs) {
    Register Reg = P.first;
    auto *MySinkTargetMBB = getSinkTargetMBB(Reg);
    forEveryMBBRegIsLiveIn(Reg, [&](MachineBasicBlock *MBB) {
      if (MBB == MySinkTargetMBB)
        return;
      auto I = SinkTgt.find(MBB);
      if (I != SinkTgt.end())
        I->second.push_back(Reg);
    });
  }
}

// Does its best to find better NewOccupancy by selecting registers defined by
// trivially rematerializable instructions with a single use to be sinked prior
// their uses. NewOccupancy on input contains the maximum occupancy constraint
// defined by the caller. Return true if NewOccupancy is higher than
// MinOccupancy.
unsigned
SinkTrivallyRematInstr::collectSinkableRegs(DenseSet<Register> &SelectedRegs,
                                            unsigned MinOccupancy,
                                            unsigned MaxOccupancy) const {
  LLVM_DEBUG(dbgs() << "Collecting sinkable regs for " << MF.getName() << '\n');
  DenseMap<Register, unsigned> SinkableRegs;
  for (unsigned I = 0, E = MRI.getNumVirtRegs(); I != E; ++I) {
    Register Reg = Register::index2VirtReg(I);
    if (isSinkableReg(Reg))
      SinkableRegs[Reg] = 0;
  }

  // Find if the game worth the candle - theoretically possible occupancy after
  // sinking: assume that maximum RP for a region is decreased by every reg
  // defined by a sinkable instruction. This is true for regions where such
  // registers are live-through but for source and sink target regions an
  // additional RP check should be made later.
  unsigned NewOccupancy = MaxOccupancy;
  DenseMap<MachineBasicBlock *, GCNRegPressure> BestRP;
  for (auto &P : SinkableRegs) {
    Register Reg = P.first;
    // Calculating RP for source region.
    auto *SrcMBB = getSinkSourceMBB(Reg);
    auto I = BestRP.try_emplace(SrcMBB, getPressure(SrcMBB)).first;
    I->second.dec(Reg, MRI);

    // Calculating RP for target and live-through regions.
    forEveryMBBRegIsLiveIn(Reg, [&](MachineBasicBlock *MBB) {
      auto I = BestRP.try_emplace(MBB, getPressure(MBB)).first;
      I->second.dec(Reg, MRI);
      ++P.second; // Num regions the reg is live-in.
    });
  }
  for (auto &MBB : MF) {
    auto I = BestRP.find(&MBB);
    const auto &RP = I != BestRP.end() ? I->second : getPressure(&MBB);
    NewOccupancy = std::min(NewOccupancy, RP.getOccupancy(ST));
  }
  if (NewOccupancy <= MinOccupancy) {
    LLVM_DEBUG(dbgs() << "The occupancy cannot be improved by sinking\n");
    return NewOccupancy;
  }

  // Predicate to prioritize registers by the number of MBBs there're live in.
  auto MostSeenFirst = [&](Register A, Register B) {
    return SinkableRegs.lookup(A) > SinkableRegs.lookup(B);
  };

  // Try to select registers for sink target regions by lowering occupancy until
  // success. TODO: binary search?
  DenseMap<MachineBasicBlock *, SmallVector<Register>> SinkSrc, SinkTgt;
  GCNRegPressure TargetRP;
  bool Success = true;
  do {
    LLVM_DEBUG(dbgs() << "Trying to reach occupancy " << NewOccupancy << '\n');
    SelectedRegs.clear();

    TargetRP = GCNRegPressure::getMaxPressure(NewOccupancy, ST);

    findExcessiveSinkSourceMBBs(SinkableRegs, TargetRP, SinkSrc);
    fromHighToLowRP(SinkSrc, NewOccupancy, [&](MachineBasicBlock *MBB) {
      auto &Regs = SinkSrc[MBB];
      sort(Regs, MostSeenFirst);
      // Leave MBBs we failed in for the next stage.
      if (selectRegsFromSinkSourceMBB(MBB, SelectedRegs, Regs, TargetRP))
        SinkSrc.erase(MBB);
      return true;
    });

    findExcessiveSinkTargetMBBs(SinkableRegs, TargetRP, SinkTgt);
    for (auto P : SinkSrc) {
      // If failed MMB is not going to be processed this is the fail.
      if (!SinkTgt.count(P.first)) {
        Success = false;
        break;
      }
    }
    if (Success) {
      cacheLiveOuts(SinkTgt);
      Success =
          fromHighToLowRP(SinkTgt, NewOccupancy, [&](MachineBasicBlock *MBB) {
            auto &Regs = SinkTgt[MBB];
            sort(Regs, MostSeenFirst);
            return selectRegsFromSinkTargetMBB(MBB, SelectedRegs, Regs,
                                               TargetRP);
          });
    }
  } while (!Success && --NewOccupancy > MinOccupancy);

  if (!Success) {
    LLVM_DEBUG(dbgs() << "No occupancy improvement\n\n");
    return MinOccupancy;
  }

  // At this point we've selected registers that can be sinked to its target
  // regions and proved these regions would have RP for the NewOccupancy. Now
  // process regions with RP excess where yet unselected sinkable registers are
  // live-through. It's already proven these regions would also have RP suitable
  // for the NewOccupancy by the theoretical occupancy test.
  SmallVector<Register, 8> Regs;
  for (auto &P : SinkableRegs) {
    if (!SelectedRegs.contains(P.first))
      Regs.push_back(P.first);
  }
  sort(Regs, MostSeenFirst);
  selectLiveThroughRegs(
      SelectedRegs, Regs, TargetRP, [&](MachineBasicBlock *MBB) {
        return SinkSrc.count(MBB) > 0 || SinkTgt.count(MBB) > 0;
      });

  LLVM_DEBUG(dbgs() << "Reached occupancy " << NewOccupancy << ", sel regs: ";
             for (Register Reg
                  : SelectedRegs) dbgs()
             << printReg(Reg) << ", ";
             dbgs() << "\n\n");
  return NewOccupancy;
}

// Does the actual sinking of trivially rematerializable instructions defining
// Regs in front of their uses.
void SinkTrivallyRematInstr::sinkTriviallyRematInstrs(
    const DenseSet<Register> &Regs) const {
  for (Register Reg : Regs) {
    MachineInstr *UserMI = &*MRI.use_instr_nodbg_begin(Reg);
    auto InsPos = MachineBasicBlock::iterator(UserMI);
    MachineInstr *DefInst = MRI.getOneDef(Reg)->getParent();
    // Rematerialize MI to its use block. Since we are only rematerializing
    // instructions that do not have any virtual reg uses, we do not need to
    // call LiveRangeEdit::allUsesAvailableAt() and
    // LiveRangeEdit::canRematerializeAt().
    TII.reMaterialize(*InsPos->getParent(), InsPos, Reg, 0, *DefInst, TRI);
    MachineInstr *NewMI = &*std::prev(InsPos);

    // Update live intervals.
    LIS.removeInterval(Reg);
    LIS.RemoveMachineInstrFromMaps(*DefInst);
    DefInst->eraseFromParent();
    LIS.InsertMachineInstrInMaps(*NewMI);
    LIS.createAndComputeVirtRegInterval(Reg);
  }
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD
Printable SinkTrivallyRematInstr::printReg(Register Reg) const {
  return llvm::printGCNRegShort(Reg, MRI);
}

LLVM_DUMP_METHOD
Printable
SinkTrivallyRematInstr::printMBBHeader(MachineBasicBlock *MBB,
                                       const SmallVectorImpl<Register> &Regs,
                                       const DenseSet<Register> &SelectedRegs,
                                       const GCNRegPressure &TargetRP) const {
  return Printable([&, MBB](raw_ostream &OS) {
    OS << "bb." << MBB->getNumber() << ", preselected regs: ";
    for (Register Reg : SelectedRegs) {
      OS << printReg(Reg) << ", ";
    }
    OS << "\nregs to select: ";
    for (Register Reg : Regs) {
      if (SelectedRegs.contains(Reg))
        continue;
      OS << printReg(Reg) << ", ";
    }
    OS << "\nTarget RP: " << print(TargetRP, &ST)
       << "Actual RP: " << print(getPressure(MBB), &ST);
  });
}
#endif