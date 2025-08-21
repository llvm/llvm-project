//===- GCNRegPressure.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the GCNRegPressure class.
///
//===----------------------------------------------------------------------===//

#include "GCNRegPressure.h"
#include "AMDGPU.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/CodeGen/RegisterPressure.h"

using namespace llvm;

#define DEBUG_TYPE "machine-scheduler"

bool llvm::isEqual(const GCNRPTracker::LiveRegSet &S1,
                   const GCNRPTracker::LiveRegSet &S2) {
  if (S1.size() != S2.size())
    return false;

  for (const auto &P : S1) {
    auto I = S2.find(P.first);
    if (I == S2.end() || I->second != P.second)
      return false;
  }
  return true;
}

///////////////////////////////////////////////////////////////////////////////
// GCNRegPressure

unsigned GCNRegPressure::getRegKind(const TargetRegisterClass *RC,
                                    const SIRegisterInfo *STI) {
  return STI->isSGPRClass(RC)
             ? SGPR
             : (STI->isAGPRClass(RC)
                    ? AGPR
                    : (STI->isVectorSuperClass(RC) ? AVGPR : VGPR));
}

void GCNRegPressure::inc(unsigned Reg,
                         LaneBitmask PrevMask,
                         LaneBitmask NewMask,
                         const MachineRegisterInfo &MRI) {
  unsigned NewNumCoveredRegs = SIRegisterInfo::getNumCoveredRegs(NewMask);
  unsigned PrevNumCoveredRegs = SIRegisterInfo::getNumCoveredRegs(PrevMask);
  if (NewNumCoveredRegs == PrevNumCoveredRegs)
    return;

  int Sign = 1;
  if (NewMask < PrevMask) {
    std::swap(NewMask, PrevMask);
    std::swap(NewNumCoveredRegs, PrevNumCoveredRegs);
    Sign = -1;
  }
  assert(PrevMask < NewMask && PrevNumCoveredRegs < NewNumCoveredRegs &&
         "prev mask should always be lesser than new");

  const TargetRegisterClass *RC = MRI.getRegClass(Reg);
  const TargetRegisterInfo *TRI = MRI.getTargetRegisterInfo();
  const SIRegisterInfo *STI = static_cast<const SIRegisterInfo *>(TRI);
  unsigned RegKind = getRegKind(RC, STI);
  if (TRI->getRegSizeInBits(*RC) != 32) {
    // Reg is from a tuple register class.
    if (PrevMask.none()) {
      unsigned TupleIdx = TOTAL_KINDS + RegKind;
      Value[TupleIdx] += Sign * TRI->getRegClassWeight(RC).RegWeight;
    }
    // Pressure scales with number of new registers covered by the new mask.
    // Note when true16 is enabled, we can no longer safely use the following
    // approach to calculate the difference in the number of 32-bit registers
    // between two masks:
    //
    // Sign *= SIRegisterInfo::getNumCoveredRegs(~PrevMask & NewMask);
    //
    // The issue is that the mask calculation `~PrevMask & NewMask` doesn't
    // properly account for partial usage of a 32-bit register when dealing with
    // 16-bit registers.
    //
    // Consider this example:
    // Assume PrevMask = 0b0010 and NewMask = 0b1111. Here, the correct register
    // usage difference should be 1, because even though PrevMask uses only half
    // of a 32-bit register, it should still be counted as a full register use.
    // However, the mask calculation yields `~PrevMask & NewMask = 0b1101`, and
    // calling `getNumCoveredRegs` returns 2 instead of 1. This incorrect
    // calculation can lead to integer overflow when Sign = -1.
    Sign *= NewNumCoveredRegs - PrevNumCoveredRegs;
  }
  Value[RegKind] += Sign;
}

bool GCNRegPressure::less(const MachineFunction &MF, const GCNRegPressure &O,
                          unsigned MaxOccupancy) const {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  unsigned DynamicVGPRBlockSize =
      MF.getInfo<SIMachineFunctionInfo>()->getDynamicVGPRBlockSize();

  const auto SGPROcc = std::min(MaxOccupancy,
                                ST.getOccupancyWithNumSGPRs(getSGPRNum()));
  const auto VGPROcc = std::min(
      MaxOccupancy, ST.getOccupancyWithNumVGPRs(getVGPRNum(ST.hasGFX90AInsts()),
                                                DynamicVGPRBlockSize));
  const auto OtherSGPROcc = std::min(MaxOccupancy,
                                ST.getOccupancyWithNumSGPRs(O.getSGPRNum()));
  const auto OtherVGPROcc =
      std::min(MaxOccupancy,
               ST.getOccupancyWithNumVGPRs(O.getVGPRNum(ST.hasGFX90AInsts()),
                                           DynamicVGPRBlockSize));

  const auto Occ = std::min(SGPROcc, VGPROcc);
  const auto OtherOcc = std::min(OtherSGPROcc, OtherVGPROcc);

  // Give first precedence to the better occupancy.
  if (Occ != OtherOcc)
    return Occ > OtherOcc;

  unsigned MaxVGPRs = ST.getMaxNumVGPRs(MF);
  unsigned MaxSGPRs = ST.getMaxNumSGPRs(MF);

  // SGPR excess pressure conditions
  unsigned ExcessSGPR = std::max(static_cast<int>(getSGPRNum() - MaxSGPRs), 0);
  unsigned OtherExcessSGPR =
      std::max(static_cast<int>(O.getSGPRNum() - MaxSGPRs), 0);

  auto WaveSize = ST.getWavefrontSize();
  // The number of virtual VGPRs required to handle excess SGPR
  unsigned VGPRForSGPRSpills = (ExcessSGPR + (WaveSize - 1)) / WaveSize;
  unsigned OtherVGPRForSGPRSpills =
      (OtherExcessSGPR + (WaveSize - 1)) / WaveSize;

  unsigned MaxArchVGPRs = ST.getAddressableNumArchVGPRs();

  // Unified excess pressure conditions, accounting for VGPRs used for SGPR
  // spills
  unsigned ExcessVGPR =
      std::max(static_cast<int>(getVGPRNum(ST.hasGFX90AInsts()) +
                                VGPRForSGPRSpills - MaxVGPRs),
               0);
  unsigned OtherExcessVGPR =
      std::max(static_cast<int>(O.getVGPRNum(ST.hasGFX90AInsts()) +
                                OtherVGPRForSGPRSpills - MaxVGPRs),
               0);
  // Arch VGPR excess pressure conditions, accounting for VGPRs used for SGPR
  // spills
  unsigned ExcessArchVGPR = std::max(
      static_cast<int>(getVGPRNum(false) + VGPRForSGPRSpills - MaxArchVGPRs),
      0);
  unsigned OtherExcessArchVGPR =
      std::max(static_cast<int>(O.getVGPRNum(false) + OtherVGPRForSGPRSpills -
                                MaxArchVGPRs),
               0);
  // AGPR excess pressure conditions
  unsigned ExcessAGPR = std::max(
      static_cast<int>(ST.hasGFX90AInsts() ? (getAGPRNum() - MaxArchVGPRs)
                                           : (getAGPRNum() - MaxVGPRs)),
      0);
  unsigned OtherExcessAGPR = std::max(
      static_cast<int>(ST.hasGFX90AInsts() ? (O.getAGPRNum() - MaxArchVGPRs)
                                           : (O.getAGPRNum() - MaxVGPRs)),
      0);

  bool ExcessRP = ExcessSGPR || ExcessVGPR || ExcessArchVGPR || ExcessAGPR;
  bool OtherExcessRP = OtherExcessSGPR || OtherExcessVGPR ||
                       OtherExcessArchVGPR || OtherExcessAGPR;

  // Give second precedence to the reduced number of spills to hold the register
  // pressure.
  if (ExcessRP || OtherExcessRP) {
    // The difference in excess VGPR pressure, after including VGPRs used for
    // SGPR spills
    int VGPRDiff = ((OtherExcessVGPR + OtherExcessArchVGPR + OtherExcessAGPR) -
                    (ExcessVGPR + ExcessArchVGPR + ExcessAGPR));

    int SGPRDiff = OtherExcessSGPR - ExcessSGPR;

    if (VGPRDiff != 0)
      return VGPRDiff > 0;
    if (SGPRDiff != 0) {
      unsigned PureExcessVGPR =
          std::max(static_cast<int>(getVGPRNum(ST.hasGFX90AInsts()) - MaxVGPRs),
                   0) +
          std::max(static_cast<int>(getVGPRNum(false) - MaxArchVGPRs), 0);
      unsigned OtherPureExcessVGPR =
          std::max(
              static_cast<int>(O.getVGPRNum(ST.hasGFX90AInsts()) - MaxVGPRs),
              0) +
          std::max(static_cast<int>(O.getVGPRNum(false) - MaxArchVGPRs), 0);

      // If we have a special case where there is a tie in excess VGPR, but one
      // of the pressures has VGPR usage from SGPR spills, prefer the pressure
      // with SGPR spills.
      if (PureExcessVGPR != OtherPureExcessVGPR)
        return SGPRDiff < 0;
      // If both pressures have the same excess pressure before and after
      // accounting for SGPR spills, prefer fewer SGPR spills.
      return SGPRDiff > 0;
    }
  }

  bool SGPRImportant = SGPROcc < VGPROcc;
  const bool OtherSGPRImportant = OtherSGPROcc < OtherVGPROcc;

  // If both pressures disagree on what is more important compare vgprs.
  if (SGPRImportant != OtherSGPRImportant) {
    SGPRImportant = false;
  }

  // Give third precedence to lower register tuple pressure.
  bool SGPRFirst = SGPRImportant;
  for (int I = 2; I > 0; --I, SGPRFirst = !SGPRFirst) {
    if (SGPRFirst) {
      auto SW = getSGPRTuplesWeight();
      auto OtherSW = O.getSGPRTuplesWeight();
      if (SW != OtherSW)
        return SW < OtherSW;
    } else {
      auto VW = getVGPRTuplesWeight();
      auto OtherVW = O.getVGPRTuplesWeight();
      if (VW != OtherVW)
        return VW < OtherVW;
    }
  }

  // Give final precedence to lower general RP.
  return SGPRImportant ? (getSGPRNum() < O.getSGPRNum()):
                         (getVGPRNum(ST.hasGFX90AInsts()) <
                          O.getVGPRNum(ST.hasGFX90AInsts()));
}

Printable llvm::print(const GCNRegPressure &RP, const GCNSubtarget *ST,
                      unsigned DynamicVGPRBlockSize) {
  return Printable([&RP, ST, DynamicVGPRBlockSize](raw_ostream &OS) {
    OS << "VGPRs: " << RP.getArchVGPRNum() << ' '
       << "AGPRs: " << RP.getAGPRNum();
    if (ST)
      OS << "(O"
         << ST->getOccupancyWithNumVGPRs(RP.getVGPRNum(ST->hasGFX90AInsts()),
                                         DynamicVGPRBlockSize)
         << ')';
    OS << ", SGPRs: " << RP.getSGPRNum();
    if (ST)
      OS << "(O" << ST->getOccupancyWithNumSGPRs(RP.getSGPRNum()) << ')';
    OS << ", LVGPR WT: " << RP.getVGPRTuplesWeight()
       << ", LSGPR WT: " << RP.getSGPRTuplesWeight();
    if (ST)
      OS << " -> Occ: " << RP.getOccupancy(*ST, DynamicVGPRBlockSize);
    OS << '\n';
  });
}

static LaneBitmask getDefRegMask(const MachineOperand &MO,
                                 const MachineRegisterInfo &MRI) {
  assert(MO.isDef() && MO.isReg() && MO.getReg().isVirtual());

  // We don't rely on read-undef flag because in case of tentative schedule
  // tracking it isn't set correctly yet. This works correctly however since
  // use mask has been tracked before using LIS.
  return MO.getSubReg() == 0 ?
    MRI.getMaxLaneMaskForVReg(MO.getReg()) :
    MRI.getTargetRegisterInfo()->getSubRegIndexLaneMask(MO.getSubReg());
}

static void
collectVirtualRegUses(SmallVectorImpl<VRegMaskOrUnit> &VRegMaskOrUnits,
                      const MachineInstr &MI, const LiveIntervals &LIS,
                      const MachineRegisterInfo &MRI) {

  auto &TRI = *MRI.getTargetRegisterInfo();
  for (const auto &MO : MI.operands()) {
    if (!MO.isReg() || !MO.getReg().isVirtual())
      continue;
    if (!MO.isUse() || !MO.readsReg())
      continue;

    Register Reg = MO.getReg();
    auto I = llvm::find_if(VRegMaskOrUnits, [Reg](const VRegMaskOrUnit &RM) {
      return RM.RegUnit == Reg;
    });

    auto &P = I == VRegMaskOrUnits.end()
                  ? VRegMaskOrUnits.emplace_back(Reg, LaneBitmask::getNone())
                  : *I;

    P.LaneMask |= MO.getSubReg() ? TRI.getSubRegIndexLaneMask(MO.getSubReg())
                                 : MRI.getMaxLaneMaskForVReg(Reg);
  }

  SlotIndex InstrSI;
  for (auto &P : VRegMaskOrUnits) {
    auto &LI = LIS.getInterval(P.RegUnit);
    if (!LI.hasSubRanges())
      continue;

    // For a tentative schedule LIS isn't updated yet but livemask should
    // remain the same on any schedule. Subreg defs can be reordered but they
    // all must dominate uses anyway.
    if (!InstrSI)
      InstrSI = LIS.getInstructionIndex(MI).getBaseIndex();

    P.LaneMask = getLiveLaneMask(LI, InstrSI, MRI, P.LaneMask);
  }
}

/// Mostly copy/paste from CodeGen/RegisterPressure.cpp
static LaneBitmask getLanesWithProperty(
    const LiveIntervals &LIS, const MachineRegisterInfo &MRI,
    bool TrackLaneMasks, Register RegUnit, SlotIndex Pos,
    LaneBitmask SafeDefault,
    function_ref<bool(const LiveRange &LR, SlotIndex Pos)> Property) {
  if (RegUnit.isVirtual()) {
    const LiveInterval &LI = LIS.getInterval(RegUnit);
    LaneBitmask Result;
    if (TrackLaneMasks && LI.hasSubRanges()) {
      for (const LiveInterval::SubRange &SR : LI.subranges()) {
        if (Property(SR, Pos))
          Result |= SR.LaneMask;
      }
    } else if (Property(LI, Pos)) {
      Result = TrackLaneMasks ? MRI.getMaxLaneMaskForVReg(RegUnit)
                              : LaneBitmask::getAll();
    }

    return Result;
  }

  const LiveRange *LR = LIS.getCachedRegUnit(RegUnit);
  if (LR == nullptr)
    return SafeDefault;
  return Property(*LR, Pos) ? LaneBitmask::getAll() : LaneBitmask::getNone();
}

/// Mostly copy/paste from CodeGen/RegisterPressure.cpp
/// Helper to find a vreg use between two indices {PriorUseIdx, NextUseIdx}.
/// The query starts with a lane bitmask which gets lanes/bits removed for every
/// use we find.
static LaneBitmask findUseBetween(unsigned Reg, LaneBitmask LastUseMask,
                                  SlotIndex PriorUseIdx, SlotIndex NextUseIdx,
                                  const MachineRegisterInfo &MRI,
                                  const SIRegisterInfo *TRI,
                                  const LiveIntervals *LIS,
                                  bool Upward = false) {
  for (const MachineOperand &MO : MRI.use_nodbg_operands(Reg)) {
    if (MO.isUndef())
      continue;
    const MachineInstr *MI = MO.getParent();
    SlotIndex InstSlot = LIS->getInstructionIndex(*MI).getRegSlot();
    bool InRange = Upward ? (InstSlot > PriorUseIdx && InstSlot <= NextUseIdx)
                          : (InstSlot >= PriorUseIdx && InstSlot < NextUseIdx);
    if (!InRange)
      continue;

    unsigned SubRegIdx = MO.getSubReg();
    LaneBitmask UseMask = TRI->getSubRegIndexLaneMask(SubRegIdx);
    LastUseMask &= ~UseMask;
    if (LastUseMask.none())
      return LaneBitmask::getNone();
  }
  return LastUseMask;
}

////////////////////////////////////////////////////////////////////////////////
// GCNRPTarget

GCNRPTarget::GCNRPTarget(const MachineFunction &MF, const GCNRegPressure &RP)
    : GCNRPTarget(RP, MF) {
  const Function &F = MF.getFunction();
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  setTarget(ST.getMaxNumSGPRs(F), ST.getMaxNumVGPRs(F));
}

GCNRPTarget::GCNRPTarget(unsigned NumSGPRs, unsigned NumVGPRs,
                         const MachineFunction &MF, const GCNRegPressure &RP)
    : GCNRPTarget(RP, MF) {
  setTarget(NumSGPRs, NumVGPRs);
}

GCNRPTarget::GCNRPTarget(unsigned Occupancy, const MachineFunction &MF,
                         const GCNRegPressure &RP)
    : GCNRPTarget(RP, MF) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  unsigned DynamicVGPRBlockSize =
      MF.getInfo<SIMachineFunctionInfo>()->getDynamicVGPRBlockSize();
  setTarget(ST.getMaxNumSGPRs(Occupancy, /*Addressable=*/false),
            ST.getMaxNumVGPRs(Occupancy, DynamicVGPRBlockSize));
}

void GCNRPTarget::setTarget(unsigned NumSGPRs, unsigned NumVGPRs) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  MaxSGPRs = std::min(ST.getAddressableNumSGPRs(), NumSGPRs);
  MaxVGPRs = std::min(ST.getAddressableNumArchVGPRs(), NumVGPRs);
  if (UnifiedRF) {
    unsigned DynamicVGPRBlockSize =
        MF.getInfo<SIMachineFunctionInfo>()->getDynamicVGPRBlockSize();
    MaxUnifiedVGPRs =
        std::min(ST.getAddressableNumVGPRs(DynamicVGPRBlockSize), NumVGPRs);
  } else {
    MaxUnifiedVGPRs = 0;
  }
}

bool GCNRPTarget::isSaveBeneficial(Register Reg) const {
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  const TargetRegisterClass *RC = MRI.getRegClass(Reg);
  const TargetRegisterInfo *TRI = MRI.getTargetRegisterInfo();
  const SIRegisterInfo *SRI = static_cast<const SIRegisterInfo *>(TRI);

  if (SRI->isSGPRClass(RC))
    return RP.getSGPRNum() > MaxSGPRs;
  unsigned NumVGPRs =
      SRI->isAGPRClass(RC) ? RP.getAGPRNum() : RP.getArchVGPRNum();
  // The addressable limit must always be respected.
  if (NumVGPRs > MaxVGPRs)
    return true;
  // For unified RFs, combined VGPR usage limit must be respected as well.
  return UnifiedRF && RP.getVGPRNum(true) > MaxUnifiedVGPRs;
}

bool GCNRPTarget::satisfied() const {
  if (RP.getSGPRNum() > MaxSGPRs || RP.getVGPRNum(false) > MaxVGPRs)
    return false;
  if (UnifiedRF && RP.getVGPRNum(true) > MaxUnifiedVGPRs)
    return false;
  return true;
}

///////////////////////////////////////////////////////////////////////////////
// GCNRPTracker

LaneBitmask llvm::getLiveLaneMask(unsigned Reg, SlotIndex SI,
                                  const LiveIntervals &LIS,
                                  const MachineRegisterInfo &MRI,
                                  LaneBitmask LaneMaskFilter) {
  return getLiveLaneMask(LIS.getInterval(Reg), SI, MRI, LaneMaskFilter);
}

LaneBitmask llvm::getLiveLaneMask(const LiveInterval &LI, SlotIndex SI,
                                  const MachineRegisterInfo &MRI,
                                  LaneBitmask LaneMaskFilter) {
  LaneBitmask LiveMask;
  if (LI.hasSubRanges()) {
    for (const auto &S : LI.subranges())
      if ((S.LaneMask & LaneMaskFilter).any() && S.liveAt(SI)) {
        LiveMask |= S.LaneMask;
        assert(LiveMask == (LiveMask & MRI.getMaxLaneMaskForVReg(LI.reg())));
      }
  } else if (LI.liveAt(SI)) {
    LiveMask = MRI.getMaxLaneMaskForVReg(LI.reg());
  }
  LiveMask &= LaneMaskFilter;
  return LiveMask;
}

GCNRPTracker::LiveRegSet llvm::getLiveRegs(SlotIndex SI,
                                           const LiveIntervals &LIS,
                                           const MachineRegisterInfo &MRI) {
  GCNRPTracker::LiveRegSet LiveRegs;
  for (unsigned I = 0, E = MRI.getNumVirtRegs(); I != E; ++I) {
    auto Reg = Register::index2VirtReg(I);
    if (!LIS.hasInterval(Reg))
      continue;
    auto LiveMask = getLiveLaneMask(Reg, SI, LIS, MRI);
    if (LiveMask.any())
      LiveRegs[Reg] = LiveMask;
  }
  return LiveRegs;
}

void GCNRPTracker::reset(const MachineInstr &MI,
                         const LiveRegSet *LiveRegsCopy,
                         bool After) {
  const MachineFunction &MF = *MI.getMF();
  MRI = &MF.getRegInfo();
  if (LiveRegsCopy) {
    if (&LiveRegs != LiveRegsCopy)
      LiveRegs = *LiveRegsCopy;
  } else {
    LiveRegs = After ? getLiveRegsAfter(MI, LIS)
                     : getLiveRegsBefore(MI, LIS);
  }

  MaxPressure = CurPressure = getRegPressure(*MRI, LiveRegs);
}

void GCNRPTracker::reset(const MachineRegisterInfo &MRI_,
                         const LiveRegSet &LiveRegs_) {
  MRI = &MRI_;
  LiveRegs = LiveRegs_;
  LastTrackedMI = nullptr;
  MaxPressure = CurPressure = getRegPressure(MRI_, LiveRegs_);
}

/// Mostly copy/paste from CodeGen/RegisterPressure.cpp
LaneBitmask GCNRPTracker::getLastUsedLanes(Register RegUnit,
                                           SlotIndex Pos) const {
  return getLanesWithProperty(
      LIS, *MRI, true, RegUnit, Pos.getBaseIndex(), LaneBitmask::getNone(),
      [](const LiveRange &LR, SlotIndex Pos) {
        const LiveRange::Segment *S = LR.getSegmentContaining(Pos);
        return S != nullptr && S->end == Pos.getRegSlot();
      });
}

////////////////////////////////////////////////////////////////////////////////
// GCNUpwardRPTracker

void GCNUpwardRPTracker::recede(const MachineInstr &MI) {
  assert(MRI && "call reset first");

  LastTrackedMI = &MI;

  if (MI.isDebugInstr())
    return;

  // Kill all defs.
  GCNRegPressure DefPressure, ECDefPressure;
  bool HasECDefs = false;
  for (const MachineOperand &MO : MI.all_defs()) {
    if (!MO.getReg().isVirtual())
      continue;

    Register Reg = MO.getReg();
    LaneBitmask DefMask = getDefRegMask(MO, *MRI);

    // Treat a def as fully live at the moment of definition: keep a record.
    if (MO.isEarlyClobber()) {
      ECDefPressure.inc(Reg, LaneBitmask::getNone(), DefMask, *MRI);
      HasECDefs = true;
    } else
      DefPressure.inc(Reg, LaneBitmask::getNone(), DefMask, *MRI);

    auto I = LiveRegs.find(Reg);
    if (I == LiveRegs.end())
      continue;

    LaneBitmask &LiveMask = I->second;
    LaneBitmask PrevMask = LiveMask;
    LiveMask &= ~DefMask;
    CurPressure.inc(Reg, PrevMask, LiveMask, *MRI);
    if (LiveMask.none())
      LiveRegs.erase(I);
  }

  // Update MaxPressure with defs pressure.
  DefPressure += CurPressure;
  if (HasECDefs)
    DefPressure += ECDefPressure;
  MaxPressure = max(DefPressure, MaxPressure);

  // Make uses alive.
  SmallVector<VRegMaskOrUnit, 8> RegUses;
  collectVirtualRegUses(RegUses, MI, LIS, *MRI);
  for (const VRegMaskOrUnit &U : RegUses) {
    LaneBitmask &LiveMask = LiveRegs[U.RegUnit];
    LaneBitmask PrevMask = LiveMask;
    LiveMask |= U.LaneMask;
    CurPressure.inc(U.RegUnit, PrevMask, LiveMask, *MRI);
  }

  // Update MaxPressure with uses plus early-clobber defs pressure.
  MaxPressure = HasECDefs ? max(CurPressure + ECDefPressure, MaxPressure)
                          : max(CurPressure, MaxPressure);

  assert(CurPressure == getRegPressure(*MRI, LiveRegs));
}

////////////////////////////////////////////////////////////////////////////////
// GCNDownwardRPTracker

bool GCNDownwardRPTracker::reset(const MachineInstr &MI,
                                 const LiveRegSet *LiveRegsCopy) {
  MRI = &MI.getParent()->getParent()->getRegInfo();
  LastTrackedMI = nullptr;
  MBBEnd = MI.getParent()->end();
  NextMI = &MI;
  NextMI = skipDebugInstructionsForward(NextMI, MBBEnd);
  if (NextMI == MBBEnd)
    return false;
  GCNRPTracker::reset(*NextMI, LiveRegsCopy, false);
  return true;
}

bool GCNDownwardRPTracker::advanceBeforeNext(MachineInstr *MI,
                                             bool UseInternalIterator) {
  assert(MRI && "call reset first");
  SlotIndex SI;
  const MachineInstr *CurrMI;
  if (UseInternalIterator) {
    if (!LastTrackedMI)
      return NextMI == MBBEnd;

    assert(NextMI == MBBEnd || !NextMI->isDebugInstr());
    CurrMI = LastTrackedMI;

    SI = NextMI == MBBEnd
             ? LIS.getInstructionIndex(*LastTrackedMI).getDeadSlot()
             : LIS.getInstructionIndex(*NextMI).getBaseIndex();
  } else { //! UseInternalIterator
    SI = LIS.getInstructionIndex(*MI).getBaseIndex();
    CurrMI = MI;
  }

  assert(SI.isValid());

  // Remove dead registers or mask bits.
  SmallSet<Register, 8> SeenRegs;
  for (auto &MO : CurrMI->operands()) {
    if (!MO.isReg() || !MO.getReg().isVirtual())
      continue;
    if (MO.isUse() && !MO.readsReg())
      continue;
    if (!UseInternalIterator && MO.isDef())
      continue;
    if (!SeenRegs.insert(MO.getReg()).second)
      continue;
    const LiveInterval &LI = LIS.getInterval(MO.getReg());
    if (LI.hasSubRanges()) {
      auto It = LiveRegs.end();
      for (const auto &S : LI.subranges()) {
        if (!S.liveAt(SI)) {
          if (It == LiveRegs.end()) {
            It = LiveRegs.find(MO.getReg());
            if (It == LiveRegs.end())
              llvm_unreachable("register isn't live");
          }
          auto PrevMask = It->second;
          It->second &= ~S.LaneMask;
          CurPressure.inc(MO.getReg(), PrevMask, It->second, *MRI);
        }
      }
      if (It != LiveRegs.end() && It->second.none())
        LiveRegs.erase(It);
    } else if (!LI.liveAt(SI)) {
      auto It = LiveRegs.find(MO.getReg());
      if (It == LiveRegs.end())
        llvm_unreachable("register isn't live");
      CurPressure.inc(MO.getReg(), It->second, LaneBitmask::getNone(), *MRI);
      LiveRegs.erase(It);
    }
  }

  MaxPressure = max(MaxPressure, CurPressure);

  LastTrackedMI = nullptr;

  return UseInternalIterator && (NextMI == MBBEnd);
}

void GCNDownwardRPTracker::advanceToNext(MachineInstr *MI,
                                         bool UseInternalIterator) {
  if (UseInternalIterator) {
    LastTrackedMI = &*NextMI++;
    NextMI = skipDebugInstructionsForward(NextMI, MBBEnd);
  } else {
    LastTrackedMI = MI;
  }

  const MachineInstr *CurrMI = LastTrackedMI;

  // Add new registers or mask bits.
  for (const auto &MO : CurrMI->all_defs()) {
    Register Reg = MO.getReg();
    if (!Reg.isVirtual())
      continue;
    auto &LiveMask = LiveRegs[Reg];
    auto PrevMask = LiveMask;
    LiveMask |= getDefRegMask(MO, *MRI);
    CurPressure.inc(Reg, PrevMask, LiveMask, *MRI);
  }

  MaxPressure = max(MaxPressure, CurPressure);
}

bool GCNDownwardRPTracker::advance(MachineInstr *MI, bool UseInternalIterator) {
  if (UseInternalIterator && NextMI == MBBEnd)
    return false;

  advanceBeforeNext(MI, UseInternalIterator);
  advanceToNext(MI, UseInternalIterator);
  if (!UseInternalIterator) {
    // We must remove any dead def lanes from the current RP
    advanceBeforeNext(MI, true);
  }
  return true;
}

bool GCNDownwardRPTracker::advance(MachineBasicBlock::const_iterator End) {
  while (NextMI != End)
    if (!advance()) return false;
  return true;
}

bool GCNDownwardRPTracker::advance(MachineBasicBlock::const_iterator Begin,
                                   MachineBasicBlock::const_iterator End,
                                   const LiveRegSet *LiveRegsCopy) {
  reset(*Begin, LiveRegsCopy);
  return advance(End);
}

Printable llvm::reportMismatch(const GCNRPTracker::LiveRegSet &LISLR,
                               const GCNRPTracker::LiveRegSet &TrackedLR,
                               const TargetRegisterInfo *TRI, StringRef Pfx) {
  return Printable([&LISLR, &TrackedLR, TRI, Pfx](raw_ostream &OS) {
    for (auto const &P : TrackedLR) {
      auto I = LISLR.find(P.first);
      if (I == LISLR.end()) {
        OS << Pfx << printReg(P.first, TRI) << ":L" << PrintLaneMask(P.second)
           << " isn't found in LIS reported set\n";
      } else if (I->second != P.second) {
        OS << Pfx << printReg(P.first, TRI)
           << " masks doesn't match: LIS reported " << PrintLaneMask(I->second)
           << ", tracked " << PrintLaneMask(P.second) << '\n';
      }
    }
    for (auto const &P : LISLR) {
      auto I = TrackedLR.find(P.first);
      if (I == TrackedLR.end()) {
        OS << Pfx << printReg(P.first, TRI) << ":L" << PrintLaneMask(P.second)
           << " isn't found in tracked set\n";
      }
    }
  });
}

GCNRegPressure
GCNDownwardRPTracker::bumpDownwardPressure(const MachineInstr *MI,
                                           const SIRegisterInfo *TRI) const {
  assert(!MI->isDebugOrPseudoInstr() && "Expect a nondebug instruction.");

  SlotIndex SlotIdx;
  SlotIdx = LIS.getInstructionIndex(*MI).getRegSlot();

  // Account for register pressure similar to RegPressureTracker::recede().
  RegisterOperands RegOpers;
  RegOpers.collect(*MI, *TRI, *MRI, true, /*IgnoreDead=*/false);
  RegOpers.adjustLaneLiveness(LIS, *MRI, SlotIdx);
  GCNRegPressure TempPressure = CurPressure;

  for (const VRegMaskOrUnit &Use : RegOpers.Uses) {
    Register Reg = Use.RegUnit;
    if (!Reg.isVirtual())
      continue;
    LaneBitmask LastUseMask = getLastUsedLanes(Reg, SlotIdx);
    if (LastUseMask.none())
      continue;
    // The LastUseMask is queried from the liveness information of instruction
    // which may be further down the schedule. Some lanes may actually not be
    // last uses for the current position.
    // FIXME: allow the caller to pass in the list of vreg uses that remain
    // to be bottom-scheduled to avoid searching uses at each query.
    SlotIndex CurrIdx;
    const MachineBasicBlock *MBB = MI->getParent();
    MachineBasicBlock::const_iterator IdxPos = skipDebugInstructionsForward(
        LastTrackedMI ? LastTrackedMI : MBB->begin(), MBB->end());
    if (IdxPos == MBB->end()) {
      CurrIdx = LIS.getMBBEndIdx(MBB);
    } else {
      CurrIdx = LIS.getInstructionIndex(*IdxPos).getRegSlot();
    }

    LastUseMask =
        findUseBetween(Reg, LastUseMask, CurrIdx, SlotIdx, *MRI, TRI, &LIS);
    if (LastUseMask.none())
      continue;

    auto It = LiveRegs.find(Reg);
    LaneBitmask LiveMask = It != LiveRegs.end() ? It->second : LaneBitmask(0);
    LaneBitmask NewMask = LiveMask & ~LastUseMask;
    TempPressure.inc(Reg, LiveMask, NewMask, *MRI);
  }

  // Generate liveness for defs.
  for (const VRegMaskOrUnit &Def : RegOpers.Defs) {
    Register Reg = Def.RegUnit;
    if (!Reg.isVirtual())
      continue;
    auto It = LiveRegs.find(Reg);
    LaneBitmask LiveMask = It != LiveRegs.end() ? It->second : LaneBitmask(0);
    LaneBitmask NewMask = LiveMask | Def.LaneMask;
    TempPressure.inc(Reg, LiveMask, NewMask, *MRI);
  }

  return TempPressure;
}

bool GCNUpwardRPTracker::isValid() const {
  const auto &SI = LIS.getInstructionIndex(*LastTrackedMI).getBaseIndex();
  const auto LISLR = llvm::getLiveRegs(SI, LIS, *MRI);
  const auto &TrackedLR = LiveRegs;

  if (!isEqual(LISLR, TrackedLR)) {
    dbgs() << "\nGCNUpwardRPTracker error: Tracked and"
              " LIS reported livesets mismatch:\n"
           << print(LISLR, *MRI);
    reportMismatch(LISLR, TrackedLR, MRI->getTargetRegisterInfo());
    return false;
  }

  auto LISPressure = getRegPressure(*MRI, LISLR);
  if (LISPressure != CurPressure) {
    dbgs() << "GCNUpwardRPTracker error: Pressure sets different\nTracked: "
           << print(CurPressure) << "LIS rpt: " << print(LISPressure);
    return false;
  }
  return true;
}

Printable llvm::print(const GCNRPTracker::LiveRegSet &LiveRegs,
                      const MachineRegisterInfo &MRI) {
  return Printable([&LiveRegs, &MRI](raw_ostream &OS) {
    const TargetRegisterInfo *TRI = MRI.getTargetRegisterInfo();
    for (unsigned I = 0, E = MRI.getNumVirtRegs(); I != E; ++I) {
      Register Reg = Register::index2VirtReg(I);
      auto It = LiveRegs.find(Reg);
      if (It != LiveRegs.end() && It->second.any())
        OS << ' ' << printVRegOrUnit(Reg, TRI) << ':'
           << PrintLaneMask(It->second);
    }
    OS << '\n';
  });
}

void GCNRegPressure::dump() const { dbgs() << print(*this); }

static cl::opt<bool> UseDownwardTracker(
    "amdgpu-print-rp-downward",
    cl::desc("Use GCNDownwardRPTracker for GCNRegPressurePrinter pass"),
    cl::init(false), cl::Hidden);

char llvm::GCNRegPressurePrinter::ID = 0;
char &llvm::GCNRegPressurePrinterID = GCNRegPressurePrinter::ID;

INITIALIZE_PASS(GCNRegPressurePrinter, "amdgpu-print-rp", "", true, true)

// Return lanemask of Reg's subregs that are live-through at [Begin, End] and
// are fully covered by Mask.
static LaneBitmask
getRegLiveThroughMask(const MachineRegisterInfo &MRI, const LiveIntervals &LIS,
                      Register Reg, SlotIndex Begin, SlotIndex End,
                      LaneBitmask Mask = LaneBitmask::getAll()) {

  auto IsInOneSegment = [Begin, End](const LiveRange &LR) -> bool {
    auto *Segment = LR.getSegmentContaining(Begin);
    return Segment && Segment->contains(End);
  };

  LaneBitmask LiveThroughMask;
  const LiveInterval &LI = LIS.getInterval(Reg);
  if (LI.hasSubRanges()) {
    for (auto &SR : LI.subranges()) {
      if ((SR.LaneMask & Mask) == SR.LaneMask && IsInOneSegment(SR))
        LiveThroughMask |= SR.LaneMask;
    }
  } else {
    LaneBitmask RegMask = MRI.getMaxLaneMaskForVReg(Reg);
    if ((RegMask & Mask) == RegMask && IsInOneSegment(LI))
      LiveThroughMask = RegMask;
  }

  return LiveThroughMask;
}

bool GCNRegPressurePrinter::runOnMachineFunction(MachineFunction &MF) {
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  const TargetRegisterInfo *TRI = MRI.getTargetRegisterInfo();
  const LiveIntervals &LIS = getAnalysis<LiveIntervalsWrapperPass>().getLIS();

  auto &OS = dbgs();

// Leading spaces are important for YAML syntax.
#define PFX "  "

  OS << "---\nname: " << MF.getName() << "\nbody:             |\n";

  auto printRP = [](const GCNRegPressure &RP) {
    return Printable([&RP](raw_ostream &OS) {
      OS << format(PFX "  %-5d", RP.getSGPRNum())
         << format(" %-5d", RP.getVGPRNum(false));
    });
  };

  auto ReportLISMismatchIfAny = [&](const GCNRPTracker::LiveRegSet &TrackedLR,
                                    const GCNRPTracker::LiveRegSet &LISLR) {
    if (LISLR != TrackedLR) {
      OS << PFX "  mis LIS: " << llvm::print(LISLR, MRI)
         << reportMismatch(LISLR, TrackedLR, TRI, PFX "    ");
    }
  };

  // Register pressure before and at an instruction (in program order).
  SmallVector<std::pair<GCNRegPressure, GCNRegPressure>, 16> RP;

  for (auto &MBB : MF) {
    RP.clear();
    RP.reserve(MBB.size());

    OS << PFX;
    MBB.printName(OS);
    OS << ":\n";

    SlotIndex MBBStartSlot = LIS.getSlotIndexes()->getMBBStartIdx(&MBB);
    SlotIndex MBBEndSlot = LIS.getSlotIndexes()->getMBBEndIdx(&MBB);

    GCNRPTracker::LiveRegSet LiveIn, LiveOut;
    GCNRegPressure RPAtMBBEnd;

    if (UseDownwardTracker) {
      if (MBB.empty()) {
        LiveIn = LiveOut = getLiveRegs(MBBStartSlot, LIS, MRI);
        RPAtMBBEnd = getRegPressure(MRI, LiveIn);
      } else {
        GCNDownwardRPTracker RPT(LIS);
        RPT.reset(MBB.front());

        LiveIn = RPT.getLiveRegs();

        while (!RPT.advanceBeforeNext()) {
          GCNRegPressure RPBeforeMI = RPT.getPressure();
          RPT.advanceToNext();
          RP.emplace_back(RPBeforeMI, RPT.getPressure());
        }

        LiveOut = RPT.getLiveRegs();
        RPAtMBBEnd = RPT.getPressure();
      }
    } else {
      GCNUpwardRPTracker RPT(LIS);
      RPT.reset(MRI, MBBEndSlot);

      LiveOut = RPT.getLiveRegs();
      RPAtMBBEnd = RPT.getPressure();

      for (auto &MI : reverse(MBB)) {
        RPT.resetMaxPressure();
        RPT.recede(MI);
        if (!MI.isDebugInstr())
          RP.emplace_back(RPT.getPressure(), RPT.getMaxPressure());
      }

      LiveIn = RPT.getLiveRegs();
    }

    OS << PFX "  Live-in: " << llvm::print(LiveIn, MRI);
    if (!UseDownwardTracker)
      ReportLISMismatchIfAny(LiveIn, getLiveRegs(MBBStartSlot, LIS, MRI));

    OS << PFX "  SGPR  VGPR\n";
    int I = 0;
    for (auto &MI : MBB) {
      if (!MI.isDebugInstr()) {
        auto &[RPBeforeInstr, RPAtInstr] =
            RP[UseDownwardTracker ? I : (RP.size() - 1 - I)];
        ++I;
        OS << printRP(RPBeforeInstr) << '\n' << printRP(RPAtInstr) << "  ";
      } else
        OS << PFX "               ";
      MI.print(OS);
    }
    OS << printRP(RPAtMBBEnd) << '\n';

    OS << PFX "  Live-out:" << llvm::print(LiveOut, MRI);
    if (UseDownwardTracker)
      ReportLISMismatchIfAny(LiveOut, getLiveRegs(MBBEndSlot, LIS, MRI));

    GCNRPTracker::LiveRegSet LiveThrough;
    for (auto [Reg, Mask] : LiveIn) {
      LaneBitmask MaskIntersection = Mask & LiveOut.lookup(Reg);
      if (MaskIntersection.any()) {
        LaneBitmask LTMask = getRegLiveThroughMask(
            MRI, LIS, Reg, MBBStartSlot, MBBEndSlot, MaskIntersection);
        if (LTMask.any())
          LiveThrough[Reg] = LTMask;
      }
    }
    OS << PFX "  Live-thr:" << llvm::print(LiveThrough, MRI);
    OS << printRP(getRegPressure(MRI, LiveThrough)) << '\n';
  }
  OS << "...\n";
  return false;

#undef PFX
}
