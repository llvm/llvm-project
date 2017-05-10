//===------------------------- GCNRegPressure.cpp - -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
//
//===----------------------------------------------------------------------===//

#include "GCNRegPressure.h"

using namespace llvm;

#define DEBUG_TYPE "misched"

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD
void llvm::printLivesAt(SlotIndex SI,
                        const LiveIntervals &LIS,
                        const MachineRegisterInfo &MRI) {
  dbgs() << "Live regs at " << SI << ": "
         << *LIS.getInstructionFromIndex(SI);
  unsigned Num = 0;
  for (unsigned I = 0, E = MRI.getNumVirtRegs(); I != E; ++I) {
    const unsigned Reg = TargetRegisterInfo::index2VirtReg(I);
    if (MRI.reg_nodbg_empty(Reg))
      continue;
    const auto &LI = LIS.getInterval(Reg);
    if (LI.hasSubRanges()) {
      bool firstTime = true;
      for (const auto &S : LI.subranges()) {
        if (!S.liveAt(SI)) continue;
        if (firstTime) {
          dbgs() << "  " << PrintReg(Reg, MRI.getTargetRegisterInfo())
                 << '\n';
          firstTime = false;
        }
        dbgs() << "  " << S << '\n';
        ++Num;
      }
    } else if (LI.liveAt(SI)) {
      dbgs() << "  " << LI << '\n';
      ++Num;
    }
  }
  if (!Num) dbgs() << "  <none>\n";
}

static bool isEqual(const GCNRPTracker::LiveRegSet &S1,
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

static GCNRPTracker::LiveRegSet
stripEmpty(const GCNRPTracker::LiveRegSet &LR) {
  GCNRPTracker::LiveRegSet Res;
  for (const auto &P : LR) {
    if (P.second.any())
      Res.insert(P);
  }
  return Res;
}
#endif

///////////////////////////////////////////////////////////////////////////////
// GCNRegPressure

unsigned GCNRegPressure::getRegKind(unsigned Reg,
                                    const MachineRegisterInfo &MRI) {
  assert(TargetRegisterInfo::isVirtualRegister(Reg));
  const auto RC = MRI.getRegClass(Reg);
  auto STI = static_cast<const SIRegisterInfo*>(MRI.getTargetRegisterInfo());
  return STI->isSGPRClass(RC) ?
    (STI->getRegSizeInBits(*RC) == 32 ? SGPR32 : SGPR_TUPLE) :
    (STI->getRegSizeInBits(*RC) == 32 ? VGPR32 : VGPR_TUPLE);
}

void GCNRegPressure::inc(unsigned Reg,
                         LaneBitmask PrevMask,
                         LaneBitmask NewMask,
                         const MachineRegisterInfo &MRI) {
  if (NewMask == PrevMask)
    return;

  int Sign = 1;
  if (NewMask < PrevMask) {
    std::swap(NewMask, PrevMask);
    Sign = -1;
  }
#ifndef NDEBUG
  const auto MaxMask = MRI.getMaxLaneMaskForVReg(Reg);
#endif
  switch (auto Kind = getRegKind(Reg, MRI)) {
  case SGPR32:
  case VGPR32:
    assert(PrevMask.none() && NewMask == MaxMask);
    Value[Kind] += Sign;
    break;

  case SGPR_TUPLE:
  case VGPR_TUPLE:
    assert(NewMask < MaxMask || NewMask == MaxMask);
    assert(PrevMask < NewMask);

    Value[Kind == SGPR_TUPLE ? SGPR32 : VGPR32] +=
      Sign * countPopulation((~PrevMask & NewMask).getAsInteger());

    if (PrevMask.none()) {
      assert(NewMask.any());
      Value[Kind] += Sign * MRI.getPressureSets(Reg).getWeight();
    }
    break;

  default: llvm_unreachable("Unknown register kind");
  }
}

bool GCNRegPressure::less(const SISubtarget &ST,
                          const GCNRegPressure& O,
                          unsigned MaxOccupancy) const {
  const auto SGPROcc = std::min(MaxOccupancy,
                                ST.getOccupancyWithNumSGPRs(getSGPRNum()));
  const auto VGPROcc = std::min(MaxOccupancy,
                                ST.getOccupancyWithNumVGPRs(getVGPRNum()));
  const auto OtherSGPROcc = std::min(MaxOccupancy,
                                ST.getOccupancyWithNumSGPRs(O.getSGPRNum()));
  const auto OtherVGPROcc = std::min(MaxOccupancy,
                                ST.getOccupancyWithNumVGPRs(O.getVGPRNum()));

  const auto Occ = std::min(SGPROcc, VGPROcc);
  const auto OtherOcc = std::min(OtherSGPROcc, OtherVGPROcc);
  if (Occ != OtherOcc)
    return Occ > OtherOcc;

  bool SGPRImportant = SGPROcc < VGPROcc;
  const bool OtherSGPRImportant = OtherSGPROcc < OtherVGPROcc;

  // if both pressures disagree on what is more important compare vgprs
  if (SGPRImportant != OtherSGPRImportant) {
    SGPRImportant = false;
  }

  // compare large regs pressure
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
  return SGPRImportant ? (getSGPRNum() < O.getSGPRNum()):
                         (getVGPRNum() < O.getVGPRNum());
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD
void GCNRegPressure::print(raw_ostream &OS, const SISubtarget *ST) const {
  OS << "VGPRs: " << getVGPRNum();
  if (ST) OS << "(O" << ST->getOccupancyWithNumVGPRs(getVGPRNum()) << ')';
  OS << ", SGPRs: " << getSGPRNum();
  if (ST) OS << "(O" << ST->getOccupancyWithNumSGPRs(getSGPRNum()) << ')';
  OS << ", LVGPR WT: " << getVGPRTuplesWeight()
     << ", LSGPR WT: " << getSGPRTuplesWeight();
  if (ST) OS << " -> Occ: " << getOccupancy(*ST);
  OS << '\n';
}
#endif

///////////////////////////////////////////////////////////////////////////////
// GCNRPTracker

LaneBitmask llvm::getLiveLaneMask(unsigned Reg,
                                  SlotIndex SI,
                                  const LiveIntervals &LIS,
                                  const MachineRegisterInfo &MRI) {
  assert(!MRI.reg_nodbg_empty(Reg));
  LaneBitmask LiveMask;
  const auto &LI = LIS.getInterval(Reg);
  if (LI.hasSubRanges()) {
    for (const auto &S : LI.subranges())
      if (S.liveAt(SI)) {
        LiveMask |= S.LaneMask;
        assert(LiveMask < MRI.getMaxLaneMaskForVReg(Reg) ||
               LiveMask == MRI.getMaxLaneMaskForVReg(Reg));
      }
  } else if (LI.liveAt(SI)) {
    LiveMask = MRI.getMaxLaneMaskForVReg(Reg);
  }
  return LiveMask;
}

GCNRPTracker::LiveRegSet llvm::getLiveRegs(SlotIndex SI,
                                           const LiveIntervals &LIS,
                                           const MachineRegisterInfo &MRI) {
  GCNRPTracker::LiveRegSet LiveRegs;
  for (unsigned I = 0, E = MRI.getNumVirtRegs(); I != E; ++I) {
    auto Reg = TargetRegisterInfo::index2VirtReg(I);
    if (MRI.reg_nodbg_empty(Reg))
      continue;
    auto LiveMask = getLiveLaneMask(Reg, SI, LIS, MRI);
    if (LiveMask.any())
      LiveRegs[Reg] = LiveMask;
  }
  return LiveRegs;
}

void GCNUpwardRPTracker::reset(const MachineInstr &MI) {
  MRI = &MI.getParent()->getParent()->getRegInfo();
  LiveRegs = getLiveRegsAfter(MI, LIS);
  MaxPressure = CurPressure = getRegPressure(*MRI, LiveRegs);
}

LaneBitmask GCNUpwardRPTracker::getDefRegMask(const MachineOperand &MO) const {
  assert(MO.isDef() && MO.isReg() &&
    TargetRegisterInfo::isVirtualRegister(MO.getReg()));

  // We don't rely on read-undef flag because in case of tentative schedule
  // tracking it isn't set correctly yet. This works correctly however since
  // use mask has been tracked before using LIS.
  return MO.getSubReg() == 0 ?
    MRI->getMaxLaneMaskForVReg(MO.getReg()) :
    MRI->getTargetRegisterInfo()->getSubRegIndexLaneMask(MO.getSubReg());
}

LaneBitmask GCNUpwardRPTracker::getUsedRegMask(const MachineOperand &MO) const {
  assert(MO.isUse() && MO.isReg() &&
         TargetRegisterInfo::isVirtualRegister(MO.getReg()));

  if (auto SubReg = MO.getSubReg())
    return MRI->getTargetRegisterInfo()->getSubRegIndexLaneMask(SubReg);

  auto MaxMask = MRI->getMaxLaneMaskForVReg(MO.getReg());
  if (MaxMask.getAsInteger() == 1) // cannot have subregs
    return MaxMask;

  // For a tentative schedule LIS isn't updated yet but livemask should remain
  // the same on any schedule. Subreg defs can be reordered but they all must
  // dominate uses anyway.
  auto SI = LIS.getInstructionIndex(*MO.getParent()).getBaseIndex();
  return getLiveLaneMask(MO.getReg(), SI, LIS, *MRI);
}

void GCNUpwardRPTracker::recede(const MachineInstr &MI) {
  assert(MRI && "call reset first");

  LastTrackedMI = &MI;

  if (MI.isDebugValue())
    return;

  // process all defs first to ensure early clobbers are handled correctly
  // iterating over operands() to catch implicit defs
  for (const auto &MO : MI.operands()) {
    if (!MO.isReg() || !MO.isDef() ||
      !TargetRegisterInfo::isVirtualRegister(MO.getReg()))
      continue;

    auto Reg = MO.getReg();
    auto &LiveMask = LiveRegs[Reg];
    auto PrevMask = LiveMask;
    LiveMask &= ~getDefRegMask(MO);
    CurPressure.inc(Reg, PrevMask, LiveMask, *MRI);
  }

  // then all uses
  for (const auto &MO : MI.uses()) {
    if (!MO.isReg() || !MO.readsReg() ||
      !TargetRegisterInfo::isVirtualRegister(MO.getReg()))
      continue;

    auto Reg = MO.getReg();
    auto &LiveMask = LiveRegs[Reg];
    auto PrevMask = LiveMask;
    LiveMask |= getUsedRegMask(MO);
    CurPressure.inc(Reg, PrevMask, LiveMask, *MRI);
  }

  MaxPressure = max(MaxPressure, CurPressure);
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD
static void reportMismatch(const GCNRPTracker::LiveRegSet &LISLR,
                           const GCNRPTracker::LiveRegSet &TrackedLR,
                           const TargetRegisterInfo *TRI) {
  for (auto const &P : TrackedLR) {
    auto I = LISLR.find(P.first);
    if (I == LISLR.end()) {
      dbgs() << "  " << PrintReg(P.first, TRI)
             << ":L" << PrintLaneMask(P.second)
             << " isn't found in LIS reported set\n";
    }
    else if (I->second != P.second) {
      dbgs() << "  " << PrintReg(P.first, TRI)
        << " masks doesn't match: LIS reported "
        << PrintLaneMask(I->second)
        << ", tracked "
        << PrintLaneMask(P.second)
        << '\n';
    }
  }
  for (auto const &P : LISLR) {
    auto I = TrackedLR.find(P.first);
    if (I == TrackedLR.end()) {
      dbgs() << "  " << PrintReg(P.first, TRI)
             << ":L" << PrintLaneMask(P.second)
             << " isn't found in tracked set\n";
    }
  }
}

bool GCNUpwardRPTracker::isValid() const {
  const auto &SI = LIS.getInstructionIndex(*LastTrackedMI).getBaseIndex();
  const auto LISLR = llvm::getLiveRegs(SI, LIS, *MRI);
  const auto TrackedLR = stripEmpty(LiveRegs);

  if (!isEqual(LISLR, TrackedLR)) {
    dbgs() << "\nGCNUpwardRPTracker error: Tracked and"
              " LIS reported livesets mismatch:\n";
    printLivesAt(SI, LIS, *MRI);
    reportMismatch(LISLR, TrackedLR, MRI->getTargetRegisterInfo());
    return false;
  }

  auto LISPressure = getRegPressure(*MRI, LISLR);
  if (LISPressure != CurPressure) {
    dbgs() << "GCNUpwardRPTracker error: Pressure sets different\nTracked: ";
    CurPressure.print(dbgs());
    dbgs() << "LIS rpt: ";
    LISPressure.print(dbgs());
    return false;
  }
  return true;
}

#endif
