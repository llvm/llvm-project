//===- GCNRegPressure.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the GCNRegPressure class, which tracks registry pressure
/// by bookkeeping number of SGPR/VGPRs used, weights for large SGPR/VGPRs. It
/// also implements a compare function, which compares different register
/// pressures, and declares one with max occupancy as winner.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_GCNREGPRESSURE_H
#define LLVM_LIB_TARGET_AMDGPU_GCNREGPRESSURE_H

#include "GCNSubtarget.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/RegisterPressure.h"
#include <algorithm>
#include <array>

namespace llvm {

class MachineRegisterInfo;
class raw_ostream;
class SlotIndex;

struct GCNRegPressure {
  enum RegKind { SGPR, VGPR, AGPR, AVGPR, TOTAL_KINDS };

  static constexpr const char *getName(RegKind Kind) {
    const char *Names[] = {"SGPR", "VGPR", "AGPR", "AVGPR"};
    assert(Kind < TOTAL_KINDS);
    return Names[Kind];
  }

  GCNRegPressure() {
    clear();
  }

  bool empty() const {
    return !Value[SGPR] && !Value[VGPR] && !Value[AGPR] && !Value[AVGPR];
  }

  void clear() { Value.fill(0); }

  unsigned getNumRegs(RegKind Kind) const {
    assert(Kind < TOTAL_KINDS);
    return Value[Kind];
  }

  /// \returns the SGPR32 pressure
  unsigned getSGPRNum() const { return Value[SGPR]; }
  /// \returns the aggregated ArchVGPR32, AccVGPR32, and Pseudo AVGPR pressure
  /// dependent upon \p UnifiedVGPRFile
  unsigned getVGPRNum(bool UnifiedVGPRFile) const {
    if (UnifiedVGPRFile) {
      return Value[AGPR]
                 ? getUnifiedVGPRNum(Value[VGPR], Value[AGPR], Value[AVGPR])
                 : Value[VGPR] + Value[AVGPR];
    }
    // AVGPR assignment priority is based on the width of the register. Account
    // AVGPR pressure as VGPR.
    return std::max(Value[VGPR] + Value[AVGPR], Value[AGPR]);
  }

  /// Returns the aggregated VGPR pressure, assuming \p NumArchVGPRs ArchVGPRs
  /// \p NumAGPRs AGPRS, and \p NumAVGPRs AVGPRs for a target with a unified
  /// VGPR file.
  inline static unsigned getUnifiedVGPRNum(unsigned NumArchVGPRs,
                                           unsigned NumAGPRs,
                                           unsigned NumAVGPRs) {

    // Assume AVGPRs will be assigned as VGPRs.
    return alignTo(NumArchVGPRs + NumAVGPRs,
                   AMDGPU::IsaInfo::getArchVGPRAllocGranule()) +
           NumAGPRs;
  }

  /// \returns the ArchVGPR32 pressure, plus the AVGPRS which we assume will be
  /// allocated as VGPR
  unsigned getArchVGPRNum() const { return Value[VGPR] + Value[AVGPR]; }
  /// \returns the AccVGPR32 pressure
  unsigned getAGPRNum() const { return Value[AGPR]; }
  /// \returns the AVGPR32 pressure
  unsigned getAVGPRNum() const { return Value[AVGPR]; }

  unsigned getVGPRTuplesWeight() const {
    return std::max(Value[TOTAL_KINDS + VGPR] + Value[TOTAL_KINDS + AVGPR],
                    Value[TOTAL_KINDS + AGPR]);
  }
  unsigned getSGPRTuplesWeight() const { return Value[TOTAL_KINDS + SGPR]; }

  unsigned getOccupancy(const GCNSubtarget &ST,
                        unsigned DynamicVGPRBlockSize) const {
    return std::min(ST.getOccupancyWithNumSGPRs(getSGPRNum()),
                    ST.getOccupancyWithNumVGPRs(getVGPRNum(ST.hasGFX90AInsts()),
                                                DynamicVGPRBlockSize));
  }

  void inc(unsigned Reg,
           LaneBitmask PrevMask,
           LaneBitmask NewMask,
           const MachineRegisterInfo &MRI);

  bool higherOccupancy(const GCNSubtarget &ST, const GCNRegPressure &O,
                       unsigned DynamicVGPRBlockSize) const {
    return getOccupancy(ST, DynamicVGPRBlockSize) >
           O.getOccupancy(ST, DynamicVGPRBlockSize);
  }

  /// Compares \p this GCNRegpressure to \p O, returning true if \p this is
  /// less. Since GCNRegpressure contains different types of pressures, and due
  /// to target-specific pecularities (e.g. we care about occupancy rather than
  /// raw register usage), we determine if \p this GCNRegPressure is less than
  /// \p O based on the following tiered comparisons (in order order of
  /// precedence):
  /// 1. Better occupancy
  /// 2. Less spilling (first preference to VGPR spills, then to SGPR spills)
  /// 3. Less tuple register pressure (first preference to VGPR tuples if we
  /// determine that SGPR pressure is not important)
  /// 4. Less raw register pressure (first preference to VGPR tuples if we
  /// determine that SGPR pressure is not important)
  bool less(const MachineFunction &MF, const GCNRegPressure &O,
            unsigned MaxOccupancy = std::numeric_limits<unsigned>::max()) const;

  bool operator==(const GCNRegPressure &O) const { return Value == O.Value; }

  bool operator!=(const GCNRegPressure &O) const {
    return !(*this == O);
  }

  GCNRegPressure &operator+=(const GCNRegPressure &RHS) {
    for (unsigned I = 0; I < ValueArraySize; ++I)
      Value[I] += RHS.Value[I];
    return *this;
  }

  GCNRegPressure &operator-=(const GCNRegPressure &RHS) {
    for (unsigned I = 0; I < ValueArraySize; ++I)
      Value[I] -= RHS.Value[I];
    return *this;
  }

  void dump() const;

  static RegKind getRegKind(unsigned Reg, const MachineRegisterInfo &MRI) {
    const TargetRegisterInfo *TRI = MRI.getTargetRegisterInfo();
    const SIRegisterInfo *STI = static_cast<const SIRegisterInfo *>(TRI);
    return (RegKind)getRegKind(MRI.getRegClass(Reg), STI);
  }

private:
  static constexpr unsigned ValueArraySize = TOTAL_KINDS * 2;

  /// Pressure for all register kinds (first all regular registers kinds, then
  /// all tuple register kinds).
  std::array<unsigned, ValueArraySize> Value;

  static unsigned getRegKind(const TargetRegisterClass *RC,
                             const SIRegisterInfo *STI);

  friend GCNRegPressure max(const GCNRegPressure &P1,
                            const GCNRegPressure &P2);

  friend Printable print(const GCNRegPressure &RP, const GCNSubtarget *ST,
                         unsigned DynamicVGPRBlockSize);
};

inline GCNRegPressure max(const GCNRegPressure &P1, const GCNRegPressure &P2) {
  GCNRegPressure Res;
  for (unsigned I = 0; I < GCNRegPressure::ValueArraySize; ++I)
    Res.Value[I] = std::max(P1.Value[I], P2.Value[I]);
  return Res;
}

inline GCNRegPressure operator+(const GCNRegPressure &P1,
                                const GCNRegPressure &P2) {
  GCNRegPressure Sum = P1;
  Sum += P2;
  return Sum;
}

inline GCNRegPressure operator-(const GCNRegPressure &P1,
                                const GCNRegPressure &P2) {
  GCNRegPressure Diff = P1;
  Diff -= P2;
  return Diff;
}

////////////////////////////////////////////////////////////////////////////////
// GCNRPTarget

/// Models a register pressure target, allowing to evaluate and track register
/// savings against that target from a starting \ref GCNRegPressure.
class GCNRPTarget {
public:
  /// Sets up the target such that the register pressure starting at \p RP does
  /// not show register spilling on function \p MF (w.r.t. the function's
  /// mininum target occupancy).
  GCNRPTarget(const MachineFunction &MF, const GCNRegPressure &RP);

  /// Sets up the target such that the register pressure starting at \p RP does
  /// not use more than \p NumSGPRs SGPRs and \p NumVGPRs VGPRs on function \p
  /// MF.
  GCNRPTarget(unsigned NumSGPRs, unsigned NumVGPRs, const MachineFunction &MF,
              const GCNRegPressure &RP);

  /// Sets up the target such that the register pressure starting at \p RP does
  /// not prevent achieving an occupancy of at least \p Occupancy on function
  /// \p MF.
  GCNRPTarget(unsigned Occupancy, const MachineFunction &MF,
              const GCNRegPressure &RP);

  /// Changes the target (same semantics as constructor).
  void setTarget(unsigned NumSGPRs, unsigned NumVGPRs);

  const GCNRegPressure &getCurrentRP() const { return RP; }

  void setRP(const GCNRegPressure &NewRP) { RP = NewRP; }

  /// Determines whether saving virtual register \p Reg will be beneficial
  /// towards achieving the RP target.
  bool isSaveBeneficial(Register Reg) const;

  /// Saves virtual register \p Reg with lanemask \p Mask.
  void saveReg(Register Reg, LaneBitmask Mask, const MachineRegisterInfo &MRI) {
    RP.inc(Reg, Mask, LaneBitmask::getNone(), MRI);
  }

  /// Whether the current RP is at or below the defined pressure target.
  bool satisfied() const;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  friend raw_ostream &operator<<(raw_ostream &OS, const GCNRPTarget &Target) {
    OS << "Actual/Target: " << Target.RP.getSGPRNum() << '/' << Target.MaxSGPRs
       << " SGPRs, " << Target.RP.getArchVGPRNum() << '/' << Target.MaxVGPRs
       << " ArchVGPRs, " << Target.RP.getAGPRNum() << '/' << Target.MaxVGPRs
       << " AGPRs";

    if (Target.MaxUnifiedVGPRs) {
      OS << ", " << Target.RP.getVGPRNum(true) << '/' << Target.MaxUnifiedVGPRs
         << " VGPRs (unified)";
    }
    return OS;
  }
#endif

private:
  const MachineFunction &MF;
  const bool UnifiedRF;

  /// Current register pressure.
  GCNRegPressure RP;

  /// Target number of SGPRs.
  unsigned MaxSGPRs;
  /// Target number of ArchVGPRs and AGPRs.
  unsigned MaxVGPRs;
  /// Target number of overall VGPRs for subtargets with unified RFs. Always 0
  /// for subtargets with non-unified RFs.
  unsigned MaxUnifiedVGPRs;

  GCNRPTarget(const GCNRegPressure &RP, const MachineFunction &MF)
      : MF(MF), UnifiedRF(MF.getSubtarget<GCNSubtarget>().hasGFX90AInsts()),
        RP(RP) {}
};

///////////////////////////////////////////////////////////////////////////////
// GCNRPTracker

class GCNRPTracker {
public:
  using LiveRegSet = DenseMap<unsigned, LaneBitmask>;

protected:
  const LiveIntervals &LIS;
  LiveRegSet LiveRegs;
  GCNRegPressure CurPressure, MaxPressure;
  const MachineInstr *LastTrackedMI = nullptr;
  mutable const MachineRegisterInfo *MRI = nullptr;

  GCNRPTracker(const LiveIntervals &LIS_) : LIS(LIS_) {}

  void reset(const MachineInstr &MI, const LiveRegSet *LiveRegsCopy,
             bool After);

  /// Mostly copy/paste from CodeGen/RegisterPressure.cpp
  void bumpDeadDefs(ArrayRef<VRegMaskOrUnit> DeadDefs);

  LaneBitmask getLastUsedLanes(Register RegUnit, SlotIndex Pos) const;

public:
  // reset tracker and set live register set to the specified value.
  void reset(const MachineRegisterInfo &MRI_, const LiveRegSet &LiveRegs_);
  // live regs for the current state
  const decltype(LiveRegs) &getLiveRegs() const { return LiveRegs; }
  const MachineInstr *getLastTrackedMI() const { return LastTrackedMI; }

  void clearMaxPressure() { MaxPressure.clear(); }

  GCNRegPressure getPressure() const { return CurPressure; }

  decltype(LiveRegs) moveLiveRegs() {
    return std::move(LiveRegs);
  }
};

GCNRPTracker::LiveRegSet
getLiveRegs(SlotIndex SI, const LiveIntervals &LIS,
            const MachineRegisterInfo &MRI,
            GCNRegPressure::RegKind RegKind = GCNRegPressure::TOTAL_KINDS);

////////////////////////////////////////////////////////////////////////////////
// GCNUpwardRPTracker

class GCNUpwardRPTracker : public GCNRPTracker {
public:
  GCNUpwardRPTracker(const LiveIntervals &LIS_) : GCNRPTracker(LIS_) {}

  using GCNRPTracker::reset;

  /// reset tracker at the specified slot index \p SI.
  void reset(const MachineRegisterInfo &MRI, SlotIndex SI) {
    GCNRPTracker::reset(MRI, llvm::getLiveRegs(SI, LIS, MRI));
  }

  /// reset tracker to the end of the \p MBB.
  void reset(const MachineBasicBlock &MBB) {
    SlotIndex MBBLastSlot = LIS.getSlotIndexes()->getMBBLastIdx(&MBB);
    reset(MBB.getParent()->getRegInfo(), MBBLastSlot);
  }

  /// reset tracker to the point just after \p MI (in program order).
  void reset(const MachineInstr &MI) {
    reset(MI.getMF()->getRegInfo(), LIS.getInstructionIndex(MI).getDeadSlot());
  }

  /// Move to the state of RP just before the \p MI . If \p UseInternalIterator
  /// is set, also update the internal iterators. Setting \p UseInternalIterator
  /// to false allows for an externally managed iterator / program order.
  void recede(const MachineInstr &MI);

  /// \p returns whether the tracker's state after receding MI corresponds
  /// to reported by LIS.
  bool isValid() const;

  const GCNRegPressure &getMaxPressure() const { return MaxPressure; }

  void resetMaxPressure() { MaxPressure = CurPressure; }

  GCNRegPressure getMaxPressureAndReset() {
    GCNRegPressure RP = MaxPressure;
    resetMaxPressure();
    return RP;
  }
};

////////////////////////////////////////////////////////////////////////////////
// GCNDownwardRPTracker

class GCNDownwardRPTracker : public GCNRPTracker {
  // Last position of reset or advanceBeforeNext
  MachineBasicBlock::const_iterator NextMI;

  MachineBasicBlock::const_iterator MBBEnd;

public:
  GCNDownwardRPTracker(const LiveIntervals &LIS_) : GCNRPTracker(LIS_) {}

  using GCNRPTracker::reset;

  MachineBasicBlock::const_iterator getNext() const { return NextMI; }

  /// \p return MaxPressure and clear it.
  GCNRegPressure moveMaxPressure() {
    auto Res = MaxPressure;
    MaxPressure.clear();
    return Res;
  }

  /// Reset tracker to the point before the \p MI
  /// filling \p LiveRegs upon this point using LIS.
  /// \p returns false if block is empty except debug values.
  bool reset(const MachineInstr &MI, const LiveRegSet *LiveRegs = nullptr);

  /// Move to the state right before the next MI or after the end of MBB.
  /// \p returns false if reached end of the block.
  /// If \p UseInternalIterator is true, then internal iterators are used and
  /// set to process in program order. If \p UseInternalIterator is false, then
  /// it is assumed that the tracker is using an externally managed iterator,
  /// and advance* calls will not update the state of the iterator. In such
  /// cases, the tracker will move to the state right before the provided \p MI
  /// and use LIS for RP calculations.
  bool advanceBeforeNext(MachineInstr *MI = nullptr,
                         bool UseInternalIterator = true);

  /// Move to the state at the MI, advanceBeforeNext has to be called first.
  /// If \p UseInternalIterator is true, then internal iterators are used and
  /// set to process in program order. If \p UseInternalIterator is false, then
  /// it is assumed that the tracker is using an externally managed iterator,
  /// and advance* calls will not update the state of the iterator. In such
  /// cases, the tracker will move to the state at the provided \p MI .
  void advanceToNext(MachineInstr *MI = nullptr,
                     bool UseInternalIterator = true);

  /// Move to the state at the next MI. \p returns false if reached end of
  /// block. If \p UseInternalIterator is true, then internal iterators are used
  /// and set to process in program order. If \p UseInternalIterator is false,
  /// then it is assumed that the tracker is using an externally managed
  /// iterator, and advance* calls will not update the state of the iterator. In
  /// such cases, the tracker will move to the state right before the provided
  /// \p MI and use LIS for RP calculations.
  bool advance(MachineInstr *MI = nullptr, bool UseInternalIterator = true);

  /// Advance instructions until before \p End.
  bool advance(MachineBasicBlock::const_iterator End);

  /// Reset to \p Begin and advance to \p End.
  bool advance(MachineBasicBlock::const_iterator Begin,
               MachineBasicBlock::const_iterator End,
               const LiveRegSet *LiveRegsCopy = nullptr);

  /// Mostly copy/paste from CodeGen/RegisterPressure.cpp
  /// Calculate the impact \p MI will have on CurPressure and \return the
  /// speculated pressure. In order to support RP Speculation, this does not
  /// rely on the implicit program ordering in the LiveIntervals.
  GCNRegPressure bumpDownwardPressure(const MachineInstr *MI,
                                      const SIRegisterInfo *TRI) const;
};

/// \returns the LaneMask of live lanes of \p Reg at position \p SI. Only the
/// active lanes of \p LaneMaskFilter will be set in the return value. This is
/// used, for example, to limit the live lanes to a specific subreg when
/// calculating use masks.
LaneBitmask getLiveLaneMask(unsigned Reg, SlotIndex SI,
                            const LiveIntervals &LIS,
                            const MachineRegisterInfo &MRI,
                            LaneBitmask LaneMaskFilter = LaneBitmask::getAll());

LaneBitmask getLiveLaneMask(const LiveInterval &LI, SlotIndex SI,
                            const MachineRegisterInfo &MRI,
                            LaneBitmask LaneMaskFilter = LaneBitmask::getAll());

/// creates a map MachineInstr -> LiveRegSet
/// R - range of iterators on instructions
/// After - upon entry or exit of every instruction
/// Note: there is no entry in the map for instructions with empty live reg set
/// Complexity = O(NumVirtRegs * averageLiveRangeSegmentsPerReg * lg(R))
template <typename Range>
DenseMap<MachineInstr*, GCNRPTracker::LiveRegSet>
getLiveRegMap(Range &&R, bool After, LiveIntervals &LIS) {
  std::vector<SlotIndex> Indexes;
  Indexes.reserve(std::distance(R.begin(), R.end()));
  auto &SII = *LIS.getSlotIndexes();
  for (MachineInstr *I : R) {
    auto SI = SII.getInstructionIndex(*I);
    Indexes.push_back(After ? SI.getDeadSlot() : SI.getBaseIndex());
  }
  llvm::sort(Indexes);

  auto &MRI = (*R.begin())->getParent()->getParent()->getRegInfo();
  DenseMap<MachineInstr *, GCNRPTracker::LiveRegSet> LiveRegMap;
  SmallVector<SlotIndex, 32> LiveIdxs, SRLiveIdxs;
  for (unsigned I = 0, E = MRI.getNumVirtRegs(); I != E; ++I) {
    auto Reg = Register::index2VirtReg(I);
    if (!LIS.hasInterval(Reg))
      continue;
    auto &LI = LIS.getInterval(Reg);
    LiveIdxs.clear();
    if (!LI.findIndexesLiveAt(Indexes, std::back_inserter(LiveIdxs)))
      continue;
    if (!LI.hasSubRanges()) {
      for (auto SI : LiveIdxs)
        LiveRegMap[SII.getInstructionFromIndex(SI)][Reg] =
            MRI.getMaxLaneMaskForVReg(Reg);
    } else
      for (const auto &S : LI.subranges()) {
        // constrain search for subranges by indexes live at main range
        SRLiveIdxs.clear();
        S.findIndexesLiveAt(LiveIdxs, std::back_inserter(SRLiveIdxs));
        for (auto SI : SRLiveIdxs)
          LiveRegMap[SII.getInstructionFromIndex(SI)][Reg] |= S.LaneMask;
      }
  }
  return LiveRegMap;
}

inline GCNRPTracker::LiveRegSet getLiveRegsAfter(const MachineInstr &MI,
                                                 const LiveIntervals &LIS) {
  return getLiveRegs(LIS.getInstructionIndex(MI).getDeadSlot(), LIS,
                     MI.getParent()->getParent()->getRegInfo());
}

inline GCNRPTracker::LiveRegSet getLiveRegsBefore(const MachineInstr &MI,
                                                  const LiveIntervals &LIS) {
  return getLiveRegs(LIS.getInstructionIndex(MI).getBaseIndex(), LIS,
                     MI.getParent()->getParent()->getRegInfo());
}

template <typename Range>
GCNRegPressure getRegPressure(const MachineRegisterInfo &MRI,
                              Range &&LiveRegs) {
  GCNRegPressure Res;
  for (const auto &RM : LiveRegs)
    Res.inc(RM.first, LaneBitmask::getNone(), RM.second, MRI);
  return Res;
}

bool isEqual(const GCNRPTracker::LiveRegSet &S1,
             const GCNRPTracker::LiveRegSet &S2);

Printable print(const GCNRegPressure &RP, const GCNSubtarget *ST = nullptr,
                unsigned DynamicVGPRBlockSize = 0);

Printable print(const GCNRPTracker::LiveRegSet &LiveRegs,
                const MachineRegisterInfo &MRI);

Printable reportMismatch(const GCNRPTracker::LiveRegSet &LISLR,
                         const GCNRPTracker::LiveRegSet &TrackedL,
                         const TargetRegisterInfo *TRI, StringRef Pfx = "  ");

struct GCNRegPressurePrinter : public MachineFunctionPass {
  static char ID;

public:
  GCNRegPressurePrinter() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LiveIntervalsWrapperPass>();
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

LLVM_ABI void dumpMaxRegPressure(MachineFunction &MF,
                                 GCNRegPressure::RegKind Kind,
                                 LiveIntervals &LIS,
                                 const MachineLoopInfo *MLI);

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_GCNREGPRESSURE_H
