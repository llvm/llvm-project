//===---------------------- GCNRegPressure.h -*- C++ -*--------------------===//
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

#ifndef LLVM_LIB_TARGET_AMDGPU_GCNREGPRESSURE_H
#define LLVM_LIB_TARGET_AMDGPU_GCNREGPRESSURE_H

#include "AMDGPUSubtarget.h"

#include <limits>

namespace llvm {

struct GCNRegPressure {
  enum RegKind {
    SGPR32,
    SGPR_TUPLE,
    VGPR32,
    VGPR_TUPLE,
    TOTAL_KINDS
  };

  GCNRegPressure() {
    clear();
  }

  bool empty() const { return getSGPRNum() == 0 && getVGPRNum() == 0; }

  void clear() { std::fill(&Value[0], &Value[TOTAL_KINDS], 0); }

  unsigned getSGPRNum() const { return Value[SGPR32]; }
  unsigned getVGPRNum() const { return Value[VGPR32]; }

  unsigned getVGPRTuplesWeight() const { return Value[VGPR_TUPLE]; }
  unsigned getSGPRTuplesWeight() const { return Value[SGPR_TUPLE]; }

  unsigned getOccupancy(const SISubtarget &ST) const {
    return std::min(ST.getOccupancyWithNumSGPRs(getSGPRNum()),
                    ST.getOccupancyWithNumVGPRs(getVGPRNum()));
  }

  void inc(unsigned Reg,
           LaneBitmask PrevMask,
           LaneBitmask NewMask,
           const MachineRegisterInfo &MRI);

  bool higherOccupancy(const SISubtarget &ST, const GCNRegPressure& O) const {
    return getOccupancy(ST) > O.getOccupancy(ST);
  }

  bool less(const SISubtarget &ST, const GCNRegPressure& O,
    unsigned MaxOccupancy = std::numeric_limits<unsigned>::max()) const;

  bool operator==(const GCNRegPressure &O) const {
    return std::equal(&Value[0], &Value[TOTAL_KINDS], O.Value);
  }

  bool operator!=(const GCNRegPressure &O) const {
    return !(*this == O);
  }

  void print(raw_ostream &OS, const SISubtarget *ST=nullptr) const;
  void dump() const { print(dbgs()); }

private:
  unsigned Value[TOTAL_KINDS];

  static unsigned getRegKind(unsigned Reg, const MachineRegisterInfo &MRI);

  friend GCNRegPressure max(const GCNRegPressure &P1,
                            const GCNRegPressure &P2);
};

inline GCNRegPressure max(const GCNRegPressure &P1, const GCNRegPressure &P2) {
  GCNRegPressure Res;
  for (unsigned I = 0; I < GCNRegPressure::TOTAL_KINDS; ++I)
    Res.Value[I] = std::max(P1.Value[I], P2.Value[I]);
  return Res;
}

class GCNRPTracker {
public:
  typedef DenseMap<unsigned, LaneBitmask> LiveRegSet;

protected:
  LiveRegSet LiveRegs;
  GCNRegPressure CurPressure, MaxPressure;
  const MachineInstr *LastTrackedMI = nullptr;
  mutable const MachineRegisterInfo *MRI = nullptr;
  GCNRPTracker() {}
public:
  // live regs for the current state
  const decltype(LiveRegs) &getLiveRegs() const { return LiveRegs; }
  const MachineInstr *getLastTrackedMI() const { return LastTrackedMI; }

  // returns MaxPressure, resetting it
  decltype(MaxPressure) moveMaxPressure() {
    auto Res = MaxPressure;
    MaxPressure.clear();
    return Res;
  }
  decltype(LiveRegs) moveLiveRegs() {
    return std::move(LiveRegs);
  }
};

class GCNUpwardRPTracker : public GCNRPTracker {
  const LiveIntervals &LIS;
  LaneBitmask getDefRegMask(const MachineOperand &MO) const;
  LaneBitmask getUsedRegMask(const MachineOperand &MO) const;
public:
  GCNUpwardRPTracker(const LiveIntervals &LIS_) : LIS(LIS_) {}
  // reset tracker to the point just below MI
  // filling live regs upon this point using LIS
  void reset(const MachineInstr &MI);

  // move to the state just above the MI
  void recede(const MachineInstr &MI);

  // checks whether the tracker's state after receding MI corresponds
  // to reported by LIS
  bool isValid() const;
};

LaneBitmask getLiveLaneMask(unsigned Reg,
                            SlotIndex SI,
                            const LiveIntervals &LIS,
                            const MachineRegisterInfo &MRI);

GCNRPTracker::LiveRegSet getLiveRegs(SlotIndex SI,
                                     const LiveIntervals &LIS,
                                     const MachineRegisterInfo &MRI);

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

void printLivesAt(SlotIndex SI,
                  const LiveIntervals &LIS,
                  const MachineRegisterInfo &MRI);

} // End namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_GCNREGPRESSURE_H
