//===- RegisterPressure.h - Dynamic Register Pressure -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the RegisterPressure class which can be used to track
// MachineInstr level register pressure.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_REGISTERPRESSURE_H
#define LLVM_CODEGEN_REGISTERPRESSURE_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SparseSet.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/MC/LaneBitmask.h"
#include "llvm/Support/Compiler.h"
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <vector>

namespace llvm {

class LiveIntervals;
class MachineFunction;
class MachineInstr;
class MachineRegisterInfo;
class RegisterClassInfo;

struct VRegMaskOrUnit {
  VirtRegOrUnit VRegOrUnit;
  LaneBitmask LaneMask;

  VRegMaskOrUnit(VirtRegOrUnit VRegOrUnit, LaneBitmask LaneMask)
      : VRegOrUnit(VRegOrUnit), LaneMask(LaneMask) {}
};

/// Base class for register pressure results.
struct RegisterPressure {
  /// Map of max reg pressure indexed by pressure set ID, not class ID.
  std::vector<unsigned> MaxSetPressure;

  /// List of live in virtual registers or physical register units.
  SmallVector<VRegMaskOrUnit, 8> LiveInRegs;
  SmallVector<VRegMaskOrUnit, 8> LiveOutRegs;

  LLVM_ABI void dump(const TargetRegisterInfo *TRI) const;
};

/// RegisterPressure computed within a region of instructions delimited by
/// TopIdx and BottomIdx.  During pressure computation, the maximum pressure per
/// register pressure set is increased. Once pressure within a region is fully
/// computed, the live-in and live-out sets are recorded.
///
/// This is preferable to RegionPressure when LiveIntervals are available,
/// because delimiting regions by SlotIndex is more robust and convenient than
/// holding block iterators. The block contents can change without invalidating
/// the pressure result.
struct IntervalPressure : RegisterPressure {
  /// Record the boundary of the region being tracked.
  SlotIndex TopIdx;
  SlotIndex BottomIdx;

  LLVM_ABI void reset();

  LLVM_ABI void openTop(SlotIndex NextTop);

  LLVM_ABI void openBottom(SlotIndex PrevBottom);
};

/// RegisterPressure computed within a region of instructions delimited by
/// TopPos and BottomPos. This is a less precise version of IntervalPressure for
/// use when LiveIntervals are unavailable.
struct RegionPressure : RegisterPressure {
  /// Record the boundary of the region being tracked.
  MachineBasicBlock::const_iterator TopPos;
  MachineBasicBlock::const_iterator BottomPos;

  LLVM_ABI void reset();

  LLVM_ABI void openTop(MachineBasicBlock::const_iterator PrevTop);

  LLVM_ABI void openBottom(MachineBasicBlock::const_iterator PrevBottom);
};

/// Capture a change in pressure for a single pressure set. UnitInc may be
/// expressed in terms of upward or downward pressure depending on the client
/// and will be dynamically adjusted for current liveness.
///
/// Pressure increments are tiny, typically 1-2 units, and this is only for
/// heuristics, so we don't check UnitInc overflow. Instead, we may have a
/// higher level assert that pressure is consistent within a region. We also
/// effectively ignore dead defs which don't affect heuristics much.
class PressureChange {
  uint16_t PSetID = 0; // ID+1. 0=Invalid.
  int16_t UnitInc = 0;

public:
  PressureChange() = default;
  PressureChange(unsigned id): PSetID(id + 1) {
    assert(id < std::numeric_limits<uint16_t>::max() && "PSetID overflow.");
  }

  bool isValid() const { return PSetID > 0; }

  unsigned getPSet() const {
    assert(isValid() && "invalid PressureChange");
    return PSetID - 1;
  }

  // If PSetID is invalid, return UINT16_MAX to give it lowest priority.
  unsigned getPSetOrMax() const {
    return (PSetID - 1) & std::numeric_limits<uint16_t>::max();
  }

  int getUnitInc() const { return UnitInc; }

  void setUnitInc(int Inc) { UnitInc = Inc; }

  bool operator==(const PressureChange &RHS) const {
    return PSetID == RHS.PSetID && UnitInc == RHS.UnitInc;
  }

  LLVM_ABI void dump() const;
};

/// List of PressureChanges in order of increasing, unique PSetID.
///
/// Use a small fixed number, because we can fit more PressureChanges in an
/// empty SmallVector than ever need to be tracked per register class. If more
/// PSets are affected, then we only track the most constrained.
class PressureDiff {
  // The initial design was for MaxPSets=4, but that requires PSet partitions,
  // which are not yet implemented. (PSet partitions are equivalent PSets given
  // the register classes actually in use within the scheduling region.)
  enum { MaxPSets = 16 };

  PressureChange PressureChanges[MaxPSets];

  using iterator = PressureChange *;

  iterator nonconst_begin() { return &PressureChanges[0]; }
  iterator nonconst_end() { return &PressureChanges[MaxPSets]; }

public:
  using const_iterator = const PressureChange *;

  const_iterator begin() const { return &PressureChanges[0]; }
  const_iterator end() const { return &PressureChanges[MaxPSets]; }

  LLVM_ABI void addPressureChange(VirtRegOrUnit VRegOrUnit, bool IsDec,
                                  const MachineRegisterInfo *MRI);

  LLVM_ABI void dump(const TargetRegisterInfo &TRI) const;
};

/// List of registers defined and used by a machine instruction.
class RegisterOperands {
public:
  /// List of virtual registers and register units read by the instruction.
  SmallVector<VRegMaskOrUnit, 8> Uses;
  /// List of virtual registers and register units defined by the
  /// instruction which are not dead.
  SmallVector<VRegMaskOrUnit, 8> Defs;
  /// List of virtual registers and register units defined by the
  /// instruction but dead.
  SmallVector<VRegMaskOrUnit, 8> DeadDefs;

  /// Analyze the given instruction \p MI and fill in the Uses, Defs and
  /// DeadDefs list based on the MachineOperand flags.
  LLVM_ABI void collect(const MachineInstr &MI, const TargetRegisterInfo &TRI,
                        const MachineRegisterInfo &MRI, bool TrackLaneMasks,
                        bool IgnoreDead);

  /// Use liveness information to find dead defs not marked with a dead flag
  /// and move them to the DeadDefs vector.
  LLVM_ABI void detectDeadDefs(const MachineInstr &MI,
                               const LiveIntervals &LIS);

  /// Use liveness information to find out which uses/defs are partially
  /// undefined/dead and adjust the VRegMaskOrUnits accordingly.
  /// If \p AddFlagsMI is given then missing read-undef and dead flags will be
  /// added to the instruction.
  LLVM_ABI void adjustLaneLiveness(const LiveIntervals &LIS,
                                   const MachineRegisterInfo &MRI,
                                   SlotIndex Pos,
                                   MachineInstr *AddFlagsMI = nullptr);
};

/// Array of PressureDiffs.
class PressureDiffs {
  PressureDiff *PDiffArray = nullptr;
  unsigned Size = 0;
  unsigned Max = 0;

public:
  PressureDiffs() = default;
  PressureDiffs &operator=(const PressureDiffs &other) = delete;
  PressureDiffs(const PressureDiffs &other) = delete;
  ~PressureDiffs() { free(PDiffArray); }

  void clear() { Size = 0; }

  LLVM_ABI void init(unsigned N);

  PressureDiff &operator[](unsigned Idx) {
    assert(Idx < Size && "PressureDiff index out of bounds");
    return PDiffArray[Idx];
  }
  const PressureDiff &operator[](unsigned Idx) const {
    return const_cast<PressureDiffs*>(this)->operator[](Idx);
  }

  /// Record pressure difference induced by the given operand list to
  /// node with index \p Idx.
  LLVM_ABI void addInstruction(unsigned Idx, const RegisterOperands &RegOpers,
                               const MachineRegisterInfo &MRI);
};

/// Store the effects of a change in pressure on things that MI scheduler cares
/// about.
///
/// Excess records the value of the largest difference in register units beyond
/// the target's pressure limits across the affected pressure sets, where
/// largest is defined as the absolute value of the difference. Negative
/// ExcessUnits indicates a reduction in pressure that had already exceeded the
/// target's limits.
///
/// CriticalMax records the largest increase in the tracker's max pressure that
/// exceeds the critical limit for some pressure set determined by the client.
///
/// CurrentMax records the largest increase in the tracker's max pressure that
/// exceeds the current limit for some pressure set determined by the client.
struct RegPressureDelta {
  PressureChange Excess;
  PressureChange CriticalMax;
  PressureChange CurrentMax;

  RegPressureDelta() = default;

  bool operator==(const RegPressureDelta &RHS) const {
    return Excess == RHS.Excess && CriticalMax == RHS.CriticalMax
      && CurrentMax == RHS.CurrentMax;
  }
  bool operator!=(const RegPressureDelta &RHS) const {
    return !operator==(RHS);
  }
  LLVM_ABI void dump() const;
};

/// A set of live virtual registers and physical register units.
///
/// This is a wrapper around a SparseSet which deals with mapping register unit
/// and virtual register indexes to an index usable by the sparse set.
class LiveRegSet {
private:
  struct IndexMaskPair {
    unsigned Index;
    LaneBitmask LaneMask;

    IndexMaskPair(unsigned Index, LaneBitmask LaneMask)
        : Index(Index), LaneMask(LaneMask) {}

    unsigned getSparseSetIndex() const {
      return Index;
    }
  };

  using RegSet = SparseSet<IndexMaskPair>;
  RegSet Regs;
  unsigned NumRegUnits = 0u;

  unsigned getSparseIndexFromVirtRegOrUnit(VirtRegOrUnit VRegOrUnit) const {
    if (VRegOrUnit.isVirtualReg())
      return VRegOrUnit.asVirtualReg().virtRegIndex() + NumRegUnits;
    assert(static_cast<unsigned>(VRegOrUnit.asMCRegUnit()) < NumRegUnits);
    return static_cast<unsigned>(VRegOrUnit.asMCRegUnit());
  }

  VirtRegOrUnit getVirtRegOrUnitFromSparseIndex(unsigned SparseIndex) const {
    if (SparseIndex >= NumRegUnits)
      return VirtRegOrUnit(Register::index2VirtReg(SparseIndex - NumRegUnits));
    return VirtRegOrUnit(static_cast<MCRegUnit>(SparseIndex));
  }

public:
  LLVM_ABI void clear();
  LLVM_ABI void init(const MachineRegisterInfo &MRI);

  LaneBitmask contains(VirtRegOrUnit VRegOrUnit) const {
    unsigned SparseIndex = getSparseIndexFromVirtRegOrUnit(VRegOrUnit);
    RegSet::const_iterator I = Regs.find(SparseIndex);
    if (I == Regs.end())
      return LaneBitmask::getNone();
    return I->LaneMask;
  }

  /// Mark the \p Pair.LaneMask lanes of \p Pair.Reg as live.
  /// Returns the previously live lanes of \p Pair.Reg.
  LaneBitmask insert(VRegMaskOrUnit Pair) {
    unsigned SparseIndex = getSparseIndexFromVirtRegOrUnit(Pair.VRegOrUnit);
    auto InsertRes = Regs.insert(IndexMaskPair(SparseIndex, Pair.LaneMask));
    if (!InsertRes.second) {
      LaneBitmask PrevMask = InsertRes.first->LaneMask;
      InsertRes.first->LaneMask |= Pair.LaneMask;
      return PrevMask;
    }
    return LaneBitmask::getNone();
  }

  /// Clears the \p Pair.LaneMask lanes of \p Pair.Reg (mark them as dead).
  /// Returns the previously live lanes of \p Pair.Reg.
  LaneBitmask erase(VRegMaskOrUnit Pair) {
    unsigned SparseIndex = getSparseIndexFromVirtRegOrUnit(Pair.VRegOrUnit);
    RegSet::iterator I = Regs.find(SparseIndex);
    if (I == Regs.end())
      return LaneBitmask::getNone();
    LaneBitmask PrevMask = I->LaneMask;
    I->LaneMask &= ~Pair.LaneMask;
    return PrevMask;
  }

  size_t size() const {
    return Regs.size();
  }

  void appendTo(SmallVectorImpl<VRegMaskOrUnit> &To) const {
    for (const IndexMaskPair &P : Regs) {
      VirtRegOrUnit VRegOrUnit = getVirtRegOrUnitFromSparseIndex(P.Index);
      if (P.LaneMask.any())
        To.emplace_back(VRegOrUnit, P.LaneMask);
    }
  }
};

/// Track the current register pressure at some position in the instruction
/// stream, and remember the high water mark within the region traversed. This
/// does not automatically consider live-through ranges. The client may
/// independently adjust for global liveness.
///
/// Each RegPressureTracker only works within a MachineBasicBlock. Pressure can
/// be tracked across a larger region by storing a RegisterPressure result at
/// each block boundary and explicitly adjusting pressure to account for block
/// live-in and live-out register sets.
///
/// RegPressureTracker holds a reference to a RegisterPressure result that it
/// computes incrementally. During downward tracking, P.BottomIdx or P.BottomPos
/// is invalid until it reaches the end of the block or closeRegion() is
/// explicitly called. Similarly, P.TopIdx is invalid during upward
/// tracking. Changing direction has the side effect of closing region, and
/// traversing past TopIdx or BottomIdx reopens it.
class RegPressureTracker {
  const MachineFunction *MF = nullptr;
  const TargetRegisterInfo *TRI = nullptr;
  const RegisterClassInfo *RCI = nullptr;
  const MachineRegisterInfo *MRI = nullptr;
  const LiveIntervals *LIS = nullptr;

  /// We currently only allow pressure tracking within a block.
  const MachineBasicBlock *MBB = nullptr;

  /// Track the max pressure within the region traversed so far.
  RegisterPressure &P;

  /// Run in two modes dependending on whether constructed with IntervalPressure
  /// or RegisterPressure. If requireIntervals is false, LIS are ignored.
  bool RequireIntervals;

  /// True if UntiedDefs will be populated.
  bool TrackUntiedDefs = false;

  /// True if lanemasks should be tracked.
  bool TrackLaneMasks = false;

  /// Register pressure corresponds to liveness before this instruction
  /// iterator. It may point to the end of the block or a DebugValue rather than
  /// an instruction.
  MachineBasicBlock::const_iterator CurrPos;

  /// Pressure map indexed by pressure set ID, not class ID.
  std::vector<unsigned> CurrSetPressure;

  /// Set of live registers.
  LiveRegSet LiveRegs;

  /// Set of vreg defs that start a live range.
  SparseSet<Register, Register, VirtReg2IndexFunctor> UntiedDefs;
  /// Live-through pressure.
  std::vector<unsigned> LiveThruPressure;

public:
  RegPressureTracker(IntervalPressure &rp) : P(rp), RequireIntervals(true) {}
  RegPressureTracker(RegionPressure &rp) : P(rp), RequireIntervals(false) {}

  LLVM_ABI void reset();

  LLVM_ABI void init(const MachineFunction *mf, const RegisterClassInfo *rci,
                     const LiveIntervals *lis, const MachineBasicBlock *mbb,
                     MachineBasicBlock::const_iterator pos, bool TrackLaneMasks,
                     bool TrackUntiedDefs);

  /// Force liveness of virtual registers or physical register
  /// units. Particularly useful to initialize the livein/out state of the
  /// tracker before the first call to advance/recede.
  LLVM_ABI void addLiveRegs(ArrayRef<VRegMaskOrUnit> Regs);

  /// Get the MI position corresponding to this register pressure.
  MachineBasicBlock::const_iterator getPos() const { return CurrPos; }

  // Reset the MI position corresponding to the register pressure. This allows
  // schedulers to move instructions above the RegPressureTracker's
  // CurrPos. Since the pressure is computed before CurrPos, the iterator
  // position changes while pressure does not.
  void setPos(MachineBasicBlock::const_iterator Pos) { CurrPos = Pos; }

  /// Recede across the previous instruction.
  LLVM_ABI void recede(SmallVectorImpl<VRegMaskOrUnit> *LiveUses = nullptr);

  /// Recede across the previous instruction.
  /// This "low-level" variant assumes that recedeSkipDebugValues() was
  /// called previously and takes precomputed RegisterOperands for the
  /// instruction.
  LLVM_ABI void recede(const RegisterOperands &RegOpers,
                       SmallVectorImpl<VRegMaskOrUnit> *LiveUses = nullptr);

  /// Recede until we find an instruction which is not a DebugValue.
  LLVM_ABI void recedeSkipDebugValues();

  /// Advance across the current instruction.
  LLVM_ABI void advance();

  /// Advance across the current instruction.
  /// This is a "low-level" variant of advance() which takes precomputed
  /// RegisterOperands of the instruction.
  LLVM_ABI void advance(const RegisterOperands &RegOpers);

  /// Finalize the region boundaries and recored live ins and live outs.
  LLVM_ABI void closeRegion();

  /// Initialize the LiveThru pressure set based on the untied defs found in
  /// RPTracker.
  LLVM_ABI void initLiveThru(const RegPressureTracker &RPTracker);

  /// Copy an existing live thru pressure result.
  void initLiveThru(ArrayRef<unsigned> PressureSet) {
    LiveThruPressure.assign(PressureSet.begin(), PressureSet.end());
  }

  ArrayRef<unsigned> getLiveThru() const { return LiveThruPressure; }

  /// Get the resulting register pressure over the traversed region.
  /// This result is complete if closeRegion() was explicitly invoked.
  RegisterPressure &getPressure() { return P; }
  const RegisterPressure &getPressure() const { return P; }

  /// Get the register set pressure at the current position, which may be less
  /// than the pressure across the traversed region.
  const std::vector<unsigned> &getRegSetPressureAtPos() const {
    return CurrSetPressure;
  }

  LLVM_ABI bool isTopClosed() const;
  LLVM_ABI bool isBottomClosed() const;

  LLVM_ABI void closeTop();
  LLVM_ABI void closeBottom();

  /// Consider the pressure increase caused by traversing this instruction
  /// bottom-up. Find the pressure set with the most change beyond its pressure
  /// limit based on the tracker's current pressure, and record the number of
  /// excess register units of that pressure set introduced by this instruction.
  LLVM_ABI void
  getMaxUpwardPressureDelta(const MachineInstr *MI, PressureDiff *PDiff,
                            RegPressureDelta &Delta,
                            ArrayRef<PressureChange> CriticalPSets,
                            ArrayRef<unsigned> MaxPressureLimit);

  LLVM_ABI void
  getUpwardPressureDelta(const MachineInstr *MI,
                         /*const*/ PressureDiff &PDiff, RegPressureDelta &Delta,
                         ArrayRef<PressureChange> CriticalPSets,
                         ArrayRef<unsigned> MaxPressureLimit) const;

  /// Consider the pressure increase caused by traversing this instruction
  /// top-down. Find the pressure set with the most change beyond its pressure
  /// limit based on the tracker's current pressure, and record the number of
  /// excess register units of that pressure set introduced by this instruction.
  LLVM_ABI void
  getMaxDownwardPressureDelta(const MachineInstr *MI, RegPressureDelta &Delta,
                              ArrayRef<PressureChange> CriticalPSets,
                              ArrayRef<unsigned> MaxPressureLimit);

  /// Find the pressure set with the most change beyond its pressure limit after
  /// traversing this instruction either upward or downward depending on the
  /// closed end of the current region.
  void getMaxPressureDelta(const MachineInstr *MI,
                           RegPressureDelta &Delta,
                           ArrayRef<PressureChange> CriticalPSets,
                           ArrayRef<unsigned> MaxPressureLimit) {
    if (isTopClosed())
      return getMaxDownwardPressureDelta(MI, Delta, CriticalPSets,
                                         MaxPressureLimit);

    assert(isBottomClosed() && "Uninitialized pressure tracker");
    return getMaxUpwardPressureDelta(MI, nullptr, Delta, CriticalPSets,
                                     MaxPressureLimit);
  }

  /// Get the pressure of each PSet after traversing this instruction bottom-up.
  LLVM_ABI void getUpwardPressure(const MachineInstr *MI,
                                  std::vector<unsigned> &PressureResult,
                                  std::vector<unsigned> &MaxPressureResult);

  /// Get the pressure of each PSet after traversing this instruction top-down.
  LLVM_ABI void getDownwardPressure(const MachineInstr *MI,
                                    std::vector<unsigned> &PressureResult,
                                    std::vector<unsigned> &MaxPressureResult);

  void getPressureAfterInst(const MachineInstr *MI,
                            std::vector<unsigned> &PressureResult,
                            std::vector<unsigned> &MaxPressureResult) {
    if (isTopClosed())
      return getUpwardPressure(MI, PressureResult, MaxPressureResult);

    assert(isBottomClosed() && "Uninitialized pressure tracker");
    return getDownwardPressure(MI, PressureResult, MaxPressureResult);
  }

  bool hasUntiedDef(Register VirtReg) const {
    return UntiedDefs.count(VirtReg);
  }

  LLVM_ABI void dump() const;

  LLVM_ABI void increaseRegPressure(VirtRegOrUnit VRegOrUnit,
                                    LaneBitmask PreviousMask,
                                    LaneBitmask NewMask);
  LLVM_ABI void decreaseRegPressure(VirtRegOrUnit VRegOrUnit,
                                    LaneBitmask PreviousMask,
                                    LaneBitmask NewMask);

protected:
  /// Add Reg to the live out set and increase max pressure.
  LLVM_ABI void discoverLiveOut(VRegMaskOrUnit Pair);
  /// Add Reg to the live in set and increase max pressure.
  LLVM_ABI void discoverLiveIn(VRegMaskOrUnit Pair);

  /// Get the SlotIndex for the first nondebug instruction including or
  /// after the current position.
  LLVM_ABI SlotIndex getCurrSlot() const;

  LLVM_ABI void bumpDeadDefs(ArrayRef<VRegMaskOrUnit> DeadDefs);

  LLVM_ABI void bumpUpwardPressure(const MachineInstr *MI);
  LLVM_ABI void bumpDownwardPressure(const MachineInstr *MI);

  LLVM_ABI void
  discoverLiveInOrOut(VRegMaskOrUnit Pair,
                      SmallVectorImpl<VRegMaskOrUnit> &LiveInOrOut);

  LLVM_ABI LaneBitmask getLastUsedLanes(VirtRegOrUnit VRegOrUnit,
                                        SlotIndex Pos) const;
  LLVM_ABI LaneBitmask getLiveLanesAt(VirtRegOrUnit VRegOrUnit,
                                      SlotIndex Pos) const;
  LLVM_ABI LaneBitmask getLiveThroughAt(VirtRegOrUnit VRegOrUnit,
                                        SlotIndex Pos) const;
};

LLVM_ABI void dumpRegSetPressure(ArrayRef<unsigned> SetPressure,
                                 const TargetRegisterInfo *TRI);

} // end namespace llvm

#endif // LLVM_CODEGEN_REGISTERPRESSURE_H
