//===-- RegisterPressure.h - Dynamic Register Pressure -*- C++ -*-------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the RegisterPressure class which can be used to track
// MachineInstr level register pressure.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_REGISTERPRESSURE_H
#define LLVM_CODEGEN_REGISTERPRESSURE_H

#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/ADT/SparseSet.h"

namespace llvm {

class LiveIntervals;
class RegisterClassInfo;
class MachineInstr;

/// Base class for register pressure results.
struct RegisterPressure {
  /// Map of max reg pressure indexed by pressure set ID, not class ID.
  std::vector<unsigned> MaxSetPressure;

  /// List of live in registers.
  SmallVector<unsigned,8> LiveInRegs;
  SmallVector<unsigned,8> LiveOutRegs;

  /// Increase register pressure for each pressure set impacted by this register
  /// class. Normally called by RegPressureTracker, but may be called manually
  /// to account for live through (global liveness).
  void increase(const TargetRegisterClass *RC, const TargetRegisterInfo *TRI);

  /// Decrease register pressure for each pressure set impacted by this register
  /// class. This is only useful to account for spilling or rematerialization.
  void decrease(const TargetRegisterClass *RC, const TargetRegisterInfo *TRI);

  void dump(const TargetRegisterInfo *TRI) const;
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

  void reset();

  void openTop(SlotIndex NextTop);

  void openBottom(SlotIndex PrevBottom);
};

/// RegisterPressure computed within a region of instructions delimited by
/// TopPos and BottomPos. This is a less precise version of IntervalPressure for
/// use when LiveIntervals are unavailable.
struct RegionPressure : RegisterPressure {
  /// Record the boundary of the region being tracked.
  MachineBasicBlock::const_iterator TopPos;
  MachineBasicBlock::const_iterator BottomPos;

  void reset();

  void openTop(MachineBasicBlock::const_iterator PrevTop);

  void openBottom(MachineBasicBlock::const_iterator PrevBottom);
};

/// An element of pressure difference that identifies the pressure set and
/// amount of increase or decrease in units of pressure.
struct PressureElement {
  unsigned PSetID;
  int UnitIncrease;

  PressureElement(): PSetID(~0U), UnitIncrease(0) {}
  PressureElement(unsigned id, int inc): PSetID(id), UnitIncrease(inc) {}

  bool isValid() const { return PSetID != ~0U; }
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
  PressureElement Excess;
  PressureElement CriticalMax;
  PressureElement CurrentMax;

  RegPressureDelta() {}
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
  const MachineFunction     *MF;
  const TargetRegisterInfo  *TRI;
  const RegisterClassInfo   *RCI;
  const MachineRegisterInfo *MRI;
  const LiveIntervals       *LIS;

  /// We currently only allow pressure tracking within a block.
  const MachineBasicBlock *MBB;

  /// Track the max pressure within the region traversed so far.
  RegisterPressure &P;

  /// Run in two modes dependending on whether constructed with IntervalPressure
  /// or RegisterPressure. If requireIntervals is false, LIS are ignored.
  bool RequireIntervals;

  /// Register pressure corresponds to liveness before this instruction
  /// iterator. It may point to the end of the block rather than an instruction.
  MachineBasicBlock::const_iterator CurrPos;

  /// Pressure map indexed by pressure set ID, not class ID.
  std::vector<unsigned> CurrSetPressure;

  /// List of live registers.
  SparseSet<unsigned> LivePhysRegs;
  SparseSet<unsigned, VirtReg2IndexFunctor> LiveVirtRegs;

public:
  RegPressureTracker(IntervalPressure &rp) :
    MF(0), TRI(0), RCI(0), LIS(0), MBB(0), P(rp), RequireIntervals(true) {}

  RegPressureTracker(RegionPressure &rp) :
    MF(0), TRI(0), RCI(0), LIS(0), MBB(0), P(rp), RequireIntervals(false) {}

  void init(const MachineFunction *mf, const RegisterClassInfo *rci,
            const LiveIntervals *lis, const MachineBasicBlock *mbb,
            MachineBasicBlock::const_iterator pos);

  /// Force liveness of registers. Particularly useful to initialize the
  /// livein/out state of the tracker before the first call to advance/recede.
  void addLiveRegs(ArrayRef<unsigned> Regs);

  /// Get the MI position corresponding to this register pressure.
  MachineBasicBlock::const_iterator getPos() const { return CurrPos; }

  // Reset the MI position corresponding to the register pressure. This allows
  // schedulers to move instructions above the RegPressureTracker's
  // CurrPos. Since the pressure is computed before CurrPos, the iterator
  // position changes while pressure does not.
  void setPos(MachineBasicBlock::const_iterator Pos) { CurrPos = Pos; }

  /// Recede across the previous instruction.
  bool recede();

  /// Advance across the current instruction.
  bool advance();

  /// Finalize the region boundaries and recored live ins and live outs.
  void closeRegion();

  /// Get the resulting register pressure over the traversed region.
  /// This result is complete if either advance() or recede() has returned true,
  /// or if closeRegion() was explicitly invoked.
  RegisterPressure &getPressure() { return P; }
  const RegisterPressure &getPressure() const { return P; }

  /// Get the register set pressure at the current position, which may be less
  /// than the pressure across the traversed region.
  std::vector<unsigned> &getRegSetPressureAtPos() { return CurrSetPressure; }

  void discoverPhysLiveIn(unsigned Reg);
  void discoverPhysLiveOut(unsigned Reg);

  void discoverVirtLiveIn(unsigned Reg);
  void discoverVirtLiveOut(unsigned Reg);

  bool isTopClosed() const;
  bool isBottomClosed() const;

  void closeTop();
  void closeBottom();

  /// Consider the pressure increase caused by traversing this instruction
  /// bottom-up. Find the pressure set with the most change beyond its pressure
  /// limit based on the tracker's current pressure, and record the number of
  /// excess register units of that pressure set introduced by this instruction.
  void getMaxUpwardPressureDelta(const MachineInstr *MI,
                                 RegPressureDelta &Delta,
                                 ArrayRef<PressureElement> CriticalPSets,
                                 ArrayRef<unsigned> MaxPressureLimit);

  /// Consider the pressure increase caused by traversing this instruction
  /// top-down. Find the pressure set with the most change beyond its pressure
  /// limit based on the tracker's current pressure, and record the number of
  /// excess register units of that pressure set introduced by this instruction.
  void getMaxDownwardPressureDelta(const MachineInstr *MI,
                                   RegPressureDelta &Delta,
                                   ArrayRef<PressureElement> CriticalPSets,
                                   ArrayRef<unsigned> MaxPressureLimit);

  /// Find the pressure set with the most change beyond its pressure limit after
  /// traversing this instruction either upward or downward depending on the
  /// closed end of the current region.
  void getMaxPressureDelta(const MachineInstr *MI, RegPressureDelta &Delta,
                           ArrayRef<PressureElement> CriticalPSets,
                           ArrayRef<unsigned> MaxPressureLimit) {
    if (isTopClosed())
      return getMaxDownwardPressureDelta(MI, Delta, CriticalPSets,
                                         MaxPressureLimit);

    assert(isBottomClosed() && "Uninitialized pressure tracker");
    return getMaxUpwardPressureDelta(MI, Delta, CriticalPSets,
                                     MaxPressureLimit);
  }

  /// Get the pressure of each PSet after traversing this instruction bottom-up.
  void getUpwardPressure(const MachineInstr *MI,
                         std::vector<unsigned> &PressureResult,
                         std::vector<unsigned> &MaxPressureResult);

  /// Get the pressure of each PSet after traversing this instruction top-down.
  void getDownwardPressure(const MachineInstr *MI,
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

protected:
  void increasePhysRegPressure(ArrayRef<unsigned> Regs);
  void decreasePhysRegPressure(ArrayRef<unsigned> Regs);

  void increaseVirtRegPressure(ArrayRef<unsigned> Regs);
  void decreaseVirtRegPressure(ArrayRef<unsigned> Regs);

  void bumpUpwardPressure(const MachineInstr *MI);
  void bumpDownwardPressure(const MachineInstr *MI);
};
} // end namespace llvm

#endif
