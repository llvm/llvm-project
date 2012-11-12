//===-- RegisterPressure.cpp - Dynamic Register Pressure ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the RegisterPressure class which can be used to track
// MachineInstr level register pressure.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/LiveInterval.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterClassInfo.h"
#include "llvm/CodeGen/RegisterPressure.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

/// Increase register pressure for each set impacted by this register class.
static void increaseSetPressure(std::vector<unsigned> &CurrSetPressure,
                                std::vector<unsigned> &MaxSetPressure,
                                const TargetRegisterClass *RC,
                                const TargetRegisterInfo *TRI) {
  unsigned Weight = TRI->getRegClassWeight(RC).RegWeight;
  for (const int *PSet = TRI->getRegClassPressureSets(RC);
       *PSet != -1; ++PSet) {
    CurrSetPressure[*PSet] += Weight;
    if (&CurrSetPressure != &MaxSetPressure
        && CurrSetPressure[*PSet] > MaxSetPressure[*PSet]) {
      MaxSetPressure[*PSet] = CurrSetPressure[*PSet];
    }
  }
}

/// Decrease register pressure for each set impacted by this register class.
static void decreaseSetPressure(std::vector<unsigned> &CurrSetPressure,
                                const TargetRegisterClass *RC,
                                const TargetRegisterInfo *TRI) {
  unsigned Weight = TRI->getRegClassWeight(RC).RegWeight;
  for (const int *PSet = TRI->getRegClassPressureSets(RC);
       *PSet != -1; ++PSet) {
    assert(CurrSetPressure[*PSet] >= Weight && "register pressure underflow");
    CurrSetPressure[*PSet] -= Weight;
  }
}

/// Directly increase pressure only within this RegisterPressure result.
void RegisterPressure::increase(const TargetRegisterClass *RC,
                                const TargetRegisterInfo *TRI) {
  increaseSetPressure(MaxSetPressure, MaxSetPressure, RC, TRI);
}

/// Directly decrease pressure only within this RegisterPressure result.
void RegisterPressure::decrease(const TargetRegisterClass *RC,
                                const TargetRegisterInfo *TRI) {
  decreaseSetPressure(MaxSetPressure, RC, TRI);
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void RegisterPressure::dump(const TargetRegisterInfo *TRI) const {
  dbgs() << "Live In: ";
  for (unsigned i = 0, e = LiveInRegs.size(); i < e; ++i)
    dbgs() << PrintReg(LiveInRegs[i], TRI) << " ";
  dbgs() << '\n';
  dbgs() << "Live Out: ";
  for (unsigned i = 0, e = LiveOutRegs.size(); i < e; ++i)
    dbgs() << PrintReg(LiveOutRegs[i], TRI) << " ";
  dbgs() << '\n';
  for (unsigned i = 0, e = MaxSetPressure.size(); i < e; ++i) {
    if (MaxSetPressure[i] != 0)
      dbgs() << TRI->getRegPressureSetName(i) << "=" << MaxSetPressure[i]
             << '\n';
  }
}
#endif

/// Increase the current pressure as impacted by these physical registers and
/// bump the high water mark if needed.
void RegPressureTracker::increasePhysRegPressure(ArrayRef<unsigned> Regs) {
  for (unsigned I = 0, E = Regs.size(); I != E; ++I)
    increaseSetPressure(CurrSetPressure, P.MaxSetPressure,
                        TRI->getMinimalPhysRegClass(Regs[I]), TRI);
}

/// Simply decrease the current pressure as impacted by these physcial
/// registers.
void RegPressureTracker::decreasePhysRegPressure(ArrayRef<unsigned> Regs) {
  for (unsigned I = 0, E = Regs.size(); I != E; ++I)
    decreaseSetPressure(CurrSetPressure, TRI->getMinimalPhysRegClass(Regs[I]),
                        TRI);
}

/// Increase the current pressure as impacted by these virtual registers and
/// bump the high water mark if needed.
void RegPressureTracker::increaseVirtRegPressure(ArrayRef<unsigned> Regs) {
  for (unsigned I = 0, E = Regs.size(); I != E; ++I)
    increaseSetPressure(CurrSetPressure, P.MaxSetPressure,
                        MRI->getRegClass(Regs[I]), TRI);
}

/// Simply decrease the current pressure as impacted by these virtual registers.
void RegPressureTracker::decreaseVirtRegPressure(ArrayRef<unsigned> Regs) {
  for (unsigned I = 0, E = Regs.size(); I != E; ++I)
    decreaseSetPressure(CurrSetPressure, MRI->getRegClass(Regs[I]), TRI);
}

/// Clear the result so it can be used for another round of pressure tracking.
void IntervalPressure::reset() {
  TopIdx = BottomIdx = SlotIndex();
  MaxSetPressure.clear();
  LiveInRegs.clear();
  LiveOutRegs.clear();
}

/// Clear the result so it can be used for another round of pressure tracking.
void RegionPressure::reset() {
  TopPos = BottomPos = MachineBasicBlock::const_iterator();
  MaxSetPressure.clear();
  LiveInRegs.clear();
  LiveOutRegs.clear();
}

/// If the current top is not less than or equal to the next index, open it.
/// We happen to need the SlotIndex for the next top for pressure update.
void IntervalPressure::openTop(SlotIndex NextTop) {
  if (TopIdx <= NextTop)
    return;
  TopIdx = SlotIndex();
  LiveInRegs.clear();
}

/// If the current top is the previous instruction (before receding), open it.
void RegionPressure::openTop(MachineBasicBlock::const_iterator PrevTop) {
  if (TopPos != PrevTop)
    return;
  TopPos = MachineBasicBlock::const_iterator();
  LiveInRegs.clear();
}

/// If the current bottom is not greater than the previous index, open it.
void IntervalPressure::openBottom(SlotIndex PrevBottom) {
  if (BottomIdx > PrevBottom)
    return;
  BottomIdx = SlotIndex();
  LiveInRegs.clear();
}

/// If the current bottom is the previous instr (before advancing), open it.
void RegionPressure::openBottom(MachineBasicBlock::const_iterator PrevBottom) {
  if (BottomPos != PrevBottom)
    return;
  BottomPos = MachineBasicBlock::const_iterator();
  LiveInRegs.clear();
}

/// Setup the RegPressureTracker.
///
/// TODO: Add support for pressure without LiveIntervals.
void RegPressureTracker::init(const MachineFunction *mf,
                              const RegisterClassInfo *rci,
                              const LiveIntervals *lis,
                              const MachineBasicBlock *mbb,
                              MachineBasicBlock::const_iterator pos)
{
  MF = mf;
  TRI = MF->getTarget().getRegisterInfo();
  RCI = rci;
  MRI = &MF->getRegInfo();
  MBB = mbb;

  if (RequireIntervals) {
    assert(lis && "IntervalPressure requires LiveIntervals");
    LIS = lis;
  }

  CurrPos = pos;
  while (CurrPos != MBB->end() && CurrPos->isDebugValue())
    ++CurrPos;

  CurrSetPressure.assign(TRI->getNumRegPressureSets(), 0);

  if (RequireIntervals)
    static_cast<IntervalPressure&>(P).reset();
  else
    static_cast<RegionPressure&>(P).reset();
  P.MaxSetPressure = CurrSetPressure;

  LivePhysRegs.clear();
  LivePhysRegs.setUniverse(TRI->getNumRegs());
  LiveVirtRegs.clear();
  LiveVirtRegs.setUniverse(MRI->getNumVirtRegs());
}

/// Does this pressure result have a valid top position and live ins.
bool RegPressureTracker::isTopClosed() const {
  if (RequireIntervals)
    return static_cast<IntervalPressure&>(P).TopIdx.isValid();
  return (static_cast<RegionPressure&>(P).TopPos ==
          MachineBasicBlock::const_iterator());
}

/// Does this pressure result have a valid bottom position and live outs.
bool RegPressureTracker::isBottomClosed() const {
  if (RequireIntervals)
    return static_cast<IntervalPressure&>(P).BottomIdx.isValid();
  return (static_cast<RegionPressure&>(P).BottomPos ==
          MachineBasicBlock::const_iterator());
}

/// Set the boundary for the top of the region and summarize live ins.
void RegPressureTracker::closeTop() {
  if (RequireIntervals)
    static_cast<IntervalPressure&>(P).TopIdx =
      LIS->getInstructionIndex(CurrPos).getRegSlot();
  else
    static_cast<RegionPressure&>(P).TopPos = CurrPos;

  assert(P.LiveInRegs.empty() && "inconsistent max pressure result");
  P.LiveInRegs.reserve(LivePhysRegs.size() + LiveVirtRegs.size());
  P.LiveInRegs.append(LivePhysRegs.begin(), LivePhysRegs.end());
  for (SparseSet<unsigned>::const_iterator I =
         LiveVirtRegs.begin(), E = LiveVirtRegs.end(); I != E; ++I)
    P.LiveInRegs.push_back(*I);
  std::sort(P.LiveInRegs.begin(), P.LiveInRegs.end());
  P.LiveInRegs.erase(std::unique(P.LiveInRegs.begin(), P.LiveInRegs.end()),
                     P.LiveInRegs.end());
}

/// Set the boundary for the bottom of the region and summarize live outs.
void RegPressureTracker::closeBottom() {
  if (RequireIntervals)
    if (CurrPos == MBB->end())
      static_cast<IntervalPressure&>(P).BottomIdx = LIS->getMBBEndIdx(MBB);
    else
      static_cast<IntervalPressure&>(P).BottomIdx =
        LIS->getInstructionIndex(CurrPos).getRegSlot();
  else
    static_cast<RegionPressure&>(P).BottomPos = CurrPos;

  assert(P.LiveOutRegs.empty() && "inconsistent max pressure result");
  P.LiveOutRegs.reserve(LivePhysRegs.size() + LiveVirtRegs.size());
  P.LiveOutRegs.append(LivePhysRegs.begin(), LivePhysRegs.end());
  for (SparseSet<unsigned>::const_iterator I =
         LiveVirtRegs.begin(), E = LiveVirtRegs.end(); I != E; ++I)
    P.LiveOutRegs.push_back(*I);
  std::sort(P.LiveOutRegs.begin(), P.LiveOutRegs.end());
  P.LiveOutRegs.erase(std::unique(P.LiveOutRegs.begin(), P.LiveOutRegs.end()),
                      P.LiveOutRegs.end());
}

/// Finalize the region boundaries and record live ins and live outs.
void RegPressureTracker::closeRegion() {
  if (!isTopClosed() && !isBottomClosed()) {
    assert(LivePhysRegs.empty() && LiveVirtRegs.empty() &&
           "no region boundary");
    return;
  }
  if (!isBottomClosed())
    closeBottom();
  else if (!isTopClosed())
    closeTop();
  // If both top and bottom are closed, do nothing.
}

/// Return true if Reg aliases a register in Regs SparseSet.
static bool hasRegAlias(unsigned Reg, SparseSet<unsigned> &Regs,
                        const TargetRegisterInfo *TRI) {
  assert(!TargetRegisterInfo::isVirtualRegister(Reg) && "only for physregs");
  for (MCRegAliasIterator AI(Reg, TRI, true); AI.isValid(); ++AI)
    if (Regs.count(*AI))
      return true;
  return false;
}

/// Return true if Reg aliases a register in unsorted Regs SmallVector.
/// This is only valid for physical registers.
static SmallVectorImpl<unsigned>::iterator
findRegAlias(unsigned Reg, SmallVectorImpl<unsigned> &Regs,
             const TargetRegisterInfo *TRI) {
  for (MCRegAliasIterator AI(Reg, TRI, true); AI.isValid(); ++AI) {
    SmallVectorImpl<unsigned>::iterator I =
      std::find(Regs.begin(), Regs.end(), *AI);
    if (I != Regs.end())
      return I;
  }
  return Regs.end();
}

/// Return true if Reg can be inserted into Regs SmallVector. For virtual
/// register, do a linear search. For physical registers check for aliases.
static SmallVectorImpl<unsigned>::iterator
findReg(unsigned Reg, bool isVReg, SmallVectorImpl<unsigned> &Regs,
        const TargetRegisterInfo *TRI) {
  if(isVReg)
    return std::find(Regs.begin(), Regs.end(), Reg);
  return findRegAlias(Reg, Regs, TRI);
}

/// Collect this instruction's unique uses and defs into SmallVectors for
/// processing defs and uses in order.
template<bool isVReg>
struct RegisterOperands {
  SmallVector<unsigned, 8> Uses;
  SmallVector<unsigned, 8> Defs;
  SmallVector<unsigned, 8> DeadDefs;

  /// Push this operand's register onto the correct vector.
  void collect(const MachineOperand &MO, const TargetRegisterInfo *TRI) {
    if (MO.readsReg()) {
      if (findReg(MO.getReg(), isVReg, Uses, TRI) == Uses.end())
      Uses.push_back(MO.getReg());
    }
    if (MO.isDef()) {
      if (MO.isDead()) {
        if (findReg(MO.getReg(), isVReg, DeadDefs, TRI) == DeadDefs.end())
          DeadDefs.push_back(MO.getReg());
      }
      else if (findReg(MO.getReg(), isVReg, Defs, TRI) == Defs.end())
        Defs.push_back(MO.getReg());
    }
  }
};
typedef RegisterOperands<false> PhysRegOperands;
typedef RegisterOperands<true> VirtRegOperands;

/// Collect physical and virtual register operands.
static void collectOperands(const MachineInstr *MI,
                            PhysRegOperands &PhysRegOpers,
                            VirtRegOperands &VirtRegOpers,
                            const TargetRegisterInfo *TRI,
                            const MachineRegisterInfo *MRI) {
  for(ConstMIBundleOperands OperI(MI); OperI.isValid(); ++OperI) {
    const MachineOperand &MO = *OperI;
    if (!MO.isReg() || !MO.getReg())
      continue;

    if (TargetRegisterInfo::isVirtualRegister(MO.getReg()))
      VirtRegOpers.collect(MO, TRI);
    else if (MRI->isAllocatable(MO.getReg()))
      PhysRegOpers.collect(MO, TRI);
  }
  // Remove redundant physreg dead defs.
  for (unsigned i = PhysRegOpers.DeadDefs.size(); i > 0; --i) {
    unsigned Reg = PhysRegOpers.DeadDefs[i-1];
    if (findRegAlias(Reg, PhysRegOpers.Defs, TRI) != PhysRegOpers.Defs.end())
      PhysRegOpers.DeadDefs.erase(&PhysRegOpers.DeadDefs[i-1]);
  }
}

/// Force liveness of registers.
void RegPressureTracker::addLiveRegs(ArrayRef<unsigned> Regs) {
  for (unsigned i = 0, e = Regs.size(); i != e; ++i) {
    if (TargetRegisterInfo::isVirtualRegister(Regs[i])) {
      if (LiveVirtRegs.insert(Regs[i]).second)
        increaseVirtRegPressure(Regs[i]);
    }
    else  {
      if (!hasRegAlias(Regs[i], LivePhysRegs, TRI)) {
        LivePhysRegs.insert(Regs[i]);
        increasePhysRegPressure(Regs[i]);
      }
    }
  }
}

/// Add PhysReg to the live in set and increase max pressure.
void RegPressureTracker::discoverPhysLiveIn(unsigned Reg) {
  assert(!LivePhysRegs.count(Reg) && "avoid bumping max pressure twice");
  if (findRegAlias(Reg, P.LiveInRegs, TRI) != P.LiveInRegs.end())
    return;

  // At live in discovery, unconditionally increase the high water mark.
  P.LiveInRegs.push_back(Reg);
  P.increase(TRI->getMinimalPhysRegClass(Reg), TRI);
}

/// Add PhysReg to the live out set and increase max pressure.
void RegPressureTracker::discoverPhysLiveOut(unsigned Reg) {
  assert(!LivePhysRegs.count(Reg) && "avoid bumping max pressure twice");
  if (findRegAlias(Reg, P.LiveOutRegs, TRI) != P.LiveOutRegs.end())
    return;

  // At live out discovery, unconditionally increase the high water mark.
  P.LiveOutRegs.push_back(Reg);
  P.increase(TRI->getMinimalPhysRegClass(Reg), TRI);
}

/// Add VirtReg to the live in set and increase max pressure.
void RegPressureTracker::discoverVirtLiveIn(unsigned Reg) {
  assert(!LiveVirtRegs.count(Reg) && "avoid bumping max pressure twice");
  if (std::find(P.LiveInRegs.begin(), P.LiveInRegs.end(), Reg) !=
      P.LiveInRegs.end())
    return;

  // At live in discovery, unconditionally increase the high water mark.
  P.LiveInRegs.push_back(Reg);
  P.increase(MRI->getRegClass(Reg), TRI);
}

/// Add VirtReg to the live out set and increase max pressure.
void RegPressureTracker::discoverVirtLiveOut(unsigned Reg) {
  assert(!LiveVirtRegs.count(Reg) && "avoid bumping max pressure twice");
  if (std::find(P.LiveOutRegs.begin(), P.LiveOutRegs.end(), Reg) !=
      P.LiveOutRegs.end())
    return;

  // At live out discovery, unconditionally increase the high water mark.
  P.LiveOutRegs.push_back(Reg);
  P.increase(MRI->getRegClass(Reg), TRI);
}

/// Recede across the previous instruction.
bool RegPressureTracker::recede() {
  // Check for the top of the analyzable region.
  if (CurrPos == MBB->begin()) {
    closeRegion();
    return false;
  }
  if (!isBottomClosed())
    closeBottom();

  // Open the top of the region using block iterators.
  if (!RequireIntervals && isTopClosed())
    static_cast<RegionPressure&>(P).openTop(CurrPos);

  // Find the previous instruction.
  do
    --CurrPos;
  while (CurrPos != MBB->begin() && CurrPos->isDebugValue());

  if (CurrPos->isDebugValue()) {
    closeRegion();
    return false;
  }
  SlotIndex SlotIdx;
  if (RequireIntervals)
    SlotIdx = LIS->getInstructionIndex(CurrPos).getRegSlot();

  // Open the top of the region using slot indexes.
  if (RequireIntervals && isTopClosed())
    static_cast<IntervalPressure&>(P).openTop(SlotIdx);

  PhysRegOperands PhysRegOpers;
  VirtRegOperands VirtRegOpers;
  collectOperands(CurrPos, PhysRegOpers, VirtRegOpers, TRI, MRI);

  // Boost pressure for all dead defs together.
  increasePhysRegPressure(PhysRegOpers.DeadDefs);
  increaseVirtRegPressure(VirtRegOpers.DeadDefs);
  decreasePhysRegPressure(PhysRegOpers.DeadDefs);
  decreaseVirtRegPressure(VirtRegOpers.DeadDefs);

  // Kill liveness at live defs.
  // TODO: consider earlyclobbers?
  for (unsigned i = 0, e = PhysRegOpers.Defs.size(); i < e; ++i) {
    unsigned Reg = PhysRegOpers.Defs[i];
    if (LivePhysRegs.erase(Reg))
      decreasePhysRegPressure(Reg);
    else
      discoverPhysLiveOut(Reg);
  }
  for (unsigned i = 0, e = VirtRegOpers.Defs.size(); i < e; ++i) {
    unsigned Reg = VirtRegOpers.Defs[i];
    if (LiveVirtRegs.erase(Reg))
      decreaseVirtRegPressure(Reg);
    else
      discoverVirtLiveOut(Reg);
  }

  // Generate liveness for uses.
  for (unsigned i = 0, e = PhysRegOpers.Uses.size(); i < e; ++i) {
    unsigned Reg = PhysRegOpers.Uses[i];
    if (!hasRegAlias(Reg, LivePhysRegs, TRI)) {
      increasePhysRegPressure(Reg);
      LivePhysRegs.insert(Reg);
    }
  }
  for (unsigned i = 0, e = VirtRegOpers.Uses.size(); i < e; ++i) {
    unsigned Reg = VirtRegOpers.Uses[i];
    if (!LiveVirtRegs.count(Reg)) {
      // Adjust liveouts if LiveIntervals are available.
      if (RequireIntervals) {
        const LiveInterval *LI = &LIS->getInterval(Reg);
        if (!LI->killedAt(SlotIdx))
          discoverVirtLiveOut(Reg);
      }
      increaseVirtRegPressure(Reg);
      LiveVirtRegs.insert(Reg);
    }
  }
  return true;
}

/// Advance across the current instruction.
bool RegPressureTracker::advance() {
  // Check for the bottom of the analyzable region.
  if (CurrPos == MBB->end()) {
    closeRegion();
    return false;
  }
  if (!isTopClosed())
    closeTop();

  SlotIndex SlotIdx;
  if (RequireIntervals)
    SlotIdx = LIS->getInstructionIndex(CurrPos).getRegSlot();

  // Open the bottom of the region using slot indexes.
  if (isBottomClosed()) {
    if (RequireIntervals)
      static_cast<IntervalPressure&>(P).openBottom(SlotIdx);
    else
      static_cast<RegionPressure&>(P).openBottom(CurrPos);
  }

  PhysRegOperands PhysRegOpers;
  VirtRegOperands VirtRegOpers;
  collectOperands(CurrPos, PhysRegOpers, VirtRegOpers, TRI, MRI);

  // Kill liveness at last uses.
  for (unsigned i = 0, e = PhysRegOpers.Uses.size(); i < e; ++i) {
    unsigned Reg = PhysRegOpers.Uses[i];
    if (!hasRegAlias(Reg, LivePhysRegs, TRI))
      discoverPhysLiveIn(Reg);
    else {
      // Allocatable physregs are always single-use before regalloc.
      decreasePhysRegPressure(Reg);
      LivePhysRegs.erase(Reg);
    }
  }
  for (unsigned i = 0, e = VirtRegOpers.Uses.size(); i < e; ++i) {
    unsigned Reg = VirtRegOpers.Uses[i];
    if (RequireIntervals) {
      const LiveInterval *LI = &LIS->getInterval(Reg);
      if (LI->killedAt(SlotIdx)) {
        if (LiveVirtRegs.erase(Reg))
          decreaseVirtRegPressure(Reg);
        else
          discoverVirtLiveIn(Reg);
      }
    }
    else if (!LiveVirtRegs.count(Reg)) {
      discoverVirtLiveIn(Reg);
      increaseVirtRegPressure(Reg);
    }
  }

  // Generate liveness for defs.
  for (unsigned i = 0, e = PhysRegOpers.Defs.size(); i < e; ++i) {
    unsigned Reg = PhysRegOpers.Defs[i];
    if (!hasRegAlias(Reg, LivePhysRegs, TRI)) {
      increasePhysRegPressure(Reg);
      LivePhysRegs.insert(Reg);
    }
  }
  for (unsigned i = 0, e = VirtRegOpers.Defs.size(); i < e; ++i) {
    unsigned Reg = VirtRegOpers.Defs[i];
    if (LiveVirtRegs.insert(Reg).second)
      increaseVirtRegPressure(Reg);
  }

  // Boost pressure for all dead defs together.
  increasePhysRegPressure(PhysRegOpers.DeadDefs);
  increaseVirtRegPressure(VirtRegOpers.DeadDefs);
  decreasePhysRegPressure(PhysRegOpers.DeadDefs);
  decreaseVirtRegPressure(VirtRegOpers.DeadDefs);

  // Find the next instruction.
  do
    ++CurrPos;
  while (CurrPos != MBB->end() && CurrPos->isDebugValue());
  return true;
}

/// Find the max change in excess pressure across all sets.
static void computeExcessPressureDelta(ArrayRef<unsigned> OldPressureVec,
                                       ArrayRef<unsigned> NewPressureVec,
                                       RegPressureDelta &Delta,
                                       const TargetRegisterInfo *TRI) {
  int ExcessUnits = 0;
  unsigned PSetID = ~0U;
  for (unsigned i = 0, e = OldPressureVec.size(); i < e; ++i) {
    unsigned POld = OldPressureVec[i];
    unsigned PNew = NewPressureVec[i];
    int PDiff = (int)PNew - (int)POld;
    if (!PDiff) // No change in this set in the common case.
      continue;
    // Only consider change beyond the limit.
    unsigned Limit = TRI->getRegPressureSetLimit(i);
    if (Limit > POld) {
      if (Limit > PNew)
        PDiff = 0;            // Under the limit
      else
        PDiff = PNew - Limit; // Just exceeded limit.
    }
    else if (Limit > PNew)
      PDiff = Limit - POld;   // Just obeyed limit.

    if (std::abs(PDiff) > std::abs(ExcessUnits)) {
      ExcessUnits = PDiff;
      PSetID = i;
    }
  }
  Delta.Excess.PSetID = PSetID;
  Delta.Excess.UnitIncrease = ExcessUnits;
}

/// Find the max change in max pressure that either surpasses a critical PSet
/// limit or exceeds the current MaxPressureLimit.
///
/// FIXME: comparing each element of the old and new MaxPressure vectors here is
/// silly. It's done now to demonstrate the concept but will go away with a
/// RegPressureTracker API change to work with pressure differences.
static void computeMaxPressureDelta(ArrayRef<unsigned> OldMaxPressureVec,
                                    ArrayRef<unsigned> NewMaxPressureVec,
                                    ArrayRef<PressureElement> CriticalPSets,
                                    ArrayRef<unsigned> MaxPressureLimit,
                                    RegPressureDelta &Delta) {
  Delta.CriticalMax = PressureElement();
  Delta.CurrentMax = PressureElement();

  unsigned CritIdx = 0, CritEnd = CriticalPSets.size();
  for (unsigned i = 0, e = OldMaxPressureVec.size(); i < e; ++i) {
    unsigned POld = OldMaxPressureVec[i];
    unsigned PNew = NewMaxPressureVec[i];
    if (PNew == POld) // No change in this set in the common case.
      continue;

    while (CritIdx != CritEnd && CriticalPSets[CritIdx].PSetID < i)
      ++CritIdx;

    if (CritIdx != CritEnd && CriticalPSets[CritIdx].PSetID == i) {
      int PDiff = (int)PNew - (int)CriticalPSets[CritIdx].UnitIncrease;
      if (PDiff > Delta.CriticalMax.UnitIncrease) {
        Delta.CriticalMax.PSetID = i;
        Delta.CriticalMax.UnitIncrease = PDiff;
      }
    }

    // Find the greatest increase above MaxPressureLimit.
    // (Ignores negative MDiff).
    int MDiff = (int)PNew - (int)MaxPressureLimit[i];
    if (MDiff > Delta.CurrentMax.UnitIncrease) {
      Delta.CurrentMax.PSetID = i;
      Delta.CurrentMax.UnitIncrease = PNew;
    }
  }
}

/// Record the upward impact of a single instruction on current register
/// pressure. Unlike the advance/recede pressure tracking interface, this does
/// not discover live in/outs.
///
/// This is intended for speculative queries. It leaves pressure inconsistent
/// with the current position, so must be restored by the caller.
void RegPressureTracker::bumpUpwardPressure(const MachineInstr *MI) {
  // Account for register pressure similar to RegPressureTracker::recede().
  PhysRegOperands PhysRegOpers;
  VirtRegOperands VirtRegOpers;
  collectOperands(MI, PhysRegOpers, VirtRegOpers, TRI, MRI);

  // Boost max pressure for all dead defs together.
  // Since CurrSetPressure and MaxSetPressure
  increasePhysRegPressure(PhysRegOpers.DeadDefs);
  increaseVirtRegPressure(VirtRegOpers.DeadDefs);
  decreasePhysRegPressure(PhysRegOpers.DeadDefs);
  decreaseVirtRegPressure(VirtRegOpers.DeadDefs);

  // Kill liveness at live defs.
  for (unsigned i = 0, e = PhysRegOpers.Defs.size(); i < e; ++i) {
    unsigned Reg = PhysRegOpers.Defs[i];
    if (!findReg(Reg, false, PhysRegOpers.Uses, TRI))
      decreasePhysRegPressure(PhysRegOpers.Defs);
  }
  for (unsigned i = 0, e = VirtRegOpers.Defs.size(); i < e; ++i) {
    unsigned Reg = VirtRegOpers.Defs[i];
    if (!findReg(Reg, true, VirtRegOpers.Uses, TRI))
      decreaseVirtRegPressure(VirtRegOpers.Defs);
  }
  // Generate liveness for uses.
  for (unsigned i = 0, e = PhysRegOpers.Uses.size(); i < e; ++i) {
    unsigned Reg = PhysRegOpers.Uses[i];
    if (!hasRegAlias(Reg, LivePhysRegs, TRI))
      increasePhysRegPressure(Reg);
  }
  for (unsigned i = 0, e = VirtRegOpers.Uses.size(); i < e; ++i) {
    unsigned Reg = VirtRegOpers.Uses[i];
    if (!LiveVirtRegs.count(Reg))
      increaseVirtRegPressure(Reg);
  }
}

/// Consider the pressure increase caused by traversing this instruction
/// bottom-up. Find the pressure set with the most change beyond its pressure
/// limit based on the tracker's current pressure, and return the change in
/// number of register units of that pressure set introduced by this
/// instruction.
///
/// This assumes that the current LiveOut set is sufficient.
///
/// FIXME: This is expensive for an on-the-fly query. We need to cache the
/// result per-SUnit with enough information to adjust for the current
/// scheduling position. But this works as a proof of concept.
void RegPressureTracker::
getMaxUpwardPressureDelta(const MachineInstr *MI, RegPressureDelta &Delta,
                          ArrayRef<PressureElement> CriticalPSets,
                          ArrayRef<unsigned> MaxPressureLimit) {
  // Snapshot Pressure.
  // FIXME: The snapshot heap space should persist. But I'm planning to
  // summarize the pressure effect so we don't need to snapshot at all.
  std::vector<unsigned> SavedPressure = CurrSetPressure;
  std::vector<unsigned> SavedMaxPressure = P.MaxSetPressure;

  bumpUpwardPressure(MI);

  computeExcessPressureDelta(SavedPressure, CurrSetPressure, Delta, TRI);
  computeMaxPressureDelta(SavedMaxPressure, P.MaxSetPressure, CriticalPSets,
                          MaxPressureLimit, Delta);
  assert(Delta.CriticalMax.UnitIncrease >= 0 &&
         Delta.CurrentMax.UnitIncrease >= 0 && "cannot decrease max pressure");

  // Restore the tracker's state.
  P.MaxSetPressure.swap(SavedMaxPressure);
  CurrSetPressure.swap(SavedPressure);
}

/// Helper to find a vreg use between two indices [PriorUseIdx, NextUseIdx).
static bool findUseBetween(unsigned Reg,
                           SlotIndex PriorUseIdx, SlotIndex NextUseIdx,
                           const MachineRegisterInfo *MRI,
                           const LiveIntervals *LIS) {
  for (MachineRegisterInfo::use_nodbg_iterator
         UI = MRI->use_nodbg_begin(Reg), UE = MRI->use_nodbg_end();
         UI != UE; UI.skipInstruction()) {
      const MachineInstr* MI = &*UI;
      SlotIndex InstSlot = LIS->getInstructionIndex(MI).getRegSlot();
      if (InstSlot >= PriorUseIdx && InstSlot < NextUseIdx)
        return true;
  }
  return false;
}

/// Record the downward impact of a single instruction on current register
/// pressure. Unlike the advance/recede pressure tracking interface, this does
/// not discover live in/outs.
///
/// This is intended for speculative queries. It leaves pressure inconsistent
/// with the current position, so must be restored by the caller.
void RegPressureTracker::bumpDownwardPressure(const MachineInstr *MI) {
  // Account for register pressure similar to RegPressureTracker::recede().
  PhysRegOperands PhysRegOpers;
  VirtRegOperands VirtRegOpers;
  collectOperands(MI, PhysRegOpers, VirtRegOpers, TRI, MRI);

  // Kill liveness at last uses. Assume allocatable physregs are single-use
  // rather than checking LiveIntervals.
  decreasePhysRegPressure(PhysRegOpers.Uses);
  if (RequireIntervals) {
    SlotIndex SlotIdx = LIS->getInstructionIndex(MI).getRegSlot();
    for (unsigned i = 0, e = VirtRegOpers.Uses.size(); i < e; ++i) {
      unsigned Reg = VirtRegOpers.Uses[i];
      const LiveInterval *LI = &LIS->getInterval(Reg);
      // FIXME: allow the caller to pass in the list of vreg uses that remain to
      // be bottom-scheduled to avoid searching uses at each query.
      SlotIndex CurrIdx = LIS->getInstructionIndex(CurrPos).getRegSlot();
      if (LI->killedAt(SlotIdx)
          && !findUseBetween(Reg, CurrIdx, SlotIdx, MRI, LIS)) {
        decreaseVirtRegPressure(Reg);
      }
    }
  }

  // Generate liveness for defs.
  increasePhysRegPressure(PhysRegOpers.Defs);
  increaseVirtRegPressure(VirtRegOpers.Defs);

  // Boost pressure for all dead defs together.
  increasePhysRegPressure(PhysRegOpers.DeadDefs);
  increaseVirtRegPressure(VirtRegOpers.DeadDefs);
  decreasePhysRegPressure(PhysRegOpers.DeadDefs);
  decreaseVirtRegPressure(VirtRegOpers.DeadDefs);
}

/// Consider the pressure increase caused by traversing this instruction
/// top-down. Find the register class with the most change in its pressure limit
/// based on the tracker's current pressure, and return the number of excess
/// register units of that pressure set introduced by this instruction.
///
/// This assumes that the current LiveIn set is sufficient.
void RegPressureTracker::
getMaxDownwardPressureDelta(const MachineInstr *MI, RegPressureDelta &Delta,
                            ArrayRef<PressureElement> CriticalPSets,
                            ArrayRef<unsigned> MaxPressureLimit) {
  // Snapshot Pressure.
  std::vector<unsigned> SavedPressure = CurrSetPressure;
  std::vector<unsigned> SavedMaxPressure = P.MaxSetPressure;

  bumpDownwardPressure(MI);

  computeExcessPressureDelta(SavedPressure, CurrSetPressure, Delta, TRI);
  computeMaxPressureDelta(SavedMaxPressure, P.MaxSetPressure, CriticalPSets,
                          MaxPressureLimit, Delta);
  assert(Delta.CriticalMax.UnitIncrease >= 0 &&
         Delta.CurrentMax.UnitIncrease >= 0 && "cannot decrease max pressure");

  // Restore the tracker's state.
  P.MaxSetPressure.swap(SavedMaxPressure);
  CurrSetPressure.swap(SavedPressure);
}

/// Get the pressure of each PSet after traversing this instruction bottom-up.
void RegPressureTracker::
getUpwardPressure(const MachineInstr *MI,
                  std::vector<unsigned> &PressureResult,
                  std::vector<unsigned> &MaxPressureResult) {
  // Snapshot pressure.
  PressureResult = CurrSetPressure;
  MaxPressureResult = P.MaxSetPressure;

  bumpUpwardPressure(MI);

  // Current pressure becomes the result. Restore current pressure.
  P.MaxSetPressure.swap(MaxPressureResult);
  CurrSetPressure.swap(PressureResult);
}

/// Get the pressure of each PSet after traversing this instruction top-down.
void RegPressureTracker::
getDownwardPressure(const MachineInstr *MI,
                    std::vector<unsigned> &PressureResult,
                    std::vector<unsigned> &MaxPressureResult) {
  // Snapshot pressure.
  PressureResult = CurrSetPressure;
  MaxPressureResult = P.MaxSetPressure;

  bumpDownwardPressure(MI);

  // Current pressure becomes the result. Restore current pressure.
  P.MaxSetPressure.swap(MaxPressureResult);
  CurrSetPressure.swap(PressureResult);
}
