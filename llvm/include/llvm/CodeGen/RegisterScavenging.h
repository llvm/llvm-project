//===- RegisterScavenging.h - Machine register scavenging -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This file declares the machine register scavenger class. It can provide
/// information such as unused register at any point in a machine basic block.
/// It also provides a mechanism to make registers available by evicting them
/// to spill slots.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_REGISTERSCAVENGING_H
#define LLVM_CODEGEN_REGISTERSCAVENGING_H

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/LiveRegUnits.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/MC/LaneBitmask.h"

namespace llvm {

class MachineInstr;
class TargetInstrInfo;
class TargetRegisterClass;
class TargetRegisterInfo;

class RegScavenger {
  const TargetRegisterInfo *TRI;
  const TargetInstrInfo *TII;
  MachineRegisterInfo* MRI;
  MachineBasicBlock *MBB = nullptr;
  MachineBasicBlock::iterator MBBI;
  unsigned NumRegUnits = 0;

  /// True if RegScavenger is currently tracking the liveness of registers.
  bool Tracking = false;

  /// Information on scavenged registers (held in a spill slot).
  struct ScavengedInfo {
    ScavengedInfo(int FI = -1) : FrameIndex(FI) {}

    /// A spill slot used for scavenging a register post register allocation.
    int FrameIndex;

    /// If non-zero, the specific register is currently being
    /// scavenged. That is, it is spilled to this scavenging stack slot.
    Register Reg;

    /// The instruction that restores the scavenged register from stack.
    const MachineInstr *Restore = nullptr;
  };

  /// A vector of information on scavenged registers.
  SmallVector<ScavengedInfo, 2> Scavenged;

  LiveRegUnits LiveUnits;

  // These BitVectors are only used internally to forward(). They are members
  // to avoid frequent reallocations.
  BitVector KillRegUnits, DefRegUnits;
  BitVector TmpRegUnits;

public:
  RegScavenger() = default;

  /// Start tracking liveness from the begin of basic block \p MBB.
  void enterBasicBlock(MachineBasicBlock &MBB);

  /// Start tracking liveness from the end of basic block \p MBB.
  /// Use backward() to move towards the beginning of the block. This is
  /// preferred to enterBasicBlock() and forward() because it does not depend
  /// on the presence of kill flags.
  void enterBasicBlockEnd(MachineBasicBlock &MBB);

  /// Move the internal MBB iterator and update register states.
  void forward();

  /// Move the internal MBB iterator and update register states until
  /// it has processed the specific iterator.
  void forward(MachineBasicBlock::iterator I) {
    if (!Tracking && MBB->begin() != I) forward();
    while (MBBI != I) forward();
  }

  /// Update internal register state and move MBB iterator backwards.
  /// Contrary to unprocess() this method gives precise results even in the
  /// absence of kill flags.
  void backward();

  /// Call backward() as long as the internal iterator does not point to \p I.
  void backward(MachineBasicBlock::iterator I) {
    while (MBBI != I)
      backward();
  }

  /// Move the internal MBB iterator but do not update register states.
  void skipTo(MachineBasicBlock::iterator I) {
    if (I == MachineBasicBlock::iterator(nullptr))
      Tracking = false;
    MBBI = I;
  }

  MachineBasicBlock::iterator getCurrentPosition() const { return MBBI; }

  /// Return if a specific register is currently used.
  bool isRegUsed(Register Reg, bool includeReserved = true) const;

  /// Return all available registers in the register class in Mask.
  BitVector getRegsAvailable(const TargetRegisterClass *RC);

  /// Find an unused register of the specified register class.
  /// Return 0 if none is found.
  Register FindUnusedReg(const TargetRegisterClass *RC) const;

  /// Add a scavenging frame index.
  void addScavengingFrameIndex(int FI) {
    Scavenged.push_back(ScavengedInfo(FI));
  }

  /// Query whether a frame index is a scavenging frame index.
  bool isScavengingFrameIndex(int FI) const {
    for (SmallVectorImpl<ScavengedInfo>::const_iterator I = Scavenged.begin(),
         IE = Scavenged.end(); I != IE; ++I)
      if (I->FrameIndex == FI)
        return true;

    return false;
  }

  /// Get an array of scavenging frame indices.
  void getScavengingFrameIndices(SmallVectorImpl<int> &A) const {
    for (SmallVectorImpl<ScavengedInfo>::const_iterator I = Scavenged.begin(),
         IE = Scavenged.end(); I != IE; ++I)
      if (I->FrameIndex >= 0)
        A.push_back(I->FrameIndex);
  }

  /// Make a register of the specific register class
  /// available and do the appropriate bookkeeping. SPAdj is the stack
  /// adjustment due to call frame, it's passed along to eliminateFrameIndex().
  /// Returns the scavenged register.
  /// This is deprecated as it depends on the quality of the kill flags being
  /// present; Use scavengeRegisterBackwards() instead!
  ///
  /// If \p AllowSpill is false, fail if a spill is required to make the
  /// register available, and return NoRegister.
  Register scavengeRegister(const TargetRegisterClass *RC,
                            MachineBasicBlock::iterator I, int SPAdj,
                            bool AllowSpill = true);
  Register scavengeRegister(const TargetRegisterClass *RegClass, int SPAdj,
                            bool AllowSpill = true) {
    return scavengeRegister(RegClass, MBBI, SPAdj, AllowSpill);
  }

  /// Make a register of the specific register class available from the current
  /// position backwards to the place before \p To. If \p RestoreAfter is true
  /// this includes the instruction following the current position.
  /// SPAdj is the stack adjustment due to call frame, it's passed along to
  /// eliminateFrameIndex().
  /// Returns the scavenged register.
  ///
  /// If \p AllowSpill is false, fail if a spill is required to make the
  /// register available, and return NoRegister.
  Register scavengeRegisterBackwards(const TargetRegisterClass &RC,
                                     MachineBasicBlock::iterator To,
                                     bool RestoreAfter, int SPAdj,
                                     bool AllowSpill = true);

  /// Tell the scavenger a register is used.
  void setRegUsed(Register Reg, LaneBitmask LaneMask = LaneBitmask::getAll());

private:
  /// Returns true if a register is reserved. It is never "unused".
  bool isReserved(Register Reg) const { return MRI->isReserved(Reg); }

  /// setUsed / setUnused - Mark the state of one or a number of register units.
  ///
  void setUsed(const BitVector &RegUnits) {
    LiveUnits.addUnits(RegUnits);
  }
  void setUnused(const BitVector &RegUnits) {
    LiveUnits.removeUnits(RegUnits);
  }

  /// Processes the current instruction and fill the KillRegUnits and
  /// DefRegUnits bit vectors.
  void determineKillsAndDefs();

  /// Add all Reg Units that Reg contains to BV.
  void addRegUnits(BitVector &BV, MCRegister Reg);

  /// Remove all Reg Units that \p Reg contains from \p BV.
  void removeRegUnits(BitVector &BV, MCRegister Reg);

  /// Return the candidate register that is unused for the longest after
  /// StartMI. UseMI is set to the instruction where the search stopped.
  ///
  /// No more than InstrLimit instructions are inspected.
  Register findSurvivorReg(MachineBasicBlock::iterator StartMI,
                           BitVector &Candidates,
                           unsigned InstrLimit,
                           MachineBasicBlock::iterator &UseMI);

  /// Initialize RegisterScavenger.
  void init(MachineBasicBlock &MBB);

  /// Spill a register after position \p After and reload it before position
  /// \p UseMI.
  ScavengedInfo &spill(Register Reg, const TargetRegisterClass &RC, int SPAdj,
                       MachineBasicBlock::iterator Before,
                       MachineBasicBlock::iterator &UseMI);
};

/// Replaces all frame index virtual registers with physical registers. Uses the
/// register scavenger to find an appropriate register to use.
void scavengeFrameVirtualRegs(MachineFunction &MF, RegScavenger &RS);

} // end namespace llvm

#endif // LLVM_CODEGEN_REGISTERSCAVENGING_H
