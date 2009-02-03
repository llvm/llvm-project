//===-- RegisterScavenging.h - Machine register scavenging ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the machine register scavenger class. It can provide
// information such as unused register at any point in a machine basic block.
// It also provides a mechanism to make registers availbale by evicting them
// to spill slots.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_REGISTER_SCAVENGING_H
#define LLVM_CODEGEN_REGISTER_SCAVENGING_H

#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"

namespace llvm {

class MachineRegisterInfo;
class TargetRegisterInfo;
class TargetInstrInfo;
class TargetRegisterClass;

class RegScavenger {
  const TargetRegisterInfo *TRI;
  const TargetInstrInfo *TII;
  MachineRegisterInfo* MRI;
  MachineBasicBlock *MBB;
  MachineBasicBlock::iterator MBBI;
  unsigned NumPhysRegs;

  /// Tracking - True if RegScavenger is currently tracking the liveness of 
  /// registers.
  bool Tracking;

  /// ScavengingFrameIndex - Special spill slot used for scavenging a register
  /// post register allocation.
  int ScavengingFrameIndex;

  /// ScavengedReg - If none zero, the specific register is currently being
  /// scavenged. That is, it is spilled to the special scavenging stack slot.
  unsigned ScavengedReg;

  /// ScavengedRC - Register class of the scavenged register.
  ///
  const TargetRegisterClass *ScavengedRC;

  /// ScavengeRestore - Instruction that restores the scavenged register from
  /// stack.
  const MachineInstr *ScavengeRestore;

  /// CalleeSavedrRegs - A bitvector of callee saved registers for the target.
  ///
  BitVector CalleeSavedRegs;

  /// ReservedRegs - A bitvector of reserved registers.
  ///
  BitVector ReservedRegs;

  /// RegsAvailable - The current state of all the physical registers immediately
  /// before MBBI. One bit per physical register. If bit is set that means it's
  /// available, unset means the register is currently being used.
  BitVector RegsAvailable;

  /// ImplicitDefed - If bit is set that means the register is defined by an
  /// implicit_def instructions. That means it can be clobbered at will.
  BitVector ImplicitDefed;

  /// CurrDist - Distance from MBB entry to the current instruction MBBI.
  ///
  unsigned CurrDist;

  /// DistanceMap - Keep track the distance of a MI from the start of the
  /// current basic block.
  DenseMap<MachineInstr*, unsigned> DistanceMap;

public:
  RegScavenger()
    : MBB(NULL), NumPhysRegs(0), Tracking(false),
      ScavengingFrameIndex(-1), ScavengedReg(0), ScavengedRC(NULL) {}

  /// enterBasicBlock - Start tracking liveness from the begin of the specific
  /// basic block.
  void enterBasicBlock(MachineBasicBlock *mbb);

  /// forward / backward - Move the internal MBB iterator and update register
  /// states.
  void forward();
  void backward();

  /// forward / backward - Move the internal MBB iterator and update register
  /// states until it has processed the specific iterator.
  void forward(MachineBasicBlock::iterator I) {
    if (!Tracking && MBB->begin() != I) forward();
    while (MBBI != I) forward();
  }
  void backward(MachineBasicBlock::iterator I) {
    while (MBBI != I) backward();
  }

  /// skipTo - Move the internal MBB iterator but do not update register states.
  ///
  void skipTo(MachineBasicBlock::iterator I) { MBBI = I; }

  /// isReserved - Returns true if a register is reserved. It is never "unused".
  bool isReserved(unsigned Reg) const { return ReservedRegs[Reg]; }

  /// isUsed / isUsed - Test if a register is currently being used.
  ///
  bool isUsed(unsigned Reg) const   { return !RegsAvailable[Reg]; }
  bool isUnused(unsigned Reg) const { return RegsAvailable[Reg]; }

  bool isImplicitlyDefined(unsigned Reg) const { return ImplicitDefed[Reg]; }

  /// getRegsUsed - return all registers currently in use in used.
  void getRegsUsed(BitVector &used, bool includeReserved);

  /// setUsed / setUnused - Mark the state of one or a number of registers.
  ///
  void setUsed(unsigned Reg, bool ImpDef = false);
  void setUsed(BitVector Regs, bool ImpDef = false) {
    RegsAvailable &= ~Regs;
    if (ImpDef)
      ImplicitDefed |= Regs;
    else
      ImplicitDefed &= ~Regs;
  }
  void setUnused(unsigned Reg, const MachineInstr *MI);
  void setUnused(BitVector Regs) {
    RegsAvailable |= Regs;
    ImplicitDefed &= ~Regs;
  }

  /// FindUnusedReg - Find a unused register of the specified register class
  /// from the specified set of registers. It return 0 is none is found.
  unsigned FindUnusedReg(const TargetRegisterClass *RegClass,
                         const BitVector &Candidates) const;

  /// FindUnusedReg - Find a unused register of the specified register class.
  /// Exclude callee saved registers if directed. It return 0 is none is found.
  unsigned FindUnusedReg(const TargetRegisterClass *RegClass,
                         bool ExCalleeSaved = false) const;

  /// setScavengingFrameIndex / getScavengingFrameIndex - accessor and setter of
  /// ScavengingFrameIndex.
  void setScavengingFrameIndex(int FI) { ScavengingFrameIndex = FI; }
  int getScavengingFrameIndex() const { return ScavengingFrameIndex; }

  /// scavengeRegister - Make a register of the specific register class
  /// available and do the appropriate bookkeeping. SPAdj is the stack
  /// adjustment due to call frame, it's passed along to eliminateFrameIndex().
  /// Returns the scavenged register.
  unsigned scavengeRegister(const TargetRegisterClass *RegClass,
                            MachineBasicBlock::iterator I, int SPAdj);
  unsigned scavengeRegister(const TargetRegisterClass *RegClass, int SPAdj) {
    return scavengeRegister(RegClass, MBBI, SPAdj);
  }

private:
  /// restoreScavengedReg - Restore scavenged by loading it back from the
  /// emergency spill slot. Mark it used.
  void restoreScavengedReg();

  MachineInstr *findFirstUse(MachineBasicBlock *MBB,
                             MachineBasicBlock::iterator I, unsigned Reg,
                             unsigned &Dist);
};
 
} // End llvm namespace

#endif
