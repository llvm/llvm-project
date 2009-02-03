//===- X86RegisterInfo.h - X86 Register Information Impl --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the X86 implementation of the TargetRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef X86REGISTERINFO_H
#define X86REGISTERINFO_H

#include "llvm/Target/TargetRegisterInfo.h"
#include "X86GenRegisterInfo.h.inc"

namespace llvm {
  class Type;
  class TargetInstrInfo;
  class X86TargetMachine;

/// N86 namespace - Native X86 register numbers
///
namespace N86 {
  enum {
    EAX = 0, ECX = 1, EDX = 2, EBX = 3, ESP = 4, EBP = 5, ESI = 6, EDI = 7
  };
}

namespace X86 {
  /// SubregIndex - The index of various sized subregister classes. Note that 
  /// these indices must be kept in sync with the class indices in the 
  /// X86RegisterInfo.td file.
  enum SubregIndex {
    SUBREG_8BIT = 1, SUBREG_16BIT = 2, SUBREG_32BIT = 3
  };
}

/// DWARFFlavour - Flavour of dwarf regnumbers
///
namespace DWARFFlavour {
  enum {
    X86_64 = 0, X86_32_DarwinEH = 1, X86_32_Generic = 2
  };
} 
  
class X86RegisterInfo : public X86GenRegisterInfo {
public:
  X86TargetMachine &TM;
  const TargetInstrInfo &TII;

private:
  /// Is64Bit - Is the target 64-bits.
  ///
  bool Is64Bit;

  /// IsWin64 - Is the target on of win64 flavours
  ///
  bool IsWin64;

  /// SlotSize - Stack slot size in bytes.
  ///
  unsigned SlotSize;

  /// StackAlign - Default stack alignment.
  ///
  unsigned StackAlign;

  /// StackPtr - X86 physical register used as stack ptr.
  ///
  unsigned StackPtr;

  /// FramePtr - X86 physical register used as frame ptr.
  ///
  unsigned FramePtr;

public:
  X86RegisterInfo(X86TargetMachine &tm, const TargetInstrInfo &tii);

  /// getX86RegNum - Returns the native X86 register number for the given LLVM
  /// register identifier.
  static unsigned getX86RegNum(unsigned RegNo);

  unsigned getStackAlignment() const { return StackAlign; }

  /// getDwarfRegNum - allows modification of X86GenRegisterInfo::getDwarfRegNum
  /// (created by TableGen) for target dependencies.
  int getDwarfRegNum(unsigned RegNum, bool isEH) const;

  /// Code Generation virtual methods...
  /// 

  /// getCrossCopyRegClass - Returns a legal register class to copy a register
  /// in the specified class to or from. Returns NULL if it is possible to copy
  /// between a two registers of the specified class.
  const TargetRegisterClass *
  getCrossCopyRegClass(const TargetRegisterClass *RC) const;

  /// getCalleeSavedRegs - Return a null-terminated list of all of the
  /// callee-save registers on this target.
  const unsigned *getCalleeSavedRegs(const MachineFunction* MF = 0) const;

  /// getCalleeSavedRegClasses - Return a null-terminated list of the preferred
  /// register classes to spill each callee-saved register with.  The order and
  /// length of this list match the getCalleeSavedRegs() list.
  const TargetRegisterClass* const*
  getCalleeSavedRegClasses(const MachineFunction *MF = 0) const;

  /// getReservedRegs - Returns a bitset indexed by physical register number
  /// indicating if a register is a special register that has particular uses and
  /// should be considered unavailable at all times, e.g. SP, RA. This is used by
  /// register scavenger to determine what registers are free.
  BitVector getReservedRegs(const MachineFunction &MF) const;

  bool hasFP(const MachineFunction &MF) const;

  bool needsStackRealignment(const MachineFunction &MF) const;

  bool hasReservedCallFrame(MachineFunction &MF) const;

  void eliminateCallFramePseudoInstr(MachineFunction &MF,
                                     MachineBasicBlock &MBB,
                                     MachineBasicBlock::iterator MI) const;

  void eliminateFrameIndex(MachineBasicBlock::iterator MI,
                           int SPAdj, RegScavenger *RS = NULL) const;

  void processFunctionBeforeFrameFinalized(MachineFunction &MF) const;
  void processFunctionBeforeCalleeSavedScan(MachineFunction &MF,
                                            RegScavenger *RS = NULL) const;

  void emitPrologue(MachineFunction &MF) const;
  void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const;

  void emitFrameMoves(MachineFunction &MF,
                      unsigned FrameLabelId, unsigned ReadyLabelId) const;

  // Debug information queries.
  unsigned getRARegister() const;
  unsigned getFrameRegister(MachineFunction &MF) const;
  int getFrameIndexOffset(MachineFunction &MF, int FI) const;
  void getInitialFrameState(std::vector<MachineMove> &Moves) const;

  // Exception handling queries.
  unsigned getEHExceptionRegister() const;
  unsigned getEHHandlerRegister() const;
};

// getX86SubSuperRegister - X86 utility function. It returns the sub or super
// register of a specific X86 register.
// e.g. getX86SubSuperRegister(X86::EAX, MVT::i16) return X86:AX
unsigned getX86SubSuperRegister(unsigned, MVT, bool High=false);

} // End llvm namespace

#endif
