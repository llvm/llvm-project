//===- PPCInstrInfo.h - PowerPC Instruction Information ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the PowerPC implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef POWERPC32_INSTRUCTIONINFO_H
#define POWERPC32_INSTRUCTIONINFO_H

#include "PPC.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "PPCRegisterInfo.h"

namespace llvm {

/// PPCII - This namespace holds all of the PowerPC target-specific
/// per-instruction flags.  These must match the corresponding definitions in
/// PPC.td and PPCInstrFormats.td.
namespace PPCII {
enum {
  // PPC970 Instruction Flags.  These flags describe the characteristics of the
  // PowerPC 970 (aka G5) dispatch groups and how they are formed out of
  // raw machine instructions.

  /// PPC970_First - This instruction starts a new dispatch group, so it will
  /// always be the first one in the group.
  PPC970_First = 0x1,
  
  /// PPC970_Single - This instruction starts a new dispatch group and
  /// terminates it, so it will be the sole instruction in the group.
  PPC970_Single = 0x2,

  /// PPC970_Cracked - This instruction is cracked into two pieces, requiring
  /// two dispatch pipes to be available to issue.
  PPC970_Cracked = 0x4,
  
  /// PPC970_Mask/Shift - This is a bitmask that selects the pipeline type that
  /// an instruction is issued to.
  PPC970_Shift = 3,
  PPC970_Mask = 0x07 << PPC970_Shift
};
enum PPC970_Unit {
  /// These are the various PPC970 execution unit pipelines.  Each instruction
  /// is one of these.
  PPC970_Pseudo = 0 << PPC970_Shift,   // Pseudo instruction
  PPC970_FXU    = 1 << PPC970_Shift,   // Fixed Point (aka Integer/ALU) Unit
  PPC970_LSU    = 2 << PPC970_Shift,   // Load Store Unit
  PPC970_FPU    = 3 << PPC970_Shift,   // Floating Point Unit
  PPC970_CRU    = 4 << PPC970_Shift,   // Control Register Unit
  PPC970_VALU   = 5 << PPC970_Shift,   // Vector ALU
  PPC970_VPERM  = 6 << PPC970_Shift,   // Vector Permute Unit
  PPC970_BRU    = 7 << PPC970_Shift    // Branch Unit
};
}
  
  
class PPCInstrInfo : public TargetInstrInfoImpl {
  PPCTargetMachine &TM;
  const PPCRegisterInfo RI;

  bool StoreRegToStackSlot(MachineFunction &MF,
                           unsigned SrcReg, bool isKill, int FrameIdx,
                           const TargetRegisterClass *RC,
                           SmallVectorImpl<MachineInstr*> &NewMIs) const;
  void LoadRegFromStackSlot(MachineFunction &MF,
                            unsigned DestReg, int FrameIdx,
                            const TargetRegisterClass *RC,
                            SmallVectorImpl<MachineInstr*> &NewMIs) const;
public:
  explicit PPCInstrInfo(PPCTargetMachine &TM);

  /// getRegisterInfo - TargetInstrInfo is a superset of MRegister info.  As
  /// such, whenever a client has an instance of instruction info, it should
  /// always be able to get register info as well (through this method).
  ///
  virtual const PPCRegisterInfo &getRegisterInfo() const { return RI; }

  /// getPointerRegClass - Return the register class to use to hold pointers.
  /// This is used for addressing modes.
  virtual const TargetRegisterClass *getPointerRegClass() const;  

  /// Return true if the instruction is a register to register move and return
  /// the source and dest operands and their sub-register indices by reference.
  virtual bool isMoveInstr(const MachineInstr &MI,
                           unsigned &SrcReg, unsigned &DstReg,
                           unsigned &SrcSubIdx, unsigned &DstSubIdx) const;

  unsigned isLoadFromStackSlot(const MachineInstr *MI,
                               int &FrameIndex) const;
  unsigned isStoreToStackSlot(const MachineInstr *MI,
                              int &FrameIndex) const;

  // commuteInstruction - We can commute rlwimi instructions, but only if the
  // rotate amt is zero.  We also have to munge the immediates a bit.
  virtual MachineInstr *commuteInstruction(MachineInstr *MI, bool NewMI) const;
  
  virtual void insertNoop(MachineBasicBlock &MBB, 
                          MachineBasicBlock::iterator MI) const;


  // Branch analysis.
  virtual bool AnalyzeBranch(MachineBasicBlock &MBB, MachineBasicBlock *&TBB,
                             MachineBasicBlock *&FBB,
                             SmallVectorImpl<MachineOperand> &Cond,
                             bool AllowModify) const;
  virtual unsigned RemoveBranch(MachineBasicBlock &MBB) const;
  virtual unsigned InsertBranch(MachineBasicBlock &MBB, MachineBasicBlock *TBB,
                                MachineBasicBlock *FBB,
                            const SmallVectorImpl<MachineOperand> &Cond) const;
  virtual bool copyRegToReg(MachineBasicBlock &MBB,
                            MachineBasicBlock::iterator MI,
                            unsigned DestReg, unsigned SrcReg,
                            const TargetRegisterClass *DestRC,
                            const TargetRegisterClass *SrcRC) const;
  
  virtual void storeRegToStackSlot(MachineBasicBlock &MBB,
                                   MachineBasicBlock::iterator MBBI,
                                   unsigned SrcReg, bool isKill, int FrameIndex,
                                   const TargetRegisterClass *RC) const;

  virtual void storeRegToAddr(MachineFunction &MF, unsigned SrcReg, bool isKill,
                              SmallVectorImpl<MachineOperand> &Addr,
                              const TargetRegisterClass *RC,
                              SmallVectorImpl<MachineInstr*> &NewMIs) const;

  virtual void loadRegFromStackSlot(MachineBasicBlock &MBB,
                                    MachineBasicBlock::iterator MBBI,
                                    unsigned DestReg, int FrameIndex,
                                    const TargetRegisterClass *RC) const;

  virtual void loadRegFromAddr(MachineFunction &MF, unsigned DestReg,
                               SmallVectorImpl<MachineOperand> &Addr,
                               const TargetRegisterClass *RC,
                               SmallVectorImpl<MachineInstr*> &NewMIs) const;
  
  /// foldMemoryOperand - PowerPC (like most RISC's) can only fold spills into
  /// copy instructions, turning them into load/store instructions.
  virtual MachineInstr* foldMemoryOperandImpl(MachineFunction &MF,
                                              MachineInstr* MI,
                                              const SmallVectorImpl<unsigned> &Ops,
                                              int FrameIndex) const;

  virtual MachineInstr* foldMemoryOperandImpl(MachineFunction &MF,
                                              MachineInstr* MI,
                                              const SmallVectorImpl<unsigned> &Ops,
                                              MachineInstr* LoadMI) const {
    return 0;
  }

  virtual bool canFoldMemoryOperand(const MachineInstr *MI,
                                    const SmallVectorImpl<unsigned> &Ops) const;
  
  virtual bool BlockHasNoFallThrough(const MachineBasicBlock &MBB) const;
  virtual
  bool ReverseBranchCondition(SmallVectorImpl<MachineOperand> &Cond) const;
  
  /// GetInstSize - Return the number of bytes of code the specified
  /// instruction may be.  This returns the maximum number of bytes.
  ///
  virtual unsigned GetInstSizeInBytes(const MachineInstr *MI) const;
};

}

#endif
