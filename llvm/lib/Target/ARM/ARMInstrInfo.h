//===- ARMInstrInfo.h - ARM Instruction Information -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the ARM implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef ARMINSTRUCTIONINFO_H
#define ARMINSTRUCTIONINFO_H

#include "llvm/Target/TargetInstrInfo.h"
#include "ARMRegisterInfo.h"
#include "ARM.h"

namespace llvm {
  class ARMSubtarget;

/// ARMII - This namespace holds all of the target specific flags that
/// instruction info tracks.
///
namespace ARMII {
  enum {
    //===------------------------------------------------------------------===//
    // Instruction Flags.

    //===------------------------------------------------------------------===//
    // This four-bit field describes the addressing mode used.

    AddrModeMask  = 0xf,
    AddrModeNone  = 0,
    AddrMode1     = 1,
    AddrMode2     = 2,
    AddrMode3     = 3,
    AddrMode4     = 4,
    AddrMode5     = 5,
    AddrModeT1    = 6,
    AddrModeT2    = 7,
    AddrModeT4    = 8,
    AddrModeTs    = 9,  // i8 * 4 for pc and sp relative data

    // Size* - Flags to keep track of the size of an instruction.
    SizeShift     = 4,
    SizeMask      = 7 << SizeShift,
    SizeSpecial   = 1,   // 0 byte pseudo or special case.
    Size8Bytes    = 2,
    Size4Bytes    = 3,
    Size2Bytes    = 4,
    
    // IndexMode - Unindex, pre-indexed, or post-indexed. Only valid for load
    // and store ops 
    IndexModeShift = 7,
    IndexModeMask  = 3 << IndexModeShift,
    IndexModePre   = 1,
    IndexModePost  = 2,
    
    //===------------------------------------------------------------------===//
    // Misc flags.

    // UnaryDP - Indicates this is a unary data processing instruction, i.e.
    // it doesn't have a Rn operand.
    UnaryDP       = 1 << 9,

    //===------------------------------------------------------------------===//
    // Instruction encoding formats.
    //
    FormShift     = 10,
    FormMask      = 0x1f << FormShift,

    // Pseudo instructions
    Pseudo        = 0  << FormShift,

    // Multiply instructions
    MulFrm        = 1  << FormShift,

    // Branch instructions
    BrFrm         = 2  << FormShift,
    BrMiscFrm     = 3  << FormShift,

    // Data Processing instructions
    DPFrm         = 4  << FormShift,
    DPSoRegFrm    = 5  << FormShift,

    // Load and Store
    LdFrm         = 6  << FormShift,
    StFrm         = 7  << FormShift,
    LdMiscFrm     = 8  << FormShift,
    StMiscFrm     = 9  << FormShift,
    LdStMulFrm    = 10 << FormShift,

    // Miscellaneous arithmetic instructions
    ArithMiscFrm  = 11 << FormShift,

    // Extend instructions
    ExtFrm        = 12 << FormShift,

    // VFP formats
    VFPUnaryFrm   = 13 << FormShift,
    VFPBinaryFrm  = 14 << FormShift,
    VFPConv1Frm   = 15 << FormShift,
    VFPConv2Frm   = 16 << FormShift,
    VFPConv3Frm   = 17 << FormShift,
    VFPConv4Frm   = 18 << FormShift,
    VFPConv5Frm   = 19 << FormShift,
    VFPLdStFrm    = 20 << FormShift,
    VFPLdStMulFrm = 21 << FormShift,
    VFPMiscFrm    = 22 << FormShift,

    // Thumb format
    ThumbFrm      = 23 << FormShift,

    //===------------------------------------------------------------------===//
    // Field shifts - such shifts are used to set field while generating
    // machine instructions.
    M_BitShift     = 5,
    ShiftImmShift  = 5,
    ShiftShift     = 7,
    N_BitShift     = 7,
    ImmHiShift     = 8,
    SoRotImmShift  = 8,
    RegRsShift     = 8,
    ExtRotImmShift = 10,
    RegRdLoShift   = 12,
    RegRdShift     = 12,
    RegRdHiShift   = 16,
    RegRnShift     = 16,
    S_BitShift     = 20,
    W_BitShift     = 21,
    AM3_I_BitShift = 22,
    D_BitShift     = 22,
    U_BitShift     = 23,
    P_BitShift     = 24,
    I_BitShift     = 25,
    CondShift      = 28
  };
}

class ARMInstrInfo : public TargetInstrInfoImpl {
  const ARMRegisterInfo RI;
public:
  explicit ARMInstrInfo(const ARMSubtarget &STI);

  /// getRegisterInfo - TargetInstrInfo is a superset of MRegister info.  As
  /// such, whenever a client has an instance of instruction info, it should
  /// always be able to get register info as well (through this method).
  ///
  virtual const ARMRegisterInfo &getRegisterInfo() const { return RI; }

  /// getPointerRegClass - Return the register class to use to hold pointers.
  /// This is used for addressing modes.
  virtual const TargetRegisterClass *getPointerRegClass() const;

  /// Return true if the instruction is a register to register move and return
  /// the source and dest operands and their sub-register indices by reference.
  virtual bool isMoveInstr(const MachineInstr &MI,
                           unsigned &SrcReg, unsigned &DstReg,
                           unsigned &SrcSubIdx, unsigned &DstSubIdx) const;

  virtual unsigned isLoadFromStackSlot(const MachineInstr *MI,
                                       int &FrameIndex) const;
  virtual unsigned isStoreToStackSlot(const MachineInstr *MI,
                                      int &FrameIndex) const;
  
  void reMaterialize(MachineBasicBlock &MBB, MachineBasicBlock::iterator MI,
                     unsigned DestReg, const MachineInstr *Orig) const;

  virtual MachineInstr *convertToThreeAddress(MachineFunction::iterator &MFI,
                                              MachineBasicBlock::iterator &MBBI,
                                              LiveVariables *LV) const;

  // Branch analysis.
  virtual bool AnalyzeBranch(MachineBasicBlock &MBB, MachineBasicBlock *&TBB,
                             MachineBasicBlock *&FBB,
                             SmallVectorImpl<MachineOperand> &Cond) const;
  virtual unsigned RemoveBranch(MachineBasicBlock &MBB) const;
  virtual unsigned InsertBranch(MachineBasicBlock &MBB, MachineBasicBlock *TBB,
                                MachineBasicBlock *FBB,
                            const SmallVectorImpl<MachineOperand> &Cond) const;
  virtual bool copyRegToReg(MachineBasicBlock &MBB,
                            MachineBasicBlock::iterator I,
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
  virtual bool spillCalleeSavedRegisters(MachineBasicBlock &MBB,
                                         MachineBasicBlock::iterator MI,
                                 const std::vector<CalleeSavedInfo> &CSI) const;
  virtual bool restoreCalleeSavedRegisters(MachineBasicBlock &MBB,
                                           MachineBasicBlock::iterator MI,
                                 const std::vector<CalleeSavedInfo> &CSI) const;
  
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

  // Predication support.
  virtual bool isPredicated(const MachineInstr *MI) const;

  ARMCC::CondCodes getPredicate(const MachineInstr *MI) const {
    int PIdx = MI->findFirstPredOperandIdx();
    return PIdx != -1 ? (ARMCC::CondCodes)MI->getOperand(PIdx).getImm() 
                      : ARMCC::AL;
  }

  virtual
  bool PredicateInstruction(MachineInstr *MI,
                            const SmallVectorImpl<MachineOperand> &Pred) const;

  virtual
  bool SubsumesPredicate(const SmallVectorImpl<MachineOperand> &Pred1,
                         const SmallVectorImpl<MachineOperand> &Pred2) const;

  virtual bool DefinesPredicate(MachineInstr *MI,
                                std::vector<MachineOperand> &Pred) const;
    
  /// GetInstSize - Returns the size of the specified MachineInstr.
  ///
  virtual unsigned GetInstSizeInBytes(const MachineInstr* MI) const;
};

}

#endif
