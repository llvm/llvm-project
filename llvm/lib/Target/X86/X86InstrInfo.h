//===- X86InstrInfo.h - X86 Instruction Information ------------*- C++ -*- ===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the X86 implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef X86INSTRUCTIONINFO_H
#define X86INSTRUCTIONINFO_H

#include "llvm/Target/TargetInstrInfo.h"
#include "X86.h"
#include "X86RegisterInfo.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Target/TargetRegisterInfo.h"

namespace llvm {
  class X86RegisterInfo;
  class X86TargetMachine;

namespace X86 {
  // X86 specific condition code. These correspond to X86_*_COND in
  // X86InstrInfo.td. They must be kept in synch.
  enum CondCode {
    COND_A  = 0,
    COND_AE = 1,
    COND_B  = 2,
    COND_BE = 3,
    COND_E  = 4,
    COND_G  = 5,
    COND_GE = 6,
    COND_L  = 7,
    COND_LE = 8,
    COND_NE = 9,
    COND_NO = 10,
    COND_NP = 11,
    COND_NS = 12,
    COND_O  = 13,
    COND_P  = 14,
    COND_S  = 15,

    // Artificial condition codes. These are used by AnalyzeBranch
    // to indicate a block terminated with two conditional branches to
    // the same location. This occurs in code using FCMP_OEQ or FCMP_UNE,
    // which can't be represented on x86 with a single condition. These
    // are never used in MachineInstrs.
    COND_NE_OR_P,
    COND_NP_OR_E,

    COND_INVALID
  };
    
  // Turn condition code into conditional branch opcode.
  unsigned GetCondBranchFromCond(CondCode CC);
  
  /// GetOppositeBranchCondition - Return the inverse of the specified cond,
  /// e.g. turning COND_E to COND_NE.
  CondCode GetOppositeBranchCondition(X86::CondCode CC);

}
  
/// X86II - This namespace holds all of the target specific flags that
/// instruction info tracks.
///
namespace X86II {
  enum {
    //===------------------------------------------------------------------===//
    // Instruction types.  These are the standard/most common forms for X86
    // instructions.
    //

    // PseudoFrm - This represents an instruction that is a pseudo instruction
    // or one that has not been implemented yet.  It is illegal to code generate
    // it, but tolerated for intermediate implementation stages.
    Pseudo         = 0,

    /// Raw - This form is for instructions that don't have any operands, so
    /// they are just a fixed opcode value, like 'leave'.
    RawFrm         = 1,

    /// AddRegFrm - This form is used for instructions like 'push r32' that have
    /// their one register operand added to their opcode.
    AddRegFrm      = 2,

    /// MRMDestReg - This form is used for instructions that use the Mod/RM byte
    /// to specify a destination, which in this case is a register.
    ///
    MRMDestReg     = 3,

    /// MRMDestMem - This form is used for instructions that use the Mod/RM byte
    /// to specify a destination, which in this case is memory.
    ///
    MRMDestMem     = 4,

    /// MRMSrcReg - This form is used for instructions that use the Mod/RM byte
    /// to specify a source, which in this case is a register.
    ///
    MRMSrcReg      = 5,

    /// MRMSrcMem - This form is used for instructions that use the Mod/RM byte
    /// to specify a source, which in this case is memory.
    ///
    MRMSrcMem      = 6,

    /// MRM[0-7][rm] - These forms are used to represent instructions that use
    /// a Mod/RM byte, and use the middle field to hold extended opcode
    /// information.  In the intel manual these are represented as /0, /1, ...
    ///

    // First, instructions that operate on a register r/m operand...
    MRM0r = 16,  MRM1r = 17,  MRM2r = 18,  MRM3r = 19, // Format /0 /1 /2 /3
    MRM4r = 20,  MRM5r = 21,  MRM6r = 22,  MRM7r = 23, // Format /4 /5 /6 /7

    // Next, instructions that operate on a memory r/m operand...
    MRM0m = 24,  MRM1m = 25,  MRM2m = 26,  MRM3m = 27, // Format /0 /1 /2 /3
    MRM4m = 28,  MRM5m = 29,  MRM6m = 30,  MRM7m = 31, // Format /4 /5 /6 /7

    // MRMInitReg - This form is used for instructions whose source and
    // destinations are the same register.
    MRMInitReg = 32,

    FormMask       = 63,

    //===------------------------------------------------------------------===//
    // Actual flags...

    // OpSize - Set if this instruction requires an operand size prefix (0x66),
    // which most often indicates that the instruction operates on 16 bit data
    // instead of 32 bit data.
    OpSize      = 1 << 6,

    // AsSize - Set if this instruction requires an operand size prefix (0x67),
    // which most often indicates that the instruction address 16 bit address
    // instead of 32 bit address (or 32 bit address in 64 bit mode).
    AdSize      = 1 << 7,

    //===------------------------------------------------------------------===//
    // Op0Mask - There are several prefix bytes that are used to form two byte
    // opcodes.  These are currently 0x0F, 0xF3, and 0xD8-0xDF.  This mask is
    // used to obtain the setting of this field.  If no bits in this field is
    // set, there is no prefix byte for obtaining a multibyte opcode.
    //
    Op0Shift    = 8,
    Op0Mask     = 0xF << Op0Shift,

    // TB - TwoByte - Set if this instruction has a two byte opcode, which
    // starts with a 0x0F byte before the real opcode.
    TB          = 1 << Op0Shift,

    // REP - The 0xF3 prefix byte indicating repetition of the following
    // instruction.
    REP         = 2 << Op0Shift,

    // D8-DF - These escape opcodes are used by the floating point unit.  These
    // values must remain sequential.
    D8 = 3 << Op0Shift,   D9 = 4 << Op0Shift,
    DA = 5 << Op0Shift,   DB = 6 << Op0Shift,
    DC = 7 << Op0Shift,   DD = 8 << Op0Shift,
    DE = 9 << Op0Shift,   DF = 10 << Op0Shift,

    // XS, XD - These prefix codes are for single and double precision scalar
    // floating point operations performed in the SSE registers.
    XD = 11 << Op0Shift,  XS = 12 << Op0Shift,

    // T8, TA - Prefix after the 0x0F prefix.
    T8 = 13 << Op0Shift,  TA = 14 << Op0Shift,

    //===------------------------------------------------------------------===//
    // REX_W - REX prefixes are instruction prefixes used in 64-bit mode.
    // They are used to specify GPRs and SSE registers, 64-bit operand size,
    // etc. We only cares about REX.W and REX.R bits and only the former is
    // statically determined.
    //
    REXShift    = 12,
    REX_W       = 1 << REXShift,

    //===------------------------------------------------------------------===//
    // This three-bit field describes the size of an immediate operand.  Zero is
    // unused so that we can tell if we forgot to set a value.
    ImmShift = 13,
    ImmMask  = 7 << ImmShift,
    Imm8     = 1 << ImmShift,
    Imm16    = 2 << ImmShift,
    Imm32    = 3 << ImmShift,
    Imm64    = 4 << ImmShift,

    //===------------------------------------------------------------------===//
    // FP Instruction Classification...  Zero is non-fp instruction.

    // FPTypeMask - Mask for all of the FP types...
    FPTypeShift = 16,
    FPTypeMask  = 7 << FPTypeShift,

    // NotFP - The default, set for instructions that do not use FP registers.
    NotFP      = 0 << FPTypeShift,

    // ZeroArgFP - 0 arg FP instruction which implicitly pushes ST(0), f.e. fld0
    ZeroArgFP  = 1 << FPTypeShift,

    // OneArgFP - 1 arg FP instructions which implicitly read ST(0), such as fst
    OneArgFP   = 2 << FPTypeShift,

    // OneArgFPRW - 1 arg FP instruction which implicitly read ST(0) and write a
    // result back to ST(0).  For example, fcos, fsqrt, etc.
    //
    OneArgFPRW = 3 << FPTypeShift,

    // TwoArgFP - 2 arg FP instructions which implicitly read ST(0), and an
    // explicit argument, storing the result to either ST(0) or the implicit
    // argument.  For example: fadd, fsub, fmul, etc...
    TwoArgFP   = 4 << FPTypeShift,

    // CompareFP - 2 arg FP instructions which implicitly read ST(0) and an
    // explicit argument, but have no destination.  Example: fucom, fucomi, ...
    CompareFP  = 5 << FPTypeShift,

    // CondMovFP - "2 operand" floating point conditional move instructions.
    CondMovFP  = 6 << FPTypeShift,

    // SpecialFP - Special instruction forms.  Dispatch by opcode explicitly.
    SpecialFP  = 7 << FPTypeShift,

    // Lock prefix
    LOCKShift = 19,
    LOCK = 1 << LOCKShift,

    // Segment override prefixes. Currently we just need ability to address
    // stuff in gs and fs segments.
    SegOvrShift = 20,
    SegOvrMask  = 3 << SegOvrShift,
    FS          = 1 << SegOvrShift,
    GS          = 2 << SegOvrShift,

    // Bits 22 -> 23 are unused
    OpcodeShift   = 24,
    OpcodeMask    = 0xFF << OpcodeShift
  };
}

inline static bool isScale(const MachineOperand &MO) {
  return MO.isImm() &&
    (MO.getImm() == 1 || MO.getImm() == 2 ||
     MO.getImm() == 4 || MO.getImm() == 8);
}

inline static bool isMem(const MachineInstr *MI, unsigned Op) {
  if (MI->getOperand(Op).isFI()) return true;
  return Op+4 <= MI->getNumOperands() &&
    MI->getOperand(Op  ).isReg() && isScale(MI->getOperand(Op+1)) &&
    MI->getOperand(Op+2).isReg() &&
    (MI->getOperand(Op+3).isImm() ||
     MI->getOperand(Op+3).isGlobal() ||
     MI->getOperand(Op+3).isCPI() ||
     MI->getOperand(Op+3).isJTI());
}

class X86InstrInfo : public TargetInstrInfoImpl {
  X86TargetMachine &TM;
  const X86RegisterInfo RI;
  
  /// RegOp2MemOpTable2Addr, RegOp2MemOpTable0, RegOp2MemOpTable1,
  /// RegOp2MemOpTable2 - Load / store folding opcode maps.
  ///
  DenseMap<unsigned*, unsigned> RegOp2MemOpTable2Addr;
  DenseMap<unsigned*, unsigned> RegOp2MemOpTable0;
  DenseMap<unsigned*, unsigned> RegOp2MemOpTable1;
  DenseMap<unsigned*, unsigned> RegOp2MemOpTable2;
  
  /// MemOp2RegOpTable - Load / store unfolding opcode map.
  ///
  DenseMap<unsigned*, std::pair<unsigned, unsigned> > MemOp2RegOpTable;
  
public:
  explicit X86InstrInfo(X86TargetMachine &tm);

  /// getRegisterInfo - TargetInstrInfo is a superset of MRegister info.  As
  /// such, whenever a client has an instance of instruction info, it should
  /// always be able to get register info as well (through this method).
  ///
  virtual const X86RegisterInfo &getRegisterInfo() const { return RI; }

  /// Return true if the instruction is a register to register move and return
  /// the source and dest operands and their sub-register indices by reference.
  virtual bool isMoveInstr(const MachineInstr &MI,
                           unsigned &SrcReg, unsigned &DstReg,
                           unsigned &SrcSubIdx, unsigned &DstSubIdx) const;

  unsigned isLoadFromStackSlot(const MachineInstr *MI, int &FrameIndex) const;
  unsigned isStoreToStackSlot(const MachineInstr *MI, int &FrameIndex) const;

  bool isReallyTriviallyReMaterializable(const MachineInstr *MI) const;
  void reMaterialize(MachineBasicBlock &MBB, MachineBasicBlock::iterator MI,
                     unsigned DestReg, const MachineInstr *Orig) const;

  bool isInvariantLoad(const MachineInstr *MI) const;

  /// convertToThreeAddress - This method must be implemented by targets that
  /// set the M_CONVERTIBLE_TO_3_ADDR flag.  When this flag is set, the target
  /// may be able to convert a two-address instruction into a true
  /// three-address instruction on demand.  This allows the X86 target (for
  /// example) to convert ADD and SHL instructions into LEA instructions if they
  /// would require register copies due to two-addressness.
  ///
  /// This method returns a null pointer if the transformation cannot be
  /// performed, otherwise it returns the new instruction.
  ///
  virtual MachineInstr *convertToThreeAddress(MachineFunction::iterator &MFI,
                                              MachineBasicBlock::iterator &MBBI,
                                              LiveVariables *LV) const;

  /// commuteInstruction - We have a few instructions that must be hacked on to
  /// commute them.
  ///
  virtual MachineInstr *commuteInstruction(MachineInstr *MI, bool NewMI) const;

  // Branch analysis.
  virtual bool isUnpredicatedTerminator(const MachineInstr* MI) const;
  virtual bool AnalyzeBranch(MachineBasicBlock &MBB, MachineBasicBlock *&TBB,
                             MachineBasicBlock *&FBB,
                             SmallVectorImpl<MachineOperand> &Cond) const;
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
                                   MachineBasicBlock::iterator MI,
                                   unsigned SrcReg, bool isKill, int FrameIndex,
                                   const TargetRegisterClass *RC) const;

  virtual void storeRegToAddr(MachineFunction &MF, unsigned SrcReg, bool isKill,
                              SmallVectorImpl<MachineOperand> &Addr,
                              const TargetRegisterClass *RC,
                              SmallVectorImpl<MachineInstr*> &NewMIs) const;

  virtual void loadRegFromStackSlot(MachineBasicBlock &MBB,
                                    MachineBasicBlock::iterator MI,
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
  
  /// foldMemoryOperand - If this target supports it, fold a load or store of
  /// the specified stack slot into the specified machine instruction for the
  /// specified operand(s).  If this is possible, the target should perform the
  /// folding and return true, otherwise it should return false.  If it folds
  /// the instruction, it is likely that the MachineInstruction the iterator
  /// references has been changed.
  virtual MachineInstr* foldMemoryOperandImpl(MachineFunction &MF,
                                              MachineInstr* MI,
                                           const SmallVectorImpl<unsigned> &Ops,
                                              int FrameIndex) const;

  /// foldMemoryOperand - Same as the previous version except it allows folding
  /// of any load and store from / to any address, not just from a specific
  /// stack slot.
  virtual MachineInstr* foldMemoryOperandImpl(MachineFunction &MF,
                                              MachineInstr* MI,
                                           const SmallVectorImpl<unsigned> &Ops,
                                              MachineInstr* LoadMI) const;

  /// canFoldMemoryOperand - Returns true if the specified load / store is
  /// folding is possible.
  virtual bool canFoldMemoryOperand(const MachineInstr*,
                                    const SmallVectorImpl<unsigned> &) const;

  /// unfoldMemoryOperand - Separate a single instruction which folded a load or
  /// a store or a load and a store into two or more instruction. If this is
  /// possible, returns true as well as the new instructions by reference.
  virtual bool unfoldMemoryOperand(MachineFunction &MF, MachineInstr *MI,
                           unsigned Reg, bool UnfoldLoad, bool UnfoldStore,
                           SmallVectorImpl<MachineInstr*> &NewMIs) const;

  virtual bool unfoldMemoryOperand(SelectionDAG &DAG, SDNode *N,
                           SmallVectorImpl<SDNode*> &NewNodes) const;

  /// getOpcodeAfterMemoryUnfold - Returns the opcode of the would be new
  /// instruction after load / store are unfolded from an instruction of the
  /// specified opcode. It returns zero if the specified unfolding is not
  /// possible.
  virtual unsigned getOpcodeAfterMemoryUnfold(unsigned Opc,
                                      bool UnfoldLoad, bool UnfoldStore) const;
  
  virtual bool BlockHasNoFallThrough(const MachineBasicBlock &MBB) const;
  virtual
  bool ReverseBranchCondition(SmallVectorImpl<MachineOperand> &Cond) const;

  /// IgnoreRegisterClassBarriers - Returns true if pre-register allocation
  /// live interval splitting pass should ignore barriers of the specified
  /// register class.
  bool IgnoreRegisterClassBarriers(const TargetRegisterClass *RC) const;

  const TargetRegisterClass *getPointerRegClass() const;

  // getBaseOpcodeFor - This function returns the "base" X86 opcode for the
  // specified machine instruction.
  //
  unsigned char getBaseOpcodeFor(const TargetInstrDesc *TID) const {
    return TID->TSFlags >> X86II::OpcodeShift;
  }
  unsigned char getBaseOpcodeFor(unsigned Opcode) const {
    return getBaseOpcodeFor(&get(Opcode));
  }
  
  static bool isX86_64NonExtLowByteReg(unsigned reg) {
    return (reg == X86::SPL || reg == X86::BPL ||
          reg == X86::SIL || reg == X86::DIL);
  }
  
  static unsigned sizeOfImm(const TargetInstrDesc *Desc);
  static bool isX86_64ExtendedReg(const MachineOperand &MO);
  static unsigned determineREX(const MachineInstr &MI);

  /// GetInstSize - Returns the size of the specified MachineInstr.
  ///
  virtual unsigned GetInstSizeInBytes(const MachineInstr *MI) const;

  /// getGlobalBaseReg - Return a virtual register initialized with the
  /// the global base register value. Output instructions required to
  /// initialize the register in the function entry block, if necessary.
  ///
  unsigned getGlobalBaseReg(MachineFunction *MF) const;

private:
  MachineInstr* foldMemoryOperandImpl(MachineFunction &MF,
                                      MachineInstr* MI,
                                      unsigned OpNum,
                                      const SmallVectorImpl<MachineOperand> &MOs) const;
};

} // End llvm namespace

#endif
